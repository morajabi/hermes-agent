"""Public-chat admission receipts, independent of transcript compression.

The adapter supplies physical message identities and immutable content revisions.
Receipts commit with the ordinary user row; fetching a chat never advances them.
Private provider history remains private, and explicit resets fence public history.
"""

from __future__ import annotations

import json
import time
from hermes_state_common import _RECOVERABLE_END_REASONS_SQL


class PublicContextAdmissionError(RuntimeError):
    """The prepared chat snapshot no longer belongs to this conversation."""


class SessionPublicContextMixin:
    def public_context_state(self, session_id: str) -> dict:
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row or not row["source"] or not row["session_key"]:
                raise PublicContextAdmissionError("public context requires a routed session")
            boundary = conn.execute(
                "SELECT generation, public_context_floor FROM conversation_generations "
                "WHERE source = ? AND session_key = ?", (row["source"], row["session_key"])
            ).fetchone()
            generation = int(boundary["generation"]) if boundary else 0
            refs = conn.execute(
                "SELECT chat_id, message_id, revision, kind FROM public_context_receipts "
                "WHERE source = ? AND session_key = ? AND generation = ?",
                (row["source"], row["session_key"], generation),
            ).fetchall()
            return {
                "version": 1, "source": row["source"], "session_key": row["session_key"],
                "generation": generation,
                "floor": json.loads(boundary["public_context_floor"])
                if boundary and boundary["public_context_floor"] else None,
                "accepted": [dict(ref) for ref in refs],
            }

    @staticmethod
    def _public_context_metadata(metadata):
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return metadata.get("public_context") if isinstance(metadata, dict) else None

    def _record_public_context_input(self, conn, session_id: str, message) -> None:
        metadata = message.get("display_metadata")
        batch = self._public_context_metadata(metadata)
        if not batch:
            return
        if message.get("_compressed_summary"):
            # Summaries continue previously admitted history; their metadata
            # cannot adopt a newly fetched message revision.
            return
        public_bytes = batch.get("content") or ""
        wire = message.get("api_content") or message.get("content")
        if public_bytes and not self._public_bytes_in_content(wire, public_bytes):
            raise PublicContextAdmissionError("public context bytes are missing from the admitted user row")
        self._validate_public_context_owner(conn, session_id, batch)
        for ref in batch.get("refs", []):
            self._insert_public_context_ref(conn, batch, ref, "input")

    @staticmethod
    def _public_bytes_in_content(content, public_bytes):
        if isinstance(content, str):
            return public_bytes in content
        if isinstance(content, list):
            return any(isinstance(part, dict) and isinstance(part.get("text"), str)
                       and public_bytes in part["text"] for part in content)
        return False

    def _validate_public_context_owner(self, conn, session_id, batch):
        row = conn.execute(
            "SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row or (row["source"], row["session_key"]) != (batch.get("source"), batch.get("session_key")):
            raise PublicContextAdmissionError("public context route changed")
        boundary = conn.execute(
            "SELECT generation FROM conversation_generations WHERE source = ? AND session_key = ?",
            (row["source"], row["session_key"]),
        ).fetchone()
        if (int(boundary["generation"]) if boundary else 0) != batch.get("generation"):
            raise PublicContextAdmissionError("conversation reset during public context admission")

    @staticmethod
    def _insert_public_context_ref(conn, batch, ref, kind):
        if not isinstance(ref, dict) or any(not str(ref.get(k) or "") for k in ("chat_id", "message_id", "revision")):
            raise PublicContextAdmissionError("invalid public message reference")
        conn.execute(
            "INSERT OR IGNORE INTO public_context_receipts "
            "(source, session_key, generation, chat_id, message_id, revision, kind) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (batch["source"], batch["session_key"], batch["generation"],
             str(ref["chat_id"]), str(ref["message_id"]), str(ref["revision"]), kind),
        )

    def public_context_was_accepted(self, session_id: str, batch: dict) -> bool:
        """Verify durable admission before a provider request; do not infer it from an SDK ACK."""
        with self._read_ctx() as conn:
            self._validate_public_context_owner(conn, session_id, batch)
            row = conn.execute(
                "SELECT content, api_content FROM messages WHERE session_id = ? AND role = 'user' AND "
                "json_extract(display_metadata, '$.public_context.input_owner') = ? LIMIT 1",
                (session_id, batch["input_owner"]),
            ).fetchone() if batch.get("input_owner") else None
            if row is None:
                return False
            wire = row["api_content"] or self._decode_content(row["content"])
            public_bytes = batch.get("content") or ""
            if public_bytes and not self._public_bytes_in_content(wire, public_bytes):
                return False
            return all(conn.execute(
                "SELECT 1 FROM public_context_receipts WHERE source = ? AND session_key = ? "
                "AND generation = ? AND chat_id = ? AND message_id = ? AND revision = ?",
                (batch["source"], batch["session_key"], batch["generation"],
                 str(ref["chat_id"]), str(ref["message_id"]), str(ref["revision"])),
            ).fetchone() is not None for ref in batch.get("refs", []))

    def record_public_output(self, session_id: str, batch: dict, refs: list) -> None:
        """Bind confirmed physical output to this run without rewriting past model messages."""
        def write(conn):
            self._validate_public_context_owner(conn, session_id, batch)
            for ref in refs:
                self._insert_public_context_ref(conn, batch, ref, "output")
        self._execute_write(write)

    def record_public_output_pending(self, session_id: str, batch: dict, refs: list) -> None:
        """Retain confirmed physical sends before fetching their final revisions."""
        def write(conn):
            self._validate_public_context_owner(conn, session_id, batch)
            for ref in refs:
                self._insert_public_context_ref(conn, batch, {**ref, "revision": "pending"}, "output_pending")
        self._execute_write(write)

    def resolve_public_output_pending(self, session_id: str, batch: dict, pending: list, refs: list) -> None:
        """A successful authoritative fetch resolves live outputs and deleted previews atomically."""
        def write(conn):
            self._validate_public_context_owner(conn, session_id, batch)
            expected = {(str(r["chat_id"]), str(r["message_id"])) for r in pending}
            if any((str(r["chat_id"]), str(r["message_id"])) not in expected for r in refs):
                raise PublicContextAdmissionError("output fetch returned an unexpected physical message")
            for ref in refs:
                self._insert_public_context_ref(conn, batch, ref, "output")
            for chat_id, message_id in expected:
                conn.execute("DELETE FROM public_context_receipts WHERE source = ? AND session_key = ? AND generation = ? "
                             "AND chat_id = ? AND message_id = ? AND kind = 'output_pending'",
                             (batch["source"], batch["session_key"], batch["generation"], chat_id, message_id))
        self._execute_write(write)

    def reset_public_context(self, session_id: str, floor: dict) -> None:
        """Compatibility entry point for a known segment; the boundary belongs to its route."""
        with self._read_ctx() as conn:
            row = conn.execute("SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row or not row["session_key"]:
            raise PublicContextAdmissionError("reset route unavailable")
        self.reset_public_context_route(row["source"], row["session_key"], floor)

    def reset_public_context_route(self, source: str, session_key: str, floor: dict) -> None:
        """Fence a conversation even when its in-memory route or all segments are absent."""
        if not isinstance(floor, dict) or any(not str(floor.get(k) or "") for k in ("chat_id", "message_id")):
            raise PublicContextAdmissionError("reset requires a physical chat message")
        if not source or not session_key:
            raise PublicContextAdmissionError("reset requires a canonical route")
        def write(conn):
            # Recoverable segments must not be rediscovered after a lost routing index.
            # Rotate the existing route generation once, rather than once per segment.
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'session_reset' WHERE source = ? AND session_key = ? "
                f"AND (ended_at IS NULL OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
                (time.time(), source, session_key))
            conn.execute(
                "INSERT INTO conversation_generations (source, session_key, generation, public_context_floor) "
                "VALUES (?, ?, 1, ?) ON CONFLICT(source, session_key) DO UPDATE SET "
                "generation = conversation_generations.generation + 1, "
                "public_context_floor = excluded.public_context_floor",
                (source, session_key, json.dumps({k: str(floor[k]) for k in ("chat_id", "message_id")}, sort_keys=True)),
            )
        self._execute_write(write)
