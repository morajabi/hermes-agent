"""Profile-local, immutable gateway inputs; the ordinary user row consumes them.

Transport ACK is adoption, never provider acceptance. Dispatch claims only avoid
duplicate process-local handoffs; a new exclusive receiver may recover unconsumed
inputs after restart. Conversation generations fence reset work independently.
"""
from __future__ import annotations

import hashlib
import json
import time


class GatewayIntakeError(RuntimeError):
    """An intake receipt cannot be adopted by this conversation."""


class SessionIntakeMixin:
    @staticmethod
    def _intake_json(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > 2_000_000:
            raise GatewayIntakeError("gateway input is too large to adopt")
        return encoded

    @staticmethod
    def _intake_generation(conn, source, session_key):
        row = conn.execute("SELECT generation FROM conversation_generations WHERE source = ? AND session_key = ?",
                           (source, session_key)).fetchone()
        return int(row[0]) if row else 0

    def adopt_gateway_intake(self, snapshot: dict, *, source: str, session_key: str,
                             owner: dict, dispatch_owner: str, expected_generation=None) -> dict:
        if not source or not session_key or not dispatch_owner:
            raise GatewayIntakeError("gateway intake requires a canonical owner")
        physical = [str(snapshot.get(key) or "") for key in (
            "receiver_id", "physical_chat_id", "physical_message_id", "revision")]
        if snapshot.get("version") != 1 or not all(physical) or not isinstance(snapshot.get("event"), dict):
            raise GatewayIntakeError("gateway intake requires an immutable physical event")
        if not all(owner.get(key) for key in ("transport_profile", "runtime_profile", "authorization_home", "runtime_home")):
            raise GatewayIntakeError("gateway intake requires physical profile ownership")
        snapshot_json, owner_json = self._intake_json(snapshot), self._intake_json(owner)
        receipt_id = hashlib.sha256(self._intake_json(
            [source, owner["authorization_home"], *physical]).encode("utf-8")).hexdigest()

        def write(conn):
            generation = self._intake_generation(conn, source, session_key)
            if expected_generation is not None and generation != expected_generation:
                raise GatewayIntakeError("conversation reset before gateway input adoption")
            row = conn.execute("SELECT * FROM gateway_intake WHERE receipt_id = ?", (receipt_id,)).fetchone()
            if row:
                if (row["source"], row["session_key"], row["generation"], row["owner_json"], row["snapshot_json"]) != (
                        source, session_key, generation, owner_json, snapshot_json):
                    raise GatewayIntakeError("gateway input changed route, generation, or immutable payload")
                dispatch = row["state"] == "pending" and row["dispatch_owner"] != dispatch_owner
                if dispatch:
                    conn.execute("UPDATE gateway_intake SET dispatch_owner = ? WHERE receipt_id = ?",
                                 (dispatch_owner, receipt_id))
            else:
                conn.execute("INSERT INTO gateway_intake (receipt_id, source, session_key, generation, owner_json, "
                             "snapshot_json, state, dispatch_owner, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                             (receipt_id, source, session_key, generation, owner_json, snapshot_json, dispatch_owner, time.time()))
                dispatch = True
            return {"receipt_id": receipt_id, "source": source, "session_key": session_key,
                    "generation": generation, "dispatch": dispatch}
        return self._execute_write(write)

    def gateway_intake_generation(self, source, session_key):
        with self._read_ctx() as conn:
            return self._intake_generation(conn, source, session_key)

    def pending_gateway_intakes(self, *, session_key=None, unclaimed_only=False) -> list:
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT i.* FROM gateway_intake i LEFT JOIN conversation_generations g "
                "ON g.source = i.source AND g.session_key = i.session_key "
                "WHERE i.state = 'pending' AND i.generation = COALESCE(g.generation, 0) "
                "AND (? IS NULL OR i.session_key = ?) AND (? = 0 OR i.dispatch_owner IS NULL) ORDER BY i.rowid",
                (session_key, session_key, int(unclaimed_only))).fetchall()
            return [{**dict(row), "snapshot": json.loads(row["snapshot_json"]),
                     "owner": json.loads(row["owner_json"])} for row in rows]

    def release_gateway_intake_dispatch(self, receipts, dispatch_owner):
        def write(conn):
            for receipt in receipts:
                conn.execute("UPDATE gateway_intake SET dispatch_owner = NULL WHERE receipt_id = ? "
                             "AND state = 'pending' AND dispatch_owner = ?",
                             (receipt["receipt_id"], dispatch_owner))
        self._execute_write(write)

    def refuse_gateway_intake(self, receipt_id):
        return self._execute_write(lambda conn: conn.execute(
            "UPDATE gateway_intake SET state = 'refused', dispatch_owner = NULL WHERE receipt_id = ? AND state = 'pending'",
            (receipt_id,)).rowcount)

    def refuse_gateway_intakes(self, receipt_ids):
        """Refuse an immutable stop cut; consumed inputs and later arrivals stay untouched."""
        def write(conn):
            refused = set()
            for receipt_id in receipt_ids:
                if conn.execute("UPDATE gateway_intake SET state = 'refused', dispatch_owner = NULL "
                                "WHERE state = 'pending' AND receipt_id = ?", (receipt_id,)).rowcount:
                    refused.add(receipt_id)
            return refused
        return self._execute_write(write)

    def _validate_gateway_intake(self, conn, session_id, receipt):
        session = conn.execute("SELECT source, session_key FROM sessions WHERE id = ?", (session_id,)).fetchone()
        row = conn.execute("SELECT * FROM gateway_intake WHERE receipt_id = ?", (receipt.get("receipt_id"),)).fetchone()
        if not session or not row or row["state"] == "refused" or (
                row["source"], row["session_key"], row["generation"]) != (
                receipt.get("source"), receipt.get("session_key"), receipt.get("generation")):
            raise GatewayIntakeError("gateway receipt is unavailable or belongs to another input")
        if (session["source"], session["session_key"]) != (row["source"], row["session_key"]):
            raise GatewayIntakeError("gateway receipt belongs to another conversation")
        if self._intake_generation(conn, row["source"], row["session_key"]) != row["generation"]:
            raise GatewayIntakeError("conversation reset before gateway input admission")
        return row

    def _consume_gateway_intake(self, conn, session_id, message):
        if message.get("role") != "user" or message.get("_compressed_summary"):
            return
        metadata = message.get("display_metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        receipts = metadata.get("gateway_intake") if isinstance(metadata, dict) else None
        if not receipts:
            return
        if not isinstance(receipts, list):
            raise GatewayIntakeError("gateway intake metadata must be a receipt list")
        for receipt in receipts:
            self._validate_gateway_intake(conn, session_id, receipt)
            conn.execute("UPDATE gateway_intake SET state = 'consumed', consumed_session_id = ?, dispatch_owner = NULL "
                         "WHERE receipt_id = ? AND state = 'pending'", (session_id, receipt["receipt_id"]))

    def gateway_intake_was_consumed(self, session_id, receipts):
        with self._read_ctx() as conn:
            return bool(receipts) and all(
                (row := self._validate_gateway_intake(conn, session_id, receipt))["state"] == "consumed"
                and row["consumed_session_id"] == session_id for receipt in receipts)

    def gateway_intake_is_pending(self, receipts):
        with self._read_ctx() as conn:
            for receipt in receipts:
                row = conn.execute("SELECT source, session_key, generation, state FROM gateway_intake WHERE receipt_id = ?",
                                   (receipt.get("receipt_id"),)).fetchone()
                if (not row or tuple(row) != (receipt.get("source"), receipt.get("session_key"),
                                               receipt.get("generation"), "pending")
                        or self._intake_generation(conn, row["source"], row["session_key"]) != row["generation"]):
                    return False
            return bool(receipts)
