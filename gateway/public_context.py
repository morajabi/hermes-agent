"""Opt-in adapter boundary for recording confirmed public output identities."""
import asyncio
import logging
import json
import uuid

logger = logging.getLogger(__name__)


async def prepare_public_admission(store, adapter, event, session_id, session_key):
    prepare = getattr(adapter, "prepare_public_context", None)
    if not callable(prepare):
        return None
    db = store._db_for_key(session_key)
    if not hasattr(db, "public_context_state"):
        return None
    state = await asyncio.to_thread(db.public_context_state, session_id)
    state = await reconcile_public_outputs(db, adapter, event.source, session_id, state)
    snapshot = await prepare(event, state=state)
    if snapshot is None:
        return None
    batch = {k: state[k] for k in ("version", "source", "session_key", "generation")}
    batch.update(refs=snapshot["refs"], content=snapshot.get("content") or "")
    source = event.source
    namespace = [source.platform.value, source.profile, source.scope_id,
                 source.chat_id, source.thread_id, str(event.message_id), snapshot.get("input_revision")]
    batch["input_owner"] = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(namespace))) if event.message_id else str(uuid.uuid4())
    event._public_context, event._public_context_session_id = batch, session_id
    return batch


def public_send_ids(result):
    if not getattr(result, "success", False):
        return ()
    raw = getattr(result, "raw_response", None) or {}
    ids = raw.get("message_ids") if isinstance(raw, dict) else ()
    return tuple(str(mid) for mid in (getattr(result, "message_id", None),
                 *(getattr(result, "continuation_message_ids", None) or ()), *(ids or ())) if mid)


async def record_event_delivery(runner, event, adapter, result):
    batch = getattr(event, "_public_context", None)
    if not batch:
        return
    await record_public_delivery(runner.session_store, adapter, event.source,
                                getattr(event, "_public_context_session_id", None),
                                batch, public_send_ids(result))


async def record_public_delivery(store, adapter, source, session_id, batch, message_ids):
    references = getattr(adapter, "public_output_references", None)
    if not batch or not session_id or not callable(references) or not message_ids:
        return
    ids = tuple(dict.fromkeys(message_ids))
    pending = [{"chat_id": str(source.thread_id or source.chat_id), "message_id": str(mid)} for mid in ids]
    db = store._db_for_key(batch["session_key"])
    # A provenance write failure cannot be treated as successful adoption. The
    # confirmed send itself is never retried by this path.
    await asyncio.to_thread(db.record_public_output_pending, session_id, batch, pending)
    try:
        refs = await references(source, message_ids=ids)
        await asyncio.to_thread(db.resolve_public_output_pending, session_id, batch, pending, refs)
    except Exception:
        # The next admission must reconcile this durable receipt before it can
        # import chat history. A lookup failure must never resend the output.
        logger.warning("Public output revision pending for %s", batch["session_key"], exc_info=True)


async def reconcile_public_outputs(db, adapter, source, session_id, state):
    pending = [r for r in state["accepted"] if r["kind"] == "output_pending"]
    if not pending:
        return state
    refs = await adapter.public_output_references(source, message_ids=tuple(r["message_id"] for r in pending))
    await asyncio.to_thread(db.resolve_public_output_pending, session_id, state, pending, refs)
    return await asyncio.to_thread(db.public_context_state, session_id)
