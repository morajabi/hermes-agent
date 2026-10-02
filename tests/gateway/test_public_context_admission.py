"""Real SQLite admission, restart, reset and compression contracts."""
from types import SimpleNamespace
import sqlite3
import asyncio
import json

import pytest

from hermes_state import SessionDB
from hermes_state_public_context import PublicContextAdmissionError


def route(db, session_id="live", key="bot:chat", parent=None):
    db.create_session(session_id, source="test-public-chat", session_key=key,
                      chat_id="42", parent_session_id=parent)
    return session_id


def batch(db, sid="live", *, message_id="1", revision="v1", chat_id="42", owner="input-1"):
    state = db.public_context_state(sid)
    return {**{k: state[k] for k in ("version", "source", "session_key", "generation")},
            "input_owner": owner, "refs": [{"chat_id": chat_id, "message_id": message_id, "revision": revision}]}


def test_fetch_does_not_accept_and_row_receipts_commit_together(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = batch(db)
    assert not db.public_context_was_accepted("live", prepared)
    db.append_message("live", "user", "request", display_metadata={"public_context": prepared})
    assert db.public_context_was_accepted("live", prepared)
    db.close()
    restored = SessionDB(tmp_path / "state.db")
    assert restored.public_context_was_accepted("live", prepared)
    assert restored.public_context_state("live")["accepted"][0]["message_id"] == "1"
    restored.close()


def test_failed_row_insert_rolls_back_receipts(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = batch(db)
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_input BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'disk failure'); END"))
    with pytest.raises(Exception, match="disk failure"):
        db.append_message("live", "user", "request", display_metadata={"public_context": prepared})
    assert db.public_context_state("live")["accepted"] == []
    db.close()


def test_compression_continues_receipts_reset_fences_and_rejects_stale_snapshot(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = batch(db)
    db.append_message("live", "user", "request", display_metadata={"public_context": prepared})
    db.end_session("live", "compression")
    route(db, "compressed", parent="live")
    assert db.public_context_state("compressed")["accepted"]
    stale = batch(db, "compressed", message_id="2", owner="input-2")
    db.reset_public_context("compressed", {"chat_id": "42", "message_id": "7"})
    route(db, "reset", parent="compressed")
    assert db.public_context_state("reset")["floor"] == {"chat_id": "42", "message_id": "7"}
    assert db.public_context_state("reset")["accepted"] == []
    with pytest.raises(PublicContextAdmissionError, match="reset"):
        db.append_message("reset", "user", "stale", display_metadata={"public_context": stale})
    assert not db.get_messages("reset")
    db.close()


def test_receipts_bind_revision_physical_chat_and_profile_store(tmp_path):
    homes = [SessionDB(tmp_path / p / "state.db") for p in ("a", "b")]
    for db in homes:
        route(db)
    prepared = batch(homes[0])
    homes[0].append_message("live", "user", "request", display_metadata={"public_context": prepared})
    homes[0].record_public_output("live", prepared, [
        {"chat_id": "42", "message_id": "2", "revision": "answer"},
        {"chat_id": "43", "message_id": "2", "revision": "other physical chat"},
    ])
    assert len(homes[0].public_context_state("live")["accepted"]) == 3
    assert not homes[1].public_context_state("live")["accepted"]
    edited = batch(homes[0], revision="v2", owner="input-edit")
    assert not homes[0].public_context_was_accepted("live", edited)
    for db in homes:
        db.close()


def test_provider_boundary_refuses_failed_persistence_even_for_empty_refs(tmp_path):
    from agent.turn_context import _persist_turn_start
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = batch(db)
    prepared["refs"] = []
    agent = SimpleNamespace(session_id="live", _session_db=db,
                            _ensure_db_session=lambda: None, _persist_session=lambda *args: None)
    with pytest.raises(PublicContextAdmissionError, match="persisted"):
        _persist_turn_start(agent, [{"role": "user", "content": "request",
                                   "display_metadata": {"public_context": prepared}}], [], None)
    db.close()


def test_public_bytes_are_required_in_same_insert_and_multimodal_row(tmp_path):
    from agent.turn_context import _stage_turn_user_message
    from agent.session_persistence import durable_user_row_content
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = {**batch(db), "content": "PUBLIC CARD:\ncanary"}
    with pytest.raises(PublicContextAdmissionError, match="bytes"):
        db.append_message("live", "user", "ask", display_metadata={"public_context": prepared})
    assert not db.get_messages("live") and not db.public_context_state("live")["accepted"]
    image = {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
    user, _ = _stage_turn_user_message(SimpleNamespace(), [{"type": "text", "text": "ask"}, image],
                                      "clean ask", None, "1", None, {"public_context": prepared})
    content, api_content = durable_user_row_content(SimpleNamespace(_persist_user_message_override="clean ask"), user,
                                                   user["content"], user.get("api_content"))
    assert image in content
    db.append_message("live", "user", content, api_content=api_content, display_metadata=user["display_metadata"])
    assert db.public_context_was_accepted("live", prepared)
    db.close()


def test_cold_reset_and_old_schema_reopen_preserve_floor_without_route_index(tmp_path):
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE conversation_generations (source TEXT NOT NULL, session_key TEXT NOT NULL, "
                     "generation INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(source, session_key))")
    db = SessionDB(path)
    db.reset_public_context_route("test-public-chat", "bot:chat", {"chat_id": "42", "message_id": "7"})
    route(db)
    assert db.public_context_state("live")["floor"] == {"chat_id": "42", "message_id": "7"}
    db.end_session("live", "agent_close")
    db.reset_public_context_route("test-public-chat", "bot:chat", {"chat_id": "42", "message_id": "9"})
    assert db.get_session("live")["end_reason"] == "session_reset"
    db.close()
    reopened = SessionDB(path)
    route(reopened, "after-reset")
    assert reopened.public_context_state("after-reset")["floor"]["message_id"] == "9"
    reopened.close()


def test_confirmed_output_lookup_failure_is_reconciled_without_resend(tmp_path):
    async def scenario():
        from gateway.public_context import record_public_delivery, reconcile_public_outputs
        db = SessionDB(tmp_path / "state.db")
        route(db)
        prepared = batch(db)
        calls = []
        async def output_refs(source, *, message_ids):
            calls.append(message_ids)
            if len(calls) == 1:
                raise RuntimeError("temporary lookup failure")
            return [{"chat_id": "42", "message_id": mid, "revision": "answer"} for mid in message_ids]
        adapter = SimpleNamespace(public_output_references=output_refs)
        store = SimpleNamespace(_db_for_key=lambda key: db)
        source = SimpleNamespace(chat_id="42", thread_id=None)
        await record_public_delivery(store, adapter, source, "live", prepared, ["2", "3", "2"])
        assert {r["kind"] for r in db.public_context_state("live")["accepted"]} == {"output_pending"}
        db.close()
        db = SessionDB(tmp_path / "state.db")
        state = await reconcile_public_outputs(db, adapter, source, "live", db.public_context_state("live"))
        assert {(r["message_id"], r["kind"]) for r in state["accepted"]} == {("2", "output"), ("3", "output")}
        assert calls == [("2", "3"), ("2", "3")]
        db.close()
    asyncio.run(scenario())

@pytest.mark.parametrize("multimodal", [False, True])
def test_actual_codex_turn_receives_durably_admitted_public_payload(tmp_path, monkeypatch, multimodal):
    import run_agent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = {**batch(db), "content": "PUBLIC CARDS AND CRON:\nsyzygy-canary"}
    agent = run_agent.AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                              api_mode="codex_app_server", quiet_mode=True, skip_context_files=True,
                              skip_memory=True, session_id="live", session_db=db)
    captured = []
    def provider_boundary(**kwargs):
        assert db.public_context_was_accepted("live", prepared)
        captured.append(kwargs["user_message"])
        return {"completed": True, "final_response": "ok", "messages": kwargs["messages"]}
    monkeypatch.setattr(agent, "_run_codex_app_server_turn", provider_boundary)
    monkeypatch.setattr(agent, "_spawn_background_review", lambda *a, **k: None)
    request = [{"type": "text", "text": "discuss"}, {"type": "image_url", "image_url": {"url": "https://stub.invalid/a.png"}}] if multimodal else "discuss"
    agent.run_conversation(request, persist_user_message="discuss", persist_user_display_metadata={"public_context": prepared})
    assert db._public_bytes_in_content(captured[0], prepared["content"])
    if multimodal:
        assert "image_url" in json.dumps(captured)
    db.close()


def test_stream_physical_ids_survive_segment_and_preview_rotation():
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.platforms.base import SendResult
    async def scenario():
        sent = []
        async def send(*args, **kwargs):
            mid = str(len(sent) + 1)
            sent.append(kwargs.get("content"))
            return SendResult(success=True, message_id=mid, continuation_message_ids=[mid + "-tail"])
        adapter = SimpleNamespace(send=send)
        consumer = GatewayStreamConsumer(adapter, "42")
        assert await consumer._send_commentary("commentary")
        consumer._fallback_final_send = True
        consumer._accumulated = "segment tail"
        await consumer._flush_segment_tail_on_edit_failure()
        await consumer._send_with_flood_retry(content="final", reply_to=None, retry_log="retry %s")
        consumer._preview_message_ids.clear()
        consumer._segment_preview_message_ids.clear()
        assert set(consumer.delivered_message_ids) == {"1", "1-tail", "2", "2-tail", "3", "3-tail"}
        assert sent == ["commentary", "segment tail", "final"]
    asyncio.run(scenario())


@pytest.mark.parametrize("multimodal", [False, True])
def test_actual_openai_wire_contains_atomic_public_context_and_media(tmp_path, monkeypatch, multimodal):
    import httpx
    import run_agent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Configure the synthetic model through the real model-capability surface;
    # otherwise unknown models intentionally receive a vision-tool text fallback.
    (tmp_path / "config.yaml").write_text("model:\n  supports_vision: true\n", encoding="utf-8")
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *args, **kwargs: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *args, **kwargs: 0.0)
    db = SessionDB(tmp_path / "state.db")
    route(db)
    prepared = {**batch(db), "content": "PUBLIC CARD AND CRON:\natomic-wire-canary"}
    captured = []
    def transport(transport, request):
        assert request.url.host == "example.invalid"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "unit-model", "context_length": 256000}]})
        assert request.method == "POST"
        assert db.public_context_was_accepted("live", prepared), "Provider request preceded durable context admission"
        body = json.loads(request.read())
        captured.append(body["messages"])
        chunk = {"id": "public-input", "object": "chat.completion.chunk", "created": 1, "model": "unit-model"}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Checked public context"}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        ]
        wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=wire.encode())
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    agent = run_agent.AIAgent(model="unit-model", api_key="stub", base_url="https://example.invalid/v1",
                              provider="openai", api_mode="chat_completions", quiet_mode=True,
                              skip_context_files=True, skip_memory=True, session_id="live", session_db=db)
    monkeypatch.setattr(agent, "_spawn_background_review", lambda *args, **kwargs: None)
    image = {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}
    prompt = [{"type": "text", "text": "discuss"}, image] if multimodal else "discuss"
    result = agent.run_conversation(prompt, persist_user_message="discuss",
                                    persist_user_display_metadata={"public_context": prepared})
    assert result["final_response"] == "Checked public context"
    assert len(captured) == 1
    users = [message for message in captured[0] if message["role"] == "user"]
    assert len(users) == 1 and json.dumps(users).count("atomic-wire-canary") == 1
    assert all("atomic-wire-canary" not in json.dumps(message) for message in captured[0] if message["role"] == "system")
    if multimodal:
        assert image in users[0]["content"]
    db.close()
