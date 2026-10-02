"""Exercise real proxy/queued entry paths rather than only receipt helpers."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.run import GatewayRunner
from hermes_state import SessionDB


def _db_batch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("live", source="test-public-chat", session_key="bot:chat", chat_id="42")
    state = db.public_context_state("live")
    batch = {**{k: state[k] for k in ("version", "source", "session_key", "generation")},
             "input_owner": "proxy-input", "content": "PUBLIC CRON:\nsyzygy-canary",
             "refs": [{"chat_id": "42", "message_id": "1", "revision": "v1"}]}
    return db, batch


@pytest.mark.parametrize("fail_write", [False, True])
def test_actual_proxy_entry_commits_exact_public_input_before_http(tmp_path, monkeypatch, fail_write):
    async def scenario():
        db, batch = _db_batch(tmp_path)
        if fail_write:
            db._execute_write(lambda c: c.execute(
                "CREATE TRIGGER refuse_input BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'input disk failure'); END"))
        calls = []
        class Response:
            status = 200
            @property
            def content(self):
                return self
            async def iter_any(self):
                yield b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n'
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        class HTTP:
            def post(self, url, *, json, headers):
                assert db.public_context_was_accepted("live", batch)
                assert batch["content"] in json["messages"][-1]["content"]
                calls.append(json)
                return Response()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: HTTP())
        monkeypatch.setattr("agent.secret_scope.get_secret", lambda *args, **kwargs: None)
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace(_db_for_key=lambda key: db)
        runner._get_proxy_url = lambda: "https://proxy.invalid"
        runner._run_still_current_fn = lambda *args: lambda: True
        runner._thread_metadata_for_source = lambda *args: None
        runner._proxy_stream_consumer = lambda *args: None
        runner._delivery_adapter_for = lambda source: None
        source = SimpleNamespace(chat_id="42", thread_id=None)
        kwargs = dict(message="discuss", context_prompt="stable system", history=[], source=source,
                      session_id="live", session_key="bot:chat", inbound_message_id="1",
                      persist_user_message="discuss", persist_user_display_metadata={"public_context": batch})
        if fail_write:
            with pytest.raises(Exception, match="input disk failure"):
                await runner._run_agent_inner(**kwargs)
            assert not calls and not db.public_context_state("live")["accepted"]
        else:
            result = await runner._run_agent_inner(**kwargs)
            assert len(calls) == 1 and result["agent_persisted"] is True
            rows = db.get_messages("live")
            assert [m["role"] for m in rows] == ["user", "assistant"]
            assert rows[0]["content"] == "discuss" and batch["content"] in rows[0]["api_content"]
            assert rows[1]["content"] == "answer"
        db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["worker_failure", "worker_cancel", "transformed_final"])
def test_actual_agent_entry_records_stream_output_through_terminal_cleanup(tmp_path, monkeypatch, outcome):
    """Real executor/stream/cancellation wiring; only the provider is deterministic."""
    import threading
    from gateway.config import Platform
    from tests.gateway.test_run_progress_topics import MetadataEditProgressCaptureAdapter, _run_with_agent

    async def scenario():
        db, batch = _db_batch(tmp_path)
        physical_send = threading.Event()
        worker_release = threading.Event()
        entered_runners, revisions = [], {}

        class Adapter(MetadataEditProgressCaptureAdapter):
            async def send(self, chat_id, content, **kwargs):
                result = await super().send(chat_id, content, **kwargs)
                revisions[str(result.message_id)] = content
                physical_send.set()
                return result

            async def edit_message(self, chat_id, message_id, content, **kwargs):
                result = await super().edit_message(chat_id, message_id, content, **kwargs)
                revisions[str(message_id)] = content
                return result

            async def public_output_references(self, source, *, message_ids):
                return [{"chat_id": "42", "message_id": mid, "revision": revisions[mid]} for mid in message_ids]

        class Agent:
            def __init__(self, **kwargs):
                self.stream_delta_callback = kwargs.get("stream_delta_callback")
                self.tools = []

            def run_conversation(self, message, **kwargs):
                assert self.stream_delta_callback is not None
                self.stream_delta_callback("commentary")
                assert physical_send.wait(timeout=5), "The actual stream consumer never sent"
                if outcome == "worker_failure":
                    raise RuntimeError("provider failed after commentary")
                if outcome == "worker_cancel":
                    assert worker_release.wait(timeout=5)
                return {"final_response": "complete answer", "response_transformed": True,
                        "messages": [], "api_calls": 1}

        ordinary_entry = GatewayRunner._run_agent

        async def public_entry(runner, **kwargs):
            entered_runners.append(runner)
            runner.session_store._db_for_key = lambda key: db
            kwargs["persist_user_display_metadata"] = {"public_context": batch}
            return await ordinary_entry(runner, **kwargs)

        monkeypatch.setattr(GatewayRunner, "_run_agent", public_entry)
        task = asyncio.create_task(_run_with_agent(
            monkeypatch, tmp_path, Agent, session_id="live", platform=Platform.MATRIX,
            chat_id="42", chat_type="group", thread_id=None, adapter_cls=Adapter,
            config_data={"display": {"tool_progress": "off", "interim_assistant_messages": False},
                         "streaming": {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1}}))
        if outcome == "worker_cancel":
            deadline = asyncio.get_running_loop().time() + 5
            while not physical_send.is_set():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0)
            task.cancel()
            worker_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "worker_failure":
            with pytest.raises(RuntimeError, match="provider failed after commentary"):
                await task
        else:
            adapter, result = await task
            if outcome == "transformed_final":
                assert result["already_sent"] is True
                assert any(edit["content"] == "complete answer" for edit in adapter.edits)
        refs = db.public_context_state("live")["accepted"]
        assert refs and all(ref["kind"] == "output" and ref["revision"] == revisions[ref["message_id"]]
                            for ref in refs)
        if outcome == "transformed_final":
            assert any(ref["revision"] == "complete answer" for ref in refs)
        assert entered_runners and not entered_runners[0]._running_agents
        db.close()

    asyncio.run(scenario())


def test_queued_first_response_cancellation_records_text_before_held_media(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from tests.gateway.test_delivery_ledger_producer import _Adapter

    async def scenario():
        db, batch = _db_batch(tmp_path)
        held_media = asyncio.Event()

        class Adapter(_Adapter):
            async def send_document(self, **kwargs):
                held_media.set()
                await asyncio.Event().wait()

            async def public_output_references(self, source, *, message_ids):
                return [{"chat_id": "42", "message_id": mid, "revision": "delivered"} for mid in message_ids]

        adapter = Adapter()
        adapter._record_delivery_obligation = AsyncMock(return_value=None)
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace(_db_for_key=lambda key: db)
        runner._pop_post_delivery_callback = lambda *args: None
        source = SessionSource(platform=Platform.SLACK, chat_id="42", chat_type="channel")
        turn = SimpleNamespace(mute_notification_reply=False, source=source,
                               session_id="live", session_key="bot:chat", run_generation=1,
                               event_message_id="1", inbound_message_id="1", _status_thread_metadata=None,
                               stream_consumer_holder=[None], persist_user_display_metadata={"public_context": batch})
        doc = tmp_path / "queued.pdf"
        doc.write_bytes(b"queued attachment")
        response = {"final_response": f"answer\n\nMEDIA: {doc}"}
        task = asyncio.create_task(runner._run_agent_deliver_first_response(turn, adapter, response, response, None))
        await asyncio.wait_for(held_media.wait(), timeout=5)
        assert adapter.sent == ["answer"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [(ref["message_id"], ref["kind"]) for ref in db.public_context_state("live")["accepted"]] == [
            ("m1", "output")]
        assert adapter.sent == ["answer"]
        db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["worker_failure", "worker_cancel", "reconcile", "reset", "write_failure"])
def test_stream_cleanup_records_final_revision_before_releasing_owner(tmp_path, outcome):
    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.turn_context import TurnContext
    from tests.gateway.test_delivery_ledger_producer import _Adapter

    async def scenario():
        db, batch = _db_batch(tmp_path)
        revision, released, edits = ["commentary"], [], []

        class Adapter(_Adapter):
            async def edit_message(self, chat_id, message_id, content, **kwargs):
                assert not released
                edits.append(content)
                revision[0] = "final"
                return SendResult(success=True, message_id=message_id)

            async def public_output_references(self, source, *, message_ids):
                assert not released, "The next owner must not read before our output is recorded"
                return [{"chat_id": "42", "message_id": mid, "revision": revision[0]} for mid in message_ids]

        adapter = Adapter()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace(_db_for_key=lambda key: db)
        runner._delivery_adapter_for = lambda source: adapter
        runner._release_running_agent_state = lambda *args, **kwargs: released.append(True)
        runner._draining = False
        source = SessionSource(platform=Platform.SLACK, chat_id="42", chat_type="channel")
        consumer = GatewayStreamConsumer(adapter, "42")
        delivered = await adapter.send("42", "commentary")
        consumer._track_delivered_result(delivered)
        consumer._message_id = delivered.message_id
        consumer._final_content_delivered = True
        consumer.delivered_final_matches = lambda text: False
        turn = TurnContext(source=source, session_id="live", session_key="bot:chat", run_generation=1,
                           stream_consumer_holder=[consumer], _run_still_current=lambda: outcome != "reset",
                           persist_user_display_metadata={"public_context": batch})
        completed = {"final_response": "complete answer"} if outcome == "reconcile" else None
        if outcome == "reset":
            db.reset_public_context_route("test-public-chat", "bot:chat", {"chat_id": "42", "message_id": "7"})
        elif outcome == "write_failure":
            db._execute_write(lambda conn: conn.execute(
                "CREATE TRIGGER refuse_output BEFORE INSERT ON public_context_receipts "
                "WHEN NEW.kind = 'output_pending' BEGIN SELECT RAISE(ABORT, 'output disk failure'); END"))

        async def idle_task():
            await asyncio.Event().wait()

        tracking = asyncio.create_task(idle_task())

        async def worker_lifecycle():
            try:
                if outcome == "worker_cancel":
                    raise asyncio.CancelledError()
                if outcome == "worker_failure":
                    raise RuntimeError("worker failed after commentary")
            finally:
                await runner._run_agent_cleanup_turn_tasks(
                    turn, progress_task=None, log_task=None, interrupt_monitor=None,
                    _notify_task=None, tracking_task=tracking, stream_task=None, completed_response=completed)

        if outcome == "worker_cancel":
            expected = asyncio.CancelledError
        elif outcome == "worker_failure":
            expected = RuntimeError
        elif outcome == "reset":
            from hermes_state_public_context import PublicContextAdmissionError
            expected = PublicContextAdmissionError
        elif outcome == "write_failure":
            import sqlite3
            expected = sqlite3.IntegrityError
        else:
            expected = None
        if expected:
            with pytest.raises(expected):
                await worker_lifecycle()
        else:
            await worker_lifecycle()
        assert released == [True] and tracking.cancelled()
        refs = db.public_context_state("live")["accepted"]
        if outcome in {"reset", "write_failure"}:
            assert not refs
        else:
            assert [(ref["message_id"], ref["revision"], ref["kind"]) for ref in refs] == [
                ("m1", "final" if outcome == "reconcile" else "commentary", "output")]
        assert edits == (["complete answer"] if outcome == "reconcile" else [])
        assert adapter.sent == ["commentary"]
        db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["cancel", "reset", "write_failure"])
def test_base_cancellation_preserves_confirmed_text_before_held_attachment(tmp_path, outcome):
    """A confirmed text remains our output when cancellation interrupts its attachment."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.public_context import prepare_public_admission
    from gateway.session import SessionSource

    async def scenario():
        db, _ = _db_batch(tmp_path)
        store = SimpleNamespace(_db_for_key=lambda key: db)
        attachment_started = asyncio.Event()
        sent, prepared_states = [], []
        reference_calls = []

        class Adapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(PlatformConfig(enabled=True), Platform.SLACK)

            async def connect(self, *, is_reconnect=False):
                return True

            async def disconnect(self):
                pass

            async def get_chat_info(self, chat_id):
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                sent.append(content)
                return SendResult(success=True, message_id="2")

            async def send_document(self, **kwargs):
                attachment_started.set()
                await asyncio.Event().wait()

            async def prepare_public_context(self, event, *, state):
                prepared_states.append(state)
                accepted = {ref["message_id"] for ref in state["accepted"]}
                return {"refs": [{"chat_id": "42", "message_id": mid, "revision": "v1"}
                                 for mid in ("1", "2", "3") if mid not in accepted],
                        "content": "PUBLIC CRON:\nsyzygy-canary"}

            async def public_output_references(self, source, *, message_ids):
                reference_calls.append(message_ids)
                if len(reference_calls) == 1:
                    raise RuntimeError("revision lookup temporarily unavailable")
                return [{"chat_id": "42", "message_id": mid, "revision": "v1"}
                        for mid in message_ids]

        adapter = Adapter()
        adapter._record_delivery_obligation = AsyncMock(return_value=None)
        adapter._start_typing_refresh = lambda *args: None
        adapter.gateway_runner = SimpleNamespace(
            session_store=store, _delivery_adapter_for=lambda source: adapter)
        source = SessionSource(platform=Platform.SLACK, chat_id="42", chat_type="channel")
        event = MessageEvent(text="discuss", message_type=MessageType.TEXT,
                             source=source, message_id="1")
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"test attachment")

        async def handler(event):
            admission = await prepare_public_admission(store, adapter, event, "live", "bot:chat")
            # Only the triggering input is present before our reply is delivered.
            admission["refs"] = [ref for ref in admission["refs"] if ref["message_id"] == "1"]
            db.append_message("live", "user", admission["content"],
                              display_metadata={"public_context": admission})
            return f"answer\n\nMEDIA: {doc}"

        adapter._message_handler = handler
        assert adapter._start_session_processing(event, "bot:chat")
        task = adapter._session_tasks["bot:chat"]
        await asyncio.wait_for(attachment_started.wait(), timeout=5)
        assert sent == ["answer"]
        if outcome == "reset":
            db.reset_public_context_route("test-public-chat", "bot:chat", {"chat_id": "42", "message_id": "7"})
        elif outcome == "write_failure":
            db._execute_write(lambda conn: conn.execute(
                "CREATE TRIGGER refuse_output BEFORE INSERT ON public_context_receipts "
                "WHEN NEW.kind = 'output_pending' BEGIN SELECT RAISE(ABORT, 'output disk failure'); END"))
        task.cancel()
        if outcome == "reset":
            from hermes_state_public_context import PublicContextAdmissionError
            expected = PublicContextAdmissionError
        elif outcome == "write_failure":
            import sqlite3
            expected = sqlite3.IntegrityError
        else:
            expected = asyncio.CancelledError
        with pytest.raises(expected):
            await task
        assert "bot:chat" not in adapter._active_sessions and "bot:chat" not in adapter._session_tasks
        if outcome != "cancel":
            assert not any(ref["kind"].startswith("output")
                           for ref in db.public_context_state("live")["accepted"])
            assert sent == ["answer"] and not reference_calls
            db.close()
            return
        assert {(ref["message_id"], ref["kind"]) for ref in db.public_context_state("live")["accepted"]} == {
            ("1", "input"), ("2", "output_pending")}
        db.close()
        db = SessionDB(tmp_path / "state.db")
        next_event = MessageEvent(text="continue", message_type=MessageType.TEXT,
                                  source=source, message_id="3")
        next_admission = await prepare_public_admission(store, adapter, next_event, "live", "bot:chat")
        assert [ref["message_id"] for ref in next_admission["refs"]] == ["3"]
        assert ("2", "output") in {(ref["message_id"], ref["kind"])
                                   for ref in prepared_states[-1]["accepted"]}
        assert sent == ["answer"] and reference_calls == [("2",), ("2",)]
        db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("recursion_capped", [False, True])
def test_interrupted_queued_entry_records_confirmed_stream_before_early_return(tmp_path, recursion_capped):
    async def scenario():
        db, batch = _db_batch(tmp_path)
        source = SimpleNamespace(chat_id="42", thread_id=None)
        async def references(source, *, message_ids):
            return [{"chat_id": "42", "message_id": mid, "revision": "output"} for mid in message_ids]
        adapter = SimpleNamespace(_active_sessions={}, public_output_references=references)
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace(_db_for_key=lambda key: db)
        runner._MAX_INTERRUPT_DEPTH = 0 if recursion_capped else 5
        runner._delivery_adapter_for = lambda source: adapter
        runner._await_stream_task = AsyncMock()
        runner._is_goal_continuation_event = lambda event: False
        runner._session_key_for_source = lambda source: "bot:chat"
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None)
        turn = SimpleNamespace(source=source, session_id="live", session_key="bot:chat", run_generation=1,
                               _interrupt_depth=0, history=[], _status_thread_metadata=None,
                               persist_user_display_metadata={"public_context": batch},
                               stream_consumer_holder=[SimpleNamespace(delivered_message_ids=["2", "3"])],
                               result_holder=[{"interrupted": True}])
        pending_event = None if recursion_capped else SimpleNamespace(source=source, reply_expected=None)
        result = {"interrupted": True, "messages": []}
        await runner._run_agent_queued_followup(turn, adapter, "next", pending_event, {}, result, object())
        assert {(r["message_id"], r["kind"]) for r in db.public_context_state("live")["accepted"]} == {
            ("2", "output"), ("3", "output")}
        runner._await_stream_task.assert_awaited_once()
        db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["restart_text", "restart_multimodal", "reset_before_post"])
def test_proxy_preserves_admitted_context_after_restart_and_fences_reset_before_post(tmp_path, monkeypatch, mode):
    async def scenario():
        db, first_batch = _db_batch(tmp_path)
        calls, current = [], [True]
        typing_started, release_typing = asyncio.Event(), asyncio.Event()
        class Response:
            status = 200
            @property
            def content(self):
                return self
            async def iter_any(self):
                yield b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n'
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        class HTTP:
            def post(self, url, *, json, headers):
                assert current[0], "A stale turn must never cross the remote provider boundary"
                assert db.public_context_was_accepted("live", active_batch[0])
                calls.append(json)
                return Response()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: HTTP())
        monkeypatch.setattr("agent.secret_scope.get_secret", lambda *args, **kwargs: None)
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace(_db_for_key=lambda key: db)
        runner._get_proxy_url = lambda: "https://proxy.invalid"
        runner._run_still_current_fn = lambda *args: lambda: current[0]
        runner._thread_metadata_for_source = lambda *args: None
        runner._proxy_stream_consumer = lambda *args: None
        async def held_typing(*args, **kwargs):
            typing_started.set()
            await release_typing.wait()
        adapter = SimpleNamespace(send_typing=held_typing) if mode == "reset_before_post" else None
        runner._delivery_adapter_for = lambda source: adapter
        source = SimpleNamespace(chat_id="42", thread_id=None)
        image = {"type": "image_url", "image_url": {"url": "https://proxy.invalid/image.png"}}
        first_request = [{"type": "text", "text": "discuss"}, image] if mode == "restart_multimodal" else "discuss"
        active_batch = [first_batch]
        kwargs = dict(message=first_request, context_prompt="stable system", history=[], source=source,
                      session_id="live", session_key="bot:chat", inbound_message_id="1",
                      persist_user_message="discuss", persist_user_display_metadata={"public_context": first_batch})
        if mode == "reset_before_post":
            task = asyncio.create_task(runner._run_agent_inner(**kwargs))
            await asyncio.wait_for(typing_started.wait(), timeout=5)
            assert db.public_context_was_accepted("live", first_batch)
            db.reset_public_context_route("test-public-chat", "bot:chat", {"chat_id": "42", "message_id": "7"})
            current[0] = False
            release_typing.set()
            result = await task
            assert not calls and result["final_response"] == ""
            assert [row["role"] for row in db.get_messages("live")] == ["user"]
        else:
            await runner._run_agent_inner(**kwargs)
            first_wire = calls[0]["messages"][-1]["content"]
            assert db._public_bytes_in_content(first_wire, first_batch["content"])
            db.close()
            # A fresh local handle and a stateless remote exercise fallback history.
            db = SessionDB(tmp_path / "state.db")
            history = db.get_messages("live")
            state = db.public_context_state("live")
            assert state["accepted"] and (history[0]["api_content"] or history[0]["content"]) == first_wire
            second_batch = {**{k: state[k] for k in ("version", "source", "session_key", "generation")},
                            "input_owner": "second-proxy-input", "refs": [], "content": ""}
            active_batch[0] = second_batch
            kwargs.update(message="continue", history=history, persist_user_message="continue",
                          inbound_message_id="2", persist_user_display_metadata={"public_context": second_batch})
            await runner._run_agent_inner(**kwargs)
            assert len(calls) == 2
            previous_user = calls[1]["messages"][1]
            assert previous_user["content"] == first_wire
            assert json.dumps(calls[1]["messages"]).count("syzygy-canary") == 1
            assert "syzygy-canary" not in json.dumps(calls[1]["messages"][-1])
            if mode == "restart_multimodal":
                assert image in previous_user["content"]
        db.close()
    asyncio.run(scenario())
