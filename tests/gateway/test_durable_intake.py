"""Physical transport adoption, real SQLite receipts and the existing FIFO."""
import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_intake import DurableIntakeRefused, intake_metadata
from gateway.session import SessionContext, SessionSource, SessionStore
from hermes_state import SessionDB
from hermes_state_intake import GatewayIntakeError


def snapshot(mid="1", rev="1", text="request", receiver="worker"):
    return {"version": 1, "receiver_id": receiver, "physical_chat_id": "42",
            "physical_message_id": mid, "revision": rev, "event": {"text": text}}


def owner(home):
    return {"transport_profile": "default", "runtime_profile": "default",
            "authorization_home": str(home), "runtime_home": str(home)}


def adopt(db, home, value=None, dispatcher="boot1", key="bot:42"):
    return db.adopt_gateway_intake(value or snapshot(), source="telegram", session_key=key,
                                    owner=owner(home), dispatch_owner=dispatcher)


def test_existing_state_db_upgrade_preserves_history_generation_and_reconciles_additive_intake(tmp_path):
    from hermes_state_common import SCHEMA_SQL, SCHEMA_VERSION
    # Reconstruct the immediately preceding schema: same durable tables, without
    # the new floor column or either additive public-context/intake table.
    legacy_sql = SCHEMA_SQL.replace("    public_context_floor TEXT,\n", "")
    start = legacy_sql.index("-- Public message revisions admitted")
    end = legacy_sql.index("-- Per-backend liveness heartbeat", start)
    legacy_sql = legacy_sql[:start] + legacy_sql[end:]
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(legacy_sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.execute("INSERT INTO sessions (id, source, session_key, started_at) VALUES ('legacy', 'telegram', 'bot:42', 1)")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('legacy', 'user', 'existing history', 2)")
        conn.execute("INSERT INTO conversation_generations VALUES ('telegram', 'bot:42', 7)")
        assert "public_context_floor" not in {row[1] for row in conn.execute("PRAGMA table_info(conversation_generations)")}
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'gateway_intake'").fetchone() is None
    db = SessionDB(path)  # ordinary supported initialization, no migration helper bypass
    assert [row["content"] for row in db.get_messages("legacy")] == ["existing history"]
    assert db.public_context_state("legacy")["generation"] == 7
    assert db.public_context_state("legacy")["floor"] is None
    receipt = adopt(db, tmp_path, snapshot(text="public input"))
    batch = {"version": 1, "source": "telegram", "session_key": "bot:42", "generation": 7,
             "content": "public input", "refs": [{"chat_id": "42", "message_id": "1", "revision": "1"}]}
    db.append_message("legacy", "user", "public input", display_metadata={"public_context": batch, "gateway_intake": [receipt]})
    assert db.gateway_intake_was_consumed("legacy", [receipt])
    assert len(db.public_context_state("legacy")["accepted"]) == 1
    db.reset_public_context_route("telegram", "bot:42", {"chat_id": "42", "message_id": "9"})
    db.close()
    reopened = SessionDB(path)
    assert [row["content"] for row in reopened.get_messages("legacy")] == ["existing history", "public input"]
    state = reopened.public_context_state("legacy")
    assert state["generation"] == 8 and state["floor"] == {"chat_id": "42", "message_id": "9"}
    assert state["accepted"] == [] and reopened.pending_gateway_intakes() == []
    fresh = adopt(reopened, tmp_path, snapshot(mid="10", text="post-reset input"))
    assert fresh["generation"] == 8
    reopened.create_session("after-reset", source="telegram", session_key="bot:42")
    reopened.append_message("after-reset", "user", "post-reset input", display_metadata={"gateway_intake": [fresh]})
    assert reopened.gateway_intake_was_consumed("after-reset", [fresh])
    reopened.close()


def test_atomic_user_row_consumption_and_failed_insert_rollback(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("live", source="telegram", session_key="bot:42")
    receipt = adopt(db, tmp_path)
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_user BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT, 'write failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match="write failed"):
        db.append_message("live", "user", "request", display_metadata={"gateway_intake": [receipt]})
    assert not db.get_messages("live")
    assert db.gateway_intake_is_pending([receipt])
    db._execute_write(lambda conn: conn.execute("DROP TRIGGER reject_user"))
    db.append_message("live", "user", "request", display_metadata={"gateway_intake": [receipt]})
    assert db.gateway_intake_was_consumed("live", [receipt])
    db.close()
    restored = SessionDB(tmp_path / "state.db")
    assert restored.gateway_intake_was_consumed("live", [receipt])
    assert restored.pending_gateway_intakes() == []
    assert adopt(restored, tmp_path, dispatcher="boot2")["dispatch"] is False
    restored.close()


def test_reset_and_session_rotation_cannot_reconsume_old_physical_input(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("before", source="telegram", session_key="bot:42")
    receipt = adopt(db, tmp_path)
    db.append_message("before", "user", "request", display_metadata={"gateway_intake": [receipt]})
    db.create_session("rotation", source="telegram", session_key="bot:42")
    assert not db.gateway_intake_was_consumed("rotation", [receipt])
    db.reset_public_context_route("telegram", "bot:42", {"chat_id": "42", "message_id": "9"})
    assert not db.pending_gateway_intakes()
    with pytest.raises(GatewayIntakeError, match="generation"):
        adopt(db, tmp_path)
    db.create_session("after", source="telegram", session_key="bot:42")
    with pytest.raises(GatewayIntakeError, match="reset"):
        db.append_message("after", "user", "request", display_metadata={"gateway_intake": [receipt]})
    fresh = adopt(db, tmp_path, snapshot("10"))
    assert fresh["generation"] == 1
    db.close()


def test_immutable_payload_route_and_profile_are_bound_before_dispatch(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first = adopt(db, tmp_path)
    assert first["dispatch"] is True
    assert adopt(db, tmp_path)["dispatch"] is False
    with pytest.raises(GatewayIntakeError, match="immutable"):
        adopt(db, tmp_path, snapshot(text="changed"))
    with pytest.raises(GatewayIntakeError, match="route"):
        adopt(db, tmp_path, key="bot:43")
    second = adopt(db, tmp_path, snapshot("2"))
    rows = db.pending_gateway_intakes()
    assert [r["receipt_id"] for r in rows] == [first["receipt_id"], second["receipt_id"]]
    db.close()
    reopened = SessionDB(tmp_path / "state.db")
    assert adopt(reopened, tmp_path, dispatcher="boot2")["dispatch"] is True
    other = SessionDB(tmp_path / "other" / "state.db")
    assert other.pending_gateway_intakes() == []
    reopened.close()
    other.close()


def test_provider_gate_refuses_unconsumed_input_and_consumed_input_is_never_replayed(tmp_path, monkeypatch):
    from agent.turn_context import _persist_turn_start
    import run_agent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("live", source="telegram", session_key="bot:42")
    receipt = adopt(db, tmp_path)
    metadata = {"gateway_intake": [receipt]}
    blocked = SimpleNamespace(session_id="live", _session_db=db, _ensure_db_session=lambda: None,
                              _persist_session=lambda *args: None)
    with pytest.raises(GatewayIntakeError, match="persisted"):
        _persist_turn_start(blocked, [{"role": "user", "content": "request", "display_metadata": metadata}], [], None)
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)
    agent = run_agent.AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                              api_mode="codex_app_server", quiet_mode=True, skip_context_files=True,
                              skip_memory=True, session_id="live", session_db=db)
    reached = []
    def provider_boundary(**kwargs):
        assert db.gateway_intake_was_consumed("live", [receipt])
        reached.append(kwargs["user_message"])
        raise RuntimeError("provider accepted; connection lost")
    monkeypatch.setattr(agent, "_run_codex_app_server_turn", provider_boundary)
    with pytest.raises(RuntimeError, match="connection lost"):
        agent.run_conversation("request", persist_user_display_metadata=metadata)
    assert reached == ["request"]
    assert not db.pending_gateway_intakes()
    db.close()


class IntakeAdapter(BasePlatformAdapter):
    durable_intake = True

    def __init__(self, receiver="worker"):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False,
                                        extra={"allow_from": "20", "group_allow_from": "20"}), Platform.TELEGRAM)
        self.receiver, self.visible, self.sent = receiver, {}, []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="answer")

    async def send_typing(self, *args, **kwargs):
        pass

    async def stop_typing(self, *args, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    def event(self, mid="1", *, text="request", author="20", rev="1", kind=MessageType.TEXT, chat_id="42"):
        self.visible[mid] = {"text": text, "author": author, "rev": rev, "kind": kind.value}
        source = self.build_source(chat_id=chat_id, chat_type="group", user_id=author)
        source.author_kind_verified = True  # this fixture's current sender directory
        return MessageEvent(text=text, message_type=kind, message_id=mid,
                            source=source,
                            metadata={"revision": rev})

    def serialize_durable_intake(self, event):
        live = self.visible.get(event.message_id)
        if not live or live["rev"] != event.metadata["revision"] or live["author"] != event.source.user_id:
            raise DurableIntakeRefused("current message version is unavailable")
        return {"version": 1, "receiver_id": self.receiver, "physical_chat_id": event.source.chat_id,
                "physical_message_id": event.message_id, "revision": live["rev"],
                "event": {"text": event.text, "source": event.source.to_dict(),
                          "kind": event.message_type.value, "allow_gateway_control": event.allow_gateway_control,
                          "media_urls": list(event.media_urls), "media_types": list(event.media_types),
                          "media_text_inlined": list(event.media_text_inlined)}}

    def restore_durable_intake(self, frozen):
        if frozen["receiver_id"] != self.receiver:
            raise DurableIntakeRefused("receiving account changed")
        live = self.visible.get(frozen["physical_message_id"])
        if live is None or live["rev"] != frozen["revision"]:
            raise DurableIntakeRefused("current source no longer proves original revision")
        if live.get("recipient") == "other":
            raise DurableIntakeRefused("current recipient excludes this bot")
        source = self.build_source(chat_id=frozen["physical_chat_id"], chat_type="group", user_id=live["author"])
        source.author_kind_verified = True  # freshly revalidated, never from frozen source
        return MessageEvent(text=frozen["event"]["text"], message_id=frozen["physical_message_id"],
                            source=source,
                            message_type=MessageType(frozen["event"]["kind"]),
                            metadata={"revision": live["rev"]},
                            media_urls=list(frozen["event"]["media_urls"]),
                            media_types=list(frozen["event"]["media_types"]),
                            media_text_inlined=list(frozen["event"]["media_text_inlined"]),
                            allow_gateway_control=frozen["event"]["allow_gateway_control"])


def wire(runner, adapter):
    adapter.gateway_runner = runner
    runner._wire_adapter_handlers(adapter)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(sessions_dir=tmp_path / "sessions", group_sessions_per_user=False)
    runner._primary_profile_name, runner._draining = "default", False
    runner._busy_input_mode, runner._busy_text_mode = "steer", "queue"
    runner.pairing_store, runner.pairing_stores = None, {}
    runner.session_store = SessionStore(runner.config.sessions_dir, runner.config)
    adapter = IntakeAdapter()
    runner.adapters, runner._profile_adapters = {adapter.platform: adapter}, {}
    wire(runner, adapter)
    handled = []
    async def handle(event):
        admitted = await runner._hm_admit_event(event)
        if admitted is not None:
            current, source, _ = admitted
            route = runner.session_store.get_or_create_session(source)
            db = runner.session_store._db_for_key(route.session_key)
            db.append_message(route.session_id, "user", current.text, display_metadata=intake_metadata(current))
            assert db.gateway_intake_was_consumed(route.session_id, current._gateway_intake_receipts)
            handled.append(current.message_id)
            if route.session_key not in adapter._pending_messages:
                successor = runner._promote_queued_event(route.session_key, adapter, None)
                if successor is not None:
                    adapter._pending_messages[route.session_key] = successor
        return None
    adapter.set_message_handler(handle)
    yield SimpleNamespace(runner=runner, adapter=adapter, handled=handled, handle=handle, home=tmp_path)
    runner.session_store.close_all_db_handles()


async def finish_tasks(adapter):
    for _ in range(20):
        tasks = list(adapter._background_tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks)
    raise AssertionError("transport queue did not settle")


def use_real_command_pipeline(rig, monkeypatch, *, before_provider=None):
    """Real Base/command/adoption/SQLite path; only the provider transport is offline."""
    import run_agent
    monkeypatch.setattr("agent.turn_context._maybe_title_session_at_turn_start", lambda *a, **k: None)
    rig.runner._external_drain_active = False
    rig.runner._persist_active_agents = lambda: None
    async def off_loop(function, *args):
        return await asyncio.to_thread(function, *args)
    rig.runner._run_in_executor_with_context = off_loop
    payloads = []
    async def provider_turn(event, source, key, generation):
        if before_provider is not None:
            await before_provider(event)
        route = rig.runner.session_store.get_or_create_session(source)
        db = rig.runner.session_store._db_for_key(key)
        agent = run_agent.AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai",
                                  api_mode="codex_app_server", quiet_mode=True, skip_context_files=True,
                                  skip_memory=True, session_id=route.session_id, session_db=db)
        def transport(**kwargs):
            assert db.gateway_intake_was_consumed(route.session_id, event._gateway_intake_receipts)
            payloads.append((event.message_id, kwargs["user_message"]))
            return {"final_response": "done", "completed": True, "api_calls": 1}
        monkeypatch.setattr(agent, "_run_codex_app_server_turn", transport)
        result = agent.run_conversation(event.text, persist_user_display_metadata=intake_metadata(event))
        rig.handled.append(event.message_id)
        next_event, _text = await rig.runner._run_agent_drain_pending(result, rig.adapter, source, key)
        if next_event is not None:
            await rig.runner._validate_durable_intake_event(next_event)
            return await provider_turn(next_event, next_event.source, key, generation)
        return result["final_response"]
    rig.runner._handle_message_with_agent = provider_turn
    rig.adapter.set_message_handler(rig.runner._primary_message_handler())
    return payloads


def register_real_skill(rig, monkeypatch):
    from agent import skill_commands
    directory = rig.home / "skills" / "intake-probe"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: intake-probe\ndescription: offline probe\n---\nUse the exact probe checklist.\n")
    monkeypatch.setattr(skill_commands, "get_skill_commands", lambda: {
        "/intake-probe": {"name": "intake-probe", "skill_dir": str(directory)}})


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/plan release request", "/intake-probe release request"])
async def test_real_llm_command_rewrite_is_adopted_before_actual_provider_payload(rig, monkeypatch, command):
    register_real_skill(rig, monkeypatch)
    payloads = use_real_command_pipeline(rig, monkeypatch)
    event = rig.adapter.event(text=command)
    await rig.adapter.handle_message(event)
    await finish_tasks(rig.adapter)
    assert event._gateway_durable_adopted is True and event.allow_gateway_control is False
    assert event._gateway_intake_control_completed is False
    assert payloads == [("1", event.text)] and "release request" in event.text
    assert event.text != command and event._gateway_intake_snapshot["event"]["text"] == event.text


@pytest.mark.asyncio
async def test_resolved_command_failed_durable_write_has_zero_provider_or_transport_disposition(rig, monkeypatch):
    payloads = use_real_command_pipeline(rig, monkeypatch)
    event = rig.adapter.event(text="/plan release request")
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(event.source))
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_intake BEFORE INSERT ON gateway_intake BEGIN SELECT RAISE(ABORT, 'disk failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match="disk failed"):
        await rig.adapter.handle_message(event)
    assert not payloads and not db.pending_gateway_intakes()
    assert not event._gateway_durable_adopted and not event._gateway_intake_control_completed
    assert not event._gateway_intake_refused and not event._gateway_accepted


@pytest.mark.asyncio
async def test_command_arrival_generation_survives_awaited_resolution_and_reset(rig, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    payloads = use_real_command_pipeline(rig, monkeypatch)
    event = rig.adapter.event(text="/plan release request")
    key = rig.runner._session_key_for_source(event.source)
    db = rig.runner.session_store._db_for_key(key)
    original_ack = rig.runner._send_command_ack
    async def held_ack(source, text, label):
        assert event._gateway_intake_generation == 0
        entered.set()
        await release.wait()
        await original_ack(source, text, label)
    monkeypatch.setattr(rig.runner, "_send_command_ack", held_ack)
    received = asyncio.create_task(rig.adapter.handle_message(event))
    await asyncio.wait_for(entered.wait(), 2)
    db.reset_public_context_route("telegram", key, {"chat_id": "42", "message_id": "9"})
    release.set()
    await received
    await finish_tasks(rig.adapter)
    assert event._gateway_intake_generation == 0  # no post-reset recapture
    assert event._gateway_intake_refused is True
    assert not event._gateway_durable_adopted and not event._gateway_intake_control_completed
    assert not payloads and not db.pending_gateway_intakes()
    fresh = rig.adapter.event("10", text="/plan fresh request")
    monkeypatch.setattr(rig.runner, "_send_command_ack", original_ack)
    await rig.adapter.handle_message(fresh)
    await finish_tasks(rig.adapter)
    assert fresh._gateway_intake_receipts[0]["generation"] == 1
    assert payloads == [("10", fresh.text)]


@pytest.mark.asyncio
async def test_failed_arrival_fence_keeps_deterministic_control_but_refuses_resolved_llm(rig, monkeypatch):
    payloads = use_real_command_pipeline(rig, monkeypatch)
    event = rig.adapter.event(text="/whoami")
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(event.source))
    def unavailable(*_args):
        raise RuntimeError("private failure detail must not enter a snapshot")
    monkeypatch.setattr(db, "gateway_intake_generation", unavailable)
    await rig.adapter.handle_message(event)
    assert event._gateway_intake_generation_unavailable is True
    assert event._gateway_intake_control_completed is True and not event._gateway_durable_adopted
    llm = rig.adapter.event("2", text="/plan release request")
    with pytest.raises(GatewayIntakeError, match="arrival generation fence"):
        await rig.adapter.handle_message(llm)
    assert llm._gateway_intake_generation_unavailable is True
    assert not llm._gateway_durable_adopted and not llm._gateway_intake_control_completed
    assert not llm._gateway_intake_refused and not llm._gateway_accepted
    assert llm._gateway_intake_snapshot is None and not payloads and not db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_cold_command_recovery_preserves_resolved_bytes_without_rerunning_hooks_or_builder(rig, monkeypatch):
    class BeforeUserRowCrash(RuntimeError):
        pass
    crashes = []
    async def crash(event):
        crashes.append(event.message_id)
        raise BeforeUserRowCrash("crash before ordinary row")
    use_real_command_pipeline(rig, monkeypatch, before_provider=crash)
    event = rig.adapter.event(text="/plan release request")
    await rig.adapter.handle_message(event)
    await finish_tasks(rig.adapter)
    assert crashes == ["1"]
    frozen = event.text
    assert event._gateway_durable_adopted is True and not event._gateway_intake_control_completed
    rig.runner.session_store.close_all_db_handles()
    rig.runner.session_store = SessionStore(rig.runner.config.sessions_dir, rig.runner.config)
    wire(rig.runner, rig.adapter)
    payloads = use_real_command_pipeline(rig, monkeypatch)
    monkeypatch.setattr("agent.plan_prompt.build_plan_prompt", lambda *a: pytest.fail("command builder replayed"))
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", lambda *a, **k: pytest.fail("dispatch hook replayed"))
    rig.runner._gateway_intake_dispatch_owner = "after-command-crash"
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert payloads == [("1", frozen)]


@pytest.mark.asyncio
async def test_resolved_command_handoff_returns_before_provider_without_blocking_later_control(rig, monkeypatch):
    gate, started = asyncio.Event(), asyncio.Event()
    async def hold(event):
        started.set()
        await gate.wait()
    payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=hold)
    event = rig.adapter.event(text="/plan release request")
    await asyncio.wait_for(rig.adapter.handle_message(event), 2)
    await asyncio.wait_for(started.wait(), 2)
    assert event._gateway_durable_adopted is True and event._gateway_accepted is True
    assert not event._gateway_intake_control_completed and not payloads
    status = rig.adapter.event("2", text="/whoami")
    await asyncio.wait_for(rig.adapter.handle_message(status), 2)
    assert status._gateway_intake_control_completed is True and not status._gateway_durable_adopted
    gate.set()
    await finish_tasks(rig.adapter)
    assert payloads == [("1", event.text)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["current-proof", "preparation"])
async def test_initial_base_failure_releases_pending_claim_without_retry_spin_and_reconnect_recovers(rig, monkeypatch, stage):
    unavailable = True
    attempts = []
    original_restore = rig.adapter.restore_durable_intake
    def current_source(frozen):
        if stage == "current-proof" and unavailable:
            attempts.append("current-proof")
            raise RuntimeError("current source proof temporarily unavailable")
        return original_restore(frozen)
    async def preparation(event):
        if stage == "preparation" and unavailable:
            attempts.append("preparation")
            raise RuntimeError("preparation temporarily unavailable")
    monkeypatch.setattr(rig.adapter, "restore_durable_intake", current_source)
    payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=preparation)
    event = rig.adapter.event()
    await rig.adapter.handle_message(event)
    assert event._gateway_durable_adopted is True and event._gateway_accepted is True
    await finish_tasks(rig.adapter)
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(event.source))
    assert attempts == [stage] and not payloads
    assert len(db.pending_gateway_intakes(unclaimed_only=True)) == 1
    assert event._gateway_accepted is False

    # The real primary reconnect lifecycle invokes the durable drain; only its
    # unrelated status/voice/delivery collaborators are offline in this fixture.
    unavailable = False
    rig.runner.delivery_router = SimpleNamespace(adapters=rig.runner.adapters)
    rig.runner._failed_platforms = {rig.adapter.platform: {}}
    for name in ("_sync_voice_mode_state_to_adapter", "_bind_voice_input_callback",
                 "_update_platform_runtime_status", "_schedule_planned_restart_replay",
                 "_schedule_resume_pending_sessions"):
        monkeypatch.setattr(rig.runner, name, lambda *a, **k: None)
    async def unrelated(*args, **kwargs):
        pass
    monkeypatch.setattr(rig.runner, "_redeliver_failed_obligations_for_platform", unrelated)
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", unrelated)
    await rig.runner._install_reconnected_adapter(rig.adapter.platform, rig.adapter)
    await finish_tasks(rig.adapter)
    assert payloads == [("1", "request")] and not db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_initial_base_provider_failure_never_reopens_consumed_input(rig):
    async def consumed_then_failed(event):
        await rig.handle(event)  # actual ordinary-row transaction consumed this receipt
        raise RuntimeError("provider acceptance uncertain")
    rig.adapter.set_message_handler(consumed_then_failed)
    event = rig.adapter.event()
    await rig.adapter.handle_message(event)
    await finish_tasks(rig.adapter)
    key = rig.runner._session_key_for_source(event.source)
    db = rig.runner.session_store._db_for_key(key)
    session = rig.runner.session_store.get_or_create_session(event.source)
    assert db.gateway_intake_was_consumed(session.session_id, event._gateway_intake_receipts)
    assert rig.handled == ["1"] and not db.pending_gateway_intakes()
    rig.runner._gateway_intake_dispatch_owner = "restart-after-consumption"
    await rig.runner._drain_durable_intakes()
    assert rig.handled == ["1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "decline"])
async def test_real_base_busy_handoff_failure_preserves_distinct_text_media_receipts_for_fifo_recovery(rig, monkeypatch, failure):
    gate, started = asyncio.Event(), asyncio.Event()
    observed = []
    async def preparation(event):
        if event.message_id == "1":
            started.set()
            await gate.wait()
        observed.append((event.message_id, event.text, list(event.media_urls), list(event.media_types),
                         list(event.media_text_inlined), event._gateway_intake_receipts[0]["receipt_id"]))
    payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=preparation)
    async def busy_failed(event, key):
        if failure == "raise":
            raise RuntimeError("busy collaborator temporarily unavailable")
        return False
    rig.adapter.set_busy_session_handler(busy_failed)
    first = rig.adapter.event(text="first request")
    await rig.adapter.handle_message(first)
    await started.wait()
    physical = []
    for mid, text in [("2", "second photo"), ("3", "third photo")]:
        event = rig.adapter.event(mid, text=text, kind=MessageType.PHOTO)
        event.media_urls = ["https://stub.invalid/photo-" + mid]
        event.media_types = ["image/jpeg"]
        event.media_text_inlined = [False]
        await rig.adapter.handle_message(event)
        assert event._gateway_durable_adopted is True and event._gateway_accepted is False
        physical.append(event)
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(first.source))
    assert [row["snapshot"]["physical_message_id"] for row in db.pending_gateway_intakes(unclaimed_only=True)] == ["2", "3"]
    assert not rig.adapter._pending_messages and not payloads
    gate.set()
    await finish_tasks(rig.adapter)
    assert payloads == [("1", "first request"), ("2", "second photo"), ("3", "third photo")]
    assert observed[1:] == [(event.message_id, event.text, event.media_urls, event.media_types,
                            event.media_text_inlined, event._gateway_intake_receipts[0]["receipt_id"])
                           for event in physical]
    assert not db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_actual_dispatch_hook_bytes_and_physical_disposition_survive_recovery(rig, monkeypatch):
    calls = []
    async def rewrite(_name, **kwargs):
        calls.append(kwargs["event"].message_id)
        return [{"action": "rewrite", "text": "authorized exact rewritten request"}]
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", rewrite)
    event = rig.adapter.event()
    await rig.adapter._durable_intake_handler(event)
    assert event._gateway_durable_adopted is True and calls == ["1"]
    assert event._gateway_intake_snapshot["event"]["text"] == "authorized exact rewritten request"
    rig.runner._gateway_intake_dispatch_owner = "after-hook-crash"
    payloads = use_real_command_pipeline(rig, monkeypatch)
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert calls == ["1"] and payloads == [("1", "authorized exact rewritten request")]


@pytest.mark.asyncio
async def test_busy_explicit_queue_steer_and_skill_keep_physical_inputs_fifo(rig, monkeypatch):
    register_real_skill(rig, monkeypatch)
    gate, started = asyncio.Event(), asyncio.Event()
    async def hold(event):
        if event.message_id == "1":
            started.set()
            await gate.wait()
    payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=hold)
    await rig.adapter.handle_message(rig.adapter.event())
    await started.wait()
    queued = [rig.adapter.event("2", text="/queue second input"),
              rig.adapter.event("3", text="/steer third input"),
              rig.adapter.event("4", text="/intake-probe fourth input")]
    for event in queued:
        event.source.author_kind_verified = True
        await rig.adapter.handle_message(event)
        assert event._gateway_durable_adopted is True and not event._gateway_intake_control_completed
    assert any("queued" in text.lower() for text in rig.adapter.sent)
    assert not payloads
    gate.set()
    await finish_tasks(rig.adapter)
    assert [mid for mid, _text in payloads] == ["1", "2", "3", "4"]
    assert payloads[1:3] == [("2", "second input"), ("3", "third input")]
    assert "fourth input" in payloads[3][1]
    assert all(event.source.author_kind_verified for event in queued)


@pytest.mark.asyncio
async def test_startup_control_is_retryable_without_queued_destructive_replay_and_wakes_keep_normal_admission(rig):
    rig.runner._startup_restore_in_progress = True
    event = rig.adapter.event(text="/reset")
    with pytest.raises(GatewayIntakeError, match="startup restoration"):
        await rig.adapter.handle_message(event)
    assert not event._gateway_durable_adopted and not event._gateway_intake_control_completed
    assert not event._gateway_intake_refused and not event._gateway_accepted
    assert not getattr(rig.runner, "_startup_restore_queue", [])
    assert not rig.adapter._pending_messages
    rig.runner._startup_restore_in_progress = False
    handled = []
    async def wake_handler(event):
        assert (await rig.runner._hm_admit_event(event))[2] is True
        handled.append(event.text)
    rig.adapter.set_message_handler(wake_handler)
    wake = rig.adapter.event("2", text="trusted completion")
    wake.internal = True
    await rig.adapter.handle_message(wake)
    assert wake._gateway_accepted is True and not wake._gateway_durable_adopted
    await finish_tasks(rig.adapter)
    assert handled == ["trusted completion"]


async def recursive_fixture(rig, monkeypatch):
    event = rig.adapter.event("2", text="frozen queued request")
    await rig.adapter._durable_intake_handler(event)
    event._gateway_accepted = True
    key = rig.runner._session_key_for_source(event.source)
    route = rig.runner.session_store.get_or_create_session(event.source)
    db = rig.runner.session_store._db_for_key(key)
    rig.runner._session_db = None  # no resident-agent cache; real profile store remains authoritative
    generation = rig.runner._begin_session_run_generation(key)
    ctx = SimpleNamespace(source=event.source, session_id=route.session_id, session_key=key,
                          run_generation=generation, _interrupt_depth=0, history=[], context_prompt="",
                          channel_prompt=None, _status_thread_metadata=None,
                          stream_consumer_holder=[None], persist_user_display_metadata=None,
                          result_holder=[None])
    prepared, providers = [], []
    async def prepare(**kwargs):
        prepared.append(kwargs["event"].text)
        return kwargs["event"].text
    async def pins(*args, **kwargs):
        pass
    async def execute(**kwargs):
        assert asyncio.current_task()._gateway_intake_event is event
        db.append_message(route.session_id, "user", kwargs["message"],
                          display_metadata=kwargs["persist_user_display_metadata"])
        assert db.gateway_intake_was_consumed(route.session_id, event._gateway_intake_receipts)
        providers.append(kwargs["message"])
        return {"final_response": "queued answer", "messages": db.get_messages(route.session_id)}
    monkeypatch.setattr(rig.runner, "_prepare_profile_scoped_inbound_message_text", prepare)
    monkeypatch.setattr(rig.runner, "_pinned_channel_inputs", lambda _k, prompt, source, **kwargs: (prompt, source))
    monkeypatch.setattr(rig.runner, "_persist_prompt_pins", pins)
    monkeypatch.setattr(rig.runner, "_run_agent", execute)
    previous = {"interrupted": True, "messages": []}
    async def recurse():
        return await rig.runner._run_agent_queued_followup(ctx, rig.adapter, event.text, event,
                                                         "", previous, None)
    return SimpleNamespace(event=event, db=db, prepared=prepared, providers=providers,
                           previous=previous, recurse=recurse, route=route, context=ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["public-context", "transcript", "cancel"])
async def test_real_ordinary_preparation_exit_releases_claim_and_task_source_for_recovery(rig, monkeypatch, failure):
    from gateway.session_context import get_current_turn_source
    from gateway.session_transcript import TranscriptReadError

    use_real_command_pipeline(rig, monkeypatch)
    # Exercise the real ordinary handler and its real preparer, not the fixture's
    # provider shortcut. Only unrelated hooks/context rendering are deterministic.
    rig.runner._handle_message_with_agent = GatewayRunner._handle_message_with_agent.__get__(rig.runner)
    rig.runner._session_db = None
    rig.runner.hooks = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(rig.runner, "_pinned_session_context_prompt", lambda *a, **k: "stable context")
    monkeypatch.setattr(rig.runner, "_hmwa_first_contact_notes", AsyncMock())
    providers, attempts, cleared = [], [], []
    entered = asyncio.Event()

    async def unavailable(event, **kwargs):
        actor = get_current_turn_source()
        assert actor.author_kind_verified and (actor.user_id, actor.chat_id, actor.is_bot) == (
            event.source.user_id, event.source.chat_id, event.source.is_bot)
        attempts.append(failure)
        entered.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        raise RuntimeError("public lookup temporarily unavailable")

    async def unreadable(session_id):
        attempts.append(failure)
        raise TranscriptReadError(session_id)

    async def provider(**kwargs):
        providers.append(kwargs)
        pytest.fail("unprepared input reached the provider")

    monkeypatch.setattr(rig.adapter, "prepare_public_context", unavailable, raising=False)
    if failure == "transcript":
        monkeypatch.setattr(rig.runner.async_session_store, "load_transcript", unreadable)
    monkeypatch.setattr(rig.runner, "_run_agent", provider)
    ordinary = rig.runner._primary_message_handler()
    async def checked_handler(event):
        try:
            return await ordinary(event)
        finally:
            cleared.append(get_current_turn_source())
    rig.adapter.set_message_handler(checked_handler)
    event = rig.adapter.event(text="ordinary exact request")
    await rig.adapter.handle_message(event)
    assert event._gateway_durable_adopted and event._gateway_accepted
    if failure == "cancel":
        await asyncio.wait_for(entered.wait(), 3)
        await rig.adapter.cancel_background_tasks()  # shutdown, not an explicit /stop
    else:
        await finish_tasks(rig.adapter)
    key = rig.runner._session_key_for_source(event.source)
    db = rig.runner.session_store._db_for_key(key)
    assert attempts == [failure] and not providers and cleared == [None]
    assert event._gateway_accepted is False and not event._gateway_intake_refused
    assert len(db.pending_gateway_intakes(unclaimed_only=True)) == 1
    # A later actual Base/Runner/user-row/provider lifecycle can recover it once.
    payloads = use_real_command_pipeline(rig, monkeypatch)
    await rig.runner._drain_durable_intakes(unclaimed_only=True)
    await finish_tasks(rig.adapter)
    assert payloads == [("1", "ordinary exact request")] and not db.pending_gateway_intakes()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lookup", "cancel"])
async def test_real_recursive_late_public_exit_defers_promoted_fifo_tail_without_overtaking(rig, monkeypatch, failure):
    recursive = await recursive_fixture(rig, monkeypatch)
    key = recursive.route.session_key
    physical = [recursive.event]
    for mid in ("3", "4"):
        event = rig.adapter.event(mid, text="queued request " + mid)
        await rig.adapter._durable_intake_handler(event)
        physical.append(event)
    for event in physical:
        rig.runner._enqueue_fifo(key, event, rig.adapter)
    oldest, _text = await rig.runner._run_agent_drain_pending(recursive.previous, rig.adapter, recursive.event.source, key)
    assert oldest is recursive.event and rig.adapter._pending_messages[key] is physical[1]
    entered = asyncio.Event()
    async def unavailable(event, **kwargs):
        assert event is oldest
        assert asyncio.current_task()._gateway_intake_event is oldest
        entered.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        raise RuntimeError("late public lookup temporarily unavailable")
    monkeypatch.setattr(rig.adapter, "prepare_public_context", unavailable, raising=False)
    task = asyncio.create_task(recursive.recurse())
    await asyncio.wait_for(entered.wait(), 3)
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="late public lookup"):
            await task
    assert task._gateway_intake_event is None
    assert recursive.prepared == [oldest.text] and not recursive.providers
    assert not rig.adapter._pending_messages and not rig.runner._overflow_queue(key)
    assert [r["snapshot"]["physical_message_id"] for r in recursive.db.pending_gateway_intakes(unclaimed_only=True)] == ["2", "3", "4"]
    await rig.runner._drain_durable_intakes(unclaimed_only=True)
    await finish_tasks(rig.adapter)
    assert rig.handled == ["2", "3", "4"] and not recursive.db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_recovery_holds_newer_same_route_after_transient_oldest_but_other_route_progresses(rig, monkeypatch):
    physical = [rig.adapter.event("1", text="oldest"), rig.adapter.event("2", text="newer"),
                rig.adapter.event("3", text="unrelated", chat_id="99")]
    for event in physical:
        await rig.adapter._durable_intake_handler(event)
    rig.runner._gateway_intake_dispatch_owner = "new-recovery-incarnation"
    restore = rig.adapter.restore_durable_intake
    attempts = []
    def unavailable(snapshot):
        attempts.append(snapshot["physical_message_id"])
        if snapshot["physical_message_id"] == "1":
            raise RuntimeError("oldest source lookup temporarily unavailable")
        return restore(snapshot)
    monkeypatch.setattr(rig.adapter, "restore_durable_intake", unavailable)
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert "2" not in attempts and rig.handled == ["3"]
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(physical[0].source))
    assert [r["snapshot"]["physical_message_id"] for r in db.pending_gateway_intakes()] == ["1", "2"]
    monkeypatch.setattr(rig.adapter, "restore_durable_intake", restore)
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert rig.handled == ["3", "1", "2"] and not db.pending_gateway_intakes()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["adopt", "finish", "processing"])
async def test_concurrent_recovery_preserves_real_adoption_handoff_and_live_task_ownership(rig, monkeypatch, phase):
    held, release = asyncio.Event(), asyncio.Event()
    calls = []
    original_adopt, original_finish = rig.adapter._durable_intake_handler, rig.adapter._durable_intake_finish
    async def adopter(event):
        calls.append(event.message_id)
        admitted = await original_adopt(event)
        if phase == "adopt":
            held.set()
            await release.wait()
        return admitted
    async def finish(event, **kwargs):
        if phase == "finish" and not kwargs.get("turn_complete"):
            held.set()
            await release.wait()
        return await original_finish(event, **kwargs)
    async def processing(event):
        if phase == "processing":
            held.set()
        await release.wait()
        return await rig.handle(event)
    rig.adapter._durable_intake_handler, rig.adapter._durable_intake_finish = adopter, finish
    rig.adapter.set_message_handler(processing)
    event = rig.adapter.event()
    arrival = asyncio.create_task(rig.adapter.handle_message(event))
    try:
        await asyncio.wait_for(held.wait(), 3)
        key = rig.runner._session_key_for_source(event.source)
        db = rig.runner.session_store._db_for_key(key)
        assert db.gateway_intake_is_pending(event._gateway_intake_receipts)
        if phase == "adopt":
            assert key not in rig.adapter._active_sessions and key not in rig.adapter._session_tasks
        if phase == "processing":
            assert not rig.runner._session_state(key).persistent.intake_lock.locked()
            assert rig.adapter._session_tasks[key]._gateway_intake_event is event
        await asyncio.wait_for(rig.runner._drain_durable_intakes(unclaimed_only=True), 3)
        assert calls == ["1"] and not rig.handled
        assert db.gateway_intake_is_pending(event._gateway_intake_receipts)
    finally:
        release.set()
    await arrival
    await finish_tasks(rig.adapter)
    assert rig.handled == ["1"] and not db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_failed_terminal_release_recovers_ownerless_self_nonce_on_real_reconnect(rig, monkeypatch):
    event = rig.adapter.event(text="unconsumed exact request")
    key = rig.runner._session_key_for_source(event.source)
    db = rig.runner.session_store._db_for_key(key)
    release_claim = db.release_gateway_intake_dispatch
    unavailable, releases = True, []
    def fail_release(receipts, owner):
        releases.append(tuple(receipt["receipt_id"] for receipt in receipts))
        if unavailable:
            raise RuntimeError("claim-release storage temporarily unavailable")
        return release_claim(receipts, owner)
    monkeypatch.setattr(db, "release_gateway_intake_dispatch", fail_release)
    started, fail_preparation = asyncio.Event(), asyncio.Event()
    async def failed_preparation(event):
        if event.message_id == "1":
            started.set()
            await fail_preparation.wait()
            raise RuntimeError("preparation failed before user-row consumption")
    initial_payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=failed_preparation)
    await rig.adapter.handle_message(event)
    await asyncio.wait_for(started.wait(), 3)
    finished_task = rig.adapter._session_tasks[key]
    newer = rig.adapter.event("2", text="newer same-route request")
    await rig.adapter.handle_message(newer)
    assert newer._gateway_accepted and newer._gateway_durable_adopted
    fail_preparation.set()
    await finish_tasks(rig.adapter)
    assert len(releases) == 1 and not rig.handled and not initial_payloads
    assert [row["snapshot"]["physical_message_id"] for row in db.pending_gateway_intakes()] == ["1", "2"]
    assert db.pending_gateway_intakes()[0]["dispatch_owner"] == rig.runner._intake_dispatch_owner()
    assert not db.pending_gateway_intakes(unclaimed_only=True)
    assert finished_task.done() and not rig.adapter._active_sessions
    # A retained completed Task is not execution ownership. The real reconnect
    # wiring must recover this exact pending receipt without a process restart.
    rig.adapter._session_tasks[key] = finished_task
    unavailable = False
    payloads = use_real_command_pipeline(rig, monkeypatch)
    rig.runner.delivery_router = SimpleNamespace(adapters=rig.runner.adapters)
    rig.runner._failed_platforms = {rig.adapter.platform: {}}
    for name in ("_sync_voice_mode_state_to_adapter", "_bind_voice_input_callback",
                 "_update_platform_runtime_status", "_schedule_planned_restart_replay",
                 "_schedule_resume_pending_sessions"):
        monkeypatch.setattr(rig.runner, name, lambda *a, **k: None)
    async def unrelated(*args, **kwargs):
        pass
    monkeypatch.setattr(rig.runner, "_redeliver_failed_obligations_for_platform", unrelated)
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", unrelated)
    await rig.runner._install_reconnected_adapter(rig.adapter.platform, rig.adapter)
    await finish_tasks(rig.adapter)
    assert payloads == [("1", "unconsumed exact request"), ("2", "newer same-route request")]
    assert len(releases) == 3
    assert not db.pending_gateway_intakes()


def test_stop_receipt_cut_never_reopens_consumed_input_or_refuses_later_receipt(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("live", source="telegram", session_key="bot:42")
    consumed = adopt(db, tmp_path, snapshot("1"))
    queued = adopt(db, tmp_path, snapshot("2"))
    db.append_message("live", "user", "already accepted", display_metadata={"gateway_intake": [consumed]})
    cut = tuple(row["receipt_id"] for row in db.pending_gateway_intakes())
    later = adopt(db, tmp_path, snapshot("3"))
    assert db.refuse_gateway_intakes(cut) == {queued["receipt_id"]}
    assert db.gateway_intake_was_consumed("live", [consumed])
    assert not db.gateway_intake_is_pending([queued]) and db.gateway_intake_is_pending([later])
    db.close()


@pytest.mark.asyncio
async def test_real_stop_cut_refuses_queued_and_journal_overflow_preserves_wake_and_later_input_across_reopen(rig, monkeypatch):
    started = asyncio.Event()
    async def hold_first(event):
        if event.message_id == "1":
            started.set()
            await asyncio.Event().wait()
    payloads = use_real_command_pipeline(rig, monkeypatch, before_provider=hold_first)
    rig.runner._session_db = None
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", lambda **kwargs: 0)
    ordinary, wakes = rig.adapter._message_handler, []
    async def with_wakes(event):
        if event.internal:
            assert (await rig.runner._hm_admit_event(event))[2] is True
            wakes.append(event.text)
            key = rig.runner._session_key_for_source(event.source)
            if key not in rig.adapter._pending_messages:
                successor = rig.runner._promote_queued_event(key, rig.adapter, None)
                if successor is not None:
                    rig.adapter._pending_messages[key] = successor
            return None
        return await ordinary(event)
    rig.adapter.set_message_handler(with_wakes)
    first = rig.adapter.event("1", text="active before provider")
    await rig.adapter.handle_message(first)
    await asyncio.wait_for(started.wait(), 3)
    key = rig.runner._session_key_for_source(first.source)
    db = rig.runner.session_store._db_for_key(key)
    rig.runner._BUSY_QUEUE_MAX_PENDING = 1
    second, third = rig.adapter.event("2", text="queued B"), rig.adapter.event("3", text="journal-only C")
    await rig.adapter.handle_message(second)
    await rig.adapter.handle_message(third)
    assert second._gateway_accepted and not third._gateway_accepted
    wake = rig.adapter.event("wake", text="trusted completion")
    wake.internal = True
    assert (await rig.runner._hm_admit_event(wake))[2] is True
    rig.runner._enqueue_fifo(key, wake, rig.adapter)
    cut_ready, release_cut, cut_ids = threading.Event(), threading.Event(), []
    refuse = db.refuse_gateway_intakes
    def held_refusal(receipt_ids):
        cut_ids.extend(receipt_ids)
        cut_ready.set()
        assert release_cut.wait(timeout=5), "stop cut was not released"
        return refuse(receipt_ids)
    monkeypatch.setattr(db, "refuse_gateway_intakes", held_refusal)
    stop = rig.adapter.event("stop", text="/stop")
    stop_task = asyncio.create_task(rig.adapter.handle_message(stop))
    try:
        assert await asyncio.to_thread(cut_ready.wait, 3)
        assert set(cut_ids) == {event._gateway_intake_receipts[0]["receipt_id"] for event in (first, second, third)}
        rig.runner._BUSY_QUEUE_MAX_PENDING = 8
        later = rig.adapter.event("4", text="new input after stop cut")
        await rig.adapter.handle_message(later)
        assert later._gateway_durable_adopted and later._gateway_intake_receipts[0]["receipt_id"] not in cut_ids
    finally:
        release_cut.set()
    await stop_task
    await finish_tasks(rig.adapter)
    assert stop._gateway_intake_control_completed and not stop._gateway_durable_adopted
    assert payloads == [("4", "new input after stop cut")] and wakes == ["trusted completion"]
    assert not db.pending_gateway_intakes()
    assert first._gateway_intake_refused and second._gateway_intake_refused
    assert db.gateway_intake_generation("telegram", key) == 0  # /stop is not a conversation reset
    rig.runner.session_store.close_all_db_handles()
    rig.runner.session_store = SessionStore(rig.runner.config.sessions_dir, rig.runner.config)
    rig.runner._gateway_intake_dispatch_owner = "after-stop-reopen"
    wire(rig.runner, rig.adapter)
    rig.adapter.set_message_handler(with_wakes)
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert payloads == [("4", "new input after stop cut")]
    fresh = rig.adapter.event("5", text="fresh after reopen")
    await rig.adapter.handle_message(fresh)
    await finish_tasks(rig.adapter)
    assert payloads[-1] == ("5", "fresh after reopen") and [mid for mid, _text in payloads] == ["4", "5"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["acl", "edit", "reset", "recipient"])
async def test_actual_recursive_followup_refuses_current_source_before_preparation_or_provider(rig, monkeypatch, change):
    recursive = await recursive_fixture(rig, monkeypatch)
    if change == "acl":
        rig.adapter.config.extra["group_allow_from"] = "different-user"
    elif change == "edit":
        rig.adapter.visible["2"]["rev"] = "2"
    elif change == "reset":
        recursive.db.reset_public_context_route("telegram", recursive.route.session_key,
                                               {"chat_id": "42", "message_id": "9"})
    else:
        rig.adapter.visible["2"]["recipient"] = "other"
    assert await recursive.recurse() is recursive.previous
    assert not recursive.prepared and not recursive.providers
    assert not recursive.db.gateway_intake_is_pending(recursive.event._gateway_intake_receipts)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["current-proof", "preparation"])
async def test_actual_recursive_transient_failure_releases_pending_claim_for_current_source_recovery(rig, monkeypatch, stage):
    recursive = await recursive_fixture(rig, monkeypatch)
    original_restore = rig.adapter.restore_durable_intake
    if stage == "current-proof":
        def unknown(_snapshot):
            raise RuntimeError("current sender proof temporarily unavailable")
        monkeypatch.setattr(rig.adapter, "restore_durable_intake", unknown)
    else:
        async def unavailable(**kwargs):
            raise RuntimeError("media preparation temporarily unavailable")
        monkeypatch.setattr(rig.runner, "_prepare_profile_scoped_inbound_message_text", unavailable)
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        await recursive.recurse()
    assert not recursive.providers and recursive.event._gateway_accepted is False
    assert len(recursive.db.pending_gateway_intakes(unclaimed_only=True)) == 1
    monkeypatch.setattr(rig.adapter, "restore_durable_intake", original_restore)
    await rig.runner._drain_durable_intakes(unclaimed_only=True)
    await finish_tasks(rig.adapter)
    assert rig.handled == ["2"] and not recursive.db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_actual_recursive_followup_preserves_frozen_bytes_and_receipts(rig, monkeypatch):
    recursive = await recursive_fixture(rig, monkeypatch)
    result = await recursive.recurse()
    assert recursive.prepared == recursive.providers == ["frozen queued request"]
    assert result["queued_terminal_inbound_id"] == "2"
    assert not recursive.db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_actual_recursive_depth_cap_restores_oldest_before_two_newer_inputs_without_media_or_receipt_merge(rig, monkeypatch):
    recursive = await recursive_fixture(rig, monkeypatch)
    recursive.context._interrupt_depth = rig.runner._MAX_INTERRUPT_DEPTH
    recursive.context.result_holder[0] = recursive.previous
    key = recursive.route.session_key
    physical = [recursive.event]
    for mid in ("3", "4"):
        event = rig.adapter.event(mid, text="photo " + mid, kind=MessageType.PHOTO)
        event.media_urls, event.media_types, event.media_text_inlined = ["https://stub.invalid/photo-" + mid], ["image/jpeg"], [False]
        await rig.adapter._durable_intake_handler(event)
        physical.append(event)
    for event in physical:
        rig.runner._enqueue_fifo(key, event, rig.adapter)
    oldest, _text = await rig.runner._run_agent_drain_pending(recursive.previous, rig.adapter, recursive.event.source, key)
    assert oldest is recursive.event and rig.adapter._pending_messages[key] is physical[1]
    assert await recursive.recurse() is recursive.previous
    assert not recursive.prepared and not recursive.providers
    assert rig.adapter._pending_messages[key] is oldest
    assert rig.runner._overflow_queue(key) == physical[1:]
    consumed = []
    for expected in physical:
        event, text = await rig.runner._run_agent_drain_pending(recursive.previous, rig.adapter, expected.source, key)
        assert event is expected and text == expected.text
        assert rig.adapter.serialize_durable_intake(event) == event._gateway_intake_snapshot
        assert await rig.runner._hm_admit_event(event) is not None
        recursive.db.append_message(recursive.route.session_id, "user", text, display_metadata=intake_metadata(event))
        consumed.append(event._gateway_intake_receipts[0]["receipt_id"])
    assert len(set(consumed)) == 3 and not recursive.db.pending_gateway_intakes()
    assert not rig.adapter._pending_messages and not rig.runner._overflow_queue(key)


@pytest.mark.asyncio
async def test_actual_recursive_depth_cap_missing_receiver_releases_pending_claim_for_recovery(rig, monkeypatch):
    recursive = await recursive_fixture(rig, monkeypatch)
    recursive.context._interrupt_depth = rig.runner._MAX_INTERRUPT_DEPTH
    recursive.context.result_holder[0] = recursive.previous
    rig.runner.adapters.pop(rig.adapter.platform)
    assert rig.runner._intake_adapter_for(recursive.event.source) is None
    assert await recursive.recurse() is recursive.previous
    assert not recursive.prepared and not recursive.providers
    assert recursive.event._gateway_accepted is False
    assert len(recursive.db.pending_gateway_intakes(unclaimed_only=True)) == 1
    assert not rig.adapter._pending_messages
    rig.runner.adapters[rig.adapter.platform] = rig.adapter
    await rig.runner._drain_durable_intakes(unclaimed_only=True)
    await finish_tasks(rig.adapter)
    assert rig.handled == ["2"] and not recursive.db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_actual_recursive_preparation_policy_refusal_is_terminal_without_provider_or_replay(rig, monkeypatch):
    recursive = await recursive_fixture(rig, monkeypatch)
    async def refused(**kwargs):
        recursive.prepared.append(kwargs["event"].text)
        return None  # current preparation contract: user-notified context-reference policy refusal
    monkeypatch.setattr(rig.runner, "_prepare_profile_scoped_inbound_message_text", refused)
    assert await recursive.recurse() is recursive.previous
    assert recursive.prepared == ["frozen queued request"] and not recursive.providers
    assert recursive.event._gateway_intake_refused is True
    assert not recursive.db.pending_gateway_intakes()
    rig.runner._gateway_intake_dispatch_owner = "after-policy-refusal"
    await rig.runner._drain_durable_intakes()
    assert not rig.handled


@pytest.mark.asyncio
async def test_real_base_adoption_precedes_handoff_and_refuses_failed_durable_write(rig):
    event = rig.adapter.event()
    key = rig.runner._session_key_for_source(event.source)
    db = rig.runner.session_store._db_for_key(key)
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER reject_intake BEFORE INSERT ON gateway_intake BEGIN SELECT RAISE(ABORT, 'disk failed'); END"))
    with pytest.raises(sqlite3.IntegrityError, match="disk failed"):
        await rig.adapter.handle_message(event)
    assert event._gateway_durable_adopted is False and event._gateway_accepted is False
    assert not rig.handled and not db.pending_gateway_intakes()
    db._execute_write(lambda conn: conn.execute("DROP TRIGGER reject_intake"))
    await rig.adapter.handle_message(event)
    assert event._gateway_durable_adopted is True
    assert event._gateway_intake_receipts and db.gateway_intake_is_pending(event._gateway_intake_receipts)
    duplicate = rig.adapter.event()
    await rig.adapter.handle_message(duplicate)
    assert duplicate._gateway_durable_adopted is True and duplicate._gateway_accepted is False
    await finish_tasks(rig.adapter)
    assert rig.handled == ["1"] and not db.pending_gateway_intakes()


@pytest.mark.asyncio
async def test_opted_in_receiver_requires_actual_runner_wiring_before_any_handoff(rig):
    event = rig.adapter.event()
    rig.adapter._durable_intake_handler = None
    with pytest.raises(RuntimeError, match="host durable intake boundary"):
        await rig.adapter.handle_message(event)
    assert event._gateway_durable_adopted is False and event._gateway_intake_refused is False
    assert event._gateway_accepted is False and not rig.handled
    wire(rig.runner, rig.adapter)
    assert rig.adapter.durable_intake_version == 1 and callable(rig.adapter._durable_intake_drain)
    rig.adapter.set_message_handler(rig.handle)
    await rig.adapter.handle_message(event)
    await finish_tasks(rig.adapter)
    assert event._gateway_durable_adopted is True and rig.handled == ["1"]


@pytest.mark.asyncio
async def test_busy_physical_inputs_use_fifo_without_steering_merging_or_loss_at_cap(rig):
    gate, started = asyncio.Event(), asyncio.Event()
    async def block_first(event):
        if event.message_id == "1":
            started.set()
            await gate.wait()
        return await rig.handle(event)
    rig.adapter.set_message_handler(block_first)
    first = rig.adapter.event()
    await rig.adapter.handle_message(first)
    await started.wait()
    key = rig.runner._session_key_for_source(first.source)
    rig.runner._session_state(key).turn.agent = SimpleNamespace(
        steer=lambda text: pytest.fail("durable input was steered"),
        interrupt=lambda: pytest.fail("durable input interrupted admitted work"))
    rig.runner._BUSY_QUEUE_MAX_PENDING = 1
    second = rig.adapter.event("2", kind=MessageType.PHOTO, text="image request")
    third = rig.adapter.event("3", text="distinct follow-up")
    await rig.adapter.handle_message(second)
    await rig.adapter.handle_message(third)
    assert second._gateway_accepted is True and third._gateway_accepted is False
    assert third._gateway_durable_adopted is True
    db = rig.runner.session_store._db_for_key(key)
    assert [r["snapshot"]["physical_message_id"] for r in db.pending_gateway_intakes(unclaimed_only=True)] == ["3"]
    assert rig.adapter._pending_messages[key] is second and second.text == "image request"
    gate.set()
    await finish_tasks(rig.adapter)
    assert rig.handled == ["1", "2", "3"]
    assert db.pending_gateway_intakes() == []


@pytest.mark.asyncio
async def test_actual_restart_revalidates_receiver_visibility_revision_and_authorization(rig):
    old = rig.adapter.event()
    await rig.adapter._durable_intake_handler(old)  # crash before process-local handoff
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(old.source))
    assert db.pending_gateway_intakes()
    # Fresh host dispatch incarnation recovers the same physical queue through
    # the real current-source restorer and normal Base/user-row path.
    rig.runner.session_store.close_all_db_handles()
    rig.runner.session_store = SessionStore(rig.runner.config.sessions_dir, rig.runner.config)
    wire(rig.runner, rig.adapter)
    rig.adapter.set_message_handler(rig.handle)
    db = rig.runner.session_store._db_for_key(rig.runner._session_key_for_source(old.source))
    rig.runner._gateway_intake_dispatch_owner = "restart-incarnation"
    await rig.runner._drain_durable_intakes()
    await finish_tasks(rig.adapter)
    assert rig.handled == ["1"] and db.pending_gateway_intakes() == []
    for mid, refusal in [("2", "account"), ("3", "revision"), ("4", "auth"), ("5", "deleted")]:
        event = rig.adapter.event(mid)
        await rig.adapter._durable_intake_handler(event)
        if refusal == "account":
            rig.adapter.receiver = "different-account"
        elif refusal == "revision":
            rig.adapter.visible[mid]["rev"] = "2"
        elif refusal == "auth":
            rig.adapter.config.extra["group_allow_from"] = "other-user"
        else:
            rig.adapter.visible.pop(mid)
        rig.runner._gateway_intake_dispatch_owner = "restart-" + mid
        await rig.runner._drain_durable_intakes()
        await finish_tasks(rig.adapter)
        assert rig.handled == ["1"] and db.pending_gateway_intakes() == []
        rig.adapter.receiver = "worker"
        rig.adapter.config.extra["group_allow_from"] = "20"


@pytest.mark.asyncio
async def test_controls_complete_synchronously_without_durable_replay_and_wire_metadata_cannot_claim_receipt(rig):
    commands = []
    async def command_handler(event):
        assert event.get_command() == "reset"
        commands.append(event.message_id)
        return "reset complete"
    rig.adapter.set_message_handler(command_handler)
    event = rig.adapter.event(text="/reset")
    event.metadata["gateway_intake"] = [{"receipt_id": "forged"}]
    assert intake_metadata(event) == {} and intake_metadata(None) == {}
    unadopted = MessageEvent("legacy input")
    unadopted._gateway_intake_receipts = ({"receipt_id": "not-adopted"},)
    assert intake_metadata(unadopted) == {}
    await rig.adapter.handle_message(event)
    assert event._gateway_intake_control_completed is True and event._gateway_durable_adopted is False
    assert commands == ["1"] and rig.adapter.sent == ["reset complete"]
    assert not list(rig.adapter._background_tasks)
    rig.runner._gateway_intake_dispatch_owner = "restart"
    await rig.runner._drain_durable_intakes()
    assert commands == ["1"]


@pytest.mark.asyncio
async def test_current_edit_and_reset_refuse_already_queued_input_before_normal_user_row(rig):
    event = rig.adapter.event()
    await rig.adapter._durable_intake_handler(event)
    rig.adapter.visible["1"]["rev"] = "2"
    assert await rig.runner._hm_admit_event(event) is None
    assert not rig.handled
    fresh = rig.adapter.event("2")
    await rig.adapter._durable_intake_handler(fresh)
    key = rig.runner._session_key_for_source(fresh.source)
    db = rig.runner.session_store._db_for_key(key)
    db.reset_public_context_route("telegram", key, {"chat_id": "42", "message_id": "9"})
    assert await rig.runner._hm_admit_event(fresh) is None
    assert not db.pending_gateway_intakes()


def test_durable_pending_uses_existing_db_instead_of_plain_text_shutdown_spool():
    from gateway.shutdown_flush import _serialise_value
    event = MessageEvent("physical input")
    event._gateway_intake_receipts = ({"receipt_id": "host-owned"},)
    event._gateway_durable_adopted = True
    assert _serialise_value(event) is None
    assert _serialise_value(MessageEvent("legacy input")) == {"text": "legacy input"}


@pytest.mark.asyncio
async def test_concurrent_adoption_is_fifo_per_route_and_reset_control_fences_waiters(rig):
    gate, serializing, other_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = rig.adapter.serialize_durable_intake
    serialized = []
    async def delayed(event):
        serialized.append(event.message_id)
        if event.message_id in {"1", "4"}:
            serializing.set()
            await gate.wait()
        return original(event)
    rig.adapter.serialize_durable_intake = delayed
    async def handle(event):
        if event.get_command() == "reset":
            assert await rig.runner._hm_admit_event(event) is not None
            key = rig.runner._session_key_for_source(event.source)
            rig.runner.session_store._db_for_key(key).reset_public_context_route(
                "telegram", key, {"chat_id": "42", "message_id": "9"})
            return "reset complete"
        result = await rig.handle(event)
        if event.message_id == "3":
            other_done.set()
        return result
    rig.adapter.set_message_handler(handle)
    first, second = rig.adapter.event(), rig.adapter.event("2")
    first_task = asyncio.create_task(rig.adapter.handle_message(first))
    await serializing.wait()
    second_task = asyncio.create_task(rig.adapter.handle_message(second))
    await asyncio.sleep(0)
    await rig.adapter.handle_message(rig.adapter.event("3", chat_id="43"))
    await asyncio.wait_for(other_done.wait(), 2)
    assert "2" not in serialized and rig.handled == ["3"]
    gate.set()
    await asyncio.gather(first_task, second_task)
    await finish_tasks(rig.adapter)
    assert rig.handled == ["3", "1", "2"]

    gate.clear()
    serializing.clear()
    before_reset, waiter = rig.adapter.event("4"), rig.adapter.event("5")
    first_task = asyncio.create_task(rig.adapter.handle_message(before_reset))
    await serializing.wait()
    second_task = asyncio.create_task(rig.adapter.handle_message(waiter))
    await asyncio.sleep(0)
    reset = rig.adapter.event("9", text="/reset")
    await asyncio.wait_for(rig.adapter.handle_message(reset), 2)  # bypass the held intake lock
    assert reset._gateway_intake_control_completed is True
    gate.set()
    await asyncio.gather(first_task, second_task)
    assert before_reset._gateway_intake_refused is True and waiter._gateway_intake_refused is True
    assert before_reset._gateway_durable_adopted is False and waiter._gateway_durable_adopted is False
    assert rig.handled == ["3", "1", "2"]
    after = rig.adapter.event("10")
    await rig.adapter.handle_message(after)
    await finish_tasks(rig.adapter)
    assert after._gateway_intake_receipts[0]["generation"] == 1 and rig.handled[-1] == "10"


@pytest.mark.asyncio
async def test_actual_two_profile_same_physical_ids_keep_receiving_bot_and_store(rig):
    rig.runner.config.multiplex_profiles = True
    profile = rig.home / "profiles" / "bot_b"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    other = IntakeAdapter("worker_b")
    other.set_owner_profile("bot_b")
    rig.runner._profile_adapters = {"bot_b": {other.platform: other}}
    wire(rig.runner, other)
    other.set_message_handler(rig.handle)
    events = [(rig.adapter, rig.adapter.event()), (other, other.event()),
              (rig.adapter, rig.adapter.event("2"))]
    stores = []
    for adapter, event in events:
        await adapter.handle_message(event)
        await finish_tasks(adapter)
        assert event._gateway_durable_adopted is True
        receipt = event._gateway_intake_receipts[0]
        db = rig.runner.session_store._db_for_key(receipt["session_key"])
        stores.append(db)
        assert (db.db_path.parent == profile) == (adapter is other)
        assert receipt["session_key"].startswith("agent:bot_b:" if adapter is other else "agent:main:")
    assert stores[0] is stores[2] and stores[0] is not stores[1]
    assert events[0][1]._gateway_intake_receipts[0]["receipt_id"] != events[1][1]._gateway_intake_receipts[0]["receipt_id"]
    assert rig.handled == ["1", "1", "2"]


def test_current_turn_actor_proof_is_native_task_local_and_never_serialized(monkeypatch):
    from gateway import session_context as context
    from gateway.session_identity import replace_source
    monkeypatch.setenv("HERMES_CURRENT_TURN_SOURCE", "human")
    context.reset_session_vars()
    unknown = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="20")
    assert context.get_current_turn_source() is None
    tokens = context.set_session_vars(current_turn_source=unknown)
    assert context.get_current_turn_source() is None
    human = replace_source(unknown, author_kind_verified=True)
    tokens = context.set_session_vars(current_turn_source=human)
    human.user_id = "changed-after-binding"
    assert context.get_current_turn_source().user_id == "20"
    assert "HERMES_CURRENT_TURN_SOURCE" not in context._VAR_MAP
    assert "author_kind_verified" not in human.to_dict()
    assert SessionSource.from_dict({**human.to_dict(), "author_kind_verified": True}).author_kind_verified is False
    assert replace_source(context.get_current_turn_source(), user_id="bot").author_kind_verified is False
    assert replace_source(context.get_current_turn_source(), is_bot=True).author_kind_verified is False
    context.clear_session_vars(tokens)
    assert context.get_current_turn_source() is None
    bot = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="30",
                        is_bot=True, author_kind_verified=True)
    context.set_session_vars(current_turn_source=bot)
    assert context.get_current_turn_source().is_bot is True
    context.reset_session_vars()
    assert context.get_current_turn_source() is None


@pytest.mark.asyncio
async def test_actual_runner_actor_binding_is_concurrent_and_entry_reset_drops_inherited_actor(rig):
    from gateway import session_context as context
    both_ready = asyncio.Event()
    ready = []
    async def turn(uid, is_bot):
        context.reset_session_vars()
        assert context.get_current_turn_source() is None
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id=uid,
                               is_bot=is_bot, author_kind_verified=True)
        bound = SessionContext(source=source, connected_platforms=[source.platform], home_channels={}, session_key="bot:42")
        tokens = rig.runner._set_session_env(bound)
        try:
            ready.append(uid)
            if len(ready) == 2:
                both_ready.set()
            await both_ready.wait()
            current = context.get_current_turn_source()
            assert (current.user_id, current.is_bot) == (uid, is_bot)
            assert (await asyncio.to_thread(context.get_current_turn_source)).user_id == uid
            async def child_entry():
                context.reset_session_vars()
                assert context.get_current_turn_source() is None
            await asyncio.create_task(child_entry())
            assert context.get_current_turn_source().user_id == uid
        finally:
            rig.runner._clear_session_env(tokens)
        assert context.get_current_turn_source() is None
    await asyncio.gather(turn("20", False), turn("30", True))
    unknown = SessionContext(source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="20"),
                             connected_platforms=[], home_channels={}, session_key="bot:42")
    tokens = rig.runner._set_session_env(unknown)
    assert context.get_current_turn_source() is None
    rig.runner._clear_session_env(tokens)
