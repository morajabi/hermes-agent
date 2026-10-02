"""Real tool/store/profile handoff; only transport and inference are simulated."""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cron import jobs, scheduler
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from hermes_constants import get_hermes_home
from tools.cronjob_tools import cronjob
from tools.registry import registry


@pytest.fixture
def profiles(tmp_path, monkeypatch, make_cron_provider):
    root = tmp_path / "hermes"
    chief, scout = root / "profiles" / "chief", root / "profiles" / "scout"
    for home in (chief, scout):
        home.mkdir(parents=True)
    chief.joinpath("config.yaml").write_text("cron:\n  allowed_executors: [scout]\n", encoding="utf-8")
    scout.joinpath("config.yaml").write_text("model: scout-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(chief))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ready.wait(timeout=5)

    class Adapter:
        is_connected = True
        denied = False
        context = "public task v1; carried mention is historical"

        def __init__(self):
            self.requests = []
            self.denied_chats = set()

        async def fetch_public_task_context(self, source):
            assert asyncio.get_running_loop() is loop
            self.requests.append((get_hermes_home(), source))
            if self.denied or source.chat_id in self.denied_chats:
                raise PermissionError("Executor has no access to the task chat")
            return self.context

    adapter = Adapter()
    import gateway.run as gateway_run

    runner = SimpleNamespace(
        _gateway_loop=loop,
        _adapters_for_profile=lambda profile: {Platform.TELEGRAM: adapter} if profile == "scout" else {},
        _is_shared_bot_satellite=lambda profile: False,
    )
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    registered = []
    provider = make_cron_provider(register_job=lambda job: registered.append((get_hermes_home(), job)))
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: provider)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="public-work", thread_id="task-7",
                           chat_type="group", user_id="human-9", profile="chief", author_kind_verified=True)
    tokens = set_session_vars(
        platform="telegram", chat_id="public-work", thread_id="task-7", user_id="human-9",
        profile="chief", async_delivery=True, current_turn_source=source,
    )
    data = SimpleNamespace(
        chief=chief, scout=scout, adapter=adapter, loop=loop, runner=runner, registered=registered, source=source,
        selector={"platform": "telegram", "chat_id": "public-work", "thread_id": "task-7"},
    )
    try:
        yield data
    finally:
        clear_session_vars(tokens)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def create_worker(profiles, **overrides):
    return json.loads(registry.dispatch("cronjob_manage", {
        "action": "create", "prompt": "Review current public task and report progress",
        "schedule": "every 5m", "execute_as": "scout", "attach_to_session": True,
        "public_task_context": profiles.selector, **overrides,
    }))


def test_actual_tool_owns_worker_job_and_all_management_stays_scoped(profiles):
    # A -> B -> A is a real context switch; source values must survive but ownership must change.
    result = create_worker(profiles)
    assert result["success"] is True
    assert get_hermes_home() == profiles.chief
    with jobs.use_cron_store(profiles.chief):
        assert jobs.list_jobs(include_disabled=True) == []
    with jobs.use_cron_store(profiles.scout):
        worker = jobs.get_job(result["job_id"])
    assert profiles.registered == [(profiles.scout, worker)]
    assert worker["public_task_context"] == profiles.selector
    assert worker["attach_to_session"] is True
    assert worker["origin"]["profile"] == worker["origin"]["executor_profile"] == "scout"
    assert worker["origin"]["creator_profile"] == "chief"
    assert worker["origin"]["user_id"] == "human-9"
    assert all(home == profiles.scout and source.profile == "scout"
               for home, source in profiles.adapter.requests)
    assert json.loads(cronjob("list"))["count"] == 0
    assert json.loads(cronjob("list", execute_as="scout"))["count"] == 1
    for action, kwargs in (("get", {}), ("update", {"name": "Scout review"}),
                           ("pause", {}), ("resume", {}), ("delete", {})):
        response = json.loads(cronjob(action, job_id=worker["id"], execute_as="scout", **kwargs))
        assert response["success"] is True
        assert get_hermes_home() == profiles.chief
    with jobs.use_cron_store(profiles.scout):
        assert jobs.list_jobs(include_disabled=True) == []
    # Omission preserves the existing owner and adds no optional task-context key.
    local = json.loads(cronjob("create", prompt="Chief-only summary", schedule="every 5m", deliver="local"))
    assert local["success"] is True
    assert profiles.registered[-1][0] == profiles.chief
    with jobs.use_cron_store(profiles.chief):
        assert "public_task_context" not in jobs.get_job(local["job_id"])


@pytest.mark.parametrize("invalid", ["unconfigured", "path", "noncanonical", "unknown", "unserved", "no_acl", "forged_selector"])
def test_create_denies_untrusted_executor_or_unreadable_task_before_any_write(profiles, invalid):
    options = {}
    if invalid == "unconfigured":
        profiles.chief.joinpath("config.yaml").write_text("cron: {}\n", encoding="utf-8")
    elif invalid == "path":
        options["execute_as"] = "../scout"
    elif invalid == "noncanonical":
        options["execute_as"] = "Scout"
    elif invalid == "unknown":
        options["execute_as"] = "missing"
        profiles.chief.joinpath("config.yaml").write_text("cron:\n  allowed_executors: [missing]\n", encoding="utf-8")
    elif invalid == "unserved":
        profiles.runner._adapters_for_profile = lambda profile: {}
    elif invalid == "no_acl":
        profiles.adapter.denied = True
    else:
        options["public_task_context"] = {**profiles.selector, "profile": "chief"}
    result = create_worker(profiles, **options)
    assert result["success"] is False
    assert profiles.registered == []
    for home in (profiles.chief, profiles.scout):
        with jobs.use_cron_store(home):
            assert jobs.list_jobs(include_disabled=True) == []
    assert get_hermes_home() == profiles.chief


@pytest.mark.parametrize("author", ["bot", "unknown", "env_only"])
def test_cross_profile_create_and_manage_require_the_current_verified_human(profiles, monkeypatch, author):
    created = create_worker(profiles, deliver="local")
    assert created["success"] is True
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="public-work", thread_id="task-7",
                           chat_type="group", user_id="bot-11" if author == "bot" else "human-9", profile="chief",
                           is_bot=author == "bot", author_kind_verified=author == "bot")
    if author == "env_only":
        monkeypatch.setenv("HERMES_CURRENT_TURN_SOURCE", json.dumps({"is_bot": False, "author_kind_verified": True}))
    tokens = set_session_vars(platform="telegram", chat_id=source.chat_id, thread_id=source.thread_id,
                              user_id=source.user_id, profile="chief", current_turn_source=source,
                              async_delivery=True)
    before = list(profiles.registered)
    try:
        for action in ("create", "list", "get", "update", "pause", "resume", "delete", "run"):
            params = {"action": action, "execute_as": "scout", "job_id": created["job_id"]}
            if action == "create":
                params.update(schedule="every 5m", prompt="Unsolicited worker task", public_task_context=profiles.selector)
            elif action == "update":
                params["name"] = "Changed by bot"
            result = json.loads(registry.dispatch("cronjob_manage", params))
            assert result["success"] is False and "authenticated human" in result["error"]
            assert "jobs" not in result and "job" not in result
            assert get_hermes_home() == profiles.chief
        assert profiles.registered == before
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize("action", ["get", "update", "run", "pause", "resume", "remove", "delete", "list"])
def test_stored_public_task_revocation_refuses_every_management_action_without_new_selector(profiles, monkeypatch, action):
    created = create_worker(profiles, deliver="local")
    assert created["success"] is True
    with jobs.use_cron_store(profiles.scout):
        before = jobs.get_job(created["job_id"])
    profiles.adapter.denied = True
    ran = MagicMock()
    monkeypatch.setattr(scheduler, "run_one_job", ran)
    params = {"action": action, "execute_as": "scout", "job_id": created["job_id"]}
    if action == "update":
        params["name"] = "Task access was revoked"
    result = json.loads(registry.dispatch("cronjob_manage", params))
    assert result["success"] is False and "access" in result["error"]
    assert "jobs" not in result and "job" not in result
    assert get_hermes_home() == profiles.chief
    with jobs.use_cron_store(profiles.scout):
        assert jobs.get_job(created["job_id"]) == before
    ran.assert_not_called()


def test_replacing_selector_cannot_bypass_revoked_stored_task_or_selected_store(profiles):
    from gateway.run import _profile_runtime_scope

    created = create_worker(profiles, deliver="local")
    assert created["success"] is True
    with jobs.use_cron_store(profiles.scout):
        before = jobs.get_job(created["job_id"])
    profiles.adapter.denied_chats.add("public-work")
    replacement = {**profiles.selector, "chat_id": "other-public-task"}
    result = json.loads(cronjob("update", execute_as="scout", job_id=created["job_id"],
                               public_task_context=replacement))
    assert result["success"] is False and "access" in result["error"]
    assert any(source.chat_id == "other-public-task" for _, source in profiles.adapter.requests)
    with _profile_runtime_scope(profiles.scout), jobs.use_cron_store(profiles.scout):
        assert jobs.get_job(created["job_id"]) == before
        result = json.loads(cronjob("list"))
        assert result["success"] is False and "jobs" not in result
    assert get_hermes_home() == profiles.chief


def test_manual_run_executes_scout_and_restart_replays_one_completion_from_chief_session_store(profiles, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import _profile_runtime_scope
    from gateway.session import SessionStore
    from tools import async_delegation
    from tools.process_registry import process_registry

    async_delegation._reset_for_tests()
    completions = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completions)
    store = SessionStore(sessions_dir=profiles.chief / "sessions",
                         config=GatewayConfig(multiplex_profiles=True, sessions_dir=profiles.chief / "sessions"))
    session = store.get_or_create_session(profiles.source)
    db = store._db_for_key(session.session_key)
    assert db.db_path == profiles.chief / "state.db"
    db.append_message(session.session_id, "user", "Run the Scout public task and report here")
    created = create_worker(profiles, deliver="local")
    assert created["success"] is True
    tokens = set_session_vars(platform="telegram", chat_id=profiles.source.chat_id,
                              thread_id=profiles.source.thread_id, user_id=profiles.source.user_id,
                              profile="chief", session_key=session.session_key, session_id=session.session_id,
                              current_turn_source=profiles.source, async_delivery=True)
    executed = []

    def inference(job, **kwargs):
        executed.append((get_hermes_home(), job["id"], kwargs["runtime_data_prompt"]))
        return True, "Scoped Scout output", "Scoped Scout output", None

    monkeypatch.setattr(scheduler, "run_job", inference)
    try:
        result = json.loads(registry.dispatch("cronjob_manage", {
            "action": "run", "execute_as": "scout", "job_id": created["job_id"],
        }, session_id=session.session_id))
        assert result["success"] is True and result["job"]["execution_mode"] == "background"
        delegation_id = result["job"]["delegation_id"]
        event = completions.get(timeout=15)
        assert event["delegation_id"] == delegation_id and event["status"] == "completed"
        assert event["session_key"] == session.session_key
        assert event["parent_session_id"] == session.session_id
        assert event["user_id"] == profiles.source.user_id
        assert "Scoped Scout output" in event["summary"]
        assert completions.empty()
        assert len(executed) == 1 and executed[0][:2] == (profiles.scout, created["job_id"])
        assert executed[0][2].endswith(profiles.adapter.context)
        assert get_hermes_home() == profiles.chief
        chief_receipt = async_delegation.get_durable_delegation(delegation_id)
        assert chief_receipt["origin_session"] == session.session_key
        assert chief_receipt["state"] == "completed" and chief_receipt["delivery_state"] == "pending"
        with _profile_runtime_scope(profiles.scout), jobs.use_cron_store(profiles.scout):
            assert async_delegation.get_durable_delegation(delegation_id) is None
            assert jobs.get_job(created["job_id"])["last_status"] == "ok"
            outputs = [path for path in (jobs.get_cron_output_dir() / created["job_id"]).iterdir() if path.is_file()]
            assert len(outputs) == 1 and "Scoped Scout output" in outputs[0].read_text()
        assert not (profiles.chief / "cron" / "output" / created["job_id"]).exists()
        assert db.get_messages(session.session_id)[0]["content"] == "Run the Scout public task and report here"

        # Restart discards only in-memory delegation state. The actual Chief database is retained.
        async_delegation._reset_for_tests()
        replay = queue.Queue()
        assert async_delegation.restore_undelivered_completions(replay) == 1
        restored = replay.get_nowait()
        assert restored["restored"] is True and restored["session_key"] == session.session_key
        assert restored["parent_session_id"] == session.session_id and replay.empty()
        claim = async_delegation.claim_event_delivery(restored, "chief-gateway-after-restart")
        assert claim and async_delegation.claim_event_delivery(restored, "second-consumer") is None
        async_delegation.complete_event_delivery(restored, claim)
        assert async_delegation.get_durable_delegation(delegation_id)["delivery_state"] == "delivered"
        assert async_delegation.restore_undelivered_completions(queue.Queue()) == 0
    finally:
        async_delegation._reset_for_tests()
        clear_session_vars(tokens)
        store.close_all_db_handles()


def test_each_fire_refreshes_public_context_in_actual_agent_prompt_and_revocation_stops_run(profiles, monkeypatch):
    result = create_worker(profiles, deliver="local")
    assert result["success"] is True
    from gateway.run import _profile_runtime_scope
    import httpx

    prompts = []
    provider_messages = []
    private_marker = "chief-private-provider-transcript"
    profiles.chief.joinpath("sessions").mkdir(exist_ok=True)
    profiles.chief.joinpath("sessions", "secret.json").write_text(private_marker, encoding="utf-8")

    def provider_transport(transport, request):
        assert request.url.host == "example.invalid"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "scout-model", "context_length": 256000}]})
        assert request.method == "POST"
        body = json.loads(request.read())
        prompts.append((get_hermes_home(), json.dumps(body["messages"])))
        provider_messages.append(body["messages"])
        chunk = {"id": "public-task", "object": "chat.completion.chunk", "created": 1, "model": "scout-model"}
        chunks = [
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Public task checked"}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
        ]
        wire = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=wire.encode())

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", provider_transport)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0.0)
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda job: None)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda *a, **kw: {
        "api_key": "test-key", "provider": "openrouter", "base_url": "https://example.invalid/v1",
        "api_mode": "chat_completions",
    })
    with _profile_runtime_scope(profiles.scout), jobs.use_cron_store(profiles.scout):
        for version in (1, 2):
            profiles.adapter.context = f"current public task v{version}"
            job = jobs.get_job(result["job_id"])
            assert scheduler.run_one_job(job, adapters={Platform.TELEGRAM: profiles.adapter}, loop=profiles.loop)
            assert jobs.get_job(job["id"])["last_status"] == "ok"
            assert f"current public task v{version}" in prompts[-1][1]
        profiles.adapter.denied = True
        assert scheduler.run_one_job(jobs.get_job(result["job_id"]),
                                     adapters={Platform.TELEGRAM: profiles.adapter}, loop=profiles.loop)
        assert jobs.get_job(result["job_id"])["last_status"] == "error"
    assert len(prompts) == 2
    assert all(home == profiles.scout and private_marker not in prompt for home, prompt in prompts)
    for version, messages in enumerate(provider_messages, start=1):
        assert any(message["role"] == "user" and f"current public task v{version}" in json.dumps(message)
                   for message in messages)
        assert all("current public task v" not in json.dumps(message) for message in messages
                   if message["role"] == "system")
    assert "current public task v1" not in prompts[1][1]
    assert get_hermes_home() == profiles.chief


def test_managed_worker_receives_only_fire_time_projection_and_missing_handoff_fails_closed(profiles, monkeypatch):
    result = create_worker(profiles, deliver="local")
    from gateway.run import _profile_runtime_scope

    handoffs = []
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job, **kw: handoffs.append((job, kw)) or True)
    with _profile_runtime_scope(profiles.scout), jobs.use_cron_store(profiles.scout):
        worker = jobs.get_job(result["job_id"])
        profiles.adapter.context = "fresh projection immediately before managed handoff"
        assert scheduler.run_one_job(worker, adapters={Platform.TELEGRAM: profiles.adapter}, loop=profiles.loop)
        assert handoffs[0][1]["runtime_data_prompt"].endswith(profiles.adapter.context)
        assert "runtime_data_prompt" not in jobs.get_job(worker["id"])
        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", worker["execution_id"])
        ran = MagicMock()
        monkeypatch.setattr(scheduler, "run_job", ran)
        assert scheduler.run_one_job(worker)
        ran.assert_not_called()
        assert jobs.get_job(worker["id"])["last_status"] == "error"


def test_actual_managed_payload_carries_public_projection_without_changing_job_definition(profiles, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tests.cron.test_restart_safe_worker import _stub_external_worker_launch
    from tools.process_registry import GatewayChildDispatch

    monkeypatch.setattr("tools.process_registry.restart_safe_gateway_child_argv",
                        lambda command, **kw: GatewayChildDispatch("scoped", ["scope", "--", *command]))
    spawned, payloads, handoff, observed = _stub_external_worker_launch(scheduler, monkeypatch)
    # This serialization fixture has no OS worker to reap after its simulated handoff.
    monkeypatch.setattr("cron.scheduler_detached_worker.reap_terminal_worker_in_background", lambda process: None)
    created = create_worker(profiles, deliver="local")
    assert created["success"] is True
    profiles.adapter.context = "head-public\n" + "x" * 20000 + "\ntail-public"
    with _profile_runtime_scope(profiles.scout), jobs.use_cron_store(profiles.scout):
        job = jobs.get_job(created["job_id"])
        job["execution_id"] = "exec-1"
        projection = scheduler._fetch_public_task_context(job, adapters={Platform.TELEGRAM: profiles.adapter},
                                                         loop=profiles.loop)
        assert scheduler._launch_external_cron_worker(job, runtime_data_prompt=projection)
        assert "runtime_data_prompt" not in jobs.get_job(job["id"])
    assert payloads[0]["job"] == job
    assert payloads[0]["job"]["origin"] == job["origin"]
    assert job["origin"]["executor_profile"] == job["origin"]["profile"] == "scout"
    assert job["origin"]["creator_profile"] == "chief" and job["origin"]["user_id"] == "human-9"
    assert payloads[0]["runtime_data_prompt"] == projection
    assert len(projection) < 8400 and "head-public" in projection and "tail-public" in projection
    assert payloads[0]["profile_home"] == str(profiles.scout.resolve())
    assert "runtime_data_prompt" not in job
    assert get_hermes_home() == profiles.chief
