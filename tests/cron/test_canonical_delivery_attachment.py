"""Confirmed transport and continuable attachment have separate run diagnostics."""

import asyncio
import threading
from dataclasses import replace

import pytest

from cron.scheduler import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource, SessionStore


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self, store):
        super().__init__(PlatformConfig(enabled=True, token="offline-fixture"), Platform.TELEGRAM)
        self.set_session_store(store)
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"type": "group", "name": "fixture"}

    async def resolve_delivery_source(self, chat_id, *, user_id=None, thread_id=None, **kwargs):
        return self.build_source(chat_id=chat_id, chat_type="group", user_id=user_id, thread_id=thread_id)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="80")


@pytest.fixture
def delivery_runtime(tmp_path, monkeypatch):
    config = GatewayConfig(group_sessions_per_user=False, platforms={
        Platform.TELEGRAM: PlatformConfig(enabled=True, token="offline-fixture"),
    })
    registration = PlatformEntry(name="telegram", label="Fixture", adapter_factory=RecordingAdapter,
                                 check_fn=lambda: True, delivery_source_resolver=RecordingAdapter.resolve_delivery_source)
    original_get = platform_registry.get
    monkeypatch.setattr(platform_registry, "get", lambda name: registration if str(getattr(name, "value", name)) == "telegram" else original_get(name))
    store = SessionStore(tmp_path / "sessions", config)
    adapter = RecordingAdapter(store)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="201", chat_type="group", user_id="77")
    session = store.get_or_create_session(source)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    assert ready.wait(5)
    job = {"id": "fixture-brief", "name": "Brief", "deliver": "origin", "attach_to_session": True,
           "origin": {"platform": "telegram", "chat_id": "201", "chat_type": "group", "user_id": "42"}}
    yield job, adapter, store, source, session, loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()
    store.close_all_db_handles()


@pytest.mark.parametrize("attach", [True, False])
def test_live_delivery_attaches_to_shared_reply_route_or_reports_failure(delivery_runtime, attach):
    job, adapter, store, source, session, loop = delivery_runtime
    if not attach:
        store.suspend_session(session.session_key)
    error = _deliver_result(job, "The fixture brief", adapters={Platform.TELEGRAM: adapter}, loop=loop)
    assert len(adapter.sent) == 1
    messages = store.load_transcript(session.session_id)
    if attach:
        assert error is None
        assert [m["role"] for m in messages] == ["user"]
        assert "The fixture brief" in messages[0]["content"]
        assert store.get_or_create_session(replace(source, user_id="42")).session_id == session.session_id
    else:
        assert "delivery attachment failed" in error and "message already sent" in error
        assert messages == []


def test_standalone_success_reports_unavailable_attachment_without_resending(delivery_runtime, monkeypatch):
    job, adapter, store, source, session, loop = delivery_runtime
    sends = []

    async def standalone(*args, **kwargs):
        sends.append((args, kwargs))
        return {"success": True, "message_id": "80"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone)
    error = _deliver_result(job, "Standalone brief", adapters=None, loop=None)
    assert len(sends) == 1 and adapter.sent == []
    assert "owning gateway runtime is unavailable" in error
    assert "message already sent" in error
    assert store.load_transcript(session.session_id) == []
