"""Cron attachment uses the current reply route and respects lifecycle boundaries."""

from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_attachment_uses_current_policy_and_owner_after_restart(tmp_path, monkeypatch):
    homes = {p: tmp_path / "profiles" / p for p in ("scout", "chief")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda p: homes[p])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda p: p in homes)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    config = GatewayConfig(multiplex_profiles=True, group_sessions_per_user=True)
    store = SessionStore(tmp_path / "sessions", config)
    scout = SessionSource(platform=Platform.TELEGRAM, chat_id="201", chat_type="group",
                          user_id="42", profile="scout")
    chief = replace(scout, profile="chief")
    private = store.get_or_create_session(scout)
    sibling = store.get_or_create_session(chief)
    # Sharing changes the exact reply key. A sender-first origin scan still picks private.
    config.group_sessions_per_user = False
    shared = store.get_or_create_session(replace(scout, user_id="77"))
    db = store._db_for_key(shared.session_key)
    assert db.find_session_by_origin(platform="telegram", chat_id="201", thread_id="", user_id="42") == private.session_id
    activity = shared.updated_at
    store.close_all_db_handles()
    restored = SessionStore(tmp_path / "sessions", config)
    try:
        for owner in ("chief", "scout", "chief"):
            token = set_hermes_home_override(str(homes[owner]))
            try:
                assert restored.append_delivery_to_session(scout, "[Cron delivery: brief]\nshared result")
            finally:
                reset_hermes_home_override(token)
        assert restored.get_or_create_session(scout, touch_activity=False).session_id == shared.session_id
        assert restored.lookup_by_session_key(shared.session_key).updated_at == activity
        owner_db = restored._db_for_key(shared.session_key)
        assert [m["role"] for m in owner_db.get_messages_as_conversation(shared.session_id)] == ["user"] * 3
        assert owner_db.get_messages_as_conversation(private.session_id) == []
        assert restored._db_for_key(sibling.session_key).get_messages_as_conversation(sibling.session_id) == []
        assert not restored.append_delivery_to_session(replace(scout, chat_id="missing"), "cold")
    finally:
        restored.close_all_db_handles()


@pytest.mark.parametrize("transition", ["compression", "reset", "busy", "suspended"])
def test_attachment_race_follows_only_compression(tmp_path, monkeypatch, transition):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="20", chat_type="dm", user_id="42")
    entry = store.get_or_create_session(source)
    old_id, key, activity = entry.session_id, entry.session_key, entry.updated_at
    db = store._db_for_key(key)
    append = db.append_message
    raced = False

    def append_after_transition(*args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            if transition == "compression":
                db.end_session(old_id, "compression")
                db.create_session("compression-child", "telegram", session_key=key, chat_id="20",
                                  chat_type="dm", user_id="42", parent_session_id=old_id)
            elif transition == "reset":
                store.reset_session(key)
            elif transition == "busy":
                assert db.try_acquire_session_turn_lease(old_id, "test-turn", ttl_seconds=60)
            else:
                store.suspend_session(key)
                store.get_or_create_session(source, touch_activity=False)
        return append(*args, **kwargs)

    monkeypatch.setattr(db, "append_message", append_after_transition)
    save_entries = store._save_entries
    def save_without_routing_lock():
        assert not store._lock.locked(), "persistence held the routing lock"
        return save_entries()
    monkeypatch.setattr(store, "_save_entries", save_without_routing_lock)
    try:
        if transition == "busy":
            from hermes_state_errors import SessionTurnLeaseLostError
            with pytest.raises(SessionTurnLeaseLostError):
                store.append_delivery_to_session(source, "brief")
        elif transition in {"reset", "suspended"}:
            with pytest.raises(ValueError, match="no longer active"):
                store.append_delivery_to_session(source, "brief")
            current = store.lookup_by_session_key(key)
            assert current.session_id != old_id
            assert db.get_messages_as_conversation(current.session_id) == []
        else:
            assert store.append_delivery_to_session(source, "brief")
            current = store.lookup_by_session_key(key)
            assert current.session_id == "compression-child"
            assert current.updated_at == activity
            assert [m["content"] for m in db.get_messages_as_conversation(current.session_id)] == ["brief"]
        assert db.get_messages_as_conversation(old_id) == []
    finally:
        db.release_session_turn_lease(old_id, "test-turn")
        store.close_all_db_handles()
