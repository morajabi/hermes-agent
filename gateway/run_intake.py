"""Opt-in transport adoption in the existing profile store and message FIFO.

ACK means the authorized immutable input is durable. The ordinary user-row
transaction consumes its receipt before any provider request; a consumed input is
never restarted merely because delivery or provider acceptance was uncertain.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
import weakref
from pathlib import Path

from gateway.session_identity import identity_of
from hermes_state_intake import GatewayIntakeError

logger = logging.getLogger("gateway.run")


class DurableIntakeRefused(GatewayIntakeError):
    """Permanent current-source/receiver/authorization refusal; never replay it."""


def intake_metadata(event):
    """Host-only user-row metadata; legacy and synthetic inputs carry no receipts."""
    receipts = getattr(event, "_gateway_intake_receipts", ())
    if (getattr(event, "_gateway_durable_adopted", False) is not True
            or not isinstance(receipts, (tuple, list)) or not receipts):
        return {}
    return {"gateway_intake": [dict(receipt) for receipt in receipts]}


async def _await_value(value):
    return await value if inspect.isawaitable(value) else value


class GatewayIntakeMixin:
    def _intake_lock_for_event(self, event):
        # Control traffic must not wait behind adoption (especially /stop and
        # approvals). It cannot enter the durable LLM replay path.
        if self._intake_is_control(event):
            return None
        key = self._session_key_for_source(event.source)
        self._capture_intake_generation(event, key)
        return self._session_state(key).persistent.intake_lock

    def _capture_intake_generation(self, event, key):
        if event._gateway_intake_generation is None:
            adapter = self._intake_adapter_for(event.source)
            db = self._intake_db(key, self._intake_owner(event, adapter))
            event._gateway_intake_generation = db.gateway_intake_generation(event.source.platform.value, key)

    def _intake_dispatch_owner(self):
        owner = getattr(self, "_gateway_intake_dispatch_owner", None)
        if owner is None:
            owner = self._gateway_intake_dispatch_owner = uuid.uuid4().hex
        return owner

    def _intake_owner(self, event, adapter):
        source = getattr(event, "source", event)
        identity = identity_of(source)
        if identity is None or self._intake_adapter_for(source) is not adapter:
            raise DurableIntakeRefused("gateway input has no receiving profile identity")
        return {"transport_profile": identity.transport_profile, "runtime_profile": identity.runtime_profile,
                "authorization_home": str(identity.authorization_home.resolve()),
                "runtime_home": str(identity.runtime_home.resolve())}

    def _intake_db(self, session_key, owner):
        store = getattr(self, "session_store", None)
        db = store._db_for_key(session_key) if store is not None else None
        if (db is None or not callable(getattr(db, "adopt_gateway_intake", None))
                or Path(db.db_path).resolve() != Path(owner["runtime_home"]) / "state.db"):
            raise GatewayIntakeError("gateway intake requires its physical profile StateDB")
        return db

    def _intake_is_control(self, event):
        """Classify before adoption using the same pending-control owners as dispatch."""
        if event.internal:
            return True
        if not event.allow_gateway_control:
            return False
        if event.get_command() or event.prompt_response:
            return True
        key = self._session_key_for_source(event.source)
        state = self._peek_session_state(key)
        if state is not None and state.persistent.update_prompt_pending:
            return True
        from tools import clarify_gateway, slash_confirm
        from tools.approval import has_blocking_approval
        text = (event.text or "").strip()
        clarify = clarify_gateway.get_pending_for_session(key, include_choice_prompts=True)
        if clarify is not None:
            _response, reason = clarify_gateway._coerce_text_response_detailed(clarify, text)
            if reason != "prose":
                return True
        if slash_confirm.get_pending(key) and (
                self._SLASH_CONFIRM_CMD_CHOICES.get(event.get_command())
                or self._slash_confirm_text_choices().get(text.lstrip("!/").lower())):
            return True
        return bool(has_blocking_approval(key) and self._plaintext_approval_words().get(text.lower()))

    def _make_durable_intake_handler(self, adapter):
        async def handler(event):
            registered, profile = self._owning_profile(adapter, adapter.platform)
            if not registered:
                raise DurableIntakeRefused("unregistered receiving adapter")
            source = event.source
            transport = self._transport_owner(source)
            if transport is not None and transport[0] is not adapter:
                raise DurableIntakeRefused("input belongs to another receiving bot")
            source._transport_adapter_ref = weakref.ref(adapter)
            home = getattr(self.session_store, "_routing_home", None)
            self._canonicalize(source, transport_profile=profile, primary_home=home)
            identity = identity_of(source)
            if identity is None:
                raise DurableIntakeRefused("unresolved receiving profile")
            if identity.multiplexed:
                from gateway.run import _async_profile_runtime_scope
                async with _async_profile_runtime_scope(identity.runtime_home):
                    return await self._adopt_durable_intake(event, adapter)
            return await self._adopt_durable_intake(event, adapter)

        return handler if self._multiplex_on() else self._standalone_scoped(handler)

    async def _adopt_durable_intake(self, original, adapter):
        # Re-dispatch of an already-adopted process-local event (startup/FIFO) is
        # not a second physical input. Admission and current visibility run again.
        if getattr(original, "_gateway_durable_adopted", False) is True:
            return original
        if original.internal:
            return original  # existing trusted wake admission, never transport ACK
        if self._intake_is_control(original):
            if (getattr(self, "_startup_restore_in_progress", False)
                    and not getattr(original, "_hermes_startup_restore_replay", False)):
                raise GatewayIntakeError("gateway control awaits startup restoration")
            # A slash may resolve to LLM input after awaited command/skill work.
            # Capture its arrival fence now; a broken DB must not block /stop.
            if original.get_command() and original._gateway_intake_generation is None:
                try:
                    self._capture_intake_generation(original, self._session_key_for_source(original.source))
                except Exception:
                    original._gateway_intake_generation_unavailable = True
            original._gateway_intake_control = True
            return original
        self._capture_intake_generation(original, self._session_key_for_source(original.source))
        admitted = await self._hm_admit_event(original, intake=True)
        if admitted is None:
            raise DurableIntakeRefused("current sender admission refused gateway input")
        event, source, _internal = admitted
        if self._intake_is_control(event):
            event._gateway_intake_control = original._gateway_intake_control = True
            return event
        # A queued conversational "yes" must never later resolve a newly opened
        # approval. Only fresh control traffic can enter the control interceptors.
        event.allow_gateway_control = False
        snapshot = await _await_value(adapter.serialize_durable_intake(event))
        if snapshot is None:
            raise DurableIntakeRefused("adapter cannot freeze this conversational input")
        owner = self._intake_owner(event, adapter)
        key, platform = self._session_key_for_source(source), source.platform.value
        db = self._intake_db(key, owner)
        try:
            receipt = await asyncio.to_thread(db.adopt_gateway_intake, snapshot, source=platform,
                                             session_key=key, owner=owner,
                                             dispatch_owner=self._intake_dispatch_owner(),
                                             expected_generation=original._gateway_intake_generation)
        except GatewayIntakeError as exc:
            raise DurableIntakeRefused(str(exc)) from exc
        host_receipt = {name: receipt[name] for name in ("receipt_id", "source", "session_key", "generation")}
        for target in (event, original):
            target._gateway_intake_receipts = (host_receipt,)
            target._gateway_intake_snapshot = snapshot
            target._gateway_intake_owner = owner
            target._gateway_intake_prepared = True
            target._gateway_intake_generation = receipt["generation"]
            target._gateway_durable_adopted = True
            target._bot_loop_admitted = True
        return event if receipt["dispatch"] else None

    async def _adopt_resolved_llm_input(self, event):
        """Command/hook resolution may yield LLM input; adopt before queue or provider.

        Only resolved bytes enter this profile-private receipt. Restored inputs
        validate the original physical source and never rerun command side effects.
        """
        adapter = self._intake_adapter_for(event.source)
        if (event.internal or getattr(event, "_gateway_durable_adopted", False) is True
                or getattr(adapter, "durable_intake", False) is not True):
            return event
        if not callable(getattr(adapter, "_durable_intake_handler", None)):
            raise GatewayIntakeError("resolved LLM input requires the host intake boundary")
        if event._gateway_intake_generation_unavailable:
            raise GatewayIntakeError("resolved LLM input has no arrival generation fence")
        event.allow_gateway_control = False
        event._gateway_intake_control = False
        event._gateway_intake_prepared = True  # dispatch hooks already resolved this event
        lock = self._intake_lock_for_event(event)
        try:
            async with lock:
                adopted = await adapter._durable_intake_handler(event)
                if adopted is not None:
                    admitted = await self._hm_admit_event(adopted)
                    return admitted[0] if admitted is not None else None
        except DurableIntakeRefused:
            event._gateway_intake_refused = True
            return None
        return None

    async def _validate_durable_intake_event(self, event):
        """Re-prove current visibility and authority before a queued turn is used."""
        receipts = getattr(event, "_gateway_intake_receipts", ())
        if not receipts:
            return
        adapter = self._intake_adapter_for(event.source)
        if adapter is None or getattr(adapter, "durable_intake", False) is not True:
            raise DurableIntakeRefused("receiving adapter is unavailable")
        owner = self._intake_owner(event, adapter)
        if owner != event._gateway_intake_owner:
            raise DurableIntakeRefused("receiving physical profile changed")
        db = self._intake_db(receipts[0]["session_key"], owner)
        if not await asyncio.to_thread(db.gateway_intake_is_pending, receipts):
            raise DurableIntakeRefused("gateway input was consumed, refused, or reset")
        restored = await _await_value(adapter.restore_durable_intake(event._gateway_intake_snapshot))
        if restored is None:
            raise DurableIntakeRefused("current source admission refused gateway input")
        # Restore does not grant authority from old serialized authorization bits.
        restored.source._transport_adapter_ref = weakref.ref(adapter)
        self._canonicalize(restored.source, transport_profile=self._owning_profile(adapter, adapter.platform)[1],
                           primary_home=getattr(self.session_store, "_routing_home", None))
        if (not self._is_user_authorized_for_source(restored.source)
                or self._intake_owner(restored, adapter) != owner
                or self._session_key_for_source(restored.source) != receipts[0]["session_key"]
                or await _await_value(adapter.serialize_durable_intake(restored)) != event._gateway_intake_snapshot):
            raise DurableIntakeRefused("current gateway input differs from its adopted version")
        event.source.is_bot = restored.source.is_bot
        event.source.role_authorized = restored.source.role_authorized
        event.source.author_kind_verified = restored.source.author_kind_verified is True

    def _durable_intake_is_queued(self, event, session_key):
        """Only an actual existing queue owner can retain a completed handoff."""
        receipts = {item["receipt_id"] for item in getattr(event, "_gateway_intake_receipts", ())}
        adapter = self._intake_adapter_for(event.source)
        queued = [getattr(adapter, "_pending_messages", {}).get(session_key)]
        queued.extend(self._overflow_queue(session_key) or ())
        queued.extend(getattr(self, "_startup_restore_queue", ()) or ())
        return any(receipts.intersection(item["receipt_id"] for item in
                   getattr(candidate, "_gateway_intake_receipts", ())) for candidate in queued)

    def _durable_intake_has_live_owner(self, row, adapter):
        """Existing adoption lock, FIFO/startup, or live Task owns a self-nonce claim."""
        key = row["session_key"]
        state = self._peek_session_state(key)
        if state is not None and state.persistent.intake_lock.locked():
            return True  # commit-to-handoff and resolved-command admission are still in flight
        task = getattr(adapter, "_session_tasks", {}).get(key)
        candidates = [getattr(adapter, "_pending_messages", {}).get(key)]
        if task is not None and not task.done():
            candidates.append(getattr(task, "_gateway_intake_event", None))
        candidates.extend(self._overflow_queue(key) or ())
        candidates.extend(getattr(self, "_startup_restore_queue", ()) or ())
        return any(any(item["receipt_id"] == row["receipt_id"] for item in
                       getattr(candidate, "_gateway_intake_receipts", ())) for candidate in candidates)

    def _detach_durable_intake_tail(self, session_key, adapter):
        """Return adopted FIFO inputs to their journal, retaining legacy/internal work.

        A transient failure of the dequeued oldest input must not let the already
        promoted next input overtake it in Base's finalizer.
        """
        pending = getattr(adapter, "_pending_messages", {})
        head = pending.get(session_key)
        overflow = self._overflow_queue(session_key)
        detached = []
        if head is not None and getattr(head, "_gateway_intake_receipts", ()) and not head.internal:
            detached.append(pending.pop(session_key))
        if overflow:
            detached.extend(item for item in overflow if getattr(item, "_gateway_intake_receipts", ())
                            and not item.internal)
            overflow[:] = [item for item in overflow if not getattr(item, "_gateway_intake_receipts", ())
                           or item.internal]
            if session_key not in pending and overflow:
                pending[session_key] = overflow.pop(0)
        return detached

    async def _settle_durable_intake_turn(self, event, *, session_key=None, run_generation=None, stopped=False):
        """Every turn exit settles pending ownership, never reopens consumed input."""
        receipts = getattr(event, "_gateway_intake_receipts", ())
        if not receipts:
            return False
        key = receipts[0]["session_key"]
        db = self._intake_db(key, event._gateway_intake_owner)
        try:
            pending = await asyncio.to_thread(db.gateway_intake_is_pending, receipts)
        except BaseException:
            task = asyncio.current_task()
            if task is not None:
                task._gateway_intake_recovery_deferred = True
            raise
        if not pending:
            return False  # ordinary user row, reset, or proven refusal is terminal
        stopped = stopped or (session_key is not None and run_generation is not None
                              and not self._is_session_run_current(session_key, run_generation))
        if stopped:
            await self._refuse_durable_intake_event(event)
            return False
        if self._durable_intake_is_queued(event, key):
            return False  # intentional startup/FIFO/depth-cap handoff still owns it
        tail = self._detach_durable_intake_tail(key, self._intake_adapter_for(event.source))
        task = asyncio.current_task()
        if task is not None:
            task._gateway_intake_recovery_deferred = True
        for pending in [event, *tail]:
            pending._gateway_accepted = False
            await self._finish_durable_intake_handoff(pending)
        return True

    async def _finish_durable_intake_handoff(self, event, *, turn_complete=False):
        if turn_complete:
            return await self._settle_durable_intake_turn(
                event, stopped=bool(getattr(asyncio.current_task(), "_gateway_intake_stop_requested", False)))
        receipts = getattr(event, "_gateway_intake_receipts", ())
        if receipts and event._gateway_accepted is not True:
            db = self._intake_db(receipts[0]["session_key"], event._gateway_intake_owner)
            await asyncio.to_thread(db.release_gateway_intake_dispatch, receipts, self._intake_dispatch_owner())

    async def _refuse_durable_intake_event(self, event):
        """Proven terminal source/payload-policy refusal, never execution or replay."""
        receipts = getattr(event, "_gateway_intake_receipts", ())
        if not receipts:
            return
        db = self._intake_db(receipts[0]["session_key"], event._gateway_intake_owner)
        refused = False
        for receipt in receipts:
            refused = bool(await asyncio.to_thread(db.refuse_gateway_intake, receipt["receipt_id"])) or refused
        if refused:
            event._gateway_intake_refused = True

    async def _refuse_durable_intake_route(self, source, session_key):
        """Stop also cancels adopted overflow that never obtained a memory slot."""
        adapter = self._intake_adapter_for(source)
        if getattr(adapter, "durable_intake", False) is not True:
            return set()
        db = self._intake_db(session_key, self._intake_owner(source, adapter))
        # This existing journal read is the stop cut. Awaited refusal uses only
        # those immutable IDs, never broadens to an input arriving after the cut.
        rows = await asyncio.to_thread(db.pending_gateway_intakes, session_key=session_key)
        receipt_ids = tuple(row["receipt_id"] for row in rows if row["source"] == source.platform.value)
        return await asyncio.to_thread(db.refuse_gateway_intakes, receipt_ids)

    def _intake_databases(self):
        store = self.session_store
        homes = {Path(store._routing_home).resolve()}
        if self._multiplex_on():
            from gateway.run import _multiplex_profile_homes
            homes.update(Path(home).resolve() for _name, home in _multiplex_profile_homes(self.config))
        for home in homes:
            db = store._open_session_db_for_active_scope(db_path=home / "state.db")
            if db is not None:
                yield db

    async def _drain_durable_intakes(self, *, adapter=None, session_key=None, unclaimed_only=False):
        """Existing startup/post-turn lifecycle drains; no observer or extra queue."""
        if not any(getattr(item, "durable_intake", False) is True
                   for item in self._primary_adapters().values()) and not any(
                getattr(item, "durable_intake", False) is True
                for items in self._profile_adapters_map().values() for item in items.values()):
            return
        for db in self._intake_databases():
            rows = await asyncio.to_thread(db.pending_gateway_intakes, session_key=session_key)
            deferred_routes = set()
            for row in rows:
                route = (row["source"], row["session_key"])
                if route in deferred_routes:
                    continue
                owner, snapshot = row["owner"], row["snapshot"]
                from gateway.config import Platform
                receiver = self._adapters_for_profile(owner["transport_profile"]).get(Platform(row["source"]))
                if (receiver is None or (adapter is not None and receiver is not adapter)
                        or getattr(receiver, "durable_intake", False) is not True):
                    continue
                state = self._peek_session_state(row["session_key"])
                if state is not None and state.persistent.intake_lock.locked():
                    deferred_routes.add(route)
                    continue  # no wait on in-flight adoption; other routes still progress
                try:
                    if row["dispatch_owner"] == self._intake_dispatch_owner():
                        if self._durable_intake_has_live_owner(row, receiver):
                            continue
                        # A failed terminal release may leave our nonce behind.
                        # Only an ownerless pending claim is eligible for later
                        # lifecycle recovery; consumption is never reopened.
                        await asyncio.to_thread(db.release_gateway_intake_dispatch,
                            (row,), self._intake_dispatch_owner())
                    elif unclaimed_only and row["dispatch_owner"] is not None:
                        continue  # another incarnation is recovered only on startup/reconnect
                    event = await _await_value(receiver.restore_durable_intake(snapshot))
                    if event is None:
                        raise DurableIntakeRefused("source is no longer readable")
                    # Restore source data is public lineage, never transport authority.
                    event.source._transport_adapter_ref = weakref.ref(receiver)
                    self._canonicalize(event.source, transport_profile=self._owning_profile(receiver, receiver.platform)[1],
                                       primary_home=getattr(self.session_store, "_routing_home", None))
                    if (self._intake_owner(event, receiver) != owner
                            or self._session_key_for_source(event.source) != row["session_key"]
                            or self._intake_is_control(event)
                            or await _await_value(receiver.serialize_durable_intake(event)) != snapshot):
                        raise DurableIntakeRefused("restored input changed physical owner, route, or payload")
                    event._gateway_intake_prepared = True  # immutable hook output was already adopted
                    event._hermes_startup_restore_replay = True
                    await receiver.handle_message(event)
                    if event._gateway_intake_refused is True:
                        await asyncio.to_thread(db.refuse_gateway_intake, row["receipt_id"])
                except DurableIntakeRefused:
                    await asyncio.to_thread(db.refuse_gateway_intake, row["receipt_id"])
                    logger.info("Refused obsolete durable gateway input %s", row["receipt_id"])
                except Exception:
                    # Visibility/DB/transport uncertainty preserves the input, never an
                    # invented authorization or an automatic provider resend.
                    logger.warning("Durable gateway input recovery deferred (%s)", row["receipt_id"], exc_info=True)
                    deferred_routes.add(route)
