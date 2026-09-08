from threading import Event, Lock
from typing import Any


class PlaybackController:
    """Session-scoped generation barrier used by every playback producer."""

    def __init__(self):
        self._lock = Lock()
        self._playback_id = 0
        self._cancel_events: dict[int, Event] = {0: Event()}
        self._tasks: dict[int, list[Any]] = {}

    def begin(self, reason: str = "") -> int:
        del reason  # reserved for metrics/logging at the caller
        with self._lock:
            previous = self._playback_id
            self._cancel_events.setdefault(previous, Event()).set()
            for task in self._tasks.pop(previous, []):
                cancel = getattr(task, "cancel", None)
                if cancel:
                    cancel()
            self._playback_id += 1
            self._cancel_events[self._playback_id] = Event()
            for stale_id in list(self._cancel_events):
                if stale_id < self._playback_id - 4:
                    self._cancel_events.pop(stale_id, None)
            return self._playback_id

    def current_id(self) -> int:
        with self._lock:
            return self._playback_id

    def is_current(self, playback_id: int) -> bool:
        with self._lock:
            return playback_id == self._playback_id

    def cancelled(self, playback_id: int) -> bool:
        with self._lock:
            event = self._cancel_events.get(playback_id)
            return playback_id != self._playback_id or bool(event and event.is_set())

    def register_task(self, playback_id: int, task: Any) -> bool:
        with self._lock:
            if playback_id != self._playback_id:
                cancel = getattr(task, "cancel", None)
                if cancel:
                    cancel()
                return False
            self._tasks.setdefault(playback_id, []).append(task)
            return True
