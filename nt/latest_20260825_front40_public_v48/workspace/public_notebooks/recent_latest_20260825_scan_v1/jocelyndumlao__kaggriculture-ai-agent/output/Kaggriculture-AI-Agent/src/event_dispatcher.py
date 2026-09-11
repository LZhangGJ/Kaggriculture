"""
event_dispatcher.py

Queued Event Dispatcher for the Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from collections import deque


class EventDispatcher:
    """
    Dispatches queued events in FIFO order.
    """

    def __init__(self):
        self._queue = deque()

    # ---------------------------------------------------------

    def dispatch(
        self,
        event: str,
        payload=None,
    ):
        """
        Queue an event.
        """
        self._queue.append(
            (
                event,
                payload,
            )
        )

    # ---------------------------------------------------------

    def next_event(self):
        """
        Remove and return the next event.
        """
        if not self._queue:
            return None

        return self._queue.popleft()

    # ---------------------------------------------------------

    def pending(self) -> int:
        """
        Return number of queued events.
        """
        return len(self._queue)

    # ---------------------------------------------------------

    def empty(self) -> bool:
        """
        Return True if no events remain.
        """
        return len(self._queue) == 0

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all queued events.
        """
        self._queue.clear()