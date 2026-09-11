"""
audit_log_manager.py

Audit Log Manager for the Kaggriculture AI Agent.

Records important system events.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from datetime import datetime, UTC


class AuditLogManager:
    """
    Stores audit log entries.
    """

    def __init__(self):
        self._logs = []

    # ---------------------------------------------------------

    def log(
        self,
        event: str,
        user: str,
        details: str,
    ):
        """
        Record an audit event.
        """

        self._logs.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": event,
                "user": user,
                "details": details,
            }
        )

    # ---------------------------------------------------------

    def all_logs(self):
        """
        Return all audit logs.
        """

        return list(self._logs)

    # ---------------------------------------------------------

    def filter_by_event(
        self,
        event: str,
    ):
        """
        Return logs matching an event type.
        """

        return [
            log
            for log in self._logs
            if log["event"] == event
        ]

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove all logs.
        """

        self._logs.clear()

    # ---------------------------------------------------------

    def count(self) -> int:
        """
        Return number of log entries.
        """

        return len(self._logs)