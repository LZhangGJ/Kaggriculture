"""
job_manager.py

Job Manager for the Kaggriculture AI Agent.

Manages asynchronous-style AI jobs.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class JobManager:
    """
    Tracks submitted jobs.
    """

    def __init__(self):
        self._jobs = {}

    # ---------------------------------------------------------

    def submit(self, name: str, job):
        """
        Register a job.
        """
        self._jobs[name] = {
            "callable": job,
            "status": "queued",
        }

    # ---------------------------------------------------------

    def run(self, name: str):
        """
        Execute a job.
        """
        if name not in self._jobs:
            raise KeyError(f"Unknown job: {name}")

        self._jobs[name]["status"] = "running"

        try:
            result = self._jobs[name]["callable"]()

            self._jobs[name]["status"] = "completed"

            return result

        except Exception:

            self._jobs[name]["status"] = "failed"

            raise

    # ---------------------------------------------------------

    def status(self, name: str):
        """
        Return job status.
        """
        return self._jobs[name]["status"]

    # ---------------------------------------------------------

    def jobs(self):
        """
        Return registered job names.
        """
        return sorted(self._jobs.keys())

    # ---------------------------------------------------------

    def remove(self, name: str):
        """
        Remove a job.
        """
        self._jobs.pop(name, None)

    # ---------------------------------------------------------

    def clear(self):
        """
        Remove every job.
        """
        self._jobs.clear()