"""
checkpoint_manager.py

Model checkpoint manager for the Kaggriculture AI Agent.

Saves and loads learned Q-values.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    """
    Save and load reinforcement learning checkpoints.
    """

    def __init__(self, filename: str = "checkpoint.json"):
        self.path = Path(filename)

    # ---------------------------------------------------------

    def save(self, q_table: dict):
        """
        Save Q-table to disk.
        """

        with self.path.open("w", encoding="utf-8") as f:
            json.dump(q_table, f, indent=2, sort_keys=True)

    # ---------------------------------------------------------

    def load(self) -> dict:
        """
        Load Q-table.
        """

        if not self.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------

    def exists(self) -> bool:
        """
        Return True if checkpoint exists.
        """

        return self.path.exists()

    # ---------------------------------------------------------

    def delete(self):
        """
        Delete checkpoint.
        """

        if self.exists():
            self.path.unlink()

    # ---------------------------------------------------------

    def size(self) -> int:
        """
        Return checkpoint size in bytes.
        """

        if not self.exists():
            return 0

        return self.path.stat().st_size