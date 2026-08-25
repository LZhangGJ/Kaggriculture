"""
base_workflow_coordinator.py

Base Workflow Coordinator for the Kaggriculture AI Agent.

Defines the common interface for workflow
coordinators.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseWorkflowCoordinator(ABC):
    """
    Base workflow coordinator.
    """

    @abstractmethod
    def clear_cache(
        self,
    ) -> None:
        """
        Clear cached workflow state.
        """

    @abstractmethod
    def prepare_features(
        self,
        *,
        state_id: str,
        game_state: dict,
        maximums: dict[str, float] | None = None,
        selected: list[str] | None = None,
    ) -> dict:
        """
        Prepare workflow features.
        """