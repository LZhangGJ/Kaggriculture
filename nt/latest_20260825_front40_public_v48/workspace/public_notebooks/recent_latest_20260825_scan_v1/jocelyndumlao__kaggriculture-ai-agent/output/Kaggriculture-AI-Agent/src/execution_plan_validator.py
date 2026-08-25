"""
execution_plan_validator.py

Execution Plan Validator for the Kaggriculture AI Agent.

Validates execution plans before they are
executed.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExecutionPlanValidator:
    """
    Validate execution plans.
    """

    # ---------------------------------------------------------

    def validate(
        self,
        *,
        plan: list[dict],
    ) -> bool:
        """
        Return True if every step contains
        the required fields.
        """

        required = {
            "worker",
            "task",
        }

        for step in plan:

            if not required.issubset(step):
                return False

        return True

    # ---------------------------------------------------------

    def invalid_steps(
        self,
        *,
        plan: list[dict],
    ) -> list[dict]:
        """
        Return invalid steps.
        """

        required = {
            "worker",
            "task",
        }

        return [
            step
            for step in plan
            if not required.issubset(step)
        ]

    # ---------------------------------------------------------

    def executable(
        self,
        *,
        plan: list[dict],
    ) -> bool:
        """
        Return True if the plan is executable.
        """

        return (
            len(plan) > 0
            and self.validate(
                plan=plan,
            )
        )