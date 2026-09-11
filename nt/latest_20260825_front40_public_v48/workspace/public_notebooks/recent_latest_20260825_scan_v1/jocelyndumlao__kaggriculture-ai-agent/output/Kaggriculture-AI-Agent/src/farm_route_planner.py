"""
farm_route_planner.py

Farm Route Planner for the Kaggriculture AI Agent.

Plans efficient routes between farm locations
using Manhattan distance.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class FarmRoutePlanner:
    """
    Plan routes around the farm.
    """

    # ---------------------------------------------------------

    def distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> int:
        """
        Return Manhattan distance.
        """

        return (
            abs(start[0] - goal[0])
            + abs(start[1] - goal[1])
        )

    # ---------------------------------------------------------

    def nearest(
        self,
        current: tuple[int, int],
        targets: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """
        Return the nearest target.
        """

        if not targets:
            return None

        return min(
            targets,
            key=lambda target: self.distance(
                current,
                target,
            ),
        )

    # ---------------------------------------------------------

    def sort_by_distance(
        self,
        current: tuple[int, int],
        targets: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Return targets ordered by distance.
        """

        return sorted(
            targets,
            key=lambda target: self.distance(
                current,
                target,
            ),
        )

    # ---------------------------------------------------------

    def total_distance(
        self,
        path: list[tuple[int, int]],
    ) -> int:
        """
        Compute total path length.
        """

        if len(path) < 2:
            return 0

        total = 0

        for index in range(
            len(path) - 1
        ):

            total += self.distance(
                path[index],
                path[index + 1],
            )

        return total

    # ---------------------------------------------------------

    def reachable(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_steps: int,
    ) -> bool:
        """
        Return True if goal can be reached
        within max_steps.
        """

        return (
            self.distance(
                start,
                goal,
            )
            <= max_steps
        )