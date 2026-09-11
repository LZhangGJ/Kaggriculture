"""
animal_scheduler_engine.py

Animal Scheduler Engine for the Kaggriculture AI Agent.

Schedules feeding, care, harvesting, and fertilizer
collection for farm animals.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AnimalSchedulerEngine:
    """
    Manage farm animal schedules.
    """

    def __init__(self):

        self.animals = {}

    # ---------------------------------------------------------

    def add_animal(
        self,
        position: tuple[int, int],
        animal: str,
        current_day: int,
    ) -> None:
        """
        Register a new animal.
        """

        self.animals[position] = {
            "animal": animal.upper(),
            "last_feed": current_day,
            "last_care": current_day,
        }

    # ---------------------------------------------------------

    def feeding_due(
        self,
        current_day: int,
    ) -> list[tuple[int, int]]:
        """
        Return animals that should be fed.
        """

        due = []

        for position, animal in self.animals.items():

            if current_day > animal["last_feed"]:
                due.append(position)

        return due

    # ---------------------------------------------------------

    def care_due(
        self,
        current_day: int,
    ) -> list[tuple[int, int]]:
        """
        Return animals needing care.
        """

        due = []

        for position, animal in self.animals.items():

            if current_day > animal["last_care"]:
                due.append(position)

        return due

    # ---------------------------------------------------------

    def feed(
        self,
        position: tuple[int, int],
        current_day: int,
    ) -> bool:
        """
        Record feeding.
        """

        if position not in self.animals:
            return False

        self.animals[position]["last_feed"] = current_day

        return True

    # ---------------------------------------------------------

    def care(
        self,
        position: tuple[int, int],
        current_day: int,
    ) -> bool:
        """
        Record animal care.
        """

        if position not in self.animals:
            return False

        self.animals[position]["last_care"] = current_day

        return True

    # ---------------------------------------------------------

    def animal_count(
        self,
    ) -> int:
        """
        Return the number of animals.
        """

        return len(self.animals)