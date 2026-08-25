"""
beam_search.py

Beam Search implementation for the Kaggriculture AI Agent.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class BeamSearch:
    """
    Keep only the best K candidate states.
    """

    def __init__(self, beam_width: int = 3):
        self.beam_width = beam_width

    # ---------------------------------------------------------

    def select(self, candidates):
        """
        Return the top-K highest scoring candidates.

        Parameters
        ----------
        candidates : list[tuple]
            List of (state, score).

        Returns
        -------
        list[tuple]
        """

        ordered = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True,
        )

        return ordered[: self.beam_width]

    # ---------------------------------------------------------

    def best(self, candidates):
        """
        Return the highest-scoring candidate.
        """

        selected = self.select(candidates)

        if not selected:
            return None

        return selected[0]

    # ---------------------------------------------------------

    def update_width(self, width: int):
        """
        Update the beam width.
        """

        if width > 0:
            self.beam_width = width
            