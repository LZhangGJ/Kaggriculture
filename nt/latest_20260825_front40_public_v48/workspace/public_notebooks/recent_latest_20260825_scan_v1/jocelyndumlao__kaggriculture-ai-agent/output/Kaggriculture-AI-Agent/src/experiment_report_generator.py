"""
experiment_report_generator.py

Experiment Report Generator for the Kaggriculture AI Agent.

Generates summaries from experiment analysis.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ExperimentReportGenerator:
    """
    Generates experiment reports.
    """

    # ---------------------------------------------------------

    def generate(self, summary: dict) -> dict:
        """
        Return a copy of the experiment summary.
        """
        return dict(summary)

    # ---------------------------------------------------------

    def to_text(self, summary: dict) -> str:
        """
        Convert a summary into a readable report.
        """
        if not summary:
            return "No experiment data available."

        return (
            f"Experiments: {summary['count']}\n"
            f"Best: {summary['best']}\n"
            f"Average: {summary['average']}\n"
            f"Worst: {summary['worst']}"
        )

    # ---------------------------------------------------------

    def has_data(self, summary: dict) -> bool:
        """
        Return True if report contains data.
        """
        return bool(summary)

    # ---------------------------------------------------------

    def metric_count(self, summary: dict) -> int:
        """
        Return experiment count.
        """
        return summary.get("count", 0)

    # ---------------------------------------------------------

    def empty_report(self) -> dict:
        """
        Return an empty report.
        """
        return {
            "count": 0,
            "best": None,
            "average": None,
            "worst": None,
        }