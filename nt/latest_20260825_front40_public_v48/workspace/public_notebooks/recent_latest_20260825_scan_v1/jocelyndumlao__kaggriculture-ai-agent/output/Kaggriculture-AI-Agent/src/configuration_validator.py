"""
configuration_validator.py

Configuration Validator for the Kaggriculture AI Agent.

Validates configuration dictionaries before application startup.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class ConfigurationValidator:
    """
    Validates configuration dictionaries.
    """

    def __init__(self):
        self._errors = []

    # ---------------------------------------------------------

    def validate(
        self,
        config: dict,
        schema: dict,
    ) -> bool:
        """
        Validate a configuration against a schema.

        Schema example:
            {
                "host": str,
                "port": int,
            }
        """

        self._errors.clear()

        for key, expected_type in schema.items():

            if key not in config:
                self._errors.append(
                    f"Missing key: {key}"
                )
                continue

            if not isinstance(
                config[key],
                expected_type,
            ):
                self._errors.append(
                    f"Invalid type for '{key}'"
                )

        return len(self._errors) == 0

    # ---------------------------------------------------------

    def errors(self):
        """
        Return validation errors.
        """

        return list(self._errors)

    # ---------------------------------------------------------

    def has_errors(self) -> bool:
        """
        Return True if validation failed.
        """

        return len(self._errors) > 0

    # ---------------------------------------------------------

    def error_count(self) -> int:
        """
        Return the number of validation errors.
        """

        return len(self._errors)