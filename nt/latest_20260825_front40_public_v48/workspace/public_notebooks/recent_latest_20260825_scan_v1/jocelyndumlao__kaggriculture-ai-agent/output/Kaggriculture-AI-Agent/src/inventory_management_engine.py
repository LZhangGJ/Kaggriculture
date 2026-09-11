"""
inventory_management_engine.py

Inventory Management Engine for the Kaggriculture AI Agent.

Tracks inventory quantities and supports
basic inventory operations.

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class InventoryManagementEngine:
    """
    Manage inventory items.
    """

    def __init__(self):

        self.inventory = {}

    # ---------------------------------------------------------

    def add(
        self,
        item: str,
        quantity: int = 1,
    ) -> None:
        """
        Add inventory.
        """

        item = item.upper()

        self.inventory[item] = (
            self.inventory.get(item, 0)
            + quantity
        )

    # ---------------------------------------------------------

    def remove(
        self,
        item: str,
        quantity: int = 1,
    ) -> bool:
        """
        Remove inventory if available.
        """

        item = item.upper()

        if self.count(item) < quantity:
            return False

        self.inventory[item] -= quantity

        if self.inventory[item] == 0:
            del self.inventory[item]

        return True

    # ---------------------------------------------------------

    def count(
        self,
        item: str,
    ) -> int:
        """
        Return quantity.
        """

        return self.inventory.get(
            item.upper(),
            0,
        )

    # ---------------------------------------------------------

    def has(
        self,
        item: str,
        quantity: int = 1,
    ) -> bool:
        """
        Return True if enough inventory exists.
        """

        return self.count(item) >= quantity

    # ---------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all inventory.
        """

        self.inventory.clear()