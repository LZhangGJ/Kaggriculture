"""
authorization_manager.py

Authorization Manager for the Kaggriculture AI Agent.

Implements Role-Based Access Control (RBAC).

Author: Jocelyn C. Dumlao
Project: Kaggriculture-AI-Agent
"""

from __future__ import annotations


class AuthorizationManager:
    """
    Manages user roles and permissions.
    """

    def __init__(self):
        self._roles = {}
        self._permissions = {}

    # ---------------------------------------------------------

    def assign_role(
        self,
        user: str,
        role: str,
    ):
        """
        Assign a role to a user.
        """

        self._roles[user] = role

    # ---------------------------------------------------------

    def role(
        self,
        user: str,
    ):
        """
        Return the user's role.
        """

        return self._roles.get(user)

    # ---------------------------------------------------------

    def revoke_role(
        self,
        user: str,
    ):
        """
        Remove a user's role.
        """

        self._roles.pop(user, None)

    # ---------------------------------------------------------

    def grant_permission(
        self,
        role: str,
        permission: str,
    ):
        """
        Grant a permission to a role.
        """

        self._permissions.setdefault(role, set())
        self._permissions[role].add(permission)

    # ---------------------------------------------------------

    def has_permission(
        self,
        user: str,
        permission: str,
    ) -> bool:
        """
        Check whether the user has the specified permission.
        """

        role = self._roles.get(user)

        if role is None:
            return False

        return permission in self._permissions.get(role, set())

    # ---------------------------------------------------------

    def list_roles(self):
        """
        Return all assigned roles.
        """

        return sorted(set(self._roles.values()))

    # ---------------------------------------------------------

    def user_count(self) -> int:
        """
        Return the number of users with assigned roles.
        """

        return len(self._roles)