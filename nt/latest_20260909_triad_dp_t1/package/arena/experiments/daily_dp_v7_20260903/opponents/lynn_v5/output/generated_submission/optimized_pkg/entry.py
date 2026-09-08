"""Collision-safe loader for the optimized pasture agent."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_POLICY = _ROOT / "agents" / "complete_terminal_frontier.py"
_SPEC = importlib.util.spec_from_file_location("optimized_pasture_policy", _POLICY)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load optimized pasture policy")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def policy(obs, configuration=None):
    return _MODULE.agent(obs, configuration)
