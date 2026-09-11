"""Shape the shop, work the pasture: guarded submission entry point."""
from __future__ import annotations

from optimized_pkg.entry import policy as _policy


# The raw competition loader selects the final callable in insertion order.
def kaggriculture_agent(obs, configuration=None):
    return _policy(obs, configuration)
