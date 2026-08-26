from pathlib import Path
import json
import sys

if "__file__" in globals():
    _ROOT = Path(globals()["__file__"]).resolve().parent
elif Path("/kaggle_simulations/agent").is_dir():
    _ROOT = Path("/kaggle_simulations/agent")
else:
    _ROOT = Path.cwd()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from meta_agent.src.search_route_policy import SearchRouteController, SearchRoutedTeammateAgent
from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes

_SOURCE = (_ROOT / "teammate_base.py").read_text(encoding="utf-8")
_TAPES = load_action_tapes(_ROOT / "route_actions.json.zlib")
_METADATA = json.loads((_ROOT / "route_library.json").read_text(encoding="utf-8"))
_POLICY_PAYLOAD = json.loads((_ROOT / "route_policy.json").read_text(encoding="utf-8"))
_NASH = json.loads((_ROOT / "opening_nash.json").read_text(encoding="utf-8"))
_ROUTE_BY_FAMILY = {
    str(value["family"]): str(value["route_id"])
    for value in _METADATA["opponent_routes"]
}
_OPENING_WEIGHTS = [
    (str(value["family"]), float(value["weight"]))
    for value in _NASH["opening_support"]
]
_EXPANDED = TeammateExpandedRouteAgent(_SOURCE, _TAPES, "searched_teammate_submission")
_CONTROLLER = SearchRouteController(
    _POLICY_PAYLOAD, _ROUTE_BY_FAMILY, _OPENING_WEIGHTS, rng_seed=None
)
_CONTROLLER.forced_opening = 'G001'
_POLICY = SearchRoutedTeammateAgent(
    _EXPANDED, _CONTROLLER, _POLICY_PAYLOAD.get("targets", ())
)


def agent(observation, configuration=None):
    return _POLICY(observation, configuration)
