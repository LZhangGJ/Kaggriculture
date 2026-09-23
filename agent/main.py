"""Mature replay route tree followed by the opening-free JointAFS R1 policy."""
import importlib.util
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r1_handoff_policy", ROOT.parent / "policy/r1/agent.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

MATURE = ROOT.parent
HANDOFF_CLASSES = frozenset(("default", "competitive_sale", "crop_succession"))
LAND_PRICES = (1000, 2000, 4000)


def sell_before_unfunded_land(observation, action, configuration=None, minimum_step=168):
    """Move already-planned, executable sales ahead of an otherwise failed land buy."""
    step = policy.observed_step(observation)
    market = list((action or {}).get("market", ()))
    if step < minimum_step or not market or not market[0] or market[0][0] != "BUY_LAND":
        return action
    seat = int(policy._get(observation, "player", 0) or 0)
    farms = list(policy._get(observation, "farms", ()) or ())
    farm = farms[seat] if seat < len(farms) else {}
    unlocked = len(policy._get(farm, "unlocked_quadrants", ()) or ())
    extra = unlocked - 1
    money = float(policy._get(farm, "money", 0.0) or 0.0)
    if not 0 <= extra < len(LAND_PRICES) or money >= LAND_PRICES[extra]:
        return action

    from meta_agent.src.market_manager import (
        MARKET_PARAMS, _market_price, _project_unit_storage, _resolved_market_params,
    )
    capacity = int(policy._get(configuration, "shedCapacity", 100) or 100)
    shed, _, _ = _project_unit_storage(observation, action, capacity)
    market_state = policy._get(observation, "market", {}) or {}
    params = _resolved_market_params(market_state)
    inventory = {
        item: int(policy._get(policy._get(market_state, "inventory", {}) or {}, item,
                             row["I0"]) or row["I0"])
        for item, row in params.items()
    }
    moved, revenue = [], 0
    for index, order in enumerate(market[1:], 1):
        if not order or order[0] != "SELL" or len(order) < 3 or order[1] not in MARKET_PARAMS:
            continue
        item = str(order[1])
        quantity = min(max(0, int(order[2])), max(0, int(shed.get(item, 0))))
        if quantity <= 0:
            continue
        moved.append(index)
        # In one market slot the rival can commit at most one unit before each
        # following quote of ours.  Price against that worst same-item sale,
        # and never rely on a later slot whose opening quote is unbounded by us.
        for unit in range(quantity):
            price = _market_price(item, inventory[item] + unit, params)
            revenue += price
            shed[item] -= 1
            if price > 1:
                inventory[item] += 1
        break
    if money + revenue < LAND_PRICES[extra]:
        return action
    result = copy.deepcopy(action)
    result["market"] = [market[index] for index in moved] + [
        order for index, order in enumerate(market) if index not in moved
    ]
    return result


def handoff_selector(value=None):
    value = os.environ.get("REPLAY_HANDOFF_SELECTOR", "") if value is None else value
    if not value or value == "0":
        return None
    from meta_agent.src.route_switch_features import route_switch_feature_names
    from meta_agent.src.search_route_policy import NumpySearchTree
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    feature_schema = payload.get("feature_schema")
    if feature_schema == "semantic_route_switch_v1":
        names = route_switch_feature_names()
    elif feature_schema == "r1_candidate_diff_v1":
        order = ["competitive_sale", "crop_succession"]
        names = [f"{name}_minus_default_{index:03d}"
                 for name in order for index in range(356)]
        if payload.get("feature_mode") != "candidate_diff" or payload.get("feature_order") != order:
            raise ValueError("invalid R1 handoff selector feature order")
    else:
        raise ValueError("invalid R1 handoff selector feature schema")
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    tree = payload.get("tree", {})
    classes = set(map(str, payload.get("classes", ())))
    tree_classes = set(map(str, tree.get("classes", ())))
    if (payload.get("schema") != "r1-handoff-selector-v1" or
            payload.get("feature_names") != names or len(names) not in (147, 712) or
            payload.get("feature_names_sha256") != digest or not classes or
            not classes <= HANDOFF_CLASSES or not tree_classes or
            not tree_classes <= classes or tree.get("class_kind") != "handoff_config" or
            not isinstance(payload.get("eligible"), bool) or
            any(index >= len(names) for index in tree.get("feature", ()) if index >= 0)):
        raise ValueError("invalid R1 handoff selector")
    selector = NumpySearchTree(tree)
    selector.eligible = payload["eligible"]
    selector.feature_schema = feature_schema
    return selector


def replay_deployment(root=ROOT):
    path = Path(root) / "replay_deployment.json"
    if not path.exists() or os.environ.get("REPLAY_DISABLE_DEPLOYMENT") == "1":
        return {}
    value = json.loads(path.read_text())
    if value.get("schema") != "replay-route-to-dynamic-v1":
        raise ValueError("invalid replay route deployment")
    files = {}
    for name, digest in value.get("sha256", {}).items():
        candidate = Path(root) / name
        if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"invalid replay route asset: {name}")
        files[name] = candidate
    return {**value, "files": files}


class ReplayThenDynamicAgent:
    """Execute searched replay routes while keeping the dynamic agent warm.

    The handoff is a step by default.  ``handoff_land`` switches it to a state
    trigger instead: as soon as the farm holds that many unlocked quadrants the
    expansion is finished and the dynamic policy takes over.  ``handoff_step``
    then becomes the deadline.  Every route in the library buys at least two
    quadrants, so the trigger does fire; the deadline is there because a few
    routes buy their last quadrant as late as day 18, and the handoff was only
    measured to be good through day 11-14.  ``handoff_floor`` stays available
    for offline scans that want to forbid an early trigger, but defaults to
    zero: the trigger itself already waits for the expansion to complete.
    """
    def __init__(self, replay, dynamic, handoff_step=288, handoff_land=None,
                 handoff_floor=0, handoff_delay_days=1, selector=None):
        self.replay = replay
        self.dynamic = dynamic
        self.handoff_step = int(handoff_step)
        self.handoff_land = None if handoff_land is None else int(handoff_land)
        self.handoff_floor = 0 if handoff_floor is None else int(handoff_floor)
        # Days to keep running the route after the purchase frame.  Anchoring the
        # handoff to the purchase rather than to an absolute step keeps it
        # correct for any opening: the tapes buy their quadrants on days 8-11.
        self.handoff_delay_days = int(handoff_delay_days)
        self.purchase_step = None
        self.handoff_selector = selector
        self.handoff_selector_done = selector is None
        self.selector_class = "off" if selector is None else None
        self.selector_step = None
        self.selector_installed_index = None
        self.selector_eligible = None if selector is None else selector.eligible
        self.selector_skip_reason = "off" if selector is None else None
        self.capital_sell_first = os.environ.get("REPLAY_CAPITAL_SELL_FIRST", "1") != "0"
        self.capital_sell_first_actions = 0

    def select_handoff(self, observation):
        self.handoff_selector_done = True
        step = policy.observed_step(observation)
        self.selector_step = step
        if step % 24:
            self.selector_skip_reason = "not_day_boundary"
            return
        if not getattr(self.dynamic, "external", False):
            self.selector_skip_reason = "external_false"
            return
        proposals = None
        if self.handoff_selector.feature_schema == "r1_candidate_diff_v1":
            proposals = self.dynamic.prepare_candidates(observation, True)
            bases = [row for row in proposals if not row.get("diagnostic")]
            default = max(bases, key=lambda row: row["score"]) if bases else None
            diagnostics = {name: [row for row in proposals if row.get("diagnostic") == name]
                           for name in ("competitive_sale", "crop_succession")}
            rows = [default, *(values[0] if len(values) == 1 else None
                               for values in diagnostics.values())]
            if any(row is None or len(row.get("features", ())) != 356 for row in rows):
                raise ValueError("invalid R1 handoff candidate features")
            vector = [float(value) - float(base)
                      for row in rows[1:] for value, base in zip(row["features"], default["features"])]
            if len(vector) != 712 or not all(math.isfinite(value) for value in vector):
                raise ValueError("invalid R1 handoff candidate difference")
        else:
            from meta_agent.src.route_switch_features import route_switch_vector
            controller = self.replay.controller
            history = copy.deepcopy(controller.history)
            history.update(observation)
            route_id = controller.route_by_family[controller.current]
            route = self.replay.expanded_agent.action_tapes[route_id]
            vector = route_switch_vector(observation, history, route)
        selected = self.handoff_selector.predict(vector)
        self.selector_class = selected
        if selected == "default":
            return
        matches = [row for row in (proposals or self.dynamic.prepare_candidates(observation))
                   if row.get("diagnostic") == selected]
        if len(matches) != 1:
            raise ValueError(f"missing unique handoff candidate: {selected}")
        self.dynamic.install_candidate(matches[0]["index"])
        self.selector_installed_index = matches[0]["index"]

    @staticmethod
    def unlocked_land(observation):
        seat = int(policy._get(observation, "player", 0) or 0)
        farms = policy._get(observation, "farms", []) or []
        farm = farms[seat] if seat < len(farms) else {}
        return len(policy._get(farm, "unlocked_quadrants", []) or [])

    def ready(self, observation):
        # Seat one carries no ``step`` key; always resolve the clock-derived step.
        step = policy.observed_step(observation)
        if step >= self.handoff_step:
            return True
        if self.handoff_land is None or step < self.handoff_floor:
            return False
        if self.purchase_step is None:
            if self.unlocked_land(observation) < self.handoff_land:
                return False
            self.purchase_step = step
        due = ((self.purchase_step + self.handoff_delay_days * 24 + 23) // 24) * 24
        return step >= due

    def __call__(self, observation, configuration=None):
        if self.ready(observation):
            if not self.handoff_selector_done:
                self.select_handoff(observation)
            return self.dynamic(observation, configuration)
        action = self.replay(observation, configuration)
        if self.capital_sell_first:
            reordered = sell_before_unfunded_land(observation, action, configuration)
            self.capital_sell_first_actions += reordered is not action
            action = reordered
        self.dynamic.observe_external(observation, action)
        return action

    def close(self):
        close = getattr(self.replay, "close", None)
        if close:
            close()
        self.dynamic.close()

    def debug(self):
        controller = self.replay.controller
        return {
            **self.dynamic.debug(), "dynamic_handoff_step": self.handoff_step,
            "dynamic_handoff_land": self.handoff_land,
            "dynamic_handoff_floor": self.handoff_floor,
            "dynamic_handoff_delay_days": self.handoff_delay_days,
            "dynamic_purchase_step": self.purchase_step,
            "selector_class": self.selector_class, "selector_step": self.selector_step,
            "selector_installed_index": self.selector_installed_index,
            "selector_eligible": self.selector_eligible,
            "selector_skip_reason": self.selector_skip_reason,
            "capital_sell_first_actions": self.capital_sell_first_actions,
            "replay_opening": controller.opening, "replay_current": controller.current,
            "replay_switched": controller.switched, "replay_switch_step": controller.switch_step,
        }


def create_replay_agent(value, instance_name):
    sys.path.insert(0, str(MATURE))
    from meta_agent.src.search_route_policy import SearchRouteController, SearchRoutedTeammateAgent
    from meta_agent.src.teammate_expanded_routes import TeammateExpandedRouteAgent, load_action_tapes
    files = value["files"]
    source = files["teammate_base.py"].read_text(encoding="utf-8")
    tapes = load_action_tapes(files["route_actions.json.zlib"])
    metadata = json.loads(files["route_library.json"].read_text())
    route_policy = json.loads(files["route_policy.json"].read_text())
    route_by_family = {
        str(row["family"]): str(row["route_id"])
        for row in metadata["opponent_routes"]
    }
    opening = str(value.get("opening", "G001"))
    weights = [(opening, 1.0)]
    expanded = TeammateExpandedRouteAgent(source, tapes, instance_name)
    controller = SearchRouteController(route_policy, route_by_family, weights)
    forced = os.environ.get("REPLAY_FORCED_OPENING")
    if forced:
        controller.forced_opening = forced
    return SearchRoutedTeammateAgent(expanded, controller, route_policy.get("targets", ()))


def _r1_binary_path():
    """Environment override so offline A/B can compare two builds of agent.so side by side."""
    override = os.environ.get("R1_BINARY_PATH")
    return Path(override) if override else ROOT.parent / "policy/r1/agent.so"


def create_agent(seat=0):
    config = json.loads((ROOT.parent / "policy/r1/config.json").read_text())
    replay = replay_deployment()
    if replay:
        dynamic = policy.Agent(config=config, binary_path=_r1_binary_path())
        route = create_replay_agent(replay, f"searched_replay_seat_{seat}")
        # Environment overrides win, so offline scans can still force a step.
        handoff = int(os.environ.get("REPLAY_HANDOFF_STEP", replay.get("handoff_step", 288)))
        land = os.environ.get("REPLAY_HANDOFF_LAND", replay.get("handoff_land"))
        floor = os.environ.get("REPLAY_HANDOFF_MIN_STEP", replay.get("handoff_floor"))
        delay = os.environ.get("REPLAY_HANDOFF_LAND_DELAY", replay.get("handoff_land_delay_days", 1))
        return ReplayThenDynamicAgent(
            route, dynamic, handoff,
            handoff_land=None if land in (None, "") else int(land),
            handoff_floor=None if floor in (None, "") else int(floor),
            handoff_delay_days=0 if delay in (None, "") else int(delay),
            selector=handoff_selector(),
        )
    return policy.Agent(config=config, binary_path=_r1_binary_path())


_instances = {}


def agent(observation, configuration=None):
    # Normalise once at the entry point so every component below (route tree,
    # tape scheduler, teammate base, R1) reads the same clock-derived step.
    observation = policy.normalize_observation(observation)
    seat = int(policy._get(observation, "player", 0))
    if seat not in _instances:
        _instances[seat] = create_agent(seat)
    return _instances[seat](observation, configuration)
