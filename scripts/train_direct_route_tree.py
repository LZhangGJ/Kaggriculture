"""Train a step-48 route-family tree from paired route/community outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np

from meta_agent.src.recurrent_meta import market_town_vector, public_farm_vector
from meta_agent.src.equilibrium_selector import EquilibriumOpeningSelector
from meta_agent.src.replay_trie_agent import ReplayTrieAgent
from meta_agent.src.selector import RouteDecision
from meta_agent.src.simple_route_tree import opponent_route_features


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "route-runtime-v8"
DESCRIPTORS = ROOT / "branch-descriptors-v8.pkl"
RECURRENT_CHECKPOINT = ROOT / "recurrent-route-best.pt"
MIXTURE = ROOT / "opening-mixture-v8.json"
CHECKPOINT = 48
OPPONENTS = (
    "public_b85", "schedule_93311715", "frontier_soil",
    "adaptive_farming", "kaito_v35",
)
CONFIG = {
    "episodeSteps": 720, "turnsPerDay": 24, "shedCapacity": 100,
    "boardSize": 10, "startingMoney": 3000, "maxMarketOrders": 10,
    "farmHandCostMult": 1,
}
_OPPONENT_CACHE: dict[str, Any] = {}
_INTERVENTION_POLICY: tuple[Any, Any] | None = None


class ForcedNextFamilySelector(EquilibriumOpeningSelector):
    """Intervene on only the step-48-to-72 semantic branch.

    All later checkpoints return to the recurrent policy.  ``force_slot`` is
    an index into the sorted legal family hashes, so the same paired opening
    can be replayed once for every legal continuation without knowing its
    exact-prefix node before the episode starts.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.force_slot = 0
        self._force_checkpoint = -1
        self._force_prefix = ""
        self._force_choice: dict[str, Any] | None = None
        self.intervention: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._force_checkpoint = -1
        self._force_prefix = ""
        self._force_choice = None
        self.intervention = None

    def _choose_candidate(self, scores: np.ndarray) -> tuple[int, bool]:
        teacher_index = int(np.argmax(scores))
        if self._force_checkpoint != CHECKPOINT:
            return teacher_index, False
        families = tuple(str(value) for value in self.selector.candidate_family_hashes)
        ordered = sorted(families)
        legal = 0 <= int(self.force_slot) < len(ordered)
        target = ordered[int(self.force_slot)] if legal else families[teacher_index]
        selected_index = families.index(target)
        self._force_choice = {
            "legal": legal,
            "families": ordered,
            "family": target,
            "teacher_family": families[teacher_index],
        }
        return selected_index, False

    def select(self, fingerprint: dict[str, Any], checkpoint: int):
        checkpoint = int(checkpoint)
        self._force_checkpoint = -1
        self._force_prefix = ""
        self._force_choice = None
        if checkpoint == CHECKPOINT and self.selector.current_route_id is not None:
            current = next(
                row for row in self.selector._rows(checkpoint)
                if str(row[0]) == str(self.selector.current_route_id)
            )
            self._force_checkpoint = checkpoint
            self._force_prefix = str(current[2])
        try:
            decision = super().select(fingerprint, checkpoint)
        finally:
            self._force_checkpoint = -1
        if checkpoint == CHECKPOINT and self._force_choice is not None:
            current = self.encoded_history[-1]
            self.intervention = {
                **self._force_choice,
                "prefix": self._force_prefix,
                "counts": opponent_route_features(
                    current.opponent_public, current.market_town, "opponent_counts"
                ),
                "context": opponent_route_features(
                    current.opponent_public, current.market_town, "opponent_context"
                ),
            }
        return decision


class FixedRouteSelector:
    def __init__(self, route_id: str) -> None:
        with (RUNTIME / "manifest.pkl").open("rb") as handle:
            manifest = pickle.load(handle)
        self.route_id = str(route_id)
        self.checkpoints = [int(value) for value in manifest["checkpoints"]]
        offset, _ = manifest["route_offsets"][self.route_id]
        with (RUNTIME / "actions.bin").open("rb") as handle:
            handle.seek(int(offset))
            self.actions = pickle.load(handle)

    def reset(self) -> None:
        pass

    def select(self, fingerprint, checkpoint: int) -> RouteDecision:
        return RouteDecision(
            checkpoint=int(checkpoint), next_checkpoint=int(checkpoint),
            route_id=self.route_id, family_hash="fixed", changed_route=False,
            score=0.0, distance={}, support=1, median_reward=0.0,
        )

    def action(self, step: int):
        if 0 <= step < len(self.actions):
            return self.actions[step]
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _worker_init() -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"


def _opponent(name: str):
    if name not in _OPPONENT_CACHE:
        from kaggrl.community_agents import community_agent_source
        from meta_agent.src.benchmark_community_round_robin import LoadedAgent
        _OPPONENT_CACHE[name] = LoadedAgent(
            community_agent_source(name), f"direct_tree_{name}_{os.getpid()}"
        )
    return _OPPONENT_CACHE[name]


def _mixture_salt(seed: int, seat: int) -> int:
    raw = f"simple-route-tree-v1:{seed}:{seat}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def _intervention_policy() -> tuple[Any, ForcedNextFamilySelector]:
    global _INTERVENTION_POLICY
    if _INTERVENTION_POLICY is None:
        selector = ForcedNextFamilySelector(
            str(RUNTIME), str(DESCRIPTORS),
            tree_q_model_path=None,
            recurrent_model_path=str(RECURRENT_CHECKPOINT),
            residual_weight=1.0,
            enable_trading=False,
            epsilon=0.0,
            top_k=16,
            prior_weight=0.0,
            seed=0,
            stochastic_softmax=True,
            mixture_path=str(MIXTURE),
            mixture_salt=0,
        )
        policy = ReplayTrieAgent(
            RUNTIME, selector_override=selector,
            manage_sells=True, lead_sells=True, lead_turns=5,
            lead_batch=20, lead_max_distance=8, repair_weeds=True,
            weed_replay_steps=8,
        )
        _INTERVENTION_POLICY = policy, selector
    return _INTERVENTION_POLICY


def _play_intervention(task: tuple[str, int, int, int]) -> dict[str, Any]:
    opponent_name, seed, seat, force_slot = task
    try:
        from fast_kaggriculture import Config, FastEnv

        policy, selector = _intervention_policy()
        selector.force_slot = int(force_slot)
        salt = _mixture_salt(seed, seat)
        selector.fixed_mixture_salt = salt
        selector.mixture_salt = salt
        opponent = _opponent(opponent_name)
        agents = [policy, opponent] if seat == 0 else [opponent, policy]
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        while not env.done:
            step = int(env.step_count)
            for observation in observations:
                observation["step"] = step
            actions = [agents[index](observations[index], CONFIG) for index in range(2)]
            observations = list(env.step(actions))
        info = selector.intervention
        if info is None:
            raise RuntimeError("step-48 intervention was not reached")
        rewards = [float(value) for value in env.rewards]
        margin = rewards[seat] - rewards[1 - seat]
        return {
            "opponent": opponent_name, "seed": seed, "seat": seat,
            "force_slot": force_slot, "prefix": info["prefix"],
            "family": info["family"], "families": info["families"],
            "teacher_family": info["teacher_family"], "legal": info["legal"],
            "counts": info["counts"], "context": info["context"],
            "margin": margin,
            "score": float(margin > 0) + 0.5 * float(margin == 0),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "task": task}


def collect_interventions(output: Path, seeds: Sequence[int], workers: int) -> None:
    # The audited step-48 nodes have at most four legal next families.
    tasks = [
        (opponent, seed, seat, slot)
        for opponent in OPPONENTS for seed in seeds for seat in (0, 1)
        for slot in range(4)
    ]
    count = min(workers or (os.cpu_count() or 1), len(tasks))
    rows = []
    with mp.get_context("spawn").Pool(count, initializer=_worker_init) as pool:
        for row in pool.imap_unordered(_play_intervention, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(tasks):
                print(f"[{len(rows):05d}/{len(tasks):05d}]", flush=True)
    errors = [row for row in rows if "error" in row]
    if errors:
        raise RuntimeError(f"{len(errors)} failures; first={errors[0]}")
    legal_rows = [row for row in rows if row["legal"]]
    legal_rows.sort(key=lambda row: (
        row["opponent"], row["seed"], row["seat"], row["force_slot"]
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefixes=np.asarray([row["prefix"] for row in legal_rows]),
        families=np.asarray([row["family"] for row in legal_rows]),
        teacher_families=np.asarray([row["teacher_family"] for row in legal_rows]),
        opponents=np.asarray([row["opponent"] for row in legal_rows]),
        seeds=np.asarray([row["seed"] for row in legal_rows], dtype=np.int64),
        seats=np.asarray([row["seat"] for row in legal_rows], dtype=np.int8),
        counts=np.stack([row["counts"] for row in legal_rows]),
        context=np.stack([row["context"] for row in legal_rows]),
        margins=np.asarray([row["margin"] for row in legal_rows], dtype=np.float32),
        scores=np.asarray([row["score"] for row in legal_rows], dtype=np.float32),
    )
    print(json.dumps({
        "output": str(output), "episodes": len(rows),
        "legal_games": len(legal_rows), "seeds": len(seeds),
    }, indent=2))


def _routes(max_per_family: int) -> tuple[list[str], dict[str, tuple[str, str]]]:
    with (RUNTIME / f"checkpoint-{CHECKPOINT:03d}.pkl").open("rb") as handle:
        rows = pickle.load(handle)
    nodes: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for route_id, reward, prefix, family, _ in rows:
        nodes[str(prefix)][str(family)].append((str(route_id), float(reward)))
    selected: list[str] = []
    mapping: dict[str, tuple[str, str]] = {}
    for prefix, families in nodes.items():
        if len(families) <= 1:
            continue
        for family, values in families.items():
            values.sort(key=lambda row: (row[1], row[0]))
            if len(values) <= max_per_family:
                sample = values
            else:
                indices = np.linspace(0, len(values) - 1, max_per_family).round().astype(int)
                sample = [values[index] for index in sorted(set(indices.tolist()))]
            for route_id, _ in sample:
                selected.append(route_id)
                mapping[route_id] = prefix, family
    return selected, mapping


def _play(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    route_id, prefix, family, opponent_name, seed, seat = task
    try:
        from fast_kaggriculture import Config, FastEnv
        selector = FixedRouteSelector(route_id)
        policy = ReplayTrieAgent(
            RUNTIME, selector_override=selector,
            manage_sells=True, lead_sells=True, lead_turns=5,
            lead_batch=20, lead_max_distance=8, repair_weeds=True,
            weed_replay_steps=8,
        )
        opponent = _opponent(opponent_name)
        agents = [policy, opponent] if seat == 0 else [opponent, policy]
        env = FastEnv(Config(), seed)
        observations = list(env.reset(seed))
        counts = context = None
        while not env.done:
            step = int(env.step_count)
            for observation in observations:
                observation["step"] = step
            if step == CHECKPOINT:
                observation = observations[seat]
                opponent_public = public_farm_vector(observation["farms"][1 - seat])
                market = market_town_vector(observation)
                counts = opponent_route_features(
                    opponent_public, market, "opponent_counts"
                )
                context = opponent_route_features(
                    opponent_public, market, "opponent_context"
                )
            actions = [agents[index](observations[index], CONFIG) for index in range(2)]
            observations = list(env.step(actions))
        rewards = [float(value) for value in env.rewards]
        margin = rewards[seat] - rewards[1 - seat]
        return {
            "route_id": route_id, "prefix": prefix, "family": family,
            "opponent": opponent_name, "seed": seed, "seat": seat,
            "counts": counts, "context": context, "margin": margin,
            "score": float(margin > 0) + 0.5 * float(margin == 0),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}", "task": task}


def collect(output: Path, seeds: Sequence[int], workers: int, max_per_family: int) -> None:
    routes, mapping = _routes(max_per_family)
    tasks = [
        (route, *mapping[route], opponent, seed, seat)
        for route in routes for opponent in OPPONENTS for seed in seeds for seat in (0, 1)
    ]
    count = min(workers or (os.cpu_count() or 1), len(tasks))
    rows = []
    with mp.get_context("spawn").Pool(count, initializer=_worker_init) as pool:
        for row in pool.imap_unordered(_play, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 500 == 0 or len(rows) == len(tasks):
                print(f"[{len(rows):05d}/{len(tasks):05d}]", flush=True)
    errors = [row for row in rows if "error" in row]
    if errors:
        raise RuntimeError(f"{len(errors)} failures; first={errors[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        route_ids=np.asarray([row["route_id"] for row in rows]),
        prefixes=np.asarray([row["prefix"] for row in rows]),
        families=np.asarray([row["family"] for row in rows]),
        opponents=np.asarray([row["opponent"] for row in rows]),
        seeds=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        seats=np.asarray([row["seat"] for row in rows], dtype=np.int8),
        counts=np.stack([row["counts"] for row in rows]),
        context=np.stack([row["context"] for row in rows]),
        margins=np.asarray([row["margin"] for row in rows], dtype=np.float32),
        scores=np.asarray([row["score"] for row in rows], dtype=np.float32),
    )
    print(json.dumps({
        "output": str(output), "routes": len(routes), "games": len(rows),
        "seeds": len(seeds),
    }, indent=2))


def _group_dataset(data, feature_set: str):
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(
        data["prefixes"], data["opponents"], data["seeds"], data["seats"]
    )):
        groups[(str(key[0]), str(key[1]), int(key[2]), int(key[3]))].append(index)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (prefix, opponent, seed, seat), indices in groups.items():
        family_values: dict[str, list[float]] = defaultdict(list)
        for index in indices:
            utility = float(data["scores"][index]) + 0.02 * float(
                np.tanh(data["margins"][index] / 20_000.0)
            )
            family_values[str(data["families"][index])].append(utility)
        means = {family: float(np.mean(values)) for family, values in family_values.items()}
        best = max(means, key=lambda family: (means[family], family))
        result[prefix].append({
            "x": np.asarray(data[feature_set][indices[0]], dtype=np.float32),
            "label": best, "payoffs": means,
            "opponent": opponent, "seed": seed, "seat": seat,
        })
    return result


def train(
    train_path: Path, valid_path: Path, output: Path, feature_set: str
) -> None:
    from sklearn.tree import DecisionTreeClassifier
    train_groups = _group_dataset(np.load(train_path), feature_set)
    valid_groups = _group_dataset(np.load(valid_path), feature_set)
    trees = {}
    metrics = {}
    regrets = []
    for prefix, train_rows in train_groups.items():
        valid_rows = valid_groups.get(prefix, [])
        labels = {row["label"] for row in train_rows}
        if len(labels) <= 1 or len(train_rows) < 20 or not valid_rows:
            continue
        x_train = np.stack([row["x"] for row in train_rows])
        y_train = np.asarray([row["label"] for row in train_rows])
        x_valid = np.stack([row["x"] for row in valid_rows])
        y_valid = np.asarray([row["label"] for row in valid_rows])
        best = None
        for depth in (1, 2, 3, 4):
            tree = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=4, random_state=20260824
            ).fit(x_train, y_train)
            prediction = tree.predict(x_valid)
            regret = np.mean([
                max(row["payoffs"].values())
                - row["payoffs"].get(str(label), min(row["payoffs"].values()))
                for row, label in zip(valid_rows, prediction)
            ])
            candidate = (-float(regret), -depth, tree)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        negative_regret, negative_depth, tree = best
        prediction = tree.predict(x_valid)
        trees[f"{CHECKPOINT}:{prefix}"] = tree
        metrics[prefix] = {
            "train_states": len(train_rows), "valid_states": len(valid_rows),
            "families": len(labels), "depth": -negative_depth,
            "accuracy": float(np.mean(prediction == y_valid)),
            "mean_regret": -negative_regret,
        }
        regrets.append(-negative_regret)
    payload = {
        "schema_version": 1, "model_kind": "direct_family",
        "feature_set": (
            "opponent_counts" if feature_set == "counts" else "opponent_context"
        ),
        "checkpoint": CHECKPOINT, "trees": trees, "metrics": metrics,
        "mean_node_regret": float(np.mean(regrets)) if regrets else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    print(json.dumps({
        "output": str(output), "trees": len(trees),
        "mean_node_regret": payload["mean_node_regret"], "metrics": metrics,
    }, indent=2))


def train_committed(
    train_path: Path, valid_path: Path, output: Path, feature_set: str
) -> None:
    from sklearn.tree import DecisionTreeRegressor
    train_data = np.load(train_path)
    valid_data = np.load(valid_path)
    route_models: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    prefixes = sorted(set(str(value) for value in train_data["prefixes"]))
    for prefix in prefixes:
        train_prefix = np.asarray(train_data["prefixes"] == prefix)
        valid_prefix = np.asarray(valid_data["prefixes"] == prefix)
        routes = sorted(set(str(value) for value in train_data["route_ids"][train_prefix]))
        models = {}
        for route in routes:
            mask = train_prefix & (train_data["route_ids"] == route)
            x = np.asarray(train_data[feature_set][mask], dtype=np.float32)
            utility = np.asarray(train_data["scores"][mask], dtype=np.float32) + 0.02 * np.tanh(
                np.asarray(train_data["margins"][mask], dtype=np.float32) / 20_000.0
            )
            models[route] = DecisionTreeRegressor(
                max_depth=2, min_samples_leaf=8, random_state=20260824
            ).fit(x, utility)
        groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for index in np.flatnonzero(valid_prefix):
            key = (
                str(valid_data["opponents"][index]),
                int(valid_data["seeds"][index]),
                int(valid_data["seats"][index]),
            )
            groups[key].append(int(index))
        regrets = []
        realized = []
        oracle = []
        selections: dict[str, int] = defaultdict(int)
        for indices in groups.values():
            x = np.asarray(valid_data[feature_set][indices[0]], dtype=np.float32).reshape(1, -1)
            predicted = max(
                models,
                key=lambda route: (float(models[route].predict(x)[0]), route),
            )
            actual = {
                str(valid_data["route_ids"][index]): float(valid_data["scores"][index])
                + 0.02 * float(np.tanh(valid_data["margins"][index] / 20_000.0))
                for index in indices
            }
            best = max(actual.values())
            chosen = actual.get(predicted, min(actual.values()))
            regrets.append(best - chosen)
            realized.append(chosen)
            oracle.append(best)
            selections[predicted] += 1
        node = f"{CHECKPOINT}:{prefix}"
        route_models[node] = models
        metrics[prefix] = {
            "routes": len(routes), "valid_states": len(groups),
            "mean_regret": float(np.mean(regrets)),
            "mean_realized_utility": float(np.mean(realized)),
            "mean_oracle_utility": float(np.mean(oracle)),
            "selected_routes": dict(selections),
        }
    payload = {
        "schema_version": 1, "model_kind": "committed_route",
        "feature_set": (
            "opponent_counts" if feature_set == "counts" else "opponent_context"
        ),
        "checkpoint": CHECKPOINT, "route_models": route_models,
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    print(json.dumps({
        "output": str(output), "nodes": len(route_models), "metrics": metrics,
    }, indent=2))


def _wilson_lower(scores: Sequence[float], z: float = 1.2815515655446004) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        return -1.0
    probability = float(np.mean(values))
    denominator = 1.0 + z * z / len(values)
    center = probability + z * z / (2.0 * len(values))
    radius = z * np.sqrt(
        probability * (1.0 - probability) / len(values)
        + z * z / (4.0 * len(values) ** 2)
    )
    return float((center - radius) / denominator)


def _intervention_states(data: Any, feature_set: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, values in enumerate(zip(
        data["prefixes"], data["opponents"], data["seeds"], data["seats"]
    )):
        groups[(str(values[0]), str(values[1]), int(values[2]), int(values[3]))].append(index)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (prefix, opponent, seed, seat), indices in groups.items():
        payoffs = {
            str(data["families"][index]): (
                float(data["scores"][index]), float(data["margins"][index])
            )
            for index in indices
        }
        result[prefix].append({
            "x": np.asarray(data[feature_set][indices[0]], dtype=np.float32),
            "opponent": opponent, "seed": seed, "seat": seat,
            "teacher_family": str(data["teacher_families"][indices[0]]),
            "payoffs": payoffs,
        })
    return result


def train_robust_intervention(
    train_path: Path, valid_path: Path, output: Path, feature_set: str
) -> None:
    """Learn opponent archetypes from robust aggregate next-family responses."""
    from sklearn.tree import DecisionTreeClassifier

    train_states = _intervention_states(np.load(train_path), feature_set)
    valid_states = _intervention_states(np.load(valid_path), feature_set)
    trees: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for prefix, rows in train_states.items():
        valid_rows = valid_states.get(prefix, [])
        if not valid_rows:
            continue
        opponents = sorted({row["opponent"] for row in rows})
        best_by_opponent: dict[str, str] = {}
        response_stats: dict[str, Any] = {}
        for opponent in opponents:
            opponent_rows = [row for row in rows if row["opponent"] == opponent]
            families = sorted(set.intersection(*(
                set(row["payoffs"]) for row in opponent_rows
            )))
            family_stats = {}
            for family in families:
                scores = [row["payoffs"][family][0] for row in opponent_rows]
                margins = [row["payoffs"][family][1] for row in opponent_rows]
                seat_lowers = [
                    _wilson_lower([
                        row["payoffs"][family][0] for row in opponent_rows
                        if row["seat"] == seat
                    ]) for seat in (0, 1)
                ]
                family_stats[family] = {
                    "games": len(scores), "score_rate": float(np.mean(scores)),
                    "mean_margin": float(np.mean(margins)),
                    "overall_lower80": _wilson_lower(scores),
                    "worst_seat_lower80": min(seat_lowers),
                    "seat_lower80": seat_lowers,
                }
            best = max(families, key=lambda family: (
                family_stats[family]["worst_seat_lower80"],
                family_stats[family]["overall_lower80"],
                family_stats[family]["score_rate"],
                family_stats[family]["mean_margin"], family,
            ))
            best_by_opponent[opponent] = best
            response_stats[opponent] = {
                "selected": best, "families": family_stats,
            }

        x_train = np.stack([row["x"] for row in rows])
        y_train = np.asarray([best_by_opponent[row["opponent"]] for row in rows])
        candidates = []
        for depth in (1, 2, 3):
            tree = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=4, random_state=20260824
            ).fit(x_train, y_train)
            predictions = tree.predict(np.stack([row["x"] for row in valid_rows]))
            realized_scores = []
            realized_margins = []
            teacher_scores = []
            teacher_margins = []
            valid_labels = []
            for row, prediction in zip(valid_rows, predictions):
                fallback = min(row["payoffs"].values(), key=lambda value: value[0])
                realized = row["payoffs"].get(str(prediction), fallback)
                teacher = row["payoffs"].get(row["teacher_family"], fallback)
                realized_scores.append(realized[0])
                realized_margins.append(realized[1])
                teacher_scores.append(teacher[0])
                teacher_margins.append(teacher[1])
                valid_labels.append(best_by_opponent[row["opponent"]])
            candidate = (
                float(np.mean(realized_scores)),
                float(np.mean(realized_margins)), -depth, tree,
                predictions, valid_labels, teacher_scores, teacher_margins,
            )
            candidates.append(candidate)
        selected = max(candidates, key=lambda row: row[:3])
        (
            realized_rate, realized_margin, negative_depth, tree,
            predictions, valid_labels, teacher_scores, teacher_margins,
        ) = selected
        trees[f"{CHECKPOINT}:{prefix}"] = tree
        metrics[prefix] = {
            "train_states": len(rows), "valid_states": len(valid_rows),
            "depth": -negative_depth, "leaves": int(tree.get_n_leaves()),
            "best_by_opponent": best_by_opponent,
            "response_stats": response_stats,
            "valid_label_accuracy": float(np.mean(predictions == valid_labels)),
            "valid_intervention_score_rate": realized_rate,
            "valid_intervention_mean_margin": realized_margin,
            "valid_teacher_score_rate": float(np.mean(teacher_scores)),
            "valid_teacher_mean_margin": float(np.mean(teacher_margins)),
        }
    payload = {
        "schema_version": 1, "model_kind": "direct_family",
        "feature_set": (
            "opponent_counts" if feature_set == "counts" else "opponent_context"
        ),
        "checkpoint": CHECKPOINT, "trees": trees, "metrics": metrics,
        "training_kind": "paired_step48_family_intervention_robust_seats",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
    print(json.dumps({
        "output": str(output), "trees": len(trees), "metrics": metrics,
    }, indent=2))


def _seed_range(value: str) -> tuple[int, ...]:
    start, separator, stop = value.partition(":")
    return tuple(range(int(start), int(stop))) if separator else (int(value),)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--seeds", type=_seed_range, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--workers", type=int, default=0)
    collect_parser.add_argument("--max-per-family", type=int, default=8)
    intervention_parser = sub.add_parser("collect-interventions")
    intervention_parser.add_argument("--seeds", type=_seed_range, required=True)
    intervention_parser.add_argument("--output", type=Path, required=True)
    intervention_parser.add_argument("--workers", type=int, default=0)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train", type=Path, required=True)
    train_parser.add_argument("--valid", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--feature-set", choices=("counts", "context"), required=True)
    commit_parser = sub.add_parser("train-commit")
    commit_parser.add_argument("--train", type=Path, required=True)
    commit_parser.add_argument("--valid", type=Path, required=True)
    commit_parser.add_argument("--output", type=Path, required=True)
    commit_parser.add_argument("--feature-set", choices=("counts", "context"), required=True)
    robust_parser = sub.add_parser("train-robust")
    robust_parser.add_argument("--train", type=Path, required=True)
    robust_parser.add_argument("--valid", type=Path, required=True)
    robust_parser.add_argument("--output", type=Path, required=True)
    robust_parser.add_argument("--feature-set", choices=("counts", "context"), required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.output, args.seeds, args.workers, args.max_per_family)
    elif args.command == "collect-interventions":
        collect_interventions(args.output, args.seeds, args.workers)
    elif args.command == "train":
        train(args.train, args.valid, args.output, args.feature_set)
    elif args.command == "train-commit":
        train_committed(args.train, args.valid, args.output, args.feature_set)
    else:
        train_robust_intervention(
            args.train, args.valid, args.output, args.feature_set
        )


if __name__ == "__main__":
    main()
