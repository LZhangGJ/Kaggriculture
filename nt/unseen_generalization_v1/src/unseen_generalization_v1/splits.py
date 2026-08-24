"""Leakage-resistant split construction for replay behavior groups."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable

from .schema import ReplaySummary, SplitAssignment


def _fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _member_key(summary: ReplaySummary, player_index: int) -> str:
    episode = summary.episode_id or summary.replay_sha256[:16]
    return f"{episode}:p{player_index}:{summary.replay_sha256[:12]}"


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Stable root selection makes output independent of replay input order.
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def build_group_splits(
    summaries: Iterable[ReplaySummary],
    *,
    seed: int = 20260824,
    validation_fraction: float = 0.15,
    family_holdout_fraction: float = 0.20,
) -> dict[str, Any]:
    """Assign connected agent/behavior groups to train, validation, or holdout.

    Members are unioned when they share an exact behavior signature or a known
    Kaggle submission ID.  This prevents episodes from the same submitted agent
    from leaking across train and validation even when stochastic events produce
    different exact action signatures.  Whole coarse route families are then
    reserved for ``family_holdout`` when at least three families are available.
    """

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= family_holdout_fraction < 1.0:
        raise ValueError("family_holdout_fraction must be in [0, 1)")

    dsu = _DisjointSet()
    signature_owner: dict[str, str] = {}
    agent_owner: dict[str, str] = {}
    row_metadata: dict[str, dict[str, Any]] = {}

    for summary in summaries:
        ordered_submission_ids = summary.submission_ids
        for player in summary.players:
            member = _member_key(summary, player.player_index)
            dsu.add(member)
            agent_hint = ""
            if len(ordered_submission_ids) == summary.player_count and player.player_index < len(ordered_submission_ids):
                agent_hint = ordered_submission_ids[player.player_index]
            metadata = {
                "episode_id": summary.episode_id,
                "replay_sha256": summary.replay_sha256,
                "player_index": player.player_index,
                "behavior_signature": player.behavior_signature,
                "coarse_family": player.coarse_family,
                "episode_seed": summary.episode_seed,
                "agent_hint": agent_hint,
            }
            row_metadata[member] = metadata

            old_signature = signature_owner.setdefault(player.behavior_signature, member)
            dsu.union(member, old_signature)
            if agent_hint:
                old_agent = agent_owner.setdefault(agent_hint, member)
                dsu.union(member, old_agent)

    components: defaultdict[str, list[str]] = defaultdict(list)
    for member in sorted(row_metadata):
        components[dsu.find(member)].append(member)

    families = sorted({str(row["coarse_family"]) for row in row_metadata.values()})
    holdout_families: set[str] = set()
    if len(families) >= 3 and family_holdout_fraction > 0:
        target = max(1, round(len(families) * family_holdout_fraction))
        ranked = sorted(families, key=lambda family: (_fraction(f"family:{family}", seed), family))
        holdout_families.update(ranked[:target])

    assignments: list[SplitAssignment] = []
    member_to_split: dict[str, str] = {}

    for component_root, members in sorted(components.items()):
        member_keys = tuple(sorted(members))
        coarse_families = tuple(sorted({str(row_metadata[member]["coarse_family"]) for member in members}))
        agent_hints = tuple(sorted({str(row_metadata[member]["agent_hint"]) for member in members if row_metadata[member]["agent_hint"]}))
        signatures = tuple(sorted({str(row_metadata[member]["behavior_signature"]) for member in members}))
        group_key = hashlib.sha256(
            ("agents=" + ",".join(agent_hints) + "|signatures=" + ",".join(signatures)).encode("utf-8")
        ).hexdigest()

        if any(family in holdout_families for family in coarse_families):
            split = "family_holdout"
            reason = "connected group contains a coarse route family reserved as unseen holdout"
        elif _fraction(f"component:{group_key}", seed) < validation_fraction:
            split = "validation"
            reason = "connected agent/behavior group selected by deterministic hash"
        else:
            split = "train"
            reason = "connected agent/behavior group assigned to development training"

        assignments.append(
            SplitAssignment(
                group_key=group_key,
                split=split,
                member_keys=member_keys,
                coarse_families=coarse_families,
                reason=reason,
            )
        )
        for member in member_keys:
            old = member_to_split.setdefault(member, split)
            if old != split:
                raise RuntimeError(f"member assigned to multiple splits: {member}")

    split_counts: defaultdict[str, int] = defaultdict(int)
    family_split_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for member, split in member_to_split.items():
        split_counts[split] += 1
        family = str(row_metadata[member]["coarse_family"])
        family_split_counts[family][split] += 1

    signature_splits: defaultdict[str, set[str]] = defaultdict(set)
    agent_splits: defaultdict[str, set[str]] = defaultdict(set)
    for member, split in member_to_split.items():
        metadata = row_metadata[member]
        signature_splits[str(metadata["behavior_signature"])].add(split)
        if metadata["agent_hint"]:
            agent_splits[str(metadata["agent_hint"])].add(split)
    signature_leaks = {key: sorted(value) for key, value in signature_splits.items() if len(value) > 1}
    agent_leaks = {key: sorted(value) for key, value in agent_splits.items() if len(value) > 1}
    if signature_leaks or agent_leaks:
        raise RuntimeError(
            f"split leakage detected; signatures={signature_leaks}, agents={agent_leaks}"
        )

    return {
        "schema_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "family_holdout_fraction": family_holdout_fraction,
        "holdout_families": sorted(holdout_families),
        "split_counts": dict(sorted(split_counts.items())),
        "family_split_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_split_counts.items())
        },
        "assignments": [assignment.to_dict() for assignment in assignments],
        "rows": {
            member: {**row_metadata[member], "split": member_to_split[member]}
            for member in sorted(member_to_split)
        },
        "leakage_checks": {
            "behavior_signature_cross_split": 0,
            "submission_id_cross_split": 0,
            "status": "PASS",
        },
    }
