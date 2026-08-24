"""Serializable schemas for the unseen-generalization corpus audit.

The first experiment deliberately uses only the Python standard library so it can
run in the existing Windows or WSL checkout before any JAX training starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class AssetKind(str, Enum):
    """Kinds of local evidence relevant to route/generalization research."""

    REPLAY = "replay"
    NOTEBOOK = "notebook"
    PYTHON = "python"
    SUBMISSION_ARCHIVE = "submission_archive"
    SUBMISSION_FILE = "submission_file"
    RECEIPT = "receipt"
    REPORT = "report"
    TABLE = "table"
    JSON = "json"
    OTHER = "other"


@dataclass(frozen=True)
class AssetRecord:
    root_label: str
    relative_path: str
    kind: AssetKind
    size_bytes: int
    mtime_ns: int
    sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        return row


@dataclass(frozen=True)
class PlayerBehavior:
    player_index: int
    final_reward: float | None
    final_status: str | None
    farmer_ops: Mapping[str, int]
    hand_ops: Mapping[str, int]
    market_ops: Mapping[str, int]
    market_items_bought: Mapping[str, int]
    market_items_sold: Mapping[str, int]
    first_op_step: Mapping[str, int]
    daily_public_snapshots: tuple[Mapping[str, Any], ...]
    behavior_signature: str
    coarse_family: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["daily_public_snapshots"] = list(self.daily_public_snapshots)
        return row


@dataclass(frozen=True)
class ReplaySummary:
    source_path: str
    replay_sha256: str
    episode_id: str | None
    submission_ids: tuple[str, ...]
    step_count: int
    player_count: int
    episode_seed: int | None
    players: tuple[PlayerBehavior, ...]
    parse_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["players"] = [player.to_dict() for player in self.players]
        row["submission_ids"] = list(self.submission_ids)
        row["parse_warnings"] = list(self.parse_warnings)
        return row


@dataclass(frozen=True)
class SplitAssignment:
    group_key: str
    split: str
    member_keys: tuple[str, ...]
    coarse_families: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["member_keys"] = list(self.member_keys)
        row["coarse_families"] = list(self.coarse_families)
        return row
