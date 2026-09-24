"""Thin Python adapter for the experiment-only native rollout JobBatch."""

from __future__ import annotations

import math

import numpy as np


OPPONENT_NAMES = {
    1: "thomas_2945_cpp", 2: "metav4_2965", 3: "replay_clean",
    4: "salemali7_2900",
    5: "fieldcraft_2887",
    6: "soil_current",
}


def ppo_games(arrays: dict, margin_weight: float = 0.1,
              margin_scale: float = 10_000.0,
              materialize_events: bool = True) -> list[dict]:
    """Expose native contiguous arrays through the current PPO list contract.

    Array slices are views; this deliberately does not duplicate the large
    state/resource tensors while the existing optimizer is migrated.
    """
    values = {name: np.asarray(value) for name, value in arrays.items()}
    session_count = len(values["seed"])
    day_sessions = values["day_session_index"]
    offsets = values["day_event_offsets"]
    games = []
    for session in range(session_count):
        days = []
        for day in np.flatnonzero(day_sessions == session):
            if not materialize_events:
                days.append({
                    "step": int(values["day_step"][day]),
                    "native_index": int(day),
                    "native_arrays": values,
                })
                continue
            events = []
            for event in range(int(offsets[day]), int(offsets[day + 1])):
                events.append({
                    "resources": values["event_resources"][event:event + 1],
                    "cell": int(values["event_cell"][event]),
                    "stage": int(values["event_stage"][event]),
                    "previous": int(values["event_previous"][event]),
                    "legal": values["event_legal"][event:event + 1],
                    "legal_mask": int(values["event_legal_mask"][event]),
                    "action": int(values["event_action"][event]),
                    "old_logprob": float(values["old_logprob"][event]),
                    "old_entropy": float(values["old_entropy"][event]),
                })
            days.append({
                "step": int(values["day_step"][day]),
                "state": {
                    "context": values["context"][day:day + 1],
                    "observation": values["observation"][day:day + 1],
                    "observation_length": values["observation_length"][day:day + 1],
                    "token_continuous": values["token_continuous"][day:day + 1],
                    "token_categories": [
                        values["token_categories"][day:day + 1, category]
                        for category in range(values["token_categories"].shape[1])
                    ],
                    "token_count": values["token_count"][day:day + 1],
                },
                "events": events,
            })
        own = float(values["own_cash"][session])
        rival = float(values["rival_cash"][session])
        margin = own - rival
        outcome = float((margin > 0) - (margin < 0))
        opponent = int(values["opponent"][session])
        games.append({
            "status": "PASS",
            "seed": int(values["seed"][session]),
            "seat": int(values["seat"][session]),
            "opponent": OPPONENT_NAMES[opponent],
            "opponent_variant": OPPONENT_NAMES[opponent],
            "route": int(values["route"][session]),
            "policy_seed": int(values["policy_seed"][session]),
            "own_cash": own,
            "opponent_cash": rival,
            "margin": margin,
            "outcome": outcome,
            "reward": outcome + margin_weight * math.tanh(margin / margin_scale),
            "frames": 719,
            "illegal": 0,
            "fallbacks": 0,
            "days": days,
        })
    return games
