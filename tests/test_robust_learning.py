from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.board_policy import encode_board_batch
from kaggriculture_lab.expert_sampling import robust_cluster_indices
from kaggriculture_lab.hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    opponent_conditioned_mode_logits,
    uncertainty_reserve_floors,
)
from kaggriculture_lab.hierarchical_schema import (
    MODE_INDEX,
    BudgetPlan,
)
from kaggriculture_lab.league import (
    assemble_seat_actions,
    league_mixture_from_payoff,
)
from kaggriculture_lab.opponent_model import OPPONENT_FEATURES, OPPONENT_HISTORY_LAGS


def test_opponent_encoder_conditions_shared_state_and_trains_belief() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=901))
    boards, globals_, units, mask = encode_board_batch(observations)
    history = torch.zeros((2, len(OPPONENT_HISTORY_LAGS), OPPONENT_FEATURES))
    history[1, -1] = 1.0
    history_mask = torch.ones((2, len(OPPONENT_HISTORY_LAGS)), dtype=torch.bool)
    model = HierarchicalKaggriculturePolicy(
        hidden_size=64,
        board_width=16,
        d_model=32,
        transformer_layers=1,
        transformer_heads=4,
        proposal_top_k=4,
    )
    context = model.encode_state(
        torch.as_tensor(boards),
        torch.as_tensor(globals_),
        torch.as_tensor(units),
        torch.as_tensor(mask),
        history,
        history_mask,
    )
    assert context.opponent_belief_logits.shape == (2, 8)
    assert context.horizon_values.shape == (2, 3)
    assert not torch.allclose(context.state[0], context.state[1])
    loss = torch.nn.functional.cross_entropy(
        context.opponent_belief_logits, torch.tensor([0, 1])
    )
    loss.backward()
    assert model.opponent_belief_head.weight.grad is not None


def test_uncertainty_prior_and_cash_floor_are_explicit_but_action_stays_raw() -> None:
    logits = torch.zeros((2, len(MODE_INDEX)))
    uncertainty = torch.tensor([0.0, 1.0])
    observations = [{"step": 0}, {"step": 200}]
    adjusted = opponent_conditioned_mode_logits(logits, uncertainty, observations)
    assert adjusted[0, MODE_INDEX["ROBUST_OPENING"]] == pytest.approx(0.5)
    assert adjusted[1, MODE_INDEX["ROBUST_OPENING"]] == pytest.approx(0.0)
    floors = uncertainty_reserve_floors(uncertainty, observations)
    assert floors.tolist() == pytest.approx([0.10, 0.20])
    plan = BudgetPlan.from_fractions(
        day=0,
        money=1000,
        reserve_fraction=0.05,
        category_fractions=[1, 1, 1, 1, 1],
        minimum_reserve_fraction=0.40,
    )
    assert plan.features[0] == pytest.approx(0.05)
    assert plan.effective_reserve_fraction == pytest.approx(0.40)
    assert plan.cash_floor == pytest.approx(400.0)


def test_cvar_sampler_and_psro_league_mixture_focus_hard_opponents() -> None:
    clusters = [index % 2 for index in range(40)]
    opponent_clusters = [0] * 20 + [1] * 20
    outcomes = [1.0] * 20 + [-1.0] * 20
    sampled = robust_cluster_indices(
        clusters,
        outcomes,
        np.random.default_rng(3),
        robust_cluster_ids=opponent_clusters,
        size=200,
        worst_fraction=1.0,
        cvar_fraction=0.25,
    )
    assert np.mean(np.asarray(opponent_clusters)[sampled] == 1) > 0.70
    mixture = league_mixture_from_payoff(
        {"psro_mixture": [0.8, 0.2], "robust_opponent_weights": [0.2, 0.8]},
        2,
    )
    assert mixture.tolist() == pytest.approx([0.5, 0.5])
    paired = assemble_seat_actions([{"id": "L"}], [{"id": "O"}], 1)
    assert paired == [[{"id": "O"}, {"id": "L"}]]
