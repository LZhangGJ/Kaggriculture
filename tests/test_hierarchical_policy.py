from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.decision_schema import (
    CANDIDATE_FEATURES,
    DecisionCandidate,
    TaskType,
)
from kaggriculture_lab.candidate_proposal import (
    PROPOSAL_AUXILIARIES,
    PROPOSAL_TASK_TYPES,
    expert_cards_from_soft_intents,
    sample_proposal_cards,
    soft_assignment_targets,
)
from kaggriculture_lab.board_policy import encode_board_batch
from kaggriculture_lab.gpu_policy import ITEMS, MARKET_INDEX, MAX_UNITS
from kaggriculture_lab.hierarchical_bc import (
    ClosureReason,
    IntentLabel,
    behavior_targets,
    closure_diagnostics,
    daily_budget_soft_targets,
    market_sequence_targets,
    match_intents_with_diagnostics,
)
from kaggriculture_lab.hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    _as_tensors,
    _decode_market,
    _select_modes,
    encode_hierarchical_batch,
    hierarchical_policy_batch,
    prepare_hierarchical_batch,
    select_task_matching,
)
from kaggriculture_lab.hierarchical_rl import (
    evaluate_hierarchical_replay,
    hierarchical_actor_critic_batch,
)
from kaggriculture_lab.hierarchical_schema import (
    BUDGET_CATEGORIES,
    MAX_MARKET_ORDERS,
    MAX_TASK_CARDS,
    PAIR_FEATURES,
    STRATEGY_MODES,
    BudgetCategory,
    BudgetPlan,
    CandidateOrigin,
    HierarchicalMemory,
    MarketBudget,
    PlannedMarketBudget,
    TaskCard,
    TaskPhase,
    TaskTerminationReason,
    compile_hierarchical_action,
    estimate_task_eta,
    generate_task_cards,
    task_resource_plan,
)
from kaggriculture_lab.hierarchical_trace import (
    IntentPhase,
    SoftIntentDistribution,
    SoftIntentHypothesis,
    infer_episode_soft_intents,
)


def _small_model() -> HierarchicalKaggriculturePolicy:
    return HierarchicalKaggriculturePolicy(
        hidden_size=64,
        board_width=16,
        d_model=64,
        transformer_layers=1,
        transformer_heads=4,
    )


def test_task_cards_are_owner_independent_and_shape_stable() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=41)[0]
    encoded = generate_task_cards(observation)

    assert 0 < len(encoded.tasks) <= MAX_TASK_CARDS
    assert encoded.tasks[0].task_type == TaskType.IDLE_OR_PASS
    assert all(card.candidate.owner_unit == -1 for card in encoded.tasks)
    assert encoded.features.shape == (MAX_TASK_CARDS, CANDIDATE_FEATURES)
    assert encoded.pair_features.shape == (
        MAX_UNITS,
        MAX_TASK_CARDS,
        PAIR_FEATURES,
    )
    assert encoded.pair_mask.shape == (MAX_UNITS, MAX_TASK_CARDS)
    assert encoded.eta_steps.shape == (MAX_UNITS, MAX_TASK_CARDS)
    assert encoded.capacity[0] == MAX_UNITS
    assert np.isfinite(encoded.features).all()
    assert np.isfinite(encoded.pair_features).all()


def test_network_shapes_gradients_and_legal_environment_step() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=42))
    encoded = encode_hierarchical_batch(observations)
    tensors = _as_tensors(encoded, "cpu")
    model = _small_model()
    modes = torch.zeros(2, dtype=torch.long)
    history = torch.zeros((2, MAX_MARKET_ORDERS), dtype=torch.long)

    assignment, market, mode_logits, switch_logits, values = model(
        *tensors[:12], modes, history, tensors[14]
    )
    assert assignment.shape == (2, MAX_UNITS, MAX_TASK_CARDS)
    assert market.shape == (2, MAX_MARKET_ORDERS, len(MARKET_INDEX))
    assert mode_logits.shape == (2, len(STRATEGY_MODES))
    assert switch_logits.shape == (2, 2)
    assert values.shape == (2,)
    loss = (
        assignment[tensors[12]].square().mean()
        + market.square().mean()
        + mode_logits.square().mean()
        + switch_logits.square().mean()
        + values.square().mean()
    )
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    policy = hierarchical_policy_batch(
        model.eval(), observations, "cpu", deterministic=True
    )
    assert all(len(action["market"]) <= MAX_MARKET_ORDERS for action in policy.actions)
    result = env.step(policy.actions)
    assert result.step == 1
    assert all(status != "ERROR" for status in result.statuses)
    assert all(memory.internal_trace_history for memory in policy.memories)
    assert policy.memories[0].internal_trace_history[-1]["source"] == "internal"


def test_task_scorer_is_permutation_equivariant() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=43))
    batch = encode_hierarchical_batch(observations)
    tensors = list(_as_tensors(batch, "cpu"))
    model = _small_model().eval()
    modes = torch.zeros(2, dtype=torch.long)
    history = torch.zeros((2, MAX_MARKET_ORDERS), dtype=torch.long)

    with torch.no_grad():
        original, _, _, _, _ = model(
            *tensors[:12], modes, history, tensors[14]
        )
        permutation = torch.randperm(MAX_TASK_CARDS)
        permuted = tensors[:4] + [value[:, permutation] for value in tensors[4:11]]
        permuted.append(tensors[11][:, :, permutation])
        reordered, _, _, _, _ = model(
            *permuted, modes, history, tensors[14]
        )

    torch.testing.assert_close(
        reordered, original[:, :, permutation], atol=1e-5, rtol=1e-5
    )


def test_matching_respects_capacity_and_assigns_each_worker_once() -> None:
    logits = torch.tensor(
        [[[9.0, 8.0, 1.0], [9.0, 7.0, 1.0], [9.0, 6.0, 1.0]]]
    )
    mask = torch.ones_like(logits, dtype=torch.bool)
    capacities = torch.tensor([[1, 1, 3]])
    selected, _, _ = select_task_matching(
        logits, mask, capacities, deterministic=True
    )
    choices = selected[0].tolist()
    assert len(choices) == 3
    assert choices.count(0) <= 1
    assert choices.count(1) <= 1


def test_market_budget_and_full_ten_order_behavior_target() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=44)[0]
    budget = MarketBudget.from_observation(observation)
    hire = MARKET_INDEX[("HIRE", None, 1)]
    before = budget.money
    assert budget.legal_mask()[hire]
    budget.apply(hire)
    assert budget.hires == 1
    assert budget.money < before

    action = {
        "market": [["BUY_SEED", "WHEAT", 1] for _ in range(MAX_MARKET_ORDERS)]
    }
    targets, active = market_sequence_targets(action)
    assert active.all()
    assert (targets != MARKET_INDEX[("NONE", None, 0)]).all()

    stop_targets, stop_active = market_sequence_targets({"market": []})
    assert stop_active[0]
    assert stop_targets[0] == MARKET_INDEX[("NONE", None, 0)]


def test_daily_budget_soft_targets_use_spend_as_category_lower_bound() -> None:
    environment = FastKaggricultureEnv(
        configuration={"episodeSteps": 4}, copy_observations=True
    )
    observations = list(environment.reset(seed=445))
    left_observations = [observations[0]]
    left_actions = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
    ]
    passive = {"farmer": ["PASS"], "hands": [], "market": []}
    result = environment.step([left_actions[0], passive])
    while not environment.done:
        left_observations.append(result.observations[0])
        left_actions.append(passive)
        result = environment.step([passive, passive])

    targets = daily_budget_soft_targets(
        left_observations,
        left_actions,
        final_observation=result.observations[0],
    )
    crop = int(BudgetCategory.CROP)
    assert len(targets) == len(left_observations)
    assert targets[0].total_spend == pytest.approx(10.0)
    assert targets[0].category_fractions[crop] == pytest.approx(1.0)
    assert targets[0].category_floor_fractions[crop] > 0.0
    assert targets[0].confidence == 1.0
    assert all(target.day == targets[0].day for target in targets)


def test_planned_market_budget_enforces_category_floor_and_bounded_emergency() -> None:
    observation = FastKaggricultureEnv(
        configuration={"episodeSteps": 4}
    ).reset(seed=446)[0]
    money = MarketBudget.from_observation(observation).money
    plan = BudgetPlan.from_fractions(
        day=0,
        money=money,
        reserve_fraction=0.99,
        category_fractions=[1.0, 0.0, 0.0, 0.0, 0.0],
        emergency_fraction=0.05,
    )
    assert len(plan.category_limits) == len(BUDGET_CATEGORIES)
    budget = PlannedMarketBudget.from_observation(observation, plan)
    seed = MARKET_INDEX[("BUY_SEED", "WHEAT", 1)]
    assert not budget.legal_mask()[seed]
    assert budget.legal_mask(emergency=True)[seed]
    budget.apply(seed, emergency=True)
    assert budget.spent_by_category[int(BudgetCategory.CROP)] == pytest.approx(10.0)
    assert budget.emergency_spent > 0.0


def test_budget_head_outputs_a_normalized_explicit_plan() -> None:
    observations = list(
        FastKaggricultureEnv(configuration={"episodeSteps": 4}).reset(seed=447)
    )
    model = _small_model().eval()
    encoded = encode_hierarchical_batch(observations)
    tensors = _as_tensors(encoded, "cpu")
    context = model.encode_state(*tensors[:4])
    output = model.predict_budget(context, torch.zeros(2, dtype=torch.long))

    assert output.reserve_fraction.shape == (2,)
    assert output.category_fractions.shape == (2, len(BUDGET_CATEGORIES))
    assert output.emergency_fraction.shape == (2,)
    torch.testing.assert_close(
        output.category_fractions.sum(dim=-1), torch.ones(2)
    )
    assert bool(((0.0 <= output.reserve_fraction) & (output.reserve_fraction <= 1.0)).all())
    assert bool(((0.0 <= output.emergency_fraction) & (output.emergency_fraction <= 0.25)).all())
    (
        output.reserve_fraction.mean()
        + output.category_fractions[:, int(BudgetCategory.CROP)].mean()
        + output.emergency_fraction.mean()
    ).backward()
    assert model.budget_category_head.weight.grad is not None


def test_candidate_closure_reports_exact_and_per_task_missing_reasons() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=440)[0]
    encoded = generate_task_cards(observation)
    idle = IntentLabel(TaskType.IDLE_OR_PASS)
    missing_market_task = IntentLabel(TaskType.HIRE_WORKER)

    _, active, closed, task_ids, reasons = match_intents_with_diagnostics(
        encoded, [idle, missing_market_task]
    )
    assert active[:2].all()
    assert closed[0]
    assert reasons[0] == int(ClosureReason.EXACT)
    assert not closed[1]
    assert reasons[1] == int(ClosureReason.TASK_TYPE_MISSING)
    assert task_ids[1] == int(TaskType.HIRE_WORKER)

    targets = behavior_targets(
        [encoded], [[idle, missing_market_task]], [{"market": []}]
    )
    report = closure_diagnostics(targets)
    assert report["intents"] == 2
    assert report["closed"] == 1
    assert report["tasks"]["HIRE_WORKER"]["reasons"]["TASK_TYPE_MISSING"] == 1


def test_executor_records_structured_invalid_selection_failures() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=441)[0]
    encoded = generate_task_cards(observation)
    memory = HierarchicalMemory()

    action = compile_hierarchical_action(
        observation,
        encoded,
        assignment=[-1] * MAX_UNITS,
        market_indices=[-1],
        memory=memory,
    )
    diagnostic = memory.executor_diagnostics()
    assert action["farmer"] == ["PASS"]
    assert diagnostic["by_reason"]["INVALID_TASK_INDEX"] >= 1
    assert diagnostic["by_reason"]["INVALID_MARKET_INDEX"] == 1
    assert diagnostic["recent_failures"][0]["step"] == 0
    assert diagnostic["recent_failures"][0]["detail"]
    assert (
        diagnostic["task_state_machine"]["current"][0]["phase"]
        == TaskPhase.IDLE.value
    )
    memory.reset()
    assert memory.executor_diagnostics()["failure_count"] == 0


def test_task_state_machine_records_phase_and_eta_transitions() -> None:
    memory = HierarchicalMemory()
    observation = {"step": 7, "day": 0}
    candidate = DecisionCandidate(
        TaskType.CROP_PRODUCTION,
        owner_unit=0,
        target_x=2,
        target_y=3,
    )
    memory.transition_task(
        0,
        TaskPhase.NAVIGATE,
        observation=observation,
        candidate=candidate,
        eta_steps=6,
        route=((0, 0), (1, 0)),
    )
    diagnostic = memory.executor_diagnostics()["task_state_machine"]
    assert diagnostic["transition_count"] == 1
    assert diagnostic["current"][0]["phase"] == "NAVIGATE"
    assert diagnostic["current"][0]["eta_steps"] == 6
    assert diagnostic["recent_transitions"][0]["task_type"] == "CROP_PRODUCTION"


def test_persistent_task_chain_advances_stage_across_day_boundary() -> None:
    memory = HierarchicalMemory()
    crop = DecisionCandidate(
        TaskType.CROP_PRODUCTION,
        owner_unit=0,
        target_x=2,
        target_y=3,
        item_id=ITEMS.index("WHEAT"),
    )
    water = DecisionCandidate(
        TaskType.WATER_CROP,
        owner_unit=0,
        target_x=2,
        target_y=3,
        item_id=ITEMS.index("WHEAT"),
    )
    memory.transition_task(
        0,
        TaskPhase.EXECUTE,
        observation={"step": 23, "day": 0},
        candidate=crop,
        eta_steps=1,
    )
    chain_id = memory.task_states[0].chain_id
    memory.transition_task(
        0,
        TaskPhase.COMPLETED,
        observation={"step": 24, "day": 1},
        termination_reason=TaskTerminationReason.STAGE_ADVANCED,
    )
    memory.transition_task(
        0,
        TaskPhase.ASSIGNED,
        observation={"step": 24, "day": 1},
        candidate=water,
        eta_steps=2,
        continue_chain=True,
    )
    memory.transition_task(
        0,
        TaskPhase.NAVIGATE,
        observation={"step": 24, "day": 1},
        candidate=water,
        eta_steps=2,
    )
    memory.record_internal_trace(
        {"step": 24, "day": 1},
        {"farmer": ["NORTH"], "hands": [], "market": []},
    )

    state = memory.task_states[0]
    hypothesis = memory.internal_trace_history[-1]["workers"][0]["hypotheses"][0]
    assert state.chain_id == chain_id
    assert state.started_step == 23
    assert state.stage_index == 1
    assert state.completed_stages == 1
    assert [task.task_type for task in state.chain_tasks] == [
        TaskType.CROP_PRODUCTION,
        TaskType.WATER_CROP,
    ]
    assert hypothesis["chain_id"] == chain_id
    assert hypothesis["stage_index"] == 1
    assert [stage["task_type"] for stage in hypothesis["chain"]] == [
        "CROP_PRODUCTION",
        "WATER_CROP",
    ]
    completed = next(
        row
        for row in memory.executor_diagnostics()["task_state_machine"]["recent_transitions"]
        if row["phase"] == TaskPhase.COMPLETED.value
    )
    assert completed["termination_reason"] == "STAGE_ADVANCED"


def test_real_eta_includes_shed_pickup_route_and_execution() -> None:
    farm = {
        "money": 1000,
        "farmer": [4, 4],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    observation = {
        "step": 0,
        "day": 0,
        "hour": 0,
        "player": 0,
        "farms": [farm, farm],
        "private": {
            "shed": {"WHEAT": 1},
            "inventories": [{}],
            "seeds": {},
        },
    }
    candidate = DecisionCandidate(
        TaskType.ANIMAL_FEED,
        owner_unit=0,
        target_x=0,
        target_y=0,
        item_id=ITEMS.index("WHEAT"),
    )
    eta = estimate_task_eta(observation, farm, 0, candidate)
    assert eta.reachable
    assert eta.immediate_goal == (4, 4)
    assert eta.movement_steps == 8
    assert eta.interaction_steps == 2
    assert eta.total_steps == 10


def test_strategy_mode_is_persistent_in_memory() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=45))
    memories = [HierarchicalMemory() for _ in observations]
    model = _small_model().eval()
    first = hierarchical_policy_batch(
        model, observations, "cpu", memories=memories, deterministic=True
    )
    chosen = first.strategy_modes.tolist()
    assert chosen == [memory.strategy_mode for memory in memories]
    plans = [memory.budget_plan for memory in memories]
    assert first.budget_action_active.all()

    second = hierarchical_policy_batch(
        model, observations, "cpu", memories=memories, deterministic=False
    )
    assert second.strategy_modes.tolist() == chosen
    assert not second.budget_action_active.any()
    assert torch.equal(second.budget_log_probs, torch.zeros_like(second.budget_log_probs))
    assert [memory.budget_plan for memory in memories] == plans


def test_strategy_mode_can_only_keep_or_switch_at_a_new_day() -> None:
    memory = HierarchicalMemory()
    mode_logits = torch.tensor([[0.0, 1.0, 10.0, 9.0, 0.0, 0.0, 0.0, 0.0]])
    switch = torch.tensor([[0.0, 10.0]])
    initial, decision, _, _ = _select_modes(
        mode_logits, switch, [{"step": 0, "day": 0}], [memory], deterministic=True
    )
    assert initial.item() == 2
    assert decision.item() == -1

    same_day, decision, _, _ = _select_modes(
        mode_logits, switch, [{"step": 5, "day": 0}], [memory], deterministic=True
    )
    assert same_day.item() == 2
    assert decision.item() == -1

    changed, decision, _, _ = _select_modes(
        mode_logits, switch, [{"step": 24, "day": 1}], [memory], deterministic=True
    )
    assert changed.item() == 3
    assert decision.item() == 1
    assert memory.mode_switches == 1

    kept, decision, _, _ = _select_modes(
        mode_logits,
        torch.tensor([[10.0, 0.0]]),
        [{"step": 48, "day": 2}],
        [memory],
        deterministic=True,
    )
    assert kept.item() == 3
    assert decision.item() == 0
    assert memory.mode_history[-1]["decision"] == "KEEP"


def test_sampled_hierarchical_policy_backpropagates() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=46))
    memories = [HierarchicalMemory() for _ in observations]
    model = _small_model().train()

    policy = hierarchical_actor_critic_batch(
        model, observations, "cpu", memories
    )
    loss = -(policy.log_probs.mean()) + policy.values.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )
    assert model.proposal_task_head.weight.grad is not None
    assert model.proposal_task_head.weight.grad.abs().sum() > 0


def test_ppo_replay_recomputes_exact_sampled_log_probability() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=460))
    memories = [HierarchicalMemory() for _ in observations]
    model = _small_model().eval()

    policy = hierarchical_actor_critic_batch(model, observations, "cpu", memories)
    assert policy.replay is not None
    replayed = evaluate_hierarchical_replay(model, policy.replay, "cpu")
    torch.testing.assert_close(
        replayed.log_probs, policy.log_probs, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(replayed.values, policy.values)
    assert policy.budget_action_active.all()
    torch.testing.assert_close(
        replayed.budget_log_probs,
        policy.budget_log_probs,
        atol=1e-5,
        rtol=1e-5,
    )
    loss = -replayed.log_probs.mean() + replayed.values.square().mean()
    loss.backward()
    assert model.proposal_task_head.weight.grad is not None
    assert model.proposal_task_head.weight.grad.abs().sum() > 0
    assert model.budget_category_head.weight.grad is not None
    assert model.budget_category_head.weight.grad.abs().sum() > 0


def test_inverse_planning_keeps_top_m_soft_intents() -> None:
    farm0 = {
        "money": 1000,
        "farmer": [0, 0],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    farm1 = {**farm0, "farmer": [9, 9]}
    observations = [
        {"step": 0, "day": 0, "player": 0, "farms": [farm0, farm1]},
        {
            "step": 1,
            "day": 0,
            "player": 0,
            "farms": [{**farm0, "farmer": [1, 0]}, farm1],
        },
        {
            "step": 2,
            "day": 0,
            "player": 0,
            "farms": [{**farm0, "farmer": [1, 0]}, farm1],
        },
    ]
    actions = [
        {"farmer": ["EAST"], "hands": [], "market": []},
        {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
        {"farmer": ["WATER"], "hands": [], "market": []},
    ]
    labels = infer_episode_soft_intents(observations, actions, horizon=4, top_m=3)

    first = labels[0][0]
    assert 1 <= len(first.hypotheses) <= 3
    assert first.best.task_type == TaskType.CROP_PRODUCTION
    assert first.best.target_x == 1 and first.best.target_y == 0
    assert first.hypotheses[0].phase == IntentPhase.NAVIGATE
    assert first.hypotheses[0].chain_operations == ("PLANT", "WATER")
    assert (
        first.hypotheses[0].chain_id
        == labels[1][0].hypotheses[0].chain_id
    )
    assert all(
        hypothesis.intent.task_type != TaskType.WATER_CROP
        for hypothesis in first.hypotheses
    )
    assert sum(value.probability for value in first.hypotheses) == pytest.approx(1.0)
    assert 0.0 <= first.confidence <= 1.0


def test_expert_injection_closes_soft_label_and_dropout_removes_it() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=47)[0]
    distribution = SoftIntentDistribution(
        unit=0,
        hypotheses=(
            SoftIntentHypothesis(
                IntentLabel(TaskType.CROP_PRODUCTION, 0, 0, ITEMS.index("WHEAT")),
                probability=1.0,
                eta_steps=1,
            ),
        ),
        confidence=1.0,
    )
    cards = expert_cards_from_soft_intents([distribution], rng=np.random.default_rng(0))
    encoded = generate_task_cards(observation, extra_cards=cards)
    targets = soft_assignment_targets(encoded, [distribution])

    assert any(card.origin == CandidateOrigin.EXPERT for card in encoded.tasks)
    assert targets.closed_mass[0] == pytest.approx(1.0)
    assert not expert_cards_from_soft_intents(
        [distribution], dropout=1.0, rng=np.random.default_rng(0)
    )


def test_cnn_proposal_shapes_gradients_and_two_stage_union() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=48))
    model = _small_model().train()
    with torch.no_grad():
        model.proposal_task_head.weight.zero_()
        model.proposal_task_head.bias.fill_(-10.0)
        model.proposal_task_head.bias[
            PROPOSAL_TASK_TYPES.index(TaskType.CROP_PRODUCTION)
        ] = 10.0
        model.proposal_item_head.weight.zero_()
        model.proposal_item_head.bias.fill_(-10.0)
        model.proposal_item_head.bias[ITEMS.index("WHEAT") + 1] = 10.0
    batch, _, _, proposal, selection = prepare_hierarchical_batch(
        model, observations, "cpu"
    )

    assert proposal.task_logits.shape == (
        2,
        len(PROPOSAL_TASK_TYPES),
        10,
        10,
    )
    assert proposal.item_logits.shape == (2, len(ITEMS) + 1, 10, 10)
    assert proposal.auxiliary_predictions.shape == (
        2,
        len(PROPOSAL_AUXILIARIES),
        10,
        10,
    )
    assert any(
        card.origin == CandidateOrigin.LEARNED
        for encoded in batch.encoded_sets
        for card in encoded.tasks
    )
    proposal.task_logits.square().mean().backward()
    assert model.proposal_task_head.weight.grad is not None
    assert selection.log_probs.shape == (2,)


def test_tensorized_proposal_mask_and_strict_filter_reject_locked_tiles() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observation = env.reset(seed=481)[0]
    boards, _, _, _ = encode_board_batch([observation])
    task_logits = torch.zeros((1, len(PROPOSAL_TASK_TYPES), 10, 10))
    task_logits[:, :, 9, 9] = 100.0
    item_logits = torch.zeros((1, len(ITEMS) + 1, 10, 10))
    auxiliaries = torch.zeros((1, len(PROPOSAL_AUXILIARIES), 10, 10))
    selection = sample_proposal_cards(
        task_logits,
        item_logits,
        auxiliaries,
        [observation],
        top_k=4,
        deterministic=True,
        locked_mask=torch.as_tensor(boards[:, 0, 1]),
    )
    assert all(int(index) % 100 != 99 for index in selection.drawn_indices[0])

    locked_card = TaskCard(
        DecisionCandidate(
            TaskType.CROP_PRODUCTION,
            target_x=9,
            target_y=9,
            item_id=ITEMS.index("WHEAT"),
        ),
        origin=CandidateOrigin.LEARNED,
    )
    encoded = generate_task_cards(observation, extra_cards=[locked_card])
    assert not any(
        card.origin == CandidateOrigin.LEARNED and card.target == (9, 9)
        for card in encoded.tasks
    )


def test_resource_deficit_creates_required_market_order_and_strict_mask() -> None:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "fed_today": False,
    }
    farm = {"money": 1000, "farmer": [0, 0], "hands": [], "tiles": tiles}
    observation = {
        "step": 0,
        "day": 0,
        "player": 0,
        "farms": [farm, farm],
        "private": {
            "shed": {},
            "inventories": [{}],
            "seeds": {},
        },
        "market": {
            "inventory": {item: 10_000 for item in ITEMS},
            "prices": {},
        },
    }
    feed = DecisionCandidate(
        TaskType.ANIMAL_FEED,
        target_x=0,
        target_y=0,
        item_id=ITEMS.index("WHEAT"),
    )
    plan = task_resource_plan(observation, farm, 0, feed)
    required = MARKET_INDEX[("BUY_PRODUCT", "WHEAT", 1)]
    assert plan.feasible and not plan.available
    assert plan.required_market_index == required

    encoded = generate_task_cards(
        observation,
        extra_cards=(
            TaskCard(feed, origin=CandidateOrigin.LEARNED),
            TaskCard(
                DecisionCandidate(
                    TaskType.ANIMAL_FEED,
                    target_x=1,
                    target_y=1,
                    item_id=ITEMS.index("WHEAT"),
                ),
                origin=CandidateOrigin.LEARNED,
            ),
        ),
    )
    valid_index = next(
        index
        for index, card in enumerate(encoded.tasks)
        if card.task_type == TaskType.ANIMAL_FEED and card.target == (0, 0)
    )
    assert encoded.pair_mask[0, valid_index]
    assert encoded.required_market_indices[0, valid_index] == required
    assert not any(
        card.task_type == TaskType.ANIMAL_FEED and card.target == (1, 1)
        for card in encoded.tasks
    )


def test_required_resource_order_precedes_learned_market_actions() -> None:
    env = FastKaggricultureEnv(configuration={"episodeSteps": 4})
    observations = list(env.reset(seed=50))
    model = _small_model().eval()
    encoded = encode_hierarchical_batch(observations)
    tensors = _as_tensors(encoded, "cpu")
    context = model.encode_state(*tensors[:4])
    modes = torch.zeros(2, dtype=torch.long)
    required = MARKET_INDEX[("BUY_PRODUCT", "WHEAT", 1)]
    required_orders = torch.tensor([[required], [required]])
    plan_context = torch.zeros((2, model.d_model))
    selected, _, _ = _decode_market(
        model,
        context,
        modes,
        tensors[14],
        observations,
        plan_context=plan_context,
        required_orders=required_orders,
        deterministic=True,
    )
    assert (selected[:, 0] == required).all()
    BudgetCategory,
    BudgetPlan,
