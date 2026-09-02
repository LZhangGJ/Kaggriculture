from kaggriculture_lab.route_planner import (
    FarmTask,
    PlanSpec,
    SchedulerConfig,
    WorkerSpec,
    evaluate_one_hire,
    mutate_plan,
    schedule_tasks,
)


def test_scheduler_respects_dependencies_and_worker_travel():
    tasks = [
        FarmTask(
            "build",
            position=(1, 0),
            release_step=0,
            deadline_step=4,
            duration=1,
            value=10,
            mandatory=True,
        ),
        FarmTask(
            "place_cow",
            position=(1, 0),
            release_step=0,
            deadline_step=6,
            duration=1,
            value=20,
            mandatory=True,
            predecessors=("build",),
        ),
    ]
    result = schedule_tasks(
        tasks,
        [WorkerSpec("farmer", (0, 0), role="farmer")],
        config=SchedulerConfig(horizon_end=10, beam_width=16),
    )
    assert result.feasible
    assert [assignment.task_id for assignment in result.assignments] == [
        "build",
        "place_cow",
    ]
    assert result.assignments[0].travel_steps == 1
    assert result.assignments[1].start_step >= result.assignments[0].finish_step


def test_hiring_is_selected_when_one_worker_cannot_cover_two_mandatory_tasks():
    tasks = [
        FarmTask(
            "feed_left",
            (0, 0),
            0,
            2,
            duration=1,
            value=100,
            mandatory=True,
        ),
        FarmTask(
            "feed_right",
            (8, 0),
            0,
            2,
            duration=1,
            value=100,
            mandatory=True,
        ),
    ]
    decision = evaluate_one_hire(
        tasks,
        [WorkerSpec("farmer", (0, 0), role="farmer")],
        hires_today=0,
        hire_step=0,
        spawn=(8, 0),
        config=SchedulerConfig(horizon_end=3, beam_width=32),
    )
    assert not decision.without_hire.feasible
    assert decision.with_hire.feasible
    assert decision.should_hire
    assert decision.hire_cost == 1


def test_plan_mutation_is_local_and_bounded():
    plan = PlanSpec(
        name="8c6s",
        cow_target=8,
        sheep_target=6,
        land_target=75,
        max_hands=4,
        liquidation_step=672,
    )
    candidates = mutate_plan(plan)
    assert candidates[0] == plan
    assert len(candidates) < 20
    assert all(0 <= candidate.cow_target <= 16 for candidate in candidates)
    assert all(25 <= candidate.land_target <= 100 for candidate in candidates)
