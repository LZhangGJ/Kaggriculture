from __future__ import annotations

from kaggriculture_lab.task_planning import (
    ReservationTable,
    bfs_shortest_path,
    movement_for_path,
    plan_prioritized_routes,
    reservation_astar,
)


def test_bfs_returns_exact_grid_distance_and_respects_blockers() -> None:
    path = bfs_shortest_path((0, 0), (2, 0), blocked={(1, 0)}, board_size=4)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 0)
    assert len(path) - 1 == 4


def test_reservation_astar_avoids_vertex_and_head_on_edge_conflicts() -> None:
    table = ReservationTable()
    table.reserve_path(((0, 0), (1, 0), (2, 0)))
    path = reservation_astar((2, 0), (0, 0), table, board_size=3, horizon=8)
    assert path
    for time in range(1, len(path)):
        assert path[time] not in table.vertices.get(time, set())
        assert (path[time], path[time - 1]) not in table.edges.get(time, set())


def test_prioritized_multi_worker_routes_have_no_timed_conflicts() -> None:
    paths, _ = plan_prioritized_routes(
        {0: (0, 0), 1: (2, 0)},
        {0: (2, 0), 1: (0, 0)},
        {0: (0,), 1: (1,)},
        board_size=3,
        horizon=8,
    )
    maximum = max(map(len, paths.values()))
    padded = {
        unit: path + (path[-1],) * (maximum - len(path))
        for unit, path in paths.items()
    }
    for time in range(1, maximum):
        assert padded[0][time] != padded[1][time]
        assert not (
            padded[0][time - 1] == padded[1][time]
            and padded[1][time - 1] == padded[0][time]
        )
    assert movement_for_path(paths[0]) in (["EAST"], ["SOUTH"])
