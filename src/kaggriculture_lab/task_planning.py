"""Grid search and prioritized multi-worker space-time reservations."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Iterable, Mapping, Sequence


Position = tuple[int, int]
TimedEdge = tuple[Position, Position]

_MOVES: tuple[tuple[str, int, int], ...] = (
    ("NORTH", 0, -1),
    ("SOUTH", 0, 1),
    ("EAST", 1, 0),
    ("WEST", -1, 0),
)


def _inside(position: Position, board_size: int) -> bool:
    return 0 <= position[0] < board_size and 0 <= position[1] < board_size


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def bfs_shortest_path(
    start: Sequence[int],
    goal: Sequence[int],
    *,
    board_size: int = 10,
    blocked: Iterable[Position] = (),
) -> tuple[Position, ...]:
    """Return an exact static shortest path, including start and goal."""

    source = (int(start[0]), int(start[1]))
    target = (int(goal[0]), int(goal[1]))
    if not _inside(source, board_size) or not _inside(target, board_size):
        return ()
    forbidden = set(blocked) - {source, target}
    frontier = [source]
    parents: dict[Position, Position | None] = {source: None}
    cursor = 0
    while cursor < len(frontier):
        current = frontier[cursor]
        cursor += 1
        if current == target:
            break
        for _, dx, dy in _MOVES:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                _inside(neighbor, board_size)
                and neighbor not in forbidden
                and neighbor not in parents
            ):
                parents[neighbor] = current
                frontier.append(neighbor)
    if target not in parents:
        return ()
    reversed_path = [target]
    while parents[reversed_path[-1]] is not None:
        reversed_path.append(parents[reversed_path[-1]])
    return tuple(reversed(reversed_path))


@dataclass
class ReservationTable:
    """Vertex and directed-edge reservations indexed by arrival time."""

    vertices: dict[int, set[Position]] = field(default_factory=dict)
    edges: dict[int, set[TimedEdge]] = field(default_factory=dict)

    def reserve_path(self, path: Sequence[Position], *, linger: int = 1) -> None:
        if not path:
            return
        for time, position in enumerate(path):
            self.vertices.setdefault(time, set()).add(position)
            if time:
                self.edges.setdefault(time, set()).add((path[time - 1], position))
        arrival = len(path) - 1
        for offset in range(1, max(linger, 0) + 1):
            self.vertices.setdefault(arrival + offset, set()).add(path[-1])
            self.edges.setdefault(arrival + offset, set()).add((path[-1], path[-1]))

    def conflicts(self, current: Position, nxt: Position, arrival: int) -> bool:
        if nxt in self.vertices.get(arrival, set()):
            return True
        # Prevent head-on swaps in the same tick.
        return (nxt, current) in self.edges.get(arrival, set())


def reservation_astar(
    start: Sequence[int],
    goal: Sequence[int],
    reservations: ReservationTable,
    *,
    board_size: int = 10,
    horizon: int = 64,
    blocked_at_first_step: Iterable[Position] = (),
) -> tuple[Position, ...]:
    """Find a shortest space-time path with WAIT, vertex, and edge constraints."""

    source = (int(start[0]), int(start[1]))
    target = (int(goal[0]), int(goal[1]))
    if not _inside(source, board_size) or not _inside(target, board_size):
        return ()
    first_blocked = set(blocked_at_first_step) - {source}
    start_state = (source, 0)
    queue: list[tuple[int, int, int, int, Position, int]] = []
    sequence = 0
    heappush(
        queue,
        (_manhattan(source, target), 0, source[1], source[0], source, 0),
    )
    parents: dict[tuple[Position, int], tuple[Position, int] | None] = {
        start_state: None
    }
    costs: dict[tuple[Position, int], int] = {start_state: 0}
    final: tuple[Position, int] | None = None
    while queue:
        _, cost, _, _, current, time = heappop(queue)
        state = (current, time)
        if cost != costs.get(state):
            continue
        if current == target:
            final = state
            break
        if time >= horizon:
            continue
        moves = (*_MOVES, ("WAIT", 0, 0))
        for move_order, (_, dx, dy) in enumerate(moves):
            nxt = (current[0] + dx, current[1] + dy)
            arrival = time + 1
            if not _inside(nxt, board_size):
                continue
            if arrival == 1 and nxt in first_blocked:
                continue
            if reservations.conflicts(current, nxt, arrival):
                continue
            next_state = (nxt, arrival)
            next_cost = cost + 1
            if next_cost >= costs.get(next_state, 1 << 30):
                continue
            costs[next_state] = next_cost
            parents[next_state] = state
            sequence += 1
            priority = next_cost + _manhattan(nxt, target)
            heappush(
                queue,
                (priority, next_cost, move_order, sequence, nxt, arrival),
            )
    if final is None:
        return ()
    reversed_path: list[Position] = []
    cursor: tuple[Position, int] | None = final
    while cursor is not None:
        reversed_path.append(cursor[0])
        cursor = parents[cursor]
    return tuple(reversed(reversed_path))


def plan_prioritized_routes(
    starts: Mapping[int, Position],
    goals: Mapping[int, Position],
    priorities: Mapping[int, tuple[int, ...]],
    *,
    board_size: int = 10,
    horizon: int = 64,
) -> tuple[dict[int, tuple[Position, ...]], ReservationTable]:
    """Plan full routes in urgency order and reserve both vertices and edges."""

    table = ReservationTable()
    paths: dict[int, tuple[Position, ...]] = {}
    order = sorted(starts, key=lambda unit: (*priorities.get(unit, ()), unit))
    remaining = set(order)
    for unit in order:
        remaining.remove(unit)
        start = starts[unit]
        goal = goals.get(unit, start)
        unplanned_starts = {starts[other] for other in remaining}
        path = reservation_astar(
            start,
            goal,
            table,
            board_size=board_size,
            horizon=horizon,
            blocked_at_first_step=unplanned_starts,
        )
        if not path:
            path = (start,)
        paths[unit] = path
        table.reserve_path(path, linger=1)
    return paths, table


def movement_for_path(path: Sequence[Position]) -> list[str]:
    if len(path) < 2 or path[0] == path[1]:
        return ["PASS"]
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    lookup = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
    return [lookup.get((dx, dy), "PASS")]
