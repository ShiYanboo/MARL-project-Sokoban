"""Reward-shaping utilities for symbolic Sokoban states."""

from collections import deque
from functools import lru_cache

import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
WALL = 0
TARGET = 2
BOX_ON_TARGET = 3
BOX = 4
PLAYER = 5


class SokobanRewardShaper:
    """Compute potential and deadlock shaping without changing base rewards."""

    def __init__(
        self,
        distance_weight=0.05,
        pushability_weight=0.02,
        deadlock_penalty=2.0,
        deadlock_penalty_mode="state",
        agent_box_distance_weight=0.005,
        useful_push_weight=0.0,
        unreachable_distance=None,
        box_target_distance_mode="reverse_push",
        agent_box_distance_mode="useful",
    ):
        self.distance_weight = float(distance_weight)
        self.pushability_weight = float(pushability_weight)
        self.deadlock_penalty = float(deadlock_penalty)
        self.deadlock_penalty_mode = deadlock_penalty_mode
        self.agent_box_distance_weight = float(agent_box_distance_weight)
        self.useful_push_weight = float(useful_push_weight)
        self.unreachable_distance = (
            None if unreachable_distance is None else float(unreachable_distance)
        )
        self.box_target_distance_mode = box_target_distance_mode
        self.agent_box_distance_mode = agent_box_distance_mode
        self._push_distance_cache = {}
        if self.deadlock_penalty_mode not in {"state", "increase"}:
            raise ValueError(
                "deadlock_penalty_mode must be either 'state' or 'increase'."
            )
        if self.box_target_distance_mode not in {"reverse_push", "bfs"}:
            raise ValueError(
                "box_target_distance_mode must be either 'reverse_push' or 'bfs'."
            )
        if self.agent_box_distance_mode not in {"useful", "adjacent"}:
            raise ValueError(
                "agent_box_distance_mode must be either 'useful' or 'adjacent'."
            )

    @property
    def enabled(self):
        return any(
            value != 0.0
            for value in (
                self.distance_weight,
                self.pushability_weight,
                self.deadlock_penalty,
                self.agent_box_distance_weight,
                self.useful_push_weight,
            )
        )

    def evaluate_transition(self, before, after):
        """Return total shaping reward and its individual components."""
        before_metrics = self.state_metrics(before)
        after_metrics = self.state_metrics(after)

        distance_reward = self.distance_weight * (
            before_metrics["box_target_distance"]
            - after_metrics["box_target_distance"]
        )
        pushability_change = (
            after_metrics["pushability"] - before_metrics["pushability"]
        )
        pushability_reward = self.pushability_weight * min(0, pushability_change)
        if self.deadlock_penalty_mode == "increase":
            deadlock_count = max(
                0,
                after_metrics["deadlocked_boxes"]
                - before_metrics["deadlocked_boxes"],
            )
        else:
            deadlock_count = after_metrics["deadlocked_boxes"]
        deadlock_reward = -self.deadlock_penalty * deadlock_count
        agent_box_distance_reward = self.agent_box_distance_weight * (
            before_metrics["agent_box_distance"]
            - after_metrics["agent_box_distance"]
        )
        useful_push_distance_delta = self._useful_push_distance_delta(
            before_metrics, after_metrics
        )
        useful_push_applied = int(useful_push_distance_delta > 0)
        useful_push_reward = self.useful_push_weight * useful_push_applied
        total = (
            distance_reward
            + pushability_reward
            + deadlock_reward
            + agent_box_distance_reward
            + useful_push_reward
        )
        return {
            "total": float(total),
            "distance": float(distance_reward),
            "pushability": float(pushability_reward),
            "deadlock": float(deadlock_reward),
            "agent_box_distance": float(agent_box_distance_reward),
            "useful_push": float(useful_push_reward),
            "box_target_distance_before": float(
                before_metrics["box_target_distance"]
            ),
            "box_target_distance_after": float(
                after_metrics["box_target_distance"]
            ),
            "pushability_before": int(before_metrics["pushability"]),
            "pushability_after": int(after_metrics["pushability"]),
            "deadlocked_boxes": int(after_metrics["deadlocked_boxes"]),
            "deadlock_penalty_count": int(deadlock_count),
            "agent_box_distance_before": float(
                before_metrics["agent_box_distance"]
            ),
            "agent_box_distance_after": float(
                after_metrics["agent_box_distance"]
            ),
            "useful_push_applied": useful_push_applied,
            "useful_push_distance_delta": float(useful_push_distance_delta),
        }

    @staticmethod
    def empty_result():
        return {
            "total": 0.0,
            "distance": 0.0,
            "pushability": 0.0,
            "deadlock": 0.0,
            "agent_box_distance": 0.0,
            "useful_push": 0.0,
            "box_target_distance_before": 0.0,
            "box_target_distance_after": 0.0,
            "pushability_before": 0,
            "pushability_after": 0,
            "deadlocked_boxes": 0,
            "deadlock_penalty_count": 0,
            "agent_box_distance_before": 0.0,
            "agent_box_distance_after": 0.0,
            "useful_push_applied": 0,
            "useful_push_distance_delta": 0.0,
        }

    def state_metrics(self, state):
        room_fixed = state["room_fixed"]
        room_state = state["room_state"]
        players = tuple(tuple(position) for position in state["players"])
        boxes = self._positions((room_state == BOX) | (room_state == BOX_ON_TARGET))
        targets = self._positions(room_fixed == TARGET)
        unreachable = self._unreachable_value(room_fixed)
        if self.box_target_distance_mode == "reverse_push":
            push_distance_maps = self._target_push_distance_maps(room_fixed, targets)
            box_target_distance = self._minimum_target_matching_distance_from_maps(
                boxes, push_distance_maps, unreachable
            )
        else:
            push_distance_maps = ()
            box_target_distance = self._minimum_target_matching_distance(
                room_fixed, boxes, targets
            )
        pushabilities = self._box_pushabilities(
            room_fixed, room_state, boxes, players
        )
        deadlocked_boxes = sum(
            pushability == 0 and room_fixed[box] != TARGET
            for box, pushability in zip(boxes, pushabilities)
        )
        if self.agent_box_distance_mode == "useful":
            agent_box_distance = self._agent_useful_position_distance(
                room_fixed,
                room_state,
                boxes,
                players,
                push_distance_maps,
                box_target_distance,
            )
        else:
            agent_box_distance = self._agent_box_distance(
                room_fixed, room_state, boxes, players
            )
        return {
            "boxes": boxes,
            "box_target_distance": box_target_distance,
            "pushability": sum(pushabilities),
            "deadlocked_boxes": deadlocked_boxes,
            "agent_box_distance": agent_box_distance,
        }

    @staticmethod
    def snapshot(env):
        return {
            "room_fixed": env.room_fixed.copy(),
            "room_state": env.room_state.copy(),
            "players": tuple(
                tuple(np.asarray(env.player_positions[index], dtype=int))
                for index in sorted(env.player_positions)
                if index in (0, 1)
            ),
        }

    def _target_push_distance_maps(self, room_fixed, targets):
        cache_key = (room_fixed.shape, room_fixed.tobytes(), targets)
        cached = self._push_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        distance_maps = tuple(
            self._reverse_push_distance_map(room_fixed, target)
            for target in targets
        )
        self._push_distance_cache[cache_key] = distance_maps
        return distance_maps

    def _reverse_push_distance_map(self, room_fixed, target):
        unreachable = self._unreachable_value(room_fixed)
        distances = np.full(room_fixed.shape, unreachable, dtype=np.float32)
        if room_fixed[target] == WALL:
            return distances

        distances[target] = 0.0
        queue = deque([target])
        while queue:
            position = queue.popleft()
            for direction in DIRECTIONS:
                previous = self._subtract(position, direction)
                player_stand = self._subtract(previous, direction)
                if (
                    self._inside(room_fixed.shape, previous)
                    and self._inside(room_fixed.shape, player_stand)
                    and room_fixed[previous] != WALL
                    and room_fixed[player_stand] != WALL
                    and distances[previous] > distances[position] + 1
                ):
                    distances[previous] = distances[position] + 1
                    queue.append(previous)
        return distances

    def _minimum_target_matching_distance_from_maps(
        self, boxes, push_distance_maps, unreachable
    ):
        if not boxes:
            return 0.0
        if not push_distance_maps:
            return float(unreachable * len(boxes))

        costs = tuple(
            tuple(
                float(push_distance_maps[target_index][box])
                for target_index in range(len(push_distance_maps))
            )
            for box in boxes
        )

        @lru_cache(maxsize=None)
        def solve(box_index, used_targets):
            if box_index == len(boxes):
                return 0.0
            best = float(unreachable * (len(boxes) - box_index))
            for target_index in range(len(push_distance_maps)):
                target_bit = 1 << target_index
                if used_targets & target_bit:
                    continue
                best = min(
                    best,
                    costs[box_index][target_index]
                    + solve(box_index + 1, used_targets | target_bit),
                )
            return best

        return solve(0, 0)

    def _minimum_target_matching_distance(self, room_fixed, boxes, targets):
        if not boxes:
            return 0.0
        if not targets:
            return float(self._unreachable_value(room_fixed) * len(boxes))

        unreachable = self._unreachable_value(room_fixed)
        target_distance_maps = tuple(
            self._distance_map(room_fixed, target, blocked=set())
            for target in targets
        )
        costs = tuple(
            tuple(
                float(target_distance_maps[target_index].get(box, unreachable))
                for target_index in range(len(targets))
            )
            for box in boxes
        )

        @lru_cache(maxsize=None)
        def solve(box_index, used_targets):
            if box_index == len(boxes):
                return 0.0
            best = float(unreachable * (len(boxes) - box_index))
            for target_index in range(len(targets)):
                target_bit = 1 << target_index
                if used_targets & target_bit:
                    continue
                best = min(
                    best,
                    costs[box_index][target_index]
                    + solve(box_index + 1, used_targets | target_bit),
                )
            return best

        return solve(0, 0)

    def _useful_push_distance_delta(self, before_metrics, after_metrics):
        before_boxes = set(before_metrics["boxes"])
        after_boxes = set(after_metrics["boxes"])
        if len(before_boxes) != len(after_boxes):
            return 0.0

        removed = before_boxes - after_boxes
        added = after_boxes - before_boxes
        if len(removed) != 1 or len(added) != 1:
            return 0.0

        return (
            before_metrics["box_target_distance"]
            - after_metrics["box_target_distance"]
        )

    def _box_pushabilities(self, room_fixed, room_state, boxes, players):
        reachable = self._reachable_by_any_agent(
            room_fixed, room_state, boxes, players
        )
        occupied = set(boxes) | set(players)
        result = []
        for box in boxes:
            pushability = 0
            for direction in DIRECTIONS:
                front = self._add(box, direction)
                rear = self._subtract(box, direction)
                if (
                    self._inside(room_fixed.shape, front)
                    and self._inside(room_fixed.shape, rear)
                    and room_fixed[front] != WALL
                    and front not in occupied
                    and rear in reachable
                ):
                    pushability += 1
            result.append(pushability)
        return result

    def _reachable_by_any_agent(self, room_fixed, room_state, boxes, players):
        reachable = set()
        box_set = set(boxes)
        player_set = set(players)
        for player in players:
            blocked = box_set | (player_set - {player})
            reachable.update(self._distance_map(room_fixed, player, blocked))
        return reachable

    def _agent_useful_position_distance(
        self,
        room_fixed,
        room_state,
        boxes,
        players,
        push_distance_maps,
        current_box_target_distance,
    ):
        if not boxes:
            return 0.0
        unreachable = self._unreachable_value(room_fixed)
        box_set = set(boxes)
        player_set = set(players)
        distance_maps = []
        for player in players:
            blocked = box_set | (player_set - {player})
            distance_maps.append(self._distance_map(room_fixed, player, blocked))

        total = 0.0
        for box_index, box in enumerate(boxes):
            useful_cells = self._useful_rear_cells(
                room_fixed,
                box_index,
                boxes,
                box_set,
                player_set,
                distance_maps,
                push_distance_maps,
                current_box_target_distance,
            )
            distance = min(
                (
                    distance_map[cell]
                    for distance_map in distance_maps
                    for cell in useful_cells
                    if cell in distance_map
                ),
                default=unreachable,
            )
            total += distance
        return total

    def _agent_box_distance(self, room_fixed, room_state, boxes, players):
        if not boxes:
            return 0.0
        unreachable = self._unreachable_value(room_fixed)
        box_set = set(boxes)
        player_set = set(players)
        distance_maps = []
        for player in players:
            blocked = box_set | (player_set - {player})
            distance_maps.append(self._distance_map(room_fixed, player, blocked))

        total = 0.0
        for box in boxes:
            adjacent_cells = [
                self._add(box, direction)
                for direction in DIRECTIONS
                if self._inside(room_fixed.shape, self._add(box, direction))
                and room_fixed[self._add(box, direction)] != WALL
                and self._add(box, direction) not in box_set
            ]
            distance = min(
                (
                    distance_map[cell] + 1
                    for distance_map in distance_maps
                    for cell in adjacent_cells
                    if cell in distance_map
                ),
                default=unreachable,
            )
            total += distance
        return total

    def _useful_rear_cells(
        self,
        room_fixed,
        box_index,
        boxes,
        box_set,
        player_set,
        player_distance_maps,
        push_distance_maps,
        current_box_target_distance,
    ):
        useful_cells = []
        box = boxes[box_index]
        for direction in DIRECTIONS:
            front = self._add(box, direction)
            rear = self._subtract(box, direction)
            if not self._is_current_push_candidate(
                room_fixed,
                front,
                rear,
                box_set,
                player_set,
                player_distance_maps,
            ):
                continue

            next_boxes = list(boxes)
            next_boxes[box_index] = front
            next_distance = self._minimum_target_matching_distance_from_maps(
                tuple(next_boxes), push_distance_maps, self._unreachable_value(room_fixed)
            )
            if next_distance < current_box_target_distance:
                useful_cells.append(rear)
        return useful_cells

    def _is_current_push_candidate(
        self,
        room_fixed,
        front,
        rear,
        box_set,
        player_set,
        player_distance_maps,
    ):
        if not (
            self._inside(room_fixed.shape, front)
            and self._inside(room_fixed.shape, rear)
            and room_fixed[front] != WALL
            and room_fixed[rear] != WALL
        ):
            return False
        if front in box_set or front in player_set or rear in box_set:
            return False
        return any(rear in distance_map for distance_map in player_distance_maps)

    @staticmethod
    def _distance_map(room_fixed, start, blocked):
        if room_fixed[start] == WALL:
            return {}
        blocked = set(blocked)
        blocked.discard(start)
        distances = {start: 0}
        queue = deque([start])
        while queue:
            position = queue.popleft()
            for direction in DIRECTIONS:
                neighbor = SokobanRewardShaper._add(position, direction)
                if (
                    SokobanRewardShaper._inside(room_fixed.shape, neighbor)
                    and room_fixed[neighbor] != WALL
                    and neighbor not in blocked
                    and neighbor not in distances
                ):
                    distances[neighbor] = distances[position] + 1
                    queue.append(neighbor)
        return distances

    @staticmethod
    def _positions(mask):
        return tuple(tuple(position) for position in np.argwhere(mask))

    @staticmethod
    def _inside(shape, position):
        return 0 <= position[0] < shape[0] and 0 <= position[1] < shape[1]

    @staticmethod
    def _add(position, direction):
        return position[0] + direction[0], position[1] + direction[1]

    @staticmethod
    def _subtract(position, direction):
        return position[0] - direction[0], position[1] - direction[1]

    def _unreachable_value(self, room_fixed):
        if self.unreachable_distance is not None:
            return self.unreachable_distance
        return float(room_fixed.size)
