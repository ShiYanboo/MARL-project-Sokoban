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
        agent_box_distance_weight=0.005,
    ):
        self.distance_weight = float(distance_weight)
        self.pushability_weight = float(pushability_weight)
        self.deadlock_penalty = float(deadlock_penalty)
        self.agent_box_distance_weight = float(agent_box_distance_weight)

    @property
    def enabled(self):
        return any(
            value != 0.0
            for value in (
                self.distance_weight,
                self.pushability_weight,
                self.deadlock_penalty,
                self.agent_box_distance_weight,
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
        deadlock_reward = -self.deadlock_penalty * after_metrics["deadlocked_boxes"]
        agent_box_distance_reward = self.agent_box_distance_weight * (
            before_metrics["agent_box_distance"]
            - after_metrics["agent_box_distance"]
        )
        total = (
            distance_reward
            + pushability_reward
            + deadlock_reward
            + agent_box_distance_reward
        )
        return {
            "total": float(total),
            "distance": float(distance_reward),
            "pushability": float(pushability_reward),
            "deadlock": float(deadlock_reward),
            "agent_box_distance": float(agent_box_distance_reward),
            "box_target_distance_before": float(
                before_metrics["box_target_distance"]
            ),
            "box_target_distance_after": float(
                after_metrics["box_target_distance"]
            ),
            "pushability_before": int(before_metrics["pushability"]),
            "pushability_after": int(after_metrics["pushability"]),
            "deadlocked_boxes": int(after_metrics["deadlocked_boxes"]),
            "agent_box_distance_before": float(
                before_metrics["agent_box_distance"]
            ),
            "agent_box_distance_after": float(
                after_metrics["agent_box_distance"]
            ),
        }

    @staticmethod
    def empty_result():
        return {
            "total": 0.0,
            "distance": 0.0,
            "pushability": 0.0,
            "deadlock": 0.0,
            "agent_box_distance": 0.0,
            "box_target_distance_before": 0.0,
            "box_target_distance_after": 0.0,
            "pushability_before": 0,
            "pushability_after": 0,
            "deadlocked_boxes": 0,
            "agent_box_distance_before": 0.0,
            "agent_box_distance_after": 0.0,
        }

    def state_metrics(self, state):
        room_fixed = state["room_fixed"]
        room_state = state["room_state"]
        players = tuple(tuple(position) for position in state["players"])
        boxes = self._positions((room_state == BOX) | (room_state == BOX_ON_TARGET))
        targets = self._positions(room_fixed == TARGET)
        pushabilities = self._box_pushabilities(
            room_fixed, room_state, boxes, players
        )
        deadlocked_boxes = sum(
            pushability == 0 and room_fixed[box] != TARGET
            for box, pushability in zip(boxes, pushabilities)
        )
        return {
            "box_target_distance": self._minimum_target_matching_distance(
                room_fixed, boxes, targets
            ),
            "pushability": sum(pushabilities),
            "deadlocked_boxes": deadlocked_boxes,
            "agent_box_distance": self._agent_box_distance(
                room_fixed, room_state, boxes, players
            ),
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

    def _minimum_target_matching_distance(self, room_fixed, boxes, targets):
        if not boxes:
            return 0.0
        unreachable = float(room_fixed.size)
        distance_maps = [
            self._distance_map(room_fixed, target, blocked=set())
            for target in targets
        ]
        costs = tuple(
            tuple(
                distance_maps[target_index].get(box, unreachable)
                for target_index in range(len(targets))
            )
            for box in boxes
        )

        @lru_cache(maxsize=None)
        def solve(box_index, used_targets):
            if box_index == len(boxes):
                return 0.0
            best = float("inf")
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

    def _agent_box_distance(self, room_fixed, room_state, boxes, players):
        if not boxes:
            return 0.0
        unreachable = float(room_fixed.size)
        box_set = set(boxes)
        player_set = set(players)
        distance_maps = []
        for player in players:
            blocked = box_set | (player_set - {player})
            distance_maps.append(self._distance_map(room_fixed, player, blocked))

        total = 0.0
        for box in boxes:
            approach_cells = [
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
                    for cell in approach_cells
                    if cell in distance_map
                ),
                default=unreachable,
            )
            total += distance
        return total

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
