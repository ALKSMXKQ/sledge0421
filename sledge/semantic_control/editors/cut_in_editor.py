from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


# Scheme A:
# Instead of asking simulation to execute a full lane change,
# directly place the cut-in vehicle in a "half-inserted" or "already inserted"
# hazardous state near / inside ego lane, ahead of ego.
#
# The target y is intentionally placed INSIDE the ego lane boundary
# (|y| < lane_half_width), so that simulation snapping is more likely to keep
# the vehicle on ego-lane rails rather than the adjacent lane.
_SEVERITY_CFG: Dict[str, Dict[str, float]] = {
    "mild": {
        "front_x_min": 8.0,
        "front_x_max": 12.0,
        "insert_y_abs_min": 1.15,
        "insert_y_abs_max": 1.55,
        "heading_mag_min": 0.02,
        "heading_mag_max": 0.07,
        "rel_speed_min": 0.0,
        "rel_speed_max": 1.0,
    },
    "moderate": {
        "front_x_min": 5.5,
        "front_x_max": 8.5,
        "insert_y_abs_min": 0.70,
        "insert_y_abs_max": 1.15,
        "heading_mag_min": 0.03,
        "heading_mag_max": 0.10,
        "rel_speed_min": 0.4,
        "rel_speed_max": 1.5,
    },
    "aggressive": {
        "front_x_min": 3.0,
        "front_x_max": 6.0,
        "insert_y_abs_min": 0.25,
        "insert_y_abs_max": 0.75,
        "heading_mag_min": 0.05,
        "heading_mag_max": 0.14,
        "rel_speed_min": 0.8,
        "rel_speed_max": 2.0,
    },
}


class CutInEditor:
    """
    Front-insert cut-in proxy for simulation.

    Key design:
    - vehicle is already partially inserted into ego lane
    - vehicle is placed ahead of ego
    - vehicle heading still has a slight tendency toward lane center
    - remove blocking same-lane vehicles near the insertion zone
    """

    def __init__(self) -> None:
        self._ego_lane_y = 0.0
        self._lane_half_width_m = 1.8
        self._clearance_radius_m = 2.2
        self._spawn_x_jitter = [0.0, -0.8, 0.8, -1.5, 1.5]
        self._min_speed = 2.5
        self._max_speed = 15.0

    def edit(self, scene: SledgeVectorRaw, spec: PromptSpec) -> Tuple[SledgeVectorRaw, SceneEditResult]:
        edited = copy.deepcopy(scene)

        severity = getattr(spec, "severity_level", "moderate")
        if severity not in _SEVERITY_CFG:
            severity = "moderate"
        cfg = _SEVERITY_CFG[severity]

        lane_y = self._ego_lane_y
        ego_speed = self._estimate_ego_speed(edited)

        if getattr(spec, "side", "auto") == "left":
            side_sign = 1.0
        elif getattr(spec, "side", "auto") == "right":
            side_sign = -1.0
        else:
            side_sign = self._choose_side(edited)

        # Put the vehicle ahead of ego and already inside ego-lane boundary
        target_x = 0.5 * (cfg["front_x_min"] + cfg["front_x_max"])
        insert_y_abs = 0.5 * (cfg["insert_y_abs_min"] + cfg["insert_y_abs_max"])
        target_y = side_sign * insert_y_abs

        # Slight heading toward lane center to preserve "recently cut in" feel.
        heading_mag = 0.5 * (cfg["heading_mag_min"] + cfg["heading_mag_max"])
        heading = -side_sign * heading_mag

        rel_speed = 0.5 * (cfg["rel_speed_min"] + cfg["rel_speed_max"])
        speed = float(np.clip(ego_speed + rel_speed, self._min_speed, self._max_speed))

        target_x, target_y = self._resolve_spawn_overlap(edited, target_x, target_y)

        # Remove one blocking same-lane vehicle in front insertion zone
        removed_indices = self._clear_conflicting_same_lane_vehicle(
            edited.vehicles,
            target_x=target_x,
            lane_y=lane_y,
        )

        vehicle_idx = self._allocate_vehicle_slot(edited.vehicles)
        self._set_vehicle_state(
            edited.vehicles,
            vehicle_idx,
            x=target_x,
            y=target_y,
            heading=heading,
            width=1.9,
            length=4.8,
            velocity=speed,
        )

        rois = self._build_rois(
            vehicle_state=edited.vehicles.states[vehicle_idx],
            lane_y=lane_y,
            insert_x=target_x,
        )

        result = SceneEditResult(
            prompt_spec=spec,
            primary_actor_type="vehicle",
            primary_actor_index=vehicle_idx,
            conflict_point_xy=[float(target_x), float(lane_y)],
            preserved_rois=rois,
            removed_vehicle_indices=removed_indices,
            notes=[
                f"scenario set to front-insert cut-in proxy ({severity})",
                f"ego speed estimate is {ego_speed:.2f} m/s",
                f"vehicle placed ahead at x={target_x:.2f} m",
                f"vehicle inserted into ego lane at y={target_y:.2f} m",
                f"vehicle heading set to {heading:.3f} rad toward lane center",
                f"vehicle speed set to {speed:.2f} m/s",
                f"side_sign={side_sign:+.0f}",
                f"removed blocking vehicles: {removed_indices}",
            ],
        )
        return edited, result

    def _estimate_ego_speed(self, scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
        if ego_states.size == 0:
            return 6.0
        if ego_states.size >= 2:
            speed = float(np.linalg.norm(ego_states[:2]))
        else:
            speed = float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    def _choose_side(self, scene: SledgeVectorRaw) -> float:
        vehicles = self._valid_agent_states(scene.vehicles)
        if len(vehicles) == 0:
            return 1.0
        left_count = int(np.sum(vehicles[:, AgentIndex.Y] > 1.5))
        right_count = int(np.sum(vehicles[:, AgentIndex.Y] < -1.5))
        return 1.0 if left_count <= right_count else -1.0

    def _resolve_spawn_overlap(self, scene: SledgeVectorRaw, target_x: float, target_y: float) -> Tuple[float, float]:
        for dx in self._spawn_x_jitter:
            cand_x = target_x + dx
            if self._is_clear(scene, cand_x, target_y):
                return cand_x, target_y
        return target_x, target_y

    def _is_clear(self, scene: SledgeVectorRaw, x: float, y: float) -> bool:
        p = np.array([x, y], dtype=np.float32)
        for elem in [scene.vehicles, scene.pedestrians, scene.static_objects]:
            valid = np.asarray(elem.mask).astype(bool)
            if not np.any(valid):
                continue
            centers = np.asarray(elem.states)[valid, :2]
            if centers.size == 0:
                continue
            dists = np.linalg.norm(centers - p[None, :], axis=1)
            if np.any(dists < self._clearance_radius_m):
                return False
        return True

    def _clear_conflicting_same_lane_vehicle(
        self,
        elem: SledgeVectorElement,
        target_x: float,
        lane_y: float,
    ) -> List[int]:
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask).astype(bool)
        removed: List[int] = []

        candidates = np.where(mask)[0]
        for idx in candidates:
            x = float(states[idx, AgentIndex.X])
            y = float(states[idx, AgentIndex.Y])

            if abs(y - lane_y) < 1.2 and (target_x - 2.0) <= x <= (target_x + 5.0):
                elem.mask[idx] = False
                removed.append(int(idx))
                break

        return removed

    def _allocate_vehicle_slot(self, elem: SledgeVectorElement) -> int:
        valid = np.asarray(elem.mask).astype(bool)
        invalid = np.where(~valid)[0]
        if len(invalid) > 0:
            idx = int(invalid[0])
            elem.mask[idx] = True
            return idx

        states = np.asarray(elem.states)
        distances = np.linalg.norm(states[:, :2], axis=1)
        idx = int(np.argmax(distances))
        elem.mask[idx] = True
        return idx

    @staticmethod
    def _set_vehicle_state(
        elem: SledgeVectorElement,
        idx: int,
        x: float,
        y: float,
        heading: float,
        width: float,
        length: float,
        velocity: float,
    ) -> None:
        elem.states[idx, AgentIndex.X] = x
        elem.states[idx, AgentIndex.Y] = y
        elem.states[idx, AgentIndex.HEADING] = heading
        elem.states[idx, AgentIndex.WIDTH] = width
        elem.states[idx, AgentIndex.LENGTH] = length
        elem.states[idx, AgentIndex.VELOCITY] = velocity
        elem.mask[idx] = True

    def _build_rois(
        self,
        vehicle_state: np.ndarray,
        lane_y: float,
        insert_x: float,
    ) -> list[SceneEditROI]:
        x = float(vehicle_state[AgentIndex.X])
        y = float(vehicle_state[AgentIndex.Y])
        width = float(max(vehicle_state[AgentIndex.WIDTH], 1.8))
        length = float(max(vehicle_state[AgentIndex.LENGTH], 4.5))
        y0, y1 = sorted([y, lane_y])

        return [
            SceneEditROI(
                x_min=x - length / 2 - 1.0,
                y_min=y - width / 2 - 1.0,
                x_max=x + length / 2 + 1.0,
                y_max=y + width / 2 + 1.0,
                tag="vehicle",
            ),
            SceneEditROI(
                x_min=x - 2.0,
                y_min=y0 - 1.0,
                x_max=insert_x + 3.0,
                y_max=y1 + 1.0,
                tag="merge_corridor",
            ),
            SceneEditROI(
                x_min=insert_x - 2.0,
                y_min=lane_y - self._lane_half_width_m - 0.8,
                x_max=insert_x + 3.0,
                y_max=lane_y + self._lane_half_width_m + 0.8,
                tag="ego_lane_entry_anchor",
            ),
        ]

    @staticmethod
    def _valid_agent_states(elem: SledgeVectorElement) -> np.ndarray:
        valid = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        return states[valid] if np.any(valid) else np.zeros((0, states.shape[-1]), dtype=states.dtype)