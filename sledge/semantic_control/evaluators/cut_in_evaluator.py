from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVector

LABEL_THRESH = 0.3

_DYNAMIC_TARGETS = {
    "mild": {
        "t_center_peak": 2.2,
        "t_center_half_width": 0.9,
        "cross_x_peak": 11.0,
        "cross_x_half_width": 4.5,
        "start_x_low": 0.0,
        "start_x_high": 9.0,
        "adjacent_y_low": 1.4,
        "adjacent_y_high": 5.0,
    },
    "moderate": {
        "t_center_peak": 1.6,
        "t_center_half_width": 0.7,
        "cross_x_peak": 8.0,
        "cross_x_half_width": 3.5,
        "start_x_low": -2.0,
        "start_x_high": 7.0,
        "adjacent_y_low": 1.2,
        "adjacent_y_high": 5.2,
    },
    "aggressive": {
        "t_center_peak": 1.0,
        "t_center_half_width": 0.45,
        "cross_x_peak": 5.5,
        "cross_x_half_width": 2.5,
        "start_x_low": -4.0,
        "start_x_high": 5.0,
        "adjacent_y_low": 1.0,
        "adjacent_y_high": 5.5,
    },
}

_INSERT_TARGETS = {
    "mild": {
        "front_x_peak": 10.0,
        "front_x_half_width": 5.5,
        "insert_y_peak": 1.35,
        "insert_y_half_width": 0.80,
        "ttc_peak": 2.0,
        "ttc_half_width": 1.0,
    },
    "moderate": {
        "front_x_peak": 7.0,
        "front_x_half_width": 4.2,
        "insert_y_peak": 0.95,
        "insert_y_half_width": 0.65,
        "ttc_peak": 1.35,
        "ttc_half_width": 0.75,
    },
    "aggressive": {
        "front_x_peak": 4.5,
        "front_x_half_width": 3.0,
        "insert_y_peak": 0.50,
        "insert_y_half_width": 0.45,
        "ttc_peak": 0.90,
        "ttc_half_width": 0.45,
    },
}


@dataclass
class PromptAlignmentResult:
    total: float
    details: Dict[str, float]
    notes: List[str] = field(default_factory=list)
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": float(self.total),
            "details": {k: float(v) for k, v in self.details.items()},
            "notes": list(self.notes),
            "accepted": bool(self.accepted),
        }


class CutInAlignmentEvaluator:
    """
    Hybrid cut-in evaluator for vector-space evaluation.

    This evaluator supports two compatible cut-in interpretations:
      1) dynamic cut-in proxy:
         a vehicle starts from an adjacent-lane region and is moving toward ego lane.
      2) front-insert state proxy:
         a vehicle is already partially inserted / inserted ahead of ego.

    Final score = max(dynamic_score, insert_score), so legacy edited caches and
    newer front-insert style scenes can be evaluated under one consistent rule.
    """

    def __init__(
        self,
        lane_half_width_m: float = 1.8,
        prediction_horizon_s: float = 4.0,
        prediction_dt_s: float = 0.1,
    ) -> None:
        self.lane_half_width_m = lane_half_width_m
        self.prediction_horizon_s = prediction_horizon_s
        self.prediction_dt_s = prediction_dt_s
        self._ego_lane_y = 0.0

    def evaluate(self, sledge_vector: SledgeVector, prompt_spec: Any = None) -> PromptAlignmentResult:
        vehicles = self._collect_valid_vehicles(sledge_vector)
        if len(vehicles) == 0:
            return PromptAlignmentResult(
                total=0.0,
                details={
                    "dynamic_total": 0.0,
                    "insert_total": 0.0,
                    "selected_mode_dynamic": 0.0,
                    "selected_mode_insert": 0.0,
                },
                notes=["no valid vehicle detected"],
                accepted=False,
            )

        severity = getattr(prompt_spec, "severity_level", "moderate") if prompt_spec is not None else "moderate"
        severity = severity if severity in _DYNAMIC_TARGETS else "moderate"

        dynamic_target = _DYNAMIC_TARGETS[severity]
        insert_target = _INSERT_TARGETS[severity]
        lane_y = self._ego_lane_y

        dynamic_best = None
        dynamic_best_total = -1.0
        insert_best = None
        insert_best_total = -1.0

        for veh in vehicles:
            dyn_metrics = self._score_dynamic_cut_in_vehicle(veh, lane_y, dynamic_target)
            if dyn_metrics["vehicle_presence_score"] > 0.0 and dyn_metrics["total"] > dynamic_best_total:
                dynamic_best_total = dyn_metrics["total"]
                dynamic_best = dyn_metrics

            ins_metrics = self._score_front_insert_vehicle(veh, lane_y, insert_target)
            if ins_metrics["vehicle_presence_score"] > 0.0 and ins_metrics["total"] > insert_best_total:
                insert_best_total = ins_metrics["total"]
                insert_best = ins_metrics

        dynamic_total = float(max(dynamic_best_total, 0.0))
        insert_total = float(max(insert_best_total, 0.0))

        if dynamic_total >= insert_total:
            best_mode = "dynamic"
            best = dynamic_best if dynamic_best is not None else self._empty_metrics("no dynamic cut-in candidate")
            total = dynamic_total
        else:
            best_mode = "insert"
            best = insert_best if insert_best is not None else self._empty_metrics("no front-insert candidate")
            total = insert_total

        accepted = total >= 0.58
        notes = list(best.get("notes", []))
        notes.append(
            f"cut-in semantics enabled: hybrid evaluator, selected_mode={best_mode}, severity={severity}"
        )

        details = {
            "dynamic_total": dynamic_total,
            "insert_total": insert_total,
            "selected_mode_dynamic": 1.0 if best_mode == "dynamic" else 0.0,
            "selected_mode_insert": 1.0 if best_mode == "insert" else 0.0,
        }

        for key, value in best.items():
            if key == "notes":
                continue
            try:
                details[key] = float(value)
            except Exception:
                pass

        return PromptAlignmentResult(
            total=float(np.clip(total, 0.0, 1.0)),
            details=details,
            notes=notes,
            accepted=accepted,
        )

    def _collect_valid_vehicles(self, sledge_vector: SledgeVector) -> List[np.ndarray]:
        states = np.asarray(sledge_vector.vehicles.states)
        masks = np.asarray(sledge_vector.vehicles.mask)

        if states.size == 0:
            return []
        if states.ndim == 1:
            states = states[None, :]
        if masks.ndim == 0:
            masks = np.asarray([masks])

        vehicles: List[np.ndarray] = []
        for state, mask in zip(states, masks):
            valid = bool(mask) if isinstance(mask, (bool, np.bool_)) else float(mask) >= LABEL_THRESH
            if valid:
                vehicles.append(np.asarray(state, dtype=np.float32))
        return vehicles

    def _score_dynamic_cut_in_vehicle(self, veh: np.ndarray, lane_y: float, target: Dict[str, float]) -> Dict[str, Any]:
        x0 = float(veh[AgentIndex.X])
        y0 = float(veh[AgentIndex.Y])
        heading = float(veh[AgentIndex.HEADING])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        lat_abs = abs(y0 - lane_y)
        if speed < 0.25:
            return self._empty_metrics("vehicle is almost stationary, not a dynamic cut-in candidate")
        if x0 < -10.0 or x0 > 20.0:
            return self._empty_metrics("vehicle starts outside practical dynamic cut-in longitudinal range")
        if lat_abs < 1.0 or lat_abs > 6.0:
            return self._empty_metrics("vehicle is not in a plausible adjacent-lane band for dynamic cut-in")

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)

        toward_ego_lane = (y0 > lane_y and vy < 0.0) or (y0 < lane_y and vy > 0.0)
        lateral_ratio = abs(vy) / max(1e-3, abs(vx) + abs(vy))

        vehicle_presence_score = 1.0
        adjacent_origin_score = self._band_score(lat_abs, target["adjacent_y_low"], target["adjacent_y_high"])
        merge_direction_score = float(np.clip(0.55 * float(toward_ego_lane) + 0.45 * np.clip(lateral_ratio / 0.16, 0.0, 1.0), 0.0, 1.0))

        times = np.arange(0.0, self.prediction_horizon_s + 1e-6, self.prediction_dt_s, dtype=np.float32)
        future_x = x0 + vx * times
        future_y = y0 + vy * times

        center_band = np.abs(future_y - lane_y) <= 0.9
        front_band = future_x >= 1.5
        crossing = center_band & front_band

        if np.any(crossing):
            idx = int(np.argmax(crossing))
            t_cross = float(times[idx])
            x_cross = float(future_x[idx])
            front_insert_score = self._triangular_score(x_cross, target["cross_x_peak"], target["cross_x_half_width"])
            cut_in_immediacy_score = self._triangular_score(t_cross, target["t_center_peak"], target["t_center_half_width"])
        else:
            t_cross = float("inf")
            x_cross = float("inf")
            min_center_dist = float(np.min(np.abs(future_y - lane_y)))
            front_insert_score = float(np.clip(1.0 - abs(float(np.mean(future_x)) - target["cross_x_peak"]) / max(1e-3, target["cross_x_half_width"] * 2.0), 0.0, 0.35))
            cut_in_immediacy_score = float(np.clip(1.0 - min_center_dist / 3.0, 0.0, 0.30))

        pre_merge_proximity_score = self._band_score(x0, target["start_x_low"], target["start_x_high"])

        raw_total = (
            0.10 * vehicle_presence_score
            + 0.22 * adjacent_origin_score
            + 0.20 * merge_direction_score
            + 0.25 * front_insert_score
            + 0.13 * cut_in_immediacy_score
            + 0.10 * pre_merge_proximity_score
        )

        origin_gate = 1.0 if adjacent_origin_score >= 0.25 else 0.45
        direction_gate = 1.0 if merge_direction_score >= 0.25 else 0.55
        total = float(np.clip(raw_total * origin_gate * direction_gate, 0.0, 1.0))

        notes: List[str] = []
        if adjacent_origin_score < 0.25:
            notes.append("dynamic mode: vehicle is not clearly starting from adjacent lane")
        if merge_direction_score < 0.25:
            notes.append("dynamic mode: vehicle is not clearly moving toward ego lane")
        if front_insert_score < 0.35:
            notes.append("dynamic mode: predicted merge point is not clearly ahead of ego")
        if cut_in_immediacy_score < 0.25:
            notes.append("dynamic mode: timing does not match requested severity")
        notes.append(
            f"dynamic debug: start=(x={x0:.2f}, y={y0:.2f}), vx={vx:.2f}, vy={vy:.2f}, "
            f"t_cross={t_cross if np.isfinite(t_cross) else -1:.2f}, "
            f"x_cross={x_cross if np.isfinite(x_cross) else -999:.2f}"
        )

        return {
            "vehicle_presence_score": vehicle_presence_score,
            "adjacent_origin_score": float(np.clip(adjacent_origin_score, 0.0, 1.0)),
            "merge_direction_score": float(np.clip(merge_direction_score, 0.0, 1.0)),
            "front_insert_score": float(np.clip(front_insert_score, 0.0, 1.0)),
            "cut_in_immediacy_score": float(np.clip(cut_in_immediacy_score, 0.0, 1.0)),
            "pre_merge_proximity_score": float(np.clip(pre_merge_proximity_score, 0.0, 1.0)),
            "total": total,
            "notes": notes,
        }

    def _score_front_insert_vehicle(self, veh: np.ndarray, lane_y: float, target: Dict[str, float]) -> Dict[str, Any]:
        x0 = float(veh[AgentIndex.X])
        y0 = float(veh[AgentIndex.Y])
        heading = float(veh[AgentIndex.HEADING])
        speed = float(max(0.0, veh[AgentIndex.VELOCITY]))

        if speed < 0.2:
            return self._empty_metrics("vehicle is almost stationary, not a front-insert cut-in candidate")
        if x0 < 0.5 or x0 > 22.0:
            return self._empty_metrics("vehicle is not in practical front-insert longitudinal range")

        lat_abs = abs(y0 - lane_y)
        if lat_abs > self.lane_half_width_m + 0.8:
            return self._empty_metrics("vehicle is still too far from ego lane to be a front-insert state")

        vehicle_presence_score = 1.0

        lane_insert_state_score = self._lane_insert_state_score(lat_abs)

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        toward_center = 1.0 if (y0 > lane_y and vy < 0.0) or (y0 < lane_y and vy > 0.0) else 0.0
        small_heading = float(np.clip(1.0 - abs(heading) / 0.50, 0.0, 1.0))
        heading_consistency_score = float(np.clip(0.25 * toward_center + 0.75 * small_heading, 0.0, 1.0))

        front_insert_score = self._triangular_score(x0, target["front_x_peak"], target["front_x_half_width"])

        pseudo_ttc = x0 / max(speed, 1e-3)
        hazard_immediacy_score = self._triangular_score(pseudo_ttc, target["ttc_peak"], target["ttc_half_width"])

        proximity_score = float(np.clip(1.0 - max(0.0, x0 - 16.0) / 10.0, 0.0, 1.0))

        raw_total = (
            0.10 * vehicle_presence_score
            + 0.28 * lane_insert_state_score
            + 0.10 * heading_consistency_score
            + 0.30 * front_insert_score
            + 0.14 * hazard_immediacy_score
            + 0.08 * proximity_score
        )

        lane_gate = 1.0 if lane_insert_state_score >= 0.30 else 0.45
        front_gate = 1.0 if front_insert_score >= 0.35 else 0.40
        total = float(np.clip(raw_total * lane_gate * front_gate, 0.0, 1.0))

        notes: List[str] = []
        if lane_insert_state_score < 0.30:
            notes.append("insert mode: vehicle is not sufficiently inserted near ego-lane area")
        if heading_consistency_score < 0.25:
            notes.append("insert mode: heading weakly supports a recent cut-in interpretation")
        if front_insert_score < 0.35:
            notes.append("insert mode: vehicle is not clearly inserted ahead of ego")
        if hazard_immediacy_score < 0.20:
            notes.append("insert mode: front-insert severity does not match requested tier")
        notes.append(
            f"insert debug: state=(x={x0:.2f}, y={y0:.2f}), vx={vx:.2f}, vy={vy:.2f}, ttc={pseudo_ttc:.2f}"
        )

        return {
            "vehicle_presence_score": vehicle_presence_score,
            "lane_insert_state_score": float(np.clip(lane_insert_state_score, 0.0, 1.0)),
            "heading_consistency_score": float(np.clip(heading_consistency_score, 0.0, 1.0)),
            "front_insert_score": float(np.clip(front_insert_score, 0.0, 1.0)),
            "hazard_immediacy_score": float(np.clip(hazard_immediacy_score, 0.0, 1.0)),
            "proximity_score": float(np.clip(proximity_score, 0.0, 1.0)),
            "total": total,
            "notes": notes,
        }

    def _lane_insert_state_score(self, lat_abs: float) -> float:
        if lat_abs <= 0.60:
            return 0.95
        if lat_abs <= 1.80:
            return float(0.75 + 0.20 * (1.80 - lat_abs) / 1.20)
        if lat_abs <= 2.40:
            return float(0.45 + 0.30 * (2.40 - lat_abs) / 0.60)
        return 0.0

    @staticmethod
    def _empty_metrics(reason: str) -> Dict[str, Any]:
        return {
            "vehicle_presence_score": 0.0,
            "total": 0.0,
            "notes": [reason],
        }

    @staticmethod
    def _band_score(value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 1.0
        if value < low:
            return float(np.clip(1.0 - (low - value) / max(1e-6, low), 0.0, 1.0))
        return float(np.clip(1.0 - (value - high) / max(1e-6, high), 0.0, 1.0))

    @staticmethod
    def _triangular_score(value: float, peak: float, half_width: float) -> float:
        if half_width <= 0:
            return 0.0
        return float(np.clip(1.0 - abs(value - peak) / half_width, 0.0, 1.0))
