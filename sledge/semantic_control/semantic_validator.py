from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVectorRaw
from sledge.semantic_control.geometry_metrics import (
    compute_lateral_gap_to_lane_boundary,
    compute_lateral_interaction_ttc,
    compute_longitudinal_ttc,
    compute_merging_proxy_metrics,
    estimate_ego_speed,
    line_of_sight_intersects_box,
    state_heading,
    state_speed,
    state_xy,
    value_in_range,
    wrap_angle,
)
from sledge.semantic_control.hazard_spec import HazardSemanticSpec


@dataclass
class _ReportContext:
    spec: HazardSemanticSpec
    actor_elem_name: str = "none"
    actor_index: int = -1
    occluder_index: int = -1
    occluder_elem_name: str = "vehicles"
    static_obstacle_index: int = -1
    anchor: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def validate_scene_against_spec(scene: SledgeVectorRaw, ctx, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Strict semantic validation used by PrimitiveOps.

    This version adds:
      1. TTC checks
      2. gap checks
      3. line-of-sight OBB occlusion checks
      4. collision overlap proxy
    """
    return _validate(scene, ctx)


def validate_scene_against_report(scene: SledgeVectorRaw, report_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone validation from an existing semantic_report.json.

    This allows you to run strict validation after generation without rerunning
    the editor.
    """
    report = report_payload.get("report", report_payload)
    spec_data = report.get("spec")
    if spec_data is None:
        raise ValueError("Cannot find report['spec'] in semantic report payload.")

    spec = HazardSemanticSpec.from_dict(spec_data)

    edit_result = report_payload.get("edit_result", {})
    if not edit_result:
        edit_result = report.get("edit_result", {})

    primary_actor_type = edit_result.get("primary_actor_type", spec.actor_layer.primary_actor)
    primary_actor_index = int(edit_result.get("primary_actor_index", -1))

    actor_elem_name = _infer_elem_name(primary_actor_type)
    pedestrian_index = int(edit_result.get("pedestrian_index", -1))
    if primary_actor_type == "pedestrian" and pedestrian_index >= 0:
        primary_actor_index = pedestrian_index
        actor_elem_name = "pedestrians"

    conflict_point_xy = edit_result.get("conflict_point_xy", [0.0, 0.0])
    anchor = {
        "x": float(conflict_point_xy[0]) if len(conflict_point_xy) > 0 else 0.0,
        "y": float(conflict_point_xy[1]) if len(conflict_point_xy) > 1 else 0.0,
        "lane_y": float(conflict_point_xy[1]) if len(conflict_point_xy) > 1 else 0.0,
        "ego_speed": estimate_ego_speed(scene),
    }

    ctx = _ReportContext(
        spec=spec,
        actor_elem_name=actor_elem_name,
        actor_index=primary_actor_index,
        occluder_index=int(edit_result.get("occluder_index", -1)),
        occluder_elem_name=str(edit_result.get("occluder_elem_name", report.get("occluder_elem_name", "vehicles"))),
        static_obstacle_index=int(edit_result.get("static_obstacle_index", -1)),
        anchor=anchor,
        extra=dict(report.get("extra", {})),
    )
    return _validate(scene, ctx)


def _validate(scene: SledgeVectorRaw, ctx) -> Dict[str, Any]:
    spec = ctx.spec
    checks: Dict[str, Dict[str, Any]] = {}

    checks["actor_exists"] = _check_actor_exists(scene, ctx)

    if not checks["actor_exists"]["passed"]:
        return _finalize_checks(checks)

    actor_elem = _get_ctx_elem(scene, ctx)
    actor_state = _project_state_for_validation(actor_elem.states[ctx.actor_index], ctx)

    conflict_type = spec.interaction_layer.conflict_type

    if conflict_type in {"lateral_conflict", "crossing_path_conflict"}:
        checks["lateral_position"] = _check_lateral_position(scene, ctx, actor_state)
        checks["crossing_heading"] = _check_crossing_heading(scene, ctx, actor_state)
        checks["lateral_gap_in_range"] = _check_lateral_gap(scene, ctx, actor_state)
        checks["ttc_in_range"] = _check_lateral_ttc(scene, ctx, actor_state)

    elif conflict_type == "longitudinal_conflict":
        checks["longitudinal_gap_in_range"] = _check_longitudinal_gap_and_ttc(scene, ctx, actor_state)
        checks["speed_relation"] = _check_speed_relation(scene, ctx, actor_state)
        checks["ttc_in_range"] = _check_longitudinal_ttc(scene, ctx, actor_state)

    elif conflict_type == "oncoming_conflict":
        checks["oncoming_position"] = _check_oncoming_position(scene, ctx, actor_state)
        checks["oncoming_heading"] = _check_oncoming_heading(scene, ctx, actor_state)
        checks["speed_relation"] = _check_speed_relation(scene, ctx, actor_state)
        checks["ttc_in_range"] = _check_oncoming_ttc(scene, ctx, actor_state)

    elif conflict_type == "merging_conflict":
        checks["merging_position"] = _check_merging_position(scene, ctx, actor_state)
        checks["merging_heading"] = _check_merging_heading(scene, ctx, actor_state)
        checks["lateral_intrusion_in_range"] = _check_merging_intrusion(scene, ctx, actor_state)
        checks["ttc_in_range"] = _check_merging_ttc(scene, ctx, actor_state)

    elif conflict_type == "lane_blocking_conflict":
        checks["lane_blocker_position"] = _check_lane_blocker_position(scene, ctx, actor_state)
        checks["longitudinal_gap_in_range"] = _check_blocker_gap(scene, ctx, actor_state)

    if spec.object_layer.occlusion.enabled:
        checks["occluder_exists"] = _check_occluder_exists(scene, ctx)
        checks["occluder_between_ego_and_actor"] = _check_occluder_between_ego_and_actor(scene, ctx, actor_state)
        checks["line_of_sight_occlusion"] = _check_line_of_sight_occlusion(scene, ctx, actor_state)

    if spec.object_layer.static_obstacle.enabled:
        checks["static_obstacle_exists"] = _check_static_obstacle_exists(scene, ctx)

    checks["no_initial_collision"] = {
        "required": spec.validation_layer.require_no_initial_collision and not spec.risk_layer.collision_allowed,
        "passed": _check_no_initial_collision(scene, ctx),
    }

    return _finalize_checks(checks)


def _finalize_checks(checks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    required = [v for v in checks.values() if v.get("required", True)]
    passed = [bool(v.get("passed", False)) for v in required]
    ssr = sum(passed) / max(1, len(passed))
    return {
        "checks": checks,
        "semantic_satisfaction_rate": float(ssr),
        "overall_pass": bool(all(passed)),
    }


def _infer_elem_name(primary_actor_type: str) -> str:
    if primary_actor_type == "pedestrian":
        return "pedestrians"
    if primary_actor_type == "static_obstacle":
        return "static_objects"
    return "vehicles"


def _get_ctx_elem(scene: SledgeVectorRaw, ctx):
    if ctx.actor_elem_name == "vehicles":
        return scene.vehicles
    if ctx.actor_elem_name == "pedestrians":
        return scene.pedestrians
    if ctx.actor_elem_name == "static_objects":
        return scene.static_objects
    raise ValueError(f"Invalid actor_elem_name={ctx.actor_elem_name}")


def _check_actor_exists(scene: SledgeVectorRaw, ctx) -> Dict[str, Any]:
    if ctx.actor_index < 0:
        return {"required": True, "passed": False, "reason": "actor_index < 0"}
    elem = _get_ctx_elem(scene, ctx)
    if ctx.actor_index >= len(elem.mask):
        return {"required": True, "passed": False, "reason": "actor_index out of range"}
    return {
        "required": True,
        "passed": bool(elem.mask[ctx.actor_index]),
        "elem_name": ctx.actor_elem_name,
        "actor_index": int(ctx.actor_index),
    }


def _check_lateral_position(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    x, y = state_xy(actor_state)
    return {
        "required": True,
        "passed": bool(0.0 <= x <= 35.0 and abs(y) <= 12.0),
        "x": x,
        "y": y,
    }


def _check_crossing_heading(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    heading = state_heading(actor_state)
    ok = abs(abs(wrap_angle(heading)) - np.pi / 2.0) < 0.8
    return {
        "required": ctx.spec.validation_layer.require_direction_match,
        "passed": bool(ok),
        "heading": heading,
    }


def _check_lateral_gap(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    metrics = compute_lateral_gap_to_lane_boundary(
        actor_state,
        lane_y=float(ctx.anchor.get("lane_y", 0.0)),
        lane_half_width_m=0.5 * float(ctx.spec.road_layer.lane_width_m),
    )
    target = ctx.spec.risk_layer.lateral_gap_range_m
    # A small tolerance avoids false negatives from discrete slot/overlap repair.
    ok = value_in_range(metrics["lateral_gap_to_lane_boundary_m"], target, tolerance=0.5)
    return {
        "required": ctx.spec.validation_layer.require_gap_in_range,
        "passed": bool(ok),
        "target_lateral_gap_range_m": target,
        **metrics,
    }


def _check_lateral_ttc(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    ego_speed = float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene)))
    conflict_x = float(ctx.anchor.get("x", state_xy(actor_state)[0]))
    lane_y = float(ctx.anchor.get("lane_y", 0.0))
    metrics = compute_lateral_interaction_ttc(actor_state, ego_speed_mps=ego_speed, lane_y=lane_y, conflict_x=conflict_x)

    ttc_range = ctx.spec.risk_layer.ttc_range_s
    ttc = metrics["interaction_ttc_s"]
    time_gap = metrics["arrival_time_gap_s"]

    ok = (
        np.isfinite(ttc)
        and value_in_range(ttc, ttc_range, tolerance=0.6)
        and time_gap <= 1.5
    )

    return {
        "required": ctx.spec.validation_layer.require_ttc_in_range,
        "passed": bool(ok),
        "target_ttc_range_s": ttc_range,
        **metrics,
    }


def _check_longitudinal_gap_and_ttc(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    ego_speed = float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene)))
    metrics = compute_longitudinal_ttc(actor_state, ego_speed_mps=ego_speed)
    target = ctx.spec.risk_layer.gap_range_m
    ok = value_in_range(metrics["bumper_gap_m"], target, tolerance=2.5) or value_in_range(
        metrics["center_gap_m"], target, tolerance=2.5
    )
    return {
        "required": ctx.spec.validation_layer.require_gap_in_range,
        "passed": bool(ok),
        "target_gap_range_m": target,
        **metrics,
    }


def _check_speed_relation(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    speed = state_speed(actor_state)
    relation = ctx.spec.interaction_layer.speed_relation
    if relation == "slow_lead":
        ok = speed <= 6.5
    elif relation == "stopped":
        ok = speed <= 0.5
    elif relation == "fast_approach":
        ok = speed >= 5.0
    else:
        ok = speed >= 0.0
    return {
        "required": True,
        "passed": bool(ok),
        "speed_mps": speed,
        "speed_relation": relation,
    }


def _check_longitudinal_ttc(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    ego_speed = float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene)))
    metrics = compute_longitudinal_ttc(actor_state, ego_speed_mps=ego_speed)
    target = ctx.spec.risk_layer.ttc_range_s
    ok = np.isfinite(metrics["ttc_s"]) and value_in_range(metrics["ttc_s"], target, tolerance=0.8)
    return {
        "required": ctx.spec.validation_layer.require_ttc_in_range,
        "passed": bool(ok),
        "target_ttc_range_s": target,
        **metrics,
    }


def _check_oncoming_position(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    x, y = state_xy(actor_state)
    target_y = float(ctx.anchor.get("opposite_lane_y", ctx.spec.road_layer.lane_width_m))
    ok = 4.0 <= x <= 40.0 and abs(y - target_y) <= 1.8
    return {
        "required": True,
        "passed": bool(ok),
        "x": x,
        "y": y,
        "target_opposite_lane_y": target_y,
    }


def _check_oncoming_heading(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    heading = state_heading(actor_state)
    ok = abs(abs(wrap_angle(heading)) - np.pi) < 0.45
    return {
        "required": ctx.spec.validation_layer.require_direction_match,
        "passed": bool(ok),
        "heading": heading,
    }


def _check_oncoming_ttc(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    ego_speed = float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene)))
    actor_speed = state_speed(actor_state)
    x, y = state_xy(actor_state)
    actor_length = float(max(actor_state[AgentIndex.LENGTH], 0.5))
    ego_front = 2.5
    bumper_gap = max(0.0, x - ego_front - 0.5 * actor_length)
    closing_speed = max(ego_speed + actor_speed, 0.1)
    ttc = bumper_gap / closing_speed
    target = ctx.spec.risk_layer.ttc_range_s
    ok = np.isfinite(ttc) and value_in_range(ttc, target, tolerance=0.8)
    return {
        "required": ctx.spec.validation_layer.require_ttc_in_range,
        "passed": bool(ok),
        "target_ttc_range_s": target,
        "ttc_s": float(ttc),
        "bumper_gap_m": float(bumper_gap),
        "closing_speed_mps": float(closing_speed),
        "ego_speed_mps": float(ego_speed),
        "actor_speed_mps": float(actor_speed),
        "actor_xy": [float(x), float(y)],
    }


def _check_merging_position(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    x, y = state_xy(actor_state)
    ok = 0.0 <= x <= 25.0 and abs(y) <= 2.4
    return {
        "required": True,
        "passed": bool(ok),
        "x": x,
        "y": y,
    }


def _check_merging_heading(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    heading = state_heading(actor_state)
    ok = abs(heading) <= 0.35
    return {
        "required": ctx.spec.validation_layer.require_direction_match,
        "passed": bool(ok),
        "heading": heading,
    }


def _check_merging_intrusion(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    metrics = compute_merging_proxy_metrics(
        actor_state,
        ego_speed_mps=float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene))),
        lane_y=float(ctx.anchor.get("lane_y", 0.0)),
        lane_half_width_m=0.5 * float(ctx.spec.road_layer.lane_width_m),
    )
    target = ctx.spec.risk_layer.lateral_intrusion_range_m
    ok = value_in_range(metrics["lateral_intrusion_m"], target, tolerance=0.5)
    return {
        "required": ctx.spec.validation_layer.require_gap_in_range,
        "passed": bool(ok),
        "target_lateral_intrusion_range_m": target,
        **metrics,
    }


def _check_merging_ttc(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    metrics = compute_merging_proxy_metrics(
        actor_state,
        ego_speed_mps=float(ctx.anchor.get("ego_speed", estimate_ego_speed(scene))),
        lane_y=float(ctx.anchor.get("lane_y", 0.0)),
        lane_half_width_m=0.5 * float(ctx.spec.road_layer.lane_width_m),
    )
    target = ctx.spec.risk_layer.ttc_range_s
    ttc = metrics["time_to_actor_x_s"]
    ok = np.isfinite(ttc) and value_in_range(ttc, target, tolerance=1.0)
    return {
        "required": ctx.spec.validation_layer.require_ttc_in_range,
        "passed": bool(ok),
        "target_ttc_range_s": target,
        **metrics,
    }


def _check_lane_blocker_position(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    x, y = state_xy(actor_state)
    ok = 0.0 <= x <= 30.0 and abs(y) <= 3.0
    return {
        "required": True,
        "passed": bool(ok),
        "x": x,
        "y": y,
    }


def _check_blocker_gap(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    x, y = state_xy(actor_state)
    target = ctx.spec.risk_layer.gap_range_m
    ok = value_in_range(max(0.0, x), target, tolerance=3.0)
    return {
        "required": ctx.spec.validation_layer.require_gap_in_range,
        "passed": bool(ok),
        "target_gap_range_m": target,
        "center_gap_m": max(0.0, x),
        "y": y,
    }


def _check_occluder_exists(scene: SledgeVectorRaw, ctx) -> Dict[str, Any]:
    elem = _get_occluder_elem(scene, ctx)
    ok = (
        elem is not None
        and ctx.occluder_index >= 0
        and ctx.occluder_index < len(elem.mask)
        and bool(elem.mask[ctx.occluder_index])
    )
    return {
        "required": True,
        "passed": bool(ok),
        "occluder_index": int(ctx.occluder_index),
        "occluder_elem_name": str(getattr(ctx, "occluder_elem_name", "vehicles")),
    }


def _check_occluder_between_ego_and_actor(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    occ_state = _get_occluder_state(scene, ctx)
    if occ_state is None:
        return {"required": True, "passed": False, "reason": "invalid occluder index"}

    occ_state = _project_state_for_validation(occ_state, ctx)
    ax, ay = state_xy(actor_state)
    ox, oy = state_xy(occ_state)

    actor_dist2 = ax * ax + ay * ay
    if actor_dist2 <= 1e-4:
        return {"required": True, "passed": False, "reason": "actor too close to ego"}

    ratio = (ox * ax + oy * ay) / actor_dist2
    perpendicular = abs(ox * ay - oy * ax) / np.sqrt(actor_dist2)

    ok = 0.1 <= ratio <= 0.95 and perpendicular <= 3.0
    return {
        "required": True,
        "passed": bool(ok),
        "projection_ratio": float(ratio),
        "perpendicular_distance_m": float(perpendicular),
        "actor_xy": [float(ax), float(ay)],
        "occluder_xy": [float(ox), float(oy)],
    }


def _check_line_of_sight_occlusion(scene: SledgeVectorRaw, ctx, actor_state: np.ndarray) -> Dict[str, Any]:
    occ_state = _get_occluder_state(scene, ctx)
    if occ_state is None:
        return {"required": True, "passed": False, "reason": "invalid occluder index"}

    occ_state = _project_state_for_validation(occ_state, ctx)
    actor_xy = state_xy(actor_state)
    ego_xy = (0.0, 0.0)

    # Small margin allows partial occlusion to count even if the line grazes the box.
    intersects = line_of_sight_intersects_box(ego_xy, actor_xy, occ_state, margin=0.25)

    return {
        "required": ctx.spec.validation_layer.require_visibility_match,
        "passed": bool(intersects),
        "ego_xy": [0.0, 0.0],
        "actor_xy": [float(actor_xy[0]), float(actor_xy[1])],
        "occluder_index": int(ctx.occluder_index),
        "occluder_elem_name": str(getattr(ctx, "occluder_elem_name", "vehicles")),
    }


def _check_static_obstacle_exists(scene: SledgeVectorRaw, ctx) -> Dict[str, Any]:
    ok = (
        ctx.static_obstacle_index >= 0
        and ctx.static_obstacle_index < len(scene.static_objects.mask)
        and bool(scene.static_objects.mask[ctx.static_obstacle_index])
    )
    return {
        "required": True,
        "passed": bool(ok),
        "static_obstacle_index": int(ctx.static_obstacle_index),
    }


def _check_no_initial_collision(scene: SledgeVectorRaw, ctx=None) -> bool:
    """
    Conservative axis-aligned overlap proxy.

    It also checks a proxy ego box at the origin because agents that overlap or
    nearly overlap ego can be filtered before they reach SimulationLog.
    When semantic_validation_time_offset_s is available, dynamic agents are
    projected to the same frame-0 convention used by SledgeBoard.
    """
    boxes = [("ego", -1, 0.0, 0.0, 5.0, 2.1)]
    for elem_name, elem in [
        ("vehicles", scene.vehicles),
        ("pedestrians", scene.pedestrians),
        ("static_objects", scene.static_objects),
    ]:
        mask = np.asarray(elem.mask).astype(bool)
        states = np.asarray(elem.states)
        for idx in np.where(mask)[0]:
            s = _project_state_for_validation(states[idx], ctx) if ctx is not None else states[idx]
            x = float(s[AgentIndex.X])
            y = float(s[AgentIndex.Y])
            w = float(max(s[AgentIndex.WIDTH], 0.5))
            l = float(max(s[AgentIndex.LENGTH], 0.5))
            boxes.append((elem_name, int(idx), x, y, l, w))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            name_i, idx_i, ax, ay, al, aw = boxes[i]
            name_j, idx_j, bx, by, bl, bw = boxes[j]

            # Slightly relaxed for different object classes but strict enough to
            # catch ego/occluder overlaps.
            scale = 0.42
            if name_i != name_j:
                scale = 0.35
            if name_i == "ego" or name_j == "ego":
                scale = 0.42

            if abs(ax - bx) < scale * (al + bl) and abs(ay - by) < scale * (aw + bw):
                return False
    return True


def _validation_time_offset_s(ctx) -> float:
    extra = getattr(ctx, "extra", {}) or {}
    if not bool(extra.get("use_projected_validation", False)):
        return 0.0
    return float(extra.get("semantic_validation_time_offset_s", 0.0))


def _project_state_for_validation(state: np.ndarray, ctx) -> np.ndarray:
    t = _validation_time_offset_s(ctx)
    projected = np.asarray(state, dtype=np.float32).copy()
    if t <= 0.0:
        return projected
    speed = float(max(projected[AgentIndex.VELOCITY], 0.0))
    heading = float(projected[AgentIndex.HEADING])
    projected[AgentIndex.X] = float(projected[AgentIndex.X]) + speed * np.cos(heading) * t
    projected[AgentIndex.Y] = float(projected[AgentIndex.Y]) + speed * np.sin(heading) * t
    return projected


def _get_occluder_elem(scene: SledgeVectorRaw, ctx):
    elem_name = str(getattr(ctx, "occluder_elem_name", "vehicles"))
    if elem_name == "vehicles":
        return scene.vehicles
    if elem_name == "static_objects":
        return scene.static_objects
    return None


def _get_occluder_state(scene: SledgeVectorRaw, ctx):
    elem = _get_occluder_elem(scene, ctx)
    if elem is None or ctx.occluder_index < 0 or ctx.occluder_index >= len(elem.mask):
        return None
    if not bool(elem.mask[ctx.occluder_index]):
        return None
    return elem.states[ctx.occluder_index]
