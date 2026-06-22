from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control.prompt_spec import SceneEditROI


SLEDGEBOARD_FRAME0_OFFSET_S = 2.1
DEFAULT_EGO_LENGTH_M = 5.0
DEFAULT_EGO_WIDTH_M = 2.1


@dataclass(frozen=True)
class OccluderSpec:
    """Executable occluder definition used by add_or_select_occluder."""

    name: str
    elem_name: str  # "vehicles" or "static_objects"
    width: float
    length: float
    default_heading: float
    velocity: float = 0.0


OCCLUDER_SPECS: Dict[str, OccluderSpec] = {
    # Use pi/2 for parked vehicles/trucks by default. This keeps the long vehicle
    # axis along the lateral direction and avoids placing a 5m/7.5m box directly
    # on top of the ego front bumper when the pedestrian is close.
    "parked_vehicle": OccluderSpec("parked_vehicle", "vehicles", width=2.1, length=5.0, default_heading=math.pi / 2.0),
    "truck": OccluderSpec("truck", "vehicles", width=2.5, length=7.5, default_heading=math.pi / 2.0),
    "bus": OccluderSpec("bus", "vehicles", width=2.6, length=9.0, default_heading=math.pi / 2.0),
    "barrier": OccluderSpec("barrier", "static_objects", width=0.6, length=2.0, default_heading=math.pi / 2.0),
    "construction_object": OccluderSpec("construction_object", "static_objects", width=1.2, length=1.2, default_heading=0.0),
    "static_object": OccluderSpec("static_object", "static_objects", width=1.2, length=2.0, default_heading=math.pi / 2.0),
}

_STATIC_OCCLUDER_ALIASES = {
    "barrier",
    "road_barrier",
    "construction",
    "construction_object",
    "cone",
    "roadblock",
    "static",
    "static_object",
    "static_obstacle",
    "generic_object",
}

_VEHICLE_OCCLUDER_ALIASES = {
    "car": "parked_vehicle",
    "vehicle": "parked_vehicle",
    "parked_car": "parked_vehicle",
    "parked_vehicle": "parked_vehicle",
    "occluding_vehicle": "parked_vehicle",
    "truck": "truck",
    "bus": "bus",
    "large_vehicle": "truck",
}


class PrimitiveOps:
    """
    Reusable low-level semantic editing operations.

    These ops are intentionally smaller than scenario-level editors:
    - select/spawn actor
    - select road anchor
    - place actor according to relation
    - add occluder/static obstacle
    - build protection rois
    - validate semantic constraints
    """

    def select_road_anchor(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        lane_y = self._estimate_lane_center_y(scene)
        ego_speed = self._estimate_ego_speed(scene)
        anchor_type = params.get("anchor_type", "ego_future_path")
        lane_context = params.get("lane_context", "ego_path")
        risk = ctx.spec.risk_layer
        longitudinal_mid = 0.5 * (risk.longitudinal_distance_range_m[0] + risk.longitudinal_distance_range_m[1])
        gap_mid = 0.5 * (risk.gap_range_m[0] + risk.gap_range_m[1])
        if anchor_type == "ego_lane_front":
            x, y = gap_mid, lane_y
        elif anchor_type == "adjacent_lane":
            x, y = gap_mid, lane_y
        else:
            x, y = longitudinal_mid, lane_y
        ctx.anchor = {"anchor_type": anchor_type, "lane_context": lane_context, "x": float(x), "y": float(y), "lane_y": float(lane_y), "ego_speed": float(ego_speed)}
        ctx.notes.append(f"select_road_anchor: {ctx.anchor}")
        return scene, ctx, {"op": "select_road_anchor", "anchor": dict(ctx.anchor)}

    def generate_or_adjust_road_geometry(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        layout = str(params.get("generated_road_layout", "none") or "none")
        if layout in {"", "none"}:
            return scene, ctx, {"op": "generate_or_adjust_road_geometry", "layout": layout, "num_lines": 0}

        lane_width = float(params.get("lane_width_m", ctx.spec.road_layer.lane_width_m))
        radius = float(params.get("generated_road_radius_m", ctx.spec.road_layer.generated_road_radius_m))
        scene.lines.mask[...] = False

        if layout == "unprotected_left_turn":
            line_specs = self._make_unprotected_left_turn_lines(lane_width=lane_width)
            ctx.anchor.update(
                {
                    "lane_y": 0.0,
                    "x": 14.0,
                    "y": 0.0,
                    "opposite_lane_y": float(lane_width),
                    "road_layout": layout,
                }
            )
        elif layout == "roundabout_entry":
            line_specs = self._make_roundabout_entry_lines(lane_width=lane_width, radius=radius)
            ctx.anchor.update(
                {
                    "lane_y": 0.0,
                    "x": max(6.0, radius - 4.0),
                    "y": 0.0,
                    "roundabout_center_x": float(radius + 4.0),
                    "roundabout_center_y": 0.0,
                    "roundabout_radius": float(radius),
                    "ego_route_goal_x": float(radius + 4.0 - lane_width),
                    "ego_route_goal_y": float(-radius - 4.5),
                    "road_layout": layout,
                }
            )
        else:
            return scene, ctx, {"op": "generate_or_adjust_road_geometry", "layout": layout, "num_lines": 0, "skipped": True}

        max_slots = int(scene.lines.states.shape[0])
        written = 0
        for slot, points in enumerate(line_specs[:max_slots]):
            self._write_polyline(scene.lines, slot, np.asarray(points, dtype=np.float32))
            written += 1

        ctx.extra["generated_road_layout"] = layout
        ctx.extra["generated_road_num_lines"] = int(written)
        ctx.notes.append(f"generate_or_adjust_road_geometry: layout={layout}, lines={written}")
        return scene, ctx, {"op": "generate_or_adjust_road_geometry", "layout": layout, "num_lines": int(written)}

    def select_or_spawn_actor(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        actor = params.get("primary_actor", ctx.spec.actor_layer.primary_actor)
        role = params.get("actor_role", ctx.spec.actor_layer.actor_role)
        prefer_existing = bool(params.get("prefer_existing_actor", False))
        if actor in {"pedestrian", "person", "walker"}:
            elem_name, elem = "pedestrians", scene.pedestrians
        elif actor in {"vehicle", "lead_vehicle", "cutin_vehicle", "rear_vehicle", "cyclist"}:
            elem_name, elem = "vehicles", scene.vehicles
        elif actor == "static_obstacle":
            elem_name, elem = "static_objects", scene.static_objects
        else:
            raise ValueError(f"Unsupported primary_actor={actor}")
        idx = self._select_existing_or_allocate(elem, prefer_existing=prefer_existing)
        ctx.actor_elem_name = elem_name
        ctx.actor_index = int(idx)
        ctx.notes.append(f"select_or_spawn_actor: {actor}/{role} -> {elem_name}[{idx}]")
        return scene, ctx, {"op": "select_or_spawn_actor", "actor": actor, "role": role, "elem_name": elem_name, "index": int(idx)}

    def place_actor_laterally(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._crossing_direction_to_side_sign(direction)
        lane_y = float(ctx.anchor.get("lane_y", 0.0))
        longitudinal_range = params.get("longitudinal_distance_range_m", ctx.spec.risk_layer.longitudinal_distance_range_m)
        lateral_gap_range = params.get("lateral_gap_range_m", ctx.spec.risk_layer.lateral_gap_range_m)
        x = 0.5 * (float(longitudinal_range[0]) + float(longitudinal_range[1]))
        lateral_gap = 0.5 * (float(lateral_gap_range[0]) + float(lateral_gap_range[1]))
        lane_half_width = 1.8
        y = lane_y + side_sign * (lane_half_width + lateral_gap + 0.8)
        x, y = self._resolve_overlap(scene, x, y, ignore=(ctx.actor_elem_name, idx))
        width, length = self._default_size_for_elem(ctx.actor_elem_name, ctx.spec.actor_layer.primary_actor)
        heading = -side_sign * math.pi / 2.0
        speed = max(0.1, float(ctx.spec.risk_layer.target_actor_speed_mps))
        self._set_agent_state(elem, idx, x=x, y=y, heading=heading, width=width, length=length, velocity=speed)
        ctx.anchor["x"], ctx.anchor["y"] = float(x), float(lane_y)
        ctx.extra["conflict_lane_y"] = float(lane_y)
        ctx.notes.append(f"place_actor_laterally: {ctx.actor_elem_name}[{idx}] at ({x:.2f}, {y:.2f})")
        return scene, ctx, {"op": "place_actor_laterally", "index": int(idx), "x": float(x), "y": float(y), "direction": direction}

    def set_lateral_or_crossing_motion(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._crossing_direction_to_side_sign(direction)
        speed = float(params.get("target_actor_speed_mps", ctx.spec.risk_layer.target_actor_speed_mps))
        heading = -side_sign * math.pi / 2.0
        elem.states[idx, AgentIndex.HEADING] = heading
        elem.states[idx, AgentIndex.VELOCITY] = max(0.1, speed)
        elem.mask[idx] = True
        return scene, ctx, {"op": "set_lateral_or_crossing_motion", "index": int(idx), "heading": float(heading), "speed": float(max(0.1, speed)), "direction": direction}

    def place_actor_longitudinally(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        """Place a lead vehicle in front of ego with a real bumper-to-bumper gap.

        Earlier code treated ``gap_range_m`` as the *center x* of the lead
        vehicle.  For hard-brake scenes this is unsafe because a 4.8m lead
        vehicle at x=5--8m can leave only a very small bumper gap, and the
        planner may immediately overlap/pass through the lead vehicle.

        This implementation interprets ``gap_range_m`` as bumper-to-bumper
        headway, derives a center position using ego/lead half lengths, and
        additionally enforces a TTC-consistent minimum gap for high-speed ego
        cases.  The generated scene remains dangerous, but should not start
        as an unavoidable overlap.
        """
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index

        if ctx.actor_elem_name != "vehicles":
            raise ValueError("place_actor_longitudinally is intended for vehicle-like lead actors.")

        risk = ctx.spec.risk_layer
        inter = ctx.spec.interaction_layer
        risk_level = str(risk.risk_level or "moderate")

        width, length = self._default_size_for_elem(ctx.actor_elem_name, ctx.spec.actor_layer.primary_actor)
        ego_length = DEFAULT_EGO_LENGTH_M
        ego_front = ego_length / 2.0
        lead_half = length / 2.0

        ego_speed = float(ctx.anchor.get("ego_speed", self._estimate_ego_speed(scene)))
        lead_speed = self._estimate_longitudinal_lead_speed(
            ego_speed=ego_speed,
            relation=str(inter.speed_relation),
            risk_level=risk_level,
            target_actor_speed=getattr(risk, "target_actor_speed_mps", None),
        )

        gap_range = params.get("gap_range_m", risk.gap_range_m)
        gap_mid = 0.5 * (float(gap_range[0]) + float(gap_range[1]))

        # If the JSON provides TTC, enforce enough distance so ego does not
        # immediately run through the lead vehicle.
        ttc_range = getattr(risk, "ttc_range_s", None)
        if ttc_range is not None and len(ttc_range) >= 2:
            target_ttc = 0.5 * (float(ttc_range[0]) + float(ttc_range[1]))
        else:
            target_ttc = {"mild": 3.2, "moderate": 2.4, "aggressive": 1.8}.get(risk_level, 2.4)

        closing_speed = max(ego_speed - lead_speed, 0.5)
        ttc_gap = closing_speed * target_ttc

        # Hard-brake should be challenging, not physically impossible.
        min_gap_by_severity = {"mild": 12.0, "moderate": 9.0, "aggressive": 7.5}.get(risk_level, 9.0)
        desired_bumper_gap = max(gap_mid, ttc_gap, min_gap_by_severity)

        # Convert bumper gap to lead-center x.
        desired_x = ego_front + lead_half + desired_bumper_gap
        y = float(ctx.anchor.get("lane_y", 0.0))

        # Remove vehicles that would occupy the same lead-car corridor.
        removed = self._clear_conflicting_same_lane_vehicle(
            scene.vehicles,
            target_x=desired_x,
            lane_y=y,
            ignore_index=int(idx),
        )
        ctx.removed_vehicle_indices.extend(removed)

        # Try only forward shifts first.  Moving the lead backwards can create
        # ego overlap, so we avoid negative jitter for longitudinal conflicts.
        candidate_xs = [desired_x, desired_x + 1.5, desired_x + 3.0, desired_x + 5.0, desired_x + 7.0]
        chosen_x, chosen_y = float(candidate_xs[-1]), float(y)
        for cand_x in candidate_xs:
            if self._is_clear(scene, cand_x, y, ignore=(ctx.actor_elem_name, idx), radius=3.2):
                # Explicit bumper-gap guard against ego overlap.
                bumper_gap = cand_x - ego_front - lead_half
                if bumper_gap >= 6.0:
                    chosen_x, chosen_y = float(cand_x), float(y)
                    break

        self._set_agent_state(
            elem,
            idx,
            x=chosen_x,
            y=chosen_y,
            heading=0.0,
            width=width,
            length=length,
            velocity=lead_speed,
        )

        actual_gap = chosen_x - ego_front - lead_half
        actual_ttc = actual_gap / max(ego_speed - lead_speed, 0.5)
        ctx.anchor["x"], ctx.anchor["y"] = float(chosen_x), float(chosen_y)
        ctx.anchor["lead_speed"] = float(lead_speed)
        ctx.anchor["lead_bumper_gap"] = float(actual_gap)
        ctx.anchor["lead_ttc"] = float(actual_ttc)
        ctx.slowed_vehicle_indices.append(int(idx))
        ctx.notes.append(
            f"place_actor_longitudinally: {ctx.actor_elem_name}[{idx}] center=({chosen_x:.2f}, {chosen_y:.2f}), "
            f"bumper_gap={actual_gap:.2f}m, lead_speed={lead_speed:.2f}m/s, ttc={actual_ttc:.2f}s, removed={removed}"
        )
        return scene, ctx, {
            "op": "place_actor_longitudinally",
            "index": int(idx),
            "x": float(chosen_x),
            "y": float(chosen_y),
            "lead_speed": float(lead_speed),
            "bumper_gap": float(actual_gap),
            "ttc": float(actual_ttc),
            "removed": removed,
        }

    def set_speed_relation(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        relation = params.get("speed_relation", ctx.spec.interaction_layer.speed_relation)
        ego_speed = float(ctx.anchor.get("ego_speed", self._estimate_ego_speed(scene)))
        risk_level = ctx.spec.risk_layer.risk_level

        if relation in {"slow_lead", "stopped"}:
            # Prefer the speed already computed by place_actor_longitudinally,
            # so the lead position and TTC remain consistent.
            if "lead_speed" in ctx.anchor:
                speed = float(ctx.anchor["lead_speed"])
            else:
                speed = self._estimate_longitudinal_lead_speed(
                    ego_speed=ego_speed,
                    relation=str(relation),
                    risk_level=str(risk_level),
                    target_actor_speed=getattr(ctx.spec.risk_layer, "target_actor_speed_mps", None),
                )
            ctx.slowed_vehicle_indices.append(int(idx))
        elif relation == "fast_approach":
            speed = ego_speed + float(ctx.spec.risk_layer.target_relative_speed_mps)
        else:
            speed = ego_speed

        if AgentIndex.VELOCITY < elem.states.shape[1]:
            elem.states[idx, AgentIndex.VELOCITY] = float(speed)
        elem.mask[idx] = True
        return scene, ctx, {"op": "set_speed_relation", "relation": relation, "speed": float(speed)}

    def place_oncoming_actor(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        if ctx.actor_elem_name != "vehicles":
            raise ValueError("place_oncoming_actor is intended for vehicle actors.")

        risk = ctx.spec.risk_layer
        distance_range = params.get("longitudinal_distance_range_m", risk.longitudinal_distance_range_m)
        display_x = max(10.0, 0.5 * (float(distance_range[0]) + float(distance_range[1])))
        y = float(ctx.anchor.get("opposite_lane_y", ctx.spec.road_layer.lane_width_m))
        width, length = self._default_size_for_elem(ctx.actor_elem_name, ctx.spec.actor_layer.primary_actor)
        speed = max(0.1, float(risk.target_actor_speed_mps))
        heading = math.pi

        frame0_offset_s = float(params.get("frame0_time_offset_s", SLEDGEBOARD_FRAME0_OFFSET_S))
        compensate_frame0 = bool(params.get("compensate_frame0_offset", True))
        if compensate_frame0:
            raw_x, raw_y = self._display_to_raw_position(display_x, y, heading, speed, frame0_offset_s)
        else:
            raw_x, raw_y = display_x, y
        if compensate_frame0 and raw_x > 31.0:
            raw_x = 31.0
            display_x = raw_x + speed * math.cos(heading) * frame0_offset_s

        # Keep the opposing lane clear around the injected actor so it remains
        # easy to inspect in SledgeBoard and does not start in collision.
        removed = []
        states = np.asarray(scene.vehicles.states)
        valid = np.asarray(scene.vehicles.mask).astype(bool)
        for other_idx in np.where(valid)[0]:
            if int(other_idx) == int(idx):
                continue
            ox, oy = float(states[other_idx, AgentIndex.X]), float(states[other_idx, AgentIndex.Y])
            if abs(oy - y) < 1.5 and min(abs(ox - raw_x), abs(ox - display_x)) < 7.0:
                scene.vehicles.mask[other_idx] = False
                removed.append(int(other_idx))

        self._set_agent_state(elem, idx, x=raw_x, y=raw_y, heading=heading, width=width, length=length, velocity=speed)
        ctx.anchor["x"], ctx.anchor["y"] = float(display_x), 0.0
        ctx.anchor["opposing_speed"] = float(speed)
        ctx.extra["semantic_validation_time_offset_s"] = frame0_offset_s
        ctx.extra["use_projected_validation"] = compensate_frame0
        ctx.extra["oncoming_display_x"] = float(display_x)
        ctx.extra["oncoming_display_y"] = float(y)
        ctx.removed_vehicle_indices.extend(removed)
        ctx.notes.append(
            f"place_oncoming_actor: vehicles[{idx}] raw=({raw_x:.2f}, {raw_y:.2f}), "
            f"display=({display_x:.2f}, {y:.2f}), speed={speed:.2f}"
        )
        return scene, ctx, {
            "op": "place_oncoming_actor",
            "index": int(idx),
            "x": float(raw_x),
            "y": float(raw_y),
            "display_x": float(display_x),
            "display_y": float(y),
            "speed": float(speed),
            "removed": removed,
        }

    def set_oncoming_motion(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        target_speed = params.get("target_actor_speed_mps", ctx.spec.risk_layer.target_actor_speed_mps)
        speed = max(0.1, float(target_speed))
        elem.states[idx, AgentIndex.HEADING] = math.pi
        elem.states[idx, AgentIndex.VELOCITY] = speed
        elem.mask[idx] = True
        return scene, ctx, {"op": "set_oncoming_motion", "index": int(idx), "heading": float(math.pi), "speed": float(speed)}

    def place_actor_for_merging(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._merge_direction_to_side_sign(direction)
        gap_range = params.get("gap_range_m", ctx.spec.risk_layer.gap_range_m)
        intrusion_range = params.get("lateral_intrusion_range_m", ctx.spec.risk_layer.lateral_intrusion_range_m)
        layout = str(ctx.extra.get("generated_road_layout", ""))
        if layout == "roundabout_entry":
            roundabout_center_x = float(ctx.anchor.get("roundabout_center_x", ctx.spec.road_layer.generated_road_radius_m + 4.0))
            roundabout_radius = float(ctx.anchor.get("roundabout_radius", ctx.spec.road_layer.generated_road_radius_m))
            x = max(6.0, min(0.5 * (float(gap_range[0]) + float(gap_range[1])), roundabout_center_x - roundabout_radius + 3.2))
        else:
            x = 0.5 * (float(gap_range[0]) + float(gap_range[1]))
        intrusion = 0.5 * (float(intrusion_range[0]) + float(intrusion_range[1]))
        width, length = self._default_size_for_elem(ctx.actor_elem_name, ctx.spec.actor_layer.primary_actor)
        lane_half_width = 0.5 * float(ctx.spec.road_layer.lane_width_m)
        center_abs_y = max(0.15, lane_half_width + 0.5 * width - intrusion)
        y = side_sign * center_abs_y
        x, y = self._resolve_overlap(scene, x, y, ignore=(ctx.actor_elem_name, idx), radius=2.2)
        ego_speed = float(ctx.anchor.get("ego_speed", self._estimate_ego_speed(scene)))
        speed = float(np.clip(ego_speed + 1.0, 2.5, 15.0))
        heading = -side_sign * 0.08
        removed = self._clear_conflicting_same_lane_vehicle(scene.vehicles, target_x=x, lane_y=0.0, ignore_index=idx)
        ctx.removed_vehicle_indices.extend(removed)
        self._set_agent_state(elem, idx, x=x, y=y, heading=heading, width=width, length=length, velocity=speed)
        ctx.anchor["x"], ctx.anchor["y"] = float(x), 0.0
        ctx.notes.append(f"place_actor_for_merging: {ctx.actor_elem_name}[{idx}] at ({x:.2f}, {y:.2f})")
        return scene, ctx, {"op": "place_actor_for_merging", "index": int(idx), "x": float(x), "y": float(y), "direction": direction, "intrusion": float(intrusion), "removed": removed}

    def set_merging_motion(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._merge_direction_to_side_sign(direction)
        heading = -side_sign * 0.08
        elem.states[idx, AgentIndex.HEADING] = heading
        elem.mask[idx] = True
        return scene, ctx, {"op": "set_merging_motion", "index": int(idx), "heading": float(heading)}

    def add_roundabout_cross_traffic(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        """Populate a generated roundabout with traffic from all four approaches."""
        if str(ctx.extra.get("generated_road_layout", "")) != "roundabout_entry":
            return scene, ctx, {"op": "add_roundabout_cross_traffic", "skipped": True, "reason": "not_roundabout_entry"}

        cx = float(ctx.anchor.get("roundabout_center_x", ctx.spec.road_layer.generated_road_radius_m + 4.0))
        cy = float(ctx.anchor.get("roundabout_center_y", 0.0))
        radius = float(ctx.anchor.get("roundabout_radius", ctx.spec.road_layer.generated_road_radius_m))
        lane_width = float(ctx.spec.road_layer.lane_width_m)
        speed = max(2.0, min(float(ctx.spec.risk_layer.target_actor_speed_mps), 8.5))
        width, length = self._default_size_for_elem("vehicles", "vehicle")

        traffic_specs = [
            # Circulating traffic bearing down on the ego entry.
            ("circulating_yield_threat", [(cx - radius + 2.0, cy + 5.8), (cx - radius + 3.2, cy + 7.2)], -math.pi / 2.0, speed),
            ("circulating_platoon", [(cx - radius + 6.8, cy + 10.2), (cx - radius + 8.4, cy + 10.8)], -0.65 * math.pi, max(2.0, speed - 1.0)),
            # Vehicles entering from all four arms.
            ("west_entry_queue", [(7.2, 0.0), (8.6, -0.2), (10.0, 0.2)], 0.0, 0.5),
            ("north_entry_vehicle", [(cx, cy + radius + 8.0), (cx, cy + radius + 11.0)], -math.pi / 2.0, max(2.0, speed - 2.0)),
            ("east_entry_vehicle", [(cx + radius + 9.0, cy), (cx + radius + 12.0, cy)], math.pi, max(2.0, speed - 1.5)),
            ("south_entry_vehicle", [(cx, cy - radius - 9.5), (cx, cy - radius - 12.5)], math.pi / 2.0, max(2.0, speed - 1.5)),
            # Four outbound vehicles make the exits visually usable.
            ("west_exit_vehicle", [(cx - radius - 7.0, cy + lane_width), (cx - radius - 10.0, cy + lane_width)], math.pi, max(2.0, speed - 2.5)),
            ("east_exit_vehicle", [(cx + radius + 7.0, cy - lane_width), (cx + radius + 10.0, cy - lane_width)], 0.0, max(2.0, speed - 2.5)),
            ("north_exit_vehicle", [(cx + lane_width, cy + radius + 7.0), (cx + lane_width, cy + radius + 10.0)], math.pi / 2.0, max(2.0, speed - 2.5)),
            ("south_exit_vehicle", [(cx - lane_width, cy - radius - 7.0), (cx - lane_width, cy - radius - 10.0)], -math.pi / 2.0, max(2.0, speed - 2.5)),
        ]

        placed: List[Dict[str, Any]] = []
        for role, candidates, heading, vehicle_speed in traffic_specs:
            idx = self._select_existing_or_allocate(scene.vehicles, prefer_existing=False)
            placed_xy = self._try_place_vehicle_from_candidates(
                scene=scene,
                idx=idx,
                candidates=candidates,
                heading=heading,
                width=width,
                length=length,
                velocity=vehicle_speed,
                ignore=[("vehicles", idx)],
            )
            if placed_xy is None:
                scene.vehicles.mask[idx] = False
                continue
            placed.append(
                {
                    "role": role,
                    "index": int(idx),
                    "x": float(placed_xy[0]),
                    "y": float(placed_xy[1]),
                    "heading": float(heading),
                    "speed": float(vehicle_speed),
                }
            )

        ctx.extra["roundabout_cross_traffic"] = placed
        ctx.anchor["x"] = float(ctx.anchor.get("ego_route_goal_x", cx + radius + lane_width + 14.0))
        ctx.anchor["y"] = float(ctx.anchor.get("ego_route_goal_y", 0.0))
        ctx.notes.append(f"add_roundabout_cross_traffic: placed={len(placed)}")
        return scene, ctx, {
            "op": "add_roundabout_cross_traffic",
            "placed": placed,
            "num_placed": len(placed),
            "ego_route_goal": [float(ctx.anchor["x"]), float(ctx.anchor["y"])],
        }

    def place_static_or_slow_actor_as_blocker(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        if ctx.actor_elem_name != "static_objects":
            idx = self._select_existing_or_allocate(scene.static_objects, prefer_existing=False)
            ctx.actor_elem_name, ctx.actor_index = "static_objects", int(idx)
        elem = self._get_ctx_elem(scene, ctx)
        idx = ctx.actor_index
        gap_range = params.get("gap_range_m", ctx.spec.risk_layer.gap_range_m)
        x = 0.5 * (float(gap_range[0]) + float(gap_range[1]))
        y = float(ctx.anchor.get("lane_y", 0.0))
        x, y = self._resolve_overlap(scene, x, y, ignore=(ctx.actor_elem_name, idx), radius=1.8)
        self._set_agent_state(elem, idx, x=x, y=y, heading=0.0, width=1.8, length=3.0, velocity=0.0)
        ctx.static_obstacle_index = int(idx)
        ctx.anchor["x"], ctx.anchor["y"] = float(x), float(y)
        ctx.notes.append(f"place_static_or_slow_actor_as_blocker: static_objects[{idx}] at ({x:.2f}, {y:.2f})")
        return scene, ctx, {"op": "place_static_or_slow_actor_as_blocker", "index": int(idx), "x": float(x), "y": float(y)}

    def add_or_select_static_obstacle(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        if ctx.static_obstacle_index >= 0: idx = ctx.static_obstacle_index
        else:
            idx = self._select_existing_or_allocate(scene.static_objects, prefer_existing=False)
            ctx.static_obstacle_index = int(idx)
        elem = scene.static_objects
        current = elem.states[idx]
        x = float(current[AgentIndex.X]) if elem.mask[idx] else float(ctx.anchor.get("x", 8.0))
        y = float(current[AgentIndex.Y]) if elem.mask[idx] else float(ctx.anchor.get("y", 0.0))
        obstacle_size = params.get("obstacle_size", "normal")
        if obstacle_size == "large": width, length = 1.8, 3.5
        elif obstacle_size == "small": width, length = 0.8, 0.8
        else: width, length = 1.2, 2.0
        self._set_agent_state(elem, idx, x=x, y=y, heading=0.0, width=width, length=length, velocity=0.0)
        ctx.notes.append(f"add_or_select_static_obstacle: static_objects[{idx}] type={params.get('obstacle_type')}")
        return scene, ctx, {"op": "add_or_select_static_obstacle", "index": int(idx), "obstacle_type": params.get("obstacle_type"), "x": float(x), "y": float(y)}

    def add_or_select_occluder(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        """
        Add an occluder with geometry-aware placement.

        The old implementation used ox=0.55*ped_x and heading=0 for all
        occluding vehicles. For an aggressive pedestrian case this can put a
        5m parked vehicle too close to the ego box, so the vehicle exists in
        sledge_vector.gz but disappears from SimulationLog after observation
        filtering. This implementation jointly solves pedestrian distance,
        occluder size, occluder layer, and ego-overlap avoidance.
        """
        if ctx.actor_index < 0:
            raise ValueError("add_or_select_occluder requires a selected primary actor first.")

        requested_type = params.get("occluder_type", ctx.spec.object_layer.occlusion.occluder_type)
        candidate_types = self._occluder_candidate_types(requested_type)

        last_error: Optional[Exception] = None
        for occ_type in candidate_types:
            spec = self._get_occluder_spec(occ_type)
            elem = self._elem_by_name(scene, spec.elem_name)

            # Prefer slot 0 for newly injected occluders. This makes debug output
            # stable: a vehicle occluder should appear as track token 0_0 in
            # SimulationLog if it survives the observation pipeline.
            idx = self._select_preferred_or_allocate(elem, preferred_index=0, prefer_existing=False)

            try:
                layout = self._plan_occluded_actor_layout(
                    scene=scene,
                    ctx=ctx,
                    occluder_spec=spec,
                    occluder_index=idx,
                    params=params,
                )
            except RuntimeError as exc:
                elem.mask[idx] = False
                last_error = exc
                continue

            actor_elem = self._get_ctx_elem(scene, ctx)
            actor_raw = layout["actor_raw"]
            self._set_agent_state(
                actor_elem,
                ctx.actor_index,
                x=actor_raw["x"],
                y=actor_raw["y"],
                heading=actor_raw["heading"],
                width=actor_raw["width"],
                length=actor_raw["length"],
                velocity=actor_raw["velocity"],
            )

            occ_raw = layout["occluder_raw"]
            self._set_agent_state(
                elem,
                idx,
                x=occ_raw["x"],
                y=occ_raw["y"],
                heading=occ_raw["heading"],
                width=occ_raw["width"],
                length=occ_raw["length"],
                velocity=occ_raw["velocity"],
            )

            ctx.occluder_index = int(idx)
            ctx.occluder_elem_name = spec.elem_name
            if spec.elem_name == "static_objects":
                ctx.static_obstacle_index = int(idx)

            # Store the frame-0 display layout so validators and reports use the
            # same semantics as SledgeBoard/SimulationLog frame 0.
            ctx.extra["occluder_layout"] = layout
            ctx.extra["semantic_validation_time_offset_s"] = float(
                params.get("frame0_time_offset_s", SLEDGEBOARD_FRAME0_OFFSET_S)
            )
            ctx.extra["use_projected_validation"] = bool(params.get("compensate_frame0_offset", True))

            actor_display = layout["actor_display"]
            ctx.anchor["x"] = float(actor_display["x"])
            ctx.anchor["y"] = float(ctx.anchor.get("lane_y", 0.0))
            ctx.extra["conflict_lane_y"] = float(ctx.anchor.get("lane_y", 0.0))

            ctx.notes.append(
                "add_or_select_occluder: "
                f"{spec.elem_name}[{idx}] type={spec.name} "
                f"raw=({occ_raw['x']:.2f},{occ_raw['y']:.2f}) "
                f"display=({layout['occluder_display']['x']:.2f},{layout['occluder_display']['y']:.2f}); "
                f"actor display=({actor_display['x']:.2f},{actor_display['y']:.2f})"
            )

            return scene, ctx, {
                "op": "add_or_select_occluder",
                "index": int(idx),
                "occluder_type": spec.name,
                "occluder_elem_name": spec.elem_name,
                "x": float(occ_raw["x"]),
                "y": float(occ_raw["y"]),
                "display_x": float(layout["occluder_display"]["x"]),
                "display_y": float(layout["occluder_display"]["y"]),
                "actor_display_x": float(actor_display["x"]),
                "actor_display_y": float(actor_display["y"]),
            }

        raise RuntimeError(f"Failed to place occluder safely for requested_type={requested_type}: {last_error}")

    def apply_risk_scaling(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        return scene, ctx, {"op": "apply_risk_scaling", "risk_level": params.get("risk_level"), "ttc_range_s": params.get("ttc_range_s"), "gap_range_m": params.get("gap_range_m")}

    def build_protection_rois(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        protected_targets = set(params.get("protected_targets", []))
        ctx.rois.clear()
        if "primary_actor" in protected_targets and ctx.actor_index >= 0:
            elem = self._get_ctx_elem(scene, ctx)
            ctx.rois.append(self._roi_from_state(elem.states[ctx.actor_index], "primary_actor"))
        if "secondary_actor" in protected_targets and ctx.occluder_index >= 0:
            occ_state = self._get_occluder_state(scene, ctx)
            if occ_state is not None:
                ctx.rois.append(self._roi_from_state(occ_state, "occluder"))
        if "static_obstacle" in protected_targets and ctx.static_obstacle_index >= 0:
            ctx.rois.append(self._roi_from_state(scene.static_objects.states[ctx.static_obstacle_index], "static_obstacle"))
        if "conflict_corridor" in protected_targets and ctx.actor_index >= 0:
            elem = self._get_ctx_elem(scene, ctx)
            actor_state = elem.states[ctx.actor_index]
            x, y = float(actor_state[AgentIndex.X]), float(actor_state[AgentIndex.Y])
            ax, ay = float(ctx.anchor.get("x", x)), float(ctx.anchor.get("y", 0.0))
            ctx.rois.append(SceneEditROI(x_min=min(x, ax) - 1.5, y_min=min(y, ay) - 1.5, x_max=max(x, ax) + 1.5, y_max=max(y, ay) + 1.5, tag="conflict_corridor"))
        if "road_anchor" in protected_targets:
            x, y = float(ctx.anchor.get("x", 0.0)), float(ctx.anchor.get("y", 0.0))
            ctx.rois.append(SceneEditROI(x_min=x - 2.0, y_min=y - 2.0, x_max=x + 2.0, y_max=y + 2.0, tag="road_anchor"))
        return scene, ctx, {"op": "build_protection_rois", "num_rois": len(ctx.rois)}

    def validate_semantic_constraints(self, scene: SledgeVectorRaw, ctx, params: Dict[str, Any]):
        from sledge.semantic_control.semantic_validator import validate_scene_against_spec
        ctx.validation = validate_scene_against_spec(scene, ctx, params)
        return scene, ctx, {"op": "validate_semantic_constraints", "validation": ctx.validation}


    def _plan_occluded_actor_layout(
        self,
        scene: SledgeVectorRaw,
        ctx,
        occluder_spec: OccluderSpec,
        occluder_index: int,
        params: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Plan raw and SledgeBoard-frame0 positions for an occluded actor."""
        actor_elem = self._get_ctx_elem(scene, ctx)
        actor_state = actor_elem.states[ctx.actor_index]
        lane_y = float(ctx.anchor.get("lane_y", 0.0))
        risk = ctx.spec.risk_layer
        direction = params.get("direction", ctx.spec.interaction_layer.conflict_direction)
        side_sign = self._crossing_direction_to_side_sign(direction)

        frame0_offset_s = float(params.get("frame0_time_offset_s", SLEDGEBOARD_FRAME0_OFFSET_S))
        compensate_frame0 = bool(params.get("compensate_frame0_offset", True))

        # Dynamic pedestrian distance. Larger occluders require more room in
        # front of ego. If the spec requests a farther distance, keep it.
        occ_heading = float(params.get("occluder_heading", occluder_spec.default_heading))
        occ_hx = self._half_extent_x(occluder_spec.length, occluder_spec.width, occ_heading)
        ego_front_x = DEFAULT_EGO_LENGTH_M / 2.0
        ego_margin = float(params.get("ego_clearance_m", 1.0))
        actor_margin = float(params.get("actor_clearance_m", 0.8))
        min_actor_x = ego_front_x + ego_margin + 2.0 * occ_hx + actor_margin

        risk_mid_x = 0.5 * (
            float(risk.longitudinal_distance_range_m[0]) + float(risk.longitudinal_distance_range_m[1])
        )
        severity = str(risk.risk_level)
        severity_floor = {"mild": 13.0, "moderate": 11.0, "aggressive": 9.5}.get(severity, 10.5)
        actor_display_x = max(float(params.get("actor_display_x_m", risk_mid_x)), min_actor_x, severity_floor)

        lane_half_width = 0.5 * float(ctx.spec.road_layer.lane_width_m)
        lateral_gap = 0.5 * (float(risk.lateral_gap_range_m[0]) + float(risk.lateral_gap_range_m[1]))
        actor_display_y = float(
            params.get(
                "actor_display_y_m",
                lane_y + side_sign * (lane_half_width + lateral_gap + 0.75),
            )
        )

        actor_heading = float(params.get("actor_heading", -side_sign * math.pi / 2.0))
        actor_speed = max(0.1, float(params.get("target_actor_speed_mps", risk.target_actor_speed_mps)))
        actor_width = float(max(actor_state[AgentIndex.WIDTH], 0.75))
        actor_length = float(max(actor_state[AgentIndex.LENGTH], 0.75))

        if compensate_frame0:
            actor_raw_x, actor_raw_y = self._display_to_raw_position(
                actor_display_x, actor_display_y, actor_heading, actor_speed, frame0_offset_s
            )
        else:
            actor_raw_x, actor_raw_y = actor_display_x, actor_display_y

        actor_raw = {
            "x": float(actor_raw_x),
            "y": float(actor_raw_y),
            "heading": actor_heading,
            "width": actor_width,
            "length": actor_length,
            "velocity": actor_speed,
        }
        actor_display = {
            "x": float(actor_display_x),
            "y": float(actor_display_y),
            "heading": actor_heading,
            "width": actor_width,
            "length": actor_length,
            "velocity": actor_speed,
        }

        occupied = self._collect_occupied_boxes(
            scene,
            ignore=[(ctx.actor_elem_name, ctx.actor_index), (occluder_spec.elem_name, occluder_index)],
            include_ego=True,
            use_display_time=False,
        )
        actor_box_raw = self._make_aabb_from_values(
            actor_raw["x"], actor_raw["y"], actor_raw["heading"], actor_width, actor_length, margin=0.25
        )
        if self._box_overlaps_any(actor_box_raw, occupied):
            # The actor itself should not be spawned into another object. Try a
            # farther x before failing.
            actor_display_x += 1.5
            if compensate_frame0:
                actor_raw_x, actor_raw_y = self._display_to_raw_position(
                    actor_display_x, actor_display_y, actor_heading, actor_speed, frame0_offset_s
                )
            else:
                actor_raw_x, actor_raw_y = actor_display_x, actor_display_y
            actor_raw["x"], actor_raw["y"] = float(actor_raw_x), float(actor_raw_y)
            actor_display["x"] = float(actor_display_x)

        # Place the occluder along the ego->actor display line. The occluder is
        # static, so raw and display positions are the same unless a nonzero
        # occluder velocity is explicitly provided.
        ratio_candidates = [0.55, 0.50, 0.60, 0.45, 0.65, 0.70]
        x_offsets = [0.0, 0.6, -0.6, 1.2, -1.2, 2.0]
        y_offsets = [0.0, -0.35, 0.35, -0.7, 0.7]

        for extra_actor_x in [0.0, 1.0, 2.0, 3.5, 5.0]:
            cur_actor_display_x = actor_display_x + extra_actor_x
            cur_actor_display = dict(actor_display)
            cur_actor_display["x"] = float(cur_actor_display_x)
            cur_actor_raw = dict(actor_raw)
            if extra_actor_x != 0.0:
                if compensate_frame0:
                    arx, ary = self._display_to_raw_position(
                        cur_actor_display_x, actor_display_y, actor_heading, actor_speed, frame0_offset_s
                    )
                else:
                    arx, ary = cur_actor_display_x, actor_display_y
                cur_actor_raw["x"], cur_actor_raw["y"] = float(arx), float(ary)

            for ratio in ratio_candidates:
                base_x = ratio * cur_actor_display_x
                base_y = ratio * actor_display_y
                for dx in x_offsets:
                    for dy in y_offsets:
                        occ_display_x = base_x + dx
                        occ_display_y = base_y + dy
                        if not (ego_front_x + 0.5 < occ_display_x < cur_actor_display_x - 0.5):
                            continue

                        occ_display = {
                            "x": float(occ_display_x),
                            "y": float(occ_display_y),
                            "heading": occ_heading,
                            "width": float(occluder_spec.width),
                            "length": float(occluder_spec.length),
                            "velocity": float(occluder_spec.velocity),
                        }
                        if compensate_frame0:
                            orx, ory = self._display_to_raw_position(
                                occ_display_x,
                                occ_display_y,
                                occ_heading,
                                occluder_spec.velocity,
                                frame0_offset_s,
                            )
                        else:
                            orx, ory = occ_display_x, occ_display_y
                        occ_raw = dict(occ_display)
                        occ_raw["x"], occ_raw["y"] = float(orx), float(ory)

                        raw_occ_box = self._make_aabb_from_values(
                            occ_raw["x"], occ_raw["y"], occ_heading, occluder_spec.width, occluder_spec.length, margin=0.30
                        )
                        raw_actor_box = self._make_aabb_from_values(
                            cur_actor_raw["x"], cur_actor_raw["y"], actor_heading, actor_width, actor_length, margin=0.20
                        )
                        display_occ_box = self._make_aabb_from_values(
                            occ_display_x, occ_display_y, occ_heading, occluder_spec.width, occluder_spec.length, margin=0.30
                        )
                        display_actor_box = self._make_aabb_from_values(
                            cur_actor_display_x, actor_display_y, actor_heading, actor_width, actor_length, margin=0.20
                        )
                        display_ego_box = self._make_aabb_from_values(
                            0.0, 0.0, 0.0, DEFAULT_EGO_WIDTH_M, DEFAULT_EGO_LENGTH_M, margin=0.30
                        )

                        if self._aabb_overlap(raw_occ_box, raw_actor_box):
                            continue
                        if self._aabb_overlap(display_occ_box, display_actor_box):
                            continue
                        if self._aabb_overlap(display_occ_box, display_ego_box):
                            continue
                        if self._box_overlaps_any(raw_occ_box, occupied):
                            continue

                        return {
                            "actor_raw": cur_actor_raw,
                            "actor_display": cur_actor_display,
                            "occluder_raw": occ_raw,
                            "occluder_display": occ_display,
                            "frame0_time_offset_s": frame0_offset_s,
                            "compensate_frame0_offset": compensate_frame0,
                        }

        raise RuntimeError(
            f"no collision-free occluder layout found for type={occluder_spec.name}, "
            f"actor_display=({actor_display_x:.2f},{actor_display_y:.2f})"
        )

    @staticmethod
    def _display_to_raw_position(display_x: float, display_y: float, heading: float, speed: float, time_offset_s: float) -> Tuple[float, float]:
        return (
            float(display_x - speed * math.cos(heading) * time_offset_s),
            float(display_y - speed * math.sin(heading) * time_offset_s),
        )

    @staticmethod
    def _half_extent_x(length: float, width: float, heading: float) -> float:
        return abs(math.cos(heading)) * length / 2.0 + abs(math.sin(heading)) * width / 2.0

    @staticmethod
    def _half_extent_y(length: float, width: float, heading: float) -> float:
        return abs(math.sin(heading)) * length / 2.0 + abs(math.cos(heading)) * width / 2.0

    def _make_aabb_from_values(self, x: float, y: float, heading: float, width: float, length: float, margin: float = 0.0) -> Tuple[float, float, float, float]:
        hx = self._half_extent_x(length, width, heading) + margin
        hy = self._half_extent_y(length, width, heading) + margin
        return float(x), float(y), float(hx), float(hy)

    @staticmethod
    def _aabb_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax, ay, ahx, ahy = a
        bx, by, bhx, bhy = b
        return abs(ax - bx) <= (ahx + bhx) and abs(ay - by) <= (ahy + bhy)

    def _box_overlaps_any(self, box: Tuple[float, float, float, float], others: List[Tuple[float, float, float, float]]) -> bool:
        return any(self._aabb_overlap(box, other) for other in others)

    def _collect_occupied_boxes(
        self,
        scene: SledgeVectorRaw,
        ignore: Optional[List[Tuple[str, int]]] = None,
        include_ego: bool = True,
        use_display_time: bool = False,
        time_offset_s: float = SLEDGEBOARD_FRAME0_OFFSET_S,
    ) -> List[Tuple[float, float, float, float]]:
        ignore_set = {(name, int(idx)) for name, idx in (ignore or [])}
        boxes: List[Tuple[float, float, float, float]] = []
        if include_ego:
            boxes.append(self._make_aabb_from_values(0.0, 0.0, 0.0, DEFAULT_EGO_WIDTH_M, DEFAULT_EGO_LENGTH_M, margin=0.30))

        for elem_name, elem in [("vehicles", scene.vehicles), ("pedestrians", scene.pedestrians), ("static_objects", scene.static_objects)]:
            valid = np.asarray(elem.mask).astype(bool)
            states = np.asarray(elem.states)
            for idx in np.where(valid)[0]:
                if (elem_name, int(idx)) in ignore_set:
                    continue
                state = np.asarray(states[idx], dtype=np.float32).copy()
                if use_display_time:
                    state = self._project_state(state, time_offset_s)
                boxes.append(
                    self._make_aabb_from_values(
                        float(state[AgentIndex.X]),
                        float(state[AgentIndex.Y]),
                        float(state[AgentIndex.HEADING]),
                        float(max(state[AgentIndex.WIDTH], 0.5)),
                        float(max(state[AgentIndex.LENGTH], 0.5)),
                        margin=0.20,
                    )
                )
        return boxes

    @staticmethod
    def _project_state(state: np.ndarray, time_s: float) -> np.ndarray:
        projected = np.asarray(state, dtype=np.float32).copy()
        speed = float(max(projected[AgentIndex.VELOCITY], 0.0))
        heading = float(projected[AgentIndex.HEADING])
        projected[AgentIndex.X] = float(projected[AgentIndex.X]) + speed * math.cos(heading) * time_s
        projected[AgentIndex.Y] = float(projected[AgentIndex.Y]) + speed * math.sin(heading) * time_s
        return projected

    @staticmethod
    def _normalize_occluder_type(occluder_type: Any) -> str:
        occ = str(occluder_type or "parked_vehicle").lower().strip()
        if occ in {"", "none"}:
            return "parked_vehicle"
        if occ == "auto":
            return "auto"
        if occ in _VEHICLE_OCCLUDER_ALIASES:
            return _VEHICLE_OCCLUDER_ALIASES[occ]
        if occ in _STATIC_OCCLUDER_ALIASES:
            if occ in {"construction", "cone", "roadblock"}:
                return "construction_object"
            if occ in {"barrier", "road_barrier"}:
                return "barrier"
            return "static_object"
        return "parked_vehicle"

    def _occluder_candidate_types(self, occluder_type: Any) -> List[str]:
        normalized = self._normalize_occluder_type(occluder_type)
        if normalized == "auto":
            # Deterministic fallback order: keep paper semantics as a parked
            # vehicle first, then fall back to smaller static occluders.
            return ["parked_vehicle", "truck", "barrier", "construction_object"]
        return [normalized]

    @staticmethod
    def _get_occluder_spec(occluder_type: str) -> OccluderSpec:
        normalized = PrimitiveOps._normalize_occluder_type(occluder_type)
        if normalized == "auto":
            normalized = "parked_vehicle"
        return OCCLUDER_SPECS.get(normalized, OCCLUDER_SPECS["parked_vehicle"])

    @staticmethod
    def _elem_by_name(scene: SledgeVectorRaw, elem_name: str) -> SledgeVectorElement:
        if elem_name == "vehicles":
            return scene.vehicles
        if elem_name == "pedestrians":
            return scene.pedestrians
        if elem_name == "static_objects":
            return scene.static_objects
        raise ValueError(f"Unsupported elem_name={elem_name}")

    @staticmethod
    def _select_preferred_or_allocate(elem: SledgeVectorElement, preferred_index: int = 0, prefer_existing: bool = False) -> int:
        valid = np.asarray(elem.mask).astype(bool)
        n = len(valid)
        if prefer_existing and np.any(valid):
            return int(np.where(valid)[0][0])
        if 0 <= preferred_index < n and not bool(valid[preferred_index]):
            elem.mask[preferred_index] = True
            return int(preferred_index)
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

    def _get_occluder_state(self, scene: SledgeVectorRaw, ctx) -> Optional[np.ndarray]:
        elem_name = getattr(ctx, "occluder_elem_name", "vehicles")
        if ctx.occluder_index < 0:
            return None
        elem = self._elem_by_name(scene, elem_name)
        if ctx.occluder_index >= len(elem.mask) or not bool(elem.mask[ctx.occluder_index]):
            return None
        return elem.states[ctx.occluder_index]

    @staticmethod
    def _select_existing_or_allocate(elem: SledgeVectorElement, prefer_existing: bool = False) -> int:
        valid = np.asarray(elem.mask).astype(bool)
        if prefer_existing and np.any(valid):
            states = np.asarray(elem.states)
            valid_indices = np.where(valid)[0]
            forward = [int(i) for i in valid_indices if float(states[i, AgentIndex.X]) > 0.0 and abs(float(states[i, AgentIndex.Y])) < 6.0]
            if forward: return int(forward[0])
            return int(valid_indices[0])
        invalid = np.where(~valid)[0]
        if len(invalid) > 0:
            idx = int(invalid[0]); elem.mask[idx] = True; return idx
        states = np.asarray(elem.states)
        distances = np.linalg.norm(states[:, :2], axis=1)
        idx = int(np.argmax(distances)); elem.mask[idx] = True; return idx

    @staticmethod
    def _estimate_longitudinal_lead_speed(
        ego_speed: float,
        relation: str,
        risk_level: str,
        target_actor_speed: Any = None,
    ) -> float:
        """Estimate a physically plausible lead-vehicle speed for hard-brake scenes.

        SLEDGE raw vectors only provide an instantaneous velocity, not an
        explicit braking trajectory.  Therefore hard-brake is represented as a
        slow/stopped lead vehicle proxy.  To avoid impossible scenes where ego
        instantly overlaps a stopped lead, aggressive cases still keep a small
        non-zero default speed unless the JSON explicitly requests zero.
        """
        try:
            if target_actor_speed is not None:
                requested = float(target_actor_speed)
                if requested >= 0.0:
                    return float(np.clip(requested, 0.0, max(ego_speed, 0.1)))
        except Exception:
            pass

        relation = str(relation or "slow_lead")
        risk_level = str(risk_level or "moderate")

        if relation == "stopped":
            # A true stopped lead is allowed, but not forced unless requested.
            return 0.0
        if risk_level == "aggressive":
            return float(max(1.5, ego_speed * 0.35))
        if risk_level == "moderate":
            return float(max(2.0, ego_speed * 0.50))
        return float(max(2.5, ego_speed * 0.65))

    @staticmethod
    def _set_agent_state(
        elem: SledgeVectorElement,
        idx: int,
        x: float,
        y: float,
        heading: float,
        width: float,
        length: float,
        velocity: float = 0.0,
    ) -> None:
        """Set a SLEDGE element state safely.

        vehicles / pedestrians use the agent state layout, which includes VELOCITY.
        static_objects use a shorter static-object layout and do not have a
        velocity column.  The previous implementation always wrote
        AgentIndex.VELOCITY, which crashes for static_objects with:

            IndexError: index 5 is out of bounds for axis 1 with size 5

        This helper keeps one call site for all element types while only writing
        fields that exist in the target state tensor.
        """
        num_state_dims = int(elem.states.shape[1])

        if AgentIndex.X < num_state_dims:
            elem.states[idx, AgentIndex.X] = float(x)
        if AgentIndex.Y < num_state_dims:
            elem.states[idx, AgentIndex.Y] = float(y)
        if AgentIndex.HEADING < num_state_dims:
            elem.states[idx, AgentIndex.HEADING] = float(heading)
        if AgentIndex.WIDTH < num_state_dims:
            elem.states[idx, AgentIndex.WIDTH] = float(width)
        if AgentIndex.LENGTH < num_state_dims:
            elem.states[idx, AgentIndex.LENGTH] = float(length)

        # static_objects do not contain a velocity column.
        if AgentIndex.VELOCITY < num_state_dims:
            elem.states[idx, AgentIndex.VELOCITY] = float(velocity)

        elem.mask[idx] = True

    @staticmethod
    def _get_ctx_elem(scene: SledgeVectorRaw, ctx) -> SledgeVectorElement:
        if ctx.actor_elem_name == "vehicles": return scene.vehicles
        if ctx.actor_elem_name == "pedestrians": return scene.pedestrians
        if ctx.actor_elem_name == "static_objects": return scene.static_objects
        raise ValueError(f"Invalid actor_elem_name={ctx.actor_elem_name}")

    @staticmethod
    def _default_size_for_elem(elem_name: str, actor: str) -> Tuple[float, float]:
        if elem_name == "pedestrians" or actor == "pedestrian": return 0.75, 0.75
        if actor == "cyclist": return 0.9, 1.8
        if elem_name == "static_objects" or actor == "static_obstacle": return 1.2, 2.0
        return 1.9, 4.8

    def _resolve_overlap(self, scene: SledgeVectorRaw, x: float, y: float, ignore: Tuple[str, int] | None = None, radius: float = 1.6) -> Tuple[float, float]:
        for dx in [0.0, -0.6, 0.6, -1.2, 1.2, -2.0, 2.0, 3.0, -3.0]:
            for dy in [0.0, -0.5, 0.5, -1.0, 1.0]:
                cx, cy = x + dx, y + dy
                if self._is_clear(scene, cx, cy, ignore=ignore, radius=radius):
                    return float(cx), float(cy)
        return float(x), float(y)

    def _try_place_vehicle_from_candidates(
        self,
        scene: SledgeVectorRaw,
        idx: int,
        candidates: List[Tuple[float, float]],
        heading: float,
        width: float,
        length: float,
        velocity: float,
        ignore: Optional[List[Tuple[str, int]]] = None,
    ) -> Optional[Tuple[float, float]]:
        ignore_pairs = list(ignore or [])
        if ("vehicles", int(idx)) not in ignore_pairs:
            ignore_pairs.append(("vehicles", int(idx)))
        occupied = self._collect_occupied_boxes(scene, ignore=ignore_pairs, include_ego=True)
        for x, y in candidates:
            box = self._make_aabb_from_values(float(x), float(y), heading, width, length, margin=0.25)
            if self._box_overlaps_any(box, occupied):
                continue
            self._set_agent_state(
                scene.vehicles,
                idx,
                x=float(x),
                y=float(y),
                heading=float(heading),
                width=float(width),
                length=float(length),
                velocity=float(velocity),
            )
            return float(x), float(y)
        return None

    @staticmethod
    def _is_clear(scene: SledgeVectorRaw, x: float, y: float, ignore: Tuple[str, int] | None = None, radius: float = 1.6) -> bool:
        p = np.array([x, y], dtype=np.float32)
        for elem_name, elem in [("vehicles", scene.vehicles), ("pedestrians", scene.pedestrians), ("static_objects", scene.static_objects)]:
            valid = np.asarray(elem.mask).astype(bool)
            if not np.any(valid): continue
            states = np.asarray(elem.states)
            for idx in np.where(valid)[0]:
                if ignore is not None and elem_name == ignore[0] and int(idx) == int(ignore[1]): continue
                center = np.asarray(states[idx, :2], dtype=np.float32)
                if float(np.linalg.norm(center - p)) < radius: return False
        return True

    @staticmethod
    def _estimate_ego_speed(scene: SledgeVectorRaw) -> float:
        ego_states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
        if ego_states.size == 0: return 6.0
        speed = float(np.linalg.norm(ego_states[:2])) if ego_states.size >= 2 else float(abs(ego_states[0]))
        return float(np.clip(speed, 2.5, 15.0))

    @staticmethod
    def _estimate_lane_center_y(scene: SledgeVectorRaw) -> float:
        valid = np.asarray(scene.vehicles.mask).astype(bool)
        if not np.any(valid): return 0.0
        states = np.asarray(scene.vehicles.states)
        vehicles = states[valid]
        forward = vehicles[(vehicles[:, AgentIndex.X] > 2.0) & (vehicles[:, AgentIndex.X] < 35.0)]
        if len(forward) == 0: return 0.0
        same_band = forward[np.abs(forward[:, AgentIndex.Y]) < 4.5]
        target = same_band if len(same_band) > 0 else forward
        return float(np.clip(np.median(target[:, AgentIndex.Y]), -1.0, 1.0))

    @staticmethod
    def _write_polyline(elem: SledgeVectorElement, slot: int, points_xy: np.ndarray) -> None:
        max_points = int(elem.states.shape[1])
        if max_points <= 0 or len(points_xy) < 2:
            return
        sampled_xy = PrimitiveOps._resample_polyline_xy(points_xy, max_points)
        diffs = np.gradient(sampled_xy, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32)
        elem.states[slot, :, 0] = sampled_xy[:, 0]
        elem.states[slot, :, 1] = sampled_xy[:, 1]
        if elem.states.shape[-1] > 2:
            elem.states[slot, :, 2] = headings
        elem.mask[slot, :] = True

    @staticmethod
    def _resample_polyline_xy(points_xy: np.ndarray, num_points: int) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32)
        seg = np.diff(points, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg_len)])
        if float(dist[-1]) <= 1e-4:
            return np.repeat(points[:1], num_points, axis=0)
        samples = np.linspace(0.0, float(dist[-1]), num_points)
        x = np.interp(samples, dist, points[:, 0])
        y = np.interp(samples, dist, points[:, 1])
        return np.stack([x, y], axis=-1).astype(np.float32)

    @staticmethod
    def _make_unprotected_left_turn_lines(lane_width: float) -> List[np.ndarray]:
        x_min, x_max = -28.0, 42.0
        y_min, y_max = -24.0, 30.0
        intersection_x = 14.0
        ego_y = 0.0
        opp_y = lane_width
        cross_x = intersection_x
        cross_x_opp = intersection_x + lane_width

        approach = np.asarray([[-24.0, ego_y], [-10.0, ego_y], [0.0, ego_y], [7.0, ego_y]], dtype=np.float32)
        turn_theta = np.linspace(-math.pi / 2.0, 0.0, 24)
        turn_center = np.asarray([7.0, 7.0], dtype=np.float32)
        turn_radius = 7.0
        turn = np.stack(
            [
                turn_center[0] + turn_radius * np.cos(turn_theta),
                turn_center[1] + turn_radius * np.sin(turn_theta),
            ],
            axis=-1,
        ).astype(np.float32)
        exit_north = np.asarray([[14.0, 7.0], [14.0, 18.0], [14.0, y_max]], dtype=np.float32)
        left_turn = np.asarray(
            np.concatenate([approach, turn[1:], exit_north[1:]], axis=0),
            dtype=np.float32,
        )
        oncoming_through = np.asarray([[x_max, opp_y], [intersection_x + 6.0, opp_y], [0.0, opp_y], [x_min, opp_y]], dtype=np.float32)
        northbound_lane = np.asarray([[cross_x_opp, y_min], [cross_x_opp, y_max]], dtype=np.float32)
        southbound_lane = np.asarray([[cross_x - lane_width, y_max], [cross_x - lane_width, y_min]], dtype=np.float32)
        crosswalk_a = np.asarray([[intersection_x - 3.0, -4.5], [intersection_x - 3.0, 8.5]], dtype=np.float32)
        crosswalk_b = np.asarray([[intersection_x + 3.0, -4.5], [intersection_x + 3.0, 8.5]], dtype=np.float32)

        return [
            left_turn,
            oncoming_through,
            northbound_lane,
            southbound_lane,
            crosswalk_a,
            crosswalk_b,
        ]

    @staticmethod
    def _make_roundabout_entry_lines(lane_width: float, radius: float) -> List[np.ndarray]:
        center_x, center_y = radius + 4.0, 0.0
        arm_len = 24.0
        inner_r = max(3.0, radius - 0.5 * lane_width)
        outer_r = radius + 0.5 * lane_width
        connector_r = outer_r + 1.2

        theta = np.linspace(-math.pi, math.pi, 128)
        inner_boundary = np.stack(
            [center_x + inner_r * np.cos(theta), center_y + inner_r * np.sin(theta)],
            axis=-1,
        ).astype(np.float32)
        outer_boundary = np.stack(
            [center_x + outer_r * np.cos(theta), center_y + outer_r * np.sin(theta)],
            axis=-1,
        ).astype(np.float32)
        circulating_center = np.stack(
            [center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)],
            axis=-1,
        ).astype(np.float32)

        lines: List[np.ndarray] = []

        def basis(angle: float) -> Tuple[np.ndarray, np.ndarray]:
            u = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
            t = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float32)
            return u, t

        def radial_line(angle: float, offset: float, start_r: float, end_r: float) -> np.ndarray:
            u, t = basis(angle)
            p0 = np.asarray([center_x, center_y], dtype=np.float32) + u * start_r + t * offset
            p1 = np.asarray([center_x, center_y], dtype=np.float32) + u * end_r + t * offset
            return np.asarray([p0, p1], dtype=np.float32)

        def point(angle: float, r: float = radius) -> np.ndarray:
            return np.asarray([center_x + r * math.cos(angle), center_y + r * math.sin(angle)], dtype=np.float32)

        def route_line(entry_angle: float, exit_angle: float, exit_offset: float) -> np.ndarray:
            while exit_angle <= entry_angle:
                exit_angle += 2.0 * math.pi
            u_in, _ = basis(entry_angle)
            u_out, t_out = basis(exit_angle)
            approach = np.asarray([center_x, center_y], dtype=np.float32) + u_in * (outer_r + arm_len)
            entry_near = np.asarray([center_x, center_y], dtype=np.float32) + u_in * connector_r
            arc_theta = np.linspace(entry_angle, exit_angle, 48)
            arc = np.stack(
                [center_x + radius * np.cos(arc_theta), center_y + radius * np.sin(arc_theta)],
                axis=-1,
            ).astype(np.float32)
            exit_near = np.asarray([center_x, center_y], dtype=np.float32) + u_out * connector_r + t_out * exit_offset
            outbound = np.asarray([center_x, center_y], dtype=np.float32) + u_out * (outer_r + arm_len) + t_out * exit_offset
            return np.asarray(np.vstack([approach, entry_near, point(entry_angle), arc[1:-1], point(exit_angle), exit_near, outbound]), dtype=np.float32)

        # Drivable path centerlines: straight approach -> roundabout arc -> a
        # different straight exit. The first one is the ego route west->south,
        # so SledgeScenario's fixed 32m mission goal lands on a non-starting
        # straight-road exit instead of a closed loop near the origin.
        lines.extend(
            [
                route_line(math.pi, 1.5 * math.pi, -lane_width),
                route_line(0.0, math.pi, -lane_width),
                route_line(math.pi / 2.0, 3.0 * math.pi / 2.0, -lane_width),
                route_line(-math.pi / 2.0, math.pi / 2.0, -lane_width),
            ]
        )

        lines.extend([inner_boundary, outer_boundary, circulating_center])

        # Four approach pairs. Entry lanes run into the roundabout; exit lanes
        # run back to a straight road so every arm visibly supports leaving.
        # The west entry is already represented by the ego route above; adding a
        # second isolated line through the origin makes SledgeScenario select it
        # as a dead-end route.
        arm_angles = [math.pi, 0.0, math.pi / 2.0, -math.pi / 2.0]
        for angle in arm_angles:
            if abs(angle - math.pi) > 1e-6:
                lines.append(radial_line(angle, 0.0, outer_r + arm_len, connector_r))
            lines.append(radial_line(angle, -lane_width, connector_r, outer_r + arm_len))

        # Entry yield bars, short and perpendicular to each approach. These read
        # as traffic-control markings instead of stray lane-line fragments.
        yield_half = 0.75 * lane_width
        lines.extend(
            [
                np.asarray([[center_x - outer_r - 2.5, -yield_half], [center_x - outer_r - 2.5, yield_half]], dtype=np.float32),
                np.asarray([[center_x + outer_r + 2.5, -yield_half], [center_x + outer_r + 2.5, yield_half]], dtype=np.float32),
                np.asarray([[center_x - yield_half, center_y + outer_r + 2.5], [center_x + yield_half, center_y + outer_r + 2.5]], dtype=np.float32),
                np.asarray([[center_x - yield_half, center_y - outer_r - 2.5], [center_x + yield_half, center_y - outer_r - 2.5]], dtype=np.float32),
            ]
        )
        return lines

    @staticmethod
    def _crossing_direction_to_side_sign(direction: str) -> float:
        if direction == "left_to_right": return 1.0
        if direction == "right_to_left": return -1.0
        return -1.0

    @staticmethod
    def _merge_direction_to_side_sign(direction: str) -> float:
        if direction == "left_merge": return 1.0
        if direction == "right_merge": return -1.0
        return 1.0

    @staticmethod
    def _clear_conflicting_same_lane_vehicle(elem: SledgeVectorElement, target_x: float, lane_y: float, ignore_index: int | None = None) -> List[int]:
        states = np.asarray(elem.states); mask = np.asarray(elem.mask).astype(bool); removed: List[int] = []
        for idx in np.where(mask)[0]:
            if ignore_index is not None and int(idx) == int(ignore_index): continue
            x, y = float(states[idx, AgentIndex.X]), float(states[idx, AgentIndex.Y])
            if abs(y - lane_y) < 1.2 and (target_x - 2.0) <= x <= (target_x + 5.0):
                elem.mask[idx] = False; removed.append(int(idx)); break
        return removed

    @staticmethod
    def _roi_from_state(state: np.ndarray, tag: str) -> SceneEditROI:
        x, y = float(state[AgentIndex.X]), float(state[AgentIndex.Y])
        width, length = float(max(state[AgentIndex.WIDTH], 0.5)), float(max(state[AgentIndex.LENGTH], 0.5))
        return SceneEditROI(x_min=x - length / 2 - 1.0, y_min=y - width / 2 - 1.0, x_max=x + length / 2 + 1.0, y_max=y + width / 2 + 1.0, tag=tag)
