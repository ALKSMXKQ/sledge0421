from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from sledge.semantic_control.hazard_spec import HazardSemanticSpec


@dataclass
class PrimitiveOp:
    """
    A single executable semantic editing operation.

    Phase 1 only compiles these ops. Later phases will implement a PrimitiveExecutor
    that applies them to SledgeVectorRaw.
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    source_layer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compile_spec_to_ops(spec: HazardSemanticSpec) -> List[PrimitiveOp]:
    """
    Slot-level compiler:
    HazardSemanticSpec -> primitive operations.

    Important: this function must not route by canonical_type. canonical_type is
    only for logging/visualization. Generation is controlled by layer fields.
    """
    ops: List[PrimitiveOp] = []

    road = spec.road_layer
    actor = spec.actor_layer
    obj = spec.object_layer
    inter = spec.interaction_layer
    risk = spec.risk_layer
    prot = spec.protection_layer

    # 1. Road/map anchor primitive.
    ops.append(
        PrimitiveOp(
            name="select_road_anchor",
            source_layer="road_layer",
            params={
                "road_topology": road.road_topology,
                "lane_context": road.lane_context,
                "anchor_type": road.anchor_type,
                "anchor_region": road.anchor_region,
                "has_crosswalk": road.has_crosswalk,
                "has_intersection": road.has_intersection,
                "has_merge_area": road.has_merge_area,
                "allow_lane_generation": road.allow_lane_generation,
                "generated_road_layout": road.generated_road_layout,
                "generated_road_radius_m": road.generated_road_radius_m,
            },
        )
    )

    if road.allow_lane_generation:
        ops.append(
            PrimitiveOp(
                name="generate_or_adjust_road_geometry",
                source_layer="road_layer",
                params={
                    "road_topology": road.road_topology,
                    "lane_context": road.lane_context,
                    "anchor_type": road.anchor_type,
                    "anchor_region": road.anchor_region,
                    "lane_width_m": road.lane_width_m,
                    "num_lanes": road.num_lanes,
                    "generated_road_layout": road.generated_road_layout,
                    "generated_road_radius_m": road.generated_road_radius_m,
                    "has_crosswalk": road.has_crosswalk,
                    "has_intersection": road.has_intersection,
                    "has_merge_area": road.has_merge_area,
                },
            )
        )

    # 2. Primary actor primitive.
    ops.append(
        PrimitiveOp(
            name="select_or_spawn_actor",
            source_layer="actor_layer",
            params={
                "primary_actor": actor.primary_actor,
                "actor_role": actor.actor_role,
                "prefer_existing_actor": actor.prefer_existing_actor,
                "allow_actor_insertion": actor.allow_actor_insertion,
                "allow_actor_replacement": actor.allow_actor_replacement,
            },
        )
    )

    # 3. Relation/placement primitives.
    if inter.conflict_type in {"lateral_conflict", "crossing_path_conflict"}:
        ops.append(
            PrimitiveOp(
                name="place_actor_laterally",
                source_layer="interaction_layer",
                params={
                    "conflict_type": inter.conflict_type,
                    "direction": inter.conflict_direction,
                    "distance_relation": inter.distance_relation,
                    "longitudinal_distance_range_m": risk.longitudinal_distance_range_m,
                    "lateral_gap_range_m": risk.lateral_gap_range_m,
                },
            )
        )
        ops.append(
            PrimitiveOp(
                name="set_lateral_or_crossing_motion",
                source_layer="interaction_layer",
                params={
                    "direction": inter.conflict_direction,
                    "speed_relation": inter.speed_relation,
                    "target_actor_speed_mps": risk.target_actor_speed_mps,
                },
            )
        )

    elif inter.conflict_type == "longitudinal_conflict":
        ops.append(
            PrimitiveOp(
                name="place_actor_longitudinally",
                source_layer="interaction_layer",
                params={
                    "direction": inter.conflict_direction,
                    "distance_relation": inter.distance_relation,
                    "gap_range_m": risk.gap_range_m,
                    "longitudinal_distance_range_m": risk.longitudinal_distance_range_m,
                },
            )
        )
        ops.append(
            PrimitiveOp(
                name="set_speed_relation",
                source_layer="interaction_layer",
                params={
                    "speed_relation": inter.speed_relation,
                    "target_relative_speed_mps": risk.target_relative_speed_mps,
                    "target_decel_range_mps2": risk.target_decel_range_mps2,
                },
            )
        )

    elif inter.conflict_type == "oncoming_conflict":
        ops.append(
            PrimitiveOp(
                name="place_oncoming_actor",
                source_layer="interaction_layer",
                params={
                    "direction": inter.conflict_direction,
                    "distance_relation": inter.distance_relation,
                    "gap_range_m": risk.gap_range_m,
                    "longitudinal_distance_range_m": risk.longitudinal_distance_range_m,
                },
            )
        )
        ops.append(
            PrimitiveOp(
                name="set_oncoming_motion",
                source_layer="interaction_layer",
                params={
                    "speed_relation": inter.speed_relation,
                    "target_actor_speed_mps": risk.target_actor_speed_mps,
                    "target_relative_speed_mps": risk.target_relative_speed_mps,
                },
            )
        )

    elif inter.conflict_type == "merging_conflict":
        ops.append(
            PrimitiveOp(
                name="place_actor_for_merging",
                source_layer="interaction_layer",
                params={
                    "direction": inter.conflict_direction,
                    "distance_relation": inter.distance_relation,
                    "gap_range_m": risk.gap_range_m,
                    "lateral_intrusion_range_m": risk.lateral_intrusion_range_m,
                },
            )
        )
        ops.append(
            PrimitiveOp(
                name="set_merging_motion",
                source_layer="interaction_layer",
                params={
                    "direction": inter.conflict_direction,
                    "speed_relation": inter.speed_relation,
                    "target_relative_speed_mps": risk.target_relative_speed_mps,
                },
            )
        )
        if road.generated_road_layout == "roundabout_entry":
            ops.append(
                PrimitiveOp(
                    name="add_roundabout_cross_traffic",
                    source_layer="interaction_layer",
                    params={
                        "road_topology": road.road_topology,
                        "generated_road_layout": road.generated_road_layout,
                        "generated_road_radius_m": road.generated_road_radius_m,
                        "lane_width_m": road.lane_width_m,
                        "target_actor_speed_mps": risk.target_actor_speed_mps,
                        "target_relative_speed_mps": risk.target_relative_speed_mps,
                    },
                )
            )

    elif inter.conflict_type == "lane_blocking_conflict":
        ops.append(
            PrimitiveOp(
                name="place_static_or_slow_actor_as_blocker",
                source_layer="interaction_layer",
                params={
                    "distance_relation": inter.distance_relation,
                    "gap_range_m": risk.gap_range_m,
                    "obstacle_type": obj.static_obstacle.obstacle_type,
                    "obstacle_position": obj.static_obstacle.obstacle_position,
                },
            )
        )

    else:
        raise ValueError(f"Unsupported conflict_type={inter.conflict_type}")

    # 4. Object/occlusion primitives.
    if obj.occlusion.enabled:
        ops.append(
            PrimitiveOp(
                name="add_or_select_occluder",
                source_layer="object_layer",
                params={
                    "secondary_actor": actor.secondary_actor,
                    "occluder_type": obj.occlusion.occluder_type,
                    "occlusion_position": obj.occlusion.occlusion_position,
                    "occlusion_level": obj.occlusion.occlusion_level,
                    "risk_level": risk.risk_level,
                    "target_actor_speed_mps": risk.target_actor_speed_mps,
                    "longitudinal_distance_range_m": risk.longitudinal_distance_range_m,
                    "lateral_gap_range_m": risk.lateral_gap_range_m,
                    "direction": inter.conflict_direction,
                    "frame0_time_offset_s": 2.1,
                    "compensate_frame0_offset": True,
                    "avoid_ego_overlap": True,
                    "avoid_actor_overlap": True,
                    "avoid_existing_agent_overlap": True,
                },
            )
        )

    if obj.static_obstacle.enabled:
        ops.append(
            PrimitiveOp(
                name="add_or_select_static_obstacle",
                source_layer="object_layer",
                params={
                    "obstacle_type": obj.static_obstacle.obstacle_type,
                    "obstacle_position": obj.static_obstacle.obstacle_position,
                    "obstacle_size": obj.static_obstacle.obstacle_size,
                },
            )
        )

    # 5. Risk scaling.
    ops.append(
        PrimitiveOp(
            name="apply_risk_scaling",
            source_layer="risk_layer",
            params={
                "risk_level": risk.risk_level,
                "ttc_range_s": risk.ttc_range_s,
                "gap_range_m": risk.gap_range_m,
                "lateral_gap_range_m": risk.lateral_gap_range_m,
                "collision_allowed": risk.collision_allowed,
            },
        )
    )

    # 6. Repaint protection planning.
    protected_targets = []
    if prot.protect_primary_actor:
        protected_targets.append("primary_actor")
    if prot.protect_secondary_actor:
        protected_targets.append("secondary_actor")
    if prot.protect_static_obstacle:
        protected_targets.append("static_obstacle")
    if prot.protect_conflict_corridor:
        protected_targets.append("conflict_corridor")
    if prot.protect_road_anchor:
        protected_targets.append("road_anchor")

    ops.append(
        PrimitiveOp(
            name="build_protection_rois",
            source_layer="protection_layer",
            params={"protected_targets": protected_targets},
        )
    )

    # 7. Semantic validation.
    ops.append(
        PrimitiveOp(
            name="validate_semantic_constraints",
            source_layer="validation_layer",
            params=spec.validation_layer.__dict__.copy(),
        )
    )

    return ops


def ops_to_dicts(ops: List[PrimitiveOp]) -> List[Dict[str, Any]]:
    return [op.to_dict() for op in ops]
