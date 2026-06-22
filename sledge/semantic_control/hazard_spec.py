from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


NumberRange = Tuple[float, float]


def _range2(value: Any, default: NumberRange) -> NumberRange:
    """Convert a JSON list/tuple/scalar into a 2-float range."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = list(value)
        if len(vals) == 0:
            return default
        if len(vals) == 1:
            v = float(vals[0])
            return (v, v)
        a, b = float(vals[0]), float(vals[1])
        return (min(a, b), max(a, b))
    return default


@dataclass
class RoadLayer:
    """
    Road/map context where the hazard should be constructed.

    Phase 1 uses this layer mainly as an anchor selector. It does not require
    direct road topology generation yet.
    """

    road_topology: str = "straight"  # straight / curve / intersection / merge / roundabout / crosswalk_area / construction_zone
    lane_context: str = "ego_path"  # ego_path / same_lane / adjacent_lane / crossing_path / opposite_lane
    anchor_type: str = "ego_future_path"  # ego_future_path / ego_lane_front / adjacent_lane / crosswalk / intersection_center
    anchor_region: str = "front"  # front / left / right / rear / center

    num_lanes: Optional[int] = None
    lane_width_m: float = 3.5
    has_crosswalk: Optional[bool] = None
    has_intersection: Optional[bool] = None
    has_merge_area: Optional[bool] = None

    require_lane_continuity: bool = True
    require_drivable_route: bool = True
    require_no_broken_boundary: bool = True

    # Optional lane-level synthesis. Existing specs keep this disabled and only
    # edit entities on top of the source map.
    allow_lane_generation: bool = False
    generated_road_layout: str = "none"  # none / unprotected_left_turn / roundabout_entry
    generated_road_radius_m: float = 14.0


@dataclass
class ActorLayer:
    """Dynamic participant layer."""

    primary_actor: str = "vehicle"  # pedestrian / vehicle / cyclist / lead_vehicle / cutin_vehicle / rear_vehicle
    actor_role: str = "none"  # crossing_actor / braking_actor / merging_actor / approaching_actor / blocking_actor
    secondary_actor: Optional[str] = None  # occluding_vehicle / parked_vehicle / static_obstacle / none
    supporting_actors: List[str] = field(default_factory=lambda: ["ego_vehicle"])

    allow_actor_insertion: bool = True
    prefer_existing_actor: bool = False
    allow_actor_replacement: bool = True


@dataclass
class OcclusionSpec:
    enabled: bool = False
    occluder_type: str = "none"  # parked_vehicle / truck / bus / barrier / static_object
    occlusion_position: str = "between_ego_and_actor"
    occlusion_level: str = "none"  # none / partial / full


@dataclass
class StaticObstacleSpec:
    enabled: bool = False
    obstacle_type: str = "none"  # cone / barrier / parked_vehicle / construction_zone
    obstacle_position: str = "none"  # lane_boundary / ego_lane / roadside / between_ego_and_actor
    obstacle_size: str = "normal"  # small / normal / large


@dataclass
class ObjectLayer:
    """Static object and occlusion layer."""

    occlusion: OcclusionSpec = field(default_factory=OcclusionSpec)
    static_obstacle: StaticObstacleSpec = field(default_factory=StaticObstacleSpec)


@dataclass
class InteractionLayer:
    """Hazard relation between ego, road context, and actors."""

    conflict_type: str = "longitudinal_conflict"  # lateral_conflict / longitudinal_conflict / oncoming_conflict / merging_conflict / crossing_path_conflict / lane_blocking_conflict
    conflict_direction: str = "front"  # left_to_right / right_to_left / front / rear / left_merge / right_merge
    distance_relation: str = "medium"  # close / short_headway / small_gap / close_lateral_gap / medium
    speed_relation: str = "normal"  # normal / slow_lead / stopped / fast_approach / fast_crossing
    interaction_goal: str = "near_miss"  # near_miss / braking_pressure / collision_risk / yielding_conflict / trajectory_blocking


@dataclass
class RiskLayer:
    """Executable risk and numeric control layer."""

    risk_level: str = "moderate"  # mild / moderate / aggressive
    ttc_range_s: NumberRange = (2.0, 3.0)
    gap_range_m: NumberRange = (5.0, 10.0)
    lateral_gap_range_m: NumberRange = (1.0, 2.0)
    longitudinal_distance_range_m: NumberRange = (10.0, 18.0)
    lateral_intrusion_range_m: NumberRange = (0.5, 0.9)

    target_actor_speed_mps: float = 1.6
    target_relative_speed_mps: float = 3.0
    target_decel_range_mps2: NumberRange = (3.0, 5.0)

    collision_allowed: bool = False


@dataclass
class ValidationLayer:
    """Semantic constraints to be checked after editing/repainting."""

    require_actor_match: bool = True
    require_road_context_match: bool = True
    require_conflict_relation: bool = True
    require_direction_match: bool = True
    require_visibility_match: bool = False
    require_lane_validity: bool = True
    require_no_initial_collision: bool = True
    require_ttc_in_range: bool = False
    require_gap_in_range: bool = True


@dataclass
class ProtectionLayer:
    """Regions that should be protected by later repaint/inpainting stages."""

    protect_primary_actor: bool = True
    protect_secondary_actor: bool = True
    protect_static_obstacle: bool = True
    protect_conflict_corridor: bool = True
    protect_road_anchor: bool = True


@dataclass
class HazardSemanticSpec:
    """
    Unified compositional critical-scenario specification.

    A scenario is defined by semantic layers, not by a fixed scenario type.
    canonical_type is optional and only used for visualization/statistics.
    """

    spec_id: str
    description: str = ""
    canonical_type: Optional[str] = None
    raw_prompt: str = ""

    road_layer: RoadLayer = field(default_factory=RoadLayer)
    actor_layer: ActorLayer = field(default_factory=ActorLayer)
    object_layer: ObjectLayer = field(default_factory=ObjectLayer)
    interaction_layer: InteractionLayer = field(default_factory=InteractionLayer)
    risk_layer: RiskLayer = field(default_factory=RiskLayer)
    validation_layer: ValidationLayer = field(default_factory=ValidationLayer)
    protection_layer: ProtectionLayer = field(default_factory=ProtectionLayer)

    tags: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HazardSemanticSpec":
        road_data = dict(data.get("road_layer", {}))
        actor_data = dict(data.get("actor_layer", {}))
        object_data = dict(data.get("object_layer", {}))
        interaction_data = dict(data.get("interaction_layer", {}))
        risk_data = dict(data.get("risk_layer", {}))
        validation_data = dict(data.get("validation_layer", {}))
        protection_data = dict(data.get("protection_layer", {}))

        occlusion = OcclusionSpec(**dict(object_data.get("occlusion", {})))
        static_obstacle = StaticObstacleSpec(**dict(object_data.get("static_obstacle", {})))

        default_risk = RiskLayer()
        risk_data["ttc_range_s"] = _range2(risk_data.get("ttc_range_s"), default_risk.ttc_range_s)
        risk_data["gap_range_m"] = _range2(risk_data.get("gap_range_m"), default_risk.gap_range_m)
        risk_data["lateral_gap_range_m"] = _range2(
            risk_data.get("lateral_gap_range_m"), default_risk.lateral_gap_range_m
        )
        risk_data["longitudinal_distance_range_m"] = _range2(
            risk_data.get("longitudinal_distance_range_m"), default_risk.longitudinal_distance_range_m
        )
        risk_data["lateral_intrusion_range_m"] = _range2(
            risk_data.get("lateral_intrusion_range_m"), default_risk.lateral_intrusion_range_m
        )
        risk_data["target_decel_range_mps2"] = _range2(
            risk_data.get("target_decel_range_mps2"), default_risk.target_decel_range_mps2
        )

        return cls(
            spec_id=str(data.get("spec_id", "unnamed_spec")),
            description=str(data.get("description", "")),
            canonical_type=data.get("canonical_type"),
            raw_prompt=str(data.get("raw_prompt", "")),
            road_layer=RoadLayer(**road_data),
            actor_layer=ActorLayer(**actor_data),
            object_layer=ObjectLayer(occlusion=occlusion, static_obstacle=static_obstacle),
            interaction_layer=InteractionLayer(**interaction_data),
            risk_layer=RiskLayer(**risk_data),
            validation_layer=ValidationLayer(**validation_data),
            protection_layer=ProtectionLayer(**protection_data),
            tags=list(data.get("tags", [])),
            debug=dict(data.get("debug", {})),
        )

    @property
    def semantic_signature(self) -> str:
        """Compact signature for logging and grouping."""
        visibility = "occluded" if self.object_layer.occlusion.enabled else "visible"
        parts = [
            self.road_layer.road_topology,
            self.road_layer.lane_context,
            self.actor_layer.primary_actor,
            self.actor_layer.actor_role,
            self.interaction_layer.conflict_type,
            self.interaction_layer.conflict_direction,
            visibility,
            self.risk_layer.risk_level,
            self.interaction_layer.interaction_goal,
        ]
        return "/".join([p for p in parts if p and p != "none"])
