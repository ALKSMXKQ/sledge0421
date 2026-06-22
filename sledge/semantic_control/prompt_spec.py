from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptSpec:
    """
    Unified prompt specification for multiple rare scenario families.

    Supported scenario types:
        - sudden_pedestrian_crossing
        - cut_in
        - hard_brake

    The dataclass remains backward-compatible with the original crossing-only
    version while adding scenario-generic control fields.
    """

    raw_prompt: str
    normalized_prompt: str
    scenario_type: str = "generic"
    city: Optional[str] = None
    map_id: Optional[int] = None

    # Legacy / backward-compatible fields
    occluder_type: str = "none"
    side: str = "auto"
    moderate_traffic: bool = False
    yielding: bool = False
    blind_spot: bool = False
    pedestrian_emerge: bool = False
    use_existing_occluder_first: bool = False
    insert_occluder_if_missing: bool = False

    # Generic scene control fields
    severity_level: str = "moderate"  # mild / moderate / aggressive
    primary_actor_type: str = "none"  # pedestrian / vehicle / lead_vehicle
    crossing_style: str = "pedestrian_crossing"

    # Shared motion / timing parameters
    pedestrian_speed: float = 1.6
    vehicle_speed: float = 8.0
    target_speed_mps: float = 0.0
    ttc_min_s: float = 2.0
    ttc_max_s: float = 3.0

    # Geometric targets / gaps
    conflict_point_x_m: float = 12.0
    conflict_point_y_m: float = 0.0
    target_gap_min_m: float = 5.0
    target_gap_max_m: float = 12.0
    lead_distance_min_m: float = 6.0
    lead_distance_max_m: float = 15.0

    # Scenario-specific dynamic bounds
    lateral_speed_min_mps: float = 0.8
    lateral_speed_max_mps: float = 2.4
    relative_speed_min_mps: float = 2.0
    relative_speed_max_mps: float = 8.0
    target_decel_min_mps2: float = 2.0
    target_decel_max_mps2: float = 6.0

    # Editing policy
    spawn_from_roadside: bool = True
    keep_scene_minimal: bool = True
    allow_actor_insertion: bool = True
    prefer_existing_actor: bool = False

    matched_rules: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneEditROI:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    tag: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "tag": self.tag,
        }


@dataclass
class SceneEditResult:
    prompt_spec: PromptSpec
    occluder_source: str = "none"
    occluder_index: int = -1
    occluder_elem_name: str = "vehicles"
    static_obstacle_index: int = -1
    pedestrian_index: int = -1

    primary_actor_type: str = "none"
    primary_actor_index: int = -1

    conflict_point_xy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    preserved_rois: List[SceneEditROI] = field(default_factory=list)

    removed_vehicle_indices: List[int] = field(default_factory=list)
    slowed_vehicle_indices: List[int] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_spec": self.prompt_spec.to_dict(),
            "occluder_source": self.occluder_source,
            "occluder_index": self.occluder_index,
            "occluder_elem_name": self.occluder_elem_name,
            "static_obstacle_index": self.static_obstacle_index,
            "pedestrian_index": self.pedestrian_index,
            "primary_actor_type": self.primary_actor_type,
            "primary_actor_index": self.primary_actor_index,
            "conflict_point_xy": list(self.conflict_point_xy),
            "preserved_rois": [roi.to_dict() for roi in self.preserved_rois],
            "removed_vehicle_indices": list(self.removed_vehicle_indices),
            "slowed_vehicle_indices": list(self.slowed_vehicle_indices),
            "notes": list(self.notes),
        }
