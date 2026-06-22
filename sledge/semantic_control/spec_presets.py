from __future__ import annotations

from copy import deepcopy
from typing import Dict

from sledge.semantic_control.hazard_spec import HazardSemanticSpec


RISK_PRESETS: Dict[str, Dict[str, object]] = {
    "mild": {
        "ttc_range_s": (3.0, 4.2),
        "gap_range_m": (8.0, 14.0),
        "lateral_gap_range_m": (2.0, 3.0),
        "longitudinal_distance_range_m": (18.0, 25.0),
        "lateral_intrusion_range_m": (0.2, 0.4),
        "target_actor_speed_mps": 1.2,
        "target_relative_speed_mps": 2.0,
        "target_decel_range_mps2": (2.0, 3.5),
    },
    "moderate": {
        "ttc_range_s": (2.0, 3.0),
        "gap_range_m": (5.0, 10.0),
        "lateral_gap_range_m": (1.0, 2.0),
        "longitudinal_distance_range_m": (10.0, 18.0),
        "lateral_intrusion_range_m": (0.5, 0.9),
        "target_actor_speed_mps": 1.6,
        "target_relative_speed_mps": 3.0,
        "target_decel_range_mps2": (3.0, 5.0),
    },
    "aggressive": {
        "ttc_range_s": (1.2, 2.0),
        "gap_range_m": (3.5, 7.0),
        "lateral_gap_range_m": (0.3, 1.0),
        "longitudinal_distance_range_m": (5.0, 10.0),
        "lateral_intrusion_range_m": (1.0, 1.5),
        "target_actor_speed_mps": 1.9,
        "target_relative_speed_mps": 4.5,
        "target_decel_range_mps2": (4.5, 7.0),
    },
}


ROAD_ANCHOR_PRESETS: Dict[str, Dict[str, str]] = {
    "lateral_conflict": {
        "anchor_type": "ego_future_path",
        "lane_context": "ego_path",
    },
    "longitudinal_conflict": {
        "anchor_type": "ego_lane_front",
        "lane_context": "same_lane",
    },
    "oncoming_conflict": {
        "anchor_type": "intersection_center",
        "lane_context": "opposite_lane",
    },
    "merging_conflict": {
        "anchor_type": "adjacent_lane",
        "lane_context": "adjacent_lane",
    },
    "crossing_path_conflict": {
        "anchor_type": "crossing_path",
        "lane_context": "crossing_path",
    },
    "lane_blocking_conflict": {
        "anchor_type": "ego_lane_front",
        "lane_context": "same_lane",
    },
}


def apply_risk_preset(spec: HazardSemanticSpec, overwrite: bool = False) -> HazardSemanticSpec:
    """
    Fill numeric risk fields according to risk_level.

    Set overwrite=True when you want risk_level to fully control numeric fields.
    Set overwrite=False to only normalize empty/default-like fields.
    """
    out = deepcopy(spec)
    preset = RISK_PRESETS.get(out.risk_layer.risk_level, RISK_PRESETS["moderate"])

    default = HazardSemanticSpec(spec_id="_default").risk_layer

    for field_name, preset_value in preset.items():
        current_value = getattr(out.risk_layer, field_name)
        default_value = getattr(default, field_name)
        if overwrite or current_value == default_value:
            setattr(out.risk_layer, field_name, preset_value)

    return out


def apply_anchor_preset(spec: HazardSemanticSpec, overwrite: bool = False) -> HazardSemanticSpec:
    """Fill anchor fields from conflict_type if they are not explicitly specified."""
    out = deepcopy(spec)
    conflict_type = out.interaction_layer.conflict_type
    preset = ROAD_ANCHOR_PRESETS.get(conflict_type)
    if preset is None:
        return out

    if overwrite or out.road_layer.anchor_type == "ego_future_path":
        out.road_layer.anchor_type = preset["anchor_type"]
    if overwrite or out.road_layer.lane_context == "ego_path":
        out.road_layer.lane_context = preset["lane_context"]

    return out


def normalize_spec(spec: HazardSemanticSpec, overwrite_risk: bool = False) -> HazardSemanticSpec:
    """Apply all phase-1 normalization rules."""
    out = apply_risk_preset(spec, overwrite=overwrite_risk)
    out = apply_anchor_preset(out, overwrite=False)

    # Automatically tighten visibility validation for occlusion cases.
    if out.object_layer.occlusion.enabled:
        out.validation_layer.require_visibility_match = True
        if out.actor_layer.secondary_actor is None:
            out.actor_layer.secondary_actor = out.object_layer.occlusion.occluder_type

    return out
