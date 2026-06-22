from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from sledge.semantic_control.hazard_spec import HazardSemanticSpec


VALID_PRIMARY_ACTORS = {
    "pedestrian",
    "vehicle",
    "cyclist",
    "lead_vehicle",
    "cutin_vehicle",
    "rear_vehicle",
    "static_obstacle",
}

VALID_CONFLICT_TYPES = {
    "lateral_conflict",
    "longitudinal_conflict",
    "oncoming_conflict",
    "merging_conflict",
    "crossing_path_conflict",
    "lane_blocking_conflict",
}

ACTOR_CONFLICT_COMPATIBILITY: Dict[str, set[str]] = {
    "pedestrian": {"lateral_conflict", "crossing_path_conflict"},
    "cyclist": {"lateral_conflict", "crossing_path_conflict", "merging_conflict"},
    "vehicle": {"longitudinal_conflict", "oncoming_conflict", "merging_conflict", "crossing_path_conflict"},
    "lead_vehicle": {"longitudinal_conflict"},
    "cutin_vehicle": {"merging_conflict"},
    "rear_vehicle": {"longitudinal_conflict"},
    "static_obstacle": {"lane_blocking_conflict"},
}

RECOMMENDED_ROLE_BY_ACTOR = {
    "pedestrian": {"crossing_actor"},
    "cyclist": {"crossing_actor", "merging_actor"},
    "lead_vehicle": {"braking_actor"},
    "cutin_vehicle": {"merging_actor"},
    "rear_vehicle": {"approaching_actor"},
    "static_obstacle": {"blocking_actor"},
}


@dataclass
class SpecCheckReport:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_signature: str = ""

    def to_dict(self):
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized_signature": self.normalized_signature,
        }


def check_spec(spec: HazardSemanticSpec, strict: bool = False) -> SpecCheckReport:
    """
    Check whether a compositional hazard spec is logically usable.

    strict=False: only fatal incompatibilities become errors.
    strict=True: suspicious combinations also become errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    actor = spec.actor_layer.primary_actor
    role = spec.actor_layer.actor_role
    conflict = spec.interaction_layer.conflict_type
    direction = spec.interaction_layer.conflict_direction
    visibility = "occluded" if spec.object_layer.occlusion.enabled else "visible"

    if actor not in VALID_PRIMARY_ACTORS:
        errors.append(f"Unsupported primary_actor={actor}. Expected one of {sorted(VALID_PRIMARY_ACTORS)}.")

    if conflict not in VALID_CONFLICT_TYPES:
        errors.append(f"Unsupported conflict_type={conflict}. Expected one of {sorted(VALID_CONFLICT_TYPES)}.")

    allowed_conflicts = ACTOR_CONFLICT_COMPATIBILITY.get(actor)
    if allowed_conflicts is not None and conflict not in allowed_conflicts:
        errors.append(
            f"Incompatible actor/conflict pair: primary_actor={actor} does not support "
            f"conflict_type={conflict}. Allowed: {sorted(allowed_conflicts)}."
        )

    recommended_roles = RECOMMENDED_ROLE_BY_ACTOR.get(actor)
    if recommended_roles is not None and role not in recommended_roles:
        msg = f"Suspicious actor_role={role} for primary_actor={actor}. Recommended: {sorted(recommended_roles)}."
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if conflict == "longitudinal_conflict" and direction not in {"front", "rear"}:
        errors.append("longitudinal_conflict should use conflict_direction in {front, rear}.")

    if conflict == "oncoming_conflict" and direction not in {"front", "opposite", "auto"}:
        errors.append("oncoming_conflict should use conflict_direction in {front, opposite, auto}.")

    if conflict == "merging_conflict" and direction not in {"left_merge", "right_merge", "auto"}:
        errors.append("merging_conflict should use conflict_direction in {left_merge, right_merge, auto}.")

    if conflict in {"lateral_conflict", "crossing_path_conflict"} and direction not in {
        "left_to_right",
        "right_to_left",
        "auto",
    }:
        errors.append("lateral/crossing-path conflicts should use left_to_right, right_to_left, or auto.")

    if spec.object_layer.occlusion.enabled:
        if spec.object_layer.occlusion.occluder_type in {"none", ""}:
            errors.append("occlusion.enabled=True requires a non-empty occluder_type.")
        if spec.object_layer.occlusion.occlusion_level not in {"partial", "full"}:
            errors.append("occlusion.enabled=True requires occlusion_level in {partial, full}.")
        if spec.actor_layer.secondary_actor is None:
            warnings.append("occlusion is enabled but actor_layer.secondary_actor is None; it can be auto-filled.")

    if visibility == "visible" and spec.validation_layer.require_visibility_match:
        warnings.append("require_visibility_match=True while occlusion is disabled.")

    if spec.risk_layer.risk_level not in {"mild", "moderate", "aggressive"}:
        errors.append("risk_level must be one of {mild, moderate, aggressive}.")

    if spec.risk_layer.ttc_range_s[0] < 0.0:
        errors.append("ttc_range_s lower bound must be non-negative.")
    if spec.risk_layer.gap_range_m[0] < 0.0:
        errors.append("gap_range_m lower bound must be non-negative.")

    return SpecCheckReport(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        normalized_signature=spec.semantic_signature,
    )
