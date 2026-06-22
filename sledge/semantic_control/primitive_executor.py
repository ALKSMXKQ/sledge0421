from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import SledgeVectorRaw
from sledge.semantic_control.hazard_spec import HazardSemanticSpec
from sledge.semantic_control.primitive_compiler import PrimitiveOp
from sledge.semantic_control.primitive_ops import PrimitiveOps
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditResult, SceneEditROI


@dataclass
class EditContext:
    """
    Runtime state shared by primitive operations.

    Primitive ops should communicate through this context rather than through
    scenario-level global variables.
    """

    spec: HazardSemanticSpec
    actor_elem_name: str = "none"  # vehicles / pedestrians / static_objects
    actor_index: int = -1
    occluder_index: int = -1
    occluder_elem_name: str = "vehicles"
    static_obstacle_index: int = -1
    anchor: Dict[str, Any] = field(default_factory=dict)
    rois: List[SceneEditROI] = field(default_factory=list)
    removed_vehicle_indices: List[int] = field(default_factory=list)
    slowed_vehicle_indices: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class PrimitiveExecutor:
    """
    Execute primitive ops on SledgeVectorRaw.

    This is the phase-2 bridge:
        HazardSemanticSpec -> primitive ops -> edited SledgeVectorRaw.
    """

    def __init__(self) -> None:
        self.ops = PrimitiveOps()

    def execute(
        self,
        scene: SledgeVectorRaw,
        spec: HazardSemanticSpec,
        primitive_ops: List[PrimitiveOp],
    ) -> Tuple[SledgeVectorRaw, SceneEditResult, Dict[str, Any]]:
        ctx = EditContext(spec=spec)
        edited = scene
        op_reports: List[Dict[str, Any]] = []

        for op in primitive_ops:
            edited, ctx, report = self._dispatch(edited, ctx, op)
            op_reports.append(report)

        result = self._build_scene_edit_result(ctx)
        full_report: Dict[str, Any] = {
            "spec": spec.to_dict(),
            "semantic_signature": spec.semantic_signature,
            "ops": [op.to_dict() for op in primitive_ops],
            "op_reports": op_reports,
            "validation": ctx.validation,
            "extra": dict(ctx.extra),
            "notes": list(ctx.notes),
        }
        return edited, result, full_report

    def _dispatch(self, scene: SledgeVectorRaw, ctx: EditContext, op: PrimitiveOp):
        name = op.name
        if name == "select_road_anchor":
            return self.ops.select_road_anchor(scene, ctx, op.params)
        if name == "generate_or_adjust_road_geometry":
            return self.ops.generate_or_adjust_road_geometry(scene, ctx, op.params)
        if name == "select_or_spawn_actor":
            return self.ops.select_or_spawn_actor(scene, ctx, op.params)
        if name == "place_actor_laterally":
            return self.ops.place_actor_laterally(scene, ctx, op.params)
        if name == "set_lateral_or_crossing_motion":
            return self.ops.set_lateral_or_crossing_motion(scene, ctx, op.params)
        if name == "place_actor_longitudinally":
            return self.ops.place_actor_longitudinally(scene, ctx, op.params)
        if name == "set_speed_relation":
            return self.ops.set_speed_relation(scene, ctx, op.params)
        if name == "place_oncoming_actor":
            return self.ops.place_oncoming_actor(scene, ctx, op.params)
        if name == "set_oncoming_motion":
            return self.ops.set_oncoming_motion(scene, ctx, op.params)
        if name == "place_actor_for_merging":
            return self.ops.place_actor_for_merging(scene, ctx, op.params)
        if name == "set_merging_motion":
            return self.ops.set_merging_motion(scene, ctx, op.params)
        if name == "add_roundabout_cross_traffic":
            return self.ops.add_roundabout_cross_traffic(scene, ctx, op.params)
        if name == "place_static_or_slow_actor_as_blocker":
            return self.ops.place_static_or_slow_actor_as_blocker(scene, ctx, op.params)
        if name == "add_or_select_occluder":
            return self.ops.add_or_select_occluder(scene, ctx, op.params)
        if name == "add_or_select_static_obstacle":
            return self.ops.add_or_select_static_obstacle(scene, ctx, op.params)
        if name == "apply_risk_scaling":
            return self.ops.apply_risk_scaling(scene, ctx, op.params)
        if name == "build_protection_rois":
            return self.ops.build_protection_rois(scene, ctx, op.params)
        if name == "validate_semantic_constraints":
            return self.ops.validate_semantic_constraints(scene, ctx, op.params)
        raise ValueError(f"Unknown primitive op: {name}")

    def _build_scene_edit_result(self, ctx: EditContext) -> SceneEditResult:
        spec = ctx.spec
        legacy_spec = self._to_legacy_prompt_spec(spec)
        primary_actor_index = int(ctx.actor_index)
        pedestrian_index = primary_actor_index if ctx.actor_elem_name == "pedestrians" else -1
        conflict_x = float(ctx.anchor.get("x", 0.0))
        conflict_y = float(ctx.anchor.get("y", 0.0))
        return SceneEditResult(
            prompt_spec=legacy_spec,
            occluder_source=spec.object_layer.occlusion.occluder_type if ctx.occluder_index >= 0 else "none",
            occluder_index=int(ctx.occluder_index),
            occluder_elem_name=str(getattr(ctx, "occluder_elem_name", "vehicles")),
            static_obstacle_index=int(ctx.static_obstacle_index),
            pedestrian_index=int(pedestrian_index),
            primary_actor_type=spec.actor_layer.primary_actor,
            primary_actor_index=primary_actor_index,
            conflict_point_xy=[conflict_x, conflict_y],
            preserved_rois=list(ctx.rois),
            removed_vehicle_indices=list(ctx.removed_vehicle_indices),
            slowed_vehicle_indices=list(ctx.slowed_vehicle_indices),
            notes=list(ctx.notes),
        )

    @staticmethod
    def _to_legacy_prompt_spec(spec: HazardSemanticSpec) -> PromptSpec:
        risk = spec.risk_layer
        inter = spec.interaction_layer
        actor = spec.actor_layer
        return PromptSpec(
            raw_prompt=spec.raw_prompt or spec.description,
            normalized_prompt=spec.raw_prompt or spec.description,
            scenario_type=spec.canonical_type or "compositional",
            severity_level=risk.risk_level,
            primary_actor_type=actor.primary_actor,
            side=_direction_to_legacy_side(inter.conflict_direction),
            pedestrian_emerge=actor.primary_actor == "pedestrian",
            occluder_type=spec.object_layer.occlusion.occluder_type,
            insert_occluder_if_missing=spec.object_layer.occlusion.enabled,
            ttc_min_s=float(risk.ttc_range_s[0]),
            ttc_max_s=float(risk.ttc_range_s[1]),
            conflict_point_x_m=float(0.5 * (risk.longitudinal_distance_range_m[0] + risk.longitudinal_distance_range_m[1])),
            conflict_point_y_m=0.0,
            target_gap_min_m=float(risk.gap_range_m[0]),
            target_gap_max_m=float(risk.gap_range_m[1]),
            lead_distance_min_m=float(risk.gap_range_m[0]),
            lead_distance_max_m=float(risk.gap_range_m[1]),
            pedestrian_speed=float(risk.target_actor_speed_mps),
            target_speed_mps=float(risk.target_actor_speed_mps),
            relative_speed_min_mps=max(0.0, float(risk.target_relative_speed_mps) - 1.0),
            relative_speed_max_mps=float(risk.target_relative_speed_mps) + 1.0,
            target_decel_min_mps2=float(risk.target_decel_range_mps2[0]),
            target_decel_max_mps2=float(risk.target_decel_range_mps2[1]),
            allow_actor_insertion=actor.allow_actor_insertion,
            prefer_existing_actor=actor.prefer_existing_actor,
            matched_rules=[f"semantic_signature:{spec.semantic_signature}"],
            debug={"hazard_spec_id": spec.spec_id},
        )


def _direction_to_legacy_side(direction: str) -> str:
    if direction in {"left_to_right", "left_merge"}:
        return "left"
    if direction in {"right_to_left", "right_merge"}:
        return "right"
    return "auto"
