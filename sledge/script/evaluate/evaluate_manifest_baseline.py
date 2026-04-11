#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

_THIS = Path(__file__).resolve()
for _p in _THIS.parents:
    if (_p / "sledge").is_dir():
        sys.path.append(str(_p))
        break

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    AgentIndex,
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.semantic_control import NaturalLanguagePromptParser, PromptAlignmentEvaluator
from sledge.script.run_half_denoise_from_tiered_cache import (
    basic_scene_compliance,
    make_simulation_compatible_vector,
    summarize_multiscenario_semantics,
)

RAW_KEYS = {
    "lines",
    "vehicles",
    "pedestrians",
    "static_objects",
    "green_lights",
    "red_lights",
    "ego",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified vector-only evaluation with CSR / CSR_strict / MPA / CR / SQS / ROI-SQS / RSR."
    )
    parser.add_argument("--manifest", required=True, help="Path to scenario_manifest.csv")
    parser.add_argument(
        "--which",
        choices=["edited", "generated", "compare"],
        required=True,
        help="edited=B1, generated=B2, compare=B1 vs B2",
    )
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--edited-root", default=None, help="Root of edited raw cache. Required for compare/generated path mapping.")
    parser.add_argument("--generated-root", default=None, help="Root of generated scenario cache. Required for generated/compare.")

    parser.add_argument("--alignment-threshold", type=float, default=0.70)
    parser.add_argument("--sqs-threshold", type=float, default=0.75)
    parser.add_argument("--roi-sqs-threshold", type=float, default=0.75)

    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--manifest-min-alignment", type=float, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _deserialize_elem(obj: Any) -> SledgeVectorElement:
    if isinstance(obj, SledgeVectorElement):
        return obj
    if isinstance(obj, dict) and "states" in obj and "mask" in obj:
        return SledgeVectorElement.deserialize(obj)
    raise TypeError(f"Unsupported vector element type: {type(obj)}")


def _dict_kind(obj: Dict[str, Any]) -> str:
    if not (isinstance(obj, dict) and RAW_KEYS.issubset(obj.keys())):
        return "unknown"
    lines = obj["lines"]
    if not isinstance(lines, dict):
        return "unknown"
    states = np.asarray(lines.get("states"))
    if states.ndim >= 3 and states.shape[-1] == 2:
        return "vector"
    return "raw"


def _dict_to_vector(obj: Dict[str, Any]) -> SledgeVector:
    return SledgeVector(
        lines=_deserialize_elem(obj["lines"]),
        vehicles=_deserialize_elem(obj["vehicles"]),
        pedestrians=_deserialize_elem(obj["pedestrians"]),
        static_objects=_deserialize_elem(obj["static_objects"]),
        green_lights=_deserialize_elem(obj["green_lights"]),
        red_lights=_deserialize_elem(obj["red_lights"]),
        ego=_deserialize_elem(obj["ego"]),
    )


def _dict_to_raw(obj: Dict[str, Any]) -> SledgeVectorRaw:
    return SledgeVectorRaw(
        lines=_deserialize_elem(obj["lines"]),
        vehicles=_deserialize_elem(obj["vehicles"]),
        pedestrians=_deserialize_elem(obj["pedestrians"]),
        static_objects=_deserialize_elem(obj["static_objects"]),
        green_lights=_deserialize_elem(obj["green_lights"]),
        red_lights=_deserialize_elem(obj["red_lights"]),
        ego=_deserialize_elem(obj["ego"]),
    )


def _load_pickle_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _load_scene_as_vector(path: Path, ae_config: RVAEConfig) -> Tuple[SledgeVector, Optional[SledgeVectorRaw], str]:
    obj = _load_pickle_gz(path)
    src = type(obj).__name__

    if isinstance(obj, SledgeVector):
        return obj, None, "SledgeVector"

    if isinstance(obj, SledgeVectorRaw):
        vec, _ = sledge_raw_feature_processing(obj, ae_config)
        return vec, obj, "SledgeVectorRaw->processed_vector"

    if isinstance(obj, dict) and RAW_KEYS.issubset(obj.keys()):
        kind = _dict_kind(obj)
        if kind == "vector":
            return _dict_to_vector(obj), None, "dict_vector"
        raw = _dict_to_raw(obj)
        vec, _ = sledge_raw_feature_processing(raw, ae_config)
        return vec, raw, "dict_raw->processed_vector"

    raise TypeError(f"Unsupported scene payload type: {type(obj)}")


def _valid_rows(states: np.ndarray, mask: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    if states.size == 0:
        return np.zeros((0, states.shape[-1] if states.ndim > 0 else 0), dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    if mask.ndim == 0:
        mask = np.asarray([mask])
    valid = np.asarray(mask).astype(float) >= thresh
    return states[valid]


def _resolve_edited_scene_path(row: Dict[str, str]) -> Path:
    return Path(row["output_dir"]).resolve() / "sledge_raw.gz"


def _resolve_generated_scene_path(row: Dict[str, str], edited_root: Path, generated_root: Path) -> Path:
    rel = Path(row["output_dir"]).resolve().relative_to(edited_root.resolve())
    return generated_root.resolve() / rel / "sledge_vector.gz"


def _filter_manifest_rows(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if args.accepted_only and not _truthy(row.get("accepted", False)):
            continue
        if args.manifest_min_alignment is not None:
            if _safe_float(row.get("edited_alignment", 0.0), 0.0) < float(args.manifest_min_alignment):
                continue
        out.append(row)
    if args.max_scenes is not None:
        out = out[: args.max_scenes]
    return out


def _collect_entity_arrays(vector: SledgeVector) -> Dict[str, np.ndarray]:
    return {
        "vehicles": _valid_rows(np.asarray(vector.vehicles.states), np.asarray(vector.vehicles.mask)),
        "pedestrians": _valid_rows(np.asarray(vector.pedestrians.states), np.asarray(vector.pedestrians.mask)),
        "statics": _valid_rows(np.asarray(vector.static_objects.states), np.asarray(vector.static_objects.mask)),
        "lines": _valid_rows(np.asarray(vector.lines.states), np.asarray(vector.lines.mask)),
        "ego": np.asarray(vector.ego.states).reshape(-1),
    }


def _entity_radius(length: float, width: float) -> float:
    return 0.35 * math.sqrt(max(length, 0.1) ** 2 + max(width, 0.1) ** 2)


def _count_overlap_penalty(arr_a: np.ndarray, arr_b: np.ndarray, same_group: bool = False) -> float:
    if len(arr_a) == 0 or len(arr_b) == 0:
        return 0.0
    penalty = 0.0
    for i, a in enumerate(arr_a):
        j_start = i + 1 if same_group else 0
        for j in range(j_start, len(arr_b)):
            b = arr_b[j]
            dx = float(a[AgentIndex.X] - b[AgentIndex.X])
            dy = float(a[AgentIndex.Y] - b[AgentIndex.Y])
            dist = math.hypot(dx, dy)
            ra = _entity_radius(float(a[AgentIndex.LENGTH]), float(a[AgentIndex.WIDTH]))
            rb = _entity_radius(float(b[AgentIndex.LENGTH]), float(b[AgentIndex.WIDTH]))
            overlap = max(0.0, (ra + rb) - dist)
            if overlap > 0.0:
                penalty += min(1.0, overlap / max(1e-3, ra + rb))
    return penalty


def numeric_validity_score(vector: SledgeVector) -> float:
    elems = [
        vector.lines,
        vector.vehicles,
        vector.pedestrians,
        vector.static_objects,
        vector.green_lights,
        vector.red_lights,
        vector.ego,
    ]
    total_checks = 0
    failures = 0
    for elem in elems:
        total_checks += 2
        if np.isnan(np.asarray(elem.states)).any() or np.isinf(np.asarray(elem.states)).any():
            failures += 1
        if np.isnan(np.asarray(elem.mask)).any() or np.isinf(np.asarray(elem.mask)).any():
            failures += 1
    if np.asarray(vector.ego.states).size == 0:
        failures += 1
        total_checks += 1
    return float(np.clip(1.0 - failures / max(1, total_checks), 0.0, 1.0))


def overlap_free_score(vector: SledgeVector) -> float:
    data = _collect_entity_arrays(vector)
    vehicles = data["vehicles"]
    pedestrians = data["pedestrians"]
    statics = data["statics"]

    penalty = 0.0
    penalty += 1.00 * _count_overlap_penalty(vehicles, vehicles, same_group=True)
    penalty += 0.70 * _count_overlap_penalty(vehicles, pedestrians, same_group=False)
    penalty += 0.80 * _count_overlap_penalty(vehicles, statics, same_group=False)
    penalty += 0.50 * _count_overlap_penalty(pedestrians, pedestrians, same_group=True)

    ego_r = _entity_radius(4.9, 2.0)
    for arr, idx_len, idx_w in [
        (vehicles, AgentIndex.LENGTH, AgentIndex.WIDTH),
        (pedestrians, AgentIndex.LENGTH, AgentIndex.WIDTH),
    ]:
        for a in arr:
            dist = math.hypot(float(a[AgentIndex.X]), float(a[AgentIndex.Y]))
            rr = _entity_radius(float(a[idx_len]), float(a[idx_w]))
            overlap = max(0.0, (rr + ego_r) - dist)
            if overlap > 0.0:
                penalty += min(1.0, overlap / max(1e-3, rr + ego_r))

    denom = max(1.0, len(vehicles) + len(pedestrians) + len(statics) + 1)
    return float(np.clip(1.0 - penalty / denom, 0.0, 1.0))


def _line_points_and_dirs(lines_elem: SledgeVectorElement) -> Tuple[np.ndarray, np.ndarray]:
    states = np.asarray(lines_elem.states)
    mask = np.asarray(lines_elem.mask)
    valid = _valid_rows(states, mask)
    if valid.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    pts = []
    dirs = []
    for line in valid:
        if line.ndim != 2 or line.shape[-1] < 2 or len(line) < 2:
            continue
        for k in range(len(line) - 1):
            p0 = np.asarray(line[k, :2], dtype=np.float32)
            p1 = np.asarray(line[k + 1, :2], dtype=np.float32)
            pts.append(p0)
            dirs.append(float(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))))
        pts.append(np.asarray(line[-1, :2], dtype=np.float32))
        dirs.append(dirs[-1] if dirs else 0.0)
    if not pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(pts, axis=0), np.asarray(dirs, dtype=np.float32)


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def lane_consistency_score(vector: SledgeVector, scenario_type: str) -> float:
    vehicles = _valid_rows(np.asarray(vector.vehicles.states), np.asarray(vector.vehicles.mask))
    if len(vehicles) == 0:
        return 0.75
    line_pts, line_dirs = _line_points_and_dirs(vector.lines)
    if len(line_pts) == 0:
        return 0.50

    scores = []
    for v in vehicles:
        px, py = float(v[AgentIndex.X]), float(v[AgentIndex.Y])
        d = np.linalg.norm(line_pts - np.asarray([px, py], dtype=np.float32), axis=1)
        idx = int(np.argmin(d))
        lane_dist = float(d[idx])
        lane_heading = float(line_dirs[idx])
        veh_heading = float(v[AgentIndex.HEADING])
        heading_diff = abs(_wrap_angle(veh_heading - lane_heading))
        pos_score_regular = float(np.clip(1.0 - lane_dist / 2.2, 0.0, 1.0))
        heading_score = float(np.clip(1.0 - heading_diff / 0.8, 0.0, 1.0))

        if scenario_type == "cut_in":
            lat_abs = abs(py)
            if lat_abs <= 0.6:
                pos_score = max(pos_score_regular, 0.92)
            elif lat_abs <= 1.8:
                pos_score = max(pos_score_regular, 0.75)
            elif lat_abs <= 2.6:
                pos_score = max(pos_score_regular, 0.50)
            else:
                pos_score = pos_score_regular
            s = 0.65 * pos_score + 0.35 * heading_score
        else:
            s = 0.60 * pos_score_regular + 0.40 * heading_score
        if 0.0 < px < 25.0:
            s = min(1.0, s * 1.05)
        scores.append(float(np.clip(s, 0.0, 1.0)))
    return _mean(scores)


def lane_connectivity_score(vector: SledgeVector) -> float:
    states = np.asarray(vector.lines.states)
    mask = np.asarray(vector.lines.mask)
    lines = _valid_rows(states, mask)
    if len(lines) == 0:
        return 0.40

    segments = []
    for line in lines:
        if line.ndim != 2 or len(line) < 2:
            continue
        start = np.asarray(line[0, :2], dtype=np.float32)
        end = np.asarray(line[-1, :2], dtype=np.float32)
        mean_x = float(np.mean(line[:, 0]))
        head0 = float(math.atan2(float(line[min(1, len(line)-1), 1] - line[0, 1]), float(line[min(1, len(line)-1), 0] - line[0, 0])))
        head1 = float(math.atan2(float(line[-1, 1] - line[max(0, len(line)-2), 1]), float(line[-1, 0] - line[max(0, len(line)-2), 0])))
        segments.append((start, end, mean_x, head0, head1, line))

    if not segments:
        return 0.45

    anchor = np.asarray([5.0, 0.0], dtype=np.float32)
    seed_idx = int(np.argmin([np.min(np.linalg.norm(seg[5][:, :2] - anchor[None, :], axis=1)) for seg in segments]))

    adj = defaultdict(list)
    connection_gaps = []
    connection_heading = []

    for i, seg_i in enumerate(segments):
        end_i = seg_i[1]
        head_i = seg_i[4]
        mx_i = seg_i[2]
        for j, seg_j in enumerate(segments):
            if i == j:
                continue
            start_j = seg_j[0]
            head_j = seg_j[3]
            mx_j = seg_j[2]
            gap = float(np.linalg.norm(end_i - start_j))
            hdiff = abs(_wrap_angle(head_i - head_j))
            if gap <= 2.8 and hdiff <= 0.65 and mx_j >= mx_i - 1.0:
                adj[i].append((j, gap, hdiff))
                connection_gaps.append(gap)
                connection_heading.append(hdiff)

    visited = {seed_idx}
    frontier = [seed_idx]
    max_x = segments[seed_idx][2]
    total_reachable = 1
    while frontier:
        cur = frontier.pop(0)
        max_x = max(max_x, segments[cur][2])
        for nxt, _, _ in adj.get(cur, []):
            if nxt not in visited:
                visited.add(nxt)
                frontier.append(nxt)
                total_reachable += 1

    ego_forward_path_score = float(np.clip((max_x + 10.0) / 25.0, 0.0, 1.0))
    if connection_gaps:
        gap_smoothness_score = float(np.clip(1.0 - float(np.mean(connection_gaps)) / 2.8, 0.0, 1.0))
        heading_continuity_score = float(np.clip(1.0 - float(np.mean(connection_heading)) / 0.65, 0.0, 1.0))
    else:
        gap_smoothness_score = 0.55
        heading_continuity_score = 0.55

    connectivity = 0.50 * ego_forward_path_score + 0.30 * gap_smoothness_score + 0.20 * heading_continuity_score
    if total_reachable <= 1:
        connectivity *= 0.8
    return float(np.clip(connectivity, 0.0, 1.0))


def layout_plausibility_score(vector: SledgeVector, scenario_type: str) -> float:
    data = _collect_entity_arrays(vector)
    vehicles = data["vehicles"]
    pedestrians = data["pedestrians"]
    statics = data["statics"]

    frame_scores = []
    for arr in [vehicles, pedestrians, statics]:
        if len(arr) == 0:
            continue
        xs = arr[:, 0]
        ys = arr[:, 1]
        inside = ((xs > -20.0) & (xs < 50.0) & (np.abs(ys) < 15.0)).astype(np.float32)
        frame_scores.append(float(np.mean(inside)))
    base = _mean(frame_scores) if frame_scores else 0.8

    bonus = 0.0
    if scenario_type == "hard_brake" and len(vehicles) > 0:
        same_lane = vehicles[(vehicles[:, AgentIndex.X] > 2.0) & (vehicles[:, AgentIndex.X] < 25.0) & (np.abs(vehicles[:, AgentIndex.Y]) < 1.8)]
        bonus = 0.1 if len(same_lane) > 0 else -0.1
    elif scenario_type in {"pedestrian_crossing", "sudden_pedestrian_crossing"} and len(pedestrians) > 0:
        py = np.abs(pedestrians[:, AgentIndex.Y])
        bonus = 0.1 if np.any((py > 1.5) & (py < 8.0)) else -0.1
    elif scenario_type == "cut_in" and len(vehicles) > 0:
        cand = vehicles[(vehicles[:, AgentIndex.X] > 1.0) & (vehicles[:, AgentIndex.X] < 20.0) & (np.abs(vehicles[:, AgentIndex.Y]) < 2.8)]
        bonus = 0.1 if len(cand) > 0 else -0.1

    return float(np.clip(base + bonus, 0.0, 1.0))


def kinematic_plausibility_score(vector: SledgeVector) -> float:
    vehicles = _valid_rows(np.asarray(vector.vehicles.states), np.asarray(vector.vehicles.mask))
    pedestrians = _valid_rows(np.asarray(vector.pedestrians.states), np.asarray(vector.pedestrians.mask))

    scores = []
    if len(vehicles) > 0:
        v_speed = vehicles[:, AgentIndex.VELOCITY]
        v_heading = np.abs(vehicles[:, AgentIndex.HEADING])
        v_len = vehicles[:, AgentIndex.LENGTH]
        v_wid = vehicles[:, AgentIndex.WIDTH]

        speed_score = np.mean(np.clip(1.0 - np.maximum(0.0, v_speed - 18.0) / 10.0, 0.0, 1.0))
        heading_score = np.mean(np.clip(1.0 - np.maximum(0.0, v_heading - 1.2) / 1.5, 0.0, 1.0))
        size_ok = ((v_len > 2.5) & (v_len < 8.5) & (v_wid > 1.3) & (v_wid < 3.2)).astype(np.float32)
        size_score = float(np.mean(size_ok))
        scores.append(float(0.45 * speed_score + 0.20 * heading_score + 0.35 * size_score))

    if len(pedestrians) > 0:
        p_speed = pedestrians[:, AgentIndex.VELOCITY]
        p_score = np.mean(np.clip(1.0 - np.maximum(0.0, p_speed - 4.0) / 3.0, 0.0, 1.0))
        scores.append(float(p_score))

    return _mean(scores) if scores else 0.85


def compute_sqs(vector: SledgeVector, scenario_type: str) -> Dict[str, float]:
    numeric = numeric_validity_score(vector)
    overlap = overlap_free_score(vector)
    lane_cons = lane_consistency_score(vector, scenario_type=scenario_type)
    lane_conn = lane_connectivity_score(vector)
    layout = layout_plausibility_score(vector, scenario_type=scenario_type)
    kinematic = kinematic_plausibility_score(vector)

    sqs = (
        0.10 * numeric
        + 0.20 * overlap
        + 0.20 * lane_cons
        + 0.20 * lane_conn
        + 0.15 * layout
        + 0.15 * kinematic
    )
    return {
        "numeric_validity_score": float(numeric),
        "overlap_free_score": float(overlap),
        "lane_consistency_score": float(lane_cons),
        "lane_connectivity_score": float(lane_conn),
        "layout_plausibility_score": float(layout),
        "kinematic_plausibility_score": float(kinematic),
        "sqs": float(np.clip(sqs, 0.0, 1.0)),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _roi_from_edit_report(scene_dir: Path) -> List[Dict[str, float]]:
    path = scene_dir / "edit_report.json"
    if not path.exists():
        return []
    try:
        payload = _load_json(path)
        return list(payload.get("preserved_rois", []))
    except Exception:
        return []


def _point_in_any_roi(x: float, y: float, rois: List[Dict[str, float]]) -> bool:
    for roi in rois:
        if float(roi["x_min"]) <= x <= float(roi["x_max"]) and float(roi["y_min"]) <= y <= float(roi["y_max"]):
            return True
    return False


def _filter_elem_by_roi(elem: SledgeVectorElement, rois: List[Dict[str, float]], is_line: bool = False) -> SledgeVectorElement:
    states = np.asarray(elem.states).copy()
    mask = np.asarray(elem.mask).copy()

    if states.size == 0:
        return SledgeVectorElement(states=states, mask=mask)

    if is_line:
        if states.ndim != 3:
            return SledgeVectorElement(states=states, mask=np.zeros_like(mask))
        keep = []
        for line in states:
            ok = False
            for pt in line:
                if _point_in_any_roi(float(pt[0]), float(pt[1]), rois):
                    ok = True
                    break
            keep.append(ok)
        keep = np.asarray(keep, dtype=bool)
        if mask.ndim == 0:
            mask = np.asarray([mask])
        new_mask = (np.asarray(mask).astype(bool) & keep)
        return SledgeVectorElement(states=states, mask=new_mask)

    if states.ndim == 1:
        states = states[None, :]
        if mask.ndim == 0:
            mask = np.asarray([mask])

    keep = []
    for row in states:
        keep.append(_point_in_any_roi(float(row[0]), float(row[1]), rois))
    keep = np.asarray(keep, dtype=bool)
    new_mask = (np.asarray(mask).astype(bool) & keep)
    return SledgeVectorElement(states=states, mask=new_mask)


def build_roi_vector(vector: SledgeVector, rois: List[Dict[str, float]]) -> SledgeVector:
    if not rois:
        return vector
    return SledgeVector(
        lines=_filter_elem_by_roi(vector.lines, rois, is_line=True),
        vehicles=_filter_elem_by_roi(vector.vehicles, rois, is_line=False),
        pedestrians=_filter_elem_by_roi(vector.pedestrians, rois, is_line=False),
        static_objects=_filter_elem_by_roi(vector.static_objects, rois, is_line=False),
        green_lights=_filter_elem_by_roi(vector.green_lights, rois, is_line=True),
        red_lights=_filter_elem_by_roi(vector.red_lights, rois, is_line=True),
        ego=vector.ego,
    )


def _evaluate_vector_scene(
    vector: SledgeVector,
    raw_scene: Optional[SledgeVectorRaw],
    prompt_spec: Any,
    alignment_evaluator: PromptAlignmentEvaluator,
    threshold: float,
    sqs_threshold: float,
    roi_sqs_threshold: float,
    rois: List[Dict[str, float]],
) -> Dict[str, Any]:
    alignment = alignment_evaluator.evaluate(vector, prompt_spec)
    semantic = summarize_multiscenario_semantics(
        alignment=alignment,
        prompt_spec=prompt_spec,
        vector=vector,
        threshold=float(threshold),
    )

    if raw_scene is not None:
        sim_vector = make_simulation_compatible_vector(vector, raw_scene)
        compliance = basic_scene_compliance(sim_vector)
    else:
        compliance = basic_scene_compliance(vector)

    sqs_dict = compute_sqs(vector, scenario_type=str(getattr(prompt_spec, "scenario_type", "generic")))
    roi_vector = build_roi_vector(vector, rois)
    roi_sqs_dict = compute_sqs(roi_vector, scenario_type=str(getattr(prompt_spec, "scenario_type", "generic")))

    return {
        "alignment_total": float(getattr(alignment, "total", 0.0)),
        "alignment_accepted": bool(getattr(alignment, "accepted", False)),
        "semantic_pass": bool(semantic["semantic_pass"]),
        "compliant": bool(compliance["compliant"]),
        "compliance_issues": list(compliance["issues"]),
        "semantic_summary": semantic,
        "alignment_details": dict(getattr(alignment, "details", {}) or {}),
        "alignment_notes": list(getattr(alignment, "notes", []) or []),
        **sqs_dict,
        "sqs_threshold": float(sqs_threshold),
        "quality_pass": bool(float(sqs_dict["sqs"]) >= float(sqs_threshold)),
        "roi_sqs": float(roi_sqs_dict["sqs"]),
        "roi_sqs_threshold": float(roi_sqs_threshold),
        "roi_quality_pass": bool(float(roi_sqs_dict["sqs"]) >= float(roi_sqs_threshold)),
    }


def _group_stats_single(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key_name, "unknown"))].append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, items in groups.items():
        out[key] = {
            "N": len(items),
            "CSR": _mean([1.0 if bool(r["alignment_accepted"]) else 0.0 for r in items]),
            "CSR_strict": _mean([1.0 if bool(r["semantic_pass"]) else 0.0 for r in items]),
            "MPA": _mean([float(r["alignment_total"]) for r in items]),
            "CR": _mean([1.0 if bool(r["compliant"]) else 0.0 for r in items]),
            "SQS": _mean([float(r["sqs"]) for r in items]),
            "QSR": _mean([1.0 if bool(r["quality_pass"]) else 0.0 for r in items]),
            "ROI_SQS": _mean([float(r["roi_sqs"]) for r in items]),
            "ROI_QSR": _mean([1.0 if bool(r["roi_quality_pass"]) else 0.0 for r in items]),
            "numeric_validity_score": _mean([float(r["numeric_validity_score"]) for r in items]),
            "overlap_free_score": _mean([float(r["overlap_free_score"]) for r in items]),
            "lane_consistency_score": _mean([float(r["lane_consistency_score"]) for r in items]),
            "lane_connectivity_score": _mean([float(r["lane_connectivity_score"]) for r in items]),
            "layout_plausibility_score": _mean([float(r["layout_plausibility_score"]) for r in items]),
            "kinematic_plausibility_score": _mean([float(r["kinematic_plausibility_score"]) for r in items]),
        }
    return out


def _group_stats_compare(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(key_name, "unknown"))].append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, items in groups.items():
        out[key] = {
            "N": len(items),
            "edited_MPA": _mean([float(r["edited_alignment_total"]) for r in items]),
            "generated_MPA": _mean([float(r["generated_alignment_total"]) for r in items]),
            "delta_MPA": _mean([float(r["delta_MPA"]) for r in items]),
            "edited_SQS": _mean([float(r["edited_sqs"]) for r in items]),
            "generated_SQS": _mean([float(r["generated_sqs"]) for r in items]),
            "delta_SQS": _mean([float(r["delta_SQS"]) for r in items]),
            "edited_ROI_SQS": _mean([float(r["edited_roi_sqs"]) for r in items]),
            "generated_ROI_SQS": _mean([float(r["generated_roi_sqs"]) for r in items]),
            "delta_ROI_SQS": _mean([float(r["delta_ROI_SQS"]) for r in items]),
            "generated_CSR": _mean([1.0 if bool(r["generated_alignment_accepted"]) else 0.0 for r in items]),
            "generated_CSR_strict": _mean([1.0 if bool(r["generated_semantic_pass"]) else 0.0 for r in items]),
            "generated_CR": _mean([1.0 if bool(r["generated_compliant"]) else 0.0 for r in items]),
            "generated_QSR": _mean([1.0 if bool(r["generated_quality_pass"]) else 0.0 for r in items]),
            "generated_ROI_QSR": _mean([1.0 if bool(r["generated_roi_quality_pass"]) else 0.0 for r in items]),
            "RSR": _mean([float(r["RSR"]) for r in items]),
        }
    return out


def _write_group_csv(path: Path, rows: Dict[str, Dict[str, Any]], key_name: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([key_name])
        return
    metric_keys = list(next(iter(rows.values())).keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name] + metric_keys)
        writer.writeheader()
        for key, val in rows.items():
            writer.writerow({key_name: key, **val})


def main() -> None:
    args = build_argparser().parse_args()

    if args.which in {"generated", "compare"} and (not args.generated_root or not args.edited_root):
        raise ValueError("--edited-root and --generated-root are required for which=generated/compare")

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    ae_cfg_dict = OmegaConf.to_container(cfg.autoencoder_model.config, resolve=True)
    if not isinstance(ae_cfg_dict, dict):
        raise TypeError(f"Expected autoencoder_model.config to resolve to dict, got {type(ae_cfg_dict)}")
    filtered = {k: v for k, v in ae_cfg_dict.items() if k in RVAEConfig.__annotations__}
    ae_config = RVAEConfig(**filtered)

    prompt_parser = NaturalLanguagePromptParser()
    alignment_evaluator = PromptAlignmentEvaluator()

    manifest_rows = _load_manifest_rows(Path(args.manifest).resolve())
    manifest_rows = _filter_manifest_rows(manifest_rows, args)

    edited_root = Path(args.edited_root).resolve() if args.edited_root else None
    generated_root = Path(args.generated_root).resolve() if args.generated_root else None

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as fp:
        for idx, row in enumerate(manifest_rows, start=1):
            try:
                prompt = str(row["prompt"])
                scenario_type = str(row["scenario_type"])
                severity_level = str(row["severity_level"])

                prompt_spec = prompt_parser.parse(prompt)
                prompt_spec.scenario_type = scenario_type
                prompt_spec.severity_level = severity_level

                base_record = {
                    "scene_index": idx,
                    "which": args.which,
                    "prompt": prompt,
                    "scenario_type": scenario_type,
                    "severity_level": severity_level,
                    "manifest_edited_alignment": _safe_float(row.get("edited_alignment", 0.0), 0.0),
                    "manifest_accepted": _truthy(row.get("accepted", False)),
                }

                edited_scene_dir = Path(row["output_dir"]).resolve()
                rois = _roi_from_edit_report(edited_scene_dir)

                if args.which in {"edited", "compare"}:
                    edited_scene_path = _resolve_edited_scene_path(row)
                    edited_vector, edited_raw, edited_fmt = _load_scene_as_vector(edited_scene_path, ae_config)
                    edited_eval = _evaluate_vector_scene(
                        vector=edited_vector,
                        raw_scene=edited_raw,
                        prompt_spec=prompt_spec,
                        alignment_evaluator=alignment_evaluator,
                        threshold=float(args.alignment_threshold),
                        sqs_threshold=float(args.sqs_threshold),
                        roi_sqs_threshold=float(args.roi_sqs_threshold),
                        rois=rois,
                    )
                    if args.which == "edited":
                        record = {
                            **base_record,
                            "scene_path": str(edited_scene_path),
                            "source_format": edited_fmt,
                            **edited_eval,
                        }
                    else:
                        record = {
                            **base_record,
                            "edited_scene_path": str(edited_scene_path),
                            "edited_source_format": edited_fmt,
                            **{f"edited_{k}": v for k, v in edited_eval.items()},
                        }
                else:
                    record = dict(base_record)

                if args.which in {"generated", "compare"}:
                    generated_scene_path = _resolve_generated_scene_path(row, edited_root, generated_root)
                    if not generated_scene_path.exists():
                        raise FileNotFoundError(f"Generated scene not found: {generated_scene_path}")
                    generated_vector, generated_raw, generated_fmt = _load_scene_as_vector(generated_scene_path, ae_config)
                    generated_eval = _evaluate_vector_scene(
                        vector=generated_vector,
                        raw_scene=generated_raw,
                        prompt_spec=prompt_spec,
                        alignment_evaluator=alignment_evaluator,
                        threshold=float(args.alignment_threshold),
                        sqs_threshold=float(args.sqs_threshold),
                        roi_sqs_threshold=float(args.roi_sqs_threshold),
                        rois=rois,
                    )
                    if args.which == "generated":
                        record = {
                            **base_record,
                            "scene_path": str(generated_scene_path),
                            "source_format": generated_fmt,
                            **generated_eval,
                            "RSR": 1.0 if (
                                generated_eval["semantic_pass"]
                                and generated_eval["compliant"]
                                and generated_eval["roi_quality_pass"]
                            ) else 0.0,
                        }
                    else:
                        record.update(
                            {
                                "generated_scene_path": str(generated_scene_path),
                                "generated_source_format": generated_fmt,
                                **{f"generated_{k}": v for k, v in generated_eval.items()},
                                "delta_MPA": float(generated_eval["alignment_total"]) - float(edited_eval["alignment_total"]),
                                "delta_SQS": float(generated_eval["sqs"]) - float(edited_eval["sqs"]),
                                "delta_ROI_SQS": float(generated_eval["roi_sqs"]) - float(edited_eval["roi_sqs"]),
                                "RSR": 1.0 if (
                                    generated_eval["semantic_pass"]
                                    and generated_eval["compliant"]
                                    and generated_eval["roi_quality_pass"]
                                ) else 0.0,
                            }
                        )

                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.append(record)

            except Exception as exc:
                errors.append(
                    {
                        "scene_index": idx,
                        "row": row,
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    if args.which == "edited":
        summary_rows = results
        by_scenario_type = _group_stats_single(summary_rows, "scenario_type")
        by_severity_level = _group_stats_single(summary_rows, "severity_level")
        summary = {
            "which": args.which,
            "N": len(summary_rows),
            "num_errors": len(errors),
            "CSR": _mean([1.0 if bool(r["alignment_accepted"]) else 0.0 for r in summary_rows]),
            "CSR_strict": _mean([1.0 if bool(r["semantic_pass"]) else 0.0 for r in summary_rows]),
            "MPA": _mean([float(r["alignment_total"]) for r in summary_rows]),
            "CR": _mean([1.0 if bool(r["compliant"]) else 0.0 for r in summary_rows]),
            "SQS": _mean([float(r["sqs"]) for r in summary_rows]),
            "QSR": _mean([1.0 if bool(r["quality_pass"]) else 0.0 for r in summary_rows]),
            "ROI_SQS": _mean([float(r["roi_sqs"]) for r in summary_rows]),
            "ROI_QSR": _mean([1.0 if bool(r["roi_quality_pass"]) else 0.0 for r in summary_rows]),
            "by_scenario_type": by_scenario_type,
            "by_severity_level": by_severity_level,
        }
    elif args.which == "generated":
        summary_rows = results
        by_scenario_type = _group_stats_single(summary_rows, "scenario_type")
        by_severity_level = _group_stats_single(summary_rows, "severity_level")
        summary = {
            "which": args.which,
            "N": len(summary_rows),
            "num_errors": len(errors),
            "CSR": _mean([1.0 if bool(r["alignment_accepted"]) else 0.0 for r in summary_rows]),
            "CSR_strict": _mean([1.0 if bool(r["semantic_pass"]) else 0.0 for r in summary_rows]),
            "MPA": _mean([float(r["alignment_total"]) for r in summary_rows]),
            "CR": _mean([1.0 if bool(r["compliant"]) else 0.0 for r in summary_rows]),
            "SQS": _mean([float(r["sqs"]) for r in summary_rows]),
            "QSR": _mean([1.0 if bool(r["quality_pass"]) else 0.0 for r in summary_rows]),
            "ROI_SQS": _mean([float(r["roi_sqs"]) for r in summary_rows]),
            "ROI_QSR": _mean([1.0 if bool(r["roi_quality_pass"]) else 0.0 for r in summary_rows]),
            "RSR": _mean([float(r["RSR"]) for r in summary_rows]),
            "by_scenario_type": by_scenario_type,
            "by_severity_level": by_severity_level,
        }
    else:
        summary_rows = results
        by_scenario_type = _group_stats_compare(summary_rows, "scenario_type")
        by_severity_level = _group_stats_compare(summary_rows, "severity_level")
        summary = {
            "which": args.which,
            "N": len(summary_rows),
            "num_errors": len(errors),
            "edited_MPA": _mean([float(r["edited_alignment_total"]) for r in summary_rows]),
            "generated_MPA": _mean([float(r["generated_alignment_total"]) for r in summary_rows]),
            "delta_MPA": _mean([float(r["delta_MPA"]) for r in summary_rows]),
            "edited_SQS": _mean([float(r["edited_sqs"]) for r in summary_rows]),
            "generated_SQS": _mean([float(r["generated_sqs"]) for r in summary_rows]),
            "delta_SQS": _mean([float(r["delta_SQS"]) for r in summary_rows]),
            "edited_ROI_SQS": _mean([float(r["edited_roi_sqs"]) for r in summary_rows]),
            "generated_ROI_SQS": _mean([float(r["generated_roi_sqs"]) for r in summary_rows]),
            "delta_ROI_SQS": _mean([float(r["delta_ROI_SQS"]) for r in summary_rows]),
            "generated_CSR": _mean([1.0 if bool(r["generated_alignment_accepted"]) else 0.0 for r in summary_rows]),
            "generated_CSR_strict": _mean([1.0 if bool(r["generated_semantic_pass"]) else 0.0 for r in summary_rows]),
            "generated_CR": _mean([1.0 if bool(r["generated_compliant"]) else 0.0 for r in summary_rows]),
            "generated_QSR": _mean([1.0 if bool(r["generated_quality_pass"]) else 0.0 for r in summary_rows]),
            "generated_ROI_QSR": _mean([1.0 if bool(r["generated_roi_quality_pass"]) else 0.0 for r in summary_rows]),
            "RSR": _mean([float(r["RSR"]) for r in summary_rows]),
            "by_scenario_type": by_scenario_type,
            "by_severity_level": by_severity_level,
        }

    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "batch_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    _write_group_csv(out_dir / "by_scenario_type.csv", by_scenario_type, "scenario_type")
    _write_group_csv(out_dir / "by_severity_level.csv", by_severity_level, "severity_level")

    if errors:
        with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote: {out_dir / 'results.jsonl'}")
    print(f"[OK] wrote: {out_dir / 'batch_summary.json'}")
    print(f"[OK] wrote: {out_dir / 'by_scenario_type.csv'}")
    if args.which == "edited":
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"CSR={summary['CSR']:.6f}, CSR_strict={summary['CSR_strict']:.6f}, "
            f"MPA={summary['MPA']:.6f}, CR={summary['CR']:.6f}, "
            f"SQS={summary['SQS']:.6f}, ROI_SQS={summary['ROI_SQS']:.6f}"
        )
    elif args.which == "generated":
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"CSR={summary['CSR']:.6f}, CSR_strict={summary['CSR_strict']:.6f}, "
            f"MPA={summary['MPA']:.6f}, CR={summary['CR']:.6f}, "
            f"SQS={summary['SQS']:.6f}, ROI_SQS={summary['ROI_SQS']:.6f}, RSR={summary['RSR']:.6f}"
        )
    else:
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"edited_MPA={summary['edited_MPA']:.6f}, generated_MPA={summary['generated_MPA']:.6f}, "
            f"delta_MPA={summary['delta_MPA']:.6f}, edited_ROI_SQS={summary['edited_ROI_SQS']:.6f}, "
            f"generated_ROI_SQS={summary['generated_ROI_SQS']:.6f}, delta_ROI_SQS={summary['delta_ROI_SQS']:.6f}, "
            f"RSR={summary['RSR']:.6f}"
        )


if __name__ == "__main__":
    main()
