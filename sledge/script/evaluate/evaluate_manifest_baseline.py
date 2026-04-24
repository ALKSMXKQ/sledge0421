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
        choices=["original", "edited", "generated", "compare"],
        required=True,
        help="original=B0, edited=B1, generated=B2, compare=B1 vs B2",
    )
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--edited-root", default=None, help="Root of edited raw cache. Required for compare/generated path mapping.")
    parser.add_argument("--generated-root", default=None, help="Root of generated scenario cache. Required for generated/compare.")
    parser.add_argument("--original-root", default=None, help="Root of original autoencoder cache. Required for which=original.")

    parser.add_argument("--alignment-threshold", type=float, default=0.70)
    parser.add_argument("--sqs-threshold", type=float, default=0.75)
    parser.add_argument("--roi-sqs-threshold", type=float, default=0.75)

    parser.add_argument("--context-margin", type=float, default=6.0)
    parser.add_argument("--context-distance-cap", type=float, default=10.0)

    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--manifest-min-alignment", type=float, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _mean_optional(xs: List[Any]) -> Optional[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            fx = float(x)
        except Exception:
            continue
        if math.isnan(fx) or math.isinf(fx):
            continue
        vals.append(fx)
    return float(sum(vals) / len(vals)) if vals else None


def _soft_threshold_score(value: float, threshold: float, floor: float) -> float:
    value = float(value)
    threshold = float(threshold)
    floor = float(floor)

    if threshold <= floor:
        return 1.0 if value >= threshold else 0.0
    if value >= threshold:
        return 1.0
    if value <= floor:
        return 0.0
    return float((value - floor) / (threshold - floor))


def _compute_rsr_soft(
    alignment_total: float,
    semantic_pass: bool,
    compliant: bool,
    sqs: float,
    roi_sqs: float,
    alignment_threshold: float,
    sqs_threshold: float,
    roi_sqs_threshold: float,
) -> float:
    if not bool(compliant):
        return 0.0

    semantic_soft = 1.0 if bool(semantic_pass) else _soft_threshold_score(
        alignment_total, alignment_threshold, 0.55
    )
    quality_soft = _soft_threshold_score(sqs, sqs_threshold, 0.60)
    roi_soft = _soft_threshold_score(roi_sqs, roi_sqs_threshold, 0.60)

    rsr = 0.50 * semantic_soft + 0.15 * quality_soft + 0.35 * roi_soft
    return float(np.clip(rsr, 0.0, 1.0))


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
    states = np.asarray(states)
    mask = np.asarray(mask)
    if states.size == 0:
        width = states.shape[-1] if states.ndim > 0 else 0
        return np.zeros((0, width), dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    if mask.ndim == 0:
        mask = np.asarray([mask])
    valid = np.asarray(mask).astype(float) >= thresh
    if valid.ndim > 1:
        valid = np.any(valid, axis=tuple(range(1, valid.ndim)))
    valid = np.asarray(valid).reshape(-1)
    if len(valid) != len(states):
        valid = np.resize(valid, (len(states),))
    return np.asarray(states[valid], dtype=np.float32)


def _resolve_edited_scene_path(row: Dict[str, str]) -> Path:
    return Path(row["output_dir"]).resolve() / "sledge_raw.gz"


def _resolve_generated_scene_path(row: Dict[str, str], edited_root: Path, generated_root: Path) -> Path:
    rel = Path(row["output_dir"]).resolve().relative_to(edited_root.resolve())
    return generated_root.resolve() / rel / "sledge_vector.gz"

def _resolve_original_cache_scene_path(row: Dict[str, str], edited_root: Path, original_root: Path) -> Path:
    rel = Path(row["output_dir"]).resolve().relative_to(edited_root.resolve())
    return original_root.resolve() / rel / "sledge_raw.gz"


ORIGINAL_SCENE_PATH_KEYS = [
    "original_scene_path",
    "source_scene_path",
    "source_path",
    "input_scene_path",
    "original_path",
    "scene_path",
]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_original_scene_path(
    row: Dict[str, str],
    edited_scene_dir: Path,
    generated_scene_path: Optional[Path] = None,
) -> Optional[Path]:
    candidates: List[Path] = []

    for key in ORIGINAL_SCENE_PATH_KEYS:
        val = row.get(key)
        if val:
            candidates.append(Path(val))

    edit_report = edited_scene_dir / "edit_report.json"
    if edit_report.exists():
        try:
            payload = _load_json(edit_report)
            for key in ORIGINAL_SCENE_PATH_KEYS:
                val = payload.get(key)
                if val:
                    candidates.append(Path(val))
        except Exception:
            pass

    if generated_scene_path is not None:
        scenario_label = generated_scene_path.parent / "scenario_label.json"
        if scenario_label.exists():
            try:
                payload = _load_json(scenario_label)
                for key in ORIGINAL_SCENE_PATH_KEYS:
                    val = payload.get(key)
                    if val:
                        candidates.append(Path(val))
            except Exception:
                pass

    for cand in candidates:
        try:
            if cand.exists():
                return cand.resolve()
        except Exception:
            continue
    return None


def _resolve_original_input_scene_path(
    row: Dict[str, str],
    edited_scene_dir: Path,
    edited_root: Optional[Path],
    original_root: Optional[Path],
) -> Optional[Path]:
    direct = _resolve_original_scene_path(row, edited_scene_dir, None)
    if direct is not None:
        return direct
    if edited_root is not None and original_root is not None:
        try:
            mapped = _resolve_original_cache_scene_path(row, edited_root, original_root)
            if mapped.exists():
                return mapped.resolve()
        except Exception:
            pass
    return None


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


def _point_in_any_roi(x: float, y: float, rois: List[Dict[str, float]]) -> bool:
    for roi in rois:
        if float(roi["x_min"]) <= x <= float(roi["x_max"]) and float(roi["y_min"]) <= y <= float(roi["y_max"]):
            return True
    return False


def _roi_from_edit_report(scene_dir: Path) -> List[Dict[str, float]]:
    path = scene_dir / "edit_report.json"
    if not path.exists():
        return []
    try:
        payload = _load_json(path)
        return list(payload.get("preserved_rois", []))
    except Exception:
        return []


def _expand_rois(rois: List[Dict[str, float]], margin: float) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for roi in rois:
        out.append(
            {
                "x_min": float(roi["x_min"]) - float(margin),
                "x_max": float(roi["x_max"]) + float(margin),
                "y_min": float(roi["y_min"]) - float(margin),
                "y_max": float(roi["y_max"]) + float(margin),
            }
        )
    return out


def _point_in_any_box(x: float, y: float, boxes: List[Dict[str, float]]) -> bool:
    for box in boxes:
        if float(box["x_min"]) <= x <= float(box["x_max"]) and float(box["y_min"]) <= y <= float(box["y_max"]):
            return True
    return False


def _filter_elem_by_context_ring(
    elem: SledgeVectorElement,
    rois: List[Dict[str, float]],
    context_margin: float,
    is_line: bool = False,
) -> SledgeVectorElement:
    states = np.asarray(elem.states).copy()
    mask = np.asarray(elem.mask).copy()

    if states.size == 0 or not rois:
        return SledgeVectorElement(states=states, mask=np.zeros_like(mask))

    expanded = _expand_rois(rois, context_margin)

    if is_line:
        if states.ndim != 3:
            return SledgeVectorElement(states=states, mask=np.zeros_like(mask))
        keep = []
        for line in states:
            in_ctx = False
            in_roi = False
            for pt in line:
                x, y = float(pt[0]), float(pt[1])
                if _point_in_any_box(x, y, expanded):
                    in_ctx = True
                if _point_in_any_box(x, y, rois):
                    in_roi = True
            keep.append(in_ctx and not in_roi)
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
        x, y = float(row[0]), float(row[1])
        in_ctx = _point_in_any_box(x, y, expanded)
        in_roi = _point_in_any_box(x, y, rois)
        keep.append(in_ctx and not in_roi)
    keep = np.asarray(keep, dtype=bool)
    new_mask = (np.asarray(mask).astype(bool) & keep)
    return SledgeVectorElement(states=states, mask=new_mask)


def build_context_ring_vector(vector: SledgeVector, rois: List[Dict[str, float]], context_margin: float) -> SledgeVector:
    if not rois:
        return vector
    return SledgeVector(
        lines=_filter_elem_by_context_ring(vector.lines, rois, context_margin, is_line=True),
        vehicles=_filter_elem_by_context_ring(vector.vehicles, rois, context_margin, is_line=False),
        pedestrians=_filter_elem_by_context_ring(vector.pedestrians, rois, context_margin, is_line=False),
        static_objects=_filter_elem_by_context_ring(vector.static_objects, rois, context_margin, is_line=False),
        green_lights=_filter_elem_by_context_ring(vector.green_lights, rois, context_margin, is_line=True),
        red_lights=_filter_elem_by_context_ring(vector.red_lights, rois, context_margin, is_line=True),
        ego=vector.ego,
    )


def _filter_rows_by_context(arr: np.ndarray, rois: List[Dict[str, float]], context_margin: float) -> np.ndarray:
    if arr is None or len(arr) == 0 or not rois:
        return np.zeros((0, arr.shape[-1] if isinstance(arr, np.ndarray) and arr.ndim > 1 else 0), dtype=np.float32)
    expanded = _expand_rois(rois, context_margin)
    kept = []
    for row in arr:
        x = float(row[AgentIndex.X])
        y = float(row[AgentIndex.Y])
        in_context = _point_in_any_box(x, y, expanded)
        in_roi = _point_in_any_box(x, y, rois)
        if in_context and not in_roi:
            kept.append(row)
    if not kept:
        return np.zeros((0, arr.shape[-1]), dtype=np.float32)
    return np.stack(kept, axis=0)


def _context_positions(vector: SledgeVector, rois: List[Dict[str, float]], context_margin: float) -> Dict[str, np.ndarray]:
    veh = _filter_rows_by_context(
        _valid_rows(np.asarray(vector.vehicles.states), np.asarray(vector.vehicles.mask)),
        rois,
        context_margin,
    )
    ped = _filter_rows_by_context(
        _valid_rows(np.asarray(vector.pedestrians.states), np.asarray(vector.pedestrians.mask)),
        rois,
        context_margin,
    )
    sta = _filter_rows_by_context(
        _valid_rows(np.asarray(vector.static_objects.states), np.asarray(vector.static_objects.mask)),
        rois,
        context_margin,
    )
    return {
        "vehicles": veh[:, :2] if len(veh) > 0 else np.zeros((0, 2), dtype=np.float32),
        "pedestrians": ped[:, :2] if len(ped) > 0 else np.zeros((0, 2), dtype=np.float32),
        "statics": sta[:, :2] if len(sta) > 0 else np.zeros((0, 2), dtype=np.float32),
    }


def _bidirectional_chamfer_mean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    a_to_b = float(np.mean(np.min(dists, axis=1)))
    b_to_a = float(np.mean(np.min(dists, axis=0)))
    return 0.5 * (a_to_b + b_to_a)


def context_change_magnitude(
    original_vector: Optional[SledgeVector],
    target_vector: SledgeVector,
    rois: List[Dict[str, float]],
    context_margin: float,
    distance_cap: float,
) -> Optional[float]:
    if original_vector is None or not rois:
        return None

    original_ctx = _context_positions(original_vector, rois, context_margin)
    target_ctx = _context_positions(target_vector, rois, context_margin)

    class_scores: List[float] = []
    for key in ["vehicles", "pedestrians", "statics"]:
        a = original_ctx[key]
        b = target_ctx[key]
        if len(a) == 0 and len(b) == 0:
            continue
        chamfer = _bidirectional_chamfer_mean(a, b)
        if math.isinf(chamfer):
            chamfer_norm = 1.0
        else:
            chamfer_norm = float(np.clip(chamfer / max(float(distance_cap), 1e-6), 0.0, 1.0))
        count_gap = abs(len(a) - len(b)) / max(1, max(len(a), len(b)))
        class_change = 0.7 * chamfer_norm + 0.3 * float(count_gap)
        class_scores.append(float(np.clip(class_change, 0.0, 1.0)))

    return _mean(class_scores) if class_scores else 0.0


def _rows_in_rois(arr: np.ndarray, rois: List[Dict[str, float]]) -> np.ndarray:
    if arr is None or len(arr) == 0 or not rois:
        return np.zeros((0, arr.shape[-1] if isinstance(arr, np.ndarray) and arr.ndim > 1 else 0), dtype=np.float32)
    kept = []
    for row in arr:
        x = float(row[AgentIndex.X])
        y = float(row[AgentIndex.Y])
        if _point_in_any_box(x, y, rois):
            kept.append(row)
    if not kept:
        return np.zeros((0, arr.shape[-1]), dtype=np.float32)
    return np.stack(kept, axis=0)


def _triangular_score(x: float, low: float, mode: float, high: float) -> float:
    if x <= low or x >= high:
        return 0.0
    if x == mode:
        return 1.0
    if x < mode:
        return float((x - low) / max(mode - low, 1e-6))
    return float((high - x) / max(high - mode, 1e-6))


def _safe_mean_from_list(xs: List[float], default: float = 0.5) -> float:
    return float(sum(xs) / len(xs)) if xs else float(default)


def _state_value(arr: np.ndarray, idx: int, default: float = 0.0) -> float:
    try:
        flat = np.asarray(arr).reshape(-1)
        if idx < 0 or idx >= flat.size:
            return float(default)
        val = float(flat[idx])
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return float(default)


def _target_rows(vector: SledgeVector, scenario_type: str, rois: List[Dict[str, float]]) -> Tuple[np.ndarray, str]:
    vehicles = _valid_rows(np.asarray(vector.vehicles.states), np.asarray(vector.vehicles.mask))
    pedestrians = _valid_rows(np.asarray(vector.pedestrians.states), np.asarray(vector.pedestrians.mask))

    veh_roi = _rows_in_rois(vehicles, rois)
    ped_roi = _rows_in_rois(pedestrians, rois)

    if scenario_type in {"sudden_pedestrian_crossing", "pedestrian_crossing"}:
        if len(ped_roi) > 0:
            return ped_roi, "pedestrian"
        return pedestrians, "pedestrian"
    if scenario_type in {"hard_brake", "cut_in"}:
        if len(veh_roi) > 0:
            return veh_roi, "vehicle"
        return vehicles, "vehicle"
    if len(veh_roi) > 0:
        return veh_roi, "vehicle"
    if len(ped_roi) > 0:
        return ped_roi, "pedestrian"
    width = vehicles.shape[-1] if vehicles.ndim == 2 else (pedestrians.shape[-1] if pedestrians.ndim == 2 else 0)
    return np.zeros((0, width), dtype=np.float32), "unknown"


def _ego_row(vector: SledgeVector) -> np.ndarray:
    default = np.asarray([0.0, 0.0, 0.0, 0.0, 2.0, 4.9], dtype=np.float32)
    ego_arr = np.asarray(vector.ego.states)

    if ego_arr.size == 0:
        return default.copy()

    # Some payloads store ego as shape (1,), (1,1), or other malformed forms.
    # Prefer flattening, but always pad to a safe minimum length.
    ego = ego_arr.reshape(-1).astype(np.float32)

    need = max(
        int(getattr(AgentIndex, "X", 0)),
        int(getattr(AgentIndex, "Y", 1)),
        int(getattr(AgentIndex, "HEADING", 2)),
        int(getattr(AgentIndex, "VELOCITY", 3)),
        int(getattr(AgentIndex, "WIDTH", 4)),
        int(getattr(AgentIndex, "LENGTH", 5)),
    ) + 1

    if ego.size < need:
        padded = default.copy()
        padded[: min(ego.size, padded.size)] = ego[: min(ego.size, padded.size)]
        ego = padded
    return ego.astype(np.float32)


def compute_context_coherence(
    vector: SledgeVector,
    rois: List[Dict[str, float]],
    context_margin: float,
) -> Dict[str, float]:
    if not rois:
        context_overlap = float(overlap_free_score(vector))
        context_layout = float(layout_plausibility_score(vector, scenario_type="generic"))
        context_lane = float(lane_consistency_score(vector, scenario_type="generic"))
    else:
        ctx_vector = build_context_ring_vector(vector, rois, context_margin)
        context_overlap = float(overlap_free_score(ctx_vector))
        context_layout = float(layout_plausibility_score(ctx_vector, scenario_type="generic"))
        context_lane = float(lane_consistency_score(ctx_vector, scenario_type="generic"))
    context_coherence = float(np.clip(0.40 * context_overlap + 0.30 * context_layout + 0.30 * context_lane, 0.0, 1.0))
    return {
        "context_overlap_score": context_overlap,
        "context_layout_score": context_layout,
        "context_lane_score": context_lane,
        "context_coherence_score": context_coherence,
    }


def compute_context_roi_sqs(
    vector: SledgeVector,
    rois: List[Dict[str, float]],
    context_margin: float,
) -> Dict[str, float]:
    ctx = compute_context_coherence(vector, rois=rois, context_margin=context_margin)
    roi_sqs = float(np.clip(0.45 * ctx["context_overlap_score"] + 0.30 * ctx["context_layout_score"] + 0.25 * ctx["context_lane_score"], 0.0, 1.0))
    return {
        "roi_context_overlap_score": float(ctx["context_overlap_score"]),
        "roi_context_layout_score": float(ctx["context_layout_score"]),
        "roi_context_lane_score": float(ctx["context_lane_score"]),
        "roi_sqs": roi_sqs,
        "sqs": roi_sqs,
        "SQS": roi_sqs,
        "score": roi_sqs,
    }


def compute_interaction_plausibility(
    vector: SledgeVector,
    scenario_type: str,
    rois: List[Dict[str, float]],
) -> Dict[str, float]:
    targets, _ = _target_rows(vector, scenario_type=scenario_type, rois=rois)
    ego = _ego_row(vector)

    if len(targets) == 0:
        default = 0.45
        return {
            "interaction_relation_score": default,
            "interaction_kinematic_score": default,
            "interaction_geometry_score": default,
            "interaction_plausibility_score": default,
            "target_count": 0.0,
        }

    ego_x = _state_value(ego, int(getattr(AgentIndex, "X", 0)), 0.0)
    ego_y = _state_value(ego, int(getattr(AgentIndex, "Y", 1)), 0.0)

    relation_scores: List[float] = []
    kinematic_scores: List[float] = []
    geometry_scores: List[float] = []

    for t in targets:
        x = _state_value(t, int(getattr(AgentIndex, "X", 0)), 0.0)
        y = _state_value(t, int(getattr(AgentIndex, "Y", 1)), 0.0)
        speed = max(0.0, _state_value(t, int(getattr(AgentIndex, "VELOCITY", 3)), 0.0))
        heading = _state_value(t, int(getattr(AgentIndex, "HEADING", 2)), 0.0)
        dist_ego = math.hypot(x - ego_x, y - ego_y)

        if scenario_type in {"sudden_pedestrian_crossing", "pedestrian_crossing"}:
            relation = 0.55 * _triangular_score(abs(y), 0.5, 3.5, 10.0) + 0.45 * _triangular_score(x, -5.0, 8.0, 28.0)
            kinetic = 0.70 * float(np.clip(1.0 - max(0.0, speed - 4.5) / 3.0, 0.0, 1.0)) + 0.30 * 1.0
            geometry = 0.65 * _triangular_score(dist_ego, 0.8, 4.0, 12.0) + 0.35 * float(np.clip(1.0 - max(0.0, abs(y) - 10.0) / 6.0, 0.0, 1.0))
        elif scenario_type == "hard_brake":
            relation = 0.55 * _triangular_score(x, 1.5, 8.0, 25.0) + 0.45 * _triangular_score(abs(y), 0.0, 0.8, 2.8)
            kinetic = 0.50 * float(np.clip(1.0 - max(0.0, speed - 18.0) / 10.0, 0.0, 1.0)) + 0.50 * float(np.clip(1.0 - abs(heading) / 1.2, 0.0, 1.0))
            geometry = 0.70 * _triangular_score(dist_ego, 1.5, 7.0, 20.0) + 0.30 * float(np.clip(1.0 - max(0.0, abs(y) - 3.0) / 3.0, 0.0, 1.0))
        elif scenario_type == "cut_in":
            relation = 0.55 * _triangular_score(x, 0.0, 8.0, 22.0) + 0.45 * _triangular_score(abs(y), 0.3, 1.8, 4.5)
            kinetic = 0.55 * float(np.clip(1.0 - abs(heading) / 1.0, 0.0, 1.0)) + 0.45 * float(np.clip(1.0 - max(0.0, speed - 20.0) / 10.0, 0.0, 1.0))
            geometry = 0.65 * _triangular_score(dist_ego, 1.2, 5.5, 15.0) + 0.35 * _triangular_score(abs(y), 0.3, 1.8, 4.5)
        else:
            relation = float(np.clip(1.0 - max(0.0, dist_ego - 20.0) / 20.0, 0.0, 1.0))
            kinetic = float(np.clip(1.0 - max(0.0, speed - 18.0) / 10.0, 0.0, 1.0))
            geometry = float(np.clip(1.0 - max(0.0, abs(y) - 8.0) / 8.0, 0.0, 1.0))

        relation_scores.append(float(np.clip(relation, 0.0, 1.0)))
        kinematic_scores.append(float(np.clip(kinetic, 0.0, 1.0)))
        geometry_scores.append(float(np.clip(geometry, 0.0, 1.0)))

    relation_score = _safe_mean_from_list(relation_scores, default=0.45)
    kinematic_score = _safe_mean_from_list(kinematic_scores, default=0.45)
    geometry_score = _safe_mean_from_list(geometry_scores, default=0.45)
    interaction = float(np.clip(0.40 * relation_score + 0.30 * kinematic_score + 0.30 * geometry_score, 0.0, 1.0))
    return {
        "interaction_relation_score": float(relation_score),
        "interaction_kinematic_score": float(kinematic_score),
        "interaction_geometry_score": float(geometry_score),
        "interaction_plausibility_score": float(interaction),
        "target_count": float(len(targets)),
    }


def compute_sqs(
    vector: SledgeVector,
    scenario_type: str,
    rois: Optional[List[Dict[str, float]]] = None,
    context_margin: float = 6.0,
    compliance_score: float = 1.0,
) -> Dict[str, float]:
    rois = rois or []

    numeric = numeric_validity_score(vector)
    lane_conn = lane_connectivity_score(vector)
    topology_score = float(lane_conn)
    compliance_score = float(np.clip(compliance_score, 0.0, 1.0))
    basic_feasibility_score = float(
        np.clip(0.35 * numeric + 0.35 * compliance_score + 0.30 * topology_score, 0.0, 1.0)
    )

    ctx = compute_context_coherence(vector, rois=rois, context_margin=context_margin)
    interaction_scores = compute_interaction_plausibility(vector, scenario_type=scenario_type, rois=rois)

    context_overlap = float(ctx["context_overlap_score"])
    context_layout = float(ctx["context_layout_score"])
    context_lane = float(ctx["context_lane_score"])
    context_coherence_score = float(ctx["context_coherence_score"])

    interaction_relation_score = float(interaction_scores["interaction_relation_score"])
    interaction_kinematic_score = float(interaction_scores["interaction_kinematic_score"])
    interaction_geometry_score = float(interaction_scores["interaction_geometry_score"])
    interaction_plausibility_score = float(interaction_scores["interaction_plausibility_score"])
    target_count = int(interaction_scores["target_count"])

    sqs = float(
        np.clip(
            0.30 * basic_feasibility_score
            + 0.35 * context_coherence_score
            + 0.35 * interaction_plausibility_score,
            0.0,
            1.0,
        )
    )

    return {
        "numeric_validity_score": float(numeric),
        "overlap_free_score": float(context_overlap),
        "lane_consistency_score": float(context_lane),
        "lane_connectivity_score": float(topology_score),
        "layout_plausibility_score": float(context_layout),
        "kinematic_plausibility_score": float(interaction_kinematic_score),
        "compliance_score": float(compliance_score),
        "topology_score": float(topology_score),
        "basic_feasibility_score": float(basic_feasibility_score),
        "context_overlap_score": float(context_overlap),
        "context_layout_score": float(context_layout),
        "context_lane_score": float(context_lane),
        "context_coherence_score": float(context_coherence_score),
        "interaction_relation_score": float(interaction_relation_score),
        "interaction_kinematic_score": float(interaction_kinematic_score),
        "interaction_geometry_score": float(interaction_geometry_score),
        "interaction_plausibility_score": float(interaction_plausibility_score),
        "target_count": int(target_count),
        "sqs": float(sqs),
        "SQS": float(sqs),
        "score": float(sqs),
    }


def _evaluate_vector_scene(
    vector: SledgeVector,
    raw_scene: Optional[SledgeVectorRaw],
    prompt_spec: Any,
    alignment_evaluator: PromptAlignmentEvaluator,
    threshold: float,
    sqs_threshold: float,
    roi_sqs_threshold: float,
    rois: List[Dict[str, float]],
    original_vector: Optional[SledgeVector] = None,
    context_margin: float = 6.0,
    context_distance_cap: float = 10.0,
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

    scenario_type = str(getattr(prompt_spec, "scenario_type", "generic"))
    sqs_dict = compute_sqs(
        vector,
        scenario_type=scenario_type,
        rois=rois,
        context_margin=float(context_margin),
        compliance_score=1.0 if bool(compliance["compliant"]) else 0.0,
    )
    roi_sqs_dict = compute_context_roi_sqs(
        vector,
        rois=rois,
        context_margin=float(context_margin),
    )
    ctx_change = context_change_magnitude(
        original_vector=original_vector,
        target_vector=vector,
        rois=rois,
        context_margin=float(context_margin),
        distance_cap=float(context_distance_cap),
    )

    sqs_value = float(sqs_dict.get("sqs", sqs_dict.get("SQS", sqs_dict.get("score", 0.0))))
    roi_sqs_value = float(roi_sqs_dict.get("roi_sqs", roi_sqs_dict.get("sqs", roi_sqs_dict.get("SQS", roi_sqs_dict.get("score", 0.0)))))

    sqs_dict["sqs"] = sqs_value
    roi_sqs_dict["roi_sqs"] = roi_sqs_value

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
        **{k: v for k, v in roi_sqs_dict.items() if k not in {"sqs", "SQS", "score"}},
        "sqs_threshold": float(sqs_threshold),
        "quality_pass": bool(sqs_value >= float(sqs_threshold)),
        "roi_sqs": roi_sqs_value,
        "roi_sqs_threshold": float(roi_sqs_threshold),
        "roi_quality_pass": bool(roi_sqs_value >= float(roi_sqs_threshold)),
        "context_change_magnitude": ctx_change,
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
            "context_change_magnitude": _mean_optional([r.get("context_change_magnitude") for r in items]),
            "compliance_score": _mean([float(r["compliance_score"]) for r in items]),
            "topology_score": _mean([float(r["topology_score"]) for r in items]),
            "basic_feasibility_score": _mean([float(r["basic_feasibility_score"]) for r in items]),
            "context_overlap_score": _mean([float(r["context_overlap_score"]) for r in items]),
            "context_layout_score": _mean([float(r["context_layout_score"]) for r in items]),
            "context_lane_score": _mean([float(r["context_lane_score"]) for r in items]),
            "context_coherence_score": _mean([float(r["context_coherence_score"]) for r in items]),
            "interaction_relation_score": _mean([float(r["interaction_relation_score"]) for r in items]),
            "interaction_kinematic_score": _mean([float(r["interaction_kinematic_score"]) for r in items]),
            "interaction_geometry_score": _mean([float(r["interaction_geometry_score"]) for r in items]),
            "interaction_plausibility_score": _mean([float(r["interaction_plausibility_score"]) for r in items]),
            "target_count": _mean([float(r["target_count"]) for r in items]),
            "RSR": (_mean_optional([r.get("RSR") for r in items]) or 0.0),
            "RSR_strict": (_mean_optional([r.get("RSR_strict") for r in items]) or 0.0),
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
            "edited_context_change_magnitude": _mean_optional([r.get("edited_context_change_magnitude") for r in items]),
            "generated_context_change_magnitude": _mean_optional([r.get("generated_context_change_magnitude") for r in items]),
            "delta_context_change_magnitude": _mean_optional([r.get("delta_context_change_magnitude") for r in items]),
            "edited_basic_feasibility_score": _mean([float(r["edited_basic_feasibility_score"]) for r in items]),
            "generated_basic_feasibility_score": _mean([float(r["generated_basic_feasibility_score"]) for r in items]),
            "delta_basic_feasibility_score": _mean([float(r["generated_basic_feasibility_score"]) - float(r["edited_basic_feasibility_score"]) for r in items]),
            "edited_context_coherence_score": _mean([float(r["edited_context_coherence_score"]) for r in items]),
            "generated_context_coherence_score": _mean([float(r["generated_context_coherence_score"]) for r in items]),
            "delta_context_coherence_score": _mean([float(r["generated_context_coherence_score"]) - float(r["edited_context_coherence_score"]) for r in items]),
            "edited_interaction_plausibility_score": _mean([float(r["edited_interaction_plausibility_score"]) for r in items]),
            "generated_interaction_plausibility_score": _mean([float(r["generated_interaction_plausibility_score"]) for r in items]),
            "delta_interaction_plausibility_score": _mean([float(r["generated_interaction_plausibility_score"]) - float(r["edited_interaction_plausibility_score"]) for r in items]),
            "RSR": _mean([float(r["RSR"]) for r in items]),
            "RSR_strict": _mean([float(r["RSR_strict"]) for r in items]),
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
    if args.which == "original" and (not args.original_root or not args.edited_root):
        raise ValueError("--edited-root and --original-root are required for which=original")

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
    original_root = Path(args.original_root).resolve() if args.original_root else None

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
                original_scene_path = None
                original_vector = None

                if args.which == "original":
                    original_scene_path = _resolve_original_input_scene_path(
                        row=row,
                        edited_scene_dir=edited_scene_dir,
                        edited_root=edited_root,
                        original_root=original_root,
                    )
                    if original_scene_path is None or not original_scene_path.exists():
                        raise FileNotFoundError(
                            f"Original scene not found from manifest/edit_report/original_root mapping for output_dir={row.get('output_dir', '')}"
                        )
                    original_vector_eval, original_raw_eval, original_fmt = _load_scene_as_vector(original_scene_path, ae_config)
                    original_eval = _evaluate_vector_scene(
                        vector=original_vector_eval,
                        raw_scene=original_raw_eval,
                        prompt_spec=prompt_spec,
                        alignment_evaluator=alignment_evaluator,
                        threshold=float(args.alignment_threshold),
                        sqs_threshold=float(args.sqs_threshold),
                        roi_sqs_threshold=float(args.roi_sqs_threshold),
                        rois=rois,
                        original_vector=None,
                        context_margin=float(args.context_margin),
                        context_distance_cap=float(args.context_distance_cap),
                    )
                    record = {
                        **base_record,
                        "scene_path": str(original_scene_path),
                        "source_format": original_fmt,
                        **original_eval,
                    }
                elif args.which in {"edited", "compare"}:
                    edited_scene_path = _resolve_edited_scene_path(row)
                    edited_vector, edited_raw, edited_fmt = _load_scene_as_vector(edited_scene_path, ae_config)
                    original_scene_path = _resolve_original_scene_path(row, edited_scene_dir, None)
                    if original_scene_path is not None:
                        try:
                            original_vector, _, _ = _load_scene_as_vector(original_scene_path, ae_config)
                        except Exception:
                            original_vector = None
                    edited_eval = _evaluate_vector_scene(
                        vector=edited_vector,
                        raw_scene=edited_raw,
                        prompt_spec=prompt_spec,
                        alignment_evaluator=alignment_evaluator,
                        threshold=float(args.alignment_threshold),
                        sqs_threshold=float(args.sqs_threshold),
                        roi_sqs_threshold=float(args.roi_sqs_threshold),
                        rois=rois,
                        original_vector=original_vector,
                        context_margin=float(args.context_margin),
                        context_distance_cap=float(args.context_distance_cap),
                    )
                    if args.which in {"original", "edited"}:
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
                    if original_vector is None:
                        original_scene_path = _resolve_original_scene_path(row, edited_scene_dir, generated_scene_path)
                        if original_scene_path is not None:
                            try:
                                original_vector, _, _ = _load_scene_as_vector(original_scene_path, ae_config)
                            except Exception:
                                original_vector = None
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
                        original_vector=original_vector,
                        context_margin=float(args.context_margin),
                        context_distance_cap=float(args.context_distance_cap),
                    )
                    if args.which == "generated":
                        record = {
                            **base_record,
                            "scene_path": str(generated_scene_path),
                            "source_format": generated_fmt,
                            **generated_eval,
                            "RSR_strict": 1.0 if (
                                generated_eval["semantic_pass"]
                                and generated_eval["compliant"]
                                and generated_eval["roi_quality_pass"]
                            ) else 0.0,
                            "RSR": _compute_rsr_soft(
                                alignment_total=float(generated_eval["alignment_total"]),
                                semantic_pass=bool(generated_eval["semantic_pass"]),
                                compliant=bool(generated_eval["compliant"]),
                                sqs=float(generated_eval["sqs"]),
                                roi_sqs=float(generated_eval["roi_sqs"]),
                                alignment_threshold=float(args.alignment_threshold),
                                sqs_threshold=float(args.sqs_threshold),
                                roi_sqs_threshold=float(args.roi_sqs_threshold),
                            ),
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
                                "delta_context_change_magnitude": (
                                    None
                                    if edited_eval.get("context_change_magnitude") is None or generated_eval.get("context_change_magnitude") is None
                                    else float(generated_eval["context_change_magnitude"]) - float(edited_eval["context_change_magnitude"])
                                ),
                                "RSR_strict": 1.0 if (
                                    generated_eval["semantic_pass"]
                                    and generated_eval["compliant"]
                                    and generated_eval["roi_quality_pass"]
                                ) else 0.0,
                                "RSR": _compute_rsr_soft(
                                    alignment_total=float(generated_eval["alignment_total"]),
                                    semantic_pass=bool(generated_eval["semantic_pass"]),
                                    compliant=bool(generated_eval["compliant"]),
                                    sqs=float(generated_eval["sqs"]),
                                    roi_sqs=float(generated_eval["roi_sqs"]),
                                    alignment_threshold=float(args.alignment_threshold),
                                    sqs_threshold=float(args.sqs_threshold),
                                    roi_sqs_threshold=float(args.roi_sqs_threshold),
                                ),
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

    if args.which in {"original", "edited"}:
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
            "context_change_magnitude": _mean_optional([r.get("context_change_magnitude") for r in summary_rows]),
            "basic_feasibility_score": _mean([float(r["basic_feasibility_score"]) for r in summary_rows]),
            "context_coherence_score": _mean([float(r["context_coherence_score"]) for r in summary_rows]),
            "interaction_plausibility_score": _mean([float(r["interaction_plausibility_score"]) for r in summary_rows]),
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
            "context_change_magnitude": _mean_optional([r.get("context_change_magnitude") for r in summary_rows]),
            "RSR": _mean([float(r["RSR"]) for r in summary_rows]),
            "RSR_strict": _mean([float(r["RSR_strict"]) for r in summary_rows]),
            "basic_feasibility_score": _mean([float(r["basic_feasibility_score"]) for r in summary_rows]),
            "context_coherence_score": _mean([float(r["context_coherence_score"]) for r in summary_rows]),
            "interaction_plausibility_score": _mean([float(r["interaction_plausibility_score"]) for r in summary_rows]),
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
            "edited_context_change_magnitude": _mean_optional([r.get("edited_context_change_magnitude") for r in summary_rows]),
            "generated_context_change_magnitude": _mean_optional([r.get("generated_context_change_magnitude") for r in summary_rows]),
            "delta_context_change_magnitude": _mean_optional([r.get("delta_context_change_magnitude") for r in summary_rows]),
            "edited_basic_feasibility_score": _mean([float(r["edited_basic_feasibility_score"]) for r in summary_rows]),
            "generated_basic_feasibility_score": _mean([float(r["generated_basic_feasibility_score"]) for r in summary_rows]),
            "delta_basic_feasibility_score": _mean([float(r["generated_basic_feasibility_score"]) - float(r["edited_basic_feasibility_score"]) for r in summary_rows]),
            "edited_context_coherence_score": _mean([float(r["edited_context_coherence_score"]) for r in summary_rows]),
            "generated_context_coherence_score": _mean([float(r["generated_context_coherence_score"]) for r in summary_rows]),
            "delta_context_coherence_score": _mean([float(r["generated_context_coherence_score"]) - float(r["edited_context_coherence_score"]) for r in summary_rows]),
            "edited_interaction_plausibility_score": _mean([float(r["edited_interaction_plausibility_score"]) for r in summary_rows]),
            "generated_interaction_plausibility_score": _mean([float(r["generated_interaction_plausibility_score"]) for r in summary_rows]),
            "delta_interaction_plausibility_score": _mean([float(r["generated_interaction_plausibility_score"]) - float(r["edited_interaction_plausibility_score"]) for r in summary_rows]),
            "RSR": _mean([float(r["RSR"]) for r in summary_rows]),
            "RSR_strict": _mean([float(r["RSR_strict"]) for r in summary_rows]),
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
    if args.which in {"original", "edited"}:
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"CSR={summary['CSR']:.6f}, CSR_strict={summary['CSR_strict']:.6f}, "
            f"MPA={summary['MPA']:.6f}, CR={summary['CR']:.6f}, "
            f"SQS={summary['SQS']:.6f}, ROI_SQS={summary['ROI_SQS']:.6f}, "
            f"CCM={0.0 if summary['context_change_magnitude'] is None else summary['context_change_magnitude']:.6f}"
        )
    elif args.which == "generated":
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"CSR={summary['CSR']:.6f}, CSR_strict={summary['CSR_strict']:.6f}, "
            f"MPA={summary['MPA']:.6f}, CR={summary['CR']:.6f}, "
            f"SQS={summary['SQS']:.6f}, ROI_SQS={summary['ROI_SQS']:.6f}, "
            f"CCM={0.0 if summary['context_change_magnitude'] is None else summary['context_change_magnitude']:.6f}, "
            f"RSR={summary['RSR']:.6f}, RSR_strict={summary['RSR_strict']:.6f}"
        )
    else:
        print(
            f"[OK] N={summary['N']}, errors={summary['num_errors']}, "
            f"edited_MPA={summary['edited_MPA']:.6f}, generated_MPA={summary['generated_MPA']:.6f}, "
            f"delta_MPA={summary['delta_MPA']:.6f}, edited_ROI_SQS={summary['edited_ROI_SQS']:.6f}, "
            f"generated_ROI_SQS={summary['generated_ROI_SQS']:.6f}, delta_ROI_SQS={summary['delta_ROI_SQS']:.6f}, "
            f"delta_CCM={0.0 if summary['delta_context_change_magnitude'] is None else summary['delta_context_change_magnitude']:.6f}, "
            f"RSR={summary['RSR']:.6f}, RSR_strict={summary['RSR_strict']:.6f}"
        )


if __name__ == "__main__":
    main()
