#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File:
    sledge/script/evaluate/evaluate_main_table.py

Purpose:
    Main-table evaluator for:
    1) G0 / original SLEDGE scenario cache (unlabeled natural scenes)
    2) B1 / edited controlled scenes
    3) B2 / diffusion-generated controlled scenes

Important design correction:
    - G0 does NOT have ped/cut_in/brake labels.
    - Therefore G0 only evaluates unconditional scene-quality metrics.
    - Only B1/B2 evaluate control-related metrics such as CSR/MPA/Obj_R/IRS/RCS.

Outputs:
    - results_main_table.jsonl
    - main_table.csv                  # shared unconditional quality table
    - main_table.md
    - main_table_latex.txt
    - control_table.csv              # only for controlled methods (B1/B2)
    - control_table.md
    - control_table_latex.txt
    - supplement_quality_table.csv
    - interaction_table.csv
    - main_table_by_type.csv
    - main_table_by_severity.csv
    - reference_stats.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import statistics
from collections import defaultdict
from dataclasses import dataclass, asdict, is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-8
DEFAULT_DRL_MAX = 80.0
DEFAULT_ROI_RADIUS = 20.0
DEFAULT_CONTEXT_RING_RADIUS = 35.0
SUPPORTED_SCENARIO_TYPES = ["ped_cross", "hard_brake", "cut_in"]


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SceneBundle:
    scene_id: str
    source_type: str
    scenario_type: Optional[str]
    severity: Optional[str]
    prompt_info: Dict[str, Any]
    raw: Any
    vector: Any
    path: str
    roi_info: Optional[Dict[str, Any]]


@dataclass
class SceneEvalResult:
    scene_id: str
    source_type: str
    scenario_type: str
    severity: str
    path: str

    CSR: float
    MPA: float
    CR: float
    SQS: float
    QSR: float
    ROI_SQS: float
    ROI_QSR: float
    CCM: float
    RSR: float
    RSR_strict: float

    DRL: float
    ELA: float
    Obj_P: float
    Obj_R: float
    SDM: float
    IRS: float
    RCS: float

    IRS_relation: float
    IRS_geometry: float
    IRS_kinematics: float

    scene_stat_vector: List[float]
    roi_scene_stat_vector: Optional[List[float]]

    accepted: int


# =========================================================
# General helpers
# =========================================================

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def nan() -> float:
    return float("nan")


def safe_mean(values: Sequence[float], default: float = float("nan")) -> float:
    vals = []
    for v in values:
        try:
            vf = float(v)
            if not math.isnan(vf):
                vals.append(vf)
        except Exception:
            continue
    if not vals:
        return default
    return float(sum(vals) / len(vals))


def safe_std(values: Sequence[float], default: float = 0.0) -> float:
    vals = []
    for v in values:
        try:
            vf = float(v)
            if not math.isnan(vf):
                vals.append(vf)
        except Exception:
            continue
    if len(vals) <= 1:
        return default
    return float(statistics.pstdev(vals))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def euclidean_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def angle_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def heading_diff_abs(a: float, b: float) -> float:
    return abs(angle_wrap(a - b))


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pickle_or_json_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        blob = f.read()

    try:
        return pickle.loads(blob)
    except Exception:
        pass

    try:
        return json.loads(blob.decode("utf-8"))
    except Exception:
        pass

    raise ValueError(f"Unsupported gz payload format: {path}")


# =========================================================
# Schema adapter layer
# If your internal sledge_vector.gz schema differs, modify these 4 extractors.
# =========================================================

def _object_items(obj: Any):
    if obj is None:
        return []

    if isinstance(obj, dict):
        return list(obj.items())

    if is_dataclass(obj):
        return [(f.name, getattr(obj, f.name)) for f in fields(obj)]

    if hasattr(obj, "__dict__"):
        try:
            return list(vars(obj).items())
        except Exception:
            return []

    return []


def _get_value(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]

    if is_dataclass(obj):
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)

    if hasattr(obj, "__dict__"):
        for k in keys:
            if hasattr(obj, k):
                return getattr(obj, k)

    return default


def _normalize_sequence(x: Any) -> List[Any]:
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        return list(x)

    if isinstance(x, np.ndarray):
        return list(x)

    if isinstance(x, dict):
        return list(x.values())

    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        try:
            return list(x)
        except Exception:
            pass

    # 某些容器对象会把真实列表放在 items/data/elements 里
    for key in ["items", "data", "elements", "values"]:
        v = _get_value(x, [key], None)
        if v is not None and v is not x:
            try:
                return _normalize_sequence(v)
            except Exception:
                pass

    return []


def _find_first(obj: Any, keys: Sequence[str], default: Any = None, _visited=None) -> Any:
    if _visited is None:
        _visited = set()

    if obj is None:
        return default

    obj_id = id(obj)
    if obj_id in _visited:
        return default
    _visited.add(obj_id)

    # 先查本层
    direct = _get_value(obj, keys, default=None)
    if direct is not None:
        return direct

    # 序列递归
    if isinstance(obj, (list, tuple, np.ndarray)):
        for item in obj:
            out = _find_first(item, keys, default=None, _visited=_visited)
            if out is not None:
                return out
        return default

    # dict / dataclass / object 递归
    for _, v in _object_items(obj):
        out = _find_first(v, keys, default=None, _visited=_visited)
        if out is not None:
            return out

    return default

def debug_describe(obj, name="root", depth=0, max_depth=2, max_items=8):
    indent = "  " * depth
    tname = type(obj).__name__
    print(f"{indent}{name}: type={tname}")

    if obj is None:
        return

    if depth >= max_depth:
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())[:max_items]
        print(f"{indent}  keys={keys}")
        for k in keys:
            try:
                debug_describe(obj[k], name=f"{name}.{k}", depth=depth + 1, max_depth=max_depth, max_items=max_items)
            except Exception as e:
                print(f"{indent}  [ERR dict child {k}: {e}]")
        return

    if isinstance(obj, (list, tuple)):
        print(f"{indent}  len={len(obj)}")
        for i, item in enumerate(list(obj)[:max_items]):
            try:
                debug_describe(item, name=f"{name}[{i}]", depth=depth + 1, max_depth=max_depth, max_items=max_items)
            except Exception as e:
                print(f"{indent}  [ERR seq child {i}: {e}]")
        return

    if hasattr(obj, "__dict__"):
        try:
            keys = list(vars(obj).keys())[:max_items]
            print(f"{indent}  attrs={keys}")
            for k in keys:
                try:
                    debug_describe(getattr(obj, k), name=f"{name}.{k}", depth=depth + 1, max_depth=max_depth, max_items=max_items)
                except Exception as e:
                    print(f"{indent}  [ERR attr child {k}: {e}]")
        except Exception as e:
            print(f"{indent}  [ERR vars(): {e}]")
        return

    # numpy / torch / 其他对象
    for attr in ["shape", "dtype"]:
        if hasattr(obj, attr):
            try:
                print(f"{indent}  {attr}={getattr(obj, attr)}")
            except Exception:
                pass

def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.asarray([])
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _squeeze_leading_singletons(arr: np.ndarray) -> np.ndarray:
    out = arr
    while out.ndim > 0 and out.shape[0] == 1:
        out = out[0]
    return out


def _normalize_mask(mask: Any, target_len: Optional[int] = None) -> np.ndarray:
    m = _to_numpy(mask)
    if m.size == 0:
        if target_len is None:
            return np.asarray([], dtype=bool)
        return np.ones(target_len, dtype=bool)

    m = np.asarray(m).astype(bool).reshape(-1)
    if target_len is not None and len(m) != target_len:
        if len(m) > target_len:
            m = m[:target_len]
        else:
            pad = np.ones(target_len - len(m), dtype=bool)
            m = np.concatenate([m, pad], axis=0)
    return m


def _select_last_valid_row(states: Any, mask: Any = None) -> np.ndarray:
    s = _to_numpy(states)
    s = _squeeze_leading_singletons(s)

    if s.ndim == 0:
        return np.asarray([float(s)])
    if s.ndim == 1:
        return s.astype(float)

    # 2D: [T, F] or [N, F]
    if s.ndim == 2:
        m = _normalize_mask(mask, target_len=s.shape[0])
        valid_idx = np.where(m)[0]
        if len(valid_idx) == 0:
            return s[-1].astype(float)
        return s[valid_idx[-1]].astype(float)

    # >=3D: recursively collapse first dim
    return _select_last_valid_row(s[0], mask[0] if mask is not None and np.asarray(mask).ndim > 1 else mask)


def _iter_entity_rows(states: Any, mask: Any = None) -> List[np.ndarray]:
    s = _to_numpy(states)
    s = _squeeze_leading_singletons(s)

    if s.size == 0:
        return []

    # 1D: single entity single row
    if s.ndim == 1:
        return [s.astype(float)]

    # 2D:
    # Usually [N, F] for entities or [T, F] for one entity over time.
    # Here for object groups, we treat rows as entities.
    if s.ndim == 2:
        m = _normalize_mask(mask, target_len=s.shape[0])
        rows = []
        for i in range(s.shape[0]):
            if bool(m[i]):
                rows.append(s[i].astype(float))
        return rows

    # 3D:
    # Usually [N, T, F] for N entities over time/polyline points.
    if s.ndim == 3:
        rows = []
        m = _to_numpy(mask)
        for i in range(s.shape[0]):
            mi = None
            if m.size > 0:
                if m.ndim == 1:
                    mi = m
                elif m.ndim >= 2:
                    mi = m[i]
            rows.append(_select_last_valid_row(s[i], mi).astype(float))
        return rows

    # >3D: recursively collapse
    return _iter_entity_rows(s[0], mask[0] if mask is not None and np.asarray(mask).ndim > 1 else mask)


def _iter_polylines(states: Any, mask: Any = None) -> List[np.ndarray]:
    s = _to_numpy(states)
    s = _squeeze_leading_singletons(s)

    if s.size == 0:
        return []

    # [P, F] -> one polyline
    if s.ndim == 2:
        m = _normalize_mask(mask, target_len=s.shape[0])
        pts = s[m]
        if len(pts) == 0:
            pts = s
        return [pts.astype(float)]

    # [N, P, F] -> N polylines
    if s.ndim == 3:
        polys = []
        m = _to_numpy(mask)
        for i in range(s.shape[0]):
            mi = None
            if m.size > 0:
                if m.ndim == 1:
                    mi = m
                elif m.ndim >= 2:
                    mi = m[i]
            mi = _normalize_mask(mi, target_len=s.shape[1])
            pts = s[i][mi]
            if len(pts) == 0:
                pts = s[i]
            polys.append(pts.astype(float))
        return polys

    # >3D: collapse leading dim
    return _iter_polylines(s[0], mask[0] if mask is not None and np.asarray(mask).ndim > 1 else mask)


def _safe_heading_from_row(row: np.ndarray) -> float:
    row = np.asarray(row).reshape(-1)
    candidates = []

    # 优先尝试常见 heading 位置
    for idx in [2, 3, 4, 5]:
        if idx < len(row):
            v = float(row[idx])
            if np.isfinite(v) and abs(v) <= 4 * math.pi:
                candidates.append(v)

    if candidates:
        return float(candidates[0])
    return 0.0


def _safe_speed_from_row(row: np.ndarray) -> float:
    row = np.asarray(row).reshape(-1)

    # 常见情况1：speed 直接在某个标量位置
    for idx in [3, 4, 5]:
        if idx < len(row):
            v = float(row[idx])
            if np.isfinite(v) and 0.0 <= abs(v) <= 60.0:
                return abs(v)

    # 常见情况2：vx, vy
    if len(row) >= 5:
        vx = float(row[3])
        vy = float(row[4])
        if np.isfinite(vx) and np.isfinite(vy):
            spd = math.hypot(vx, vy)
            if 0.0 <= spd <= 60.0:
                return float(spd)

    return 0.0


def _safe_size_from_row(row: np.ndarray, default_length: float, default_width: float) -> Tuple[float, float]:
    row = np.asarray(row).reshape(-1)

    tail = []
    for v in row[-6:]:
        try:
            fv = abs(float(v))
            if np.isfinite(fv):
                tail.append(fv)
        except Exception:
            pass

    cands = [v for v in tail if 0.2 <= v <= 15.0]
    if len(cands) >= 2:
        cands = sorted(cands, reverse=True)
        length = cands[0]
        width = cands[1]
        if width > length:
            length, width = width, length
        return float(length), float(width)

    return float(default_length), float(default_width)

def extract_ego_state(vector: Any) -> Dict[str, float]:
    ego_group = None
    if isinstance(vector, dict):
        ego_group = vector.get("ego", None)

    if ego_group is None:
        return {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
            "speed": 0.0,
            "length": 4.8,
            "width": 2.0,
        }

    states = ego_group.get("states", None)
    mask = ego_group.get("mask", None)

    row = _select_last_valid_row(states, mask)
    row = np.asarray(row).reshape(-1)

    x = float(row[0]) if len(row) > 0 else 0.0
    y = float(row[1]) if len(row) > 1 else 0.0
    heading = _safe_heading_from_row(row)
    speed = _safe_speed_from_row(row)
    length, width = _safe_size_from_row(row, default_length=4.8, default_width=2.0)

    return {
        "x": x,
        "y": y,
        "heading": heading,
        "speed": speed,
        "length": length,
        "width": width,
    }


def extract_objects(vector: Any) -> List[Dict[str, Any]]:
    out = []

    if not isinstance(vector, dict):
        return out

    group_specs = [
        ("vehicles", "vehicle", 4.5, 1.8),
        ("pedestrians", "pedestrian", 0.8, 0.6),
        ("static_objects", "static", 1.0, 1.0),
    ]

    for group_name, obj_type, default_length, default_width in group_specs:
        group = vector.get(group_name, None)
        if group is None or not isinstance(group, dict):
            continue

        states = group.get("states", None)
        mask = group.get("mask", None)

        rows = _iter_entity_rows(states, mask)
        for idx, row in enumerate(rows):
            row = np.asarray(row).reshape(-1)
            if row.size == 0:
                continue

            x = float(row[0]) if len(row) > 0 else 0.0
            y = float(row[1]) if len(row) > 1 else 0.0
            heading = _safe_heading_from_row(row)
            speed = _safe_speed_from_row(row)
            length, width = _safe_size_from_row(row, default_length=default_length, default_width=default_width)

            out.append({
                "track_id": f"{group_name}_{idx}",
                "type": obj_type,
                "x": x,
                "y": y,
                "heading": heading,
                "speed": speed,
                "length": length,
                "width": width,
            })

    return out


def extract_lanes(vector: Any) -> List[Dict[str, Any]]:
    lanes = []

    if not isinstance(vector, dict):
        return lanes

    line_group = vector.get("lines", None)
    if line_group is None or not isinstance(line_group, dict):
        return lanes

    states = line_group.get("states", None)
    mask = line_group.get("mask", None)

    polylines = _iter_polylines(states, mask)
    for idx, poly in enumerate(polylines):
        poly = np.asarray(poly)
        if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 2:
            continue

        coords = []
        for p in poly:
            x = float(p[0])
            y = float(p[1])
            if np.isfinite(x) and np.isfinite(y):
                coords.append((x, y))

        if len(coords) < 2:
            continue

        lanes.append({
            "lane_id": f"line_{idx}",
            "polyline": coords,
            "successors": [],
        })

    return lanes


def extract_roi(vector: Any) -> Optional[Dict[str, Any]]:
    roi = _find_first(vector, ["roi", "edit_roi", "local_roi", "mask_roi"], default=None)
    if roi is None:
        return None

    if isinstance(roi, dict):
        cx = float(_find_first(roi, ["center_x", "x", "cx"], 0.0))
        cy = float(_find_first(roi, ["center_y", "y", "cy"], 0.0))
        radius = float(_find_first(roi, ["radius", "r"], DEFAULT_ROI_RADIUS))
        return {"center_x": cx, "center_y": cy, "radius": radius}

    return None


# =========================================================
# Path / prompt parsing
# =========================================================

def derive_scene_id(path: Path) -> str:
    if path.parent.name:
        return path.parent.name
    return path.stem


def infer_scenario_type(path_str: str, prompt_info: Optional[Dict[str, Any]] = None) -> str:
    if prompt_info is not None:
        for k in ["scenario_type", "type", "prompt_type"]:
            if k in prompt_info and pd.notna(prompt_info[k]):
                v = str(prompt_info[k]).lower()
                if "ped" in v:
                    return "ped_cross"
                if "brake" in v:
                    return "hard_brake"
                if "cut" in v:
                    return "cut_in"

    p = path_str.lower()
    if "ped" in p:
        return "ped_cross"
    if "brake" in p:
        return "hard_brake"
    if "cut_in" in p or "cutin" in p:
        return "cut_in"
    return "unknown"


def infer_severity(prompt_info: Optional[Dict[str, Any]], path_str: str) -> str:
    if prompt_info is not None:
        for k in ["severity", "severity_level", "risk_level", "level"]:
            if k in prompt_info and pd.notna(prompt_info[k]):
                return str(prompt_info[k])

    p = path_str.lower()
    if "aggressive" in p:
        return "aggressive"
    if "moderate" in p:
        return "moderate"
    if "mild" in p:
        return "mild"
    return "unknown"


# =========================================================
# Geometry helpers
# =========================================================

def polyline_length(polyline: List[Tuple[float, float]]) -> float:
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(polyline)):
        total += euclidean_xy(polyline[i - 1], polyline[i])
    return float(total)


def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, Tuple[float, float], float]:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = vx * wx + vy * wy
    c2 = vx * vx + vy * vy + EPS
    t = max(0.0, min(1.0, c1 / c2))
    proj = (ax + t * vx, ay + t * vy)
    d = euclidean_xy((px, py), proj)
    return d, proj, t


def nearest_lane_and_heading(
    ego_xy: Tuple[float, float],
    lanes: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    best_lane = None
    best_dist = float("inf")
    best_heading = 0.0

    ex, ey = ego_xy
    for lane in lanes:
        poly = lane.get("polyline", [])
        if len(poly) < 2:
            continue
        for i in range(1, len(poly)):
            ax, ay = poly[i - 1]
            bx, by = poly[i]
            d, _, _ = point_to_segment_distance(ex, ey, ax, ay, bx, by)
            if d < best_dist:
                best_dist = d
                best_lane = lane
                best_heading = math.atan2(by - ay, bx - ax)

    return best_lane, float(best_dist), float(best_heading)


def build_lane_map(lanes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {lane["lane_id"]: lane for lane in lanes}


def forward_reachable_length(
    start_lane_id: Optional[str],
    lane_map: Dict[str, Dict[str, Any]],
    max_depth: int = 20
) -> float:
    if start_lane_id is None or start_lane_id not in lane_map:
        return 0.0

    visited = set()

    def dfs(lane_id: str, depth: int) -> float:
        if depth > max_depth or lane_id in visited or lane_id not in lane_map:
            return 0.0
        visited.add(lane_id)
        lane = lane_map[lane_id]
        this_len = polyline_length(lane.get("polyline", []))
        succs = lane.get("successors", [])
        if not succs:
            visited.remove(lane_id)
            return this_len

        best_tail = 0.0
        for nxt in succs:
            best_tail = max(best_tail, dfs(nxt, depth + 1))

        visited.remove(lane_id)
        return this_len + best_tail

    return float(dfs(start_lane_id, 0))


def approximate_lane_curvature(polyline: List[Tuple[float, float]]) -> float:
    if len(polyline) < 3:
        return 0.0
    headings = []
    for i in range(1, len(polyline)):
        dx = polyline[i][0] - polyline[i - 1][0]
        dy = polyline[i][1] - polyline[i - 1][1]
        headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    diffs = [abs(angle_wrap(headings[i] - headings[i - 1])) for i in range(1, len(headings))]
    return float(sum(diffs) / max(1, len(diffs)))


# =========================================================
# Object plausibility
# =========================================================

def is_vehicle(obj_type: str) -> bool:
    return "vehicle" in obj_type or obj_type in {"car", "bus", "truck"}


def is_pedestrian(obj_type: str) -> bool:
    return "ped" in obj_type or "walker" in obj_type or obj_type == "person"


def split_objects_by_type(objects: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    vehicles, pedestrians, others = [], [], []
    for obj in objects:
        if is_vehicle(obj["type"]):
            vehicles.append(obj)
        elif is_pedestrian(obj["type"]):
            pedestrians.append(obj)
        else:
            others.append(obj)
    return vehicles, pedestrians, others


def plausible_object(obj: Dict[str, Any]) -> bool:
    t = obj["type"]
    speed = abs(float(obj["speed"]))
    length = float(obj["length"])
    width = float(obj["width"])

    if is_vehicle(t):
        return (2.5 <= length <= 8.5 and 1.2 <= width <= 3.0 and speed <= 35.0)
    if is_pedestrian(t):
        return (0.2 <= length <= 1.5 and 0.2 <= width <= 1.2 and speed <= 4.5)
    return (0.1 <= length <= 15.0 and 0.1 <= width <= 6.0 and speed <= 40.0)


def approximate_overlap_ratio(objects: List[Dict[str, Any]], ego: Dict[str, float]) -> float:
    boxes = []
    boxes.append(("ego", ego["x"], ego["y"], ego["length"], ego["width"]))
    for obj in objects:
        boxes.append((obj["track_id"], obj["x"], obj["y"], obj["length"], obj["width"]))

    def overlap(a, b) -> bool:
        _, ax, ay, al, aw = a
        _, bx, by, bl, bw = b
        return (abs(ax - bx) < (al + bl) / 2.0) and (abs(ay - by) < (aw + bw) / 2.0)

    total_pairs = 0
    bad_pairs = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            total_pairs += 1
            if overlap(boxes[i], boxes[j]):
                bad_pairs += 1

    if total_pairs == 0:
        return 0.0
    return float(bad_pairs / total_pairs)


def compute_obj_precision(objects: List[Dict[str, Any]], ego: Dict[str, float]) -> float:
    if not objects:
        return 1.0
    plausible = sum(1 for obj in objects if plausible_object(obj))
    overlap_ratio = approximate_overlap_ratio(objects, ego)
    base = plausible / max(1, len(objects))
    penalty = 1.0 - overlap_ratio
    return clip01(0.8 * base + 0.2 * penalty)


# =========================================================
# Controlled-scene interaction analysis
# =========================================================

def longitudinal_lateral_in_ego_frame(
    ego: Dict[str, float],
    obj: Dict[str, Any]
) -> Tuple[float, float]:
    dx = obj["x"] - ego["x"]
    dy = obj["y"] - ego["y"]
    ch = math.cos(ego["heading"])
    sh = math.sin(ego["heading"])
    longitudinal = dx * ch + dy * sh
    lateral = -dx * sh + dy * ch
    return float(longitudinal), float(lateral)


def find_lead_vehicle_same_lane(
    ego: Dict[str, float],
    vehicles: List[Dict[str, Any]],
    lateral_thresh: float = 2.2
) -> Optional[Dict[str, Any]]:
    best = None
    best_lon = float("inf")
    for obj in vehicles:
        lon, lat = longitudinal_lateral_in_ego_frame(ego, obj)
        if lon > 0 and abs(lat) <= lateral_thresh:
            if lon < best_lon:
                best_lon = lon
                best = obj
    return best


def find_cut_in_vehicle(
    ego: Dict[str, float],
    vehicles: List[Dict[str, Any]],
    lateral_band: Tuple[float, float] = (0.5, 4.5)
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    for obj in vehicles:
        lon, lat = longitudinal_lateral_in_ego_frame(ego, obj)
        if -10.0 <= lon <= 30.0 and lateral_band[0] <= abs(lat) <= lateral_band[1]:
            heading_gap = heading_diff_abs(ego["heading"], obj["heading"])
            intrusion = max(0.0, 1.0 - abs(abs(lat) - 1.5) / 3.0)
            score = intrusion + max(0.0, 1.0 - heading_gap / 0.8) + max(0.0, 1.0 - abs(lon - 8.0) / 20.0)
            if score > best_score:
                best_score = score
                best = obj
    return best


def find_crossing_pedestrian(
    ego: Dict[str, float],
    pedestrians: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1.0
    for obj in pedestrians:
        lon, lat = longitudinal_lateral_in_ego_frame(ego, obj)
        if -5.0 <= lon <= 25.0 and abs(lat) <= 8.0:
            rel_heading = heading_diff_abs(obj["heading"], ego["heading"])
            crossingness = max(0.0, 1.0 - abs(rel_heading - math.pi / 2) / (math.pi / 2))
            dist = math.hypot(lon, lat)
            score = crossingness + max(0.0, 1.0 - dist / 25.0)
            if score > best_score:
                best_score = score
                best = obj
    return best


def compute_target_object(
    scene_type: str,
    ego: Dict[str, float],
    objects: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    vehicles, pedestrians, _ = split_objects_by_type(objects)
    if scene_type == "hard_brake":
        return find_lead_vehicle_same_lane(ego, vehicles)
    if scene_type == "cut_in":
        return find_cut_in_vehicle(ego, vehicles)
    if scene_type == "ped_cross":
        return find_crossing_pedestrian(ego, pedestrians)
    return None


def compute_ttc_proxy(ego: Dict[str, float], obj: Dict[str, Any]) -> float:
    lon, _ = longitudinal_lateral_in_ego_frame(ego, obj)
    rel_speed = max(EPS, ego["speed"] - obj["speed"])
    if lon <= 0:
        return 999.0
    return float(lon / rel_speed)


def compute_obj_recall(scene_type: str, ego: Dict[str, float], objects: List[Dict[str, Any]]) -> float:
    target = compute_target_object(scene_type, ego, objects)
    if target is None:
        return 0.0

    lon, lat = longitudinal_lateral_in_ego_frame(ego, target)
    heading_gap = heading_diff_abs(ego["heading"], target["heading"])
    dist = math.hypot(lon, lat)

    if scene_type == "hard_brake":
        score = (
            0.45 * (1.0 - min(1.0, abs(lat) / 2.5)) +
            0.35 * (1.0 - min(1.0, abs(lon - 10.0) / 30.0)) +
            0.20 * (1.0 - min(1.0, heading_gap / 0.6))
        )
        return clip01(score)

    if scene_type == "cut_in":
        intrusion = 1.0 - min(1.0, abs(abs(lat) - 1.5) / 3.0)
        score = (
            0.40 * intrusion +
            0.35 * (1.0 - min(1.0, abs(lon - 8.0) / 20.0)) +
            0.25 * (1.0 - min(1.0, heading_gap / 0.8))
        )
        return clip01(score)

    if scene_type == "ped_cross":
        rel_heading = heading_diff_abs(target["heading"], ego["heading"])
        crossingness = 1.0 - min(1.0, abs(rel_heading - math.pi / 2) / (math.pi / 2))
        score = (
            0.50 * crossingness +
            0.30 * (1.0 - min(1.0, dist / 20.0)) +
            0.20 * (1.0 if -5.0 <= lon <= 25.0 else 0.0)
        )
        return clip01(score)

    return 0.0


def nearest_object_distance(ego: Dict[str, float], objects: List[Dict[str, Any]]) -> float:
    if not objects:
        return 50.0
    ds = [euclidean_xy((ego["x"], ego["y"]), (obj["x"], obj["y"])) for obj in objects]
    return float(min(ds))


def min_ttc_feature(ego: Dict[str, float], objects: List[Dict[str, Any]]) -> float:
    if not objects:
        return 999.0
    ttcs = [compute_ttc_proxy(ego, obj) for obj in objects]
    return float(min(ttcs))


def compute_interaction_scores(scene_type: str, ego: Dict[str, float], objects: List[Dict[str, Any]]) -> Dict[str, float]:
    target = compute_target_object(scene_type, ego, objects)

    if target is None:
        return {
            "IRS": 0.0,
            "IRS_relation": 0.0,
            "IRS_geometry": 0.0,
            "IRS_kinematics": 0.0,
            "RCS": 0.0,
        }

    lon, lat = longitudinal_lateral_in_ego_frame(ego, target)
    gap = math.hypot(lon, lat)
    ttc = compute_ttc_proxy(ego, target)
    heading_gap = heading_diff_abs(ego["heading"], target["heading"])

    if scene_type == "hard_brake":
        relation = clip01(1.0 if (lon > 0 and abs(lat) <= 2.2) else 0.0)
        geometry = clip01(1.0 - min(1.0, abs(lat) / 3.0)) * clip01(1.0 - abs(lon - 8.0) / 25.0)
        kinematics = clip01(1.0 - min(1.0, ttc / 6.0))
    elif scene_type == "cut_in":
        relation = clip01(1.0 if (lon > -5.0 and abs(lat) <= 4.5) else 0.0)
        intrusion = clip01(1.0 - abs(abs(lat) - 1.5) / 3.0)
        geometry = clip01(0.6 * intrusion + 0.4 * (1.0 - min(1.0, abs(lon - 8.0) / 20.0)))
        kinematics = clip01(0.5 * (1.0 - min(1.0, heading_gap / 0.8)) + 0.5 * (1.0 - min(1.0, ttc / 6.0)))
    elif scene_type == "ped_cross":
        relation = clip01(1.0 if (-5.0 <= lon <= 25.0 and abs(lat) <= 8.0) else 0.0)
        rel_heading = heading_diff_abs(target["heading"], ego["heading"])
        crossingness = clip01(1.0 - abs(rel_heading - math.pi / 2) / (math.pi / 2))
        geometry = clip01(0.5 * crossingness + 0.5 * (1.0 - min(1.0, gap / 20.0)))
        kinematics = clip01(1.0 - min(1.0, ttc / 8.0))
    else:
        relation = 0.0
        geometry = 0.0
        kinematics = 0.0

    irs = clip01(0.35 * relation + 0.35 * geometry + 0.30 * kinematics)

    target_risk = clip01(1.0 - min(1.0, ttc / 8.0))
    obj_p = compute_obj_precision(objects, ego)
    bg_overlap = approximate_overlap_ratio(objects, ego)
    background_anomaly = clip01(0.5 * (1.0 - obj_p) + 0.5 * bg_overlap)
    rcs = float(target_risk / max(EPS, target_risk + background_anomaly))

    return {
        "IRS": irs,
        "IRS_relation": relation,
        "IRS_geometry": geometry,
        "IRS_kinematics": kinematics,
        "RCS": clip01(rcs),
    }


# =========================================================
# Common scene-quality metrics
# =========================================================

def compute_drl(ego: Dict[str, float], lanes: List[Dict[str, Any]], drl_max: float = DEFAULT_DRL_MAX) -> float:
    lane, _, _ = nearest_lane_and_heading((ego["x"], ego["y"]), lanes)
    lane_map = build_lane_map(lanes)
    start_lane_id = lane["lane_id"] if lane is not None else None
    reachable = forward_reachable_length(start_lane_id, lane_map)
    if reachable <= 0.0 and lane is not None:
        reachable = polyline_length(lane.get("polyline", []))
    return float(min(drl_max, reachable))


def compute_ela(ego: Dict[str, float], lanes: List[Dict[str, Any]], theta_max: float = math.pi / 3) -> float:
    _, lane_dist, lane_heading = nearest_lane_and_heading((ego["x"], ego["y"]), lanes)
    if not lanes:
        return 0.0
    heading_gap = heading_diff_abs(ego["heading"], lane_heading)
    align = 1.0 - min(1.0, heading_gap / max(theta_max, EPS))
    dist_penalty = 1.0 - min(1.0, lane_dist / 4.0)
    return clip01(0.8 * align + 0.2 * dist_penalty)


def extract_scene_density_features(objects: List[Dict[str, Any]], lanes: List[Dict[str, Any]]) -> Dict[str, float]:
    vehicles, pedestrians, others = split_objects_by_type(objects)
    return {
        "lane_count": float(len(lanes)),
        "veh_count": float(len(vehicles)),
        "ped_count": float(len(pedestrians)),
        "static_count": float(len(others)),
    }


def compute_sdm(density_feats: Dict[str, float], ref_stats: Dict[str, Any]) -> float:
    keys = ["lane_count", "veh_count", "ped_count", "static_count"]
    terms = []
    for k in keys:
        mu = float(ref_stats.get("density_mean", {}).get(k, 0.0))
        sigma = float(ref_stats.get("density_std", {}).get(k, 1.0))
        v = float(density_feats.get(k, 0.0))
        terms.append(abs(v - mu) / max(EPS, sigma))
    score = 1.0 - sum(terms) / max(1, len(terms))
    return clip01(score)


def compute_ccm(roi_info: Optional[Dict[str, Any]], objects: List[Dict[str, Any]]) -> float:
    if roi_info is None:
        return nan()

    cx = float(roi_info["center_x"])
    cy = float(roi_info["center_y"])
    rr = float(roi_info.get("radius", DEFAULT_ROI_RADIUS))
    ring = DEFAULT_CONTEXT_RING_RADIUS

    in_roi = 0
    in_ring = 0
    for obj in objects:
        d = euclidean_xy((obj["x"], obj["y"]), (cx, cy))
        if d <= rr:
            in_roi += 1
        elif d <= ring:
            in_ring += 1

    if (in_roi + in_ring) == 0:
        return 0.0
    return float(in_ring / max(1, in_roi + in_ring))


def compute_roi_sqs(
    roi_info: Optional[Dict[str, Any]],
    ego: Dict[str, float],
    objects: List[Dict[str, Any]],
    irs: float,
    obj_p: float
) -> float:
    if roi_info is None:
        return nan()

    cx = float(roi_info["center_x"])
    cy = float(roi_info["center_y"])
    rr = float(roi_info.get("radius", DEFAULT_ROI_RADIUS))
    roi_objs = [obj for obj in objects if euclidean_xy((obj["x"], obj["y"]), (cx, cy)) <= rr]
    if not roi_objs:
        return clip01(0.5 * obj_p + 0.5 * irs)

    roi_obj_p = compute_obj_precision(roi_objs, ego)
    return clip01(0.55 * roi_obj_p + 0.45 * irs)


# =========================================================
# Controlled-only semantic metrics
# =========================================================

def compute_cr_controlled(ela: float, obj_p: float, irs: float, scenario_type: str) -> float:
    if scenario_type == "unknown":
        return clip01(0.6 * ela + 0.4 * obj_p)
    return clip01(0.35 * ela + 0.35 * obj_p + 0.30 * irs)


def compute_sqs_controlled(ela: float, obj_p: float, sdm: float, irs: float) -> float:
    return clip01(0.25 * ela + 0.30 * obj_p + 0.20 * sdm + 0.25 * irs)


def compute_mpa(scene_type: str, obj_r: float, irs: float) -> float:
    if scene_type == "unknown":
        return 0.0
    return clip01(0.55 * obj_r + 0.45 * irs)


def compute_csr(mpa: float, cr: float) -> float:
    return clip01(0.6 * mpa + 0.4 * cr)


def compute_rsr(csr: float, cr: float, roi_sqs: float) -> float:
    roi_term = 0.0 if math.isnan(roi_sqs) else roi_sqs
    return clip01(0.40 * csr + 0.35 * cr + 0.25 * roi_term)


def compute_rsr_strict(csr: float, cr: float, roi_sqs: float, obj_r: float) -> float:
    roi_term = 0.0 if math.isnan(roi_sqs) else roi_sqs
    hard = 1.0 if (csr >= 0.55 and cr >= 0.55 and obj_r >= 0.50 and (math.isnan(roi_sqs) or roi_term >= 0.60)) else 0.0
    return float(hard)


# =========================================================
# Scene vectors for FSLD / ROI-FSLD
# =========================================================

def headway_feature(ego: Dict[str, float], vehicles: List[Dict[str, Any]]) -> float:
    lead = find_lead_vehicle_same_lane(ego, vehicles)
    if lead is None:
        return 50.0
    lon, _ = longitudinal_lateral_in_ego_frame(ego, lead)
    return float(max(0.0, lon))


def lateral_occupancy_hist(ego: Dict[str, float], objects: List[Dict[str, Any]], bins: Sequence[float]) -> List[float]:
    arr = []
    for obj in objects:
        _, lat = longitudinal_lateral_in_ego_frame(ego, obj)
        arr.append(lat)
    hist, _ = np.histogram(arr, bins=bins)
    total = float(np.sum(hist))
    if total <= 0:
        return [0.0 for _ in range(len(bins) - 1)]
    return [float(h / total) for h in hist]


def extract_scene_stat_vector(
    ego: Dict[str, float],
    objects: List[Dict[str, Any]],
    lanes: List[Dict[str, Any]],
    drl: float
) -> List[float]:
    vehicles, pedestrians, _ = split_objects_by_type(objects)
    lane_count = float(len(lanes))
    mean_curv = safe_mean([approximate_lane_curvature(l["polyline"]) for l in lanes], 0.0)
    veh_count = float(len(vehicles))
    ped_count = float(len(pedestrians))
    mean_headway = headway_feature(ego, vehicles)
    min_ttc = min_ttc_feature(ego, objects)
    nearest_dist = nearest_object_distance(ego, objects)
    lat_hist = lateral_occupancy_hist(ego, objects, bins=[-10, -4, -1.5, 1.5, 4, 10])

    return [
        float(drl),
        lane_count,
        float(mean_curv),
        veh_count,
        ped_count,
        float(mean_headway),
        float(min_ttc),
        float(nearest_dist),
        *lat_hist,
    ]


def extract_roi_scene_stat_vector(
    ego: Dict[str, float],
    objects: List[Dict[str, Any]],
    lanes: List[Dict[str, Any]],
    roi_info: Optional[Dict[str, Any]],
    drl: float,
) -> Optional[List[float]]:
    if roi_info is None:
        return None

    cx = float(roi_info["center_x"])
    cy = float(roi_info["center_y"])
    ring = float(roi_info.get("radius", DEFAULT_ROI_RADIUS)) + 10.0

    roi_objs = [obj for obj in objects if euclidean_xy((obj["x"], obj["y"]), (cx, cy)) <= ring]

    roi_lanes = []
    for lane in lanes:
        poly = lane.get("polyline", [])
        if any(euclidean_xy((x, y), (cx, cy)) <= ring for x, y in poly):
            roi_lanes.append(lane)

    local_drl = min(drl, ring * 2.0)
    return extract_scene_stat_vector(ego, roi_objs, roi_lanes, local_drl)


# =========================================================
# Fréchet distance
# =========================================================

def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance_from_stats(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    diff = mu1 - mu2
    covmean = sqrtm_psd(sigma1 @ sigma2)
    val = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(val))


def compute_distribution_stats(vectors: List[List[float]]) -> Dict[str, Any]:
    if not vectors:
        return {"mean": [], "cov": []}
    X = np.asarray(vectors, dtype=np.float64)
    mu = np.mean(X, axis=0)
    if X.shape[0] == 1:
        cov = np.eye(X.shape[1], dtype=np.float64) * 1e-6
    else:
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
    return {"mean": mu.tolist(), "cov": cov.tolist()}


# =========================================================
# Scene loading
# =========================================================

def collect_gz_files(root: Path, suffix_name: str) -> List[Path]:
    return sorted(root.rglob(suffix_name))


def load_scene_bundle_from_path(
    vector_path: Path,
    source_type: str,
    scenario_type: Optional[str] = None,
    severity: Optional[str] = None,
    prompt_info: Optional[Dict[str, Any]] = None,
) -> SceneBundle:
    vec = load_pickle_or_json_gz(vector_path)
    roi_info = extract_roi(vec)
    path_str = str(vector_path)

    stype = scenario_type
    sev = severity
    if stype is None:
        stype = infer_scenario_type(path_str, prompt_info)
    if sev is None:
        sev = infer_severity(prompt_info, path_str)

    return SceneBundle(
        scene_id=derive_scene_id(vector_path),
        source_type=source_type,
        scenario_type=stype,
        severity=sev,
        prompt_info=prompt_info or {},
        raw=None,
        vector=vec,
        path=path_str,
        roi_info=roi_info,
    )


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(manifest_path)
    return df.to_dict(orient="records")


def resolve_manifest_scene_path(
    row: Dict[str, Any],
    which: str,
    generated_root: Optional[Path],
    edited_root: Optional[Path],
) -> Path:
    """
    Resolve actual scenario file path from one manifest row.

    Your manifest columns include:
        - b2_scenario_cache_vector_path   (best direct path for generated)
        - b2_output_dir
        - output_dir
        - scene_path
        - scenario_type
        - severity_level
    """
    candidates = []

    def add_candidate(v: Any):
        if v is None:
            return
        try:
            s = str(v).strip()
        except Exception:
            return
        if s == "" or s.lower() == "nan":
            return
        candidates.append(s)

    if which == "generated":
        # ===== 1) direct full-path columns =====
        for k in [
            "b2_scenario_cache_vector_path",   # your manifest's actual best column
            "generated_path",
            "generated_relpath",
            "output_path",
            "vector_path",
        ]:
            if k in row and pd.notna(row[k]):
                add_candidate(row[k])

        # ===== 2) directory columns: append sledge_vector.gz =====
        for k in [
            "b2_output_dir",
            "output_dir",
        ]:
            if k in row and pd.notna(row[k]):
                add_candidate(Path(str(row[k])) / "sledge_vector.gz")

        # ===== 3) fallback from generated root + basename =====
        if generated_root is not None:
            scenario_type = str(row.get("scenario_type", "")).strip()
            base_name = None

            # Prefer deriving basename from direct path if available
            if "b2_scenario_cache_vector_path" in row and pd.notna(row["b2_scenario_cache_vector_path"]):
                p = Path(str(row["b2_scenario_cache_vector_path"]))
                if p.name == "sledge_vector.gz":
                    base_name = p.parent.name

            # Fallback: derive basename from output_dir
            if base_name is None and "output_dir" in row and pd.notna(row["output_dir"]):
                base_name = Path(str(row["output_dir"])).name

            if base_name is None and "b2_output_dir" in row and pd.notna(row["b2_output_dir"]):
                base_name = Path(str(row["b2_output_dir"])).name

            if scenario_type and base_name:
                add_candidate(generated_root / "log" / scenario_type / base_name / "sledge_vector.gz")

    elif which == "edited":
        # ===== 1) direct path columns =====
        for k in [
            "edited_path",
            "edited_relpath",
            "raw_path",
            "vector_path",
            "scene_path",      # your manifest contains original raw path
        ]:
            if k in row and pd.notna(row[k]):
                add_candidate(row[k])

        # ===== 2) directory columns =====
        for k in [
            "output_dir",      # your B2 proxy edited result dir
            "b2_output_dir",
        ]:
            if k in row and pd.notna(row[k]):
                d = Path(str(row[k]))
                add_candidate(d / "sledge_vector.gz")
                add_candidate(d / "sledge_raw.gz")

        # ===== 3) fallback from edited root + basename =====
        if edited_root is not None:
            scenario_type = str(row.get("scenario_type", "")).strip()
            base_name = None

            if "output_dir" in row and pd.notna(row["output_dir"]):
                base_name = Path(str(row["output_dir"])).name

            if base_name and scenario_type:
                add_candidate(edited_root / "log" / scenario_type / base_name / "sledge_vector.gz")
                add_candidate(edited_root / "log" / scenario_type / base_name / "sledge_raw.gz")

    # 去重，但保留顺序
    uniq_candidates = []
    seen = set()
    for c in candidates:
        cs = str(c)
        if cs not in seen:
            uniq_candidates.append(cs)
            seen.add(cs)

    for c in uniq_candidates:
        p = Path(c)
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cannot resolve scene path for row with scene_id={row.get('scene_id', 'N/A')} and which={which}. "
        f"Tried candidates: {uniq_candidates[:12]}"
    )


# =========================================================
# Reference stats
# =========================================================

def compute_reference_stats_from_paths(paths: List[Path], output_json: Optional[Path] = None) -> Dict[str, Any]:
    density_rows = []
    scene_vectors = []
    roi_scene_vectors = []

    for p in paths:
        try:
            bundle = load_scene_bundle_from_path(p, source_type="reference", scenario_type=None, severity=None, prompt_info={})
            ego = extract_ego_state(bundle.vector)
            objects = extract_objects(bundle.vector)
            lanes = extract_lanes(bundle.vector)

            drl = compute_drl(ego, lanes)
            density = extract_scene_density_features(objects, lanes)
            scene_vec = extract_scene_stat_vector(ego, objects, lanes, drl)
            roi_vec = extract_roi_scene_stat_vector(ego, objects, lanes, bundle.roi_info, drl)

            density_rows.append(density)
            scene_vectors.append(scene_vec)
            if roi_vec is not None:
                roi_scene_vectors.append(roi_vec)
        except Exception:
            continue

    density_mean = {}
    density_std = {}
    for k in ["lane_count", "veh_count", "ped_count", "static_count"]:
        vals = [float(r[k]) for r in density_rows] if density_rows else [0.0]
        density_mean[k] = safe_mean(vals, 0.0)
        density_std[k] = max(EPS, safe_std(vals, 1.0))

    out = {
        "density_mean": density_mean,
        "density_std": density_std,
        "scene_distribution": compute_distribution_stats(scene_vectors),
        "roi_scene_distribution": compute_distribution_stats(roi_scene_vectors),
        "num_reference_scenes": len(scene_vectors),
    }

    if output_json is not None:
        write_json(output_json, out)
    return out


# =========================================================
# Per-scene evaluation
# =========================================================

def evaluate_one_scene_g0(bundle: SceneBundle, ref_stats: Dict[str, Any]) -> SceneEvalResult:
    ego = extract_ego_state(bundle.vector)
    objects = extract_objects(bundle.vector)
    lanes = extract_lanes(bundle.vector)

    drl = compute_drl(ego, lanes)
    ela = compute_ela(ego, lanes)
    obj_p = compute_obj_precision(objects, ego)

    density_feats = extract_scene_density_features(objects, lanes)
    sdm = compute_sdm(density_feats, ref_stats)

    cr = clip01(0.5 * ela + 0.5 * obj_p)
    sqs = clip01(0.30 * ela + 0.35 * obj_p + 0.35 * sdm)

    scene_vec = extract_scene_stat_vector(ego, objects, lanes, drl)

    return SceneEvalResult(
        scene_id=bundle.scene_id,
        source_type=bundle.source_type,
        scenario_type="unlabeled",
        severity="none",
        path=bundle.path,

        CSR=nan(),
        MPA=nan(),
        CR=float(cr),
        SQS=float(sqs),
        QSR=float(1.0 if sqs >= 0.65 else 0.0),
        ROI_SQS=nan(),
        ROI_QSR=nan(),
        CCM=nan(),
        RSR=nan(),
        RSR_strict=nan(),

        DRL=float(drl),
        ELA=float(ela),
        Obj_P=float(obj_p),
        Obj_R=nan(),
        SDM=float(sdm),
        IRS=nan(),
        RCS=nan(),

        IRS_relation=nan(),
        IRS_geometry=nan(),
        IRS_kinematics=nan(),

        scene_stat_vector=[float(x) for x in scene_vec],
        roi_scene_stat_vector=None,

        accepted=1,
    )


def evaluate_one_scene_controlled(bundle: SceneBundle, ref_stats: Dict[str, Any]) -> SceneEvalResult:
    ego = extract_ego_state(bundle.vector)
    objects = extract_objects(bundle.vector)
    lanes = extract_lanes(bundle.vector)

    scene_type = bundle.scenario_type or "unknown"
    severity = bundle.severity or "unknown"

    drl = compute_drl(ego, lanes)
    ela = compute_ela(ego, lanes)
    obj_p = compute_obj_precision(objects, ego)
    obj_r = compute_obj_recall(scene_type, ego, objects)

    density_feats = extract_scene_density_features(objects, lanes)
    sdm = compute_sdm(density_feats, ref_stats)

    inter = compute_interaction_scores(scene_type, ego, objects)
    irs = inter["IRS"]
    irs_relation = inter["IRS_relation"]
    irs_geometry = inter["IRS_geometry"]
    irs_kinematics = inter["IRS_kinematics"]
    rcs = inter["RCS"]

    cr = compute_cr_controlled(ela, obj_p, irs, scene_type)
    sqs = compute_sqs_controlled(ela, obj_p, sdm, irs)
    roi_sqs = compute_roi_sqs(bundle.roi_info, ego, objects, irs, obj_p)
    ccm = compute_ccm(bundle.roi_info, objects)
    mpa = compute_mpa(scene_type, obj_r, irs)
    csr = compute_csr(mpa, cr)

    qsr = 1.0 if sqs >= 0.65 else 0.0
    roi_qsr = nan() if math.isnan(roi_sqs) else (1.0 if roi_sqs >= 0.65 else 0.0)
    rsr = compute_rsr(csr, cr, roi_sqs)
    rsr_strict = compute_rsr_strict(csr, cr, roi_sqs, obj_r)

    scene_vec = extract_scene_stat_vector(ego, objects, lanes, drl)
    roi_scene_vec = extract_roi_scene_stat_vector(ego, objects, lanes, bundle.roi_info, drl)

    accepted = 1 if (csr >= 0.50 and cr >= 0.50) else 0

    return SceneEvalResult(
        scene_id=bundle.scene_id,
        source_type=bundle.source_type,
        scenario_type=scene_type,
        severity=severity,
        path=bundle.path,

        CSR=float(csr),
        MPA=float(mpa),
        CR=float(cr),
        SQS=float(sqs),
        QSR=float(qsr),
        ROI_SQS=float(roi_sqs) if not math.isnan(roi_sqs) else nan(),
        ROI_QSR=float(roi_qsr) if not math.isnan(roi_qsr) else nan(),
        CCM=float(ccm) if not math.isnan(ccm) else nan(),
        RSR=float(rsr),
        RSR_strict=float(rsr_strict),

        DRL=float(drl),
        ELA=float(ela),
        Obj_P=float(obj_p),
        Obj_R=float(obj_r),
        SDM=float(sdm),
        IRS=float(irs),
        RCS=float(rcs),

        IRS_relation=float(irs_relation),
        IRS_geometry=float(irs_geometry),
        IRS_kinematics=float(irs_kinematics),

        scene_stat_vector=[float(x) for x in scene_vec],
        roi_scene_stat_vector=[float(x) for x in roi_scene_vec] if roi_scene_vec is not None else None,

        accepted=int(accepted),
    )


# =========================================================
# Group aggregation
# =========================================================

UNCONDITIONAL_MAIN_COLUMNS = [
    "Method", "Mode", "Count",
    "CR", "SQS", "DRL", "ELA", "Obj_P", "SDM", "FSLD"
]

CONTROL_TABLE_COLUMNS = [
    "Method", "Mode", "Count",
    "CSR", "MPA", "ROI_SQS", "Obj_R", "ROI_FSLD", "IRS", "RCS"
]

SUPPLEMENT_TABLE_COLUMNS = [
    "Method", "Mode", "Count",
    "QSR", "ROI_QSR", "CCM", "RSR", "RSR_strict"
]

INTERACTION_TABLE_COLUMNS = [
    "Method", "Mode", "Count",
    "IRS", "RCS", "IRS_relation", "IRS_geometry", "IRS_kinematics"
]

FULL_COLUMNS = [
    "Method", "Mode", "Count",
    "CSR", "MPA", "CR", "SQS", "QSR", "ROI_SQS", "ROI_QSR", "CCM",
    "RSR", "RSR_strict",
    "DRL", "ELA", "Obj_P", "Obj_R", "SDM",
    "FSLD", "ROI_FSLD",
    "IRS", "RCS",
    "IRS_relation", "IRS_geometry", "IRS_kinematics",
]


def aggregate_metric_mean(results: List[SceneEvalResult], field: str) -> float:
    return safe_mean([getattr(r, field) for r in results])


def compute_group_fsld(results: List[SceneEvalResult], ref_stats: Dict[str, Any]) -> float:
    vecs = [r.scene_stat_vector for r in results if r.scene_stat_vector is not None]
    if not vecs:
        return nan()

    gen_stats = compute_distribution_stats(vecs)
    mu_g = np.asarray(gen_stats["mean"], dtype=np.float64)
    sg_g = np.asarray(gen_stats["cov"], dtype=np.float64)
    mu_r = np.asarray(ref_stats["scene_distribution"]["mean"], dtype=np.float64)
    sg_r = np.asarray(ref_stats["scene_distribution"]["cov"], dtype=np.float64)

    if mu_g.size == 0 or mu_r.size == 0:
        return nan()
    return frechet_distance_from_stats(mu_g, sg_g, mu_r, sg_r)


def compute_group_roi_fsld(results: List[SceneEvalResult], ref_stats: Dict[str, Any]) -> float:
    vecs = [r.roi_scene_stat_vector for r in results if r.roi_scene_stat_vector is not None]
    if not vecs:
        return nan()

    gen_stats = compute_distribution_stats(vecs)
    mu_g = np.asarray(gen_stats["mean"], dtype=np.float64)
    sg_g = np.asarray(gen_stats["cov"], dtype=np.float64)
    mu_r = np.asarray(ref_stats["roi_scene_distribution"]["mean"], dtype=np.float64)
    sg_r = np.asarray(ref_stats["roi_scene_distribution"]["cov"], dtype=np.float64)

    if mu_g.size == 0 or mu_r.size == 0:
        return nan()
    return frechet_distance_from_stats(mu_g, sg_g, mu_r, sg_r)


def aggregate_results(
    results: List[SceneEvalResult],
    method_name: str,
    mode: str,
    ref_stats: Dict[str, Any]
) -> Dict[str, Any]:
    row = {
        "Method": method_name,
        "Mode": mode,
        "Count": len(results),
    }

    for field in [
        "CSR", "MPA", "CR", "SQS", "QSR", "ROI_SQS", "ROI_QSR", "CCM",
        "RSR", "RSR_strict",
        "DRL", "ELA", "Obj_P", "Obj_R", "SDM",
        "IRS", "RCS",
        "IRS_relation", "IRS_geometry", "IRS_kinematics",
    ]:
        row[field] = aggregate_metric_mean(results, field)

    row["FSLD"] = compute_group_fsld(results, ref_stats)
    row["ROI_FSLD"] = compute_group_roi_fsld(results, ref_stats)
    return row


def aggregate_by_key(
    results: List[SceneEvalResult],
    group_key: str,
    method_name: str,
    mode: str,
    ref_stats: Dict[str, Any]
) -> pd.DataFrame:
    buckets = defaultdict(list)
    for r in results:
        buckets[getattr(r, group_key)].append(r)

    rows = []
    for key in sorted(buckets.keys()):
        row = aggregate_results(buckets[key], method_name, mode, ref_stats)
        row[group_key] = key
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Table formatting
# =========================================================

def format_number(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "-"
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def to_markdown_table(df: pd.DataFrame, columns: List[str], digits: int = 3) -> str:
    show = df.copy()
    for c in columns:
        if c in {"Method", "Mode"}:
            continue
        if c in show.columns:
            show[c] = show[c].apply(lambda x: format_number(x, digits))
    return show[columns].to_markdown(index=False)


def to_latex_table(df: pd.DataFrame, columns: List[str], digits: int = 3) -> str:
    show = df.copy()
    for c in columns:
        if c in {"Method", "Mode"}:
            continue
        if c in show.columns:
            show[c] = show[c].apply(lambda x: format_number(x, digits))
    return show[columns].to_latex(index=False, escape=False)


# =========================================================
# Evaluation runners
# =========================================================

def run_g0_mode(
    scenario_cache: Path,
    ref_stats: Dict[str, Any],
    vector_name: str = "sledge_vector.gz"
) -> List[SceneEvalResult]:
    paths = collect_gz_files(scenario_cache, vector_name)
    results = []
    for p in paths:
        try:
            bundle = load_scene_bundle_from_path(
                p,
                source_type="g0",
                scenario_type=None,
                severity=None,
                prompt_info={}
            )
            results.append(evaluate_one_scene_g0(bundle, ref_stats))
        except Exception as e:
            print(f"[WARN] skip failed G0 scene {p}: {e}")
    return results


def run_manifest_mode(
    manifest_path: Path,
    which: str,
    generated_root: Optional[Path],
    edited_root: Optional[Path],
    ref_stats: Dict[str, Any],
    source_type: str,
    accepted_only: bool = False,
) -> List[SceneEvalResult]:
    rows = load_manifest_rows(manifest_path)
    results = []

    for row in rows:
        try:
            scene_path = resolve_manifest_scene_path(row, which, generated_root, edited_root)
        except Exception as e:
            print(f"[WARN] skip unresolved row scene_id={row.get('scene_id', 'N/A')}: {e}")
            continue

        prompt_info = dict(row)
        stype = infer_scenario_type(str(scene_path), prompt_info)
        severity = infer_severity(prompt_info, str(scene_path))

        try:
            bundle = load_scene_bundle_from_path(
                scene_path,
                source_type=source_type,
                scenario_type=stype,
                severity=severity,
                prompt_info=prompt_info,
            )
            res = evaluate_one_scene_controlled(bundle, ref_stats)
            if accepted_only and not bool(res.accepted):
                continue
            results.append(res)
        except Exception as e:
            print(f"[WARN] skip failed controlled scene {scene_path}: {e}")

    return results


# =========================================================
# Export
# =========================================================

def export_outputs(
    output_dir: Path,
    results: List[SceneEvalResult],
    summary_row: Dict[str, Any],
    by_type_df: pd.DataFrame,
    by_severity_df: pd.DataFrame,
    digits: int = 3,
) -> None:
    ensure_dir(output_dir)

    write_jsonl(output_dir / "results_main_table.jsonl", [asdict(r) for r in results])

    summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(output_dir / "main_table_full.csv", index=False)

    # Shared unconditional table
    main_df = summary_df[UNCONDITIONAL_MAIN_COLUMNS]
    main_df.to_csv(output_dir / "main_table.csv", index=False)
    with open(output_dir / "main_table.md", "w", encoding="utf-8") as f:
        f.write(to_markdown_table(main_df, UNCONDITIONAL_MAIN_COLUMNS, digits) + "\n")
    with open(output_dir / "main_table_latex.txt", "w", encoding="utf-8") as f:
        f.write(to_latex_table(main_df, UNCONDITIONAL_MAIN_COLUMNS, digits) + "\n")

    # Controlled table: only write if at least one control metric is defined
    control_df = summary_df[CONTROL_TABLE_COLUMNS].copy()
    metric_cols = [c for c in CONTROL_TABLE_COLUMNS if c not in {"Method", "Mode", "Count"}]
    if metric_cols and not control_df[metric_cols].isna().all(axis=None):
        control_df.to_csv(output_dir / "control_table.csv", index=False)
        with open(output_dir / "control_table.md", "w", encoding="utf-8") as f:
            f.write(to_markdown_table(control_df, CONTROL_TABLE_COLUMNS, digits) + "\n")
        with open(output_dir / "control_table_latex.txt", "w", encoding="utf-8") as f:
            f.write(to_latex_table(control_df, CONTROL_TABLE_COLUMNS, digits) + "\n")

    # Supplement table
    supplement_df = summary_df[SUPPLEMENT_TABLE_COLUMNS]
    supplement_df.to_csv(output_dir / "supplement_quality_table.csv", index=False)
    with open(output_dir / "supplement_quality_table.md", "w", encoding="utf-8") as f:
        f.write(to_markdown_table(supplement_df, SUPPLEMENT_TABLE_COLUMNS, digits) + "\n")

    # Interaction table: only for controlled methods
    interaction_df = summary_df[INTERACTION_TABLE_COLUMNS].copy()
    interaction_metric_cols = [c for c in INTERACTION_TABLE_COLUMNS if c not in {"Method", "Mode", "Count"}]
    if interaction_metric_cols and not interaction_df[interaction_metric_cols].isna().all(axis=None):
        interaction_df.to_csv(output_dir / "interaction_table.csv", index=False)
        with open(output_dir / "interaction_table.md", "w", encoding="utf-8") as f:
            f.write(to_markdown_table(interaction_df, INTERACTION_TABLE_COLUMNS, digits) + "\n")
        with open(output_dir / "interaction_table_latex.txt", "w", encoding="utf-8") as f:
            f.write(to_latex_table(interaction_df, INTERACTION_TABLE_COLUMNS, digits) + "\n")

    if not by_type_df.empty:
        by_type_df.to_csv(output_dir / "main_table_by_type.csv", index=False)
    if not by_severity_df.empty:
        by_severity_df.to_csv(output_dir / "main_table_by_severity.csv", index=False)

    print(f"[OK] wrote: {output_dir / 'results_main_table.jsonl'}")
    print(f"[OK] wrote: {output_dir / 'main_table.csv'}")
    print(f"[OK] wrote: {output_dir / 'main_table_full.csv'}")


# =========================================================
# Main
# =========================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Main-table evaluator for SLEDGE and controlled generated scenarios")

    parser.add_argument("--mode", type=str, required=True, choices=["g0", "manifest"])
    parser.add_argument("--method-name", type=str, default="Ours")

    # g0
    parser.add_argument("--scenario-cache", type=str, default=None)

    # manifest
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--which", type=str, default="generated", choices=["generated", "edited"])
    parser.add_argument("--generated-root", type=str, default=None)
    parser.add_argument("--edited-root", type=str, default=None)
    parser.add_argument("--accepted-only", action="store_true")

    # reference
    parser.add_argument("--reference-cache", type=str, default=None)
    parser.add_argument("--reference-stats-json", type=str, default=None)

    # output
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--vector-name", type=str, default="sledge_vector.gz")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    output_dir = Path(args.output)
    ensure_dir(output_dir)

    # Reference stats
    if args.reference_stats_json is not None and Path(args.reference_stats_json).exists():
        ref_stats = read_json(Path(args.reference_stats_json))
    else:
        if args.reference_cache is not None:
            ref_paths = collect_gz_files(Path(args.reference_cache), args.vector_name)
        elif args.mode == "g0" and args.scenario_cache is not None:
            ref_paths = collect_gz_files(Path(args.scenario_cache), args.vector_name)
        elif args.mode == "manifest" and args.generated_root is not None:
            ref_paths = collect_gz_files(Path(args.generated_root), args.vector_name)
        elif args.mode == "manifest" and args.edited_root is not None:
            ref_paths = collect_gz_files(Path(args.edited_root), args.vector_name)
        else:
            raise ValueError("Reference cache cannot be inferred. Please provide --reference-cache or --reference-stats-json.")

        ref_stats = compute_reference_stats_from_paths(ref_paths, output_json=output_dir / "reference_stats.json")

    # Run evaluation
    if args.mode == "g0":
        if args.scenario_cache is None:
            raise ValueError("--scenario-cache is required when --mode g0")
        results = run_g0_mode(
            scenario_cache=Path(args.scenario_cache),
            ref_stats=ref_stats,
            vector_name=args.vector_name,
        )
        mode_name = "natural_unlabeled"

    elif args.mode == "manifest":
        if args.manifest is None:
            raise ValueError("--manifest is required when --mode manifest")
        results = run_manifest_mode(
            manifest_path=Path(args.manifest),
            which=args.which,
            generated_root=Path(args.generated_root) if args.generated_root else None,
            edited_root=Path(args.edited_root) if args.edited_root else None,
            ref_stats=ref_stats,
            source_type=args.method_name.lower(),
            accepted_only=bool(args.accepted_only),
        )
        mode_name = "conditional"

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if not results:
        raise RuntimeError("No valid scenes were evaluated.")

    summary_row = aggregate_results(results, args.method_name, mode_name, ref_stats)
    by_type_df = aggregate_by_key(results, "scenario_type", args.method_name, mode_name, ref_stats)
    by_severity_df = aggregate_by_key(results, "severity", args.method_name, mode_name, ref_stats)

    export_outputs(
        output_dir=output_dir,
        results=results,
        summary_row=summary_row,
        by_type_df=by_type_df,
        by_severity_df=by_severity_df,
        digits=args.digits,
    )


if __name__ == "__main__":
    main()