from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex
from sledge.semantic_control.geometry_metrics import oriented_box_corners_from_state, state_xy
from sledge.semantic_control.io import load_raw_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an edited SLEDGE raw scene in BEV.")
    parser.add_argument("--raw", type=str, required=True, help="Path to sledge_raw.gz.")
    parser.add_argument("--semantic_report", type=str, default=None, help="Optional semantic_report.json.")
    parser.add_argument("--output_png", type=str, required=True, help="Output image path.")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-8.0, 35.0])
    parser.add_argument("--ylim", type=float, nargs=2, default=[-15.0, 15.0])
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene, _ = load_raw_scene(args.raw)
    report_payload = _load_json(args.semantic_report) if args.semantic_report else {}

    edit_result = report_payload.get("edit_result", {})
    report = report_payload.get("report", {})

    primary_actor_type = edit_result.get("primary_actor_type")
    primary_actor_index = int(edit_result.get("primary_actor_index", -1))
    pedestrian_index = int(edit_result.get("pedestrian_index", -1))
    occluder_index = int(edit_result.get("occluder_index", -1))
    static_obstacle_index = int(edit_result.get("static_obstacle_index", -1))

    if primary_actor_type == "pedestrian" and pedestrian_index >= 0:
        primary_elem_name = "pedestrians"
        primary_actor_index = pedestrian_index
    elif primary_actor_type == "static_obstacle":
        primary_elem_name = "static_objects"
    else:
        primary_elem_name = "vehicles"

    fig, ax = plt.subplots(figsize=(10, 7))

    # Ego proxy.
    ego_box = Rectangle((-2.4, -0.95), 4.8, 1.9, fill=False, linewidth=2.0, label="ego")
    ax.add_patch(ego_box)
    ax.text(0.0, 0.0, "ego", ha="center", va="center", fontsize=8)

    # Scene elements.
    _draw_element(ax, scene.vehicles, "vehicles", primary_elem_name, primary_actor_index, occluder_index, static_obstacle_index)
    _draw_element(ax, scene.pedestrians, "pedestrians", primary_elem_name, primary_actor_index, occluder_index, static_obstacle_index)
    _draw_element(ax, scene.static_objects, "static_objects", primary_elem_name, primary_actor_index, occluder_index, static_obstacle_index)

    # Conflict point and line of sight.
    conflict_xy = edit_result.get("conflict_point_xy", None)
    if conflict_xy and len(conflict_xy) >= 2:
        cx, cy = float(conflict_xy[0]), float(conflict_xy[1])
        ax.plot([cx], [cy], marker="x", markersize=8)
        ax.text(cx, cy + 0.6, "conflict", fontsize=8)

    actor_xy = _get_actor_xy(scene, primary_elem_name, primary_actor_index)
    if actor_xy is not None:
        ax.plot([0.0, actor_xy[0]], [0.0, actor_xy[1]], linestyle="--", linewidth=1.5)
        ax.text(actor_xy[0], actor_xy[1] + 0.5, "primary", fontsize=8)

    # ROIs from report.
    for roi in edit_result.get("preserved_rois", []):
        x_min, y_min = float(roi["x_min"]), float(roi["y_min"])
        x_max, y_max = float(roi["x_max"]), float(roi["y_max"])
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, linestyle=":", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x_min, y_max, roi.get("tag", "roi"), fontsize=7)

    # Text summary.
    validation = report.get("validation", {})
    ssr = validation.get("semantic_satisfaction_rate")
    overall = validation.get("overall_pass")
    semantic_signature = report.get("semantic_signature", "")

    title = args.title or "Compositional Semantic Edit"
    ax.set_title(f"{title}\nSSR={ssr}, overall_pass={overall}\n{semantic_signature}", fontsize=10)

    ax.set_xlim(args.xlim)
    ax.set_ylim(args.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved visualization to {out}")


def _load_json(path_like: Optional[str]) -> Dict:
    if not path_like:
        return {}
    with open(path_like, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _draw_element(ax, elem, elem_name: str, primary_elem_name: str, primary_index: int, occluder_index: int, static_index: int):
    mask = np.asarray(elem.mask).astype(bool)
    states = np.asarray(elem.states)

    for idx in np.where(mask)[0]:
        state = states[idx]
        corners = oriented_box_corners_from_state(state)

        linewidth = 0.8
        label = elem_name[:-1]
        if elem_name == primary_elem_name and int(idx) == int(primary_index):
            linewidth = 2.5
            label = "primary"
        elif elem_name == "vehicles" and int(idx) == int(occluder_index):
            linewidth = 2.5
            label = "occluder"
        elif elem_name == "static_objects" and int(idx) == int(static_index):
            linewidth = 2.5
            label = "static obstacle"

        patch = MplPolygon(corners, closed=True, fill=False, linewidth=linewidth)
        ax.add_patch(patch)

        x = float(state[AgentIndex.X])
        y = float(state[AgentIndex.Y])
        ax.text(x, y, f"{label}:{idx}", fontsize=6, ha="center", va="center")


def _get_actor_xy(scene, elem_name: str, idx: int) -> Optional[Tuple[float, float]]:
    if idx < 0:
        return None
    if elem_name == "vehicles":
        elem = scene.vehicles
    elif elem_name == "pedestrians":
        elem = scene.pedestrians
    elif elem_name == "static_objects":
        elem = scene.static_objects
    else:
        return None

    if idx >= len(elem.mask) or not bool(elem.mask[idx]):
        return None
    return state_xy(elem.states[idx])


if __name__ == "__main__":
    main()
