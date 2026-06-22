from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sledge.autoencoder.preprocessing.features.sledge_vector_feature import AgentIndex, SledgeVectorRaw


Point = Tuple[float, float]
Polygon = List[Point]


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def state_xy(state: np.ndarray) -> Point:
    return float(state[AgentIndex.X]), float(state[AgentIndex.Y])


def state_heading(state: np.ndarray) -> float:
    return float(state[AgentIndex.HEADING])


def state_size(state: np.ndarray) -> Tuple[float, float]:
    width = float(max(state[AgentIndex.WIDTH], 0.5))
    length = float(max(state[AgentIndex.LENGTH], 0.5))
    return width, length


def state_speed(state: np.ndarray) -> float:
    return float(max(state[AgentIndex.VELOCITY], 0.0))


def velocity_from_state(state: np.ndarray) -> Point:
    heading = state_heading(state)
    speed = state_speed(state)
    return speed * math.cos(heading), speed * math.sin(heading)


def oriented_box_corners_from_state(state: np.ndarray, margin: float = 0.0) -> Polygon:
    x, y = state_xy(state)
    heading = state_heading(state)
    width, length = state_size(state)
    return oriented_box_corners(x, y, heading, width + 2 * margin, length + 2 * margin)


def oriented_box_corners(x: float, y: float, heading: float, width: float, length: float) -> Polygon:
    """
    Return four corners of an oriented rectangle.

    Convention:
      - length axis follows heading
      - width axis is heading + pi/2
    """
    c = math.cos(heading)
    s = math.sin(heading)

    hx = 0.5 * length
    hy = 0.5 * width

    # Local corners: front-left, front-right, rear-right, rear-left.
    local = [(hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy)]
    corners: Polygon = []
    for lx, ly in local:
        wx = x + lx * c - ly * s
        wy = y + lx * s + ly * c
        corners.append((float(wx), float(wy)))
    return corners


def polygon_edges(poly: Polygon) -> Iterable[Tuple[Point, Point]]:
    for i in range(len(poly)):
        yield poly[i], poly[(i + 1) % len(poly)]


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-6) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(_orientation(a, b, p)) <= eps
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-6) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True

    return False


def point_in_polygon(point: Point, poly: Polygon) -> bool:
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / max(y2 - y1, 1e-9) + x1):
            inside = not inside
    return inside


def segment_intersects_polygon(a: Point, b: Point, poly: Polygon) -> bool:
    if point_in_polygon(a, poly) or point_in_polygon(b, poly):
        return True
    for c, d in polygon_edges(poly):
        if segments_intersect(a, b, c, d):
            return True
    return False


def line_of_sight_intersects_box(
    ego_xy: Point,
    target_xy: Point,
    occluder_state: np.ndarray,
    margin: float = 0.0,
) -> bool:
    box = oriented_box_corners_from_state(occluder_state, margin=margin)
    return segment_intersects_polygon(ego_xy, target_xy, box)


def estimate_ego_speed(scene: SledgeVectorRaw) -> float:
    ego_states = np.asarray(scene.ego.states, dtype=np.float32).reshape(-1)
    if ego_states.size == 0:
        return 6.0
    if ego_states.size >= 2:
        speed = float(np.linalg.norm(ego_states[:2]))
    else:
        speed = float(abs(ego_states[0]))
    return float(np.clip(speed, 2.5, 15.0))


def compute_lateral_interaction_ttc(
    actor_state: np.ndarray,
    ego_speed_mps: float,
    lane_y: float = 0.0,
    conflict_x: Optional[float] = None,
) -> dict:
    """
    TTC proxy for lateral/crossing conflicts.

    We estimate:
      ego_time = time for ego to reach conflict_x.
      actor_time = time for actor to reach lane_y.
      interaction_ttc = max(ego_time, actor_time) if both are future times.

    This is more useful than pure relative TTC for crossing cases.
    """
    x, y = state_xy(actor_state)
    vx, vy = velocity_from_state(actor_state)

    if conflict_x is None:
        conflict_x = x

    ego_time = float("inf")
    if ego_speed_mps > 1e-3 and conflict_x > 0:
        ego_time = conflict_x / ego_speed_mps

    actor_time = float("inf")
    if abs(vy) > 1e-3:
        t = (lane_y - y) / vy
        if t > 0:
            actor_time = t

    if math.isfinite(ego_time) and math.isfinite(actor_time):
        interaction_ttc = max(ego_time, actor_time)
        arrival_time_gap = abs(ego_time - actor_time)
    else:
        interaction_ttc = float("inf")
        arrival_time_gap = float("inf")

    return {
        "ego_time_s": ego_time,
        "actor_time_s": actor_time,
        "interaction_ttc_s": interaction_ttc,
        "arrival_time_gap_s": arrival_time_gap,
        "actor_vx_mps": vx,
        "actor_vy_mps": vy,
    }


def compute_longitudinal_ttc(
    actor_state: np.ndarray,
    ego_speed_mps: float,
    ego_length_m: float = 4.8,
) -> dict:
    x, _ = state_xy(actor_state)
    actor_speed = state_speed(actor_state)
    _, length = state_size(actor_state)

    center_gap = max(0.0, x)
    bumper_gap = max(0.0, x - 0.5 * ego_length_m - 0.5 * length)
    closing_speed = ego_speed_mps - actor_speed

    if bumper_gap > 0 and closing_speed > 1e-3:
        ttc = bumper_gap / closing_speed
    else:
        ttc = float("inf")

    return {
        "center_gap_m": center_gap,
        "bumper_gap_m": bumper_gap,
        "ego_speed_mps": ego_speed_mps,
        "actor_speed_mps": actor_speed,
        "closing_speed_mps": closing_speed,
        "ttc_s": ttc,
    }


def compute_merging_proxy_metrics(
    actor_state: np.ndarray,
    ego_speed_mps: float,
    lane_y: float = 0.0,
    lane_half_width_m: float = 1.8,
) -> dict:
    x, y = state_xy(actor_state)
    width, _ = state_size(actor_state)

    abs_y = abs(y - lane_y)
    lateral_intrusion_m = max(0.0, lane_half_width_m + 0.5 * width - abs_y)
    lateral_gap_to_lane_m = max(0.0, abs_y - lane_half_width_m - 0.5 * width)

    time_to_reach_actor_x = x / ego_speed_mps if ego_speed_mps > 1e-3 and x > 0 else float("inf")

    return {
        "x_m": x,
        "y_m": y,
        "abs_lateral_offset_m": abs_y,
        "lateral_intrusion_m": lateral_intrusion_m,
        "lateral_gap_to_lane_m": lateral_gap_to_lane_m,
        "time_to_actor_x_s": time_to_reach_actor_x,
    }


def compute_lateral_gap_to_lane_boundary(
    actor_state: np.ndarray,
    lane_y: float = 0.0,
    lane_half_width_m: float = 1.8,
) -> dict:
    _, y = state_xy(actor_state)
    width, _ = state_size(actor_state)

    abs_y = abs(y - lane_y)
    gap = max(0.0, abs_y - lane_half_width_m - 0.5 * width)
    intrusion = max(0.0, lane_half_width_m + 0.5 * width - abs_y)

    return {
        "abs_lateral_offset_m": abs_y,
        "lateral_gap_to_lane_boundary_m": gap,
        "lateral_intrusion_m": intrusion,
        "lane_half_width_m": lane_half_width_m,
    }


def value_in_range(value: float, range_pair: Sequence[float], tolerance: float = 0.0) -> bool:
    lo, hi = float(range_pair[0]), float(range_pair[1])
    return (lo - tolerance) <= value <= (hi + tolerance)
