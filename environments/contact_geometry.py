"""Shared continuous, route-aware contact geometry (no branch-id shortcuts)."""
import numpy as np


def continuous_route_distance(tree, positions, stations, distance, next_hop):
    """Station route length corrected by continuous axial displacement.

    Radial separation remains the responsibility of the independent Euclidean
    gate. At the target station the axial displacement must not disappear.
    Clot routes are anchored at their declared clot station, as in shaping.
    """
    st = np.asarray(stations)
    step = tree.points[next_hop[st]] - tree.points[st]
    length = np.linalg.norm(step, axis=-1)
    valid = length > 1e-8
    unit = step / np.maximum(length[..., None], 1e-8)
    offset = positions - tree.points[st]
    along = np.clip(np.sum(offset * unit, axis=-1), -length, length)
    axial = np.abs(np.sum(offset * tree.tangents[st], axis=-1))
    return np.maximum(distance[st] - np.where(valid, along, -axial), 0.0)


def append_flow_features(nodes, flow, route_delta, lube, max_speed,
                         time_progress, remaining, crowding, control_margin=True):
    """Append six features; never repurpose any of the original 36 slots."""
    speed = np.linalg.norm(flow, axis=-1)
    direction = route_delta / np.maximum(
        np.linalg.norm(route_delta, axis=-1, keepdims=True), 1e-8)
    along = np.sum(flow * direction, axis=-1)
    nodes[..., 36] = np.log1p(speed / max_speed)
    nodes[..., 37] = np.sign(along) * np.log1p(np.abs(along) / max_speed)
    nodes[..., 38] = lube - speed / max_speed if control_margin else 0.0
    nodes[..., 39] = time_progress
    nodes[..., 40] = remaining
    nodes[..., 41] = crowding
