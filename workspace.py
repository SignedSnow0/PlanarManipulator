"""Workspace geometry and plotting for the two-link arm in ``main.py``.

Shapely builds the joint-limited reachable polygon from boundary circles, and
Matplotlib draws that polygon on a supplied axes object. The arm only needs to
expose ``l1``, ``l2``, ``limits1``, ``limits2``, and ``is_reachable()``.
"""

import numpy as np
from shapely.geometry import MultiPolygon, Point
from shapely.ops import polygonize, unary_union


def build_workspace(arm):
    """Build the joint-limited workspace as a Shapely polygon.

    Boundary circles come from the joint limits and link lengths. They are
    noded, split into faces, and each face is kept when its representative
    point passes ``arm.is_reachable()``.

    Args:
        arm: A two-link arm exposing lengths, limits, and reachability.
        resolution: Segments used by Shapely to approximate each circle.

    Returns:
        A ``(workspace, circles)`` tuple.
    """
    q1_min, q1_max = arm.limits1
    q2_min, q2_max = arm.limits2

    circles = [
        (0, 0, np.sqrt(arm.l1**2 + arm.l2**2 + 2 * arm.l1 * arm.l2 * np.cos(q2_min))),
        (0, 0, np.sqrt(arm.l1**2 + arm.l2**2 + 2 * arm.l1 * arm.l2 * np.cos(q2_max))),
        (arm.l1 * np.cos(q1_min), arm.l1 * np.sin(q1_min), arm.l2),
        (arm.l1 * np.cos(q1_max), arm.l1 * np.sin(q1_max), arm.l2),
    ]

    # Full extension at q2 = 0 and maximum fold at q2 = +/-pi.
    if q2_min < 0 < q2_max:
        circles.append((0, 0, arm.l1 + arm.l2))
    if q2_min < np.pi < q2_max or q2_min < -np.pi < q2_max:
        circles.append((0, 0, abs(arm.l1 - arm.l2)))

    rings = [
        Point(cx, cy).buffer(r).exterior
        for cx, cy, r in circles
        if r > 0
    ]
    noded_lines = unary_union(rings)
    faces = [
        face
        for face in polygonize(noded_lines)
        if arm.is_reachable(
            face.representative_point().x, face.representative_point().y
        )
    ]
    return unary_union(faces), circles


def plot_workspace(ax, arm):
    """Draw reachable faces in white and unreachable space in light gray."""
    workspace, _ = build_workspace(arm)

    polys = workspace.geoms if isinstance(workspace, MultiPolygon) else [workspace]

    ax.set_facecolor('lightgray')
    for poly in polys:
        if poly.is_empty:
            continue
        x_ext, y_ext = poly.exterior.xy
        ax.fill(x_ext, y_ext, color='white', label='Workspace')
        ax.plot(x_ext, y_ext, color='white')

        for interior in poly.interiors:
            x_int, y_int = interior.xy
            ax.fill(x_int, y_int, color='lightgray')
            ax.plot(x_int, y_int, color='lightgray')

    max_reach = arm.l1 + arm.l2
    ax.set_xlim(-max_reach * 1.15, max_reach * 1.15)
    ax.set_ylim(-max_reach * 1.15, max_reach * 1.15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
