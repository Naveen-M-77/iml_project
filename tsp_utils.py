"""
Shared TSP utilities for delivery route optimization.

Provides:
- haversine_distance: Great-circle distance between two lat/lon points
- calculate_route_distance: Total distance along a route
- solve_tsp: Nearest-neighbor + 2-opt TSP heuristic
- plot_optimized_route: Matplotlib visualization of an optimized route
"""

import logging
import warnings
from math import radians

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; overridden when callers do plt.show()
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def haversine_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points on Earth.

    Parameters
    ----------
    coord1 : tuple of (latitude, longitude) in degrees
    coord2 : tuple of (latitude, longitude) in degrees

    Returns
    -------
    float
        Distance in kilometres.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def calculate_route_distance(route):
    """
    Calculate the total distance for an ordered sequence of (lat, lon) points.

    Parameters
    ----------
    route : list of (lat, lon) tuples

    Returns
    -------
    float
        Total distance in kilometres.
    """
    if len(route) <= 1:
        return 0.0
    return sum(haversine_distance(route[i], route[i + 1]) for i in range(len(route) - 1))


# ---------------------------------------------------------------------------
# TSP solver
# ---------------------------------------------------------------------------

def solve_tsp(points, max_points=30, max_iterations=100):
    """
    Solve TSP using nearest-neighbour heuristic + 2-opt improvement.

    Parameters
    ----------
    points : list of (lat, lon) tuples
        The first point is treated as the depot / starting location.
    max_points : int
        If ``len(points) > max_points`` the list is truncated (with a warning).
    max_iterations : int
        Maximum number of outer 2-opt improvement passes.

    Returns
    -------
    route : list of (lat, lon)
        Optimized visit order.
    distance : float
        Total route distance in km.
    """
    if len(points) > max_points:
        logger.warning(
            "TSP solver: truncating from %d to %d points (max_points limit). "
            "Route quality may be degraded.",
            len(points), max_points,
        )
        warnings.warn(
            f"TSP solver truncated route from {len(points)} to {max_points} points",
            RuntimeWarning,
            stacklevel=2,
        )
        points = points[:max_points]

    if len(points) <= 2:
        return list(points), calculate_route_distance(points)

    # --- Nearest-neighbour construction ---
    current_point = points[0]
    unvisited = list(points[1:])
    route = [current_point]

    while unvisited:
        nearest = min(unvisited, key=lambda x: haversine_distance(current_point, x))
        route.append(nearest)
        unvisited.remove(nearest)
        current_point = nearest

    # --- 2-opt improvement ---
    improved = True
    best_distance = calculate_route_distance(route)
    iterations_left = max_iterations

    while improved and iterations_left > 0:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_distance = calculate_route_distance(new_route)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
        iterations_left -= 1

    if iterations_left == 0 and improved:
        logger.info(
            "2-opt reached max_iterations=%d; further improvement may be possible.",
            max_iterations,
        )

    return route, best_distance


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_optimized_route(route_df, route_points, distance, cluster_id, save_path=None):
    """
    Visualize an optimized delivery route for a single cluster.

    Parameters
    ----------
    route_df : pandas.DataFrame
        Cluster data (must contain Restaurant_latitude/longitude columns).
    route_points : list of (lat, lon)
        Ordered route (first point = restaurant).
    distance : float
        Total route distance in km.
    cluster_id : int or str
        Cluster identifier for the plot title.
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    restaurant = route_points[0]
    deliveries = route_points[1:]

    # Restaurant
    ax.scatter(restaurant[1], restaurant[0],
               color="red", marker="X", s=200, label="Restaurant", zorder=5)

    # Delivery points
    if deliveries:
        ax.scatter([p[1] for p in deliveries], [p[0] for p in deliveries],
                   color="blue", s=50, label="Delivery Points", zorder=4)

    # Route path
    ax.plot([p[1] for p in route_points], [p[0] for p in route_points],
            color="green", linestyle="-", linewidth=1, alpha=0.7, label="Optimized Route",
            zorder=3)

    # Annotate delivery order
    for i, point in enumerate(deliveries):
        ax.annotate(str(i + 1), (point[1], point[0]),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8)

    ax.set_title(
        f"Optimized Delivery Route – Cluster {cluster_id}\n"
        f"Total Distance: {distance:.2f} km | {len(deliveries)} Deliveries"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Route plot saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)
