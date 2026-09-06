#!/usr/bin/env python3
"""
Route Optimizer — Time-based / KMeans++ Clustering + TSP

Clusters delivery locations using either **time-based** features or plain
**KMeans++**, then optimizes each cluster's delivery route with a
nearest-neighbour + 2-opt TSP heuristic.

Key features
------------
- Automatically selects the best clustering method (``time_based`` or
  ``kmeans++``) based on silhouette score (``--method auto``).
- Assigns delivery persons to clusters via the **Hungarian algorithm**
  (``scipy.optimize.linear_sum_assignment``) — minimises total distance
  from each person's average location to each cluster centroid.
- Large clusters are recursively split so that no single route exceeds
  ``--max-deliveries`` points.

Usage
-----
    python route_optimizer_time_kmeans.py --input zomato_dataset.csv
    python route_optimizer_time_kmeans.py --input zomato_dataset.csv --method time_based --max-deliveries 20
"""

import argparse
import gc
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from tsp_utils import (
    calculate_route_distance,
    haversine_distance,
    plot_optimized_route,
    solve_tsp,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the delivery dataset from *csv_path*."""
    if not os.path.isfile(csv_path):
        logger.error("File not found: %s", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute nulls, engineer time & distance features."""
    df = df.copy()

    # Numerical imputation
    for col in ["Delivery_person_Age", "Delivery_person_Ratings",
                "multiple_deliveries", "Time_taken (min)"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical imputation
    for col in ["Weather_conditions", "Road_traffic_density", "Festival", "City"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Datetime handling
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")

    if "Time_Orderd" in df.columns:
        df["Time_Orderd"] = pd.to_datetime(
            df["Time_Orderd"], format="%H:%M:%S", errors="coerce"
        ).fillna(pd.to_datetime("12:00:00", format="%H:%M:%S"))

        df["Order_Time_min"] = df["Time_Orderd"].dt.hour * 60 + df["Time_Orderd"].dt.minute
        df["time_sin"] = np.sin(2 * np.pi * df["Order_Time_min"] / 1440)
        df["time_cos"] = np.cos(2 * np.pi * df["Order_Time_min"] / 1440)

    # Time window bins
    if "Order_Time_min" in df.columns:
        df["Time_Window"] = pd.cut(
            df["Order_Time_min"],
            bins=[0, 360, 720, 1440],
            labels=["Morning", "Afternoon", "Night"],
            include_lowest=True, ordered=False,
        ).fillna("Night")
        df["is_peak_hours"] = df["Time_Orderd"].dt.hour.apply(
            lambda x: 1 if (7 <= x < 10) or (17 <= x < 20) else 0
        )

    # Adjusted distance with traffic/weather/vehicle factors
    factors_traffic = {"Low": 1, "Medium": 1.3, "High": 1.6, "Jam": 2}
    factors_weather = {"Sunny": 1, "Cloudy": 1.1, "Fog": 1.2, "Rain": 1.5,
                       "Stormy": 2, "Sandstorms": 2.2, "Windy": 1.3}
    factors_vehicle = {1: 1.5, 2: 1.3, 3: 1.1}

    def _calc_adj_dist(row):
        base = great_circle(
            (row["Restaurant_latitude"], row["Restaurant_longitude"]),
            (row["Delivery_location_latitude"], row["Delivery_location_longitude"]),
        ).km
        return (
            base
            * factors_traffic.get(row.get("Road_traffic_density", ""), 1)
            * factors_weather.get(row.get("Weather_conditions", ""), 1)
            * factors_vehicle.get(row.get("Vehicle_condition", 1), 1)
        )

    if all(c in df.columns for c in ["Restaurant_latitude", "Restaurant_longitude",
                                       "Delivery_location_latitude", "Delivery_location_longitude"]):
        df["adjusted_distance"] = df.apply(_calc_adj_dist, axis=1)
        before = len(df)
        df = df[df["adjusted_distance"] < 50]
        if len(df) < before:
            logger.info("Removed %d outlier rows (adjusted_distance >= 50 km).", before - len(df))

    # Capability score
    if "Delivery_person_Ratings" in df.columns and "Vehicle_condition" in df.columns:
        df["capability_score"] = 0.6 * df["Delivery_person_Ratings"] + 0.4 * df["Vehicle_condition"]

    logger.info("Data cleaned. %d rows remaining.", len(df))
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def perform_clustering(df, method="kmeans++", n_clusters=None, max_samples=None):
    """
    Run KMeans clustering. Returns (model, cluster_series).

    Parameters
    ----------
    method : str
        ``'kmeans++'``, ``'time_based'``, or ``'distance_optimized'``.
    n_clusters : int or None
        Defaults to number of unique delivery persons.
    max_samples : int or None
        If set, sample down before clustering.
    """
    working_df = df.copy()

    if n_clusters is None:
        n_clusters = df["Delivery_person_ID"].nunique()
        logger.info("Setting n_clusters = %d (one per delivery person).", n_clusters)

    if max_samples and len(working_df) > max_samples:
        working_df = working_df.sample(max_samples, random_state=42)
        logger.info("Sampled %d rows for clustering.", max_samples)

    feature_map = {
        "kmeans++": ["Delivery_location_latitude", "Delivery_location_longitude"],
        "time_based": ["Delivery_location_latitude", "Delivery_location_longitude",
                       "time_sin", "time_cos"],
        "distance_optimized": ["Delivery_location_latitude", "Delivery_location_longitude",
                               "Restaurant_latitude", "Restaurant_longitude"],
    }
    if method not in feature_map:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(feature_map)}")

    features = feature_map[method]

    # Verify required features exist
    missing = [f for f in features if f not in working_df.columns]
    if missing:
        logger.warning("Missing features %s for method '%s' — falling back to 'kmeans++'.", missing, method)
        method = "kmeans++"
        features = feature_map[method]

    scaler = StandardScaler()
    X = scaler.fit_transform(working_df[features])

    # Adapt n_init for large cluster counts to keep runtime reasonable
    n_init = 5 if n_clusters > 100 else 20
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=42,
        algorithm="elkan",
    )
    logger.info("Fitting KMeans (n_clusters=%d, n_init=%d)...", n_clusters, n_init)
    clusters = model.fit_predict(X)

    # Map back to original index
    cluster_series = pd.Series(-1, index=df.index, dtype="int32")
    cluster_series[working_df.index] = clusters

    return model, cluster_series


def evaluate_clustering(df, clusters, features):
    """Return dict of clustering quality metrics."""
    clustered_mask = clusters != -1
    if clustered_mask.sum() == 0:
        return {"error": "No points clustered"}

    X_full = StandardScaler().fit_transform(df.loc[clustered_mask, features])
    labels_full = clusters[clustered_mask]

    # Subsample for silhouette score if dataset is large (>10k) — it's O(n²)
    n_eval = min(len(X_full), 10000)
    if n_eval < len(X_full):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_full), n_eval, replace=False)
        X_eval, labels_eval = X_full[idx], labels_full.values[idx] if hasattr(labels_full, 'values') else labels_full[idx]
        logger.info("Evaluating silhouette on %d/%d samples.", n_eval, len(X_full))
    else:
        X_eval, labels_eval = X_full, labels_full

    metrics = {
        "silhouette": silhouette_score(X_eval, labels_eval),
        "calinski": calinski_harabasz_score(X_full, labels_full),
        "n_clusters": len(np.unique(labels_full)),
        "avg_cluster_size": clustered_mask.sum() / len(np.unique(labels_full)),
    }

    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X_full, labels_full)
    except Exception:
        logger.debug("Davies-Bouldin score unavailable.")

    cluster_counts = pd.Series(labels_full).value_counts()
    metrics["max_cluster_size"] = int(cluster_counts.max())
    metrics["min_cluster_size"] = int(cluster_counts.min())
    metrics["cluster_imbalance"] = cluster_counts.max() / max(cluster_counts.min(), 1)

    return metrics


# ---------------------------------------------------------------------------
# Smart clustering selection
# ---------------------------------------------------------------------------
def optimize_delivery_clusters(df, method="auto", max_samples=None):
    """
    Select best clustering method, cluster, and assign persons.

    Returns (model, df_with_clusters).
    """
    if "Delivery_person_ID" not in df.columns:
        raise ValueError("DataFrame must contain 'Delivery_person_ID' column.")

    n_persons = df["Delivery_person_ID"].nunique()
    logger.info("Found %d unique delivery persons.", n_persons)

    # Method selection
    if method == "auto":
        methods = ["time_based", "kmeans++"] if all(
            c in df.columns for c in ["time_sin", "time_cos"]
        ) else ["kmeans++"]
    else:
        methods = [method]

    best_score = -1
    best_result = None

    for m in methods:
        logger.info("Testing '%s' clustering...", m)
        try:
            model, clusters = perform_clustering(
                df, method=m, n_clusters=n_persons, max_samples=max_samples,
            )
            eval_features = ["Delivery_location_latitude", "Delivery_location_longitude"]
            metrics = evaluate_clustering(df, clusters, eval_features)

            logger.info(
                "  %s — silhouette: %.3f, sizes: %d–%d, imbalance: %.2f",
                m, metrics["silhouette"],
                metrics["min_cluster_size"], metrics["max_cluster_size"],
                metrics["cluster_imbalance"],
            )

            if metrics["silhouette"] > best_score:
                best_score = metrics["silhouette"]
                best_result = (m, model, clusters, metrics)
        except Exception as e:
            logger.error("Clustering failed for '%s': %s", m, e)

    if best_result is None:
        logger.warning(
            "FALLBACK: All clustering methods failed — using basic KMeans++. "
            "Output quality is DEGRADED."
        )
        model, clusters = perform_clustering(df, method="kmeans++", n_clusters=n_persons)
        eval_features = ["Delivery_location_latitude", "Delivery_location_longitude"]
        metrics = evaluate_clustering(df, clusters, eval_features)
        best_result = ("kmeans++ (fallback)", model, clusters, metrics)

    method_name, model, clusters, metrics = best_result
    logger.info("Selected '%s' with silhouette=%.3f", method_name, metrics["silhouette"])

    df["delivery_cluster"] = clusters

    # Assign delivery persons to clusters via Hungarian algorithm
    cluster_to_person = assign_persons_to_clusters(df, clusters, n_persons)
    df["assigned_person"] = df["delivery_cluster"].map(cluster_to_person)

    unmapped = df["assigned_person"].isna().sum()
    if unmapped > 0:
        logger.warning("%d rows have no assigned person (cluster = -1).", unmapped)

    return model, df


# ---------------------------------------------------------------------------
# Hungarian-algorithm person-to-cluster assignment
# ---------------------------------------------------------------------------
def assign_persons_to_clusters(df, clusters, n_persons):
    """
    Assign delivery persons to clusters by minimising total distance from
    each person's average location to each cluster centroid.

    Uses ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm).

    Returns
    -------
    dict
        ``{cluster_id: person_id}``
    """
    persons = df["Delivery_person_ID"].unique()
    unique_clusters = np.sort(np.unique(clusters[clusters != -1]))

    n_c = len(unique_clusters)
    n_p = len(persons)

    logger.info(
        "Assigning %d persons to %d clusters via Hungarian algorithm...", n_p, n_c,
    )

    # Compute each person's centroid (mean delivery location)
    person_centroids = (
        df.groupby("Delivery_person_ID")[
            ["Delivery_location_latitude", "Delivery_location_longitude"]
        ]
        .mean()
    )

    # Compute each cluster's centroid
    cluster_centroids = (
        df[df["delivery_cluster"] != -1]
        .groupby("delivery_cluster")[
            ["Delivery_location_latitude", "Delivery_location_longitude"]
        ]
        .mean()
    )

    # Build cost matrix (n_p × n_c) using vectorized haversine
    from sklearn.metrics.pairwise import haversine_distances as _sklearn_haversine

    # Align persons/clusters to numpy arrays
    p_coords = person_centroids.reindex(persons)[
        ["Delivery_location_latitude", "Delivery_location_longitude"]
    ].values  # (n_p, 2) in degrees
    c_coords = cluster_centroids.reindex(unique_clusters)[
        ["Delivery_location_latitude", "Delivery_location_longitude"]
    ].values  # (n_c, 2) in degrees

    # sklearn haversine expects radians
    p_rad = np.radians(p_coords)
    c_rad = np.radians(c_coords)

    # haversine_distances returns values in radians; multiply by Earth radius
    dist_matrix = _sklearn_haversine(p_rad, c_rad) * 6371  # (n_p, n_c) in km

    # Pad to square if needed
    size = max(n_p, n_c)
    cost_matrix = np.full((size, size), fill_value=1e9)
    cost_matrix[:n_p, :n_c] = dist_matrix

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_p and c < n_c:
            mapping[unique_clusters[c]] = persons[r]

    total_cost = sum(
        cost_matrix[r, c] for r, c in zip(row_ind, col_ind) if r < n_p and c < n_c
    )
    logger.info(
        "Hungarian assignment complete. Total assignment cost: %.2f km.", total_cost,
    )

    return mapping


# ---------------------------------------------------------------------------
# Cluster processing & TSP
# ---------------------------------------------------------------------------
def process_cluster(cluster_df, cluster_id, max_deliveries, depth=0, max_depth=5):
    """Process a single cluster: solve TSP, splitting if too large."""
    if len(cluster_df) == 0:
        return []

    if depth >= max_depth:
        logger.warning(
            "Max recursion depth reached for cluster %s — treating as single deliveries. "
            "Route quality is DEGRADED.", cluster_id,
        )
        return _handle_as_single_deliveries(cluster_df, cluster_id)

    restaurant_loc = (
        cluster_df["Restaurant_latitude"].iloc[0],
        cluster_df["Restaurant_longitude"].iloc[0],
    )
    delivery_points = list(zip(
        cluster_df["Delivery_location_latitude"],
        cluster_df["Delivery_location_longitude"],
    ))

    if len(delivery_points) == 1:
        route = [restaurant_loc, delivery_points[0]]
        dist = haversine_distance(restaurant_loc, delivery_points[0])
        return [(cluster_id, route, dist, cluster_df.iloc[[0]])]

    if len(delivery_points) > max_deliveries:
        logger.info("Splitting cluster %s (%d deliveries).", cluster_id, len(delivery_points))
        return _split_and_solve(cluster_df, cluster_id, max_deliveries, depth)

    try:
        route_points, dist = solve_tsp(
            [restaurant_loc] + delivery_points, max_iterations=100,
        )
        optimized_order = [delivery_points.index(p) for p in route_points[1:]]
        return [(cluster_id, route_points, dist, cluster_df.iloc[optimized_order])]
    except Exception as e:
        logger.warning(
            "TSP failed for cluster %s: %s — using unoptimized order. "
            "Route quality is DEGRADED.", cluster_id, e,
        )
        unopt = [restaurant_loc] + delivery_points
        return [(cluster_id, unopt, calculate_route_distance(unopt), cluster_df)]


def _split_and_solve(cluster_df, cluster_id, max_deliveries, depth):
    """Split a large cluster into sub-clusters via KMeans and recurse."""
    delivery_points = np.array(list(zip(
        cluster_df["Delivery_location_latitude"],
        cluster_df["Delivery_location_longitude"],
    )))
    n_sub = max(2, min(int(np.ceil(len(delivery_points) / max_deliveries)), 10))

    try:
        kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(delivery_points)
        
        # Prevent wasted recursion if points are duplicates/near-duplicates and can't be split
        unique_labels, counts = np.unique(sub_labels, return_counts=True)
        if len(unique_labels) < n_sub or max(counts) >= len(delivery_points) - 2:
            logger.warning(
                "Cluster %s: KMeans could not split further (duplicate/near-duplicate coordinates) "
                "— falling back after 1 attempt instead of 5.", cluster_id
            )
            return _handle_as_single_deliveries(cluster_df, cluster_id)
            
    except Exception as e:
        logger.warning(
            "Sub-clustering failed for cluster %s: %s — treating as single deliveries. "
            "Route quality is DEGRADED.", cluster_id, e,
        )
        return _handle_as_single_deliveries(cluster_df, cluster_id)

    results = []
    for i in range(n_sub):
        mask = sub_labels == i
        if mask.sum() == 0:
            continue
        sub_df = cluster_df.iloc[mask]
        results.extend(process_cluster(sub_df, f"{cluster_id}_{i}", max_deliveries, depth + 1, max_depth=5))
    return results


def _handle_as_single_deliveries(cluster_df, cluster_id):
    """Fallback: treat each delivery as its own route."""
    results = []
    restaurant_loc = (
        cluster_df["Restaurant_latitude"].iloc[0],
        cluster_df["Restaurant_longitude"].iloc[0],
    )
    for idx, row in cluster_df.iterrows():
        d = (row["Delivery_location_latitude"], row["Delivery_location_longitude"])
        route = [restaurant_loc, d]
        dist = haversine_distance(restaurant_loc, d)
        results.append((f"{cluster_id}_s{idx}", route, dist, cluster_df.loc[[idx]]))

    logger.info("Cluster %s → %d single-delivery routes.", cluster_id, len(results))
    return results


# ---------------------------------------------------------------------------
# Top-level route optimization
# ---------------------------------------------------------------------------
def optimize_all_clusters(clustered_df, max_deliveries=15, sample_clusters=5):
    """Optimise routes for every cluster and print summary."""
    required = [
        "delivery_cluster", "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
    ]
    missing = [c for c in required if c not in clustered_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_routes = []
    groups = clustered_df.groupby("delivery_cluster")

    logger.info("Optimizing %d clusters...", len(groups))
    for cid, group in tqdm(groups, total=len(groups), desc="TSP per cluster"):
        if cid == -1:
            continue
        all_routes.extend(process_cluster(group, cid, max_deliveries, depth=0))

    if not all_routes:
        logger.error("No valid routes were created.")
        return []

    total_dist = sum(r[2] for r in all_routes)
    avg_dist = total_dist / len(all_routes)
    avg_del = np.mean([len(r[3]) for r in all_routes])

    print("\n" + "=" * 60)
    print("  Optimization Summary")
    print("=" * 60)
    print(f"  Total routes created:      {len(all_routes)}")
    print(f"  Total optimized distance:  {total_dist:.2f} km")
    print(f"  Average distance/route:    {avg_dist:.2f} km")
    print(f"  Average deliveries/route:  {avg_del:.1f}")
    print("=" * 60 + "\n")

    return all_routes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Delivery route optimization using Time-based / KMeans++ Clustering + TSP.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the delivery dataset CSV file.",
    )
    parser.add_argument(
        "--method", default="auto",
        choices=["auto", "kmeans++", "time_based", "distance_optimized"],
        help="Clustering method (default: auto — tries all and picks best).",
    )
    parser.add_argument(
        "--max-deliveries", type=int, default=15,
        help="Max deliveries per route before splitting (default: 15).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max rows for clustering (default: use all data).",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save route plots to files instead of displaying.",
    )
    args = parser.parse_args()

    # Load & clean
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading data")
    logger.info("=" * 60)
    df = load_data(args.input)

    logger.info("=" * 60)
    logger.info("STEP 2 — Cleaning & feature engineering")
    logger.info("=" * 60)
    df = clean_data(df)

    # Cluster
    logger.info("=" * 60)
    logger.info("STEP 3 — Clustering (%s)", args.method)
    logger.info("=" * 60)
    model, clustered_df = optimize_delivery_clusters(
        df, method=args.method, max_samples=args.max_samples,
    )

    # Show person-cluster mapping sample
    if "assigned_person" in clustered_df.columns:
        mapping_df = (
            clustered_df[["delivery_cluster", "assigned_person"]]
            .drop_duplicates()
            .sort_values("delivery_cluster")
        )
        logger.info("Sample person–cluster mapping (first 10):")
        print(mapping_df.head(10).to_string(index=False))

    # Route optimization
    logger.info("=" * 60)
    logger.info("STEP 4 — Route optimization (TSP)")
    logger.info("=" * 60)
    all_routes = optimize_all_clusters(
        clustered_df, max_deliveries=args.max_deliveries, sample_clusters=3,
    )

    # Visualize first 3
    if all_routes:
        for cid, route_points, dist, route_df in all_routes[:3]:
            save_path = f"route_kmeans_cluster_{cid}.png" if args.save_plots else None
            plot_optimized_route(route_df, route_points, dist, cid, save_path=save_path)

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
