#!/usr/bin/env python3
"""
Route Optimizer — Agglomerative Clustering + TSP

Clusters delivery locations using Agglomerative Clustering, then optimizes
each cluster's delivery route with a nearest-neighbour + 2-opt TSP heuristic.

For datasets larger than ``--sample-size`` rows (default 8 000), the data is
sampled so that Agglomerative Clustering can run in reasonable time/memory.
Remaining rows are assigned to the nearest cluster centroid.

Usage
-----
    python route_optimizer_agglomerative.py --input zomato_dataset.csv
    python route_optimizer_agglomerative.py --input zomato_dataset.csv --max-clusters 25 --sample-size 5000
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
import psutil
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from tsp_utils import (
    calculate_route_distance,
    haversine_distance,
    plot_optimized_route,
    solve_tsp,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------
def check_memory(threshold: float = 85.0) -> bool:
    """Log current memory usage; return False if above *threshold* %."""
    mem = psutil.virtual_memory()
    logger.info(
        "Memory usage: %.2f GB / %.2f GB (%.1f%%)",
        mem.used / 1e9, mem.total / 1e9, mem.percent,
    )
    if mem.percent > threshold:
        logger.warning("Memory usage above %.0f%% — running garbage collection.", threshold)
        gc.collect()
        return False
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
REQUIRED_COLS = [
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_person_ID",
    "Time_Orderd",
    "Order_Date",
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load delivery data from *csv_path*, reading only required columns."""
    if not os.path.isfile(csv_path):
        logger.error("File not found: %s", csv_path)
        sys.exit(1)

    # Determine which of the required columns actually exist in the file
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols_to_load = [c for c in REQUIRED_COLS if c in header_cols]

    # Also load columns needed for features if available
    for extra in [
        "Weather_conditions", "Road_traffic_density", "Vehicle_condition",
        "Delivery_person_Age", "Delivery_person_Ratings",
        "multiple_deliveries", "Festival", "City", "Time_taken (min)",
    ]:
        if extra in header_cols:
            cols_to_load.append(extra)

    df = pd.read_csv(csv_path, usecols=cols_to_load)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and create time features."""
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

    # Ensure time features exist even if Time_Orderd was missing
    if "time_sin" not in df.columns:
        logger.warning("Time features missing — generating random time features (results less meaningful).")
        df["Order_Time_min"] = np.random.uniform(0, 1440, len(df))
        df["time_sin"] = np.sin(2 * np.pi * df["Order_Time_min"] / 1440)
        df["time_cos"] = np.cos(2 * np.pi * df["Order_Time_min"] / 1440)

    if "Delivery_person_ID" not in df.columns:
        logger.warning("Delivery_person_ID column missing — generating synthetic IDs.")
        df["Delivery_person_ID"] = [f"DP_{i}" for i in range(len(df))]

    logger.info("Data cleaned. %d rows remaining.", len(df))
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def perform_clustering(
    df: pd.DataFrame,
    max_clusters: int = 30,
    sample_size: int = 8000,
) -> tuple:
    """
    Perform Agglomerative Clustering.

    For datasets larger than *sample_size*, we sample down so that
    ``AgglomerativeClustering`` runs in feasible time/memory, then propagate
    labels to remaining rows by nearest-centroid assignment.

    Returns
    -------
    df : DataFrame
        With a new ``delivery_cluster`` column.
    """
    features = ["Delivery_location_latitude", "Delivery_location_longitude"]
    if "time_sin" in df.columns and "time_cos" in df.columns:
        features.extend(["time_sin", "time_cos"])

    sampled = False
    sample_idx = None

    if len(df) > sample_size:
        logger.info(
            "Dataset has %d rows — sampling %d rows for Agglomerative Clustering.",
            len(df), sample_size,
        )
        sample_idx = df.sample(sample_size, random_state=42).index
        X_sample = df.loc[sample_idx, features].astype(np.float32).values
        sampled = True
    else:
        X_sample = df[features].astype(np.float32).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    # Always use Agglomerative Clustering (the purpose of this script)
    logger.info(
        "Running AgglomerativeClustering (n_clusters=%d) on %d rows.",
        max_clusters, len(X_scaled),
    )
    model = AgglomerativeClustering(
        n_clusters=max_clusters,
        metric="euclidean",
        linkage="ward",
    )

    try:
        clusters = model.fit_predict(X_scaled)
    except Exception as e:
        logger.error("AgglomerativeClustering failed: %s", e)
        logger.warning(
            "FALLBACK: assigning random cluster labels. Output quality is DEGRADED."
        )
        df["delivery_cluster"] = np.random.randint(0, max_clusters, len(df))
        return df

    if sampled:
        # Assign sampled rows their cluster labels
        df["delivery_cluster"] = -1
        df.loc[sample_idx, "delivery_cluster"] = clusters

        # Propagate to remaining rows via nearest centroid
        logger.info("Propagating cluster labels to remaining %d rows via nearest centroid.",
                     len(df) - sample_size)

        # Compute centroids from the sample
        sample_df = df.loc[sample_idx].copy()
        centroids = sample_df.groupby("delivery_cluster")[features].mean().values
        centroid_labels = sample_df.groupby("delivery_cluster")[features].mean().index.values

        # Scale remaining rows
        remaining_idx = df.index.difference(sample_idx)
        X_remaining = scaler.transform(df.loc[remaining_idx, features].astype(np.float32).values)

        # Scale centroids
        centroids_scaled = scaler.transform(centroids.astype(np.float32))

        # Assign each remaining row to nearest centroid
        from scipy.spatial.distance import cdist
        dists = cdist(X_remaining, centroids_scaled, metric="euclidean")
        nearest = np.argmin(dists, axis=1)
        df.loc[remaining_idx, "delivery_cluster"] = centroid_labels[nearest]

        logger.info("All %d rows now have cluster labels.", len(df))
    else:
        df["delivery_cluster"] = clusters

    # Log cluster distribution
    cluster_counts = df["delivery_cluster"].value_counts()
    logger.info(
        "Cluster sizes — min: %d, max: %d, mean: %.1f",
        cluster_counts.min(), cluster_counts.max(), cluster_counts.mean(),
    )

    del X_scaled
    gc.collect()
    return df


# ---------------------------------------------------------------------------
# Route optimization
# ---------------------------------------------------------------------------
def optimize_routes(
    clustered_df: pd.DataFrame,
    max_clusters_to_process: int = 30,
    max_points_per_route: int = 20,
) -> list:
    """
    For each cluster, extract delivery points and solve TSP.

    Returns a list of dicts with keys:
        cluster_id, route, distance, deliveries
    """
    all_routes = []
    cluster_groups = list(clustered_df.groupby("delivery_cluster"))
    processed = 0

    for cluster_id, group in cluster_groups:
        if cluster_id == -1:
            continue
        if processed >= max_clusters_to_process:
            logger.warning(
                "Stopping after %d clusters (max_clusters_to_process limit).", processed
            )
            break
        if not check_memory():
            logger.warning("Stopping route optimization due to high memory usage.")
            break

        try:
            restaurant_loc = (
                group["Restaurant_latitude"].iloc[0],
                group["Restaurant_longitude"].iloc[0],
            )
            delivery_points = list(zip(
                group["Delivery_location_latitude"],
                group["Delivery_location_longitude"],
            ))

            if len(delivery_points) > max_points_per_route:
                logger.warning(
                    "Cluster %s has %d deliveries — truncating to %d. "
                    "Route quality may be degraded.",
                    cluster_id, len(delivery_points), max_points_per_route,
                )
                delivery_points = delivery_points[:max_points_per_route]

            route_points, distance = solve_tsp(
                [restaurant_loc] + delivery_points, max_iterations=100,
            )

            all_routes.append({
                "cluster_id": cluster_id,
                "route": route_points,
                "distance": distance,
                "deliveries": len(delivery_points),
            })
            processed += 1

        except Exception as e:
            logger.error("Skipping cluster %s due to error: %s", cluster_id, e)
            continue

    return all_routes





# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Delivery route optimization using Agglomerative Clustering + TSP.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the delivery dataset CSV file.",
    )
    parser.add_argument(
        "--max-clusters", type=int, default=30,
        help="Number of clusters to create (default: 30).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=8000,
        help="Max rows to feed to AgglomerativeClustering (default: 8000).",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save route plots to files instead of displaying.",
    )
    args = parser.parse_args()

    check_memory()

    # Load & clean
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading data")
    logger.info("=" * 60)
    df = load_data(args.input)

    logger.info("=" * 60)
    logger.info("STEP 2 — Cleaning data")
    logger.info("=" * 60)
    df = clean_data(df)

    # Cluster
    logger.info("=" * 60)
    logger.info("STEP 3 — Agglomerative Clustering")
    logger.info("=" * 60)
    clustered_df = perform_clustering(
        df,
        max_clusters=args.max_clusters,
        sample_size=args.sample_size,
    )

    # Optimize routes
    logger.info("=" * 60)
    logger.info("STEP 4 — Route Optimization (TSP)")
    logger.info("=" * 60)
    all_routes = optimize_routes(clustered_df)

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    if not all_routes:
        logger.error("No routes were created.")
        sys.exit(1)

    total_distance = sum(r["distance"] for r in all_routes)
    avg_distance = total_distance / len(all_routes)
    avg_deliveries = np.mean([r["deliveries"] for r in all_routes])

    # Calculate total routed deliveries
    total_routed_deliveries = int(sum(r["deliveries"] for r in all_routes))
    total_dataset_deliveries = len(df)
    routed_percentage = (total_routed_deliveries / total_dataset_deliveries) * 100

    print("\n" + "=" * 60)
    print("  Optimization Summary")
    print("=" * 60)
    print(f"  Algorithm used:            Agglomerative Clustering (Ward linkage)")
    print(f"  Total routes created:      {len(all_routes)}")
    print(f"  Total optimized distance:  {total_distance:.2f} km")
    print(f"  Average distance/route:    {avg_distance:.2f} km")
    print(f"  Average deliveries/route:  {avg_deliveries:.1f}")
    print(f"  Deliveries routed:         {total_routed_deliveries} / {total_dataset_deliveries} ({routed_percentage:.1f}%) — see truncation warnings above for clusters that were cut down")

    # Naive baseline: sum of individual restaurant→delivery distances
    # for only the points each route actually covers (apples-to-apples)
    naive_total = 0.0
    for r in all_routes:
        restaurant = r["route"][0]
        for delivery in r["route"][1:]:
            naive_total += haversine_distance(restaurant, delivery)
    if naive_total > 0:
        reduction = (1 - total_distance / naive_total) * 100
        print(f"  Naive baseline (individual trips): {naive_total:.2f} km")
        print(f"  Distance reduction vs naive:       {reduction:.1f}%")

    print("=" * 60 + "\n")

    # Visualize first 3 routes
    for i, route in enumerate(all_routes[:3]):
        save_path = f"route_agg_cluster_{route['cluster_id']}.png" if args.save_plots else None
        # Build a mini-df for the plot helper
        mini_df = pd.DataFrame({
            "Restaurant_latitude": [route["route"][0][0]],
            "Restaurant_longitude": [route["route"][0][1]],
        })
        plot_optimized_route(
            mini_df, route["route"], route["distance"],
            route["cluster_id"], save_path=save_path,
        )

    check_memory()
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
