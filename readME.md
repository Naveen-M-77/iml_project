# Cluster based Route Optimization System

This project implements intelligent delivery route optimization using **clustering** techniques and **TSP (Traveling Salesman Problem)** heuristics. It includes two approaches: one using **Agglomerative Clustering**, and another that dynamically selects between **time-based clustering** and **KMeans++**.

---

## Repository Structure

```
├── route_optimizer_agglomerative.py      # Agglomerative Clustering + TSP
├── route_optimizer_time_kmeans.py        # Time-based OR KMeans++ Clustering + TSP
├── tsp_utils.py                          # Shared utilities (TSP solver, distance computation)
├── requirements.txt                      # Python dependencies
└── readME.md                             # Project documentation
```

---

## Project Files Overview

---

### `route_optimizer_agglomerative.py`
**Approach:** Agglomerative Clustering + TSP Optimization  
1. Data Loading & Preprocessing  
2. Agglomerative Clustering (with sample-based fallback for large datasets)  
3. Route Optimization (Nearest Neighbor + 2-opt)  
4. Distance Evaluation (Haversine distance)  
5. Visualization of Routes  

---

### `route_optimizer_time_kmeans.py`
**Approach:** Time-based Binning / KMeans++ + TSP Optimization  
1. Data Loading & Preprocessing  
2. Clustering (Time Binning or KMeans++) with evaluation (Silhouette Score)  
3. Assignment of clusters to delivery persons using the Hungarian Algorithm  
4. Route Optimization (Nearest Neighbor + 2-opt)  
5. Distance Evaluation & Comparison  
6. Route Visualization  

---

### `tsp_utils.py`
**Approach:** Shared Logic for Route Optimization  
1. Haversine distance calculations  
2. Fast TSP heuristic solver (Nearest-Neighbor initialization + 2-opt iterative improvement)  
3. Route plotting utility using matplotlib  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Naveen-M-77/iml_project.git
cd delivery-route-optimizer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## How to Run the Scripts

### Run Agglomerative Clustering-based Route Optimization

```bash
python route_optimizer_agglomerative.py --input zomato_dataset.csv --save-plots
```

| Option         | Description                                     |
|----------------|-------------------------------------------------|
| `--input`      | Path to your delivery dataset (CSV)             |
| `--save-plots` | Save route visualizations as PNG images         |
| `--max-samples`| Maximum rows to sample before clustering        |

> **Note on Performance vs. Completeness:** By default, clustering parameters (like `--max-clusters 30` in the agglomerative approach) and route size limits (truncated to 20 for TSP) trade off completeness against runtime. To optimize routes for a larger portion of the total deliveries in big datasets, you can raise the number of clusters (e.g. `--max-clusters 200`), though this will significantly increase clustering and TSP runtime.

---

### Run Time-based / KMeans++ Route Optimization

```bash
python route_optimizer_time_kmeans.py --input zomato_dataset.csv --save-plots
```

| Option         | Description                                     |
|----------------|-------------------------------------------------|
| `--input`      | Path to your delivery dataset (CSV)             |
| `--save-plots` | Save route visualizations as PNG images         |
| `--max-samples`| Maximum rows to sample before clustering        |
---

## Outputs

- Cluster-wise optimized delivery routes.
- Total and average distance traveled per cluster.
- Comparison metrics vs naive baseline (e.g. 80.4% reduction in travel distance).
- `.png` images of generated routes (if `--save-plots` is used).

---

## Results

- Efficient route planning with reduced travel distance.
- Comparison between clustering techniques based on actual route length.
- Visual feedback through plotted delivery paths for each cluster.
