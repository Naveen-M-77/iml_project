
# Cluster based Route Optimization System

This project implements intelligent delivery route optimization using **clustering** techniques and **TSP (Traveling Salesman Problem)** heuristics. It includes two approaches: one using **Agglomerative Clustering**, and another that dynamically selects between **time-based clustering** and **KMeans++**.

---

## Repository Structure

```
├── route_optimizer_agglomerative.py      # Agglomerative Clustering + TSP
├── route_optimizer_time_kmeans.py        # Time-based OR KMeans++ Clustering + TSP
├── requirements.txt                      # Python dependencies
└── README.md                             
```

---

## Project Files Overview

---

### `agglomerative_clustering.py`
**Approach:** Agglomerative Clustering + TSP Optimization  
1. Data Loading & Preprocessing  
2. Agglomerative Clustering  
3. Route Optimization (Nearest Neighbor + 2-opt)  
4. Distance Evaluation (Haversine)  
5. Visualization of Routes  

---

### `time_based_k_means_pp.py`
**Approach:** Time-based Binning / KMeans++ + TSP Optimization  
1. Data Loading & Preprocessing  
2. Clustering (Time Binning or KMeans++)  
3. Route Optimization (Nearest Neighbor + 2-opt)  
4. Distance Evaluation & Comparison  
5. Route Visualization  

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
python agglomerative_clustering.py --input zomato_dataset.csv
```

| Option        | Description                            |
|---------------|----------------------------------------|
| `--input`     | Path to your delivery dataset (CSV)    |

---

### Run Time-based / KMeans++ Route Optimization

```bash
python route_optimizer_time_kmeans.py
```
---

## Outputs

- Cluster-wise optimized delivery routes.
- Total and average distance traveled per cluster.

---

---

## Results

- Efficient route planning with reduced travel distance.
- Comparison between clustering techniques.
- Visual feedback through plotted delivery paths.

---
