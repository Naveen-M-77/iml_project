
# 🚚 Cluster based Route Optimization System

This project implements intelligent delivery route optimization using **clustering** techniques and **TSP (Traveling Salesman Problem)** heuristics. It includes two approaches: one using **Agglomerative Clustering**, and another that dynamically selects between **time-based clustering** and **KMeans++**.

---

## 📁 Repository Structure

```
├── route_optimizer_agglomerative.py      # Agglomerative Clustering + TSP
├── route_optimizer_time_kmeans.py        # Time-based OR KMeans++ Clustering + TSP
├── data/                                 # Folder to store input delivery datasets
├── outputs/                              # Stores plots and result files
├── requirements.txt                      # Python dependencies
└── README.md                             # You're here!
```

---

## 🧠 Overview of Python Scripts

### 🔹 `agglomerative_clustering.py`
- **Method:** Agglomerative (hierarchical) clustering based on latitude/longitude.
- **TSP:** Nearest Neighbor + 2-opt optimization.
- **Output:** Plots optimized routes for each cluster.
- **Best for:** Pure spatial grouping.

### 🔹 `time_based_k_means_pp.py`
- **Method:** Either:
  - **Time-based Clustering**: Groups deliveries by time windows.
  - **KMeans++ Clustering**: Groups based on location, with better centroid initialization.
- **TSP:** Same as above.
- **Output:** Plots and evaluates route efficiency based on the chosen clustering method.
- **Best for:** Temporal or hybrid grouping.

---

## 🛠️ Installation

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

## 🚀 How to Run the Scripts

### ✅ Run Agglomerative Clustering-based Route Optimization

```bash
python agglomerative_clustering.py --input zomato_dataset.csv
```

| Option        | Description                            |
|---------------|----------------------------------------|
| `--input`     | Path to your delivery dataset (CSV)    |

---

### ✅ Run Time-based / KMeans++ Route Optimization

```bash
python route_optimizer_time_kmeans.py
```

| Option          | Description                                              |
|------------------|----------------------------------------------------------|
| `--input`        | Path to your delivery dataset (CSV)                     |

---

## 📊 Outputs

- Cluster-wise optimized delivery routes.
- Total and average distance traveled per cluster.

---
`

---

## 🏁 Results

- Efficient route planning with reduced travel distance.
- Comparison between clustering techniques.
- Visual feedback through plotted delivery paths.

---
