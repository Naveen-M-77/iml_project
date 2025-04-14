
# 🚚 Delivery Route Optimization using Clustering and TSP

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

### 🔹 `route_optimizer_agglomerative.py`
- **Method:** Agglomerative (hierarchical) clustering based on latitude/longitude.
- **TSP:** Nearest Neighbor + 2-opt optimization.
- **Output:** Plots optimized routes for each cluster.
- **Best for:** Pure spatial grouping.

### 🔹 `route_optimizer_time_kmeans.py`
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
git clone https://github.com/your-username/delivery-route-optimizer.git
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
python route_optimizer_agglomerative.py --input data/delivery_locations.csv --clusters 5
```

| Option        | Description                            |
|---------------|----------------------------------------|
| `--input`     | Path to your delivery dataset (CSV)    |
| `--clusters`  | Number of clusters to form             |

---

### ✅ Run Time-based / KMeans++ Route Optimization

```bash
python route_optimizer_time_kmeans.py --input data/delivery_data.csv --mode time --time_window 60
```

| Option          | Description                                              |
|------------------|----------------------------------------------------------|
| `--input`        | CSV file with columns: `latitude`, `longitude`, `time` |
| `--mode`         | `time` or `kmeans`                                       |
| `--time_window`  | Time bin size in minutes (only for `time` mode)         |
| `--clusters`     | Number of KMeans clusters (only for `kmeans` mode)      |

Example for KMeans++ mode:
```bash
python route_optimizer_time_kmeans.py --input data/delivery_data.csv --mode kmeans --clusters 4
```

---

## 📊 Outputs

- Cluster-wise optimized delivery routes.
- Total and average distance traveled per cluster.
- Matplotlib plots saved in `outputs/` folder.

---

## 📌 Sample Dataset Format

### For `route_optimizer_agglomerative.py`
```csv
latitude,longitude
12.9716,77.5946
13.0827,80.2707
...
```

### For `route_optimizer_time_kmeans.py`
```csv
latitude,longitude,time
12.9716,77.5946,09:15
13.0827,80.2707,10:45
...
```

---

## 🏁 Results

- Efficient route planning with reduced travel distance.
- Comparison between clustering techniques.
- Visual feedback through plotted delivery paths.

---

## 🤝 Contributing

Feel free to fork and improve this repo. Pull requests are welcome!

---

## 📄 License

This project is open-source and available under the MIT License.
