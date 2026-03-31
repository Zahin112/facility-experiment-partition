# Facility Assignment Problem Experiment with Partition Algorithm and Maximal Input Sequence

# 📊 Facility Assignment Analysis & Dataset Generation

This repository studies **facility assignment algorithms** under different geometric settings and analyzes their performance through **datasets, partitions, and worst-case constructions**.

---

## 🚀 Project Overview

This project consists of two main components:

### 1. Dataset Generation (`dataset_generate_final.py`)

* Generates synthetic datasets in:

  * Line (1D)
  * Plane (2D)
  * Graph (network distances)
* Computes assignments using:

  * Greedy
  * Optimal (Hungarian Algorithm)
  * Optimal-Fill
* Constructs:

  * Non-trivial instances
  * Partitions
  * Maximal (worst-case) inputs

---

## 📁 Project Structure

```
.
├── dataset_generate_final.py       # Dataset generation + algorithms
├── graph_all.py                    # Analysis + plotting
├── facility_datasets/              # Generated datasets
├── vcapacityfig/                   # Capacity plots algorithms in separate figure 
├── vfacilityfig/                   # Facility plots algorithms in separate figure
├── combinedfig/                    # Combined comparison plots algorithms in same figure
└── README.md
```
### 2. Analysis & Visualization (`graph_all.py`)

* Reads generated datasets
* Computes **average performance metrics**
* Produces plots comparing:

  * Greedy vs Optimal-Fill
  * Final vs Maximal vs Partition ratios

---

## 🧠 Algorithms Implemented

### 🔹 Greedy Assignment

* Assign each customer to the nearest available facility
* Fast but not optimal

### 🔹 Optimal Assignment

* Uses the Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`)
* Produces globally optimal assignment under capacity constraints

### 🔹 Optimal-Fill Assignment

* Incremental algorithm:
* For each prefix of customers, recompute optimal assignment
* Captures online-like behavior

---

## 🌍 Dataset Types

### 1. Line (1D) - Points lie on a line

### 2. Plane (2D) - Euclidean distance in 2D space

### 3. Graph

* Random graph generated using `networkx`
* Edge weights are random
* Distances computed using shortest paths (Floyd–Warshall)

---

## ⚙️ Pipeline


## 📦 Dependencies

Install required libraries:

```bash
pip install numpy requirements.py
```

### Step 1: Generate Dataset

```bash
python dataset_generate_final.py <mode>
```

### Modes:

* `f` → vary number of facilities
* `c` → vary capacities
* `t` → small test mode

📁 Output:

```
facility_datasets/
  facility_<facilities>_<capacity>_<space>.json
```

---

### Step 2: Analyze & Plot

```bash
python graph_all.py
```

This will:

* Parse all dataset files
* Compute average ratios
* Generate plots

---

## 📈 Metrics Computed

For each dataset:

### Assignment Ratios

* `final_ratio` → Greedy vs Optimal
* `final_of_ratio` → Optimal-Fill vs Optimal

### Maximal Ratios

* Worst-case constructed inputs

### Partition Ratios

* Maximum ratio within partitions

---

## 📊 Generated Plots

Saved into:

```
vcapacityfig/
vfacilityfig/
combinedfig/
```

### Plot Types

#### 1. Ratio vs Capacity

* Fix number of facilities
* Vary capacity

#### 2. Ratio vs Facility Count

* Fix capacity
* Vary facilities

#### 3. Combined Plots

* Compare:

  * Greedy vs Optimal-Fill in one figure
  * Final vs Maximal vs maximum Partition ratio 

---


