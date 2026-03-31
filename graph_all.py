import json, glob, re, numpy as np, sys
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Parse a single file into averaged metrics
# ------------------------------------------------------------
def process_single_file(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)

    final_ratios = []
    maximal_ratios = []
    partition_max_ratios = []
    of_final_ratios = []
    of_maximal_ratios = []
    of_partition_max_ratios = []
    
    for case in data:
        costs = case["costs"]
        
        final_ratios.append(costs["final_ratio"])
        of_final_ratios.append(costs["final_of_ratio"])
        maximal_ratios.append(costs["maximal_cost_ratio"])
        of_maximal_ratios.append(costs["of_maximal_cost_ratio"])
        
        # if costs["maximal_cost_ratio"] < 30:
        #     maximal_ratios.append(costs["maximal_cost_ratio"])

        # if costs["of_maximal_cost_ratio"] < 30:
        #     of_maximal_ratios.append(costs["of_maximal_cost_ratio"])
        
        # partition max ratio
        partitions_costs = case.get("partition costs", [])
        if partitions_costs:
            ratios = [ p["cost_ratio"] for p in partitions_costs ]
            if ratios:
                partition_max_ratios.append(max(ratios))
        of_partitions_costs = case.get("of_partition costs", [])
        if of_partitions_costs:
            ratios = [ p["cost_ratio"] for p in of_partitions_costs ]
            if ratios:
                of_partition_max_ratios.append(max(ratios))
    # print(partition_max_ratios)
    # print(of_partition_max_ratios)
    return {
        "avg_final_ratio": np.mean(final_ratios),
        "avg_maximal_ratio": np.mean(maximal_ratios),
        "avg_partition_ratio": np.mean(partition_max_ratios),
        "avg_of_final_ratio": np.mean(of_final_ratios),
        "avg_of_maximal_ratio": np.mean(of_maximal_ratios),
        "avg_of_partition_ratio": np.mean(of_partition_max_ratios),
    }

# ------------------------------------------------------------
# Plot ratios vs Capacity for each facility + space
# ------------------------------------------------------------
def plot_vs_capacity(results, space, graph):
    # collect all capacities for this (facility, space)
    fac = 30
    subset = {
        c: results[(fac, c, space)]
        for (f, c, s) in results
        if f == fac and s == space
    }
    # print(subset)
    if not subset:
        print(f"No data found for facility, space={space}")
        return

    capacities = sorted(subset.keys())
    plt.figure(figsize=(10, 6))
    plt.plot(capacities, [subset[c]["avg_final_ratio"] for c in capacities], marker='o', label="Original Ratio")
    if graph == "max":
        plt.plot(capacities, [subset[c]["avg_maximal_ratio"] for c in capacities], marker='s', label="Maximal Ratio")
    else: plt.plot(capacities, [subset[c]["avg_partition_ratio"] for c in capacities], marker='^', label="Maximum Partition Ratio")

    plt.xlabel("Capacity")
    plt.ylabel("Ratio")
    plt.title(f"Ratios vs Capacity ({space}) for Greedy Algorithm.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vcapacityfig/vcapacity_{space}_Greedy_{graph}.png", dpi=300)
    # plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(capacities, [subset[c]["avg_of_final_ratio"] for c in capacities], marker='o', label="Original Ratio")
    if graph == "max":
        plt.plot(capacities, [subset[c]["avg_of_maximal_ratio"] for c in capacities], marker='s', label="Maximal Ratio")
    else: plt.plot(capacities, [subset[c]["avg_of_partition_ratio"] for c in capacities], marker='^', label="Maximum Partition Ratio")

    plt.xlabel("Capacity")
    plt.ylabel("Ratio")
    plt.title(f"Ratios vs Capacity ({space}) for Optimal-Fill Algorithm.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vcapacityfig/vcapacity_{space}_optfill_{graph}.png", dpi=300)
    # plt.show()
    plt.close()

# ------------------------------------------------------------
# Plot ratios vs Facility Count for each capacity + space
# ------------------------------------------------------------
def plot_vs_facility(results, space, graph):
    # collect all facility sizes for this (capacity, space)
    capacity = 5
    subset = {
        f: results[(f, capacity, space)]
        for (f, c, s) in results
        if c == capacity and s == space
    }

    if not subset:
        print(f"No data found for capacity={capacity}, space={space}")
        return

    facilities = sorted(subset.keys())

    plt.figure(figsize=(10, 6))
    plt.plot(facilities, [subset[f]["avg_final_ratio"] for f in facilities], marker='o', label="Final Ratio")
    if graph == "max": plt.plot(facilities, [subset[f]["avg_maximal_ratio"] for f in facilities], marker='s', label="Maximal Ratio")
    else: plt.plot(facilities, [subset[f]["avg_partition_ratio"] for f in facilities], marker='^', label="Partition Max Ratio")

    plt.xlabel("Facility Count")
    plt.ylabel("Ratio")
    plt.title(f"Capacity {capacity} Ratios vs Facility Count ({space}) for Greedy Algorithm.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vfacilityfig/vfacility_{space}_Greedy_{graph}.png", dpi=300)
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(facilities, [subset[f]["avg_of_final_ratio"] for f in facilities], marker='o', label="Final Ratio")
    if graph == "max": plt.plot(facilities, [subset[f]["avg_of_maximal_ratio"] for f in facilities], marker='s', label="Maximal Ratio")
    else: plt.plot(facilities, [subset[f]["avg_of_partition_ratio"] for f in facilities], marker='^', label="Partition Max Ratio")

    plt.xlabel("Facility Count")
    plt.ylabel("Ratio")
    plt.title(f"Capacity {capacity} Ratios vs Facility Count ({space}) for Optimal-Fill Algorithm.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vfacilityfig/vfacility_{space}_optfill_{graph}.png", dpi=300)
    # plt.show()
    plt.close()

# ------------------------------------------------------------
# Plot ratios vs Facility Count for each capacity + space
# ------------------------------------------------------------
def combined_plots(results, space, graph):
    # collect all facility sizes for this (capacity, space)
    capacity = 5
    subset = {
        f: results[(f, capacity, space)]
        for (f, c, s) in results
        if c == capacity and s == space
    }

    if not subset:
        print(f"No data found for capacity={capacity}, space={space}")
        return
    # print(space,graph)
    # print(subset)
    facilities = sorted(subset.keys())
    plt.figure(figsize=(10, 6))
    plt.plot(facilities, [subset[f]["avg_final_ratio"] for f in facilities], marker='o', label="Final Greedy Ratio")
    plt.plot(facilities, [subset[f]["avg_of_final_ratio"] for f in facilities], marker='x', label="Final Optimal-Fill Ratio")

    if graph == "max": 
        plt.plot(facilities, [subset[f]["avg_maximal_ratio"] for f in facilities], marker='s', label="Maximal Greedy Ratio")
        plt.plot(facilities, [subset[f]["avg_of_maximal_ratio"] for f in facilities], marker='D', label="Maximal Optimal-Fill Ratio")
    else: 
        plt.plot(facilities, [subset[f]["avg_partition_ratio"] for f in facilities], marker='P', label="Partition Max Ratio For Greedy")
        plt.plot(facilities, [subset[f]["avg_of_partition_ratio"] for f in facilities], marker='^', label="Partition Max Ratio for Optimal-Fill")

    plt.xlabel("Facility Count")
    plt.ylabel("Ratio")
    plt.title(f"Combined Capacity {capacity} Ratios vs Facility Count ({space}).")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"combinedfig/vfacility_{space}_combined_{graph}.png", dpi=300)
    # plt.show()
    plt.close()

    fac = 30
    subset = {
        c: results[(fac, c, space)]
        for (f, c, s) in results
        if f == fac and s == space
    }
    # print(subset)
    if not subset:
        print(f"No data found for facility, space={space}")
        return
    # print(space,graph)
    # print(subset)
    capacities = sorted(subset.keys())
    plt.figure(figsize=(10, 6))
    plt.plot(capacities, [subset[c]["avg_final_ratio"] for c in capacities], marker='o', label="Original Greedy Ratio")
    plt.plot(capacities, [subset[c]["avg_of_final_ratio"] for c in capacities], marker='x', label="Original Optimal-Fill Ratio")
    if graph == "max":
        plt.plot(capacities, [subset[c]["avg_maximal_ratio"] for c in capacities], marker='s', label="Maximal Greedy Ratio")
        plt.plot(capacities, [subset[c]["avg_of_maximal_ratio"] for c in capacities], marker='D', label="Maximal Optimal-Fill Ratio")
    else: 
        plt.plot(capacities, [subset[c]["avg_partition_ratio"] for c in capacities], marker='^', label="Maximum Partition Ratio Greedy")
        plt.plot(capacities, [subset[c]["avg_of_partition_ratio"] for c in capacities], marker='P', label="Maximum Partition Ratio Optimal-Fill")

    plt.xlabel("Capacity")
    plt.ylabel("Ratio")
    plt.title(f"Ratios vs Capacity ({space}) for Greedy Algorithm.")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"combinedfig/vcapacity_{space}_combined_{graph}.png", dpi=300)
    # plt.show()
    plt.close()
    
# results = parse_all_facility_files("data")   # folder containing all json files
pattern = f"facility_datasets/facility_*_*_*.json"
files = glob.glob(pattern)
# Dictionary indexed by (f, c, space)
results = {}

regex = re.compile(r"facility_(\d+)_(\d+)_(\w+)\.json")
for file in files:
    m = regex.search(file)
    if not m:
        continue

    f = int(m.group(1))
    c = int(m.group(2))
    space = m.group(3)
    print(file)
    metrics = process_single_file(file)
    results[(f, c, space)] = metrics
# print(results)
# print(results.keys)

#----------- Plot Generate-----------------------
# plot_vs_capacity(results, space="line", graph ="max")
# plot_vs_capacity(results, space="plane", graph ="max")
# plot_vs_capacity(results, space="line", graph ="part")
# plot_vs_capacity(results, space="plane", graph ="part")

# plot_vs_facility(results, space="line", graph ="max")
# plot_vs_facility(results, space="plane", graph ="max")
# plot_vs_facility(results, space="line", graph ="part")
# plot_vs_facility(results, space="plane", graph ="part")

#----------- Combined Plot Generate-----------------------
combined_plots(results, space="line", graph ="max")
combined_plots(results, space="plane", graph ="max")
combined_plots(results, space="graph", graph ="max")
combined_plots(results, space="line", graph ="part")
combined_plots(results, space="plane", graph ="part")
combined_plots(results, space="graph", graph ="part")


#------------------------ Example ------------------------ 
# plot_vs_capacity(results, space="line", algo="Optimal-Fill", graph ="max")
# plot_vs_capacity(results, space="plane", algo="Optimal-Fill", graph ="max")
# plot_vs_capacity(results, space="line", algo="Optimal-Fill", graph ="part")
# plot_vs_capacity(results, space="plane", algo="Optimal-Fill", graph ="part")

# plot_vs_facility(results, space="line", algo="Optimal-Fill", graph ="max")
# plot_vs_facility(results, space="plane", algo="Optimal-Fill", graph ="max")
# plot_vs_facility(results, space="line", algo="Optimal-Fill", graph ="part")
# plot_vs_facility(results, space="plane", algo="Optimal-Fill", graph ="part")

# plot_vs_capacity(results, space="line", algo=algo)
# plot_vs_capacity(results, space="plane", algo=algo)

# plot_vs_facility(results, space="line", algo=algo)
# plot_vs_facility(results, space="plane", algo=algo)