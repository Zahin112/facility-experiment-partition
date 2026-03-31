# -*- coding: utf-8 -*-
"""partition, maximal and non trivial

### **Line**
"""

## generate dataset for line
import numpy as np, json, os, sys, time
from scipy.optimize import linear_sum_assignment
import networkx as nx
from scipy.sparse.csgraph import dijkstra, floyd_warshall

generating_space ='line' 
G = None
graph_distance_matrix = None
graph_distance_matrix_gpu = None  # Cache GPU version

def generate_positions(n, grid_size, dim=2):
    """
    Generate n positions. dim=1 -> line of length  dim=2 -> plane 100x100
    """
    # if dim == 1:
    #     return np.random.rand(n, 1) * line_length
    # else:
    #     return np.random.rand(n, 2) * 100
    return np.random.rand(n, dim) * grid_size

def compute_distance(a, b):
    """
    Compute distance matrix depending on dimension and space type.
    a: array of shape (n, dim) - typically facilities
    b: array of shape (m, dim) or (dim,) - typically customers
    Returns: distance matrix of shape (n, m) or (n,) if b is 1D
    """
    global generating_space, graph_distance_matrix
    
    # Ensure b is at least 2D
    if b.ndim == 1:
        b = b.reshape(1, -1)
        single_point = True
    else:
        single_point = False

    if generating_space == 'graph':
        # For graph: positions are node IDs
        # Use precomputed distance matrix for O(1) lookup
        n = a.shape[0]
        m = b.shape[0]
        
        # Extract node IDs
        node_ids_a = a[:, 0].astype(int)
        node_ids_b = b[:, 0].astype(int)
        
        # Vectorized lookup from precomputed matrix
        distances = graph_distance_matrix[np.ix_(node_ids_a, node_ids_b)]
    
    elif a.shape[1] == 1:  # line (1D)
        distances = np.abs(a - b.T)
    
    else:  # plane (2D)
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    # If b was a single point, return 1D array instead of 2D
    if single_point:
        distances = distances.squeeze(axis=1)

    return distances

def midpoint(p1, p2):
    """
    Compute midpoint between two points in any dimension.
    Args: p1, p2: Iterable points of same dimension (list, tuple, np.array)
    Returns: list: midpoint coordinates
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return ((p1 + p2) / 2).tolist()

def generate_graph(num_nodes, density=0.3):
    """
    Generate a random graph with weighted edges.
    Returns the graph.
    """
    global graph_distance_matrix
    
    # Create random graph
    graph = nx.erdos_renyi_graph(num_nodes, density)
    
    # Ensure graph is connected
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            graph.add_edge(node1, node2)
    
    # Add random edge weights
    for (u, v) in graph.edges():
        graph[u][v]['weight'] = np.random.uniform(1, 10)
    
    # Precompute ALL shortest path distances
    print("Precomputing graph distances...")
    start_time = time.time()   # start timer
    # dist_dict = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    
    # # Convert to numpy array for fast lookup
    # graph_distance_matrix = np.zeros((num_nodes, num_nodes))
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         graph_distance_matrix[i][j] = dist_dict[i][j]

    # graph_distance_matrix = nx.floyd_warshall_numpy(graph, weight='weight')
    # Convert to SciPy sparse adjacency matrix
    # A = nx.to_scipy_sparse_array(
    #     graph,
    #     weight="weight",
    #     format="csr",
    #     dtype=np.float32
    # )

    # # All-pairs shortest paths (C-optimized)
    # graph_distance_matrix = dijkstra(
    #     A,
    #     directed=False,
    #     unweighted=False
    # )
    adjacency = nx.to_scipy_sparse_array(graph, weight='weight')

    # Sparse Floyd-Warshall (faster for sparse graphs)
    graph_distance_matrix = floyd_warshall(adjacency, directed=False)
    print(f"Distance matrix precomputed! Time taken: {time.time() - start_time:.3f} seconds")
    
    return graph

def generate_graph_positions(n, num_nodes):
    """Generate n random node IDs from the graph. Returns positions as node IDs."""
    global G
    
    if G is None:
        raise ValueError("Graph not initialized. Call generate_graph first.")
    
    selected_nodes = np.random.choice(num_nodes, size=n, replace=False)
    positions = selected_nodes.reshape(-1, 1)  # Shape: (n, 1)
    
    return positions

def greedy_assignment(customers, facilities, capacities):
    assignments = [-1] * len(customers)
    remaining_capacity = capacities.copy()
    for i, c in enumerate(customers):
        # distances = np.linalg.norm(facilities - c, axis=1)
        distances = compute_distance(facilities, c)  # Now returns 1D array correctly
        sorted_idx = np.argsort(distances)
        for idx in sorted_idx:
            if remaining_capacity[idx] > 0:
                assignments[i] = int(idx)  # Convert numpy int to Python int
                remaining_capacity[idx] -= 1
                break
    return assignments

def hungarian_assignment(customers, facilities, capacities):
    """
    Optimized Hungarian algorithm that runs once on expanded capacity matrix.
    This gives the true globally optimal assignment.

    Args:
        customers: numpy array of customer positions (n_customers, dim)
        facilities: numpy array of facility positions (n_facilities, dim)
        capacities: list of capacity for each facility

    Returns:
        assignments: list of facility indices for each customer
    """
    n_customers = len(customers)
    n_facilities = len(facilities)

    # Create expanded facility list (duplicate facilities based on capacity)
    facility_slots = []
    slot_to_facility = []
    for fac_idx, capacity in enumerate(capacities):
        for _ in range(capacity):
            facility_slots.append(facilities[fac_idx])
            slot_to_facility.append(fac_idx)

    facility_slots = np.array(facility_slots)
    n_slots = len(facility_slots)

    # If more customers than slots, some will remain unassigned
    if n_customers > n_slots:
        # Add dummy slots with very high cost
        dummy_slots = n_customers - n_slots
        dummy_positions = np.ones((dummy_slots, facilities.shape[1])) * 1e10
        facility_slots = np.vstack([facility_slots, dummy_positions])
        slot_to_facility.extend([-1] * dummy_slots)

    # Create cost matrix
    cost_matrix = compute_distance(customers, facility_slots)

    # Run Hungarian algorithm once
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert slot assignments to facility assignments
    assignments = []
    for customer_idx in range(n_customers):
        if customer_idx in row_ind:
            idx = list(row_ind).index(customer_idx)
            slot_idx = col_ind[idx]
            if slot_idx < len(slot_to_facility):
                facility_idx = slot_to_facility[slot_idx]
                assignments.append(int(facility_idx) if facility_idx != -1 else -1)
            else:
                assignments.append(-1)
        else:
            assignments.append(-1)

    return assignments

def optimal_fill_assignment(customers, facilities, capacities):
    """
    Optimal-Fill Algorithm.
    For each new customer c_i, choose the facility f_j that results from the optimal assignment of {c_1, ..., c_i} with available capacities.
    Args: customers: numpy array (n_customers, dim) facilities: numpy array (n_facilities, dim) capacities: list of capacities for each facility
    Returns: assignments: list of facility indices (or -1 if unassigned)
    """

    n_customers = len(customers)
    final_assignment = [-1] * n_customers
    
    # Build expanded facility slots based on capacity
    facility_slots = []
    slot_to_facility = []

    for fac_idx, cap in enumerate(capacities):
        for _ in range(cap):
            facility_slots.append(facilities[fac_idx])
            slot_to_facility.append(fac_idx)
    facility_slots = np.array(facility_slots)
    # print(facility_slots)
    # print(slot_to_facility)

    for i in range(n_customers):
        # Slice the prefix customers c_1 ... c_i
        prefix_customers = customers[:i+1]

        # Create cost matrix for Hungarian. Solve optimal assignment for the prefix
        cost_matrix = compute_distance(prefix_customers, facility_slots)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Convert Hungarian slots to facility IDs || multiset subtraction
        prefix_facilities = [slot_to_facility[c] for c in col_ind]
        # Copy for mutability 
        pref = prefix_facilities[:]
        used = final_assignment[:i]
        for fac in used:
            if fac != -1:                    
                try:                    
                    pref.remove(fac)         
                except ValueError:                    
                    pass                     

        # After removing all committed assignments, 'pref' should contain exactly ONE facility for the new customer
        facility_index = pref[0] if len(pref) > 0 else -1

        final_assignment[i] = facility_index

        # print(row_ind)
        # print([slot_to_facility[c] for c in col_ind])
        # print(f"pref: {pref}")
        # print(final_assignment)
    return final_assignment

def partition_algorithm(greedy_assignments, optimal_assignments, customer_positions, facility_positions):
    """
    Create partitions of customers based on greedy and optimal assignment relationships.
    Args:
        greedy_assignments: list of facility indices from greedy algorithm
        optimal_assignments: list of facility indices from optimal algorithm
        customer_positions: numpy array of customer positions
        facility_positions: numpy array of facility positions
    Returns:
        partitions: list of lists, each containing customer indices in a partition
        partition_costs: list of dicts with costs for each partition
    """
    # n = len(customer_positions)
    # -1 is marked as removed from the input 
    valid_customers = [i for i, g in enumerate(greedy_assignments) if g != -1]
    n=len(valid_customers)
    total_customer = len(customer_positions)
    assigned = set()
    partitions = []

    while len(assigned) < n:
        # Find first unassigned customer
        c1 = None
        for c in range(total_customer):
            if (greedy_assignments[c] != -1) and c not in assigned:
                c1 = c
                break

        if c1 is None:
            break

        current_partition = [c1]
        assigned.add(c1)

        # Get facilities that c1 is assigned to
        f1_optimal = optimal_assignments[c1]

        # if f1_greedy == -1 or f1_optimal == -1:
        #     break

        current = c1
        while True:
            # Find which facility greedy assigned current customer to
            fk = greedy_assignments[current]
            next_c = -1
            for idx in range(total_customer):
                if optimal_assignments[idx] == fk and idx not in assigned:
                    next_c = idx
                    break

            if fk == -1 or next_c == -1:
                # No unassigned customer found for this facility in optimal
                break

            current_partition.append(next_c)
            assigned.add(next_c)

            # Check if we've completed the cycle
            if greedy_assignments[next_c] == f1_optimal:
                break

            current = next_c

        partitions.append(current_partition)

    # Calculate costs for each partition
    partition_costs = []
    for partition in partitions:
        greedy_cost = 0.0
        optimal_cost = 0.0

        for customer_idx in partition:
            # Greedy cost
            if greedy_assignments[customer_idx] != -1:
                dist = compute_distance(
                    facility_positions[greedy_assignments[customer_idx]:greedy_assignments[customer_idx]+1],
                    customer_positions[customer_idx]
                )[0]
                greedy_cost += float(dist)

            # Optimal cost
            if optimal_assignments[customer_idx] != -1:
                dist = compute_distance(
                    facility_positions[optimal_assignments[customer_idx]:optimal_assignments[customer_idx]+1],
                    customer_positions[customer_idx]
                )[0]
                optimal_cost += float(dist)

        partition_costs.append({
            "greedy_cost": greedy_cost,
            "optimal_cost": optimal_cost,
            "cost_ratio": greedy_cost / optimal_cost if optimal_cost !=0 else 0,
            "customers_in_partition": len(partition)
        })

    return partitions, partition_costs

def calculate_assignment_cost(customers, facilities, assignments):
    """
    Works for both line (1D) and plane (2D) problems.
    Args:
        customers: numpy array of customer positions (n_customers, dim)
        facilities: numpy array of facility positions (n_facilities, dim)
        assignments: list of facility indices for each customer (-1 for unassigned)
    Returns:
        total_cost: sum of distances for all assigned customers
    """
    total_cost = 0.0

    for customer_idx, facility_idx in enumerate(assignments):
        if facility_idx != -1:  # Only consider assigned customers
            # Calculate distance between customer and assigned facility
            distance = compute_distance(
                facilities[facility_idx:facility_idx+1],
                customers[customer_idx]
            )[0]
            total_cost += float(distance)


    return total_cost

def calculate_assignment_cost_dict(customers_dict, facilities_dict, assignments):
    """
    Works for both line (1D) and plane (2D) problems.
    Args:
        customers: dict of customer positions 
        facilities: dict of facility positions
        assignments: list of facility indices for each customer (-1 for unassigned)
    Returns:
        total_cost: sum of distances for all assigned customers
    """
    total_cost = 0.0
    customers = {s["id"]: s for s in customers_dict}
    facilities = {s["id"]: s for s in facilities_dict}

    for customer_idx, facility_idx in enumerate(assignments):
        if facility_idx != -1:  # Only consider assigned customers
            # Calculate distance between customer and assigned facility
            facility_pos = np.array(facilities[facility_idx]["position"])
            customer_pos = np.array(customers[customer_idx]["position"])

            # To preserve original behavior: facility must be 2D (1, dim)
            facility_pos = facility_pos.reshape(1, -1)

            distance = compute_distance(facility_pos, customer_pos)[0]

            total_cost += float(distance)

    return total_cost

def maximal_algorithm(partitions, final_greedy, final_optimal,customers, facilitys):
    # print(partitions, final_greedy, final_optimal,customers, facilitys)
    maximal = [
    {
        "id": c["id"],
        "position": c["position"].copy()
    }
    for c in customers
    ]
    
    for partition in partitions:
        c1_idx = partition[0]
        f1_g_idx = final_greedy[c1_idx]
        f1_o_idx = final_optimal[c1_idx]

        c1 = next(c for c in maximal if c["id"] == c1_idx)
        f1_g = next(c for c in facilitys if c["id"] == f1_g_idx)
        f1_o = next(c for c in facilitys if c["id"] == f1_o_idx)

        c1["position"] = midpoint(f1_g["position"],f1_o["position"])

        for part in partition[1:]:
            # c_idx = part
            f_o_idx = final_optimal[part]

            ck = next(c for c in maximal if c["id"] == part)
            f_o = next(c for c in facilitys if c["id"] == f_o_idx)

            ck["position"] = f_o["position"]
    
    maximal_cost_g = calculate_assignment_cost_dict(maximal, facilitys,final_greedy)
    maximal_cost_o = calculate_assignment_cost_dict(maximal, facilitys,final_optimal)

    maximal_cost = {
        "maximal_greedy_cost": maximal_cost_g,
        "maximal_optimal_cost": maximal_cost_o,
        "maximal_cost_ratio": float( maximal_cost_g / maximal_cost_o ) if maximal_cost_o!=0 else 0
    }

    return maximal, maximal_cost

def process_test_case(facility_count, capacity, dim=2, grid_size=100):
    global generating_space, G
    total_customers = facility_count * capacity

    if(generating_space == 'graph'):
        #generate graph
        G = generate_graph(4*total_customers, 0.4)
    
        # Generate positions as node IDs
        facility_positions = generate_graph_positions(facility_count, 4*total_customers)
        customer_positions = generate_graph_positions(total_customers, 4*total_customers)

        # graph_data = None
        # if generating_space == 'graph':
        edges_list = [
            {
                "from": int(u),
                "to": int(v),
                "weight": float(G[u][v]['weight'])
            }
            for u, v in G.edges()
        ]
        
        graph_data = {
            "num_nodes": len(G.nodes()),
            "edges": edges_list,
            # "distance_matrix": distance_matrix.tolist()
        }
    
    else:
        # Generate positions
        facility_positions = generate_positions(facility_count, grid_size, dim)
        customer_positions = generate_positions(total_customers, grid_size, dim)
    # # Create facility dictionaries
    # facilities_dict = [
    #     {
    #         "id": i,
    #         "position": [float(x) for x in facility_positions[i]],
    #         "capacity": capacity
    #     }
    #     for i in range(facility_count)
    # ]

    # # Create customer dictionaries
    # customers_dict = [
    #     {
    #         "id": i,
    #         "position": [float(x) for x in customer_positions[i]]
    #     }
    #     for i in range(total_customers)
    # ]

    # Create facility dictionaries
    facilities_dict = [
        {
            "id": i,
            "position": [int(facility_positions[i][0])] if generating_space == 'graph' else [float(x) for x in facility_positions[i]],
            "capacity": capacity
        }
        for i in range(facility_count)
    ]

    # Create customer dictionaries
    customers_dict = [
        {
            "id": i,
            "position": [int(customer_positions[i][0])] if generating_space == 'graph' else [float(x) for x in customer_positions[i]]
        }
        for i in range(total_customers)
    ]
    # Initial capacities for assignment
    capacities = [capacity] * facility_count

    # Calculate greedy and optimal assignments
    greedy_assignments = greedy_assignment(customer_positions, facility_positions, capacities)
    optimal_fill_assignments = optimal_fill_assignment(customer_positions, facility_positions, capacities)
    optimal_assignments = hungarian_assignment(customer_positions, facility_positions, capacities)
    # Reject test case if non trivial customers is below 5
    if sum(1 for a, b in zip(greedy_assignments, optimal_assignments) if a != b) <10:
        return 0
    if sum(1 for a, b in zip(optimal_fill_assignments, optimal_assignments) if a != b) <10:
        return 0

    # Process assignments - match greedy and optimal
    final_greedy = [-1] * total_customers
    final_optimal = [-1] * total_customers
    final_optimal_fill = [-1] * total_customers
    final_optimal_of = [-1] * total_customers   
    
    # Remove trivial for greedy
    for idx, assignment in enumerate(greedy_assignments):
        if(assignment != -1 and assignment == optimal_assignments[idx]):
            facility = next((f for f in facilities_dict if f['id'] == assignment), None)
            facility['capacity'] -= 1
        else:
            final_greedy[idx] = assignment
            final_optimal[idx] = optimal_assignments[idx]
    
    # Remove trivial for optimal fill
    for idx, assignment in enumerate(optimal_fill_assignments):
        if(assignment != -1 and assignment != optimal_assignments[idx]):
            # facility = next((f for f in facilities_dict if f['id'] == assignment), None)
            # facility['capacity'] -= 1
        # else:
            final_optimal_fill[idx] = assignment
            final_optimal_of[idx] = optimal_assignments[idx]
    # Calculate costs for all assignments
    greedy_cost = calculate_assignment_cost(customer_positions, facility_positions, greedy_assignments)
    optimal_cost = calculate_assignment_cost(customer_positions, facility_positions, optimal_assignments)
    final_greedy_cost = calculate_assignment_cost(customer_positions, facility_positions, final_greedy)
    final_optimal_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal)

    optimal_fill_cost = calculate_assignment_cost(customer_positions, facility_positions, optimal_fill_assignments)
    final_optimal_fill_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal_fill)
    final_optimal_of_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal_of)

    # Generate partitions
    partitions, partition_costs = partition_algorithm(final_greedy, final_optimal, customer_positions, facility_positions)
    maximal, maximal_costs = maximal_algorithm(partitions, final_greedy, final_optimal, customers_dict, facilities_dict)
    
    of_partitions, of_partition_costs = partition_algorithm(final_optimal_fill, final_optimal_of,customer_positions, facility_positions)
    of_maximal, of_maximal_costs = maximal_algorithm(of_partitions, final_optimal_fill, final_optimal_of,customers_dict, facilities_dict)

    return {
        "facility": facilities_dict,
        "customer": customers_dict,
        "greedy": [int(x) if x != -1 else -1 for x in greedy_assignments],
        "optimal fill": [int(x) if x != -1 else -1 for x in optimal_fill_assignments],
        "optimal": [int(x) if x != -1 else -1 for x in optimal_assignments],
        "final greedy": [int(x) if x != -1 else -1 for x in final_greedy],
        "final optimal": [int(x) if x != -1 else -1 for x in final_optimal],

        "final optimal fill": [int(x) if x != -1 else -1 for x in final_optimal_fill],
        "final optimal of": [int(x) if x != -1 else -1 for x in final_optimal_of],
        "costs": {
            "greedy_cost": float(greedy_cost),
            "optimal_cost": float(optimal_cost),
            "original_greedy_ratio": float(greedy_cost / optimal_cost) if optimal_cost > 0 else 0,
            "final_greedy_cost": float(final_greedy_cost),
            "final_optimal_cost": float(final_optimal_cost),
            "final_ratio": float(final_greedy_cost / final_optimal_cost) if final_optimal_cost > 0 else 0,
            "maximal_greedy_cost": maximal_costs['maximal_greedy_cost'],
            "maximal_optimal_cost": maximal_costs['maximal_optimal_cost'],
            "maximal_cost_ratio": maximal_costs['maximal_cost_ratio'],
            
            "optimal_fill_cost": float(optimal_fill_cost),
            "original_of_ratio": float(optimal_fill_cost / optimal_cost) if optimal_cost > 0 else 0,
            "final_optimal_fill_cost": float(final_optimal_fill_cost),
            "final_optimal_of_cost": float(final_optimal_of_cost),
            "final_of_ratio": float(final_optimal_fill_cost / final_optimal_of_cost) if final_optimal_of_cost > 0 else 0,
            "maximal_optimal_fill_cost": of_maximal_costs['maximal_greedy_cost'],
            "of_maximal_optimal_cost": of_maximal_costs['maximal_optimal_cost'],
            "of_maximal_cost_ratio": of_maximal_costs['maximal_cost_ratio'],
        },
        "partitions": partitions,
        "partition costs": partition_costs,
        "maximal input": maximal,

        "of_partitions": of_partitions,
        "of_partition costs": of_partition_costs,
        "of_maximal input":of_maximal
    }

mode = sys.argv[1]
print (mode)
test_cases_per_setting =25
if mode == "f":
    facilities_list = list(range(10, 101, 5))
    capacities_list = [5]
elif mode == "c":
    facilities_list = [30]
    capacities_list = list(range(1,16))
elif mode == "t":
    test_cases_per_setting =1
    facilities_list = [5]
    capacities_list = [3]

# Parameters
# facilities_list = list(range(10, 101, 5))
# capacities_list = list(range(1,16))
# facilities_list = [30]
# capacities_list = [1, 2]
# capacities_list = [5]
# test_cases_per_setting =1
output_folder = "facility_datasets"

os.makedirs(output_folder, exist_ok=True)

# Generate datasets for both line (1D) and plane (2D)
for dim_type in ['line', 'plane','graph']:
# for dim_type in ['graph']:
    generating_space = dim_type
    dim = 1 if dim_type == 'line' else 2
    
    print(f"Generating {dim_type} datasets...")
    for f in facilities_list:
        grid_size = 10 * f
        print(f"  Generating dataset for {f} facilities in {dim_type}.")
        
        all_tests = []
        for cap in capacities_list:
            # filename = f"facility_{f}_{dim_type}.json"
            filename = f"facility_{f}_{cap}_{dim_type}.json"
            tempfile = f"facility_{f}_{cap}_{dim_type}.jsonl"  # .jsonl extension
            filepath = os.path.join(output_folder, filename)
            tempfilepath = os.path.join(output_folder, tempfile)
            # Skip if the file already exists
            if os.path.exists(filepath):
                print(f"Skipping {filename}, file already exists.")
                continue
            if os.path.exists(tempfilepath):
                print(f"File {tempfile} exists. Loading existing test cases...")
                with open(tempfilepath, "r") as fjson:
                    # all_tests = json.load(fjson)
                    # existing_count = sum(1 for _ in fjson)
                    # with open(filepath, "r") as f:
                    for line in fjson:
                        all_tests.append(json.loads(line.strip()))
                existing_count = len(all_tests)
                print(f"Found {existing_count} existing test cases.")
            
                if existing_count >= test_cases_per_setting: # If we already have enough test cases, skip
                    print(f"Already have {existing_count} test cases. Skipping.")
                    with open(filepath, "w") as fjson:
                        json.dump(all_tests, fjson, indent=2)
                    continue
            
                t = existing_count + 1
            else:
                print(f"File {tempfile} does not exist. Creating new test cases...")
                all_tests = []
                t = 1
            while t <= test_cases_per_setting:
                print('Test Case: '+str(t), end="\r")
                data = process_test_case(f, cap, dim=dim, grid_size=grid_size)
                if data == 0:
                    print("Fully Trivial Test case found. Ignoring....")
                    continue    # retry same t again. for loop ignores any modification to loop vairable
                all_tests.append(data)
                with open(tempfilepath, "a") as ft:
                    ft.write(json.dumps(data) + "\n")
                t += 1 
            with open(filepath, "w") as fjson:
                json.dump(all_tests, fjson, indent=2)
            os.remove(tempfilepath)
            print(f"Completed {filename} with {len(all_tests)} test cases.")
        

print("Dataset generation completed for spaces!")



##########################
# compute distance old
# def compute_distance(a, b):
#     """
#     Compute distance matrix depending on dimension and space type.
#     a: array of shape (n, dim) - typically facilities
#     b: array of shape (m, dim) or (dim,) - typically customers
#     Returns: distance matrix of shape (n, m) or (n,) if b is 1D
#     """
#     global generating_space, G
    
#     # Ensure b is at least 2D
#     if b.ndim == 1:
#         b = b.reshape(1, -1)
#         single_point = True
#     else:
#         single_point = False

#     if generating_space == 'graph':
#         # For graph: positions are node IDs # a: (n, 1) array of node IDs # b: (m, 1) array of node IDs
#         # Compute shortest path distances using the graph
#         n = a.shape[0]
#         m = b.shape[0]
#         distances = np.zeros((n, m))
        
#         for i in range(n):
#             node_i = int(a[i, 0])
#             for j in range(m):
#                 node_j = int(b[j, 0])
#                 try:
#                     # Get shortest path length between nodes
#                     distances[i, j] = nx.shortest_path_length(G, node_i, node_j, weight='weight')
#                 except nx.NetworkXNoPath:
#                     # If no path exists (shouldn't happen in connected graph)
#                     distances[i, j] = float('inf')
    
#     elif a.shape[1] == 1:  # line (1D)
#         distances = np.abs(a - b.T)
    
#     else:  # plane (2D) # a[:, None, :] shape: (n, 1, 2) # b[None, :, :] shape: (1, m, 2) # Result shape: (n, m)
#         distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

#     # If b was a single point, return 1D array instead of 2D
#     if single_point:
#         distances = distances.squeeze(axis=1)

#     return distances
##########################

# from collections import Counter

# # Convert Hungarian slots → facility IDs
# prefix_facilities = [slot_to_facility[c] for c in col_ind]

# # Count multiset of Hungarian assignments
# pref_count = Counter(prefix_facilities)

# # Count multiset of already committed assignments (0..i-1)
# used_count = Counter([f for f in final_assignment[:i] if f != -1])

# # Multiset difference
# diff = pref_count - used_count

# # There MUST be exactly 1 entry corresponding to the new customer's assignment
# if len(diff) == 0:
#     facility_index = -1
# else:
#     facility_index = list(diff.elements())[0]

# # Commit assignment
# final_assignment[i] = facility_index
# if facility_index != -1:
#     remaining_capacity[facility_index] -= 1


### working def process
# def process_test_case(facility_count, capacity, dim=2, grid_size=100):
#     total_customers = facility_count * capacity

#     # Generate positions
#     facility_positions = generate_positions(facility_count, grid_size, dim)
#     customer_positions = generate_positions(total_customers, grid_size, dim)

#     # Create facility dictionaries
#     facilities_dict = [
#         {
#             "id": i,
#             "position": [float(x) for x in facility_positions[i]],
#             "capacity": capacity
#         }
#         for i in range(facility_count)
#     ]

#     # Create customer dictionaries
#     customers_dict = [
#         {
#             "id": i,
#             "position": [float(x) for x in customer_positions[i]]
#         }
#         for i in range(total_customers)
#     ]

#     # Initial capacities for assignment
#     capacities = [capacity] * facility_count

#     # Calculate greedy and optimal assignments
#     greedy_assignments = greedy_assignment(customer_positions, facility_positions, capacities)
#     optimal_assignments = hungarian_assignment(customer_positions, facility_positions, capacities)
    
#     # Process assignments - match greedy and optimal
#     final_greedy = [-1] * total_customers
#     final_optimal = [-1] * total_customers

#     for idx, assignment in enumerate(greedy_assignments):
#         if(assignment != -1 and assignment == optimal_assignments[idx]):
#             facility = next((f for f in facilities_dict if f['id'] == assignment), None)
#             facility['capacity'] -= 1
#         else:
#             final_greedy[idx] = assignment
#             final_optimal[idx] = optimal_assignments[idx]

#     # Calculate costs for all assignments
#     greedy_cost = calculate_assignment_cost(customer_positions, facility_positions, greedy_assignments)
#     optimal_cost = calculate_assignment_cost(customer_positions, facility_positions, optimal_assignments)
#     final_greedy_cost = calculate_assignment_cost(customer_positions, facility_positions, final_greedy)
#     final_optimal_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal)

#     # Generate partitions
#     partitions, partition_costs = partition_algorithm(final_greedy, final_optimal, customer_positions, facility_positions)
#     maximal, maximal_costs = maximal_algorithm(partitions, final_greedy, final_optimal, customers_dict, facilities_dict)

#     # opt fill
#     optimal_fill_assignments = optimal_fill_assignment(customer_positions, facility_positions, capacities)
#     optimal_fill_cost = calculate_assignment_cost(customer_positions, facility_positions, optimal_fill_assignments)
#     final_optimal_fill = [-1] * total_customers
#     final_optimal_of = [-1] * total_customers
#     for idx, assignment in enumerate(optimal_fill_assignments):
#         if(assignment != -1 and assignment != optimal_assignments[idx]):
#             # facility = next((f for f in facilities_dict if f['id'] == assignment), None)
#             # facility['capacity'] -= 1
#         # else:
#             final_optimal_fill[idx] = assignment
#             final_optimal_of[idx] = optimal_assignments[idx]
#     final_optimal_fill_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal_fill)
#     final_optimal_of_cost = calculate_assignment_cost(customer_positions, facility_positions, final_optimal_of)
#     of_partitions, of_partition_costs = partition_algorithm(final_optimal_fill, final_optimal_of,customer_positions, facility_positions)
#     of_maximal, of_maximal_costs = maximal_algorithm(of_partitions, final_optimal_fill, final_optimal_of,customers_dict, facilities_dict)

#     return {
#         "facility": facilities_dict,
#         "customer": customers_dict,
#         "greedy": [int(x) if x != -1 else -1 for x in greedy_assignments],
#         "optimal fill": [int(x) if x != -1 else -1 for x in optimal_fill_assignments],
#         "optimal": [int(x) if x != -1 else -1 for x in optimal_assignments],
#         "final greedy": [int(x) if x != -1 else -1 for x in final_greedy],
#         "final optimal": [int(x) if x != -1 else -1 for x in final_optimal],
#         "final optimal fill": [int(x) if x != -1 else -1 for x in final_optimal_fill],
#         "final optimal of": [int(x) if x != -1 else -1 for x in final_optimal_of],
#         "costs": {
#             "greedy_cost": float(greedy_cost),
#             "optimal_cost": float(optimal_cost),
#             "original_greedy_ratio": float(greedy_cost / optimal_cost) if optimal_cost > 0 else 0,
#             "final_greedy_cost": float(final_greedy_cost),
#             "final_optimal_cost": float(final_optimal_cost),
#             "final_ratio": float(final_greedy_cost / final_optimal_cost) if final_optimal_cost > 0 else 0,
#             "maximal_greedy_cost": maximal_costs['maximal_greedy_cost'],
#             "maximal_optimal_cost": maximal_costs['maximal_optimal_cost'],
#             "maximal_cost_ratio": maximal_costs['maximal_cost_ratio'],
            
#             "optimal_fill_cost": float(optimal_fill_cost),
#             "original_of_ratio": float(optimal_fill_cost / optimal_cost) if optimal_cost > 0 else 0,
#             "final_optimal_fill_cost": float(final_optimal_fill_cost),
#             "final_optimal_of_cost": float(final_optimal_of_cost),
#             "final_of_ratio": float(final_optimal_fill_cost / final_optimal_of_cost) if final_optimal_cost > 0 else 0,
#             "maximal_optimal_fill_cost": of_maximal_costs['maximal_optimal_cost'],
#             "maximal_cost_ratio": of_maximal_costs['maximal_cost_ratio'],
#         },
#         "partitions": partitions,
#         "partition costs": partition_costs,
#         "maximal input": maximal,
#         "of_partitions": of_partitions,
#         "of_partition costs": of_partition_costs,
#         "of_maximal input":of_maximal
#     }

