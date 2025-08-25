import os
import numpy as np
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def generate_tsp_instance(n_cities, seed=None):
    rng = np.random.default_rng(seed)
    cities = rng.random((n_cities, 2), dtype=np.float32)

    # Compute symmetric distance matrix
    dist_matrix = np.linalg.norm(cities[:, None] - cities[None, :], axis=-1)  # [n, n]
    dist_matrix = dist_matrix.astype(np.float32)

    return dist_matrix


def solve_tsp_ortools(dist_matrix):
    """Solve TSP using OR-Tools and return city order."""
    n_cities = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(n_cities, 1, 0)  # 1 vehicle, depot=0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1e6)  # scaled

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Params
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return np.array(tour, dtype=np.int32)
    else:
        return np.arange(n_cities, dtype=np.int32)  # fallback: identity


# ========================
# Configuration
# ========================
n_cities = 30
N = 1000  # ⚠ để nhỏ để test nhanh; tăng lên nếu cần
input_dim = n_cities * n_cities
save_dir = "data/tsp_data/train"
os.makedirs(save_dir, exist_ok=True)

# ========================
# Generate Data
# ========================
inputs = np.zeros((N, input_dim), dtype=np.float32)
labels = np.zeros((N, n_cities), dtype=np.int32)

# 🔴 Puzzle identifiers: ALL ZEROS
puzzle_identifiers = np.zeros(N, dtype=np.int32)

for i in range(N):
    dist_matrix = generate_tsp_instance(n_cities, seed=i)
    tour = solve_tsp_ortools(dist_matrix)

    inputs[i] = dist_matrix.flatten()
    labels[i] = tour

# 🔵 Puzzle indices
puzzle_indices = np.arange(0, N, 1, dtype=np.int64)
group_indices = np.arange(0, N, 1, dtype=np.int64)

# ========================
# Save Files
# ========================
np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
np.save(os.path.join(save_dir, "all__labels.npy"), labels)
np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)

# ========================
# Save metadata
# ========================
meta = {
    "pad_id": 0,
    "ignore_label_id": -100,
    "blank_identifier_id": 0,
    "vocab_size": n_cities,
    "seq_len": n_cities * n_cities,
    "num_puzzle_identifiers": 1,
    "total_groups": 1000,
    "mean_puzzle_examples": 1.0,
    "sets": ["all"]
}

with open(os.path.join(save_dir, "dataset.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Generated {N} TSP instances with {n_cities} cities each.")
print(f"📁 Saved to: {save_dir}")
print(f"💡 Input shape: {inputs.shape}")
print(f"💡 Label shape: {labels.shape}")
