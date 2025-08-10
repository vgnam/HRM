import os
import numpy as np
import json


def generate_tsp_instance(n_cities, seed=None):
    rng = np.random.default_rng(seed)
    cities = rng.random((n_cities, 2), dtype=np.float32)

    # Compute symmetric distance matrix
    dist_matrix = np.linalg.norm(cities[:, None] - cities[None, :], axis=-1)  # [n, n]
    dist_matrix = dist_matrix.astype(np.float32)

    tour = rng.permutation(n_cities)
    return dist_matrix, tour


# ========================
# Configuration
# ========================
n_cities = 30
N = 10000
input_dim = n_cities * n_cities
save_dir = "data/tsp_data/test"
os.makedirs(save_dir, exist_ok=True)

# ========================
# Generate Data
# ========================
inputs = np.zeros((N, input_dim), dtype=np.float32)
labels = np.zeros((N, n_cities), dtype=np.int32)

# 🔴 Puzzle identifiers: ALL ZEROS (as requested)
puzzle_identifiers = np.zeros(N, dtype=np.int32)  # All zeros

# Fill data
for i in range(N):
    dist_matrix, tour = generate_tsp_instance(n_cities)
    inputs[i] = dist_matrix.flatten()
    labels[i] = tour

# 🔵 Puzzle indices: from 0 to end (already correct)
puzzle_indices = np.arange(0, N, 1, dtype=np.int64)  # [0, 30, 60, ..., 300000]

# One group (all same size)
group_indices = np.array([0, N], dtype=np.int64)

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
    "seq_len": n_cities,
    "num_puzzle_identifiers": 1,
    "total_groups": 1,
    "mean_puzzle_examples": 1.0,
    "sets": ["all"]
}

with open(os.path.join(save_dir, "dataset.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"✅ Generated {N} TSP instances with {n_cities} cities each.")
print(f"📁 Saved to: {save_dir}")
print(f"💡 Input shape: {inputs.shape} = [N, {n_cities}x{n_cities}] (flattened distance matrix)")
print(f"💡 Label shape: {labels.shape} = [N, {n_cities}] (tour order)")
print(f"🔧 puzzle_identifiers: all zeros")
print(f"🔧 puzzle_indices: from 0 to {N * n_cities}")