from typing import Optional
import os
import csv
import json
import numpy as np
import ast
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import csv

from common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "microsoft/tsp"
    output_dir: str = "data/tsp"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0


def convert_subset(set_name: str, config: DataProcessConfig):

    # Read CSV and parse data
    inputs = []
    labels = []
    uids = []

    # 1. Load dataset directly from Hugging Face
    ds = load_dataset("microsoft/tsp", split="train")

    inputs = []
    labels = []
    uids = []

    # 2. Process rows directly
    for row in ds:
        # weight_matrix is already a list of lists, no string parsing needed
        weight_matrix = np.array(row["weight_matrix"], dtype=np.float32)

        # optimal_tour is already a list of lists → take the first tour
        first_tour = row["optimal_tour"][0]

        inputs.append(weight_matrix.flatten())  # Flatten to 1D
        labels.append([first_tour])
        uids.append(row["uid"])

    # Generate results structure
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out, uid in zip(tqdm(inputs), labels, uids):
        # Push puzzle (only single example)
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)  # All zeros as requested

    # Push group
    results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq):
        # Stack arrays and ensure they're in the right format
        arr = np.stack(seq)
        return arr

    def _labels_to_numpy(seq):
        # Handle labels (first elements of optimal tours)
        arr = np.array(seq).flatten()
        return arr

    def _uids_to_numpy(uids_list):
        # Convert UIDs to appropriate format
        return np.array(uids_list, dtype=object)

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _labels_to_numpy(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": _uids_to_numpy(uids),  # Use actual UIDs instead of indices
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Calculate metadata based on actual data
    input_size = results["inputs"].shape[1] if len(results["inputs"]) > 0 else 0
    max_city = int(np.max(results["labels"])) if len(results["labels"]) > 0 else 0

    metadata = PuzzleDatasetMetadata(
        seq_len=input_size,  # Size of flattened weight matrix
        vocab_size=max_city + 2,  # Max city index + padding

        pad_id=0,
        ignore_label_id=0,

        blank_identifier_id=0,
        num_puzzle_identifiers=1,

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
