"""Print cached multimodal dataset metadata without constructing models."""

from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cmumosi_cmumosei_iemocap_mult"

for dataset_name in ("cmumosi", "cmumosei", "iemocap"):
    with (DATA_DIR / f"{dataset_name}_data.pkl").open("rb") as stream:
        data = pickle.load(stream)
    print(dataset_name, {split: len(data[split]["text"]) for split in ("train", "valid", "test")})
    print(" keys", list(data["train"].keys()))
    print(" labels", np.asarray(data["train"]["labels"]).shape)
    for label in ("sentiment", "emotion"):
        if label in data["train"]:
            print(" ", label, np.asarray(data["train"][label]).shape)
    print(" vision", np.asarray(data["train"]["vision"]).shape)
    print(" audio", np.asarray(data["train"]["audio"]).shape)
