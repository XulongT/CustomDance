"""Read a paired inference checkpoint and immutable deployment index."""

import json
from pathlib import Path

import numpy as np
import torch

from app.models.retriever import SharedRetriever


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "customdance-retriever-inference-v1":
        raise ValueError("expected the CustomDance inference bundle, not a training checkpoint")
    return checkpoint


def model_from_checkpoint(checkpoint, device="cpu"):
    model = SharedRetriever(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def read_index(index_dir):
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if embeddings.shape[0] != len(rows):
        raise ValueError("index embedding/metadata length mismatch")
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("non-finite values in motion embedding index")
    return manifest, embeddings, rows
