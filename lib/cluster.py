"""HDBSCAN clustering with Optuna optimization and Claude-powered topic naming.

This module handles the core clustering pipeline: building extended text
summaries for embedding, loading pretrained UMAP reducers, and computing
clustering quality metrics.  Optuna-based hyperparameter tuning and
LLM-driven topic naming will be added in subsequent tasks.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from config import EMBEDDING_MODEL, MIN_COMPONENTS, RANDOM_STATE, OPTUNA_TRIALS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text summaries for embedding
# ---------------------------------------------------------------------------

def _create_extended_summary(row: dict) -> str:
    """Concatenate title + description + topics + summary for embedding.

    Builds a combined text string from available article fields, suitable
    for generating embeddings that capture the full semantic content of
    an article.

    Args:
        row: Dict-like object with optional keys ``title``, ``description``,
             ``topics``, and ``summary``.

    Returns:
        Combined text with parts separated by double newlines.
    """
    parts = []
    if row.get("title"):
        parts.append(str(row["title"]).strip())
    if row.get("description"):
        parts.append(str(row["description"]).strip())
    if row.get("topics"):
        topics = row["topics"]
        if isinstance(topics, list):
            topics_str = ", ".join(str(t).strip() for t in topics if t)
        else:
            topics_str = str(topics).strip()
        if topics_str:
            parts.append(topics_str)
    if row.get("summary") and pd.notna(row["summary"]):
        parts.append(str(row["summary"]).strip())
    return "\n\n".join(parts)


def _create_short_summary(row: dict) -> str:
    """Concatenate short_summary + topics for compact embedding text.

    Args:
        row: Dict-like object with optional keys ``short_summary`` and
             ``topics``.

    Returns:
        Combined text string.
    """
    retval = ""
    short = row.get("short_summary")
    if short and pd.notna(short):
        retval += str(short).strip()
        topics = row.get("topics")
        topics_str = str(topics).strip() if topics else ""
        if topics_str:
            retval += f" Topics: {topics_str}"
    return retval


# ---------------------------------------------------------------------------
# UMAP reducer persistence
# ---------------------------------------------------------------------------

def load_umap_reducer(path: str) -> Any:
    """Load a pretrained UMAP reducer from a pickle file.

    Args:
        path: Filesystem path to the ``.pkl`` file.

    Returns:
        The deserialized UMAP reducer object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"UMAP reducer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Clustering quality metrics
# ---------------------------------------------------------------------------

def calculate_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    clusterer: Any,
) -> dict:
    """Calculate clustering quality metrics for HDBSCAN results.

    Computes silhouette score, Calinski-Harabasz index, Davies-Bouldin index,
    noise ratio, and (when available) the HDBSCAN relative validity index.

    Args:
        embeddings: Array of shape ``(n_samples, n_features)``.
        labels: Cluster labels from HDBSCAN; ``-1`` denotes noise points.
        clusterer: Optional HDBSCAN clusterer instance (used for
                   ``relative_validity_``).

    Returns:
        Dict with keys ``silhouette``, ``calinski_harabasz``,
        ``davies_bouldin``, ``noise_ratio``, ``n_clusters``, and
        ``validity_index``.
    """
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = float((~mask).sum()) / len(labels)

    # Not enough clusters or non-noise points for meaningful metrics
    if n_clusters < 2 or mask.sum() < 2:
        return {
            "silhouette": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float("inf"),
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "validity_index": 0.0,
        }

    non_noise_embeddings = embeddings[mask]
    non_noise_labels = labels[mask]

    sil = silhouette_score(non_noise_embeddings, non_noise_labels)
    ch = calinski_harabasz_score(non_noise_embeddings, non_noise_labels)
    db = davies_bouldin_score(non_noise_embeddings, non_noise_labels)

    validity = 0.0
    if clusterer is not None and hasattr(clusterer, "relative_validity_"):
        validity = clusterer.relative_validity_

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters,
        "validity_index": validity,
    }
