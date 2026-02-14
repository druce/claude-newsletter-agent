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
import hdbscan
import optuna
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import openai
from pydantic import BaseModel

from config import EMBEDDING_MODEL, MIN_COMPONENTS, RANDOM_STATE, OPTUNA_TRIALS
from llm import create_agent
from prompts import TOPIC_WRITER

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

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


# ---------------------------------------------------------------------------
# Optuna-based HDBSCAN hyperparameter optimization
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, embeddings_array: np.ndarray) -> float:
    """Optuna objective: optimize HDBSCAN params for clustering quality.

    Returns composite score: 0.5 * silhouette + 0.5 * validity.
    """
    max_cluster_size = max(3, len(embeddings_array) // 3)
    min_cluster_size = trial.suggest_int("min_cluster_size", 3, min(50, max_cluster_size))
    min_samples = trial.suggest_int("min_samples", 2, min(30, min_cluster_size))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings_array)

    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    if n_clusters < 2 or mask.sum() < n_clusters + 1:
        return 0.0

    sil = silhouette_score(embeddings_array[mask], labels[mask])
    validity = clusterer.relative_validity_ if hasattr(clusterer, "relative_validity_") else 0.0
    return 0.5 * max(sil, 0.0) + 0.5 * max(validity, 0.0)


def optimize_hdbscan(
    embeddings_array: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """Run Optuna to find best HDBSCAN hyperparameters.

    Args:
        embeddings_array: Shape (n_samples, n_features).
        n_trials: Number of Optuna trials.

    Returns:
        Dict with keys: min_cluster_size, min_samples, labels, score, clusterer.
    """
    if len(embeddings_array) < 6:
        labels = np.array([-1] * len(embeddings_array))
        return {"min_cluster_size": 3, "min_samples": 2, "labels": labels, "score": 0.0}

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, embeddings_array), n_trials=n_trials)

    best = study.best_params
    # Re-fit with best params to get the clusterer object
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best["min_cluster_size"],
        min_samples=best["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings_array)

    return {
        "min_cluster_size": best["min_cluster_size"],
        "min_samples": best["min_samples"],
        "labels": labels,
        "score": study.best_value,
        "clusterer": clusterer,
    }


# ---------------------------------------------------------------------------
# Cluster naming (LLM-powered)
# ---------------------------------------------------------------------------

class TopicText(BaseModel):
    """LLM response schema for cluster topic naming."""
    topic: str


async def get_embeddings_df(
    df: pd.DataFrame,
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Generate embeddings for extended summaries of articles.

    Args:
        df: DataFrame with article text fields.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.

    Returns:
        DataFrame of embedding vectors, indexed like *df*.
    """
    client = openai.AsyncOpenAI()
    texts = [_create_extended_summary(row.to_dict()) for _, row in df.iterrows()]
    texts = [t if t else "empty" for t in texts]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])

    return pd.DataFrame(all_embeddings, index=df.index)


async def name_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Name each cluster using Claude via the TOPIC_WRITER prompt.

    Clusters with fewer than 2 articles or noise clusters (label < 0) get "Other".

    Args:
        df: DataFrame with ``cluster_label`` and ``title`` columns.

    Returns:
        Copy of *df* with a ``cluster_name`` column added.
    """
    result = df.copy()
    result["cluster_name"] = "Other"

    agent = create_agent(
        model=TOPIC_WRITER.model,
        system_prompt=TOPIC_WRITER.system_prompt,
        user_prompt=TOPIC_WRITER.user_prompt,
        output_type=TopicText,
        reasoning_effort=TOPIC_WRITER.reasoning_effort,
    )

    unique_labels = sorted(set(df["cluster_label"]))
    for label in unique_labels:
        if label < 0:
            continue
        cluster_df = df[df["cluster_label"] == label]
        if len(cluster_df) < 2:
            continue
        titles = cluster_df["title"].tolist()
        input_text = "\n".join(f"- {t}" for t in titles)
        try:
            parsed = await agent.prompt_dict({"input_text": input_text})
            result.loc[cluster_df.index, "cluster_name"] = parsed.topic
        except Exception as exc:
            logger.error("Failed to name cluster %d: %s", label, exc)

    return result


async def do_clustering(
    df: pd.DataFrame,
    umap_reducer_path: str = "umap_reducer.pkl",
    n_trials: int = OPTUNA_TRIALS,
) -> pd.DataFrame:
    """Top-level clustering pipeline: embed -> reduce -> optimize -> cluster -> name.

    Args:
        df: DataFrame with text columns (title, summary, description, topics).
        umap_reducer_path: Path to pretrained UMAP reducer pickle.
        n_trials: Number of Optuna trials for HDBSCAN optimization.

    Returns:
        DataFrame with ``cluster_label`` and ``cluster_name`` columns added.
    """
    # 1. Get embeddings
    embeddings_df = await get_embeddings_df(df)
    embeddings_array = embeddings_df.values

    # 2. Reduce with pretrained UMAP
    reducer = load_umap_reducer(umap_reducer_path)
    reduced = reducer.transform(embeddings_array)

    # 3. Optimize HDBSCAN
    opt_result = optimize_hdbscan(reduced, n_trials=n_trials)

    # 4. Assign labels
    result = df.copy()
    result["cluster_label"] = opt_result["labels"]

    # 5. Name clusters
    result = await name_clusters(result)

    return result
