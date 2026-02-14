# tests/test_cluster.py
"""Tests for lib/cluster.py — HDBSCAN clustering and topic naming."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


class TestCreateExtendedSummary:
    def test_concatenates_fields(self):
        from lib.cluster import _create_extended_summary
        row = {
            "title": "AI Breakthrough",
            "description": "Major discovery",
            "topics": "AI, Research",
            "summary": "Scientists found new approach",
        }
        result = _create_extended_summary(row)
        assert "AI Breakthrough" in result
        assert "Major discovery" in result
        assert "AI, Research" in result
        assert "Scientists found" in result

    def test_handles_missing_fields(self):
        from lib.cluster import _create_extended_summary
        row = {"title": "AI Breakthrough"}
        result = _create_extended_summary(row)
        assert "AI Breakthrough" in result


class TestCreateShortSummary:
    def test_concatenates_fields(self):
        from lib.cluster import _create_short_summary
        row = {"short_summary": "Brief summary", "topics": "AI, ML"}
        result = _create_short_summary(row)
        assert "Brief summary" in result
        assert "AI, ML" in result


class TestLoadUmapReducer:
    def test_loads_pickle(self, tmp_path):
        from lib.cluster import load_umap_reducer
        import pickle
        # Create a simple picklable stand-in for a UMAP reducer
        fake_reducer = {"name": "umap_reducer", "n_components": 690}
        pkl_path = tmp_path / "test_reducer.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(fake_reducer, f)

        reducer = load_umap_reducer(str(pkl_path))
        assert reducer is not None
        assert reducer["name"] == "umap_reducer"

    def test_raises_on_missing_file(self):
        from lib.cluster import load_umap_reducer
        with pytest.raises(FileNotFoundError):
            load_umap_reducer("/nonexistent/reducer.pkl")


class TestCalculateClusteringMetrics:
    def test_returns_metrics_dict(self):
        from lib.cluster import calculate_clustering_metrics
        embeddings = np.random.rand(20, 10)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1])
        metrics = calculate_clustering_metrics(embeddings, labels, clusterer=None)
        assert "silhouette" in metrics
        assert "noise_ratio" in metrics
        assert 0 <= metrics["noise_ratio"] <= 1

    def test_all_noise_returns_zero_silhouette(self):
        from lib.cluster import calculate_clustering_metrics
        embeddings = np.random.rand(10, 5)
        labels = np.array([-1] * 10)
        metrics = calculate_clustering_metrics(embeddings, labels, clusterer=None)
        assert metrics["silhouette"] == 0.0
