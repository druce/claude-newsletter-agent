# tests/test_cluster.py
"""Tests for lib/cluster.py — HDBSCAN clustering and topic naming."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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


class TestObjective:
    def test_returns_float_score(self):
        from lib.cluster import objective
        import optuna
        np.random.seed(42)
        embeddings = np.vstack([
            np.random.randn(15, 5) + [3, 0, 0, 0, 0],
            np.random.randn(15, 5) + [0, 3, 0, 0, 0],
        ])
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial, embeddings)
        assert isinstance(score, float)


class TestOptimizeHdbscan:
    def test_returns_best_params(self):
        from lib.cluster import optimize_hdbscan
        np.random.seed(42)
        # Create data with 3 clear clusters
        cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        result = optimize_hdbscan(embeddings, n_trials=5)
        assert "min_cluster_size" in result
        assert "min_samples" in result
        assert "labels" in result
        assert "score" in result

    def test_handles_small_dataset(self):
        from lib.cluster import optimize_hdbscan
        embeddings = np.random.randn(5, 10)
        result = optimize_hdbscan(embeddings, n_trials=3)
        assert "labels" in result


class TestGetEmbeddingsDf:
    @pytest.mark.asyncio
    @patch("lib.cluster.openai.AsyncOpenAI")
    async def test_returns_dataframe(self, mock_openai_cls):
        from lib.cluster import get_embeddings_df
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        df = pd.DataFrame({
            "title": ["Article A", "Article B"],
            "summary": ["Summary A", "Summary B"],
            "description": ["Desc A", "Desc B"],
            "topics": ["AI", "ML"],
        })
        result = await get_embeddings_df(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.shape[1] == 3  # 3-dim embeddings


class TestNameClusters:
    @pytest.mark.asyncio
    @patch("lib.cluster.create_agent")
    async def test_names_each_cluster(self, mock_create):
        from lib.cluster import name_clusters
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.topic = "AI Healthcare"
        mock_agent.prompt_dict.return_value = mock_result
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "title": ["AI in hospitals", "Medical AI", "AI doctor", "Sports news"],
            "cluster_label": [0, 0, 0, -1],
        })
        result = await name_clusters(df)
        assert "cluster_name" in result.columns
        assert result[result["cluster_label"] == 0].iloc[0]["cluster_name"] == "AI Healthcare"
        assert result[result["cluster_label"] == -1].iloc[0]["cluster_name"] == "Other"


class TestDoClustering:
    @pytest.mark.asyncio
    @patch("lib.cluster.name_clusters")
    @patch("lib.cluster.optimize_hdbscan")
    @patch("lib.cluster.load_umap_reducer")
    @patch("lib.cluster.get_embeddings_df")
    async def test_end_to_end_pipeline(self, mock_embed, mock_umap, mock_opt, mock_name):
        from lib.cluster import do_clustering

        # Mock embeddings
        mock_embed.return_value = pd.DataFrame(
            np.random.randn(10, 50),
            index=range(10),
        )

        # Mock UMAP reducer
        mock_reducer = MagicMock()
        mock_reducer.transform.return_value = np.random.randn(10, 20)
        mock_umap.return_value = mock_reducer

        # Mock HDBSCAN
        mock_opt.return_value = {
            "labels": np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1]),
            "score": 0.5,
            "min_cluster_size": 3,
            "min_samples": 2,
            "clusterer": MagicMock(),
        }

        # Mock naming
        async def fake_name(df):
            df = df.copy()
            df["cluster_name"] = df["cluster_label"].apply(lambda x: f"Cluster {x}" if x >= 0 else "Other")
            return df
        mock_name.side_effect = fake_name

        df = pd.DataFrame({
            "title": [f"Article {i}" for i in range(10)],
            "summary": [f"Summary {i}" for i in range(10)],
            "description": ["desc"] * 10,
            "topics": ["ai"] * 10,
        })

        result = await do_clustering(df, umap_reducer_path="fake.pkl", n_trials=3)
        assert "cluster_label" in result.columns
        assert "cluster_name" in result.columns
        assert len(result) == 10
