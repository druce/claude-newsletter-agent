# tests/test_dedupe.py
"""Tests for lib/dedupe.py — embedding-based duplicate detection."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCreateSimilarityMatrix:
    def test_identity_diagonal(self):
        from lib.dedupe import create_similarity_matrix
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        index = pd.Index([0, 1, 2])
        sim = create_similarity_matrix(embeddings, index)
        assert sim.shape == (3, 3)
        np.testing.assert_almost_equal(sim.iloc[0, 0], 1.0)
        np.testing.assert_almost_equal(sim.iloc[0, 1], 0.0)

    def test_high_similarity_detected(self):
        from lib.dedupe import create_similarity_matrix
        embeddings = np.array([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0]])
        index = pd.Index([0, 1, 2])
        sim = create_similarity_matrix(embeddings, index)
        assert sim.iloc[0, 1] > 0.9


class TestFindDuplicatePairs:
    def test_finds_pairs_above_threshold(self):
        from lib.dedupe import find_duplicate_pairs
        sim_data = {0: {0: 1.0, 1: 0.96, 2: 0.3}, 1: {0: 0.96, 1: 1.0, 2: 0.2}, 2: {0: 0.3, 1: 0.2, 2: 1.0}}
        sim_df = pd.DataFrame(sim_data)
        pairs = find_duplicate_pairs(sim_df, threshold=0.925)
        assert len(pairs) == 1
        assert (0, 1) in pairs or (1, 0) in pairs

    def test_no_pairs_below_threshold(self):
        from lib.dedupe import find_duplicate_pairs
        sim_data = {0: {0: 1.0, 1: 0.5}, 1: {0: 0.5, 1: 1.0}}
        sim_df = pd.DataFrame(sim_data)
        pairs = find_duplicate_pairs(sim_df, threshold=0.925)
        assert len(pairs) == 0


class TestFilterDuplicates:
    def test_keeps_longer_article(self):
        from lib.dedupe import filter_duplicates
        df = pd.DataFrame({
            "title": ["Article A", "Article B", "Article C"],
            "content_length": [500, 1000, 200],
        })
        pairs = [(0, 1)]  # A and B are dupes
        result = filter_duplicates(df, pairs)
        assert len(result) == 2
        assert "Article B" in result["title"].values  # B has more content
        assert "Article A" not in result["title"].values

    def test_no_pairs_returns_unchanged(self):
        from lib.dedupe import filter_duplicates
        df = pd.DataFrame({"title": ["A", "B"], "content_length": [100, 200]})
        result = filter_duplicates(df, [])
        assert len(result) == 2


class TestReadAndTruncateFiles:
    def test_reads_text_files(self, tmp_path):
        from lib.dedupe import read_and_truncate_files
        text_file = tmp_path / "article.txt"
        text_file.write_text("This is a test article about artificial intelligence.")
        df = pd.DataFrame({"text_path": [str(text_file)]})
        result = read_and_truncate_files(df, max_tokens=100)
        assert "truncated_text" in result.columns
        assert "artificial intelligence" in result.iloc[0]["truncated_text"]

    def test_handles_missing_file(self, tmp_path):
        from lib.dedupe import read_and_truncate_files
        df = pd.DataFrame({"text_path": ["/nonexistent/file.txt"]})
        result = read_and_truncate_files(df, max_tokens=100)
        assert result.iloc[0]["truncated_text"] == ""


class TestGetEmbeddingsBatch:
    @pytest.mark.asyncio
    @patch("lib.dedupe.openai.AsyncOpenAI")
    async def test_returns_embeddings(self, mock_openai_cls):
        from lib.dedupe import get_embeddings_batch
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        result = await get_embeddings_batch(["test text"], batch_size=10)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]


class TestProcessDataframeWithFiltering:
    @pytest.mark.asyncio
    @patch("lib.dedupe.get_embeddings_batch")
    async def test_end_to_end_dedup(self, mock_embed):
        from lib.dedupe import process_dataframe_with_filtering
        # Two near-identical embeddings + one different
        mock_embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
            [0.0, 1.0, 0.0],
        ]
        df = pd.DataFrame({
            "text_path": ["a.txt", "b.txt", "c.txt"],
            "content_length": [100, 200, 300],
            "truncated_text": ["text a", "text b", "text c"],
        })
        result = await process_dataframe_with_filtering(df, similarity_threshold=0.925)
        # Should drop one of the near-identical pair (keep longer)
        assert len(result) == 2
