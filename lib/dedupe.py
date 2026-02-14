"""Embedding-based duplicate detection for newsletter articles.

Syndicated articles (AP, Reuters) appear under different URLs but contain
near-identical text.  This module embeds article text via OpenAI, computes
pairwise cosine similarity, and drops the shorter duplicate from each pair.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import openai
import pandas as pd
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, MAX_EMBED_TOKENS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

async def get_embeddings_batch(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 25,
) -> List[List[float]]:
    """Get embeddings from OpenAI in batches.

    Args:
        texts: List of strings to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (one per input text).
    """
    client = openai.AsyncOpenAI()
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_and_truncate_files(
    df: pd.DataFrame, max_tokens: int = MAX_EMBED_TOKENS
) -> pd.DataFrame:
    """Read text files from *text_path* column and truncate to *max_tokens*.

    Missing or unreadable files produce an empty string.

    Returns:
        Copy of *df* with an added ``truncated_text`` column.
    """
    enc = tiktoken.encoding_for_model("gpt-4")  # cl100k_base
    texts: List[str] = []
    for path in df["text_path"]:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
            tokens = enc.encode(text)[:max_tokens]
            texts.append(enc.decode(tokens))
        except (FileNotFoundError, OSError):
            texts.append("")
    result = df.copy()
    result["truncated_text"] = texts
    return result


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def create_similarity_matrix(
    embeddings: np.ndarray, index: pd.Index
) -> pd.DataFrame:
    """Compute pairwise cosine similarity and return as a labelled DataFrame."""
    sim = cosine_similarity(embeddings)
    return pd.DataFrame(sim, index=index, columns=index)


def find_duplicate_pairs(
    similarity_df: pd.DataFrame, threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[int, int]]:
    """Return ``(i, j)`` pairs whose similarity exceeds *threshold*.

    Only the upper triangle is scanned so each pair appears once.
    """
    pairs: List[Tuple[int, int]] = []
    n = len(similarity_df)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_df.iloc[i, j] > threshold:
                pairs.append(
                    (similarity_df.index[i], similarity_df.columns[j])
                )
    return pairs


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_duplicates(
    df: pd.DataFrame, pairs: List[Tuple]
) -> pd.DataFrame:
    """For each duplicate pair, drop the article with less content.

    Uses the ``content_length`` column to decide which to keep.
    """
    to_drop: set = set()
    for idx_a, idx_b in pairs:
        if idx_a in to_drop or idx_b in to_drop:
            continue
        len_a = df.loc[idx_a, "content_length"] if idx_a in df.index else 0
        len_b = df.loc[idx_b, "content_length"] if idx_b in df.index else 0
        to_drop.add(idx_a if len_b >= len_a else idx_b)
    return df.drop(index=to_drop)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

async def process_dataframe_with_filtering(
    df: pd.DataFrame, similarity_threshold: float = SIMILARITY_THRESHOLD
) -> pd.DataFrame:
    """End-to-end dedup: embed -> similarity matrix -> filter duplicates.

    Expects *df* to already contain a ``truncated_text`` column (e.g. via
    :func:`read_and_truncate_files`) and a ``content_length`` column.

    Returns:
        Filtered DataFrame with near-duplicate rows removed.
    """
    if len(df) < 2:
        return df

    texts = df.get("truncated_text", pd.Series([""] * len(df))).tolist()
    # Replace empty strings so the embedding API doesn't reject them
    texts = [t if t else "empty" for t in texts]

    embeddings = await get_embeddings_batch(texts)
    emb_array = np.array(embeddings)

    sim_df = create_similarity_matrix(emb_array, df.index)
    pairs = find_duplicate_pairs(sim_df, similarity_threshold)

    logger.info(
        "Dedup: %d articles -> %d duplicate pairs found", len(df), len(pairs)
    )

    return filter_duplicates(df, pairs)
