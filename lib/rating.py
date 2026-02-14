"""Article scoring — recency, length, composite rating, LLM assessments, Bradley-Terry battles."""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime
from typing import List, Optional, Tuple

import choix
import numpy as np
import pandas as pd
from scipy.stats import zscore

from llm import create_agent
from prompts import BATTLE_PROMPT, RATE_QUALITY, RATE_ON_TOPIC, RATE_IMPORTANCE

logger = logging.getLogger(__name__)

LN2 = math.log(2)
MAX_ARTICLE_AGE_DAYS = 7


def compute_recency_score(published_date: Optional[datetime]) -> float:
    """Half-life of 1 day: 2 * exp(-ln2 * age_days) - 1.

    Returns 0.0 for articles older than MAX_ARTICLE_AGE_DAYS or None dates.
    """
    if published_date is None:
        return 0.0
    age = (datetime.now() - published_date).total_seconds() / 86400
    if age > MAX_ARTICLE_AGE_DAYS:
        return 0.0
    return 2 * math.exp(-LN2 * age) - 1


def compute_length_score(content_length: int) -> float:
    """log10(content_length) - 3, clipped to [0, 2]."""
    if content_length <= 0:
        return 0.0
    raw = math.log10(content_length) - 3
    return max(0.0, min(2.0, raw))


def compute_composite_rating(
    reputation: float,
    length_score: float,
    on_topic: float,
    importance: float,
    low_quality: float,
    recency: float,
    bt_zscore: float,
) -> float:
    """Composite article rating formula."""
    return reputation + length_score + on_topic + importance - low_quality + recency + bt_zscore


async def _assess_with_probs(df: pd.DataFrame, prompt_config) -> pd.Series:
    """Generic LLM probability assessment. Iterates rows, gets P(token='1')."""
    agent = create_agent(
        model=prompt_config.model,
        system_prompt=prompt_config.system_prompt,
        user_prompt=prompt_config.user_prompt,
        reasoning_effort=prompt_config.reasoning_effort,
    )

    async def _assess_one(row):
        input_text = f"Title: {row.get('title', '')}\nSummary: {row.get('summary', '')}"
        probs = await agent.run_prompt_with_probs(
            variables={"input_text": input_text},
            target_tokens=["1", "0"],
        )
        return probs.get("1", 0.0)

    # Run concurrently with semaphore (agent already has one)
    tasks = [_assess_one(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    return pd.Series(list(results), index=df.index)


async def assess_quality(df: pd.DataFrame) -> pd.Series:
    """LLM probability of low quality."""
    return await _assess_with_probs(df, RATE_QUALITY)


async def assess_on_topic(df: pd.DataFrame) -> pd.Series:
    """LLM probability of AI-relevance."""
    return await _assess_with_probs(df, RATE_ON_TOPIC)


async def assess_importance(df: pd.DataFrame) -> pd.Series:
    """LLM probability of importance."""
    return await _assess_with_probs(df, RATE_IMPORTANCE)


# ---------------------------------------------------------------------------
# Bradley-Terry battle system
# ---------------------------------------------------------------------------

def swiss_pairing(
    df: pd.DataFrame,
    battle_history: set,
) -> List[Tuple[int, int]]:
    """Create Swiss-style pairings for Bradley-Terry battles.

    Sort articles by current rating, pair adjacent articles that haven't
    fought before. If an article can't be paired adjacently, try pairing
    with the next available opponent.

    Args:
        df: DataFrame with 'id' and 'rating' columns.
        battle_history: Set of (id_a, id_b) tuples already battled.

    Returns:
        List of (id_a, id_b) tuples for this round.
    """
    sorted_df = df.sort_values("rating", ascending=False).reset_index(drop=True)
    used = set()
    pairs: List[Tuple[int, int]] = []

    for i, row in sorted_df.iterrows():
        aid = int(row["id"])
        if aid in used:
            continue

        # Try pairing with the next available opponent
        for j in range(i + 1, len(sorted_df)):
            bid = int(sorted_df.iloc[j]["id"])
            if bid in used:
                continue
            # Skip if already battled (check both orderings)
            if (aid, bid) in battle_history or (bid, aid) in battle_history:
                continue
            pairs.append((aid, bid))
            used.add(aid)
            used.add(bid)
            break

    return pairs


async def swiss_batching(
    df: pd.DataFrame,
    battle_history: set,
    batch_size: int = 6,
) -> List[List[dict]]:
    """Get pairs from swiss_pairing and group into batches.

    Each batch is a list of dicts with 'id', 'title', 'summary' — ready
    to be sent to the battle LLM.

    Args:
        df: DataFrame with 'id', 'title', 'summary', 'rating'.
        battle_history: Set of (id_a, id_b) tuples already battled.
        batch_size: Max items per batch.

    Returns:
        List of batches (each batch is a list of article dicts).
    """
    pairs = swiss_pairing(df, battle_history)
    if not pairs:
        return []

    # Flatten all unique IDs from pairs
    all_ids: List[int] = []
    for a, b in pairs:
        if a not in all_ids:
            all_ids.append(a)
        if b not in all_ids:
            all_ids.append(b)

    # Build lookup of article data
    id_to_row = {}
    for _, row in df.iterrows():
        id_to_row[int(row["id"])] = {
            "id": int(row["id"]),
            "title": str(row.get("title", "")),
            "summary": str(row.get("summary", "")),
        }

    # Group IDs into batches of batch_size
    batches: List[List[dict]] = []
    for i in range(0, len(all_ids), batch_size):
        chunk_ids = all_ids[i : i + batch_size]
        batch = [id_to_row[cid] for cid in chunk_ids if cid in id_to_row]
        if len(batch) >= 2:
            batches.append(batch)

    return batches


async def process_battle_round(
    df: pd.DataFrame,
    batches: List[List[dict]],
    agent,
) -> List[Tuple[int, int]]:
    """Process batches through the battle agent and extract pairwise wins.

    For each batch, the agent returns items in ranked order (best first).
    Higher rank beats lower rank for every pair in the ranked list.

    Args:
        df: DataFrame with article data.
        batches: List of batches (each batch is a list of article dicts).
        agent: LLM agent with prompt_list() method.

    Returns:
        List of (winner_id, loser_id) tuples.
    """
    all_wins: List[Tuple[int, int]] = []

    async def _process_batch(batch: List[dict]) -> List[Tuple[int, int]]:
        try:
            ranked = await agent.prompt_list(
                items=batch,
                item_id_field="id",
                item_list_field="items",
            )
            # Extract pairwise wins: higher rank (earlier in list) beats lower rank
            wins = []
            for i in range(len(ranked)):
                for j in range(i + 1, len(ranked)):
                    winner_id = int(ranked[i]["id"])
                    loser_id = int(ranked[j]["id"])
                    wins.append((winner_id, loser_id))
            return wins
        except Exception as e:
            logger.error(f"Error processing battle batch: {e}")
            return []

    results = await asyncio.gather(*[_process_batch(b) for b in batches])
    for batch_wins in results:
        all_wins.extend(batch_wins)

    return all_wins


async def run_bradley_terry(
    df: pd.DataFrame,
    max_rounds: int = 8,
    batch_size: int = 6,
) -> pd.Series:
    """Run iterative Bradley-Terry battles with Swiss pairing.

    Creates a battle agent, runs rounds of Swiss-paired battles,
    accumulates wins/losses, and computes BT parameters with choix.
    Stops when convergence is reached or max_rounds is exhausted.

    Args:
        df: DataFrame with 'id', 'title', 'summary', 'rating' columns.
        max_rounds: Maximum number of battle rounds.
        batch_size: Items per battle batch.

    Returns:
        pd.Series of z-score normalized BT parameters, aligned to df.index.
    """
    n = len(df)
    if n < 3:
        return pd.Series(np.zeros(n), index=df.index)

    # Build ID <-> contiguous index mapping (choix needs 0-based indices)
    ids = df["id"].tolist()
    id_to_idx = {int(aid): idx for idx, aid in enumerate(ids)}

    # Working copy with BT rating column
    bt_df = df.copy()
    bt_df["rating"] = np.linspace(1.0, 0.0, n)  # initial rating by position

    battle_history: set = set()
    all_battles: List[Tuple[int, int]] = []  # (winner_idx, loser_idx) for choix

    # Create battle agent
    agent = create_agent(
        model=BATTLE_PROMPT.model,
        system_prompt=BATTLE_PROMPT.system_prompt,
        user_prompt=BATTLE_PROMPT.user_prompt,
        reasoning_effort=BATTLE_PROMPT.reasoning_effort,
    )

    convergence_threshold = n / 100
    min_rounds = max_rounds // 2
    previous_rankings = bt_df.sort_values("rating", ascending=False)["id"].values.copy()
    all_avg_changes: List[float] = []

    for round_num in range(1, max_rounds + 1):
        logger.info(f"BT round {round_num}/{max_rounds}")

        # Swiss pair and batch
        batches = await swiss_batching(bt_df, battle_history, batch_size=batch_size)
        if not batches:
            logger.info("No more valid batches — stopping")
            break

        # Run battles
        wins = await process_battle_round(bt_df, batches, agent)

        # Update history and accumulate results
        for winner_id, loser_id in wins:
            battle_history.add((winner_id, loser_id))
            battle_history.add((loser_id, winner_id))
            # Map to contiguous indices for choix
            w_idx = id_to_idx.get(winner_id)
            l_idx = id_to_idx.get(loser_id)
            if w_idx is not None and l_idx is not None:
                all_battles.append((w_idx, l_idx))

        if not all_battles:
            continue

        # Compute BT parameters
        bt_params = choix.opt_pairwise(n, all_battles)
        bt_df["rating"] = bt_params

        # Check convergence
        new_rankings = bt_df.sort_values("rating", ascending=False)["id"].values
        ranking_change_sum = np.abs(
            np.array([np.where(new_rankings == pid)[0][0] for pid in ids])
            - np.array([np.where(previous_rankings == pid)[0][0] for pid in ids])
        ).sum()
        avg_change = ranking_change_sum / n
        all_avg_changes.append(avg_change)
        previous_rankings = new_rankings.copy()

        logger.info(f"  avg rank change: {avg_change:.2f} (threshold: {convergence_threshold:.2f})")

        if len(all_avg_changes) > min_rounds:
            last_two = (all_avg_changes[-1] + all_avg_changes[-2]) / 2
            if last_two < convergence_threshold:
                logger.info("Convergence achieved — stopping")
                break
            if len(all_avg_changes) >= 4:
                prev_two = (all_avg_changes[-3] + all_avg_changes[-4]) / 2
                if last_two < n / 5 and last_two > prev_two:
                    logger.info("Avg rank change increasing — stopping")
                    break

    # Z-score normalize
    bt_values = bt_df["rating"].values
    if np.std(bt_values) > 0:
        bt_z = zscore(bt_values, ddof=0)
    else:
        bt_z = np.zeros(n)

    return pd.Series(bt_z, index=df.index)


async def rate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Top-level article rating pipeline.

    Runs all LLM assessments concurrently, computes recency/length scores,
    runs Bradley-Terry battles, and computes composite rating.

    Args:
        df: DataFrame with columns: id, title, summary, content_length,
            published, reputation.

    Returns:
        DataFrame with new columns: low_quality, on_topic, importance,
        recency, length_score, bt_zscore, rating.
    """
    result = df.copy()
    result["title"] = result["title"].fillna("").astype(str)
    result["summary"] = result["summary"].fillna("").astype(str)
    result["content_length"] = result["content_length"].fillna(1).astype(int)
    result["reputation"] = result["reputation"].fillna(0.0)

    # Run LLM assessments concurrently
    lq_task = assess_quality(result)
    ot_task = assess_on_topic(result)
    imp_task = assess_importance(result)
    low_quality, on_topic, importance = await asyncio.gather(lq_task, ot_task, imp_task)

    result["low_quality"] = low_quality.values
    result["on_topic"] = on_topic.values
    result["importance"] = importance.values

    # Recency score
    result["recency"] = result["published"].apply(compute_recency_score)

    # Length score
    result["length_score"] = result["content_length"].apply(compute_length_score)

    # Bradley-Terry battles
    # Prepare a temporary rating column for BT pairing
    result["rating"] = 0.0
    bt_scores = await run_bradley_terry(result)
    result["bt_zscore"] = bt_scores.values

    # Composite rating
    result["rating"] = result.apply(
        lambda row: compute_composite_rating(
            reputation=row["reputation"],
            length_score=row["length_score"],
            on_topic=row["on_topic"],
            importance=row["importance"],
            low_quality=row["low_quality"],
            recency=row["recency"],
            bt_zscore=row["bt_zscore"],
        ),
        axis=1,
    )

    return result
