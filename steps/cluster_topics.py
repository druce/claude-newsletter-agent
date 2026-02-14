#!/usr/bin/env python3
"""Step 6: Cluster topics — HDBSCAN clustering + Claude-powered topic naming.

Run via: python steps/cluster_topics.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from config import NEWSAGENTDB
from db import Article
from lib.cluster import do_clustering
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _articles_to_dataframe(db_path: str) -> pd.DataFrame:
    """Load all successful articles from DB into a DataFrame for clustering."""
    articles = Article.get_all(db_path)
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        if a.status != "success":
            continue
        rows.append({
            "db_id": a.id,
            "final_url": a.final_url,
            "title": a.title,
            "summary": a.summary or a.rss_summary or "",
            "short_summary": a.short_summary or "",
            "description": a.description or "",
            "topics": a.topics or "",
            "rating": a.rating,
            "source": a.source,
        })
    return pd.DataFrame(rows)


async def cluster_topics_action(state: NewsletterAgentState) -> str:
    """Load articles, run clustering pipeline, update DB and state."""
    Article.create_table(state.db_path)

    df = _articles_to_dataframe(state.db_path)
    if df.empty:
        return "No articles to cluster"

    clustered_df = await do_clustering(df)

    # Update Article records with cluster data
    for _, row in clustered_df.iterrows():
        db_id = int(row["db_id"])
        article = Article.get(state.db_path, db_id)
        if article is None:
            continue
        article.cluster_label = str(row.get("cluster_label", -1))
        article.update(state.db_path)

    # Update state with cluster metadata
    if "cluster_name" in clustered_df.columns:
        unique_names = sorted(
            clustered_df.loc[
                clustered_df["cluster_name"] != "Other", "cluster_name"
            ].unique().tolist()
        )
        state.cluster_names = unique_names

        # Build clusters dict: {cluster_name: [url1, url2, ...]}
        clusters = {}
        for name in unique_names:
            mask = clustered_df["cluster_name"] == name
            clusters[name] = clustered_df.loc[mask, "final_url"].tolist()
        state.clusters = clusters

    n_clusters = len(state.cluster_names)
    return f"Clustered {len(clustered_df)} articles into {n_clusters} topics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Cluster topics")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="cluster_topics",
        session_id=args.session,
        db_path=args.db,
        action=cluster_topics_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
