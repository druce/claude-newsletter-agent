# Phase 3: Library Modules — Design Document

## Summary

Rewrite five library modules in `lib/` that provide the data-processing backbone for the newsletter agent. Each module is a clean rewrite of its original from the OpenAIAgentsSDK, using the new `llm.py`, `db.py`, and `config.py` infrastructure built in Phases 1-2.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-large` | Best-in-class quality; preserves pretrained UMAP reducer compatibility |
| Browser automation | Camoufox (replaces playwright-stealth) | 0% headless detection; beats Akamai/Kasada on Bloomberg/NYT; Playwright API-compatible |
| Bradley-Terry battles | Enabled | User wants full ranking pipeline |
| HDBSCAN optimization | Optuna, 50 trials + pretrained UMAP | Adaptive to varying data; proven approach |
| Token counting | tiktoken | Accurate for OpenAI embeddings; transitive dependency |
| Sources | Copy sources.yaml as-is (19 sources) | No changes needed |
| Prompt storage | `prompts.py` with templates + config | Replaces Langfuse; all models Sonnet 4.5 |
| Module structure | 5 modules mirroring CC.md (Approach A) | 1:1 mapping with originals; easy to verify |

## Module Designs

### 1. lib/scrape.py — Camoufox Browser Automation

**Responsibility:** Low-level browser automation. Given a URL, fetch the page, save HTML, extract text, return metadata.

**Key classes/functions:**

- `RateLimiter` — per-domain throttling with asyncio.Lock (ported from original)
- `ScrapeResult` dataclass — `html_path, text_path, final_url, last_updated, status`
- `async get_browser(user_data_dir) → BrowserContext` — Camoufox persistent context with module-level caching + lock
- `async scrape_url(url, title, browser_context, ...) → ScrapeResult` — fetch single URL, save to `download/html/`, extract text to `download/text/` via trafilatura
- `async scrape_urls_concurrent(urls, concurrency, rate_limit_seconds)` — batch scrape with semaphore + rate limiter
- `sanitize_filename(name) → str` — safe filesystem names
- `normalize_html(html_path) → str` — trafilatura text extraction
- `clean_url(url) → str` — strip tracking params

**Changes from original:**
- Camoufox replaces `playwright-stealth` + raw Firefox launch
- `ScrapeResult` dataclass instead of tuple return
- Per-domain daily cap (configurable via `DOMAIN_DAILY_CAP`)
- Graceful bot-block detection: log + skip instead of retry
- `trunc_tokens` moved to dedupe.py; `get_og_tags` and `parse_source_file` moved to fetch.py

**Dependencies:** `camoufox`, `trafilatura`, `aiohttp`, `tldextract`

---

### 2. lib/fetch.py — Source Processor

**Responsibility:** High-level orchestrator that loads `sources.yaml`, dispatches to the right fetcher (RSS/HTML/REST), returns standardized headline dicts.

**Key classes/functions:**

- `Fetcher` class with async context manager
  - `__init__(sources_file, max_concurrent)` — loads sources.yaml, creates aiohttp session
  - `async __aenter__/__aexit__` — manages aiohttp session + Camoufox browser lifecycle
  - `async fetch_rss(source_config) → list[dict]` — feedparser
  - `async fetch_html(source_key, source_config) → list[dict]` — uses scrape.py for landing page, then `parse_source_links()` for headline extraction
  - `async fetch_api(source_key, source_config) → list[dict]` — NewsAPI
  - `async fetch_all() → dict[str, list[dict]]` — concurrent fetch from all sources
- `parse_source_links(html_path, source_config) → list[dict]` — extract links from saved HTML using include/exclude regex patterns
- `get_og_tags(html_path) → dict` — OpenGraph metadata extraction

**Standardized result format:**
```python
{"source": str, "title": str, "url": str, "published": str|None, "rss_summary": str|None}
```

**Changes from original:** Cleaner class interface; `parse_source_file` renamed to `parse_source_links` and moved here from scrape.py; `extract_newsapi()` folded into `fetch_api()`.

**Dependencies:** `feedparser`, `aiohttp`, `pyyaml`, `beautifulsoup4`, `re`

---

### 3. lib/dedupe.py — Duplicate Detection

**Responsibility:** Find and remove near-duplicate articles using embedding cosine similarity.

**Key functions:**

- `async get_embeddings_batch(texts, model, batch_size=25) → list[list[float]]` — batched OpenAI embedding calls
- `read_and_truncate_files(df, max_tokens=8192) → pd.DataFrame` — read text from `text_path`, truncate with tiktoken
- `create_similarity_matrix(embeddings, index) → pd.DataFrame` — pairwise cosine similarity
- `find_duplicate_pairs(similarity_df, threshold=0.925) → list[tuple]` — pairs above threshold
- `filter_duplicates(df, pairs) → pd.DataFrame` — keep article with higher `content_length`
- `async process_dataframe_with_filtering(df, threshold=0.925) → pd.DataFrame` — top-level pipeline

**Changes from original:** Removed `cluster_summaries`, `nearest_neighbor_sort`, `make_bullet` (newsletter formatting, not dedup). Removed HDBSCAN dependency from this module. Single responsibility: find and remove duplicates.

**Dependencies:** `openai`, `tiktoken`, `numpy`, `pandas`, `sklearn.metrics.pairwise`

---

### 4. lib/rating.py — Article Scoring + Bradley-Terry Battles

**Responsibility:** Score articles with composite formula + rank via BT pairwise battles.

**Composite rating formula:**
```
rating = reputation + length_score + on_topic + importance - low_quality + recency + bt_zscore
```

**Key functions:**

- `async rate_articles(df, db_path) → pd.DataFrame` — top-level pipeline
- `compute_recency_score(published_date) → float` — half-life 1 day: `2 * exp(-ln2 * age) - 1`
- `compute_length_score(content_length) → float` — `log10(length) - 3`, clipped [0, 2]
- `async assess_quality(df) → pd.Series` — LLM low-quality probability
- `async assess_on_topic(df) → pd.Series` — LLM AI-relevance probability
- `async assess_importance(df) → pd.Series` — LLM importance probability

**Bradley-Terry battle system:**
- `async run_bradley_terry(df, max_rounds=8, batch_size=6) → pd.Series` — z-scored BT ratings
- `swiss_pairing(df, battle_history) → list[tuple[id, id]]`
- `async swiss_batching(df, battle_history, batch_size) → list[list[dict]]`
- `async process_battle_round(df, batches, agent) → list[tuple[winner, loser]]`
- Convergence: avg rank change < `n_stories / 100`, min rounds = `max_rounds // 2`
- Final scores via `choix.opt_pairwise()`, z-score normalized

**LLM integration:** Uses `llm.create_agent()` with `run_prompt_with_probs()` for probability extraction. Battle rounds use `prompt_list()`. All prompts from `prompts.py`.

**Dependencies:** `choix`, `numpy`, `pandas`, `scipy.stats`, `math`, `datetime`

---

### 5. lib/cluster.py — HDBSCAN Clustering + Topic Naming

**Responsibility:** Embed articles, reduce with pretrained UMAP, optimize HDBSCAN via Optuna, name clusters with Claude.

**Key functions:**

- `async do_clustering(df, umap_reducer_path, n_trials=50) → pd.DataFrame` — top-level pipeline, returns df with `cluster_label` and `cluster_name`
- `async get_embeddings_df(df, model) → pd.DataFrame` — embeddings from extended summary, batched (100 items)
- `load_umap_reducer(path) → umap.UMAP` — load pretrained reducer (3072 → 690 dims)
- `_create_extended_summary(row) → str` — title + description + topics + summary
- `_create_short_summary(row) → str` — short_summary + topics
- `optimize_hdbscan(embeddings_array, n_trials=50) → dict` — Optuna optimization
- `objective(trial, embeddings_array) → float` — 50% silhouette + 50% validity index
- `calculate_clustering_metrics(embeddings, labels, clusterer) → dict`
- `async name_clusters(df) → pd.DataFrame` — Claude names each cluster via `topic_writer` prompt

**File dependency:** Requires `umap_reducer.pkl` copied from original repo.

**Dependencies:** `hdbscan`, `umap-learn`, `optuna`, `scikit-learn`, `openai`, `numpy`, `pandas`, `pickle`

---

## prompts.py

All LLM prompts stored in a single `prompts.py` file. Each prompt is a dataclass with:
- `name` — identifier (e.g. `"battle_prompt"`)
- `system_prompt` — system prompt template
- `user_prompt` — user prompt template with `{variable}` placeholders
- `model` — LLMModel reference (all Sonnet 4.5)
- `reasoning_effort` — 0-10 scale based on prompt complexity

Reasoning effort mapping (all Sonnet 4.5):
- 0: trivial lookup (sitename)
- 2: simple binary classification (filter_urls, headline_classifier, extract_topics, canonical_topic, topic_writer)
- 4: moderate analysis (rate_quality, rate_on_topic, rate_importance, battle_prompt, dedupe_articles, extract_summaries, item_distiller, topic_cleanup, cat_assignment, cat_cleanup, critique_section)
- 6: complex generation (cat_proposal, critique_newsletter, generate_newsletter_title, write_section)
- 8: heavy editorial (draft_newsletter, improve_newsletter)

---

## Cross-cutting Concerns

**New config.py constants:**
```python
DOMAIN_DAILY_CAP = 50
SLEEP_TIME = 5
EMBEDDING_MODEL = "text-embedding-3-large"
SIMILARITY_THRESHOLD = 0.925
MAX_EMBED_TOKENS = 8192
MIN_COMPONENTS = 20
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
```

**New dependencies (requirements.txt):**
```
camoufox          # replaces playwright-stealth
beautifulsoup4    # HTML parsing
tiktoken          # token counting
optuna            # HDBSCAN optimization
```

**Remove:** `playwright-stealth`

**Copy from original repo:**
- `sources.yaml` (19 sources, verbatim)
- `umap_reducer.pkl` (pretrained UMAP model)

**Data flow:**
```
sources.yaml → fetch.py → list[dict] headlines
                  ↓ (uses scrape.py for HTML sources)
              scrape.py → download/html/, download/text/

headlines → db.Url (insert) → dedupe.py → filtered df
         → rating.py → df with ratings
         → cluster.py → df with cluster labels + names

All modules read/write via db.py models (Article, Url, Site)
All LLM calls via llm.py (create_agent) with prompts from prompts.py
All config via config.py constants
```

**Testing strategy:**
- Each module gets `tests/test_<module>.py`
- Mock external calls (Camoufox, OpenAI embeddings, aiohttp, LLM agents)
- Integration tests with small fixture data
- `pytest-asyncio` for async tests

**Error handling:** Each module's top-level function catches exceptions, logs them, returns partial results where possible. Errors logged at WARNING/ERROR level, never silently swallowed.
