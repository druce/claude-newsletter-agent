# Phase 2: LLM Layer Design

## Overview

`llm.py` — a multi-vendor LLM wrapper supporting Anthropic, OpenAI, and Gemini. Provides structured output, batch DataFrame processing, retry logic, and a unified reasoning-effort interface.

## Class Hierarchy

```
LLMModel (dataclass)           — model metadata + capabilities
LLMAgent (ABC)                 — vendor-agnostic base with shared logic
├── AnthropicAgent             — Anthropic SDK (tool_use structured output)
├── OpenAIAgent                — OpenAI SDK (json_schema structured output)
└── GeminiAgent                — Google GenAI SDK (response_schema structured output)
```

## LLMModel

Dataclass describing a model's identity and capabilities:

```python
@dataclass(frozen=True)
class LLMModel:
    model_id: str               # e.g. "claude-sonnet-4-5-20250929"
    vendor: Vendor              # enum: anthropic, openai, gemini
    supports_logprobs: bool     # native logprob support
    supports_reasoning: bool    # reasoning effort / thinking budget
    default_max_tokens: int     # default response token limit
    display_name: str           # human-readable name
```

Pre-defined instances:

| Constant | Model ID | Vendor | Logprobs | Reasoning |
|----------|----------|--------|----------|-----------|
| CLAUDE_SONNET | claude-sonnet-4-5-20250929 | anthropic | no | yes |
| CLAUDE_HAIKU | claude-haiku-4-5-20251001 | anthropic | no | yes |
| GPT5_MINI | gpt-5-mini | openai | yes | yes |
| GPT41_MINI | gpt-4.1-mini | openai | yes | no |
| GEMINI_FLASH | gemini-2.0-flash | gemini | no | yes |
| GEMINI_PRO | gemini-2.5-pro | gemini | no | yes |

## LLMAgent (Abstract Base Class)

### Constructor

```python
LLMAgent(
    model: LLMModel,
    system_prompt: str,
    user_prompt: str,           # template with {variable} placeholders
    output_type: Type[BaseModel] | None = None,  # Pydantic model for structured output
    reasoning_effort: int = 0,  # 0-10 scale
    max_concurrency: int = 12,
    temperature: float = 0.0,
)
```

### Public Methods

1. **`prompt_dict(variables: dict) -> BaseModel | str`**
   Single prompt. Substitutes variables into user_prompt, calls LLM, returns parsed structured output or raw string.

2. **`prompt_list(items: list[dict], item_id_field="id") -> list[BaseModel]`**
   Process a list of dicts. Serializes as JSON in prompt, gets back a list of structured outputs. Validates returned IDs match sent IDs.

3. **`filter_dataframe(df, chunk_size=25, value_field="output", item_list_field="results_list", item_id_field="id") -> pd.Series`**
   Chunks DataFrame, sends each chunk via prompt_list, reassembles into a Series aligned to the original index.

4. **`run_prompt_with_probs(variables: dict, target_tokens: list[str]) -> dict[str, float]`**
   Returns token probabilities. Uses native logprobs (OpenAI) or simulated confidence scores (Anthropic, Gemini).

### Internal Methods (Abstract)

- `_call_llm(system, user, output_schema) -> raw_response` — vendor-specific API call
- `_parse_structured(raw_response, output_type) -> BaseModel` — vendor-specific parsing
- `_parse_logprobs(raw_response, target_tokens) -> dict` — vendor-specific logprob extraction
- `_fatal_exceptions: tuple[Type[Exception], ...]` — class var, no retry
- `_temporary_exceptions: tuple[Type[Exception], ...]` — class var, retry with backoff

## Reasoning Effort Mapping

Numeric 0-10 maps to vendor-specific parameters:

| Level | Label | Anthropic (thinking tokens) | OpenAI | Gemini (thinking tokens) |
|-------|-------|-----------------------------|--------|--------------------------|
| 0 | none | disabled | none | disabled |
| 2 | minimal | 1,000 | low | 1,024 |
| 4 | low | 2,000 | low | 2,048 |
| 6 | medium | 4,000 | medium | 4,096 |
| 8 | high | 8,000 | high | 8,192 |
| 10 | xhigh | 16,000 | high | 16,384 |

## Structured Output

Each vendor implements structured output differently:

- **Anthropic**: Single tool definition with JSON schema from Pydantic model. Response parsed from `tool_use` content block.
- **OpenAI**: `response_format={"type": "json_schema", "json_schema": ...}`. Response parsed from `message.content`.
- **Gemini**: `generation_config={"response_schema": ...}`. Response parsed from `response.text`.

## Logprob Handling

- **OpenAI** (native): Request `logprobs=True, top_logprobs=5`. Extract token probabilities from response.
- **Anthropic/Gemini** (simulated): When logprobs requested, inject a `confidence: float` field into the output schema. LLM returns its confidence 0.0-1.0. Normalized to same `dict[str, float]` format.

## Retry Logic

Base class uses tenacity:

```python
@retry(
    retry=retry_if_exception_type(self._temporary_exceptions),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=log_retry,
)
```

Exception classification per vendor:

| Vendor | Fatal | Temporary |
|--------|-------|-----------|
| Anthropic | AuthenticationError, BadRequestError | RateLimitError, APIStatusError(5xx), APIConnectionError |
| OpenAI | AuthenticationError, BadRequestError | RateLimitError, APIStatusError(5xx), APIConnectionError |
| Gemini | InvalidArgument, PermissionDenied | ResourceExhausted, ServiceUnavailable |

## Batch Processing Flow

```
filter_dataframe(df, chunk_size=25)
  ├── split df into chunks of 25 rows
  ├── for each chunk (bounded by asyncio.Semaphore):
  │     ├── serialize chunk rows to JSON string
  │     ├── substitute into user_prompt
  │     ├── call _call_llm()
  │     ├── parse structured list response
  │     └── validate returned IDs match sent IDs
  ├── gather all chunks concurrently
  └── reassemble into pd.Series aligned to original index
```

## Dependencies

- `anthropic` (AsyncAnthropic)
- `openai` (AsyncOpenAI)
- `google-genai` (GenAI client)
- `tenacity` (retry)
- `pydantic` (structured output schemas)
- `pandas` (DataFrame operations)

## Not Included (YAGNI)

- No Langfuse/tracing integration
- No streaming support (batch workload)
- No caching layer
- No prompt versioning system
