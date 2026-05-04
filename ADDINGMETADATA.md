# Adding Model Metadata

This guide explains how to add metadata for a new model to the OpenHands Index Results repository. Every metadata field must be backed by an **official source** from the model's original provider.

## Source Requirements

- **Only official provider sources are accepted.** For example, use `openai.com` for GPT models, `anthropic.com` / `platform.claude.com` for Claude models, `ai.google.dev` for Gemini models, etc.
- **Aggregators are never valid sources.** Sites like OpenRouter, LiteLLM Hub, or any third-party pricing aggregator must not be used.
- For each field you add, you must find:
  1. An **official link** to the provider's documentation, blog post, or pricing page.
  2. An **exact quote** from that page that supports the value you chose.

## Step-by-Step Process

### 1. Create the Model Directory

Create a new directory under `results/` named after the model:

```
results/<ModelName>/
├── metadata.json
└── scores.json
```

The directory name must exactly match the `model` field value (e.g., `GPT-5.5`, `claude-sonnet-4-5`, `Qwen3.6-Plus`).

### 2. Create `metadata.json`

Create a `metadata.json` file with the following fields. See [Field Reference](#field-reference) below for details on sourcing each value.

**Example for a closed API model:**

```json
{
  "agent_name": "OpenHands",
  "agent_version": "v1.18.1",
  "model": "GPT-5.5",
  "openness": "closed_api_available",
  "country": "us",
  "tool_usage": "standard",
  "directory_name": "GPT-5.5",
  "release_date": "2026-04-23",
  "supports_vision": true,
  "input_price": 5.0,
  "output_price": 30.0,
  "cache_read_price": 0.5,
  "cache_write_price": null
}
```

**Example for an open-weights model:**

```json
{
  "agent_name": "OpenHands",
  "agent_version": "v1.8.3",
  "model": "DeepSeek-V3.2-Reasoner",
  "openness": "open_weights",
  "country": "cn",
  "tool_usage": "standard",
  "directory_name": "DeepSeek-V3.2-Reasoner",
  "release_date": "2025-12-01",
  "parameter_count_b": 685,
  "supports_vision": false,
  "input_price": 0.55,
  "output_price": 2.19,
  "cache_read_price": 0.14,
  "cache_write_price": null
}
```

### 3. Create `scores.json`

Create an empty `scores.json` file — benchmark results will be added in separate PRs:

```json
[]
```

### 4. Update the Validation Schema

Edit `scripts/validate_schema.py` to register the new model:

1. **Add to the `Model` enum:**

   ```python
   class Model(str, Enum):
       # ... existing entries ...
       YOUR_MODEL = "Your-Model-Name"
   ```

2. **Add to `MODEL_OPENNESS_MAP`:**

   ```python
   MODEL_OPENNESS_MAP: dict[Model, Openness] = {
       # ... existing entries ...
       Model.YOUR_MODEL: Openness.OPEN_WEIGHTS,       # or Openness.CLOSED_API_AVAILABLE
   }
   ```

3. **Add to `MODEL_COUNTRY_MAP`:**

   ```python
   MODEL_COUNTRY_MAP: dict[Model, Country] = {
       # ... existing entries ...
       Model.YOUR_MODEL: Country.US,   # or Country.CN, Country.FR
   }
   ```

### 5. Update `complete-models.json`

Add the new model entry at the top of the array (most recent first) with the current timestamp:

```json
[
  {
    "timestamp": "2026-04-29T14:48:56.000+00:00",
    "model-path": "results/Your-Model-Name"
  },
  ...
]
```

### 6. Add a Test

Add a test in `tests/test_validate_schema.py` to validate your new metadata:

```python
def test_valid_metadata_your_model(self, tmp_path):
    """Test valid metadata for Your-Model passes validation."""
    metadata = {
        "agent_name": "OpenHands",
        "agent_version": "v1.18.1",
        "model": "Your-Model-Name",
        "country": "us",
        "openness": "closed_api_available",
        "tool_usage": "standard",
        "directory_name": "Your-Model-Name",
        "release_date": "2026-01-01",
        "supports_vision": True,
        "input_price": 1.0,
        "output_price": 5.0,
        "cache_read_price": 0.1,
        "cache_write_price": None
    }
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))

    valid, msg = validate_metadata(metadata_file)
    assert valid is True
    assert msg == "OK"
```

### 7. Run Validation

Before submitting, run both the schema validation and the test suite:

```bash
python scripts/validate_schema.py
pytest tests/ -v
```

All tests must pass.

---

## Field Reference

Every field listed below must be sourced from the model's **official provider**. The table summarizes what each field is, where to find the value, and whether it is required.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_name` | string | Yes | Always `"OpenHands"` for main results (or `"Claude Code"`, `"Codex"`, etc. for alternative agents). |
| `agent_version` | string | Yes | Semantic version of the OpenHands SDK (or alternative agent) used for the runs, e.g. `"v1.18.1"`. **Required for consistency enforcement:** the schema validates that this value is ≤ the `agent_version` of every score entry in `scores.json`, ensuring all runs of the same model use the same or a newer agent. Set it to the earliest SDK version used across all runs for this model. |
| `model` | string | Yes | Model name, must match one of the `Model` enum values in `validate_schema.py`. |
| `openness` | string | Yes | One of: `"closed_api_available"`, `"open_weights"`, `"closed"`. |
| `country` | string | Yes | Country of origin: `"us"`, `"cn"`, or `"fr"`. |
| `tool_usage` | string | Yes | Usually `"standard"`. |
| `directory_name` | string | Yes | Must exactly match the `model` field value. |
| `release_date` | string | Yes | `YYYY-MM-DD` format. The date the model was officially released or made available. |
| `supports_vision` | boolean | Yes | Whether the model accepts image/vision inputs. |
| `parameter_count_b` | number | Conditional | Total parameter count in billions. **Required for `open_weights` models.** Set to `null` for closed models. |
| `active_parameter_count_b` | number | No | Active parameters per token in billions (relevant for Mixture-of-Experts models). |
| `input_price` | number | Yes | Price per 1 million input tokens in USD. Must be > 0. |
| `output_price` | number | Yes | Price per 1 million output tokens in USD. Must be > 0. |
| `cache_read_price` | number | No | Price per 1 million cached input tokens in USD. Set to `null` if not supported. |
| `cache_write_price` | number | No | Price per 1 million cache-write tokens in USD. Set to `null` if not supported. |
| `hide_from_leaderboard` | boolean | No | Defaults to `false`. Set to `true` to hide the model from the public leaderboard. |

### How to Source Each Field

#### `model`

- **Where to look:** The provider's official model page, API docs, or release announcement.
- **Example source:** `https://openai.com/index/gpt-5-5/` — *"Introducing GPT-5.5"*

#### `release_date`

- **Where to look:** The official announcement blog post or API changelog from the provider.
- **Example source:** `https://openai.com/index/gpt-5-5/` — *"April 23, 2026"*
- **Format:** `YYYY-MM-DD`

#### `openness`

- **Where to look:** The provider's model page or license information.
- **Values:**
  - `"closed_api_available"` — Model is only accessible via a paid API (e.g., GPT, Claude, Gemini Pro).
  - `"open_weights"` — Model weights are publicly downloadable (e.g., on Hugging Face) under an open license.
  - `"closed"` — Model has no public API access.
- **Example source:** `https://huggingface.co/deepseek-ai/DeepSeek-V3.2` — *"License: MIT"*

#### `country`

- **Where to look:** The provider's official "About" page or company registration.
- **Values:** `"us"` (United States), `"cn"` (China), `"fr"` (France).
- **Example providers:**
  - `"us"`: OpenAI, Anthropic, Google, NVIDIA
  - `"cn"`: DeepSeek, Alibaba/Qwen, Zhipu/GLM, Moonshot/Kimi, MiniMax
  - `"fr"`: Mistral

#### `supports_vision`

- **Where to look:** The model's API documentation or model card from the official provider.
- **Example source:** `https://platform.openai.com/docs/models/gpt-5-5` — *"Supports text and image inputs"*

#### `parameter_count_b`

- **Where to look:** Official model card, technical report, or announcement blog.
- **Required for** `open_weights` **models.** Should be `null` or omitted for closed models if not disclosed.
- **Example source:** `https://huggingface.co/deepseek-ai/DeepSeek-V3.2` — *"685B total parameters"*

#### `active_parameter_count_b`

- **Where to look:** Official technical report or model card (for MoE architectures).
- **Example source:** `https://huggingface.co/Qwen/Qwen3-Coder-480B` — *"35B active parameters per token"*

#### `input_price` / `output_price`

- **Where to look:** The provider's **official pricing page**.
- **Example sources:**
  - OpenAI: `https://openai.com/api/pricing/`
  - Anthropic: `https://platform.claude.com/docs/en/about-claude/pricing`
  - Google: `https://ai.google.dev/pricing`
  - Alibaba/Qwen: `https://www.alibabacloud.com/help/en/model-studio/models`
  - DeepSeek: `https://api-docs.deepseek.com/quick_start/pricing`
- **Unit:** USD per 1 million tokens.

#### `cache_read_price` / `cache_write_price`

- **Where to look:** The provider's **official pricing page** under prompt caching or context caching.
- Set to `null` if the provider does not offer prompt caching for this model.
- **Example source:** `https://platform.claude.com/docs/en/about-claude/pricing` — *"Prompt caching: Write $3.75 / MTok, Read $0.30 / MTok"*

---

## PR Description Requirements

When submitting a PR that adds metadata, the PR description **must** include evidence for each field. For every metadata field, provide:

1. **The exact value** you are setting.
2. **An exact quote** from the official source that supports the value.
3. **The exact URL** of the official source.

**Do not use aggregator sites** (e.g., OpenRouter, LiteLLM, or third-party pricing trackers) as sources. Only the original model provider's pages are accepted.

### PR Description Template

Use this structure in your PR description:

```markdown
## Summary

Added <Model-Name> model metadata to the repository.

## Field-by-Field Evidence

### model: "<Model-Name>"
- **Value**: `<Model-Name>`
- **Quote**: "<exact quote from official source>"
- **Source**: <official URL>

### release_date: "YYYY-MM-DD"
- **Value**: `YYYY-MM-DD`
- **Quote**: "<exact quote confirming the release date>"
- **Source**: <official URL>

### openness: "<value>"
- **Value**: `<value>`
- **Quote**: "<exact quote about model availability/license>"
- **Source**: <official URL>

### country: "<value>"
- **Value**: `<value>`
- **Quote**: "<exact quote confirming company origin>"
- **Source**: <official URL>

### supports_vision: <true/false>
- **Value**: `<true/false>`
- **Quote**: "<exact quote about input modalities>"
- **Source**: <official URL>

### parameter_count_b: <value>  (if applicable)
- **Value**: `<value>`
- **Quote**: "<exact quote about parameter count>"
- **Source**: <official URL>

### active_parameter_count_b: <value>  (if applicable)
- **Value**: `<value>`
- **Quote**: "<exact quote about active parameters>"
- **Source**: <official URL>

### input_price: <value>
- **Value**: `<value>`
- **Quote**: "<exact quote from official pricing page>"
- **Source**: <official URL>

### output_price: <value>
- **Value**: `<value>`
- **Quote**: "<exact quote from official pricing page>"
- **Source**: <official URL>

### cache_read_price: <value or null>
- **Value**: `<value or null>`
- **Quote**: "<exact quote or 'Not supported/mentioned in official docs'>"
- **Source**: <official URL>

### cache_write_price: <value or null>
- **Value**: `<value or null>`
- **Quote**: "<exact quote or 'Not supported/mentioned in official docs'>"
- **Source**: <official URL>

## Changes Made

1. Created `results/<Model-Name>/metadata.json`
2. Created `results/<Model-Name>/scores.json` (empty array)
3. Updated `scripts/validate_schema.py` (Model enum, MODEL_OPENNESS_MAP, MODEL_COUNTRY_MAP)
4. Updated `complete-models.json`
5. Added test in `tests/test_validate_schema.py`

## Verification

- All tests pass
- Schema validation passes

Fixes #<issue-number>
```

### Example: Good vs. Bad Sources

| Field | ✅ Good Source | ❌ Bad Source |
|-------|---------------|--------------|
| Pricing | `https://openai.com/api/pricing/` | `https://openrouter.ai/models/openai/gpt-5.5` |
| Release date | `https://blog.google/technology/ai/gemini-3/` | `https://en.wikipedia.org/wiki/Gemini_(language_model)` |
| Parameters | `https://huggingface.co/deepseek-ai/DeepSeek-V3.2` | `https://artificialanalysis.ai/models/deepseek-v3-2` |
| Vision support | `https://platform.openai.com/docs/models/gpt-5-5` | `https://docs.litellm.ai/docs/providers/openai` |

---

## Quick Checklist

Before submitting your PR, verify:

- [ ] `metadata.json` contains all required fields
- [ ] `scores.json` exists (empty array `[]` is fine)
- [ ] Model added to `Model` enum in `scripts/validate_schema.py`
- [ ] Model added to `MODEL_OPENNESS_MAP` in `scripts/validate_schema.py`
- [ ] Model added to `MODEL_COUNTRY_MAP` in `scripts/validate_schema.py`
- [ ] Model added to `complete-models.json` (at the top, most recent first)
- [ ] Test added in `tests/test_validate_schema.py`
- [ ] `python scripts/validate_schema.py` passes
- [ ] `pytest tests/ -v` passes
- [ ] PR description includes exact value, exact quote, and exact official link for every field
- [ ] All sources are from the **official provider** (no aggregators)
