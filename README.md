# Finesse Benchmark: Evaluating Long-Context Embedders with Semantic Precision

## Introduction

The **Finesse Benchmark** is a sophisticated evaluation framework designed to assess the performance of long-context embedding models on semantic understanding and information retention. Unlike traditional benchmarks that rely on superficial metrics, Finesse focuses on **Relative Semantic Similarity (RSS)**—a robust metric that measures how well models distinguish between relevant ("memory") and irrelevant ("noise") chunks in long sequences.

### Key Features
- **Modular Evaluation Modes**: Supports `merger_mode` (using sequence-merger with a base embedder), `native_mode` (direct long-context embedders like Snowflake Arctic Embed), and `BYOK_mode` (Bring Your Own Keys for external APIs via LiteLLM).
- **Dynamic Probe Generation**: Creates synthetic probes from atomic text chunks in the dataset, masking portions to test reconstruction accuracy.
- **Top-Down and Bottom-Up Scoring**: Combines **Top-Down (TD)** for contextual coherence (how well the model separates memory from noise) and **Bottom-Up (BU)** for individual chunk integrity (how well each chunk recognizes itself in compositions).
- **Reproducibility and Integrity**: Outputs include self-contained content hashes and optional model hashes for notarization and verification.
- **CLI-Driven Workflow**: Simple commands (`init`, `generate`, `score`, `checksum`) for end-to-end evaluation.
- **Dataset**: Uses the [enzoescipy/finesse-benchmark-database](https://huggingface.co/datasets/enzoescipy/finesse-benchmark-database) on Hugging Face, which provides domain-diverse atomic chunks grouped by `string_id`.

Finesse is built with [Pydantic](https://pydantic-docs.helpmanual.io/) for configuration validation, [Typer](https://typer.tiangolo.com/) for CLI, and [Torch](https://pytorch.org/) for efficient embedding computations.

## Installation

Install via pip:

```bash
pip install finesse-benchmark
```

- Requires Python 3.8+.
- For GPU acceleration: Ensure CUDA is installed and set `device: "cuda"` in config.
- Hugging Face models are downloaded automatically (use `transformers` cache).

For BYOK mode (e.g., OpenAI), install additional dependencies:

```bash
pip install litellm
```

## Quick Start

### 1. Initialize Config (Optional)
Generate a default `benchmark.yaml` template:

```bash
finesse init --output benchmark.yaml
```

Edit `benchmark.yaml` to select mode, models, and probe settings. For BYOK mode, see the dedicated section below.

### 2. Generate Raw Embeddings
Run the evaluation to generate raw probe and synthesis embeddings:

```bash
finesse generate --config benchmark.yaml --output results --samples 5 --seed 42
```

- This saves a `.pt` file (e.g., `embeddings_merger_mode_finesse-benchmark-database.pt`) containing raw data and config.
- Overrides: Use `--dataset-path` for custom HF datasets, `--samples` for more evaluations per length.

### 3. Score the Embeddings
Compute RSS scores from the raw data:

```bash
finesse score --pt-path results/embeddings_merger_mode_finesse-benchmark-database.pt --output results
```

- Outputs `benchmark_results.json` with `average_rss` (final score) and per-length scores.
- Scores are normalized and scaled (multiplied by 500 for interpretability).

### 4. Verify Integrity (Checksum)
Validate the results for tampering or reproducibility:

```bash
finesse checksum --json-path results/benchmark_results.json
```

For full provenance (model unchanged), provide the model ID:

```bash
finesse checksum --json-path results/benchmark_results.json --model-path Snowflake/snowflake-arctic-embed-l-v2.0
```

- ✅ Success if content and model hashes match.
- Only Hugging Face model IDs (e.g., `org/repo`) are accepted for `--model-path`.

## Detailed CLI Reference

All commands use [Typer](https://typer.tiangolo.com/) for intuitive interfaces. Run `finesse --help` for overview.

### `finesse init`
Generates a commented `benchmark.yaml` template.

**Options**:
- `--output`: Path to save YAML (default: `benchmark.yaml`).

**Example**:
```bash
finesse init --output my_config.yaml
```

The template includes examples for all modes and validates against `BenchmarkConfig` before saving.

### `finesse generate`
Generates raw embeddings from the dataset using the specified config.

**Options**:
- `--config` (required): Path to `benchmark.yaml`.
- `--dataset-path`: Override HF dataset path (default: from config).
- `--output`: Directory for `.pt` files (default: `results`).
- `--samples`: Samples per sequence length (overrides config).
- `--seed`: Random seed for reproducibility (overrides config).

**Output**:
- `.pt` file: Torch tensor with `config` (dict), `raw_results` (embeddings per length).

**Example**:
```bash
finesse generate --config benchmark.yaml --output ./my_results --samples 10
```

### `finesse score`
Computes TD/BU scores and final RSS from raw `.pt` data.

**Options**:
- `--pt-path` (required): Path to `.pt` file from `generate`.
- `--output`: Directory for JSON (default: `results`).

**Scoring Logic** (simplified):
- **TD Score**: Quartile gap between memory and noise similarities (excludes first/last synthesis steps for stability).
- **BU Score**: Similar gap from individual chunk perspectives.
- **Final RSS**: `((avg_TD + avg_BU) / 2) - |TD - BU|` per length, averaged across lengths, scaled by 500.

**Output**:
- `benchmark_results.json`:
  ```json
  {
    "config": {...},
    "average_rss": 42.123456,
    "length_scores": {"5": 40.5, "16": 43.7},
    "content_hash": "sha256:...",
    "model_hash": "sha256:..."  // Optional, for HF models
  }
  ```

**Example**:
```bash
finesse score --pt-path results/embeddings_byok_mode_finesse-benchmark-database.pt
```

### `finesse checksum`
Verifies JSON integrity via self-contained hash. Optional model provenance check.

**Options**:
- `--json-path` (required): Path to `benchmark_results.json`.
- `--model-path`: HF model ID for provenance (e.g., `intfloat/multilingual-e5-base`).

**Verification**:
- Recomputes `content_hash` (excludes hash itself) and compares.
- For `--model-path`: Recomputes `model_hash` from model files and compares.

**Example**:
```bash
finesse checksum --json-path results/benchmark_results.json --model-path enzoescipy/sequence-merger-tiny
```

## Output Files Explained

- **`.pt` (Raw Embeddings)**: Binary Torch file with:
  - `config`: Full benchmark config as dict.
  - `raw_results`: Dict of `{length: {"probe_embeddings": [...], "synthesis_embeddings": [...], "num_synth_steps": int}}`.
  - Used as input to `score`; enables decoupling of embedding generation (GPU-heavy) from scoring (CPU-friendly).

- **`benchmark_results.json`**: Human-readable results with:
  - `average_rss`: Overall score (higher is better; >40 indicates strong performance).
  - `length_scores`: Per-sequence-length scores (tests scaling).
  - `content_hash`: SHA-256 of config + scores (for tamper-proofing).
  - `model_hash`: SHA-256 of model files (if applicable; verifies unchanged model).

Hashes ensure reproducibility: Rerun `checksum` on shared results to confirm no alterations.

## Using BYOK Mode (Bring Your Own Keys)

BYOK mode integrates external embedding APIs (e.g., OpenAI, Cohere) via [LiteLLM](https://github.com/BerriAI/litellm) for fair comparison with open models.

### Setup
1. Edit `benchmark.yaml`:
   ```yaml
   mode: "byok_mode"

   models:
     byok_embedder:
       provider: "openai"  # 'openai', 'cohere', 'google', etc.
       name: "text-embedding-3-large"  # Provider-specific model
   ```

2. Set Environment Variables (REQUIRED; never hardcode in YAML):
   - OpenAI: `export OPENAI_API_KEY="sk-..."`
   - Cohere: `export COHERE_API_KEY="..."`
   - Google: `export GOOGLE_API_KEY="..."` (or Vertex AI creds).
   - LiteLLM auto-detects based on `provider`. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list.

   On Windows (PowerShell): `$env:OPENAI_API_KEY="sk-..."`

3. Run as usual:
   ```bash
   finesse generate --config byok_config.yaml
   ```

### Notes
- Costs: BYOK incurs API fees; start with small `--samples` (e.g., 1-5).
- Security: Keys stay in env vars—YAML remains commit-safe.
- Validation: Config validator ensures `byok_embedder` is set for `byok_mode`; others are optional/ignored.
- Example YAML (in `init` template): Uncomment and customize the BYOK section.

## Configuration Deep Dive (benchmark.yaml)

- **mode**: `"merger_mode"` (default; uses merger + base), `"native_mode"` (direct embedder), `"byok_mode"`.
- **models**: Mode-specific; unused fields default to `None` (no download).
  - `merger`: Sequence-merger path (e.g., `"enzoescipy/sequence-merger-tiny"`).
  - `base_embedder`/`native_embedder`: Embedder path (e.g., `"intfloat/multilingual-e5-base"`).
- **dataset**: HF path (default: `"enzoescipy/finesse-benchmark-database"`), split (`"train"`).
- **probe_config**:
  - `mask_ratio`: 0.15 (fraction masked).
  - `sequence_length`: `{min: 5, max: 16}` (probe lengths in tokens).
  - `samples_per_length`: 1+ (evals per length).
- **advanced**: `{batch_size: 8, device: "cuda"}` (optional).
- **seed**: 42 (reproducibility).

Pydantic ensures type safety; invalid configs raise `ValueError` on load.

## Development

- Source: `src/finesse_benchmark/`.
- Tests: Run `pytest` (add tests for scoring, hashing).
- Contributing: Fork, PR with docs/tests. Focus on new providers/modes.

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built on insights from long-context evaluation research. Thanks to Hugging Face Transformers and Pydantic teams.
