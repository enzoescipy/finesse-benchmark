# Finesse Benchmark: Evaluating Long-Context Embedders with Semantic Precision

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/enzoescipy/finesse-benchmark)
[![PyPI](https://img.shields.io/badge/PyPI-Package-green?logo=pypi)](https://pypi.org/project/finesse-benchmark/)
[![Blog](https://img.shields.io/badge/Blog-Article-orange?logo=medium)](https://www.winter-sci-dev.com/posts/embed-sequence-merger-vbert-ppe-article/)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/spaces/enzoescipy/finesse-benchmark-space)

## Introduction

The **Finesse Benchmark** is a sophisticated evaluation framework designed to assess the performance of long-context embedding models on semantic understanding and information retention. Unlike traditional benchmarks that rely on superficial metrics, Finesse focuses on **Relative Semantic Similarity (RSS)**—a robust metric that measures how well models distinguish between relevant ("memory") and irrelevant ("noise") chunks in long sequences.

### Key Features
- **Modular Evaluation Modes**: Supports `merger_mode` (using sequence-merger with a base embedder), `native_mode` (direct long-context embedders like Snowflake Arctic Embed), and `BYOK_mode` (Bring Your Own Keys for external APIs via LiteLLM).
- **Dynamic Probe Generation**: Creates synthetic probes from atomic text chunks in the dataset, masking portions to test reconstruction accuracy.
- **Top-Down and Bottom-Up Scoring**: Combines **Top-Down (TD)** for contextual coherence (how well the model separates memory from noise) and **Bottom-Up (BU)** for individual chunk integrity (how well each chunk recognizes itself in compositions).
- **Latency Measurement**: Tracks computational efficiency by measuring `total_latency` (full embedding process time in milliseconds) and `synthesis_latency` (merging-specific time in milliseconds), enabling analysis of accuracy versus speed trade-offs.
- **Reproducibility and Integrity**: Outputs include self-contained content hashes and optional model hashes for notarization and verification.
- **CLI-Driven Workflow**: Simple commands (`init`, `generate`, `score`, `checksum`) for end-to-end evaluation.
- **Dataset**: Uses the [enzoescipy/finesse-benchmark-database](https://huggingface.co/datasets/enzoescipy/finesse-benchmark-database) on Hugging Face, which provides domain-diverse atomic chunks grouped by `string_id`.

- **Results Repository**: All official and community-submitted benchmark results, including JSON reports and .pt files, are hosted and versioned in the [enzoescipy/finesse-benchmark-results](https://huggingface.co/datasets/enzoescipy/finesse-benchmark-results) dataset on Hugging Face. This serves as the central hub for leaderboard updates and comparative analysis. results are shared on the [enzoescipy/finesse-benchmark-space](https://huggingface.co/spaces/enzoescipy/finesse-benchmark-space).

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

For official leaderboard submissions, use the standardized config:

```bash
finesse init --leaderboard
```

This copies the official `benchmark.leaderboard.yaml` to `benchmark.yaml`, ensuring reproducibility with fixed seed (42), balanced sequence lengths (4-32), and samples per length (25).

Alternatively, manually copy:

```bash
cp "benchmark.leaderboard.yaml" "benchmark.yaml"
```

**Note:** For leaderboard, use the official config as-is without modifications to maintain fairness. Edit only for custom evaluations.

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
Generates a commented `benchmark.yaml` template or copies the official leaderboard config.

**Options**:
- `--leaderboard`: Use official leaderboard configuration (copies `benchmark.leaderboard.yaml` to output; default: False).
- `--output`: Path to save YAML (default: `benchmark.yaml`).

**Examples**:
- Default template:
  ```bash
  finesse init --output my_config.yaml
  ```
- Leaderboard config:
  ```bash
  finesse init --leaderboard
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
    "average_rss": 68.984345,
    "average_total_latency": 52.576509,
    "average_synthesis_latency": 52.576509,
    "length_scores": {
            "4": {
                "rss_scores": [
                    123.456,
                    234.567,
                ],
                "total_latency_scores": [
                    12.34,
                    56.78,
                ],
                "synthesis_latency_scores": [
                    1.23,
                    4.56,
                ]
            }
            "5" : ...
    }
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
  - `length_scores`: Per-sequence-length scores (tests scaling).
  - `rss_scores`: Overall score (higher is better; -1000 ~ 1000 range).
  - `total_latency_scores`: Total time (in milliseconds) for the entire embedding generation process across all samples.
  - `synthesis_latency_scores`: Time (in milliseconds) specifically for the synthesis/merging steps across all samples.
  - `content_hash`: SHA-256 of config + scores (for tamper-proofing).
  - `model_hash`: SHA-256 of model files (if applicable; verifies unchanged model).

Hashes ensure reproducibility: Rerun `checksum` on shared results to confirm no alterations. Users can upload their `benchmark_results.json` to the [enzoescipy/finesse-benchmark-space](https://huggingface.co/spaces/enzoescipy/finesse-benchmark-space) for community visibility and leaderboard integration.

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
  - `sequence_length`: `{min: 5, max: 16}` (probe lengths in tokens).
  - `samples_per_length`: 1+ (evals per length).
- **advanced**: `{batch_size: 8, device: "cuda"}` (optional).
- **seed**: 42 (reproducibility).

Pydantic ensures type safety; invalid configs raise `ValueError` on load.

## Leaderboard Submission Guide

To ensure fair, reproducible, and standardized evaluations for the official Finesse leaderboard:

1. **Initialize Official Config**:
   Use the CLI for convenience:
   ```bash
   finesse init --leaderboard
   ```
   This copies `benchmark.leaderboard.yaml` to `benchmark.yaml` with fixed settings (seed: 42, sequence_length: {min: 4, max: 32}, samples_per_length: 25).
   
   Or manually (for scripts/environments without CLI):
   ```bash
   cp "benchmark.leaderboard.yaml" "benchmark.yaml"
   ```
   
   **Important:** Do not modify this config for leaderboard submissions. Use as-is for comparability.

2. **Run Evaluation**:
   Generate embeddings:
   ```bash
   finesse generate --config benchmark.yaml --output results
   ```
   Score results:
   ```bash
   finesse score --pt-path results/embeddings_merger_mode_finesse-benchmark-database.pt --output results
   ```

3. **Verify & Submit**:
   Check integrity:
   ```bash
   finesse checksum --json-path results/benchmark_results.json

   finesse checksum --json-path results/benchmark_results.json --merger-path enzoescipy/sequence-merger-malgeum --base-embedder-path intfloat/multilingual-e5-base

   finesse checksum --json-path results/benchmark_results.json --native-path Snowflake/snowflake-arctic-embed-l-v2.0

   finesse verify --json-path results/benchmark_results.json --pt-path results/embeddings_merger_mode_finesse-benchmark-database.pt
   ```
   Submit `benchmark_results.json` to the leaderboard (via HF Spaces, GitHub, etc.). Include `content_hash` and `model_hash` for verification.

This setup guarantees all submissions use identical dataset sampling, probe generation, and randomness, focusing purely on model performance.

## Using Custom Local Synthesizers with Finesse Benchmark

The `finesse-benchmark` package follows a **dependency injection** design pattern, allowing flexible substitution of core components: the **embedder** (for generating partial embeddings) and the **synthesizer** (for merging sequences). This enables users to:

- Use standard Hugging Face models for embedding (e.g., via `HuggingFaceEmbedder`).
- Swap in custom synthesizers for local models, such as trained checkpoints (`.pt` files) from your own training pipeline.
- Ensure seamless integration with the `FinesseEvaluator` for benchmark scoring without modifying the package core.

This approach promotes modularity: you define your embedder and synthesizer engines, inject them into the evaluator, and run standardized probes to measure **Relative Sequence Strength (RSS)** scores. It's tamper-proof, scalable, and leaderboard-ready.

## Custom Synthesizer Recipe: LocalCheckpointSynthesizer

Below is the complete, battle-tested code for `LocalCheckpointSynthesizer`. It inherits from `FinesseSynthesizer` and handles dynamic architecture detection (Micro/Macro/Massive/ResidualCorrectionMoe) from checkpoint metadata.

Place this in a script (e.g., `evaluator_custom.py`) or your project module.

```python
import os
import torch
from finesse_benchmark.interfaces import FinesseSynthesizer
class LocalCheckpointSynthesizer(FinesseSynthesizer):
    def __init__(self, checkpoint_path: str, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()  # Call parent init if needed
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == 'cuda' else torch.float32)
        self.model = self._load_model()

    def _load_model(self):
        model = model.to(device=self.device, dtype=self.dtype)
        return model.eval()

    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on correct device and dtype
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)

        # Check sequence length (handle short/empty sequences)
        if embeddings.shape[1] <= 1:
            # Return as-is for single/empty sequences (Finesse expects identity transformation)
            return embeddings[0]

        with torch.no_grad():
            outputs = self.model(embeddings)
            if hasattr(outputs, 'pooler_output'):
                synthesized = outputs.pooler_output
            else:
                synthesized = outputs  # Direct output, assuming (B, D)

        # Normalize output on device
        synthesized = torch.nn.functional.normalize(synthesized, p=2, dim=1)
        return synthesized
    
    def device(self) -> torch.device:
        """Return the device's type for internal use."""
        return self.device
```

**Notes on the Recipe**:
- **Dynamic Detection**: Relies on `'model_architecture'` key in your `.pt` file's metadata (added during training, e.g., via `torch.save({'model_architecture': 'MicroTransformerSynthesizer', ...}`).
- **Error Handling**: Falls back to `MicroTransformerSynthesizer` if architecture is unknown.
- **Performance**: Uses `.eval()` and `no_grad()` for efficient inference.
- **Dependencies**: Requires your custom model classes (e.g., in `models/synthesizer.py`). Adjust imports as needed.

## Step-by-Step Usage Guide

Follow these steps to integrate your custom synthesizer into a full benchmark evaluation.

### 1. Setup Environment and Config
- Install `finesse-benchmark`: `pip install finesse-benchmark`
- Prepare your `.pt` checkpoint file (must include `'model_architecture'` and `'model_state_dict'`).
- Create as `finesse init` or use a `benchmark.yaml` config file (standard Finesse format).

Example `benchmark.yaml` snippet:
```yaml
probe_config:
  sequence_length:
    min: 4
    max: 16
    step: 4
models:
  base_embedder:
    name: "intfloat/multilingual-e5-base"  # Or your preferred embedder model
```

### 2. Define Device and dtype
Unify settings to avoid runtime errors (e.g., dtype mismatches).

```python
import torch

# Auto-detect device and set dtype
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == 'cuda' else torch.float32

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")
```

### 3. Import Modules and Load Config
```python
import yaml
from pathlib import Path

from finesse_benchmark.config import BenchmarkConfig
from finesse_benchmark.implementations import HuggingFaceEmbedder
from finesse_benchmark.evaluator import FinesseEvaluator

# Load config
with open("benchmark.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
config = BenchmarkConfig.model_validate(config_dict)
```

### 4. Instantiate Embedder and Custom Synthesizer
Inject unified device/dtype into both components.

```python
# Embedder: Standard Hugging Face model
embedder = HuggingFaceEmbedder(
    model_path=config.models.base_embedder.name,  # e.g., "BAAI/bge-base-en-v1.5"
    device=device,
    dtype=dtype
)
print(f"Loaded embedder: {config.models.base_embedder.name}")

# Custom Synthesizer: Your local .pt checkpoint
checkpoint_path = Path("checkpoints/your_model.pt")  # Full path to .pt file
synthesizer = LocalCheckpointSynthesizer(
    checkpoint_path=str(checkpoint_path),
    device=device,
    dtype=dtype
)
print(f"Loaded synthesizer: {checkpoint_path.name}")
```

### 5. Create FinesseEvaluator and Run Benchmark
Inject your custom engines and execute the evaluation.

```python
# Initialize evaluator with injected engines
evaluator = FinesseEvaluator(
    embedder_engine=embedder,
    synthesizer_engine=synthesizer,
    config=config
)

# Run raw evaluation
print("  Generating raw embeddings...")
try:
    raw_data = evaluator.raw_run()
    print("  Raw embeddings generated successfully")
except Exception as e:
    print(f"❌ Error in raw_run: {e}")
    print(traceback.format_exc())
    continue

# Extract length_results for scoring
length_results = raw_data.get('raw_results', {}).get('length_results', {})
if not length_results:
    print("  No length results found. Skipping scoring.")
    continue

from finesse_benchmark.scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

final_scores_per_length = {}
for target_length, raw in length_results.items():
    sample_results = raw.get('sample_results', [])
    if not sample_results:
        final_scores_per_length[target_length] = 0.0
        continue

    sample_scores = []
    for sample_dict in sample_results:
        probe_embeddings = sample_dict.get('chunk_embeddings')
        synthesis_embeddings = sample_dict.get('synthesis_embeddings')

        if probe_embeddings and synthesis_embeddings and len(probe_embeddings) >= 2:
            td_scores = calculate_self_attestation_scores(probe_embeddings, synthesis_embeddings)
            bu_scores = calculate_self_attestation_scores_bottom_up(probe_embeddings, synthesis_embeddings)

            avg_td = td_scores['contextual_coherence']
            avg_bu = bu_scores['bottom_up_coherence']
            imbalance = abs(avg_td - avg_bu)
            final_score = ((avg_td + avg_bu) / 2) - imbalance
            sample_scores.append(final_score)
        else:
            sample_scores.append(0.0)

    avg_length_score = np.mean(sample_scores) if sample_scores else 0.0
    final_scores_per_length[target_length] = round(avg_length_score * 500, 6)  # Scale by 500

# Calculate average RSS
avg_rss = round(np.mean(list(final_scores_per_length.values())), 6)
print(f"  Average RSS: {avg_rss}")
```

## License

Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built on insights from long-context evaluation research. Thanks to Hugging Face Transformers and Pydantic teams.