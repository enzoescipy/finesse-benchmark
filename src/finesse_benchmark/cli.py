import os
import json
import yaml
from typing import Optional
import typer
import torch
import numpy as np
import click
from .utils import get_content_hash, get_model_hash

from .config import BenchmarkConfig
from .evaluator import FinesseEvaluator
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

app = typer.Typer(no_args_is_help=True)

@app.command("generate")
def generate_raw_data(
    config_path: str = typer.Option(..., "--config", help="Path to benchmark.yaml config file"),
    dataset_path: Optional[str] = typer.Option(None, help="Override HF dataset path"),
    output_dir: str = typer.Option("results", "--output", help="Directory to save raw embedding data"),
    num_samples: Optional[int] = typer.Option(None, "--samples", help="Number of samples per sequence length"),
    num_seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for dataset shuffling reproducibility"),
):
    """
    Generate raw embeddings from the Finesse benchmark dataset.
    """
    # Load config
    if not os.path.exists(config_path):
        typer.echo(f"Error: Config file not found: {config_path}")
        raise typer.Exit(code=1)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    try:
        config = BenchmarkConfig.model_validate(yaml_data)
        typer.echo(f"Loaded config from {config_path}")
    except Exception as e:
        typer.echo(f"Error validating config: {e}")
        raise typer.Exit(code=1)
    
    # Override if provided
    if dataset_path:
        config.dataset.path = dataset_path
    if num_samples:
        config.probe_config.samples_per_length = num_samples
    if num_seed:
        config.seed = num_seed
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Init Evaluator
    typer.echo("Initializing FinesseEvaluator...")
    evaluator = FinesseEvaluator(config)

    # Run raw evaluation
    typer.echo("Generating raw embeddings...")
    raw_data = evaluator.raw_run()
    
    # Save full raw data (config + raw_results) to .pt file
    dataset_name = config.dataset.path.split('/')[-1]
    save_path = os.path.join(output_dir, f"embeddings_{config.mode}_{dataset_name}.pt")
    torch.save(raw_data, save_path)
    
    typer.echo(f"Raw data (with config) saved to {save_path}")
    length_results = raw_data['raw_results'].get('length_results', {})
    num_lengths = len(length_results)
    typer.echo(f"Processed {num_lengths} sequence lengths with raw probe and synthesis embeddings.")

@app.command("score")
def score_embeddings(
    pt_path: str = typer.Option(..., "--pt-path", help="Path to the raw .pt data file from the generate command"),
    output_dir: str = typer.Option("results", "--output", help="Directory to save scored results"),
):
    """
    Compute scores from raw embeddings data.
    """
    if not os.path.exists(pt_path):
        typer.echo(f"Error: Input .pt file not found: {pt_path}")
        raise typer.Exit(code=1)
    
    # Load full raw data
    raw_data = torch.load(pt_path)
    config_dict = raw_data['config']
    raw_results = raw_data['raw_results']
    length_results = raw_results.get('length_results', {})
    
    if not length_results:
        typer.echo("Error: No length results found in .pt file.")
        raise typer.Exit(code=1)
    
    # Compute scores per length
    final_scores_per_length = {}
    for target_length, raw in length_results.items():
        probe_embeddings = raw['probe_embeddings']
        synthesis_embeddings = raw['synthesis_embeddings']
        num_synth_steps = raw['num_synth_steps']
        num_probes = len(probe_embeddings)
        if num_probes >= 2 and num_synth_steps > 0:
            td_scores = calculate_self_attestation_scores(probe_embeddings, synthesis_embeddings)
            bu_scores = calculate_self_attestation_scores_bottom_up(probe_embeddings, synthesis_embeddings, num_synth_steps)
            avg_td = td_scores['contextual_coherence']
            avg_bu = bu_scores['bottom_up_coherence']
            imbalance = abs(avg_td - avg_bu)
            final_score = ((avg_td + avg_bu) / 2) - imbalance
            final_score *= 500
            final_scores_per_length[target_length] = final_score
        else:
            final_scores_per_length[target_length] = 0.0
    
    # Average RSS
    avg_rss = np.mean(list(final_scores_per_length.values()))
    
    # Round scores for precision control (get_content_hash will convert to str)
    avg_rss = round(avg_rss, 6)
    rounded_length_scores = {
        length: round(score, 6)
        for length, score in final_scores_per_length.items()
    }
    
    # Prepare base results without hash
    base_results = {
        'config': config_dict,
        'average_rss': avg_rss,
        'length_scores': rounded_length_scores
    }
    
    # Compute model hash for notarization (before content_hash)
    try:
        config = BenchmarkConfig.model_validate(config_dict)
        if config.mode == 'merger_mode':
            model_path = config.models.merger.name
        else:
            model_path = config.models.native_embedder.name
        model_hash = get_model_hash(model_path)
        base_results['model_hash'] = model_hash
        typer.echo(f"Model hash computed: {model_hash[:16]}... (for notarization)")
    except Exception as e:
        typer.echo(f"Warning: Could not compute model hash: {e}")
        base_results['model_hash'] = None
    
    # Create output dir before hashing to ensure debug path exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create copy for hashing with fixed frame ('content_hash': '')
    hash_data = base_results.copy()
    hash_data['content_hash'] = ''
    
    # Compute content hash on the fixed frame with debug
    content_hash = get_content_hash(hash_data, debug_file_path='results/stored_canonical.txt')
    
    # Add the hash to final results
    results = base_results.copy()
    results['content_hash'] = content_hash
    
    # Save to JSON
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w", encoding='utf-8', newline='') as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"Scored results saved to {output_path}")
    typer.echo(f"Average RSS: {avg_rss}")

@app.command("checksum")
def verify_integrity(json_path: str = typer.Option(..., "--json-path", help="Path to the results JSON file to verify")):
    """
    Verify the integrity of a results.json file using its self-contained content hash.
    """
    if not os.path.exists(json_path):
        typer.echo(f"❌ Error: File not found: {json_path}")
        raise typer.Exit(code=1)
    
    import json  # Ensure json is imported
    
    # Read original text
    with open(json_path, "r", encoding='utf-8', newline='') as f:
        original_text = f.read()
    
    # Load data
    data = json.loads(original_text)
    
    if 'content_hash' not in data:
        typer.echo("❌ Error: No 'content_hash' found in the file. This file is not notarized.")
        raise typer.Exit(code=1)
    
    stored_hash = data['content_hash']
    
    # Create copy and set fixed frame for recomputation
    verify_data = data.copy()
    verify_data['content_hash'] = ''
    recomputed_hash = get_content_hash(verify_data, debug_file_path='results/recomputed_canonical.txt')
    
    if recomputed_hash == stored_hash:
        click.echo("✅ Verification SUCCESS")
        click.echo(f"Stored: {stored_hash}")
        click.echo(f"Recomputed: {recomputed_hash}")
    else:
        click.echo("❌ Verification FAILED")
        click.echo(f"Stored: {stored_hash}")
        click.echo(f"Recomputed: {recomputed_hash}")
        raise typer.Exit(code=1)

@app.command("init")
def init_config(output_path: str = typer.Option("benchmark.yaml", "--output", help="Path to save the config file")):
    """
    Generate a default benchmark.yaml template with comments.
    """
    template = '''# Finesse Benchmark Configuration
# This file configures the benchmark modes, models, probe settings, etc.
# For merger_mode: Use sequence-merger with a base embedder.
# For native_mode: Use a long-context native embedder directly.

mode: "merger_mode"  # Options: "merger_mode" or "native_mode"

# Models Configuration
models:
  # Used only in merger_mode
  merger:
    # Hugging Face model name or local path for Sequence Merger
    name: "enzoescipy/sequence-merger-tiny"
  # merger_mode: base embedder for probes, native_mode: the main long-context embedder
  base_embedder:
    # e.g., multilingual-e5-base for merger, or longformer-base-4096 for native
    name: "intfloat/multilingual-e5-base"
  # Used only in native_mode (if separate)
  native_embedder:
    # e.g., "Snowflake/snowflake-arctic-embed-l-v2.0"
    name: "Snowflake/snowflake-arctic-embed-l-v2.0"

# Dataset Configuration
dataset:
  path: "enzoescipy/finesse-benchmark-database"  # HF dataset path
  split: "train"  # Split to use

# Probe Configuration
probe_config:
  mask_ratio: 0.15  # Token masking ratio for probes
  sequence_length:
    min: 5  # Minimum sequence length in tokens
    max: 16  # Maximum sequence length in tokens
  samples_per_length: 1  # Evaluations per length

# Advanced Settings
advanced: {}
  # batch_size: 8
  # device: "cuda"

# Seed for Reproducibility
seed: 42  # Default seed for dataset shuffling
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    # Self-validate the generated config
    try:
        with open(output_path, "r", encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        config = BenchmarkConfig.model_validate(yaml_data)
        typer.echo(f"Default benchmark.yaml generated at: {output_path}")
        typer.echo("YAML template validated successfully with BenchmarkConfig.")
        typer.echo("Edit the file to customize models, modes, and settings.")
    except Exception as e:
        typer.echo(f"Error: Generated YAML is invalid - {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()