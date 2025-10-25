import os
import json
import yaml
from typing import Optional
import typer
import torch
import numpy as np

from .config import BenchmarkConfig
from .evaluator import FinesseEvaluator
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

app = typer.Typer(no_args_is_help=True)

@app.command("generate")
def generate_raw_data(
    config_path: Optional[str] = typer.Option(None, "--config", help="Path to benchmark.yaml config file"),
    dataset_path: Optional[str] = typer.Option(None, help="Override HF dataset path"),
    output_dir: str = typer.Option("results", "--output", help="Directory to save raw embedding data"),
    num_samples: Optional[int] = typer.Option(None, "--samples", help="Number of samples to process"),
):
    """
    Generate raw embeddings from the Finesse benchmark dataset.
    """
    # Load config
    if config_path:
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
    else:
        config = BenchmarkConfig()  # Default config
        typer.echo("Using default config")
    
    # Override if provided
    if dataset_path:
        config.dataset.path = dataset_path
    if num_samples:
        config.dataset.num_samples = num_samples
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Init Evaluator
    typer.echo("Initializing FinesseEvaluator...")
    evaluator = FinesseEvaluator(config)

    # Run raw evaluation
    typer.echo("Generating raw embeddings...")
    raw_results = evaluator.raw_run()
    
    # Save raw embeddings to .pt file
    dataset_name = config.dataset.path.split('/')[-1]
    save_path = os.path.join(output_dir, f"embeddings_{config.mode}_{dataset_name}.pt")
    torch.save(raw_results, save_path)
    
    typer.echo(f"Raw embeddings saved to {save_path}")
    length_results = raw_results.get('length_results', {})
    num_lengths = len(length_results)
    typer.echo(f"Processed {num_lengths} sequence lengths with raw probe and synthesis embeddings.")

@app.command("score")
def score_embeddings(
    input_pt_path: str = typer.Argument(..., help="Path to raw embeddings .pt file"),
    output_dir: str = typer.Option("results", "--output", help="Directory to save scored results"),
):
    """
    Compute scores from raw embeddings data.
    """
    if not os.path.exists(input_pt_path):
        typer.echo(f"Error: Input .pt file not found: {input_pt_path}")
        raise typer.Exit(code=1)
    
    # Load raw data
    raw_data = torch.load(input_pt_path)
    length_results = raw_data.get('length_results', {})
    
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
            final_scores_per_length[target_length] = final_score
        else:
            final_scores_per_length[target_length] = 0.0
    
    # Average RSS
    avg_rss = np.mean(list(final_scores_per_length.values()))
    
    # Prepare results
    results = {
        'average_rss': avg_rss,
        'length_scores': final_scores_per_length
    }
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"Scored results saved to {output_path}")
    typer.echo(f"Average RSS: {avg_rss:.4f}")

if __name__ == "__main__":
    app()