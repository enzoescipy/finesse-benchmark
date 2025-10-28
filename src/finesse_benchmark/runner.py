import os
import yaml
import json
from typing import Dict, Any

from .config import BenchmarkConfig
from .cli import generate_raw_data, score_embeddings

def run_benchmark_from_config(
    config_path: str,
    output_dir: str = "results",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the full Finesse benchmark pipeline by calling generate and score functions from CLI.

    This orchestrates the existing CLI logic in a non-CLI context for scripts/notebooks,
    strictly adhering to the config.yaml as the single source of truth for all parameters.

    Args:
        config_path: Path to benchmark.yaml.
        output_dir: Directory to save .pt and .json files.
        verbose: If True, print progress (uses typer.echo internally).

    Returns:
        Dict with 'pt_path', 'json_path', 'average_rss'.
    """
    # Load config minimally to compute expected pt_path
    if verbose:
        print(f"Loading config to determine output paths...")
    
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    try:
        config = BenchmarkConfig.model_validate(yaml_data)
        if verbose:
            print("Config validated.")
    except Exception as e:
        raise ValueError(f"Error validating config: {e}")
    
    # Compute expected pt_filename strictly from config (no overrides)
    dataset_name = config.dataset.path.split('/')[-1] if '/' in config.dataset.path else config.dataset.path
    pt_filename = f"embeddings_{config.mode}_{dataset_name}.pt"
    pt_path = os.path.join(output_dir, pt_filename)
    
    if verbose:
        print(f"Step 1: Generating raw embeddings to {pt_filename}...")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Call generate_raw_data with NO overrides - force config adherence
    generate_raw_data(
        config_path=config_path,
        dataset_path=None,      # No override
        output_dir=output_dir,
        num_seed=None,          # No override
    )
    
    if not os.path.exists(pt_path):
        raise ValueError(f"Generated .pt file not found at {pt_path}. Check generation step.")
    
    if verbose:
        print(f"Step 2: Scoring .pt file...")
    
    # Call score_embeddings (no overrides needed)
    score_embeddings(
        pt_path=pt_path,
        output_dir=output_dir,
    )
    
    # Load results from json
    json_path = os.path.join(output_dir, "benchmark_results.json")
    if not os.path.exists(json_path):
        raise ValueError(f"Results .json not found at {json_path}. Check scoring step.")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    avg_rss = results_data['average_rss']
    
    if verbose:
        print(f"Pipeline completed! Average RSS: {avg_rss}")
    
    return {
        'pt_path': pt_path,
        'json_path': json_path,
        'average_rss': avg_rss
    }