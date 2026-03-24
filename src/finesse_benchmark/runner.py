import os
import yaml
import json
from typing import Dict, Any

from .config import BenchmarkConfig
from .cli import generate_raw_data, score_embeddings

def run_benchmark_from_config(
    config_path: str,
    output_dir: str,
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
    
    if verbose:
        print(f"Step 1: Generating raw embeddings ...")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Call generate_raw_data with NO overrides - force config adherence
    pt_path = generate_raw_data(
        config_path=config_path,
        output_dir=output_dir,
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
    
    if verbose:
        print(f"Pipeline completed")
    
    return {
        'pt_path': pt_path,
        'json_path': json_path,
    }