import os
import json
import yaml
from typing import Optional
import typer
import torch
import numpy as np
import traceback
from importlib import resources
from .utils import get_content_hash, get_model_hash
from typing import Dict, List
from .config import BenchmarkConfig
from .evaluator import FinesseEvaluator
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up, calculate_sample_latency

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

    This command initializes the FinesseEvaluator with your configuration and runs the raw evaluation to produce
    probe and synthesis embeddings for analysis. The output is saved as a .pt file containing the full raw_results
    along with the config for reproducibility.

    Required Arguments:
    --config: Path to your benchmark.yaml file defining models, dataset, probe settings, mode (merger_mode/native_mode/byok_mode), etc.
              This is the core blueprint for your evaluation. Use 'finesse init' to generate a template.

    Optional Arguments:
    --output: Directory where the .pt file will be saved. Defaults to 'results/'. The filename will be
              'embeddings_{mode}_{dataset_name}.pt' (e.g., embeddings_merger_mode_finesse-benchmark-database.pt).
    --dataset-path: Override the dataset path in your config (e.g., for local datasets or different HF repos).
    --samples: Override samples_per_length in probe_config (default from config is 25 for leaderboard reliability).
               Increase for more statistical power, but it will take longer to run.
    --seed: Override the random seed for reproducible dataset shuffling and sampling (default: 42).

    Usage Examples:
    $ finesse generate --config my_benchmark.yaml
       # Basic run with default settings from config.
    $ finesse generate --config leaderboard.yaml --output ./my_results --samples 50 --seed 123
       # Leaderboard config, custom output, more samples for precision, different seed.
    $ finesse generate --config byok_config.yaml --dataset-path ./local_data
       # BYOK mode with local dataset override.

    Notes:
    - For merger_mode: Uses sequence-merger with a base embedder (e.g., multilingual-e5-base).
    - For native_mode: Directly uses a long-context embedder (e.g., snowflake-arctic-embed-l).
    - For byok_mode: Requires API keys set as environment variables (e.g., OPENAI_API_KEY). Do NOT hardcode keys in YAML.
    - After running, use 'finesse score' on the output .pt to compute RSS scores.
    - Hardware Tip: Set advanced.batch_size in config based on your GPU memory; device auto-detects CUDA/CPU.
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
    
    # Validate sequence lengths - minimum length must be 4 for valid scoring
    sequence_length_min = config.probe_config.sequence_length.min
    
    if sequence_length_min < 4:
        typer.echo(f"❌ Error: Invalid sequence lengths minimum: {sequence_length_min}")
        typer.echo("   Minimum sequence length must be 4 for valid scoring.")
        typer.echo("   For lengths < 4, the scoring system cannot properly evaluate")
        typer.echo("   contextual coherence and bottom-up coherence.")
        raise typer.Exit(code=1)
    
    typer.echo(f"✅ Valid sequence lengths: {sequence_length_min}")
    
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Init Evaluator with proper engine injection based on mode
    typer.echo("Initializing FinesseEvaluator...")
    
    # Import the engine implementations
    from .implementations import HuggingFaceEmbedder, ByokEmbedder, HuggingFaceSynthesizer, MeanPoolingSynthesizer
    
    # Initialize embedder and synthesizer based on mode
    if config.mode == 'merger_mode':
        typer.echo("  Mode: merger_mode")
        # Use base embedder for embedding and sequence merger for synthesis
        embedder = HuggingFaceEmbedder(config.models.base_embedder.name)
        synthesizer = HuggingFaceSynthesizer(config.models.merger.name)
        typer.echo(f"  Embedder: {config.models.base_embedder.name}")
        typer.echo(f"  Synthesizer: {config.models.merger.name}")
    
    elif config.mode == 'native_mode':
        typer.echo("  Mode: native_mode")
        # Use native embedder for both embedding and synthesis (pass-through)
        embedder = HuggingFaceEmbedder(config.models.native_embedder.name)
        synthesizer = MeanPoolingSynthesizer()
        typer.echo(f"  Embedder: {config.models.native_embedder.name}")
        typer.echo("  Synthesizer: pass-through")
    
    elif config.mode == 'byok_mode':
        typer.echo("  Mode: byok_mode")
        if not config.models.byok_embedder:
            typer.echo("❌ Error: BYOK mode requires 'models.byok_embedder' configuration.")
            raise typer.Exit(code=1)
        
        # Use BYOK embedder for embedding and pass-through for synthesis
        embedder = ByokEmbedder(
            provider=config.models.byok_embedder.provider,
            model_name=config.models.byok_embedder.name,
            tokenizer_path=config.models.byok_embedder.tokenizer_path
        )
        synthesizer = MeanPoolingSynthesizer()
        typer.echo(f"  Embedder: {config.models.byok_embedder.provider}/{config.models.byok_embedder.name}")
        typer.echo("  Synthesizer: pass-through")
    
    else:
        typer.echo(f"❌ Error: Unknown mode '{config.mode}'")
        raise typer.Exit(code=1)
    
    # Initialize evaluator with the engines
    evaluator = FinesseEvaluator(embedder_engine=embedder, synthesizer_engine=synthesizer, config=config)

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
    Compute scores from raw embeddings data and generate the final benchmark_results.json with notarization.

    This command loads the .pt file from 'generate', computes self-attestation scores (top-down and bottom-up coherence)
    for each sequence length, calculates the Average RSS (Robustness to Sequence Scaling) metric, and produces a
    notarized JSON with content_hash and model_hash for integrity verification.

    Required Arguments:
    --pt-path: Path to the .pt embeddings file generated by the 'generate' command.
               Must contain 'config', 'raw_results' with length-specific embeddings and 'length_results'.

    Optional Arguments:
    --output: Directory to save benchmark_results.json. Defaults to 'results/'.
              The JSON includes average_rss, length_scores, config, content_hash, and model_hash.

    How Scoring Works:
    For each length (e.g., 4-32 tokens):
    - Top-Down (contextual_coherence): Measures synthesis separation from memory/noise probes.
    - Bottom-Up (bottom_up_coherence): Builds coherence incrementally across synthesis steps.
    - Final Score per Length: ((TD + BU)/2) - |TD - BU| imbalance, scaled by 500 for readability.
    - Average RSS: Mean of all length scores, indicating model robustness to increasing complexity.

    Notarization Details:
    - content_hash: SHA-256 of the results (excluding hash itself) for tamper-proof verification.
    - model_hash: Computed from the model name in config (Hugging Face ID) for provenance.
    - Use 'finesse checksum' to verify later.

    Usage Examples:
    $ finesse score --pt-path results/embeddings_merger_mode_finesse-benchmark-database.pt
       # Standard scoring on default generated file.
    $ finesse score --pt-path ./my_results/my_embeddings.pt --output ./final_scores
       # Custom paths for organized workflow.

    Notes:
    - Scores range: Higher is better (e.g., >0 indicates good separation; negative shows confusion).
    - For leaderboard submission: Use official config and 25 samples/length for fair comparison.
    - If .pt lacks data: Error will be raised; ensure 'generate' completed successfully.
    """
    if not os.path.exists(pt_path):
        typer.echo(f"Error: Input .pt file not found: {pt_path}")
        raise typer.Exit(code=1)
    
    raw_data = torch.load(pt_path, weights_only=False)
    config_dict = raw_data['config']
    metadata = raw_data.get('metadata', {})
    length_results = raw_data.get('raw_results', {}).get('length_results', {})
    
    if not length_results:
        typer.echo("Error: No length results found in .pt file.")
        raise typer.Exit(code=1)
    
    length_scores = {}
    all_individual_scores = []
    all_total_latencies = []
    all_synthesis_latencies = []
    mode = config_dict.get('mode')
    
    for target_length, raw in length_results.items():
        sample_results = raw.get('sample_results', [])
        sample_scores = []
        total_latency_list = []
        synthesis_latency_list = []
        
        for sample_dict in sample_results:
            probe_embeddings = sample_dict.get('chunk_embeddings')
            synthesis_embeddings = sample_dict.get('synthesis_embeddings')

            if probe_embeddings and synthesis_embeddings and len(probe_embeddings) >= 2:
                td_scores = calculate_self_attestation_scores(probe_embeddings, synthesis_embeddings)
                bu_scores = calculate_self_attestation_scores_bottom_up(probe_embeddings, synthesis_embeddings) # num_synth_steps removed
                
                avg_td = td_scores['contextual_coherence']
                avg_bu = bu_scores['bottom_up_coherence']
                imbalance = abs(avg_td - avg_bu)
                final_score = ((avg_td + avg_bu) / 2) - imbalance
                sample_scores.append(final_score)
            else:
                sample_scores.append(0.0)
            
            # Calculate latencies for this sample
            chunk_times = sample_dict.get('chunk_times', None)
            synth_times = sample_dict.get('synth_times', [])
            if mode and synth_times:
                try:
                    latency_dict = calculate_sample_latency(mode, chunk_times, synth_times)
                    total_latency_list.append(latency_dict['total_latency'])
                    synthesis_latency_list.append(latency_dict['synthesis_latency'])
                except Exception as e:
                    # Fallback for invalid data
                    total_latency_list.append(0.0)
                    synthesis_latency_list.append(0.0)
            else:
                total_latency_list.append(0.0)
                synthesis_latency_list.append(0.0)

        # Store scaled RSS and rounded latency scores as lists
        scaled_rss = [round(score * 500, 6) for score in sample_scores]
        scaled_total = [round(t, 6) for t in total_latency_list]
        scaled_synth = [round(s, 6) for s in synthesis_latency_list]
        
        length_scores[target_length] = {
            'rss_scores': scaled_rss,
            'total_latency_scores': scaled_total,  # ms, cold start
            'synthesis_latency_scores': scaled_synth  # ms, warm start
        }
        all_individual_scores.extend(scaled_rss)
        all_total_latencies.extend(scaled_total)
        all_synthesis_latencies.extend(scaled_synth)

    # Averages
    avg_rss = np.mean(all_individual_scores) if all_individual_scores else 0.0
    avg_rss = round(avg_rss, 6)
    avg_total_latency = np.mean(all_total_latencies) if all_total_latencies else 0.0
    avg_total_latency = round(avg_total_latency, 6)
    avg_synthesis_latency = np.mean(all_synthesis_latencies) if all_synthesis_latencies else 0.0
    avg_synthesis_latency = round(avg_synthesis_latency, 6)
    
    # Prepare base results without hash
    base_results = {
        'config': config_dict,
        'average_rss': avg_rss,
        'average_total_latency': avg_total_latency,
        'average_synthesis_latency': avg_synthesis_latency,
        'length_scores': length_scores,
        'metadata': metadata  # Includes device_info for hardware provenance from .pt
    }
    
    # Compute model hash for notarization (before content_hash)
    try:
        config = BenchmarkConfig.model_validate(config_dict)
        model_hash_dict = {}
        
        if config.mode == 'merger_mode':
            # Dual Notarization Protocol: Hash both merger and base_embedder
            merger_path = config.models.merger.name
            base_path = config.models.base_embedder.name
            model_hash_dict['merger'] = get_model_hash(merger_path)
            model_hash_dict['base_embedder'] = get_model_hash(base_path)
            typer.echo(f"Merger model hash computed: {model_hash_dict['merger'][:16]}...")
            typer.echo(f"Base embedder hash computed: {model_hash_dict['base_embedder'][:16]}...")
        elif config.mode == 'native_mode':
            native_path = config.models.native_embedder.name
            model_hash_dict['native'] = get_model_hash(native_path)
            typer.echo(f"Native model hash computed: {model_hash_dict['native'][:16]}...")
        elif config.mode == 'byok_mode':
            # Diplomat Passport Protocol: Hash the identity string for BYOK models
            provider = config.models.byok_embedder.provider
            name = config.models.byok_embedder.name
            hash_string = f"byok:{provider}:{name}"
            model_hash_dict['byok'] = get_content_hash({'identity': hash_string})
            typer.echo(f"BYOK model identity hash computed: {model_hash_dict['byok'][:16]}...")
        
        base_results['model_hash'] = model_hash_dict
    except Exception as e:
        typer.echo(f"Warning: Could not compute model hash: {e}")
        base_results['model_hash'] = None
    
# Create output dir before hashing to ensure debug path exists
    os.makedirs(output_dir, exist_ok=True)
    
# Create copy for hashing with fixed frame ('content_hash': '')
    hash_data = base_results.copy()
    hash_data['content_hash'] = ''
    
# Compute content hash on the fixed frame with debug
    content_hash = get_content_hash(hash_data)
# content_hash = get_content_hash(hash_data, debug_file_path='results/stored_canonical.txt')
    
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
def verify_integrity(
    json_path: str = typer.Option(..., "--json-path", help="Path to the results JSON file to verify"),
    merger_path: Optional[str] = typer.Option(None, "--merger-path", help="Path to the merger model (e.g., 'enzoescipy/sequence-merger-malgeum') for merger_mode provenance verification."),
    base_embedder_path: Optional[str] = typer.Option(None, "--base-embedder-path", help="Path to the base embedder model (e.g., 'intfloat/multilingual-e5-base') for merger_mode provenance verification."),
    native_path: Optional[str] = typer.Option(None, "--native-path", help="Path to the native embedder model for native_mode provenance verification."),
):
    """
    Verify the integrity of a results.json file using its self-contained content hash and optional model provenance.

    This command is the final gatekeeper for trust in your benchmark results. It recomputes the content_hash
    from the JSON structure (excluding the hash field) and compares it to the stored value. If --model-path is
    provided (as a Hugging Face model ID), it also verifies the model_hash against the actual model's hash.

    Required Arguments:
    --json-path: Path to the benchmark_results.json file from the 'score' command.
                 Must contain 'content_hash', 'model_hash' (optional), config, scores, etc.

    Optional Arguments:
    --merger-path: Hugging Face model ID (e.g., 'enzoescipy/sequence-merger-malgeum') for merger_mode provenance.
                   Only used if merger_mode is active in the config.
    --base-embedder-path: Hugging Face model ID (e.g., 'intfloat/multilingual-e5-base') for merger_mode provenance.
                          Only used if merger_mode is active in the config.
    --native-path: Hugging Face model ID (e.g., 'Snowflake/snowflake-arctic-embed-l-v2.0') for native_mode provenance.
                   Only used if native_mode is active in the config.

    Verification Steps:
    1. Content Integrity: Recompute SHA-256 of canonical JSON frame and match against stored 'content_hash'.
       SUCCESS: Results are untampered. FAILED: Alert for potential manipulation.
    2. Model Provenance (if --merger-path, --base-embedder-path, or --native-path): Compute hash of the specified model and match 'model_hash'.
       Ensures results tie to the claimed model version.

    Usage Examples:
    $ finesse checksum --json-path results/benchmark_results.json
       # Basic content verification.
    $ finesse checksum --json-path ./final/benchmark_results.json --merger-path enzoescipy/sequence-merger-malgeum --base-embedder-path intfloat/multilingual-e5-base
       # Full verification including merger_mode provenance.
    $ finesse checksum --json-path ./final/benchmark_results.json --native-path Snowflake/snowflake-arctic-embed-l-v2.0
       # Full verification including native_mode provenance.

    Security Notes:
    - Hashes are deterministic and reproducible across environments.
    - For sharing results: Include the full JSON; recipients can verify independently.
    - If model_hash is missing in JSON: Skips provenance, warns only.
    - Edge Case: Invalid JSON structure will fail loading, indicating corruption.
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
    recomputed_hash = get_content_hash(verify_data)
    # recomputed_hash = get_content_hash(verify_data, debug_file_path='results/recomputed_canonical.txt')
    
    if recomputed_hash == stored_hash:
        typer.echo("✅ Content Verification SUCCESS")
        typer.echo(f"Stored Content Hash: {stored_hash}")
        typer.echo(f"Recomputed Content Hash: {recomputed_hash}")
        
        # If any model path provided, perform model provenance check
        if merger_path or base_embedder_path or native_path:
            if 'model_hash' not in data or data['model_hash'] is None:
                typer.echo("❌ Model Provenance FAILED: No 'model_hash' in results.")
                raise typer.Exit(code=1)
            
            stored_model_hash = data['model_hash']
            config = BenchmarkConfig.model_validate(data['config'])
            
            try:
                if config.mode == 'merger_mode':
                    # Dual Notarization Protocol: Verify both merger and base_embedder
                    if not merger_path or not base_embedder_path:
                        typer.echo("❌ Model Provenance FAILED: For merger_mode, both --merger-path and --base-embedder-path must be provided.")
                        raise typer.Exit(code=1)
                    
                    # Compute hashes for both models
                    computed_merger_hash = get_model_hash(merger_path)
                    computed_base_hash = get_model_hash(base_embedder_path)
                    
                    # Get stored hashes
                    stored_merger_hash = stored_model_hash.get('merger')
                    stored_base_hash = stored_model_hash.get('base_embedder')
                    
                    if computed_merger_hash == stored_merger_hash and computed_base_hash == stored_base_hash:
                        typer.echo("✅ Model Provenance SUCCESS")
                        typer.echo(f"Merger Hash: {computed_merger_hash[:16]}... (matches)")
                        typer.echo(f"Base Embedder Hash: {computed_base_hash[:16]}... (matches)")
                    else:
                        typer.echo("❌ Model Provenance FAILED")
                        if computed_merger_hash != stored_merger_hash:
                            typer.echo(f"Merger Hash Mismatch: Computed {computed_merger_hash[:16]}..., Stored {stored_merger_hash[:16]}...")
                        if computed_base_hash != stored_base_hash:
                            typer.echo(f"Base Embedder Hash Mismatch: Computed {computed_base_hash[:16]}..., Stored {stored_base_hash[:16]}...")
                        raise typer.Exit(code=1)
                        
                elif config.mode == 'native_mode':
                    # Single model verification for native_mode
                    if not native_path:
                        typer.echo("❌ Model Provenance FAILED: For native_mode, --native-path must be provided.")
                        raise typer.Exit(code=1)
                    
                    computed_model_hash = get_model_hash(native_path)
                    stored_native_hash = stored_model_hash.get('native')
                    
                    if computed_model_hash == stored_native_hash:
                        typer.echo("✅ Model Provenance SUCCESS")
                        typer.echo(f"Native Model Hash: {computed_model_hash[:16]}... (matches)")
                    else:
                        typer.echo("❌ Model Provenance FAILED")
                        typer.echo(f"Native Hash Mismatch: Computed {computed_model_hash[:16]}..., Stored {stored_native_hash[:16]}...")
                        raise typer.Exit(code=1)
                        
                elif config.mode == 'byok_mode':
                    # Diplomat Passport Protocol for BYOK mode
                    if merger_path or base_embedder_path or native_path:
                        typer.echo("ℹ️ BYOK mode detected. Model path parameters are ignored.")
                    
                    provider = config.models.byok_embedder.provider
                    name = config.models.byok_embedder.name
                    hash_string = f"byok:{provider}:{name}"
                    computed_model_hash = get_content_hash({'identity': hash_string})
                    stored_byok_hash = stored_model_hash.get('byok')
                    
                    if computed_model_hash == stored_byok_hash:
                        typer.echo("✅ Model Provenance SUCCESS")
                        typer.echo(f"BYOK Identity Hash: {computed_model_hash[:16]}... (matches)")
                    else:
                        typer.echo("❌ Model Provenance FAILED")
                        typer.echo(f"BYOK Hash Mismatch: Computed {computed_model_hash[:16]}..., Stored {stored_byok_hash[:16]}...")
                        raise typer.Exit(code=1)
                else:
                    typer.echo("❌ Model Provenance ERROR: Unknown mode in config")
                    raise typer.Exit(code=1)
                    
            except Exception as e:
                typer.echo(f"❌ Model Provenance ERROR: {e}")
                raise typer.Exit(code=1)
        else:
            # Provide more helpful message based on config mode
            config = BenchmarkConfig.model_validate(data['config'])
            if config.mode == 'byok_mode':
                typer.echo("ℹ️ BYOK mode detected. Model provenance is based on provider/name identity.")
            elif config.mode == 'merger_mode':
                typer.echo("ℹ️ Run with --merger-path [MERGER] and --base-embedder-path [EMBEDDER] for full dual provenance verification.")
            else:
                typer.echo("ℹ️ Run with --native-path [EMBEDDER] for full provenance verification.")
    else:
        typer.echo("❌ Content Verification FAILED")
        typer.echo(f"Stored Content Hash: {stored_hash}")
        typer.echo(f"Recomputed Content Hash: {recomputed_hash}")
        raise typer.Exit(code=1)

@app.command("init")
def init_config(
    leaderboard: bool = typer.Option(False, "--leaderboard", help="Use official leaderboard configuration (copies benchmark.leaderboard.yaml)"),
    output_path: str = typer.Option("benchmark.yaml", "--output", help="Path to save the config file")):
    """
    Generate a default or leaderboard benchmark.yaml configuration template.

    This command bootstraps your evaluation setup. Use it to create a customizable YAML file defining
    benchmark mode, models, dataset, probe lengths, samples, and advanced settings. For official submissions,
    use --leaderboard to copy the standardized config.

    Optional Arguments:
    --leaderboard: If True, copies the official 'benchmark.leaderboard.yaml' (immutable for fair comparisons).
                   Includes standard models (sequence-merger-malgeum + multilingual-e5-base), dataset,
                   probe lengths 4-32, 25 samples/length, seed 42.
    --output: Path to save the generated YAML. Defaults to 'benchmark.yaml' in current directory.

    Template Contents (Default Mode):
    - mode: merger_mode (default; options: native_mode for direct long-context, byok_mode for external APIs).
    - models: merger/base_embedder/native_embedder configs (Hugging Face names).
    - dataset: HF path and split (default: enzoescipy/finesse-benchmark-database, train split).
    - probe_config: min/max sequence_length (default 5-16), samples_per_length (default 1; use 25+ for stats).
    - advanced: batch_size (default 8), device (auto CUDA/CPU).
    - seed: 42 for reproducibility.
    - BYOK Notes: Uncomment and set provider/name; API keys via env vars only (e.g., OPENAI_API_KEY).

    Leaderboard Mode Differences:
    - Fixed to merger_mode with official models.
    - Probe: 4-32 lengths, 25 samples each (balanced short-to-medium evaluation).
    - No customizations; edit your copy for experiments but use original for submissions.

    Usage Examples:
    $ finesse init --output my_config.yaml
       # Generate editable default template.
    $ finesse init --leaderboard --output leaderboard_config.yaml
       # Copy official leaderboard config; validates Pydantic schema on creation.

    Post-Generation Steps:
    - Edit the YAML (e.g., change models, lengths).
    - Validate: Run 'finesse init --leaderboard' again or manually with Pydantic to check syntax.
    - Use in 'generate': Pass as --config to start evaluation.
    """
    try:
        if leaderboard:
            resource_file = 'benchmark.leaderboard.yaml'
            typer.echo(f"Loading leaderboard config from package: {resource_file}")
            with resources.open_text('finesse_benchmark', resource_file) as f:
                content = f.read()
            typer.echo(f"Leaderboard benchmark.yaml generated at: {output_path}")
        else:
            resource_file = 'benchmark.default.yaml'
            typer.echo(f"Loading default config from package: {resource_file}")
            with resources.open_text('finesse_benchmark', resource_file) as f:
                content = f.read()
            typer.echo(f"Default benchmark.yaml generated at: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Validate the copied config
        try:
            with open(output_path, "r", encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            config = BenchmarkConfig.model_validate(yaml_data)
            if leaderboard:
                typer.echo("Leaderboard config validated successfully with BenchmarkConfig.")
            else:
                typer.echo("YAML template validated successfully with BenchmarkConfig.")
                typer.echo("Edit the file to customize models, modes, and settings.")
        except Exception as e:
            typer.echo(f"Error: Generated YAML is invalid - {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.echo(f"Error: Package resource not found: {resource_file}. Ensure it exists in the finesse_benchmark package.")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error loading config from package: {e}")
        raise typer.Exit(code=1)

@app.command("inspect")
def inspect(
    pt_path: str = typer.Option(..., "--pt-path", help="Path to the .pt embeddings file"),
    all_flag: bool = typer.Option(False, "--all", help="Inspect all lengths in the .pt file"),
    length: int = typer.Option(None, "--length", help="Specific sequence length to inspect (e.g., 8)"),
    mode: str = typer.Option("average", "--mode", help="Mode: average, stddev, worst, best, worst-time, best-time, average-time, stddev-time"),
    output_dir: str = typer.Option("inspect_plots", "--output-dir", help="Output directory for plots"),
):
    """
    Inspect similarity matrices by generating cosine similarity heatmap visualizations from a .pt embeddings file.

    This advanced inspection tool reveals the 'sine wave' oscillation patterns in model performance, highlighting
    how well synthesis embeddings separate from memory (left chunks) vs. noise (right chunks) probes. Useful for
    debugging instability in sequence-merging or long-context models. Outputs PNG heatmaps in RdBu_r colormap
    (red=high similarity to noise (bad), blue=high to memory (good), center=0).

    Required Arguments (Mutually Exclusive):
    --pt-path: [Essential] Path to the .pt file from 'generate' containing raw_results with embeddings per length.
    Exactly ONE of the following must be provided:
    --all: Flag to inspect ALL sequence lengths present in the .pt file (auto-detects numeric keys like '4','5',...,'32').
          Skips non-numeric keys (e.g., 'length_results').
    --length: Specific sequence length (integer, e.g., 8) to inspect. Targets potential 'valleys' like length 6.

    Optional Arguments:
    --mode: Analysis mode determining how the similarity matrix is aggregated/computed:
            - 'average': Mean similarity across all samples (overall trend).
            - 'stddev': Standard deviation across samples (instability/hotspots in 'sine wave').
            - 'worst': Similarity from the sample with lowest contextual_coherence score (failure case).
            - 'best': Similarity from the sample with highest score (success case).
            - 'worst-time': Worst performance with time annotations.
            - 'best-time': Best performance with time annotations.
            - 'average-time': Average performance with time annotations.
            - 'stddev-time': Standard deviation with time annotations.
            Default: 'average'. Use 'stddev' or 'worst' to diagnose issues.
    --output-dir: Folder to save PNG files. Defaults to 'inspect_plots/'. Files named 'heatmap_length_{L}_mode_{M}.png'
                  (e.g., heatmap_length_6_mode_worst.png). One file per length in --all mode.

    Heatmap Interpretation:
    - X-Axis: Chunk Index (Left: Memory Probes (should be high sim), Right: Noise Probes (should be low sim)).
    - Y-Axis: Synthesis Step Index (progression of generation).
    - Annotations: For small matrices (<10x10); otherwise, rely on colorbar.
    - Title: Includes length, mode, shape (N_synth x M_chunks), sample count.
    - If no samples for a length: Skips with warning; .pt must have valid raw_results[length_str].

    Usage Examples:
    $ finesse inspect --pt-path results/embeddings.pt --length 6 --mode worst --output-dir ./diagnostics
       # Deep-dive into length 6 failure (common 'valley' in RSS scores).
    $ finesse inspect --pt-path results/embeddings.pt --all --mode stddev
       # Full scan of all lengths for instability patterns; generates multiple PNGs.
    $ finesse inspect --pt-path my.pt --length 8 --mode best
       # Visualize peak performance at length 8.

    Workflow Integration:
    - Run after 'generate' to visualize before scoring.
    - Dependencies: Requires torch, numpy, matplotlib, seaborn (auto-installed with package).
    - Troubleshooting: If 'length not found', check .pt keys with Python: torch.load(pt)['raw_results'].keys().
    - Pro Tip: Use 'stddev' mode on --all to spot the 'sine wave' oscillations across lengths quickly.
    """
    from .inspect import generate_heatmap_for_length, generate_timeline_plot_for_length
    import torch
    import os

    plot_paths = []
    loaded_data = torch.load(pt_path, weights_only=False)
    
    # Extract length_results from the new data structure
    if 'raw_results' in loaded_data:
        raw_results_data = loaded_data['raw_results']
    else:
        raw_results_data = loaded_data

    if 'length_results' in raw_results_data:
        length_results = raw_results_data['length_results']
    else:
        length_results = raw_results_data

    try:
        if all_flag:
            # Inspect all available lengths
            available_lengths = []
            for key in length_results:
                if str(key).isdigit():
                    avail_len = int(key)
                    length_data = length_results[key]
                    # Check for the new sample_results structure
                    if 'sample_results' in length_data:
                        available_lengths.append(avail_len)
            
            available_lengths.sort()
            typer.echo(f"Inspecting all lengths: {available_lengths}")

            if not available_lengths:
                typer.echo("No valid length data found.")
                return

            for avail_len in available_lengths:
                length_data = length_results[avail_len]
                sample_results = length_data.get('sample_results', [])
                
                if not sample_results:
                    typer.echo(f"Warning: No samples found for length {avail_len}. Skipping.")
                    continue
                
                # Branch based on mode: heatmap or timeline
                if mode.endswith('-time'):
                    internal_mode = mode.replace('-time', '')
                    filename = generate_timeline_plot_for_length(
                        sample_results=sample_results,
                        length=avail_len,
                        mode=internal_mode,
                        output_dir=output_dir,
                        num_samples=len(sample_results)
                    )
                else:
                    filename = generate_heatmap_for_length(
                        sample_results=sample_results,
                        length=avail_len,
                        mode=mode,
                        output_dir=output_dir,
                        num_samples=len(sample_results)
                    )
                plot_paths.append(filename)
                typer.echo(f"Generated {'timeline plot' if mode.endswith('-time') else 'heatmap'} for length {avail_len} with mode {mode}")

        else:
            # Inspect specific length
            if length is None:
                typer.echo("Error: Please specify --all or a specific --length.")
                raise typer.Exit(code=1)
                
            length_data = length_results.get(str(length), length_results.get(length))
            if not length_data or 'sample_results' not in length_data:
                raise ValueError(f"Invalid or missing data for length {length} in .pt file.")
            
            sample_results = length_data['sample_results']
            if not sample_results:
                typer.echo(f"Warning: No samples found for length {length}.")
                return
            
            typer.echo(f"Inspecting length: {length}")
            
            # Branch based on mode: heatmap or timeline
            if mode.endswith('-time'):
                internal_mode = mode.replace('-time', '')
                filename = generate_timeline_plot_for_length(
                    sample_results=sample_results,
                    length=length,
                    mode=internal_mode,
                    output_dir=output_dir,
                    num_samples=len(sample_results)
                )
            else:
                filename = generate_heatmap_for_length(
                    sample_results=sample_results,
                    length=length,
                    mode=mode,
                    output_dir=output_dir,
                    num_samples=len(sample_results)
                )
            plot_paths.append(filename)
            typer.echo(f"Generated {'timeline plot' if mode.endswith('-time') else 'heatmap'} for length {length} with mode {mode}")

        typer.echo(f"Inspect plots saved to: {output_dir}")
        if plot_paths:
            typer.echo(f"Generated {len(plot_paths)} plots: {[os.path.basename(p) for p in plot_paths]}")
        else:
            typer.echo("No plots generated.")
        return
    except ValueError as ve:
        typer.echo(f"❌ Configuration Error: {ve}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Generation Error: {e}")
        typer.echo(traceback.format_exc())
        raise typer.Exit(code=1)

@app.command()
def verify(
    pt_path: str = typer.Option(...,"--pt-path", help="Path to the .pt probe matrix file"),
    json_path: str = typer.Option(...,"--json-path", help="Path to the .json results file")
):
    """Verify metadata consistency between .pt and .json files."""
    import torch
    import json

    try:
        # Load .pt file
        pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
        pt_metadata = pt_data['metadata']

        # Load .json file
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            json_metadata = json_data['metadata']

        # Compare metadata
        if pt_metadata == json_metadata:
            typer.echo("✅ Metadata verification: SUCCESS - Files are consistent.")
        else:
            typer.echo("❌ Metadata verification: FAILED - Files are inconsistent.")
            typer.echo("Debug: Check the differences manually.")

    except FileNotFoundError as e:
        typer.echo(f"❌ Error: File not found - {str(e)}")
        raise typer.Exit(code=1)
    except KeyError as e:
        typer.echo(f"❌ Error: Missing 'metadata' key in file - {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}")
        raise typer.Exit(code=1)



if __name__ == "__main__":
    app()