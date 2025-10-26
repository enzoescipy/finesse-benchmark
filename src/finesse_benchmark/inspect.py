import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from typing import List, Dict

from .scoring import calculate_self_attestation_scores

def generate_heatmap_for_length(
    chunk_embeddings: List[torch.Tensor],
    synth_embeddings: List[torch.Tensor],
    num_synth_steps: int,
    length: int,
    mode: str,
    output_dir: str = "inspect_plots"
) -> str:
    """
    Generate and save a single heatmap for one length, using lists of per-sample embedding lists.
    
    Args:
        chunk_embeddings: List[torch.Tensor] - lists of probe embeddings.
        synth_embeddings: List[torch.Tensor] - lists of synthesis embeddings.
        length: The sequence length.
        mode: One of 'average', 'stddev', 'worst', 'best'.
        output_dir: Directory to save the plot.
    
    Returns:
        Path to the saved plot file.
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import torch.nn.functional as F
    from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

    os.makedirs(output_dir, exist_ok=True)
    num_samples = len(chunk_embeddings)
    if num_samples == 0:
        raise ValueError(f"No samples for length {length}")

    sim_matrices: List[np.ndarray] = []
    scores: List[float] = []
    
    # Stack embeddings
    chunk_emb_tensor = torch.stack(chunk_embeddings)  # Shape: (M, d_model)
    synth_emb_tensor = torch.stack(synth_embeddings)  # Shape: (N, d_model)
    
    # Compute similarity matrix: (N_synth, M_chunks)
    sim_matrix = F.cosine_similarity(
        synth_emb_tensor.unsqueeze(1),  # (N, 1, d)
        chunk_emb_tensor.unsqueeze(0),  # (1, M, d)
        dim=2
    ).cpu().numpy()  # (N, M)
    
    sim_matrices.append(sim_matrix)
    
    # Compute proxy score
    td_scores = calculate_self_attestation_scores(chunk_embeddings, synth_embeddings)
    bu_scores = calculate_self_attestation_scores_bottom_up(chunk_embeddings, synth_embeddings, num_synth_steps)
    avg_td = td_scores['contextual_coherence']
    avg_bu = bu_scores['bottom_up_coherence']
    imbalance = abs(avg_td - avg_bu)
    final_score = ((avg_td + avg_bu) / 2) - imbalance
    final_score *= 500
    scores.append(final_score)
    
    # Aggregate based on mode
    if mode == "average":
        mat = np.mean(sim_matrices, axis=0)
    elif mode == "stddev":
        mat = np.std(sim_matrices, axis=0)
    elif mode == "worst":
        idx = np.argmin(scores)
        mat = sim_matrices[idx]
    elif mode == "best":
        idx = np.argmax(scores)
        mat = sim_matrices[idx]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mat,
        cmap='RdBu_r',
        center=0,
        annot=(mat.shape[0] < 10),
        fmt='.3f',
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.xlabel('Chunk Index (Left: Memory, Right: Noise)')
    plt.ylabel('Synthesis Step Index')
    plt.title(f"Similarity Matrix - Length: {length}, Mode: {mode} (Samples: {num_samples})")
    filename = os.path.join(output_dir, f"heatmap_length_{length}_mode_{mode}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename
