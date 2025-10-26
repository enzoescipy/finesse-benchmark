import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

def generate_heatmap_for_length(
    sample_results: List[Dict],
    length: int,
    mode: str,
    output_dir: str,
    num_samples: int = 25
):
    """
    Generate cosine similarity heatmap visualizations from raw sample results.
    
    Args:
        sample_results: List of sample dictionaries containing chunk_embeddings and synthesis_embeddings
        length: Target sequence length
        mode: 'average', 'best', 'worst', or 'stddev'
        output_dir: Directory to save the plot
        num_samples: Number of samples processed (for title)
    """
    
    if not sample_results:
        raise ValueError(f"No sample results found for length {length}")
    
    # Mode processing logic - the artist now chooses their own materials
    if mode == 'average':
        # Average across all samples
        all_chunk_embs = []
        all_synth_embs = []
        
        for sample_dict in sample_results:
            chunk_embs = sample_dict.get('chunk_embeddings', [])
            synth_embs = sample_dict.get('synthesis_embeddings', [])
            
            if chunk_embs and synth_embs:
                all_chunk_embs.append(torch.stack(chunk_embs))
                all_synth_embs.append(torch.stack(synth_embs))
        
        if not all_chunk_embs:
            raise ValueError(f"No valid embeddings found for length {length}")
        
        # Stack all samples and average
        chunk_embeddings_3d = torch.stack(all_chunk_embs)  # (num_samples, N, D)
        synth_embeddings_3d = torch.stack(all_synth_embs)  # (num_samples, N, D)
        
        chunk_embeddings = chunk_embeddings_3d.mean(dim=0)  # (N, D)
        synth_embeddings = synth_embeddings_3d.mean(dim=0)  # (N, D)
        
        # Convert to list for compatibility with existing heatmap logic
        chunk_embeddings_list = list(torch.unbind(chunk_embeddings, dim=0))
        synth_embeddings_list = list(torch.unbind(synth_embeddings, dim=0))
        
    elif mode in ['best', 'worst']:
        # Define device for scoring
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Find the best/worst sample based on scoring
        sample_scores = []
        
        for sample_dict in sample_results:
            chunk_embs = sample_dict.get('chunk_embeddings', [])
            synth_embs = sample_dict.get('synthesis_embeddings', [])
            
            if chunk_embs and synth_embs and len(chunk_embs) >= 2:
                # Convert to device for scoring
                chunk_embs_gpu = [emb.to(device) for emb in chunk_embs]
                synth_embs_gpu = [emb.to(device) for emb in synth_embs]
                
                # Calculate scores using the new scoring functions
                td_scores = calculate_self_attestation_scores(chunk_embs_gpu, synth_embs_gpu)
                bu_scores = calculate_self_attestation_scores_bottom_up(chunk_embs_gpu, synth_embs_gpu)
                
                avg_td = td_scores['contextual_coherence']
                avg_bu = bu_scores['bottom_up_coherence']
                imbalance = abs(avg_td - avg_bu)
                final_score = ((avg_td + avg_bu) / 2) - imbalance
                sample_scores.append(final_score)
            else:
                # Use extreme values for invalid samples
                sample_scores.append(-float('inf') if mode == 'best' else float('inf'))
        
        if not sample_scores:
            raise ValueError(f"Could not calculate scores for length {length}")
        
        # Find target sample
        target_idx = np.argmax(sample_scores) if mode == 'best' else np.argmin(sample_scores)
        target_sample = sample_results[target_idx]
        
        chunk_embeddings_list = target_sample['chunk_embeddings']
        synth_embeddings_list = target_sample['synthesis_embeddings']
        
    elif mode == 'stddev':
        # Calculate standard deviation across samples
        all_chunk_embs = []
        all_synth_embs = []
        
        for sample_dict in sample_results:
            chunk_embs = sample_dict.get('chunk_embeddings', [])
            synth_embs = sample_dict.get('synthesis_embeddings', [])
            
            if chunk_embs and synth_embs:
                all_chunk_embs.append(torch.stack(chunk_embs))
                all_synth_embs.append(torch.stack(synth_embs))
        
        if not all_chunk_embs:
            raise ValueError(f"No valid embeddings found for length {length}")
        
        # Stack all samples and calculate stddev
        chunk_embeddings_3d = torch.stack(all_chunk_embs)  # (num_samples, N, D)
        synth_embeddings_3d = torch.stack(all_synth_embs)  # (num_samples, N, D)
        
        chunk_embeddings = chunk_embeddings_3d.std(dim=0)  # (N, D)
        synth_embeddings = synth_embeddings_3d.std(dim=0)  # (N, D)
        
        # Convert to list for compatibility with existing heatmap logic
        chunk_embeddings_list = list(torch.unbind(chunk_embeddings, dim=0))
        synth_embeddings_list = list(torch.unbind(synth_embeddings, dim=0))
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Define device for vectorized computation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Vectorized cross-similarity computation
    chunk_emb_tensor = torch.stack(chunk_embeddings_list).to(device)
    synth_emb_tensor = torch.stack(synth_embeddings_list).to(device)
    
    cross_sim = F.cosine_similarity(
        synth_emb_tensor.unsqueeze(1), 
        chunk_emb_tensor.unsqueeze(0), 
        dim=2
    )
    cross_similarity = cross_sim.cpu().numpy()
    
    # Create the single meaningful plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Cross-similarity heatmap: Synthesis (Y) vs Chunks (X)
    sns.heatmap(cross_similarity, ax=ax, cmap='viridis', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Cosine Similarity'})
    ax.set_title(f'Cross-Similarity: Synthesis vs Chunks({mode.capitalize()} Mode)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Chunk Index (Memory Probes ←→ Noise Probes)')
    ax.set_ylabel('Synthesis Step Index')
    
    # Add interpretation guide
    ax.text(0.02, 0.98, 'High similarity = Model remembers the chunk', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'heatmap_length_{length}_mode_{mode}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath