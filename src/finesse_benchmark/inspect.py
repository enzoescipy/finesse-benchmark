import os
import torch
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
        # Find the best/worst sample based on scoring
        sample_scores = []
        
        for sample_dict in sample_results:
            chunk_embs = sample_dict.get('chunk_embeddings', [])
            synth_embs = sample_dict.get('synthesis_embeddings', [])
            
            if chunk_embs and synth_embs and len(chunk_embs) >= 2:
                # Calculate scores using the new scoring functions
                td_scores = calculate_self_attestation_scores(chunk_embs, synth_embs)
                bu_scores = calculate_self_attestation_scores_bottom_up(chunk_embs, synth_embs)
                
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
    
    # Now generate the heatmap with the processed embeddings
    num_chunks = len(chunk_embeddings_list)
    num_synth_steps = len(synth_embeddings_list)
    
    # Create similarity matrices
    probe_similarity = np.zeros((num_chunks, num_chunks))
    synthesis_similarity = np.zeros((num_synth_steps, num_synth_steps))
    
    # Compute cosine similarities
    for i in range(num_chunks):
        for j in range(num_chunks):
            sim = torch.cosine_similarity(chunk_embeddings_list[i], chunk_embeddings_list[j], dim=0)
            probe_similarity[i, j] = sim.item()
    
    for i in range(num_synth_steps):
        for j in range(num_synth_steps):
            sim = torch.cosine_similarity(synth_embeddings_list[i], synth_embeddings_list[j], dim=0)
            synthesis_similarity[i, j] = sim.item()
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Probe embeddings heatmap
    sns.heatmap(probe_similarity, ax=ax1, cmap='viridis', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Cosine Similarity'})
    ax1.set_title(f'Probe Embeddings\n({mode.capitalize()} Mode)')
    ax1.set_xlabel('Chunk Index')
    ax1.set_ylabel('Chunk Index')
    
    # Synthesis embeddings heatmap
    sns.heatmap(synthesis_similarity, ax=ax2, cmap='viridis', annot=True, fmt='.2f',
                cbar_kws={'label': 'Cosine Similarity'})
    ax2.set_title(f'Synthesis Embeddings\n({mode.capitalize()} Mode)')
    ax2.set_xlabel('Step Index')
    ax2.set_ylabel('Step Index')
    
    # Overall title
    plt.suptitle(f'Finesse Benchmark - Length {length} (N={num_samples}, Mode={mode})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'heatmap_length_{length}_mode_{mode}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath