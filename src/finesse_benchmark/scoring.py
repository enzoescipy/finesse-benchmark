import torch
import torch.nn.functional as F
import numpy as np

def calculate_self_attestation_scores(chunk_embeddings, synth_embeddings):
    """
    Calculate Top-Down self-attestation scores using Robust Separation Score (quartile-based).
    
    Robust Separation Philosophy:
    - Focus on RAG separability: Weakest memories must outperform strongest noise.
    - No rank-order violations; instead, measure quartile gap between Tier 1 and Tier 2.
    - Q1(Tier 1) - Q3(Tier 2): Ensures weakest 25% of memories > strongest 75% of noise.
    - No subjective cohesion penalty; pure practical distinction for RAG utility.
    
    Args:
        chunk_embeddings: List[torch.Tensor] - Embeddings for all chunks, each (d_model,)
        synth_embeddings: List[torch.Tensor] - Embeddings for synthesis steps, each (d_model,)
        num_chunks: Number of chunks (len(chunk_embeddings))
        chunk_ids: List of chunk IDs (for compatibility, unused in computation)
    
    Returns:
        Dictionary with 'contextual_coherence' score (average robust gaps)
    """
    # Stack embeddings
    chunk_emb_tensor = torch.stack(chunk_embeddings)  # (M, d_model)
    synth_emb_tensor = torch.stack(synth_embeddings)  # (N, d_model)
    
    # Compute similarity matrix (N_synth, M_chunks)
    sim_matrix = F.cosine_similarity(
        synth_emb_tensor.unsqueeze(1), 
        chunk_emb_tensor.unsqueeze(0), 
        dim=2
    ).cpu().numpy()
    
    N = len(sim_matrix)  # Number of synthesis steps
    M = len(sim_matrix[0])  # Number of chunks

    row_gaps = []

    # Evaluate only middle synthesis steps, excluding start (Synth(A)) and end (Synth(ABC...G))
    for i in range(1, N - 1):  # Skip first (i=0) and last (i=N-1) synthesis steps
        # Assign 2-tier system
        tier_for_chunk = []
        for j in range(M):
            if j < i:  # Memory chunks: part of the synthesis
                tier = 1
            else:  # Noise chunks: not part of the synthesis
                tier = 2
            tier_for_chunk.append(tier)

        # Collect tier indices
        tier1_js = [j for j in range(M) if tier_for_chunk[j] == 1]
        tier2_js = [j for j in range(M) if tier_for_chunk[j] == 2]

        # Collect scores for each tier
        tier1_scores = sim_matrix[i][tier1_js]
        tier2_scores = sim_matrix[i][tier2_js]

        # Calculate Robust Gap if both tiers have scores
        if len(tier1_scores) > 0 and len(tier2_scores) > 0:
            q1_t1 = np.percentile(tier1_scores, 25)
            q3_t2 = np.percentile(tier2_scores, 75)
            row_gap = q1_t1 - q3_t2
        else:
            row_gap = 0.0  # No meaningful separation possible

        row_gaps.append(row_gap)

    # Overall contextual coherence score (average robust gap)
    contextual_coherence = np.mean(row_gaps) if row_gaps else 0.0

    return {
        'contextual_coherence': float(contextual_coherence)
    }

def calculate_self_attestation_scores_bottom_up(chunk_embeddings, synth_embeddings):
    """
    Calculate Bottom-Up self-attestation scores using Robust Separation Score.
    
    For each story chunk as anchor (starting from 1 to num_synth_steps-2 to skip start and end):
    - Tier 1 (Memory): Synths that include this chunk (synth_idx >= anchor_idx)
    - Tier 2 (Noise): Synths that do not include it (synth_idx < anchor_idx)
    
    Robust Gap: Q1(Tier 1 scores) - Q3(Tier 2 scores)
    Ensures weakest 25% of including synths > strongest 75% of non-including synths.
    Pure separation without rank violations, focusing on RAG utility.
    
    Args:
        chunk_embeddings: List[torch.Tensor] - Embeddings for all chunks, each (d_model,)
        synth_embeddings: List[torch.Tensor] - Embeddings for synthesis steps, each (d_model,)
        main_story_end: Number of story chunks (unused, for compatibility)
        chunk_ids: List of chunk IDs (unused, for consistency)
    
    Returns:
        Dict with 'bottom_up_coherence' (average robust gaps over anchors)
    """
    # Stack embeddings
    chunk_emb_tensor = torch.stack(chunk_embeddings)  # (M, d_model)
    synth_emb_tensor = torch.stack(synth_embeddings)  # (N, d_model)
    
    # Compute similarity matrix for bottom-up: (M_chunks, N_synth)
    sim_bottom_up = F.cosine_similarity(
        chunk_emb_tensor.unsqueeze(1), 
        synth_emb_tensor.unsqueeze(0), 
        dim=2
    ).cpu().numpy()
    
    M_synth = len(synth_embeddings)

    row_gaps = []

    # Evaluate only middle chunks, excluding start (anchor_idx=0) and end (anchor_idx=M_synth-1)
    for anchor_idx in range(1, M_synth):  # Skip start and end anchors
        tier_for_synth = []
        for j in range(M_synth):
            if j >= anchor_idx:  # synth j includes chunks 0 to j, so includes anchor_idx if j >= anchor_idx
                tier = 1  # Memory
            else:
                tier = 2  # Noise
            tier_for_synth.append(tier)

        # Collect tier indices
        tier1_js = [j for j in range(M_synth) if tier_for_synth[j] == 1]
        tier2_js = [j for j in range(M_synth) if tier_for_synth[j] == 2]

        # Collect scores for each tier (similarities from this anchor to synths)
        tier1_scores = sim_bottom_up[anchor_idx, tier1_js]
        tier2_scores = sim_bottom_up[anchor_idx, tier2_js]

        # Calculate Robust Gap if both tiers have scores
        if len(tier1_scores) > 0 and len(tier2_scores) > 0:
            q1_t1 = np.percentile(tier1_scores, 25)
            q3_t2 = np.percentile(tier2_scores, 75)
            row_gap = q1_t1 - q3_t2
        else:
            row_gap = 0.0  # No meaningful separation

        row_gaps.append(row_gap)

    # Average over all anchors
    contextual_coherence_bottom_up = np.mean(row_gaps) if row_gaps else 0.0

    return {
        'bottom_up_coherence': float(contextual_coherence_bottom_up)
    }