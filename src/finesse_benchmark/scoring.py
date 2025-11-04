import torch
import torch.nn.functional as F
from typing import List

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
    device = chunk_embeddings[0].device
    chunk_emb_tensor = torch.stack([t.to(device) for t in chunk_embeddings])  # (M, d_model)
    synth_emb_tensor = torch.stack([t.to(device) for t in synth_embeddings])  # (N, d_model)
    
    # Compute similarity matrix (N_synth, M_chunks)
    sim_matrix = F.cosine_similarity(
        synth_emb_tensor.unsqueeze(1), 
        chunk_emb_tensor.unsqueeze(0), 
        dim=2
    )
    
    N = sim_matrix.shape[0]  # Number of synthesis steps
    M = sim_matrix.shape[1]  # Number of chunks

    row_gaps = []

    # Evaluate only middle synthesis steps, excluding start (Synth(A)) and end (Synth(ABC...G))
    for i in range(1, N - 1):  # Skip first (i=0) and last (i=N-1) synthesis steps
        # Assign 2-tier system
        tier_for_chunk = []
        for j in range(M):
            if j <= i:  # Memory chunks: part of the synthesis
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
            q1_t1 = torch.quantile((tier1_scores).to(torch.float32), 0.25)
            q3_t2 = torch.quantile((tier2_scores).to(torch.float32), 0.75)
            row_gap = q1_t1 - q3_t2
        else:
            row_gap = torch.tensor(0.0, device=sim_matrix.device)  # No meaningful separation possible

        row_gaps.append(row_gap)

    # Overall contextual coherence score (average robust gap)
    contextual_coherence = torch.mean(torch.stack(row_gaps)).item() if row_gaps else 0.0

    return {
        'contextual_coherence': contextual_coherence
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
    device = chunk_embeddings[0].device
    chunk_emb_tensor = torch.stack([t.to(device) for t in chunk_embeddings])  # (M, d_model)
    synth_emb_tensor = torch.stack([t.to(device) for t in synth_embeddings])  # (N, d_model)
    
    # Compute similarity matrix for bottom-up: (M_chunks, N_synth)
    sim_bottom_up = F.cosine_similarity(
        chunk_emb_tensor.unsqueeze(1), 
        synth_emb_tensor.unsqueeze(0), 
        dim=2
    )
    
    M_synth = sim_bottom_up.shape[1]

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
        tier1_scores = sim_bottom_up[anchor_idx][tier1_js]
        tier2_scores = sim_bottom_up[anchor_idx][tier2_js]

        # Calculate Robust Gap if both tiers have scores
        if len(tier1_scores) > 0 and len(tier2_scores) > 0:
            q1_t1 = torch.quantile((tier1_scores).to(torch.float32), 0.25)
            q3_t2 = torch.quantile((tier2_scores).to(torch.float32), 0.75)
            row_gap = q1_t1 - q3_t2
        else:
            row_gap = torch.tensor(0.0, device=sim_bottom_up.device)  # No meaningful separation

        row_gaps.append(row_gap)

    # Average over all anchors
    contextual_coherence_bottom_up = torch.mean(torch.stack(row_gaps)).item() if row_gaps else 0.0

    return {
        'bottom_up_coherence': contextual_coherence_bottom_up
    }

def calculate_sample_latency(mode: str, chunk_times: list[float], synth_times: list[float]) -> dict:
    """
    Calculate latencies for final synthesized vector production.
    
    Returns two metrics:
    - total_latency: Total time for cold start (all chunks + final synthesis).
    - synthesis_latency: Time for warm start (final synthesis only, assuming cached chunks).
    
    Args:
        mode: 'merger_mode', 'native_mode', or 'byok_mode'.
        chunk_times: List of embedding times for chunks (ms), None/empty for non-merger modes.
        synth_times: List of synthesis times (ms), last is final synthesis.
    """
    if mode == 'merger_mode':
        # Caching cost (one-time embedding of chunks)
        caching_cost = sum(chunk_times) if chunk_times else 0.0
        # Final synthesis cost (real-time assembly)
        synthesis_cost = synth_times[-1] if synth_times else 0.0
        total_latency = caching_cost + synthesis_cost
        synthesis_latency = synthesis_cost
    elif mode in ('native_mode', 'byok_mode'):
        # End-to-End: no separation of caching and synthesis
        total_cost = synth_times[-1] if synth_times else 0.0
        total_latency = total_cost
        synthesis_latency = total_cost
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    return {
        'total_latency': total_latency,
        'synthesis_latency': synthesis_latency
    }

def calculate_srs_score(probe_embedding: torch.Tensor, positive_embeddings: List[torch.Tensor], negative_embeddings: List[torch.Tensor]) -> float:
    """
    Calculate Sequence Recognition Sensitivity (SRS) score for a single probe position.
    
    SRS Philosophy:
    - Measures model's awareness of order/direction: How well does the probe (AB) separate
      from reversed probe groups (BA) in symmetric contexts?
    - Robust Separation: Weakest 25% of expected matches (positive Q1) must exceed
      strongest 75% of mismatches (negative Q3).
    - Raw gap: Q1(positive sims) - Q3(negative sims). Positive gap indicates clear directionality.
    
    Args:
        probe_embedding: 1D tensor (d_model,) - Embedding of the forward probe (e.g., synth(AB)).
        positive_embeddings: List[torch.Tensor] - Embeddings of forward-inserted groups (ABX...), each (d_model,).
        negative_embeddings: List[torch.Tensor] - Embeddings of reverse-inserted groups (BAX...), each (d_model,).

    Returns:
        float: Raw SRS gap score. Higher is better (clear separation). No normalization; aggregate upstream.
    
    Raises:
        ValueError: If inputs are empty, mismatched dims, or invalid shapes.
    """
    if len(positive_embeddings) == 0 or len(negative_embeddings) == 0:
        raise ValueError("Positive or negative embeddings list cannot be empty for SRS scoring.")

    device = probe_embedding.device
    if probe_embedding.dim() != 1:
        raise ValueError(f"Probe embedding must be 1D tensor, got {probe_embedding.dim()}D with shape {probe_embedding.shape}")

    d_model = probe_embedding.shape[0]
    for group, group_name in [(positive_embeddings, 'positive'), (negative_embeddings, 'negative')]:
        if any(emb.dim() != 1 or emb.shape[0] != d_model for emb in group):
            raise ValueError(f"All {group_name} embeddings must be 1D tensors of dim {d_model}, but some are invalid.")

    # Move to device and stack for efficiency
    probe_on_device = probe_embedding.unsqueeze(0)  # (1, d_model)
    pos_tensor = torch.stack([emb.to(device) for emb in positive_embeddings])  # (num_pos, d_model)
    neg_tensor = torch.stack([emb.to(device) for emb in negative_embeddings])  # (num_neg, d_model)

    # Compute cosine similarities
    sims_positive = F.cosine_similarity(probe_on_device, pos_tensor, dim=1)  # (num_pos,)
    sims_negative = F.cosine_similarity(probe_on_device, neg_tensor, dim=1)  # (num_neg,)

    # Calculate quartiles
    q1_positive = torch.quantile(sims_positive.float(), 0.25)
    q3_negative = torch.quantile(sims_negative.float(), 0.75)

    # Raw gap: Q1(pos) - Q3(neg)
    srs_gap = (q1_positive - q3_negative).item()

    # Cleanup
    del pos_tensor, neg_tensor, sims_positive, sims_negative

    return srs_gap