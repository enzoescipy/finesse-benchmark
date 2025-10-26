from typing import Dict, Any, Optional
import torch
from datasets import load_dataset
import numpy as np
import litellm
import typer
import tiktoken
import warnings
from transformers import AutoTokenizer

from .config import BenchmarkConfig
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up
from .interfaces import FinesseEmbedder, FinesseSynthesizer


class FinesseEvaluator:
    def __init__(self, embedder_engine: FinesseEmbedder, synthesizer_engine: FinesseSynthesizer, config: BenchmarkConfig):
        """
        새로운 엔진 기반 아키텍처의 FinesseEvaluator.
        더 이상 모델을 직접 로드하지 않고, 외부에서 주입된 엔진을 사용한다.
        
        Args:
            embedder_engine: 임베딩을 담당하는 엔진 객체
            synthesizer_engine: 합성을 담당하는 엔진 객체
            config: 벤치마크 설정
        """
        self.config = config
        self.embedder = embedder_engine
        self.synthesizer = synthesizer_engine

    def raw_run(self) -> Dict[str, Any]:
        """Finesse 벤치마크 실행: Stratified CSAT with Single-Pass Conveyor Belt (Raw mode - embeddings only)"""
        # Load dataset without any version parameter to use default
        dataset = load_dataset(
            path=self.config.dataset.path,
            split=self.config.dataset.split
        )

        # Shuffle dataset deterministically for reproducibility
        if self.config.seed is not None:
            dataset = dataset.shuffle(seed=self.config.seed)

        if self.config.dataset.num_samples:
            dataset = dataset.select(range(self.config.dataset.num_samples))

        # 단일 이터레이터 생성: 컨베이어 벨트 원칙
        iterator = iter(dataset)
        min_length, max_length = self.config.probe_config.sequence_length.min, self.config.probe_config.sequence_length.max
        total_needed_samples = (max_length - min_length + 1) * self.config.probe_config.samples_per_length
        if len(dataset) < total_needed_samples:
            raise ValueError(f"데이터셋 크기({len(dataset)})가 필요 샘플({total_needed_samples})보다 작음. 더 많은 데이터 필요.")

        length_results = {}  # 길이별 결과 저장: {'sample_results': [dicts], 'num_synth_steps': N}
        for target_length in range(min_length, max_length + 1):
            typer.echo(f"probe sequence [{target_length}] in progress ...")
            
            sample_results = []  # List of 25 dicts per length
            for index in range(self.config.probe_config.samples_per_length):
                typer.echo(f"probe sequence [{target_length}] in progress ({index}/{self.config.probe_config.samples_per_length})...")
                try:
                    sample = next(iterator)  # 다음 고유 샘플 가져옴
                    beads = sample['beads']  # List of bead texts
                    if not beads or len(beads) < target_length:
                        # Skip if not enough beads
                        continue

                    # Intelligent tailor logic: thread beads and precisely cut to reach target token size
                    chunk_texts = []
                    current_chunk = []
                    current_token_count = 0
                    target_token_size = self.config.probe_config.token_per_sample 

                    for bead_text in beads:
                        # Count tokens for this bead using the embedder's tokenizer
                        bead_token_count = self.embedder.count_tokens(bead_text)

                        # Check if adding this bead would exceed target
                        if current_token_count + bead_token_count > target_token_size:
                            # Calculate how many tokens we need to complete the current chunk
                            tokens_needed = target_token_size - current_token_count
                            
                            if tokens_needed > 0:
                                # Use the embedder's scissors to cut exactly what we need
                                needed_text = self.embedder.chunk_text(bead_text, tokens_needed)
                                current_chunk.append(needed_text)
                            
                            # Finalize the perfectly sized chunk
                            chunk_texts.append(' '.join(current_chunk))
                            current_chunk = []
                            current_token_count = 0
                            
                            # The remaining part of the bead is discarded
                            # We move to the next bead to start a new chunk
                            continue
                        
                        # Add bead to current chunk (it fits perfectly)
                        current_chunk.append(bead_text)
                        current_token_count += bead_token_count
                        
                        # If we have enough chunks, break early
                        if len(chunk_texts) >= target_length:
                            break
                    
                    # Handle any remaining partial chunk at the end
                    if current_chunk and len(chunk_texts) < target_length:
                        # Finalize the partial chunk as-is
                        chunk_texts.append(' '.join(current_chunk))
                    
                    # If we don't have enough chunks, skip this sample
                    if len(chunk_texts) < target_length:
                        continue
                    
                    # Embed the chunks using our new embedder engine
                    chunk_embeddings_tensor = self.embedder.encode(chunk_texts[:target_length])
                    chunk_embeddings = [chunk_embeddings_tensor[i] for i in range(chunk_embeddings_tensor.size(0))]

                    # Synthesis embeddings: cumulative synthesis using partial chunk stacks
                    synthesis_embeddings = []
                    for i in range(1, target_length + 1):
                        partial_embs = torch.stack(chunk_embeddings[:i]).unsqueeze(0)  # (1, i, D)
                        synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                        synthesis_embeddings.append(synth_emb)

                    # Package this sample's results
                    sample_dict = {
                        'chunk_embeddings': chunk_embeddings,
                        'synthesis_embeddings': synthesis_embeddings
                    }
                    sample_results.append(sample_dict)

                except (StopIteration, KeyError) as e:
                    if isinstance(e, StopIteration):
                        raise ValueError(f"데이터셋 소진: target_length={target_length}에서 샘플 부족.")
                    else:
                        # Skip on errors
                        continue

            # Store for this length only if we have samples
            if sample_results:
                length_results[target_length] = {
                    'sample_results': sample_results,  # List of dicts, each with chunk and synth lists
                    'num_synth_steps': target_length
                }

        return {
            'config': self.config.model_dump(),
            'raw_results': {
                'length_results': length_results
            }
        }

    def _setup_token_counter(self):
        """모드에 따라 적절한 토큰 카운터를 설정한다."""
        if self.config.mode == 'byok_mode':
            if self.config.models.byok_embedder is None:
                raise ValueError("BYOK mode requires 'models.byok_embedder' configuration.")
            
            provider = self.config.models.byok_embedder.provider
            model_name = self.config.models.byok_embedder.name
            
            # 1. De Facto Standardization: Use tiktoken for OpenAI models
            if provider == 'openai':
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                    return lambda text: len(encoding.encode(text))
                except KeyError:
                    warnings.warn(f"tiktoken encoding for {model_name} not found. Defaulting to general-purpose tokenizer.")
            
            # 2. Delegation of Autonomy: Use user-specified tokenizer
            if self.config.models.byok_embedder.tokenizer_path:
                tokenizer_path = self.config.models.byok_embedder.tokenizer_path
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
            
            # 3. Fallback with clear warning
            warnings.warn(f"BYOK provider '{provider}' is not OpenAI and no 'tokenizer_path' was provided. Falling back to default tokenizer.")
            fallback_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
            return lambda text: len(fallback_tokenizer.encode(text, add_special_tokens=False))
        
        else:  # merger_mode or native_mode
            # For local models, we need to use the embedder's tokenizer
            # But since we don't have direct access to the tokenizer anymore,
            # we'll use a default multilingual tokenizer as fallback
            fallback_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
            return lambda text: len(fallback_tokenizer.encode(text, add_special_tokens=False))