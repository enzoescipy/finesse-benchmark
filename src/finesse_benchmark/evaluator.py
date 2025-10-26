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
            attempts = 0
            max_attempts = len(dataset)  # 데이터셋 전체를 최대 시도 횟수로
            
            # 진정한 원칙: target_length 개의 청크로 구성된 시퀀스를 samples_per_length번 테스트
            while len(sample_results) < self.config.probe_config.samples_per_length:
                attempts += 1
                if attempts > max_attempts:
                    raise ValueError(f"데이터셋 소진: target_length={target_length}에서 충분한 샘플을 찾을 수 없음. {len(sample_results)}/{self.config.probe_config.samples_per_length}개만 생성됨.")
                
                typer.echo(f"probe sequence [{target_length}] in progress ({len(sample_results)}/{self.config.probe_config.samples_per_length})...")
                
                # 이 테스트를 위해 target_length 개의 청크를 생성
                chunk_texts = []
                chunk_token_count = 0
                
                # target_length 개의 청크를 생성할 때까지 데이터셋에서 구슬을 가져옴
                while len(chunk_texts) < target_length:
                    try:
                        sample = next(iterator)
                        beads = sample['beads']
                        
                        # 구슬이 없으면 다음 샘플로
                        if not beads:
                            continue
                        
                        # 이 샘플에서 하나의 청크 생성
                        current_chunk = []
                        current_token_count = 0
                        target_token_size = self.config.probe_config.token_per_sample
                        
                        for bead_text in beads:
                            bead_token_count = self.embedder.count_tokens(bead_text)
                            
                            # 토큰 수가 초과하면 정확히 잘라서 청크 완성
                            if current_token_count + bead_token_count > target_token_size:
                                tokens_needed = target_token_size - current_token_count
                                if tokens_needed > 0:
                                    needed_text = self.embedder.chunk_text(bead_text, tokens_needed)
                                    current_chunk.append(needed_text)
                                
                                # 청크 완성
                                chunk_texts.append(' '.join(current_chunk))
                                chunk_token_count += target_token_size
                                break
                            
                            # 청크에 구슬 추가
                            current_chunk.append(bead_text)
                            current_token_count += bead_token_count
                            
                            # 정확히 타겟 토큰 수에 도달하면 청크 완성
                            if current_token_count == target_token_size:
                                chunk_texts.append(' '.join(current_chunk))
                                chunk_token_count += target_token_size
                                break
                        

                    
                    except (StopIteration, KeyError):
                        # 이터레이터가 끝나면 다시 시작
                        iterator = iter(dataset)
                        continue
                
                
                # 청크들 임베딩
                chunk_embeddings_tensor = self.embedder.encode(chunk_texts)
                chunk_embeddings = [chunk_embeddings_tensor[i].cpu() for i in range(chunk_embeddings_tensor.size(0))]
                
                # 누적 합성 수행: A, AB, ABC, ..., ABCDEFG (GPU Tour Optimization)
                device = chunk_embeddings_tensor.device
                synthesis_embeddings = []
                cumulative_embeddings = torch.empty((0, chunk_embeddings[0].shape[0]), dtype=chunk_embeddings[0].dtype, device=device)

                for emb in chunk_embeddings:
                    single_on_device = emb.to(device)
                    cumulative_embeddings = torch.cat([cumulative_embeddings, single_on_device.unsqueeze(0)], dim=0)
                    partial_embs = cumulative_embeddings.unsqueeze(0)  # (1, i, D)
                    synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                    synth_emb_cpu = synth_emb.cpu()

                    # Validate synthesizer output: must be 1D tensor (d_model,)
                    if synth_emb_cpu.dim() != 1:
                        raise ValueError(
                            f"Synthesizer contract violation: synthesize() must return a 1D tensor, "
                            f"but instead returned a {synth_emb_cpu.dim()}-dimensional tensor with shape {synth_emb_cpu.shape}. "
                            "Please check your custom synthesizer's output format."
                        )

                    synthesis_embeddings.append(synth_emb_cpu)

                    # Clean up GPU tensors to prevent memory accumulation
                    del single_on_device, partial_embs, synth_emb

                del cumulative_embeddings  # Final cleanup
                
                # 샘플 결과 저장
                sample_dict = {
                    'chunk_embeddings': chunk_embeddings,
                    'synthesis_embeddings': synthesis_embeddings
                }
                sample_results.append(sample_dict)

            # 이 길이에 대한 결과 저장
            length_results[target_length] = {
                'sample_results': sample_results,
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