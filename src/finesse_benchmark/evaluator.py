from typing import Dict, Any, Optional
import torch
import time
from datasets import load_dataset
import numpy as np
import litellm
import typer
import tiktoken
import warnings
from transformers import AutoTokenizer
import importlib.metadata
import cpuinfo 

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

        # Pre-flight Qualification Check: Validate max_context_length based on mode
        if self.config.mode == 'merger_mode':
            base_embedder_config = self.config.models.base_embedder
            if base_embedder_config.max_context_length is None:
                raise ValueError("merger_mode requires 'models.base_embedder.max_context_length' to be set in config.")
            token_per_sample = self.config.probe_config.token_per_sample
            if token_per_sample > base_embedder_config.max_context_length:
                raise ValueError(
                    f"probe_config.token_per_sample ({token_per_sample}) exceeds base_embedder.max_context_length ({base_embedder_config.max_context_length}). "
                    f"Adjust token_per_sample or use a model with longer context."
                )
        elif self.config.mode == 'native_mode':
            native_embedder_config = self.config.models.native_embedder
            if native_embedder_config is None or native_embedder_config.max_context_length is None:
                raise ValueError("native_mode requires 'models.native_embedder.max_context_length' to be set in config.")
            # Additional check: Estimate max total tokens (sequence_length.max * token_per_sample)
            max_seq_len = self.config.probe_config.sequence_length.max
            estimated_max_tokens = max_seq_len * self.config.probe_config.token_per_sample + 100  # +overhead for safety
            if estimated_max_tokens > native_embedder_config.max_context_length:
                warnings.warn(
                    f"Maximum sequence may exceed native_embedder context: estimated {estimated_max_tokens} > {native_embedder_config.max_context_length}. "
                    f"Samples exceeding limit will be automatically skipped during evaluation."
                )
        elif self.config.mode == 'byok_mode':
            byok_embedder_config = self.config.models.byok_embedder
            if byok_embedder_config is None or byok_embedder_config.max_context_length is None:
                raise ValueError("byok_mode requires 'models.byok_embedder.max_context_length' to be set in config.")
            # Similar warning for BYOK
            max_seq_len = self.config.probe_config.sequence_length.max
            estimated_max_tokens = max_seq_len * self.config.probe_config.token_per_sample + 100  # +overhead
            if estimated_max_tokens > byok_embedder_config.max_context_length:
                warnings.warn(
                    f"Maximum sequence may exceed byok_embedder context: estimated {estimated_max_tokens} > {byok_embedder_config.max_context_length}. "
                    f"Samples exceeding limit will be automatically skipped during evaluation."
                )

    def raw_run(self) -> Dict[str, Any]:
        """Finesse 벤치마크 실행: Stratified CSAT with Single-Pass Conveyor Belt (Raw mode - embeddings only)"""
        # Load dataset with specific revision for declarative reproducibility
        dataset = load_dataset(
            path=self.config.dataset.path,
            split=self.config.dataset.split,
            revision=self.config.dataset.commit_hash
        )

        # Capture dataset metadata from config declaration
        dataset_metadata = {
            'path': self.config.dataset.path,
            'split': self.config.dataset.split,
            'commit_hash': self.config.dataset.commit_hash
        }

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
            # Pre-length check: Estimate if feasible
            estimated_tokens = target_length * self.config.probe_config.token_per_sample + 100  # +overhead for joins/spaces
            skip_length = False
            if self.config.mode != 'merger_mode':
                if self.config.mode == 'native_mode':
                    max_ctx = self.config.models.native_embedder.max_context_length
                else:  # byok_mode
                    max_ctx = self.config.models.byok_embedder.max_context_length
                if estimated_tokens > max_ctx:
                    typer.echo(f"Skipping entire length {target_length}: estimated {estimated_tokens} tokens > {max_ctx} limit.")
                    length_results[target_length] = {
                        'sample_results': [],
                        'num_synth_steps': target_length,
                        'skipped': True  # Flag for post-processing
                    }
                    skip_length = True
                    continue

            if skip_length:
                continue

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
                is_valid_sample = True
                
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

                # After building, runtime check (for exact after estimate)
                valid_sample = True
                if self.config.mode != 'merger_mode':
                    full_text = ' '.join(chunk_texts)
                    total_tokens = self.embedder.count_tokens(full_text)
                    
                    if self.config.mode == 'native_mode':
                        max_ctx = self.config.models.native_embedder.max_context_length
                    else:  # byok_mode
                        max_ctx = self.config.models.byok_embedder.max_context_length
                    
                    if total_tokens > max_ctx:
                        typer.echo(f"Skipping sample for length {target_length}: {total_tokens} tokens > {max_ctx} limit.")
                        valid_sample = False
                        is_valid_sample = False

                if not valid_sample:
                    continue  # Now safely continue outer loop, advancing to next attempt
                
                # 청크들 임베딩 with timing
                if self.config.mode == 'merger_mode':
                    chunk_times = []
                    chunk_embeddings_list = []
                    use_gpu_timing = torch.cuda.is_available()
                    for text in chunk_texts:
                        single_texts = [text]
                        if use_gpu_timing:
                            start = torch.cuda.Event(enable_timing=True)
                            start.record()
                        else:
                            start_cpu = time.perf_counter()
                        single_emb_tensor = self.embedder.encode(single_texts)
                        if use_gpu_timing:
                            end = torch.cuda.Event(enable_timing=True)
                            end.record()
                            torch.cuda.synchronize()
                            elapsed_ms = start.elapsed_time(end)
                        else:
                            end_cpu = time.perf_counter()
                            elapsed_ms = (end_cpu - start_cpu) * 1000
                        chunk_times.append(elapsed_ms)
                        chunk_embeddings_list.append(single_emb_tensor[0].cpu())
                    chunk_embeddings = chunk_embeddings_list
                else:
                    chunk_times = None
                    chunk_embeddings_tensor = self.embedder.encode(chunk_texts)
                    chunk_embeddings = [chunk_embeddings_tensor[i].cpu() for i in range(chunk_embeddings_tensor.size(0))]
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                use_gpu_timing = torch.cuda.is_available()
                synthesis_embeddings = []
                synth_times = []
                cumulative_embeddings = torch.empty((0, chunk_embeddings[0].shape[0]), dtype=chunk_embeddings[0].dtype, device=device)

                cpu_elapsed_list = []
                synth_start_events = []
                synth_end_events = []

                for i, emb in enumerate(chunk_embeddings):
                    single_on_device = emb.to(device)
                    cumulative_embeddings = torch.cat([cumulative_embeddings, single_on_device.unsqueeze(0)], dim=0)
                    partial_embs = cumulative_embeddings.unsqueeze(0)  # (1, i+1, D)
                    if use_gpu_timing:
                        start = torch.cuda.Event(enable_timing=True)
                        start.record()
                        synth_start_events.append(start)
                        synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                        synth_end_events.append(end)
                    else:
                        start_cpu = time.perf_counter()
                        synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                        end_cpu = time.perf_counter()
                        elapsed_ms = (end_cpu - start_cpu) * 1000
                        cpu_elapsed_list.append(elapsed_ms)
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
                
                # Compute synth_times
                if use_gpu_timing:
                    torch.cuda.synchronize()
                    for i in range(len(chunk_embeddings)):
                        elapsed_ms = synth_start_events[i].elapsed_time(synth_end_events[i])
                        synth_times.append(0.0 if i == 0 else elapsed_ms)
                else:
                    for i in range(len(chunk_embeddings)):
                        elapsed_ms = cpu_elapsed_list[i]
                        synth_times.append(0.0 if i == 0 else elapsed_ms)

                # 샘플 결과 저장
                sample_dict = {
                    'chunk_embeddings': chunk_embeddings,
                    'synthesis_embeddings': synthesis_embeddings,
                    'chunk_times': chunk_times,
                    'synth_times': synth_times
                }
                sample_results.append(sample_dict)

            # 이 길이에 대한 결과 저장
            length_results[target_length] = {
                'sample_results': sample_results,
                'num_synth_steps': target_length
            }

        # Add metadata for provenance
        try:
            version = importlib.metadata.version('finesse-benchmark')
            package_metadata = {'finesse-benchmark': version}
        except importlib.metadata.PackageNotFoundError:
            package_metadata = {'finesse-benchmark': 'unknown'}
        except Exception as e:
            package_metadata = {'finesse-benchmark': f'error: {str(e)}'}

        # Add device information for hardware provenance
        device_info = {}
        # CPU info
        try:
            cpu_info = cpuinfo.get_cpu_info()
            device_info['cpu'] = cpu_info.get('brand_raw', 'unknown')
        except Exception as e:
            device_info['cpu'] = f'error: {str(e)}'
        # GPU info
        if torch.cuda.is_available():
            try:
                device_info['gpu'] = torch.cuda.get_device_name(0)
                device_info['cuda_version'] = torch.version.cuda
            except Exception as e:
                device_info['gpu'] = f'cuda error: {str(e)}'
                device_info['cuda_version'] = 'unknown'
        else:
            device_info['gpu'] = 'none'
        # Torch version
        device_info['torch_version'] = torch.__version__

        metadata = {
            'package_versions': package_metadata,
            'dataset': dataset_metadata,
            'config': self.config.model_dump(),
            'device_info': device_info
        }

        return {
            'config': self.config.model_dump(),
            'metadata': metadata,
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