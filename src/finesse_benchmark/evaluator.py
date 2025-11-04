from typing import Dict, Any, Optional, List
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
        self.chunk_concat_sep = ' '

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



    def native_run(self) -> Dict[str, Any]:
        """Finesse 벤치마크 실행: Stratified CSAT with Single-Pass Conveyor Belt (Raw mode - embeddings only)"""
        
        dataset = self._dataset_prepare_and_validate()
        iterator = iter(dataset)

        # Find ctx
        max_ctx = None
        if self.config.mode == 'native_mode':
            max_ctx = self.config.models.native_embedder.max_context_length
        else:  # byok_mode
            max_ctx = self.config.models.byok_embedder.max_context_length

        # find min, max length
        min_length, max_length = self.config.probe_config.sequence_length.min, self.config.probe_config.sequence_length.max
        
        # Warm-up phase: Initialize models with dummy data to avoid cold-start latency
        dummy_samples = [
            "This is a warm-up dummy sentence 1.",
            "This is a warm-up dummy sentence 2."
        ]
        _ = self.embedder.encode(dummy_samples)

        # All warm-up results discarded; models now warmed up

        length_results = {}  # 길이별 결과 저장: {'sample_results': [dicts], 'num_synth_steps': N}

        for target_length in range(min_length, max_length + 1):
            # Pre-length check: Estimate if feasible
            estimated_tokens = target_length * self.config.probe_config.token_per_sample + 100  # +overhead for joins/spaces
            skip_length = False
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
            
            sample_results = []  
            attempts = 0
            max_attempts = len(dataset)  # 데이터셋 전체를 최대 시도 횟수로
            
            # target_length 개의 청크로 구성된 시퀀스를 samples_per_length번 테스트
            while len(sample_results) < self.config.probe_config.samples_per_length:
                attempts += 1
                if attempts > max_attempts:
                    raise ValueError(f"데이터셋 소진: target_length={target_length}에서 충분한 샘플을 찾을 수 없음. {len(sample_results)}/{self.config.probe_config.samples_per_length}개만 생성됨.")
                
                typer.echo(f"probe sequence [{target_length}] in progress ({len(sample_results)}/{self.config.probe_config.samples_per_length})...")

                # 이 테스트를 위해 target_length 개의 청크를 생성
                chunk_texts = self._get_text_chunck_from_database(target_length=target_length, dataset=dataset, iterator=iterator)

                # After building, runtime check (for exact after estimate)
                valid_sample = True
                full_text = self.chunk_concat_sep.join(chunk_texts)
                total_tokens = self.embedder.count_tokens(full_text)
                
                if total_tokens > max_ctx:
                    typer.echo(f"Skipping sample for length {target_length}: {total_tokens} tokens > {max_ctx} limit.")
                    valid_sample = False

                if not valid_sample:
                    continue  # Now safely continue outer loop, advancing to next attempt
                
                # 청크들 임베딩 without timing
                chunk_times = None
                chunk_embeddings_tensor = self.embedder.encode(chunk_texts)
                chunk_embeddings = [chunk_embeddings_tensor[i].cpu() for i in range(chunk_embeddings_tensor.size(0))]

                # 청크 합성 with timing: Embed cumulative texts progressively
                synthesis_embeddings = []
                synth_times = []
                cumulative_text = ""

                for i, text in enumerate(chunk_texts):
                    if i > 0:
                        cumulative_text += self.chunk_concat_sep
                    cumulative_text += text

                    start_time = time.monotonic()
                    synth_emb_tensor = self.embedder.encode([cumulative_text])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.monotonic()
                    elapsed_ms = (end_time - start_time) * 1000
                    synth_times.append(elapsed_ms)  # Record actual time for each cumulative embedding

                    synth_emb_cpu = synth_emb_tensor[0].cpu()  # Take the first (and only) embedding

                    # Validate embedder output: must be 1D tensor (d_model,)
                    if synth_emb_cpu.dim() != 1:
                        raise ValueError(
                            f"Embedder contract violation: encode() must return a 1D tensor per input, "
                            f"but instead returned a {synth_emb_cpu.dim()}-dimensional tensor with shape {synth_emb_cpu.shape}. "
                            "Please check your embedder's output format."
                        )

                    synthesis_embeddings.append(synth_emb_cpu)

                    # Clean up GPU tensors to prevent memory accumulation
                    del synth_emb_tensor

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

        return self._package_metadata_data(length_results=length_results)

    def merger_run(self) -> Dict[str, Any]:
        """Finesse 벤치마크 실행: Stratified CSAT with Single-Pass Conveyor Belt (Raw mode - embeddings only)"""
        # Load dataset with specific revision for declarative reproducibility
        dataset = self._dataset_prepare_and_validate()
        iterator = iter(dataset)

        # find min, max length
        min_length, max_length = self.config.probe_config.sequence_length.min, self.config.probe_config.sequence_length.max


        # Warm-up phase: Initialize models with dummy data to avoid cold-start latency
        dummy_samples = [
            "This is a warm-up dummy sentence 1.",
            "This is a warm-up dummy sentence 2."
        ]

        # Warm-up embedder
        _ = self.embedder.encode(dummy_samples)
        # Warm-up synthesizer incrementally
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dummy_embs = self.embedder.encode(dummy_samples)
        cumulative = torch.empty((0, dummy_embs.shape[1]), dtype=dummy_embs.dtype, device=device)
        for emb in dummy_embs:
            single_on_device = emb.unsqueeze(0).to(device)
            cumulative = torch.cat([cumulative, single_on_device], dim=0)
            partial_embs = cumulative.unsqueeze(0)  # (1, i+1, D)
            _ = self.synthesizer.synthesize(partial_embs).squeeze(0)
        del cumulative  # Cleanup

        # All warm-up results discarded; models now warmed up

        length_results = {}  # 길이별 결과 저장: {'sample_results': [dicts], 'num_synth_steps': N}

        for target_length in range(min_length, max_length + 1):
            typer.echo(f"probe sequence [{target_length}] in progress ...")
            
            sample_results = []  # List of 25 dicts per length
            attempts = 0
            max_attempts = len(dataset)  # 데이터셋 전체를 최대 시도 횟수로
            
            # target_length 개의 청크로 구성된 시퀀스를 samples_per_length번 테스트
            while len(sample_results) < self.config.probe_config.samples_per_length:
                attempts += 1
                if attempts > max_attempts:
                    raise ValueError(f"데이터셋 소진: target_length={target_length}에서 충분한 샘플을 찾을 수 없음. {len(sample_results)}/{self.config.probe_config.samples_per_length}개만 생성됨.")
                
                typer.echo(f"probe sequence [{target_length}] in progress ({len(sample_results)}/{self.config.probe_config.samples_per_length})...")

                # 이 테스트를 위해 target_length 개의 청크를 생성
                chunk_texts = self._get_text_chunck_from_database(target_length=target_length, dataset=dataset, iterator=iterator)

                # 청크들 임베딩 with timing
                chunk_times = []
                chunk_embeddings_list = []
                for text in chunk_texts:
                    single_texts = [text]
                    start_time = time.monotonic()
                    single_emb_tensor = self.embedder.encode(single_texts)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.monotonic()
                    elapsed_ms = (end_time - start_time) * 1000
                    chunk_times.append(elapsed_ms)
                    chunk_embeddings_list.append(single_emb_tensor[0].cpu())
                chunk_embeddings = chunk_embeddings_list

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                synthesis_embeddings = []
                synth_times = []
                cumulative_embeddings = torch.empty((0, chunk_embeddings[0].shape[0]), dtype=chunk_embeddings[0].dtype, device=device)

                for i, emb in enumerate(chunk_embeddings):
                    single_on_device = emb.to(device)
                    cumulative_embeddings = torch.cat([cumulative_embeddings, single_on_device.unsqueeze(0)], dim=0)
                    partial_embs = cumulative_embeddings.unsqueeze(0)  # (1, i+1, D)
                    start_time = time.monotonic()
                    synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.monotonic()
                    elapsed_ms = (end_time - start_time) * 1000
                    synth_times.append(0.0 if i == 0 else elapsed_ms)
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

        return self._package_metadata_data(length_results=length_results)



    def merger_run_srs(self) -> Dict[str, Any]:
        """Finesse 벤치마크 실행: Stratified CSAT with Single-Pass Conveyor Belt (Raw mode - embeddings only)"""
        # Load dataset with specific revision for declarative reproducibility
        dataset = self._dataset_prepare_and_validate()
        iterator = iter(dataset)

        # find min, max length
        min_length, max_length = self.config.probe_config.sequence_length.min, self.config.probe_config.sequence_length.max


        # Warm-up phase: Initialize models with dummy data to avoid cold-start latency
        dummy_samples = [
            "This is a warm-up dummy sentence 1.",
            "This is a warm-up dummy sentence 2."
        ]

        # Warm-up embedder
        _ = self.embedder.encode(dummy_samples)
        # Warm-up synthesizer incrementally
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dummy_embs = self.embedder.encode(dummy_samples)
        cumulative = torch.empty((0, dummy_embs.shape[1]), dtype=dummy_embs.dtype, device=device)
        for emb in dummy_embs:
            single_on_device = emb.unsqueeze(0).to(device)
            cumulative = torch.cat([cumulative, single_on_device], dim=0)
            partial_embs = cumulative.unsqueeze(0)  # (1, i+1, D)
            _ = self.synthesizer.synthesize(partial_embs).squeeze(0)
        del cumulative  # Cleanup

        # All warm-up results discarded; models now warmed up

        length_results = {}  # 길이별 결과 저장: {'sample_results': [dicts], 'num_synth_steps': N}

        for target_length in range(min_length, max_length + 1):
            typer.echo(f"probe sequence [{target_length}] in progress ...")
            
            sample_results = []  # List of 25 dicts per length

            # target_length 개의 청크로 구성된 시퀀스를 samples_per_length번 테스트
            while len(sample_results) < self.config.probe_config.samples_per_length:
                
                # testing probe chunks yield
                test_probe_chunks = self._get_text_chunck_from_database(target_length=target_length - 1, dataset=dataset, iterator=iterator)
                
                arbitual_probe_group = []
                current_probe = []
                for i in range(len(test_probe_chunks)):
                    current_probe.append(test_probe_chunks[i])
                    arbitual_probe_group.append(tuple(current_probe))

                # Batch embed all probe chunks for efficiency
                chunk_embeddings_tensor = self.embedder.encode(test_probe_chunks)
                probe_chunk_embeddings = [chunk_embeddings_tensor[i].cpu() for i in range(chunk_embeddings_tensor.size(0))]

                # Synthesize cumulative embeddings progressively, starting from length 2
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                synthesis_probe_embeddings = []
                cumulative_embeddings = torch.empty((0, probe_chunk_embeddings[0].shape[0]), dtype=probe_chunk_embeddings[0].dtype, device=device)

                for i, emb in enumerate(probe_chunk_embeddings):
                    single_on_device = emb.to(device).unsqueeze(0)
                    cumulative_embeddings = torch.cat([cumulative_embeddings, single_on_device], dim=0)
                    
                    if i >= 1:  # Skip synthesis for i=0 (length 1 probe), start from length 2
                        partial_embs = cumulative_embeddings.unsqueeze(0)  # (1, i+1, D)
                        synth_emb = self.synthesizer.synthesize(partial_embs).squeeze(0)
                        synth_emb_cpu = synth_emb.cpu()

                        # Validate synthesizer output
                        if synth_emb_cpu.dim() != 1:
                            raise ValueError(
                                f"Synthesizer contract violation: synthesize() must return a 1D tensor, "
                                f"but instead returned a {synth_emb_cpu.dim()}-dimensional tensor with shape {synth_emb_cpu.shape}. "
                                "Please check your custom synthesizer's output format."
                            )

                        synthesis_probe_embeddings.append(synth_emb_cpu)

                        # Clean up GPU tensors
                        del partial_embs, synth_emb
                    
                    # Clean up single_on_device regardless
                    del single_on_device

                del cumulative_embeddings  # Final cleanup

                # test group factory
                max_n_gram_len = (target_length - 2)
                n_gram_memory_chunks = self._get_text_chunck_from_database(target_length=max_n_gram_len * 2 * self.config.probe_config.group_amount, dataset=dataset, iterator=iterator)
                arbitual_n_gram_memory_positive = []
                arbitual_n_gram_memory_negative = []
                for i in range(2 * self.config.probe_config.group_amount):
                    n_gram = []
                    for j in range(max_n_gram_len):
                        n_gram.append(n_gram_memory_chunks[i * max_n_gram_len + j])
                    if (i % 2 == 0):
                        arbitual_n_gram_memory_positive.append(n_gram)
                    else:
                        arbitual_n_gram_memory_negative.append(n_gram)

                # result dict creation
                probe_len_unit_sets = {}
                for probe_len in range(2, target_length):
                    arbitual_probe = arbitual_probe_group[probe_len - 1]
                    probe_pos_unit_sets = {}

                    for probe_pos in range(target_length - probe_len):
                        arbitual_positive_group = []
                        arbitual_negative_group = []

                        for positive_n_gram in arbitual_n_gram_memory_positive:
                            crafted = positive_n_gram
                            for chunk in reversed(arbitual_probe):
                                crafted.insert(probe_pos, chunk)
                            arbitual_positive_group.append(crafted)

                        for negative_n_gram in arbitual_n_gram_memory_negative:
                            crafted = negative_n_gram
                            for chunk in arbitual_probe:
                                crafted.insert(probe_pos, chunk)
                            arbitual_negative_group.append(crafted)

                        ## TODO 1) embed the arbitual_positive_group and arbitual_negative_group

                        ## TODO psudo-code
                        probe_pos_unit_sets[str(probe_pos)] = {
                                "positive_embeddings" : [],
                                "negative_embeddings" : []
                            }
                    
                    probe_len_unit_sets[str(probe_len)] = {
                        "probe_embedding": "psudo-embedding",
                        "probe_pos_embeddings": probe_pos_unit_sets
                    }   

                    # pop the n_gram_memory
                    for positive_n_gram in arbitual_n_gram_memory_positive:
                        positive_n_gram.pop()
                    for negative_n_gram in arbitual_n_gram_memory_negative:
                        negative_n_gram.pop

                
                # 샘플 결과 저장
                sample_dict = probe_len_unit_sets
                sample_results.append(sample_dict)

            # 이 길이에 대한 결과 저장
            length_results[target_length] = {
                'sample_results': sample_results,
                'num_synth_steps': target_length
            }

        return self._package_metadata_data(length_results=length_results)














    def _package_metadata_data(self, length_results:Dict[int, Any]) -> Dict[str, Any]:
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

        # Capture dataset metadata from config declaration
        dataset_metadata = {
            'path': self.config.dataset.path,
            'split': self.config.dataset.split,
            'commit_hash': self.config.dataset.commit_hash
        }

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


    def _dataset_prepare_and_validate(self) -> Any:
        # Load dataset with specific revision for declarative reproducibility
        dataset = load_dataset(
            path=self.config.dataset.path,
            split=self.config.dataset.split,
            revision=self.config.dataset.commit_hash
        )

        # Shuffle dataset deterministically for reproducibility
        if self.config.seed is not None:
            dataset = dataset.shuffle(seed=self.config.seed)

        min_length, max_length = self.config.probe_config.sequence_length.min, self.config.probe_config.sequence_length.max
        total_needed_samples = (max_length - min_length + 1) * self.config.probe_config.samples_per_length
        if len(dataset) < total_needed_samples:
            raise ValueError(f"데이터셋 크기({len(dataset)})가 필요 샘플({total_needed_samples})보다 작음. 더 많은 데이터 필요.")

        return dataset

    def _get_text_chunck_from_database(self, target_length:int, dataset:Any, iterator:Any) -> List[str]:
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
                        chunk_texts.append(self.chunk_concat_sep.join(current_chunk))
                        chunk_token_count += target_token_size
                        break
                    
                    # 청크에 구슬 추가
                    current_chunk.append(bead_text)
                    current_token_count += bead_token_count
                    
                    # 정확히 타겟 토큰 수에 도달하면 청크 완성
                    if current_token_count == target_token_size:
                        chunk_texts.append(self.chunk_concat_sep.join(current_chunk))
                        chunk_token_count += target_token_size
                        break
                

            except (StopIteration, KeyError):
                # 이터레이터가 끝나면 다시 시작
                iterator = iter(dataset)
                continue

        
        return chunk_texts

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