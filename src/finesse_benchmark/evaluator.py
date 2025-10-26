from typing import Dict, Any, Optional
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import litellm
import typer

from .config import BenchmarkConfig
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up

class FinesseEvaluator:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}  # Dict to hold loaded models
        self._load_models()

    def _load_models(self):
        """Load models based on config mode."""
        models_cfg = self.config.models

        # Common loading for embedders
        def load_embedder(key: str):
            model_name = getattr(models_cfg, key).name
            tokenizer, model = self._load_embedder(model_name)
            self.models[key] = {"tokenizer": tokenizer, "model": model}

        if self.config.mode == "merger_mode":
            # Load merger model
            merger_name = models_cfg.merger.name
            self.models["merger"] = AutoModel.from_pretrained(
                merger_name,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device).eval()
            
            # Load base embedder
            load_embedder("base_embedder")
        
        elif self.config.mode == "native_mode":
            # Load native long-context embedder
            load_embedder("native_embedder")

        elif self.config.mode == "byok_mode":
            # Load tokenizer for probe assembly using a default multilingual model (not used for individual beads)
            tokenizer_path = "intfloat/multilingual-e5-base"
            self.models["probe_tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
        
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

    def _load_embedder(self, model_path: str):
        """Load embedder model and tokenizer from HF path."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device).eval()
        
        return tokenizer, model

    def _get_embedding(self, text: str, embedder_key: Optional[str] = None) -> torch.Tensor:
        """Get embedding for text using the specified embedder."""
        if self.config.mode == "byok_mode":
            if self.config.models.byok_embedder is None:
                raise ValueError("BYOK mode requires 'models.byok_embedder' configuration.")
            provider = self.config.models.byok_embedder.provider
            model_name = self.config.models.byok_embedder.name
            litellm_model = f"{provider}/{model_name}"
            response = litellm.embedding(model=litellm_model, input=[text])
            embedding_list = response.data[0]['embedding']
            embedding = torch.tensor(embedding_list, dtype=torch.float32)
            embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
            return embedding
        else:
            if embedder_key is None:
                raise ValueError("Embedder key required for local embedding modes")
            embedder = self.models[embedder_key]
            tokenizer = embedder["tokenizer"]
            model = embedder["model"]
            
            # Prefix based on model (e.g., "passage: " for E5)
            prefix = "passage: " if "e5" in embedder_key.lower() else ""
            input_text = prefix + text
            
            inputs = tokenizer(
                [input_text],
                max_length=512,  # Adjust based on model
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)  # Universal mean pooling for all AutoModels
            
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze(0)
            return embedding.cpu().to(torch.float32)

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

        # Dynamically determine the embedder key or None for BYOK
        if self.config.mode == 'byok_mode':
            embedder_key = None
        elif self.config.mode == 'merger_mode':
            embedder_key = 'base_embedder'
        elif self.config.mode == 'native_mode':
            embedder_key = 'native_embedder'
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

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

                    # Chunk embeddings: embed the first N beads individually
                    chunk_texts = beads[:target_length]
                    chunk_embeddings = [
                        self._get_embedding(bead_text, embedder_key) for bead_text in chunk_texts
                    ]  # List of N 1D Tensors

                    # Synthesis embeddings: cumulative synthesis using partial chunk stacks
                    synthesis_embeddings = []
                    for i in range(1, target_length + 1):
                        partial_embs = chunk_embeddings[:i]
                        if self.config.mode == 'merger_mode':
                            merger = self.models['merger'].to(self.device).eval()
                            src = torch.stack(partial_embs).unsqueeze(0).to(self.device)  # (1, i, D)
                            dtype = next(merger.parameters()).dtype
                            with torch.no_grad():
                                outputs = merger(src.to(dtype))
                                if hasattr(outputs, 'pooler_output'):
                                    synth_emb = outputs.pooler_output.squeeze(0)
                                elif hasattr(outputs, 'last_hidden_state'):
                                    synth_emb = outputs.last_hidden_state.squeeze(0).mean(dim=0)
                                else:
                                    synth_emb = outputs[0].squeeze(0) if isinstance(outputs, tuple) else outputs.squeeze(0)
                            synth_emb = synth_emb.cpu().to(torch.float32)
                        else:  # native_mode or fallback: mean pooling
                            synth_emb = torch.stack(partial_embs).mean(dim=0).cpu().to(torch.float32)
                        
                        # Normalize
                        synth_emb = torch.nn.functional.normalize(synth_emb, p=2, dim=0)
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