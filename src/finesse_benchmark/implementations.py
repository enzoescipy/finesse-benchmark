from typing import Optional
import torch
from transformers import AutoModel, AutoTokenizer
import litellm
import tiktoken
import warnings

from .interfaces import FinesseEmbedder, FinesseSynthesizer


class HuggingFaceEmbedder(FinesseEmbedder):
    """
    Hugging Face 모델을 사용하는 임베딩 엔진 구현체.
    """
    
    def __init__(self, model_path: str, prefix: Optional[str] = None):
        """
        Args:
            model_path: Hugging Face 모델 경로
            prefix: 임베딩 전 텍스트에 추가할 접두사 (예: "passage: " for E5)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.prefix = prefix or ""
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device).eval()
    
    def encode(self, texts: list[str]) -> torch.Tensor:
        """텍스트 리스트를 임베딩하여 반환한다."""
        # Add prefix if specified
        input_texts = [self.prefix + text for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token) for e5-style models
            embeddings = outputs.last_hidden_state[:, 0]
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().to(torch.float32)
    
    @property
    def device(self) -> torch.device:
        return self.device

    def count_tokens(self, text: str) -> int:
        """주어진 텍스트의 토큰 수를 반환한다."""
        # Use the embedder's own tokenizer to count tokens
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str, max_tokens: int) -> str:
        """주어진 텍스트를 최대 토큰 수만큼 정확히 잘라내서 반환한다."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Take only the first max_tokens tokens
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        
        # Decode back to text
        return self.tokenizer.decode(tokens)


class ByokEmbedder(FinesseEmbedder):
    """
    BYOK (Bring Your Own Key) API를 사용하는 임베딩 엔진 구현체.
    """
    
    def __init__(self, provider: str, model_name: str, tokenizer_path: Optional[str] = None):
        """
        Args:
            provider: API 제공자 (예: "openai")
            model_name: 모델 이름 (예: "text-embedding-3-small")
            tokenizer_path: 토크나이저 경로 (선택사항)
        """
        self.provider = provider
        self.model_name = model_name
        self.litellm_model = f"{provider}/{model_name}"
        self.device = torch.device("cpu")  # BYOK는 CPU에서 작동
        
        # Store tokenizer configuration for count_tokens method
        self.tokenizer_path = tokenizer_path
        
        # Token counter setup
        self.token_counter = self._setup_token_counter(tokenizer_path)
    
    def _setup_token_counter(self, tokenizer_path: Optional[str] = None):
        """토큰 카운터를 설정한다."""
        # 1. De Facto Standardization: Use tiktoken for OpenAI models
        if self.provider == 'openai':
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                return lambda text: len(encoding.encode(text))
            except KeyError:
                warnings.warn(f"tiktoken encoding for {self.model_name} not found. Defaulting to general-purpose tokenizer.")
        
        # 2. Delegation of Autonomy: Use user-specified tokenizer
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        
        # 3. Fallback with clear warning
        warnings.warn(f"BYOK provider '{self.provider}' is not OpenAI and no 'tokenizer_path' was provided. Falling back to default tokenizer.")
        fallback_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        return lambda text: len(fallback_tokenizer.encode(text, add_special_tokens=False))
    
    def encode(self, texts: list[str]) -> torch.Tensor:
        """API를 통해 텍스트 리스트를 임베딩하여 반환한다."""
        # Call API
        response = litellm.embedding(model=self.litellm_model, input=texts)
        
        # Extract embeddings
        embedding_lists = [item['embedding'] for item in response.data]
        embeddings = torch.tensor(embedding_lists, dtype=torch.float32)
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    @property
    def device(self) -> torch.device:
        return self.device

    def count_tokens(self, text: str) -> int:
        """주어진 텍스트의 토큰 수를 반환한다."""
        # 1. De Facto Standardization: Use tiktoken for OpenAI models
        if self.provider == 'openai':
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
            except KeyError:
                warnings.warn(f"tiktoken encoding for {self.model_name} not found. Defaulting to general-purpose tokenizer.")
        
        # 2. Delegation of Autonomy: Use user-specified tokenizer
        if self.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            return len(tokenizer.encode(text, add_special_tokens=False))
        
        # 3. Fallback with clear warning
        warnings.warn(f"BYOK provider '{self.provider}' is not OpenAI and no 'tokenizer_path' was provided. Falling back to default tokenizer.")
        fallback_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        return len(fallback_tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str, max_tokens: int) -> str:
        """주어진 텍스트를 최대 토큰 수만큼 정확히 잘라내서 반환한다."""
        # 1. De Facto Standardization: Use tiktoken for OpenAI models
        if self.provider == 'openai':
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                tokens = encoding.encode(text)
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                return encoding.decode(tokens)
            except KeyError:
                warnings.warn(f"tiktoken encoding for {self.model_name} not found. Defaulting to general-purpose tokenizer.")
        
        # 2. Delegation of Autonomy: Use user-specified tokenizer
        if self.tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            return tokenizer.decode(tokens)
        
        # 3. Fallback with clear warning
        warnings.warn(f"BYOK provider '{self.provider}' is not OpenAI and no 'tokenizer_path' was provided. Falling back to default tokenizer.")
        fallback_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        tokens = fallback_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return fallback_tokenizer.decode(tokens)


class HuggingFaceSynthesizer(FinesseSynthesizer):
    """
    Hugging Face 합성 모델(merger)을 사용하는 합성 엔진 구현체.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Hugging Face 합성 모델 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device).eval()
    
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """임베딩 시퀀스를 합성하여 반환한다."""
        # Move to device if needed
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        # Get model dtype
        dtype = next(self.model.parameters()).dtype
        
        # Run through model
        with torch.no_grad():
            outputs = self.model(embeddings.to(dtype))
            
            # Handle different output formats
            if hasattr(outputs, 'pooler_output'):
                synth_emb = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                synth_emb = outputs.last_hidden_state.mean(dim=1)
            else:
                synth_emb = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Normalize
        synth_emb = torch.nn.functional.normalize(synth_emb, p=2, dim=1)
        return synth_emb.cpu().to(torch.float32)
    
    @property
    def device(self) -> torch.device:
        return self.device


class MeanPoolingSynthesizer(FinesseSynthesizer):
    """
    단순 평균 풀링을 사용하는 합성 엔진 구현체 (native_mode용).
    """
    
    def __init__(self):
        self.device = torch.device("cpu")  # CPU 연산
    
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """임베딩 시퀀스의 평균을 계산하여 반환한다."""
        # Mean pooling
        synth_emb = embeddings.mean(dim=1)
        
        # Normalize
        synth_emb = torch.nn.functional.normalize(synth_emb, p=2, dim=1)
        return synth_emb
    
    @property
    def device(self) -> torch.device:
        return self.device