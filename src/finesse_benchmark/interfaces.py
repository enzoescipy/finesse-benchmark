from abc import ABC, abstractmethod
from typing import Optional
import torch


class FinesseEmbedder(ABC):
    """
    finesse-benchmark 시스템에 장착될 모든 '임베딩 엔진'이 따라야 할 표준 규격.
    이 엔진은 텍스트를 벡터로 변환한다.
    """
    
    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        문자열 리스트를 받아서, [N, D_model] 형태의 텐서를 반환한다.
        
        Args:
            texts: 임베딩할 문자열들의 리스트
            
        Returns:
            torch.Tensor: [N, D_model] 형태의 정규화된 임베딩 텐서
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        이 엔진이 어떤 장치(cpu, cuda) 위에서 돌아가는지 반환한다.
        
        Returns:
            torch.device: 이 엔진이 사용하는 장치
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        주어진 텍스트의 토큰 수를 반환한다.
        
        Args:
            text: 토큰 수를 셀 텍스트
            
        Returns:
            int: 텍스트의 토큰 수
        """
        pass

    @abstractmethod
    def chunk_text(self, text: str, max_tokens: int) -> str:
        """
        주어진 텍스트를 최대 토큰 수만큼 정확히 잘라내서 반환한다.
        
        Args:
            text: 자를 텍스트
            max_tokens: 최대 토큰 수
            
        Returns:
            str: 정확히 max_tokens 만큼 잘라낸 텍스트 조각
        """
        pass


class FinesseSynthesizer(ABC):
    """
    finesse-benchmark 시스템에 장착될 모든 '합성 엔진(merger)'이 따라야 할 표준 규격.
    이 엔진은 벡터의 시퀀스를 입력받아 하나의 벡터로 합성한다.
    """
    
    @abstractmethod
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        [N, Sequence, Dim] 형태의 임베딩 시퀀스를 입력받아,
        [N, Dim] 형태의 합성된 임베딩을 반환한다.
        
        Args:
            embeddings: [N, Sequence, Dim] 형태의 임베딩 시퀀스
            
        Returns:
            torch.Tensor: [N, Dim] 형태의 정규화된 합성 임베딩 텐서
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """
        이 엔진이 어떤 장치(cpu, cuda) 위에서 돌아가는지 반환한다.
        
        Returns:
            torch.device: 이 엔진이 사용하는 장치
        """
        pass