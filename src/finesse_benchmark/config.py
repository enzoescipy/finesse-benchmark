from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class SequenceLengthConfig(BaseModel):
    """시퀀스 길이 범위 설정"""
    min: int = Field(..., ge=1, description="최소 시퀀스 길이")
    max: int = Field(..., ge=1, description="최대 시퀀스 길이")

    class Config:
        validate_assignment = True

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min must be <= max")

class AutoModelSelector(BaseModel):
    """hf 모델 설정"""
    name: str = Field(..., description="모델 카드 이름")

class ProbeConfig(BaseModel):
    """프로브 생성 설정"""
    mask_ratio: float = Field(default=0.15, description="Masking 비율")
    sequence_length: SequenceLengthConfig = Field(default=SequenceLengthConfig(min=4, max=16), description="시퀀스 길이 범위. min부터 max까지 순차적으로 평가.")
    samples_per_length: int = Field(default=10, description="각 시퀀스 길이에 대해 평가할 샘플 개수. Stratified CSAT 모드에서 사용.")

class ModelsConfig(BaseModel):
    merger: AutoModelSelector = Field(default=AutoModelSelector(name="enzoescipy/sequence-merger-tiny"), description="merger_mode용 모델 설정")
    base_embedder: AutoModelSelector = Field(default=AutoModelSelector(name="intfloat/multilingual-e5-base"), description="기본 임베더 설정")
    native_embedder: AutoModelSelector = Field(default=AutoModelSelector(name="Snowflake/snowflake-arctic-embed-l-v2.0"),  description="native_mode용 임베더 설정")

class DatasetConfig(BaseModel):
    """데이터셋 설정"""
    path: str = Field(default="enzoescipy/finesse-benchmark-database", description="HF 데이터셋 경로")
    split: str = Field(default="train")
    num_samples: int = Field(default=10000)

class OutputConfig(BaseModel):
    format: str = Field(default="json")
    sign: bool = Field(default=True)

class BenchmarkConfig(BaseModel):
    mode: str = Field(default="merger_mode", description="merger_mode 또는 native_mode")
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    probe_config: ProbeConfig = Field(default_factory=ProbeConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    byok: Dict[str, str] = Field(default_factory=dict, description="API 키 등")
    advanced: Dict[str, Any] = Field(default_factory=dict, description="고급 옵션 (batch_size, device 등)")

    class Config:
        json_schema_extra = {"example": {
            "mode": "merger_mode",
            "models": {
                "merger": {"path": "enzoescipy/sequence-merger-tiny"},
                "base_embedder": {"path": "intfloat/multilingual-e5-base"},
            },
            "probe_config": {
                "mask_ratio": 0.15,
                "sequence_length": {"min": 5, "max": 16},
                "samples_per_length": 1,
            },
            # ... 기타 필드
        }}