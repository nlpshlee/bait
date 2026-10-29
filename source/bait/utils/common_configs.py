from _init import *

from dataclasses import dataclass, asdict


@dataclass
class BaseModelConfig:
    dtype: str = 'bfloat16'
    device_map: str = 'auto'
    attn_imp: str = 'eager'
    max_seq_length: int = 4096
    max_new_tokens: int = 64

    # 디코딩 옵션
    do_sample: bool = False
    temperature: float = None
    top_k: int = None
    top_p: float = None

    # 내부 상태 추출 옵션 플래그
    output_logits: bool = False
    output_scores: bool = False
    output_hidden_states: bool = False
    output_attentions: bool = False


    # def __post_init__(self):
    #     pass


    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f'# [ERROR] ModelConfig.update() [{key}]는 존재하지 않는 속성입니다.')


    def to_dict(self):
        return asdict(self)

