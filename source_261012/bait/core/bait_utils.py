from _init import *

import random

from bait.core.bait_prompts import get_generate_prompt, get_generate_prompt_dummy_doc
from bait.utils import model_utils


class INTERVENTION_OPTION:
    OFF = 0
    DUMMY = 1
    MLP = 2
    ATTN = 4
    NOT = OFF | DUMMY
    ALL = MLP | ATTN


def get_model_name_or_path(model_name: str):
    if model_name.startswith(f'Llama'):
        model_name_or_path = f'meta-llama/{model_name}-Instruct'
    elif model_name.startswith('Qwen'):
        model_name_or_path = f'Qwen/{model_name}-Instruct'
    else:
        model_name_or_path = model_name

    return model_name_or_path


def mix_contexts(contexts_fact_dict: dict, contexts_counter_dict: dict, ext_n_fact: int, ext_n_counter: int):
    if ext_n_fact <= len(contexts_fact_dict) and ext_n_counter <= len(contexts_counter_dict):
        ext_contexts_fact = random.sample(list(contexts_fact_dict.values()), ext_n_fact)
        ext_contexts_counter = random.sample(list(contexts_counter_dict.values()), ext_n_counter)

        # 셔플 전 각각의 컨텍스트에 태그(출처)를 붙여 튜플 형태로 결합
        tagged_contexts = [(ctx, 'fact') for ctx in ext_contexts_fact] + [(ctx, 'counter') for ctx in ext_contexts_counter]

        # 태그를 붙인 상태에서 셔플
        random.shuffle(tagged_contexts)

        # 태그를 제거하고 위치 기록
        mixed_contexts = []
        fact_idxs, counter_idxs = [], []

        for i, (ctx, tag) in enumerate(tagged_contexts):
            mixed_contexts.append(ctx)
            if tag == 'fact':
                fact_idxs.append(i)
            else:
                counter_idxs.append(i)

        return mixed_contexts, fact_idxs, counter_idxs

    return None, None, None


def generate_dummy_docs(model, tokenizer, queries: list, max_seq_length: int,
                        max_new_tokens: int = 192, zero_shot_max_new_tokens: int = 64):
    '''
        더미 문서(모델이 직접 쓴 내재 지식 문서)를 queries 전체 한 배치로 생성한다. 정답 라벨은 쓰지 않고 모델 출력만 쓴다.
            1) zero-shot 답 : 문서 없는 get_generate_prompt (zero-shot 분할을 정할 때와 같은 프롬프트) 로 한 번에 생성
            2) 더미        : (질문, zero-shot 답) 쌍마다 그 답을 사실처럼 서술하는 50~80단어 문단을 한 번에 생성
        배치로 생성하면 함께 묶인 질문(왼쪽 패딩)에 따라 문장이 조금 달라진다. 재현이 필요하면 같은 질문 묶음으로 호출할 것.
        반환 (zero-shot 답 리스트, 더미 리스트). zero-shot 답이 빈 질문은 더미도 빈 문자열

        예) zero_shots, dummies = generate_dummy_docs(model, tokenizer, ['George V Coast is a part of the continent of'], 4096)
            zero_shots[0] = 'Australia.', dummies[0] = 'The George V Coast is a part of the continent of Australia. ...'
    '''
    def generate(prompts, n_tokens):
        return model_utils.get_generated_texts(model, tokenizer, model.device, prompts, max_seq_length, n_tokens) if prompts else []

    zero_shots = generate([get_generate_prompt(q) for q in queries], zero_shot_max_new_tokens)
    answered = [i for i, z in enumerate(zero_shots) if z]
    dummies = [''] * len(queries)
    for i, text in zip(answered, generate([get_generate_prompt_dummy_doc(queries[i], zero_shots[i]) for i in answered],
                                          max_new_tokens)):
        dummies[i] = text
    return zero_shots, dummies
