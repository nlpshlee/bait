from _init import *

import random


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

