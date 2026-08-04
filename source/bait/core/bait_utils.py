from _init import *


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

