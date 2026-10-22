from pathlib import Path
import yaml


globals_file_path = Path(__file__).resolve()
globals_dir = globals_file_path.parent

with open(f'{globals_dir}/globals.yml', 'r') as in_file:
    data = yaml.safe_load(in_file)


class GlobalDebug:
    LOG = data['DEBUG']['LOG']
    ERROR = data['DEBUG']['ERROR']
    TRAIN = data['DEBUG']['TRAIN']
    EVAL = data['DEBUG']['EVAL']


class GlobalCommonConfig:
    SEED = data['COMMON_CONFIG']['SEED']


class GlobalModelConfig:
    DTYPE = data['MODEL_CONFIG']['DTYPE'],
    MAX_SEQ_LENGTH = data['MODEL_CONFIG']['MAX_SEQ_LENGTH'],
    MAX_NEW_TOKENS = data['MODEL_CONFIG']['MAX_NEW_TOKENS'],
    LORA_R = data['MODEL_CONFIG']['LORA_R'],
    LORA_TARGET_MODULES = data['MODEL_CONFIG']['LORA_TARGET_MODULES'],
    LORA_ALPHA = data['MODEL_CONFIG']['LORA_ALPHA'],
    GRADIENT_ACCUMULATION_STEPS = data['MODEL_CONFIG']['GRADIENT_ACCUMULATION_STEPS'],
    LEARNING_RATE = data['MODEL_CONFIG']['LEARNING_RATE'],
    EPSILON = data['MODEL_CONFIG']['EPSILON'],
    KL_PENALTY = data['MODEL_CONFIG']['KL_PENALTY'],
    USE_GRADIENT_CHECKPOINTING = data['MODEL_CONFIG']['USE_GRADIENT_CHECKPOINTING'],
    RESUME_RUN_TIME = data['MODEL_CONFIG']['RESUME_RUN_TIME']

