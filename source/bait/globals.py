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

