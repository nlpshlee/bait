from pathlib import Path
import yaml


globals_file_path = Path(__file__).resolve()
globals_dir = globals_file_path.parent

with open(f'{globals_dir}/globals.yml', 'r') as in_file:
    data = yaml.safe_load(in_file)


class DEBUG:
    LOG = data['DEBUG']['LOG']
    ERROR = data['DEBUG']['ERROR']

