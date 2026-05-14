from _init import *

import json

from msnap.utils.common_const import *
from msnap.utils import common_utils, file_utils


###########################################################################################

def str_to_dict(json_str: str):
    try:
        # 문자열을 읽을 때는, loads() 호출
        return json.loads(json_str)

    except Exception as e:
        common_utils.logging_error("json_util.str_to_dict()", e)
        return None


def to_str(input, indent=4):
    try:
        return json.dumps(input, ensure_ascii=False, indent=indent)

    except Exception as e:
        common_utils.logging_error("json_util.to_str()", e)
        return ""


def load_jsonl(in_file_path: str, encoding=ENCODING):
    try:
        if file_utils.exists(in_file_path):
            file = file_utils.open_file(in_file_path, encoding, 'r')
            datas = []

            for line in file:
                if not line.strip():
                    continue

                data = json.loads(line)
                datas.append(data)

            print(f'json_util.load_jsonl() {in_file_path} -> data_size : {len(datas)}')
            file.close()

            return datas

    except Exception as e:
        common_utils.logging_error("json_util.load_jsonl()", e)
        return None

    return None


def write_jsonl(datas, out_file_path: str, encoding=ENCODING):
    try:
        file_utils.make_parent(out_file_path)
        file = file_utils.open_file(out_file_path, encoding, 'w')

        for data in datas:
            json_string = json.dumps(data, ensure_ascii=False)
            file.write(json_string + '\n')
        file.close()

        print(f'json_util.write_jsonl() data_size : {len(datas)} -> {out_file_path}')
        return True

    except Exception as e:
        common_utils.logging_error("json_util.write_jsonl()", e)
        return False


def load_json(in_file_path: str, encoding=ENCODING):
    try:
        if file_utils.exists(in_file_path):
            file = file_utils.open_file(in_file_path, encoding, 'r')

            # 파일을 읽을 때는, load() 호출
            datas = json.load(file)

            print(f'json_util.load_json() {in_file_path} -> data_size : {len(datas)}')
            file.close()

            return datas

    except Exception as e:
        common_utils.logging_error("json_util.load_json()", e)
        return None

    return None


def write_json(input, out_file_path: str, encoding=ENCODING, indent=4):
    try:
        file_utils.make_parent(out_file_path)

        file = file_utils.open_file(out_file_path, encoding, 'w')
        file.write(to_str(input, indent))
        file.close()

        print(f'json_util.write_json() data_size : {len(input)} -> {out_file_path}')
        return True

    except Exception as e:
        common_utils.logging_error("json_util.write_json()", e)
        return False

