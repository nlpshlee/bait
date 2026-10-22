"""확증 편향 측정 전용 러너.

입력: data/create_contexts/{model}/
저장: outputs/check_confirmation_bias/{model}/{run_name}/measurements*.jsonl
분석: python analyze_confirmation_bias.py --model MODEL --run-name RUN
기존 --analyze-only 호출도 위 분석 프로그램으로 연결한다.
"""
from _init import *

import argparse
import glob
import json
import os
import sys
import time
from contextlib import ExitStack, contextmanager

from bait import globals as bait_globals
from bait.core.bait_prompts import CONTEXT_SIZE, FILE_FORMATS

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'check_confirmation_bias')
MODEL_CONFIG = bait_globals.data['MODEL_CONFIG']
DEVICE = 'cuda:0'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='문서 단위 IE 측정. 저장 결과 분석은 analyze_confirmation_bias.py에서 실행')
    p.add_argument('--model', default='Llama-3.2-3B', help='create_contexts 폴더의 모델 이름')
    p.add_argument('--n', type=int, default=100, help='분할마다 앞에서부터 사용할 문항 수')
    p.add_argument('--splits', nargs='+', default=['fact', 'counter'], choices=['fact', 'counter', 'other'],
                   help='측정 대상 분할. 기본 fact counter; other는 명시적으로 선택한 경우만 포함')
    p.add_argument('--formats', nargs='+', default=FILE_FORMATS, choices=FILE_FORMATS,
                   help='문서 형식. 기본은 5개 전부 (더미는 형식과 무관하게 문항마다 한 번만 생성해 모든 형식에 쓴다)')
    p.add_argument('--ratios', nargs='+', default=None, help="'사실:반사실' 목록. 기본 0:9 ~ 9:0")
    p.add_argument('--positions', nargs='*', default=['first', 'middle', 'last'],
                   choices=['first', 'middle', 'last'],
                   help='조건 B / C 의 추가 위치. 값 없이 --positions 만 주면 조건 A 만 측정')
    p.add_argument('--no-control', action='store_true',
                   help='대조군 C(외부 같은 편 문서) 를 측정하지 않는다. C 는 해석용이며 B / C 측정량의 절반이다')
    p.add_argument('--gpu', default='0', help='사용할 GPU 번호 하나. 프로세스 안에서는 이 GPU 가 cuda:0 이 된다')
    p.add_argument('--shard', type=int, default=0, help='GPU 여러 장일 때 이 프로세스가 맡을 몫 (0 ~ num-shards-1)')
    p.add_argument('--num-shards', type=int, default=1, help='문항을 나눌 프로세스(GPU) 수')
    p.add_argument('--dtype', default=MODEL_CONFIG['DTYPE'], help='3B 는 float32 로 IE 정밀도를 높일 수 있다')
    p.add_argument('--score-reduction', choices=['mean', 'sum'], default='mean',
                   help='mean: 기존 토큰 평균 로그확률 / sum: 정답 문자열 로그확률 합(EOS 제외). 바꾸면 새 run-name 사용')
    p.add_argument('--max-seq-length', type=int, default=MODEL_CONFIG['MAX_SEQ_LENGTH'])
    p.add_argument('--seed', type=int, default=bait_globals.GlobalCommonConfig.SEED, help='문서 집합 샘플링 시드')
    p.add_argument('--dummy-max-new-tokens', type=int, default=192, help='더미 문서 생성 최대 토큰 수 (프롬프트가 50~80단어를 요구)')
    p.add_argument('--zero-shot-max-new-tokens', type=int, default=64,
                   help='더미를 만들기 전 zero-shot 답 생성 최대 토큰 수 (check_zero_shot.ipynb 와 같은 값)')
    p.add_argument('--dummy-batch-size', type=int, default=64,
                   help='더미 생성 배치 크기. 분할 문항을 처음부터 이 크기로 잘라 생성하므로 바꾸면 더미 문장도 바뀐다')
    p.add_argument('--run-name', default='main', help='설정을 바꾸면 이름도 바꿀 것 (같은 폴더에 다른 설정이 섞이지 않게)')
    p.add_argument('--output-dir', default=None, help='결과 폴더 직접 지정. 생략하면 outputs/check_confirmation_bias/{model}/{run-name}')
    args = p.parse_args(argv)

    try:
        args.ratios = ([tuple(map(int, r.split(':'))) for r in args.ratios] if args.ratios
                       else [(n, CONTEXT_SIZE - 1 - n) for n in range(CONTEXT_SIZE)])
    except ValueError:
        p.error('--ratios 는 1:8 같은 정수 쌍이어야 한다')
    if any(len(r) != 2 or min(r) < 0 or sum(r) == 0 or max(r) >= CONTEXT_SIZE for r in args.ratios):
        p.error(f'--ratios 는 각 편이 0~{CONTEXT_SIZE - 1}이고 합이 양수인 사실:반사실 쌍이어야 한다')
    for name in ('n', 'num_shards', 'max_seq_length', 'dummy_max_new_tokens',
                 'zero_shot_max_new_tokens', 'dummy_batch_size'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} 는 양수여야 한다')
    for name in ('splits', 'formats', 'ratios', 'positions'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f'--{name} 에 중복 값이 있다')
    if not 0 <= args.shard < args.num_shards:
        p.error(f'--shard 는 0 ~ {args.num_shards - 1} 이어야 한다')
    return args


@contextmanager
def run_session(args, out_dir):
    """서로 다른 샤드만 같은 설정에 이어 쓰며, 최종 분석 중에는 측정하지 않는다."""
    from bait.core.confirmation_bias_measure import ORDER_POLICY, result_lock

    with ExitStack() as stack:
        stack.enter_context(result_lock(out_dir, directory=True, shared=True,
                                        legacy=os.path.join(out_dir, '.activity.lock')))
        logs_dir = os.path.join(out_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        with result_lock(logs_dir, directory=True, blocking=True,
                         legacy=os.path.join(out_dir, '.config.lock')):
            path = os.path.join(out_dir, 'config.json')
            config = None
            if os.path.exists(path):
                with open(path, encoding='utf-8') as config_file:
                    config = json.load(config_file)
                if not isinstance(config, dict):
                    raise SystemExit('config.json 이 올바른 설정 객체가 아니다. 해당 실행 폴더를 확인할 것')
            # splits는 이후 별도 분할을 추가할 수 있도록 비교에서 제외한다.
            excluded = {'gpu', 'shard', 'splits', 'run_name', 'output_dir'}
            expected = {'measurement_version': 3, 'document_order': ORDER_POLICY,
                        **{k: v for k, v in vars(args).items() if k not in excluded}}
            expected = json.loads(json.dumps(expected))
            if config is None and glob.glob(os.path.join(out_dir, 'measurements*.jsonl')):
                raise SystemExit('설정 기록이 없는 기존 결과에는 이어 쓸 수 없다. 새 --run-name 을 사용할 것')
            if config is not None and config != expected:
                changed = [k for k in expected.keys() | config.keys() if expected.get(k) != config.get(k)]
                raise SystemExit(f'측정 설정이 다르다: {changed}. 새 --run-name 을 사용할 것')
            if config is None:
                with open(path, 'w', encoding='utf-8') as config_file:
                    json.dump(expected, config_file, ensure_ascii=False, indent=2)
        name = 'measurements.jsonl' if args.num_shards == 1 else f'measurements.shard{args.shard}.jsonl'
        stack.enter_context(result_lock(os.path.join(out_dir, name),
                                        legacy=os.path.join(out_dir, f'.shard{args.shard}.lock')))
        yield


def measure(args, out_dir: str):
    '''이 프로세스 몫의 문항 중 아직 저장되지 않은 것을 측정한다 (GPU 별로 결과 파일을 따로 쓴다)'''
    from bait.core import bait_utils
    from bait.core.confirmation_bias_measure import (
        ORDER_POLICY, Scorer, build_position_plan, load_results,
        measure_question, prepare_dummies, save_question,
    )
    from bait.utils import common_utils, json_utils, model_utils, tokenizer_utils

    common_utils.set_seed(args.seed)
    name = 'measurements.jsonl' if args.num_shards == 1 else f'measurements.shard{args.shard}.jsonl'
    result_path = os.path.join(out_dir, name)
    done = {(m['split'], m['qid']) for m in load_results(out_dir, meta_only=True)[0]}
    model = None

    for split in args.splits:
        datas = json_utils.load_json(os.path.join(DATA_DIR, 'create_contexts', args.model,
                                                  f'bait_{args.model}_zero_shot_{split}_created_contexts.json'))[:args.n]
        # 전체 선택 문항으로 먼저 배정한다. 샤드/미완료 문항만 쓰면 재개할 때 순서가 바뀐다.
        position_plan = build_position_plan([d['id'] for d in datas], split, args.formats, args.ratios, args.seed)
        print(f'# [{split}] 위치 배정: {ORDER_POLICY}, 전체 {len(datas)}문항 기준, '
              '형식·비율별 위치 등장 횟수 차이 최대 1 (실행 중 누락 전)')
        mine = datas[args.shard::args.num_shards]                               # 이 프로세스 몫 (번갈아 나눔)
        todo = [d for d in mine if (split, d['id']) not in done]
        print(f'\n# [{split}] 이 프로세스 몫 {len(mine)} 문항 중 {len(mine) - len(todo)} 개는 이미 측정됨 -> {len(todo)} 개 측정')
        if not todo:
            continue
        if model is None:  # 전부 끝난 샤드를 다시 실행했을 때 모델을 올리지 않는다.
            model_path = bait_utils.get_model_name_or_path(args.model)
            model = model_utils.get_model(model_path, args.dtype, device=DEVICE, attn_imp='sdpa', is_eval=True)
            tokenizer = tokenizer_utils.load_tokenizer(model_path, 'left')
            scorer = Scorer(model, tokenizer, args.max_seq_length, args.score_reduction)

        started = time.time()
        dummies = prepare_dummies(model, tokenizer, datas, {d['id'] for d in todo}, args) if args.positions and todo else {}
        if dummies:
            print(f'# [{split}] 더미 {len(dummies)} 개 생성 : {time.time() - started:.0f}초')

        started = time.time()
        for i, data in enumerate(todo, start=1):
            meta, records = measure_question(scorer, data, split, args, dummies.get(data['id']),
                                             position_plan=position_plan)
            save_question(result_path, meta, records)
            common_utils.clear_gpu_memory()
            left = (time.time() - started) / i * (len(todo) - i) / 60
            print(f"# [{split}] {i}/{len(todo)} qid={data['id']} records={len(records)} skipped={meta['skipped']} "
                  f"prior={meta['prior']:+.2f} zero_shot_answer={meta.get('zero_shot_answer')} "
                  f"dummy_answer={meta.get('dummy_answer')} | 남은 시간 약 {left:.1f}분")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--analyze-only' in argv:  # 기존 실행 스크립트와 호환
        from analyze_confirmation_bias import main as analyze_main
        return analyze_main([arg for arg in argv if arg != '--analyze-only'])
    args = parse_args(argv)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU 선택 전에 torch가 로드됐다. import 순서를 확인할 것')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    out_dir = args.output_dir or os.path.join(OUTPUT_DIR, args.model, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'# 결과 폴더 : {out_dir}')
    print(f'# 측정 설정 : {vars(args)}')
    with run_session(args, out_dir):
        measure(args, out_dir)
    print('# 측정 완료. 모든 샤드가 끝나면 analyze_confirmation_bias.py로 분석할 것')


if __name__ == '__main__':
    main()
