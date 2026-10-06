'''
    확증 편향 측정 러너 (check_confirmation_bias.ipynb 의 스크립트 버전)

    1) 측정 : 분할(fact / counter / other)마다 문항별로 measure_question() 실행
              -> {out_dir}/measurements.jsonl (문항 하나 = 한 줄. 끊기면 같은 명령으로 이어서 실행)
    2) 분석 : β (사실을 믿음 vs 반사실을 믿음) + 더미 비교 (질문 단위 부트스트랩 95% 신뢰구간) -> 로그 출력, report.txt, *.csv
    3) 그림 : fig_confirmation_bias / fig_beta_by_format / fig_dummy_effect / fig_dummy_control (.pdf 논문용, .png 확인용)

    실행 예 (source/bait 에서)
        python check_confirmation_bias.py --model Llama-3.2-3B --n 100
        python check_confirmation_bias.py --model Llama-3.2-3B --analyze-only      # 저장된 결과로 분석 / 그림만

    GPU 여러 장 : 문항을 GPU 수만큼 나눠 프로세스를 하나씩 띄운다 (GPU 한 장은 문서 집합 하나로 이미 포화된다)
        python check_confirmation_bias.py --gpu 0 --shard 0 --num-shards 2 ...
        python check_confirmation_bias.py --gpu 1 --shard 1 --num-shards 2 ...
        python check_confirmation_bias.py --analyze-only ...                        # 모두 끝난 뒤 합쳐서 분석
    결과 : data/check_confirmation_bias/{model}/{run_name}/
'''
from _init import *

import argparse, os, sys, time

from bait import globals as bait_globals
from bait.core.bait_prompts import CONTEXT_SIZE, FILE_FORMATS


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
MODEL_CONFIG = bait_globals.data['MODEL_CONFIG']


def parse_args():
    p = argparse.ArgumentParser(description='확증 편향 측정 (문서 단위 knockout + 3분할 β 분해)')
    p.add_argument('--model', default='Llama-3.2-3B', help='create_contexts 폴더의 모델 이름')
    p.add_argument('--n', type=int, default=100, help='분할마다 앞에서부터 사용할 문항 수')
    p.add_argument('--splits', nargs='+', default=['fact', 'counter', 'other'], choices=['fact', 'counter', 'other'])
    p.add_argument('--formats', nargs='+', default=FILE_FORMATS, choices=FILE_FORMATS,
                   help='문서 형식. 기본은 5개 전부 (더미는 형식과 무관하게 문항마다 한 번만 생성해 모든 형식에 쓴다)')
    p.add_argument('--ratios', nargs='+', default=None, help="'사실:반사실' 목록. 기본 0:9 ~ 9:0")
    p.add_argument('--positions', nargs='*', default=['first', 'middle', 'last'],
                   help='조건 B / C 의 추가 위치. 값 없이 --positions 만 주면 조건 A 만 측정')
    p.add_argument('--no-control', action='store_true',
                   help='대조군 C(외부 같은 편 문서) 를 측정하지 않는다. C 는 해석용이며 B / C 측정량의 절반이다')
    p.add_argument('--gpu', default='0', help='사용할 GPU 번호 하나. 프로세스 안에서는 이 GPU 가 cuda:0 이 된다')
    p.add_argument('--shard', type=int, default=0, help='GPU 여러 장일 때 이 프로세스가 맡을 몫 (0 ~ num-shards-1)')
    p.add_argument('--num-shards', type=int, default=1, help='문항을 나눌 프로세스(GPU) 수')
    p.add_argument('--dtype', default=MODEL_CONFIG['DTYPE'], help='3B 는 float32 로 IE 정밀도를 높일 수 있다')
    p.add_argument('--max-seq-length', type=int, default=MODEL_CONFIG['MAX_SEQ_LENGTH'])
    p.add_argument('--seed', type=int, default=bait_globals.GlobalCommonConfig.SEED, help='문서 집합 샘플링 / 부트스트랩 시드')
    p.add_argument('--dummy-max-new-tokens', type=int, default=192, help='더미 문서 생성 최대 토큰 수 (프롬프트가 50~80단어를 요구)')
    p.add_argument('--zero-shot-max-new-tokens', type=int, default=64,
                   help='더미를 만들기 전 zero-shot 답 생성 최대 토큰 수 (check_zero_shot.ipynb 와 같은 값)')
    p.add_argument('--dummy-batch-size', type=int, default=64,
                   help='더미 생성 배치 크기. 분할 문항을 처음부터 이 크기로 잘라 생성하므로 바꾸면 더미 문장도 바뀐다')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--run-name', default='main', help='설정을 바꾸면 이름도 바꿀 것 (같은 폴더에 다른 설정이 섞이지 않게)')
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()

    args.ratios = ([tuple(map(int, r.split(':'))) for r in args.ratios] if args.ratios
                   else [(n, CONTEXT_SIZE - 1 - n) for n in range(CONTEXT_SIZE)])
    if not 0 <= args.shard < args.num_shards:
        p.error(f'--shard 는 0 ~ {args.num_shards - 1} 이어야 한다')
    return args


ARGS = parse_args()

# CUDA_VISIBLE_DEVICES 는 torch 가 불러와지기 전에 지정해야 효과가 있다. 그러면 --gpu 로 고른 GPU 가
# 프로세스 안에서는 항상 cuda:0 이 되므로 아래에서 device='cuda:0' 으로 쓴다.
# 위쪽 import 가 바뀌어 torch 가 먼저 불러와지면 --gpu 가 조용히 무시되므로 여기서 막는다.
if 'torch' in sys.modules:
    raise RuntimeError('torch 가 CUDA_VISIBLE_DEVICES 지정 전에 불러와졌다. --gpu 가 적용되지 않으므로 import 순서를 확인할 것')
os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
DEVICE = 'cuda:0'

from bait.core import bait_utils
from bait.core.confirmation_bias_analysis import beta_table, diagnostics, dummy_effect_table, dummy_table, report_text
from bait.core.confirmation_bias_measure import INSERT_AT, Scorer, load_results, measure_question, prepare_dummies, save_question
from bait.core.confirmation_bias_plot import plot_all
from bait.utils import common_utils, json_utils, model_utils, tokenizer_utils

if not set(ARGS.positions) <= set(INSERT_AT):
    raise SystemExit(f'--positions 는 {list(INSERT_AT)} 중에서 고를 것 : {ARGS.positions}')


def load_dataset_counter_ids(model: str) -> set:
    '''
        [평가용] zero-shot 답이 원 데이터의 반사실 답이었던 문항 id (check_zero_shot.ipynb 의 counter 분류 결과).
        counter 분할의 나머지와 other 분할은 반사실 답을 모델 자신의 zero-shot 답으로 바꿔 채운 문항이다 (history.md 2.1)
    '''
    path = os.path.join(DATA_DIR, 'check_zero_shot', model, f'bait_{model}_checked_zero_shot_counter.json')
    if not os.path.exists(path):
        print(f'# 경고 : {path} 가 없어 counter_dataset 그룹(beta_dataset)을 계산하지 않는다')
        return set()
    return {d['case_id'] for d in json_utils.load_json(path)}


def measure(args, out_dir: str):
    '''이 프로세스 몫의 문항 중 아직 저장되지 않은 것을 측정한다 (GPU 별로 결과 파일을 따로 쓴다)'''
    common_utils.set_seed(args.seed)
    model_path = bait_utils.get_model_name_or_path(args.model)
    model = model_utils.get_model(model_path, args.dtype, device=DEVICE, attn_imp='sdpa', is_eval=True)
    tokenizer = tokenizer_utils.load_tokenizer(model_path, 'left')
    scorer = Scorer(model, tokenizer, args.max_seq_length)

    name = 'measurements.jsonl' if args.num_shards == 1 else f'measurements.shard{args.shard}.jsonl'
    result_path = os.path.join(out_dir, name)
    done = {(m['split'], m['qid']) for m in load_results(out_dir)[0]}

    for split in args.splits:
        datas = json_utils.load_json(os.path.join(DATA_DIR, 'create_contexts', args.model,
                                                  f'bait_{args.model}_zero_shot_{split}_created_contexts.json'))[:args.n]
        mine = datas[args.shard::args.num_shards]                               # 이 프로세스 몫 (번갈아 나눔)
        todo = [d for d in mine if (split, d['id']) not in done]
        print(f'\n# [{split}] 이 프로세스 몫 {len(mine)} 문항 중 {len(mine) - len(todo)} 개는 이미 측정됨 -> {len(todo)} 개 측정')

        started = time.time()
        dummies = prepare_dummies(model, tokenizer, datas, {d['id'] for d in todo}, args) if args.positions and todo else {}
        if dummies:
            print(f'# [{split}] 더미 {len(dummies)} 개 생성 : {time.time() - started:.0f}초')

        started = time.time()
        for i, data in enumerate(todo, start=1):
            meta, records = measure_question(scorer, data, split, args, dummies.get(data['id']))
            save_question(result_path, meta, records)
            common_utils.clear_gpu_memory()
            left = (time.time() - started) / i * (len(todo) - i) / 60
            print(f"# [{split}] {i}/{len(todo)} qid={data['id']} records={len(records)} skipped={meta['skipped']} "
                  f"prior={meta['prior']:+.2f} zero_shot_answer={meta.get('zero_shot_answer')} "
                  f"dummy_answer={meta.get('dummy_answer')} | 남은 시간 약 {left:.1f}분")


def analyze(args, out_dir: str):
    metas, records = load_results(out_dir)
    if not records:
        raise SystemExit(f'측정 결과가 없다 : {out_dir}')

    ids = load_dataset_counter_ids(args.model)                             # counter_dataset 그룹 (강건성 확인용)
    beta = beta_table(records, args.n_boot, args.seed, dataset_counter_ids=ids)                 # 조건 A : 확증 편향 (비율별)
    beta_format = (beta_table(records, args.n_boot, args.seed, by='format', dataset_counter_ids=ids)
                   if len({r['format'] for r in records}) > 1 else None)                       # 형식이 여러 개면 형식별로도
    effect = dummy_effect_table(records, args.n_boot, args.seed, dataset_counter_ids=ids)      # A vs B : 더미 효과
    dummy, dummy_splits = dummy_table(records, args.n_boot, args.seed)                          # B vs C : 대조군
    diag = diagnostics(metas, records, ids)

    text = report_text(beta, beta_format, effect, dummy, dummy_splits, diag)
    print(text)
    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    for name, table in (('beta', beta), ('beta_format', beta_format), ('dummy_effect', effect), ('dummy_control', dummy),
                        ('diagnostics', diag)):
        if table is not None:
            table.to_csv(os.path.join(out_dir, f'{name}.csv'))

    for path in plot_all(beta, beta_format, effect, dummy, dummy_splits, out_dir):
        print(f'# 그림 : {path}')


def main():
    out_dir = os.path.join(DATA_DIR, 'check_confirmation_bias', ARGS.model, ARGS.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'# 결과 폴더 : {out_dir}\n# 설정 : {vars(ARGS)}')

    if not ARGS.analyze_only:
        measure(ARGS, out_dir)
        if ARGS.num_shards > 1:                                                 # 다른 GPU 가 아직 돌고 있을 수 있다
            print('# 샤드 측정 완료. 모든 샤드가 끝나면 --analyze-only 로 합쳐서 분석할 것')
            return
    analyze(ARGS, out_dir)


if __name__ == '__main__':
    main()
