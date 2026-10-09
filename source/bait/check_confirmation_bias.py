'''
    확증 편향 측정 러너 (check_confirmation_bias.ipynb 의 스크립트 버전)

    1) 측정 : 선택한 분할(기본 fact / counter)마다 문항별로 measure_question() 실행
              -> {out_dir}/measurements.jsonl (문항 하나 = 한 줄. 끊기면 같은 명령으로 이어서 실행)
    2) 분석 : 분할·형식·비율별 A/B 영향과 질문 단위 부트스트랩 95% 신뢰구간
    3) 결과 : influence_table.html (비교표 + 형식별 그래프 + 필요한 CSV 내려받기)
              fig_influence_{position}.png (전체 형식 평균 그래프)
              측정 원본·설정·로그는 재분석을 위해 유지. PDF나 별도 요약/중간 CSV는 생성하지 않는다.

    실행 예 (source/bait 에서)
        python check_confirmation_bias.py --model Llama-3.2-3B --n 100
        python check_confirmation_bias.py --model Llama-3.2-3B --analyze-only      # 저장된 결과로 분석 / 그림만

    GPU 여러 장 : 문항을 GPU 수만큼 나눠 프로세스를 하나씩 띄운다 (GPU 한 장은 문서 집합 하나로 이미 포화된다)
        python check_confirmation_bias.py --gpu 0 --shard 0 --num-shards 2 ...
        python check_confirmation_bias.py --gpu 1 --shard 1 --num-shards 2 ...
        python check_confirmation_bias.py --analyze-only ...                        # 모두 끝난 뒤 합쳐서 분석
    입력 데이터 : data/create_contexts/{model}/
    결과 : outputs/check_confirmation_bias/{model}/{run_name}/
'''
from _init import *

import argparse, glob, json, os, sys, time
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from bait import globals as bait_globals
from bait.core.bait_prompts import CONTEXT_SIZE, FILE_FORMATS


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'check_confirmation_bias')
MODEL_CONFIG = bait_globals.data['MODEL_CONFIG']


def parse_args():
    p = argparse.ArgumentParser(description='문서 단위 IE 측정 및 더미 추가 전후 영향 비교')
    p.add_argument('--model', default='Llama-3.2-3B', help='create_contexts 폴더의 모델 이름')
    p.add_argument('--n', type=int, default=100, help='분할마다 앞에서부터 사용할 문항 수')
    p.add_argument('--splits', nargs='+', default=['fact', 'counter'], choices=['fact', 'counter', 'other'],
                   help='측정 및 분석 대상 분할. 기본 fact counter; other는 명시적으로 선택한 경우만 포함')
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
    p.add_argument('--score-reduction', choices=['mean', 'sum'], default='mean',
                   help='mean: 기존 토큰 평균 로그확률 / sum: 정답 문자열 로그확률 합(EOS 제외). 바꾸면 새 run-name 사용')
    p.add_argument('--max-seq-length', type=int, default=MODEL_CONFIG['MAX_SEQ_LENGTH'])
    p.add_argument('--seed', type=int, default=bait_globals.GlobalCommonConfig.SEED, help='문서 집합 샘플링 / 부트스트랩 시드')
    p.add_argument('--dummy-max-new-tokens', type=int, default=192, help='더미 문서 생성 최대 토큰 수 (프롬프트가 50~80단어를 요구)')
    p.add_argument('--zero-shot-max-new-tokens', type=int, default=64,
                   help='더미를 만들기 전 zero-shot 답 생성 최대 토큰 수 (check_zero_shot.ipynb 와 같은 값)')
    p.add_argument('--dummy-batch-size', type=int, default=64,
                   help='더미 생성 배치 크기. 분할 문항을 처음부터 이 크기로 잘라 생성하므로 바꾸면 더미 문장도 바뀐다')
    p.add_argument('--n-boot', type=int, default=5000, help='분석 시 질문 단위 부트스트랩 반복 수 (측정 결과는 바뀌지 않음)')
    p.add_argument('--run-name', default='main', help='설정을 바꾸면 이름도 바꿀 것 (같은 폴더에 다른 설정이 섞이지 않게)')
    p.add_argument('--output-dir', default=None, help='결과 폴더 직접 지정. 생략하면 outputs/check_confirmation_bias/{model}/{run-name}')
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--preview', action='store_true',
                   help='--analyze-only와 함께 사용: 진행 중인 결과의 완성 문항만 읽어 preview/에 중간 분석 저장')
    args = p.parse_args()
    if args.preview and not args.analyze_only:
        p.error('--preview 는 --analyze-only 와 함께 사용해야 한다')

    try:
        args.ratios = ([tuple(map(int, r.split(':'))) for r in args.ratios] if args.ratios
                       else [(n, CONTEXT_SIZE - 1 - n) for n in range(CONTEXT_SIZE)])
    except ValueError:
        p.error('--ratios 는 1:8 같은 정수 쌍이어야 한다')
    if any(len(r) != 2 or min(r) < 0 or sum(r) == 0 or max(r) >= CONTEXT_SIZE for r in args.ratios):
        p.error(f'--ratios 는 각 편이 0~{CONTEXT_SIZE - 1}이고 합이 양수인 사실:반사실 쌍이어야 한다')
    for name in ('n', 'num_shards', 'max_seq_length', 'dummy_max_new_tokens',
                 'zero_shot_max_new_tokens', 'dummy_batch_size', 'n_boot'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} 는 양수여야 한다')
    for name in ('splits', 'formats', 'ratios', 'positions'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f'--{name} 에 중복 값이 있다')
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

import base64, hashlib, io, math, numbers, re, warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from bait.core import bait_utils
from bait.core.confirmation_bias_measure import (
    INSERT_AT, ORDER_POLICY, Scorer, build_position_plan, load_results,
    log_position_balance, measure_question, prepare_dummies, result_lock, save_question,
)
from bait.utils import common_utils, json_utils, model_utils, tokenizer_utils

if not set(ARGS.positions) <= set(INSERT_AT):
    raise SystemExit(f'--positions 는 {list(INSERT_AT)} 중에서 고를 것 : {ARGS.positions}')


@contextmanager
def run_session(args, out_dir):
    '''최종 분석은 측정과 배타적. 중간 분석은 읽기 범위를 고정해 측정과 동시에 실행한다.'''
    with ExitStack() as stack:
        stack.enter_context(result_lock(out_dir, directory=True, shared=not args.analyze_only or args.preview,
                                        legacy=os.path.join(out_dir, '.activity.lock')))
        if args.preview:
            preview_dir = os.path.join(out_dir, 'preview')
            os.makedirs(preview_dir, exist_ok=True)
            stack.enter_context(result_lock(preview_dir, directory=True))  # 중간 보고서끼리 동시 덮어쓰기 방지
        # config 파일을 처음 만드는 순간에도 네 프로세스가 경합할 수 있으므로 짧은 별도 잠금을 쓴다.
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
            if args.analyze_only:
                if config is None:
                    print('# 경고: 설정 기록이 없는 이전 결과다. 동일한 측정 설정인지 검증할 수 없다')
                else:
                    print(f'# 저장된 측정 설정: {config}')
            else:
                # splits 는 나중에 other 만 추가할 수 있도록 비교 대상에서 제외한다.
                excluded = {'gpu', 'shard', 'splits', 'n_boot', 'run_name', 'output_dir', 'analyze_only', 'preview'}
                expected = {'measurement_version': 3, 'document_order': ORDER_POLICY,
                            **{k: v for k, v in vars(args).items() if k not in excluded}}
                expected = json.loads(json.dumps(expected))  # ratios 의 tuple -> JSON list
                if config is None and glob.glob(os.path.join(out_dir, 'measurements*.jsonl')):
                    raise SystemExit('설정 기록이 없는 기존 결과에는 이어 쓸 수 없다. 새 --run-name 을 사용할 것')
                if config is not None and config != expected:
                    changed = [k for k in expected.keys() | config.keys() if expected.get(k) != config.get(k)]
                    raise SystemExit(f'측정 설정이 다르다: {changed}. 새 --run-name 을 사용할 것')
                if config is None:
                    with open(path, 'w', encoding='utf-8') as config_file:
                        json.dump(expected, config_file, ensure_ascii=False, indent=2)
        if not args.analyze_only:
            name = 'measurements.jsonl' if args.num_shards == 1 else f'measurements.shard{args.shard}.jsonl'
            stack.enter_context(result_lock(os.path.join(out_dir, name),
                                            legacy=os.path.join(out_dir, f'.shard{args.shard}.lock')))
        yield


def measure(args, out_dir: str):
    '''이 프로세스 몫의 문항 중 아직 저장되지 않은 것을 측정한다 (GPU 별로 결과 파일을 따로 쓴다)'''
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


# Paired A/B analysis: signed IE, document sums/means, and question bootstrap.
# Sums are sums of individual knockout effects, not an additive decomposition.
# The table and graphs below use exactly the same paired estimates.

def _cluster_estimates(frame, metrics, n_boot, seed, split):
    """Shared question draws for every condition/format/ratio within a split.

    Count weights avoid materializing bootstrap x question x all-metrics arrays.
    Feature blocks bound memory. The seed convention matches the existing
    analysis bootstrap.
    """
    wide = frame.pivot(index='qid', columns=CELL, values=metrics).sort_index()
    x = wide.to_numpy(dtype=float)
    n = len(x)
    key = json.dumps([seed, split, wide.index.tolist()], default=str).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little'))
    weights = np.zeros((n_boot, n), dtype=float)
    for start in range(0, n_boot, 128):
        count = min(128, n_boot - start)
        sampled = rng.integers(0, n, size=(count, n))
        np.add.at(weights[start:start + count], (np.arange(count)[:, None], sampled), 1.)
    estimates, intervals = {}, {}
    for start in range(0, x.shape[1], 128):
        block = x[:, start:start + 128]
        finite = np.isfinite(block)
        counts = finite.sum(axis=0)
        clean = np.where(finite, block, 0.)
        estimate = np.divide(clean.sum(axis=0), counts, out=np.full(len(counts), np.nan), where=counts > 0)
        boot = weights @ clean
        missing = counts != n
        boot[:, ~missing] /= n
        if missing.any():
            denominators = weights @ finite[:, missing].astype(float)
            boot[:, missing] = np.divide(boot[:, missing], denominators,
                                         out=np.full_like(denominators, np.nan), where=denominators > 0)
        boot[:, counts < 2] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bounds = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        for j, column in enumerate(wide.columns[start:start + 128]):
            metric, *cell = column
            cell = tuple(cell)
            estimates.setdefault(cell, {})[metric] = estimate[j]
            intervals.setdefault(cell, {}).update({f'{metric}_lo': bounds[0, j], f'{metric}_hi': bounds[1, j]})
    return estimates, intervals


KEYS = ['split', 'qid', 'format', 'n_fact', 'n_counter']
CELL = ['format', 'n_fact', 'n_counter', 'position']
METRICS = [f'{side}_mean_{condition}'
           for side in ('fact', 'counter') for condition in ('A', 'B', 'delta')]
METRICS += ['dummy_mean_B']


def _record_values(record):
    """Validate the saved measurements; never filter on their sign or size."""
    context = '/'.join(str(record.get(key, '?')) for key in KEYS)
    if record['split'] not in ('fact', 'counter'):
        raise ValueError(f'Unsupported belief split in IE analysis: {context}')
    if record['format'] == 'ALL':
        raise ValueError('ALL is reserved for the within-question format average')
    for key in ('n_fact', 'n_counter'):
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f'Invalid document count {key}: {context}')
    n_original = record['n_fact'] + record['n_counter']
    if n_original == 0:
        raise ValueError(f'An IE condition must contain an original document: {context}')
    ie = np.asarray(record['ie'], dtype=float)
    sides = np.asarray(record['sides'])
    if ie.ndim != 1 or ie.shape != sides.shape or not np.isfinite(ie).all():
        raise ValueError(f'Invalid or nonfinite saved IE: {context}')
    if 'S' in record and not np.isfinite(record['S']):
        raise ValueError(f'Nonfinite saved score: {context}')
    if not set(sides) <= {'fact', 'counter', 'added'}:
        raise ValueError(f'Unknown document label: {context}')
    is_added = record['condition'] == 'B'
    expected = {'fact': record['n_fact'], 'counter': record['n_counter'],
                'added': int(is_added)}
    if any(int(np.sum(sides == side)) != count for side, count in expected.items()):
        raise ValueError(f'Document labels/counts disagree with the saved ratio: {context}')
    position = record['position']
    if is_added:
        positions = {'first': 0, 'middle': n_original // 2, 'last': n_original}
        if position not in positions or sides[positions[position]] != 'added':
            raise ValueError(f'Added document does not match its saved position: {context}')
    elif position != 'none':
        raise ValueError(f'Baseline A must have position=none: {context}')
    values = {side: float(ie[sides == side].mean()) if expected[side] else np.nan
              for side in ('fact', 'counter')}
    values['dummy'] = float(ie[sides == 'added'][0]) if is_added else np.nan
    return values, tuple(sides[sides != 'added'])


def _paired_questions(records, expected_formats):
    baseline, added, seen = {}, [], set()
    available_formats = set()
    for record in records:
        if record.get('condition') not in ('A', 'B'):
            continue
        key = tuple(record[name] for name in KEYS)
        unique = (*key, record['condition'], record['position'])
        if unique in seen:
            raise ValueError(f'Duplicate A/B measurement: {unique}')
        seen.add(unique)
        values, order = _record_values(record)
        available_formats.add(record['format'])
        if record['condition'] == 'A':
            baseline[key] = (values, order)
        else:
            added.append((key, record['position'], values, order))

    formats = list(expected_formats) if expected_formats is not None else sorted(available_formats)
    if len(formats) != len(set(formats)) or 'ALL' in formats:
        raise ValueError('expected_formats must contain distinct original format names')
    if not available_formats <= set(formats):
        raise ValueError('Saved measurements contain formats outside expected_formats')

    rows, matched_baselines = [], set()
    unpaired_added = 0
    for key, position, after, order in added:
        if key not in baseline:
            unpaired_added += 1
            continue
        before, baseline_order = baseline[key]
        if order != baseline_order:
            raise ValueError(f'Original document label order differs between A and B: {key}')
        matched_baselines.add(key)
        row = dict(zip(KEYS, key))
        row.update(position=position, n_format_pairs=1, dummy_mean_B=after['dummy'])
        for side in ('fact', 'counter'):
            row.update({f'{side}_mean_A': before[side], f'{side}_mean_B': after[side],
                        f'{side}_mean_delta': after[side] - before[side]})
        rows.append(row)

    diagnostics = dict(expected_formats=formats, n_unpaired_B=unpaired_added,
                       n_A_without_any_B=len(baseline) - len(matched_baselines),
                       n_incomplete_format_questions=0)
    if not rows:
        return pd.DataFrame(), diagnostics

    frame = pd.DataFrame(rows)
    question_keys = ['split', 'qid', 'n_fact', 'n_counter', 'position']
    grouped = frame.groupby(question_keys, sort=False, dropna=False)
    counts = grouped['format'].nunique()
    complete = counts[counts == len(formats)].index
    diagnostics['n_incomplete_format_questions'] = int((counts != len(formats)).sum())
    # ALL requires every expected format for that question, ratio and position.
    # Thus each included question and each format within it have equal weight.
    aggregate = grouped[METRICS].mean().loc[complete].reset_index()
    aggregate['format'] = 'ALL'
    aggregate['n_format_pairs'] = len(formats)
    return pd.concat([frame, aggregate], ignore_index=True), diagnostics


def ie_tables(records, n_boot=5000, seed=42, expected_formats=None):
    """Return one raw-IE table and paired question-bootstrap 95% intervals.

    ``*_sum_*`` is the mean, across questions, of each question's sum of
    document effects; it is not the sum across the entire dataset. ``*_mean_*``
    is the mean per document, again giving each question equal weight.
    ``*_delta`` always means raw signed IE after minus before. Only columns
    named ``*_change`` reverse the counter-document sign for plotting.

    Partial runs are supported: only completed A/B pairs enter each cell.
    In ALL, a question additionally needs every expected format. C records are
    not used. Saved data permit validation of label order, not document text
    identity, which is not stored in these records.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral) or n_boot < 1:
        raise ValueError('n_boot must be a positive integer')
    questions, diagnostics = _paired_questions(records, expected_formats)
    if questions.empty:
        result = pd.DataFrame()
        result.attrs.update(diagnostics)
        return result

    rows = []
    for split, frame in questions.groupby('split', sort=False):
        estimates, intervals = _cluster_estimates(frame, METRICS, n_boot, seed, split)
        for cell, group in frame.groupby(CELL, sort=False):
            row = dict(zip(CELL, cell))
            row.update(split=split, ratio=f'{cell[1]}:{cell[2]}', n_questions=len(group),
                       n_format_pairs=int(group['n_format_pairs'].sum()))
            row.update(estimates[cell])
            row.update(intervals[cell])
            # Counts are constant within a ratio, so sums and their intervals
            # follow exactly by scaling means. An absent group has sum zero
            # and undefined mean; these are not missing observations.
            for side in ('fact', 'counter'):
                count = row[f'n_{side}']
                for condition in ('A', 'B', 'delta'):
                    for suffix in ('', '_lo', '_hi'):
                        row[f'{side}_sum_{condition}{suffix}'] = (
                            count * row[f'{side}_mean_{condition}{suffix}'] if count else 0.)
                sign = 1. if side == 'fact' else -1.
                for statistic in ('mean', 'sum'):
                    raw = f'{side}_{statistic}_delta'
                    name = f'{side}_{statistic}_change'
                    row[name] = sign * row[raw]
                    row[f'{name}_lo'] = sign * row[f'{raw}_lo' if sign > 0 else f'{raw}_hi']
                    row[f'{name}_hi'] = sign * row[f'{raw}_hi' if sign > 0 else f'{raw}_lo']
            # There is exactly one dummy per B condition.
            for suffix in ('', '_lo', '_hi'):
                row[f'dummy_sum_B{suffix}'] = row[f'dummy_mean_B{suffix}']
            rows.append(row)

    table = pd.DataFrame(rows)
    format_order = {name: index for index, name in enumerate(['ALL', *diagnostics['expected_formats']])}
    table['_split_order'] = table['split'].map({'fact': 0, 'counter': 1})
    table['_format_order'] = table['format'].map(format_order)
    table['_fact_fraction'] = table['n_fact'] / (table['n_fact'] + table['n_counter'])
    table = table.sort_values(['_split_order', '_format_order', '_fact_fraction',
                              'n_fact', 'n_counter', 'position'], kind='stable')
    table = table.drop(columns=['_split_order', '_format_order', '_fact_fraction']).reset_index(drop=True)
    table.attrs.update(diagnostics)
    return table


def as_push_table(table):
    """Convert report values once; raw IE estimates and measurements stay intact.

    Existing documents use their label direction. The dummy uses the split's
    belief direction, not an inferred label for its freely generated content.
    Negating an interval swaps its lower/upper endpoints. Missing groups and
    the nonexistent dummy-before condition are never filled with zero.
    """
    identity = ['format', 'n_fact', 'n_counter', 'position', 'split', 'ratio',
                'n_questions', 'n_format_pairs']
    result = table[identity].copy()
    for side in ('fact', 'counter', 'dummy'):
        sign = (table['split'].map({'fact': 1., 'counter': -1.})
                if side == 'dummy' else pd.Series(1. if side == 'fact' else -1., index=table.index))
        for unit in ('mean', 'sum'):
            for condition in (('B',) if side == 'dummy' else ('A', 'B', 'delta')):
                key = f'{side}_{unit}_{condition}'
                result[key] = sign * table[key]
                for bound, opposite in (('_lo', '_hi'), ('_hi', '_lo')):
                    result[key + bound] = sign * table[key + bound].where(sign > 0, table[key + opposite])
    result.attrs.update(table.attrs)
    return result


_COLORS = {'fact': '#096595', 'counter': '#bd5b23', 'dummy': '#8b4088'}
_LABELS = {'fact': '사실', 'counter': '반사실'}


def _style():
    style = {
        'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 11,
        'xtick.labelsize': 9, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 160, 'savefig.bbox': 'tight', 'axes.unicode_minus': False,
    }
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.isfile(font):
        font_manager.fontManager.addfont(font)
        style['font.family'] = font_manager.FontProperties(fname=font).get_name()
    return style


def _values(rows, column):
    return (rows[column].to_numpy(dtype=float, copy=True) if column in rows.columns
            else np.full(len(rows), np.nan))


def _ratio_key(value):
    fact, counter = map(int, str(value).split(':'))
    return fact / (fact + counter), fact, counter


def _figure(table, position, format_name, unit, n_boot, status):
    rows = table[(table['position'] == position) & (table['format'] == format_name)]
    expected = status.get('expected_ratios') or []
    expected = [':'.join(map(str, ratio)) if isinstance(ratio, (list, tuple)) else str(ratio)
                for ratio in expected]
    ratios = sorted(set(rows['ratio']) | set(expected), key=_ratio_key)
    if not ratios:
        return None
    x = np.arange(len(ratios), dtype=float)
    def limits(sides):
        values = np.concatenate([
            _values(rows, f'{side}_{unit}_{condition}{bound}')
            for side in sides for condition in (('B',) if side == 'dummy' else ('A', 'B'))
            for bound in ('', '_lo', '_hi')])
        finite = values[np.isfinite(values)]
        low, high = (min(0., float(finite.min())), max(0., float(finite.max()))) if finite.size else (-1., 1.)
        span = max(.1, high - low)
        return low - span * .06, high + span * .24

    # All four existing-document panels share a scale; the two dummy panels
    # share their own scale. A zero baseline and negative pushes are retained.
    original_limits, dummy_limits = limits(('fact', 'counter')), limits(('dummy',))
    fig, axes = plt.subplots(3, 2, figsize=(17.6, 14.2),
                             gridspec_kw={'height_ratios': [1, 1, .85]})
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.17, hspace=.43, wspace=.12)
    preview = '진행 중 결과 · ' if status.get('preview') else ''
    fig.suptitle(preview + '더미 추가 전·후, 문서의 영향이 얼마나 달라졌는가?',
                 x=.065, y=.975, ha='left', fontsize=21, fontweight='bold')
    format_label = '전체 문서 타입 평균' if format_name == 'ALL' else str(format_name)
    unit_label = '문서당 평균 push' if unit == 'mean' else '문항 내 push 합의 평균'
    position_label = {'last': '맨 뒤', 'first': '맨 앞', 'middle': '중간'}.get(position, position)
    fig.text(.065, .938, f'{format_label}  |  {unit_label}  |  더미: {position_label}', fontsize=12)
    fig.text(.065, .907, '막대 높이 = push   ·   막대 위 Δ = 추가 후 − 전 (음수: 감소 / 양수: 증가)',
             fontsize=12, color='#344453')
    fig.legend(handles=[Patch(facecolor='#edf1f4', edgecolor='#566b7d', hatch='///', label='추가 전'),
                        Patch(facecolor='#566b7d', edgecolor='#566b7d', label='추가 후')],
               loc='upper left', bbox_to_anchor=(.06, .887), ncol=2, frameon=False, fontsize=12)
    for column, split in enumerate(('fact', 'counter')):
        group = rows[rows['split'] == split].set_index('ratio').reindex(ratios)
        counts = _values(group, 'n_questions')
        axes[0, column].text(0, 1.23, '내재 지식: ' + _LABELS[split],
                             transform=axes[0, column].transAxes, fontsize=17, fontweight='bold')
        for row, side in enumerate(('fact', 'counter', 'dummy')):
            ax = axes[row, column]
            color = _COLORS[side]
            dummy = side == 'dummy'
            title = ('더미 문서 · 내재 지식 방향 (추가 후만)' if dummy else
                     _LABELS[side] + ' 문서 · ' + ('내재 지식과 일치' if side == split else '내재 지식과 반대'))
            ax.set_title(title, loc='left', color=color, fontsize=12, fontweight='bold', pad=12)
            low, high = dummy_limits if dummy else original_limits
            ax.set_ylim(low, high)
            ax.axhline(0, color='#475767', linewidth=.9)
            if '4:4' in ratios:
                at = ratios.index('4:4')
                ax.axvspan(at - .47, at + .47, color='#fff0ca', zorder=0)
            absent = (np.zeros(len(group), dtype=bool) if dummy else _values(group, 'n_' + side) == 0)
            tops = np.zeros(len(group))
            observed = np.zeros(len(group), dtype=bool)
            for condition, offset in ((('B', 0.),) if dummy else (('A', -.18), ('B', .18))):
                name = f'{side}_{unit}_{condition}'
                y, lo, hi = [_values(group, name + bound) for bound in ('', '_lo', '_hi')]
                for values in (y, lo, hi):
                    values[absent] = np.nan
                valid = np.isfinite(y)
                before = condition == 'A'
                ax.bar(x[valid] + offset, y[valid], width=.32 if not dummy else .48,
                       facecolor=(*matplotlib.colors.to_rgb(color), .13) if before else color,
                       edgecolor=color, linewidth=.8, hatch='///' if before else None, zorder=3)
                for at, value, lower, upper in zip(x[valid] + offset, y[valid], lo[valid], hi[valid]):
                    if np.isfinite([lower, upper]).all():
                        ax.vlines(at, lower, upper, color='#263b49', linewidth=.85, zorder=4)
                        ax.plot([at, at], [lower, upper], '_', color='#263b49', markersize=3.5, zorder=4)
                tops = np.maximum(tops, np.where(valid, np.maximum(y, np.nan_to_num(hi, nan=-np.inf)), 0.))
                observed |= valid
            if not dummy:
                for at, top, delta, has_value in zip(x, tops, _values(group, f'{side}_{unit}_delta'), observed):
                    if has_value and np.isfinite(delta):
                        label = f'Δ{delta:+.3f}'.replace('-', '−')
                        ax.text(at, top + (high - low) * .035, label, ha='center', va='bottom',
                                fontsize=8, color='#344453')
                for at in x[absent]:
                    ax.annotate('문서 없음', (at, 0), xytext=(0, 6), textcoords='offset points',
                                ha='center', fontsize=8, color='#6e7b87')
            tick_labels = ([f'{ratio}\nN={int(n):,}' if np.isfinite(n) else f'{ratio}\nN=0'
                            for ratio, n in zip(ratios, counts)] if dummy else ratios)
            ax.set_xticks(x, tick_labels)
            ax.set_xlim(-.6, max(len(ratios) - .4, .6))
            ax.grid(axis='y', color='#e1e7eb', linewidth=.7)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if column == 0:
                ax.set_ylabel('내재 지식 방향 영향' if dummy else unit_label, labelpad=9)
            if dummy:
                ax.set_xlabel('기존 문서 수 (사실 : 반사실)', labelpad=10)
            if not observed.any():
                ax.text(.5, .5, '추가 전후가 모두 완료된 문항 없음',
                        ha='center', va='center', transform=ax.transAxes, color='#617080')
    fig.text(.065, .089, 'push: 사실 문서 = IE / 반사실 문서 = −IE. 양수는 문서 라벨 방향, 음수는 반대 방향입니다.',
             fontsize=10, color='#526170')
    fig.text(.065, .067, '더미는 사실 분할에서 IE, 반사실 분할에서 −IE: 내재 지식 방향을 기준으로 표시합니다. 더미의 내용 일치를 보장하지 않습니다.',
             fontsize=10, color='#526170')
    fig.text(.065, .045, f'세로선: 문항 단위 부트스트랩 95% 신뢰구간 ({n_boot:,}회). 변화의 불확실성은 표의 Δ 신뢰구간을 확인하세요.',
             fontsize=10, color='#526170')
    fig.text(.065, .023, '노란 배경 4:4는 기존 문서 8개, 나머지는 9개입니다. 기존 문서 4패널은 같은 y축, 더미 2패널은 별도의 같은 y축을 씁니다.',
             fontsize=10, color='#526170')
    if status.get('preview'):
        fig.text(.065, .002, '진행 중 결과입니다. N은 완료된 짝의 문항 수이며, 진행에 따라 값이 바뀝니다.', fontsize=10, color='#945c12')
    return fig


def plot_push(table, out_dir, n_boot, status=None):
    """Return one overview PNG per position and all selectable PNGs for HTML.

    figures[position][format][unit] contains a base64 PNG. Only the ALL mean
    view is saved as a separate file; there are no PDFs or raw-IE plot copies.
    """
    if table is None or table.empty:
        return [], {}
    status = status or {}
    paths, figures = [], {}
    with plt.rc_context(_style()):
        for position in table['position'].drop_duplicates():
            formats = table.loc[table['position'] == position, 'format'].drop_duplicates()
            figures[str(position)] = {}
            overview = 'ALL' if 'ALL' in set(formats) else formats.iloc[0]
            for format_name in formats:
                variants = figures[str(position)][str(format_name)] = {}
                for unit in ('mean', 'sum'):
                    fig = _figure(table, position, format_name, unit, n_boot, status)
                    if fig is None:
                        continue
                    try:
                        with io.BytesIO() as buffer:
                            fig.savefig(buffer, format='png', facecolor='white')
                            png = buffer.getvalue()
                        variants[unit] = base64.b64encode(png).decode('ascii')
                        if format_name == overview and unit == 'mean':
                            safe = str(position).replace('/', '_').replace(os.sep, '_')
                            path = os.path.join(out_dir, f'fig_influence_{safe}.png')
                            with open(path, 'wb') as handle:
                                handle.write(png)
                            paths.append(path)
                    finally:
                        plt.close(fig)
    return paths, figures


def _script_json(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def _records(table):
    if table is None or table.empty:
        return []
    if any(name is not None for name in table.index.names):
        table = table.reset_index()
    table = table.astype(object).where(table.notna(), None)

    def plain(value):
        if hasattr(value, 'item'):
            value = value.item()
        return None if isinstance(value, float) and not math.isfinite(value) else value

    return [{str(key): plain(value) for key, value in row.items()}
            for row in table.to_dict(orient='records')]


def write_push_table(table, out_dir, n_boot, seed, figures=None, status=None):
    """Write display-only mean/sum push and paired changes to influence_table.html."""
    replacements = {
        '__PUSH_DATA__': _script_json(_records(table)),
        '__FIGURE_DATA__': _script_json(figures or {}),
        '__STATUS_DATA__': _script_json(status or {}),
        '__N_BOOT__': f'{int(n_boot):,}',
        '__SEED__': str(int(seed)),
    }
    document = re.sub('|'.join(map(re.escape, replacements)),
                      lambda match: replacements[match.group()], _HTML)
    path = os.path.join(out_dir, 'influence_table.html')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(document)
    return [path]


_HTML = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>더미 추가 전후 문서 push 비교</title>
<style>
:root{color-scheme:light;--ink:#172b3b;--muted:#536779;--line:#d9e2e8;--fact:#096595;--counter:#bd5b23;--dummy:#8b4088}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:14px/1.65 system-ui,-apple-system,"Noto Sans CJK KR",sans-serif}main{max-width:1580px;margin:auto;padding:28px 24px 50px}h1{font-size:27px;line-height:1.4;margin:0 0 12px}h2{font-size:19px;margin:0 0 12px}p{margin:8px 0 12px}.card{margin-top:18px;background:white;border:1px solid var(--line);border-radius:10px;padding:20px}.muted,.ci{color:var(--muted)}.muted{font-size:13px}.status{padding:12px 16px;background:#eef4f8;border:1px solid #c8d8e5;border-radius:8px;margin:14px 0}.status.preview{background:#fff6e7;border-color:#e5c891}.status strong{display:block;font-size:15px}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:end}.controls label{display:grid;gap:5px;font-weight:600}select,button{background:white;color:var(--ink);font:inherit;border:1px solid #aabac7;border-radius:6px;padding:7px 10px}button{cursor:pointer}.controls .check{display:flex;align-items:center;gap:6px;padding-bottom:7px}.fact{color:var(--fact)}.counter{color:var(--counter)}.dummy{color:var(--dummy)}.table-wrap{overflow:auto;max-height:74vh;border:1px solid var(--line);border-radius:7px;margin-top:13px}table{border-collapse:separate;border-spacing:0;width:100%;min-width:1050px;white-space:nowrap;font-size:12px}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid #e2e8ed;vertical-align:middle}th.label,td.label{text-align:left}thead th{position:sticky;top:0;background:#eaf1f6;z-index:2;line-height:1.45}thead tr:nth-child(2) th{top:35px;background:#f2f6f9}.sum{background:#faf8fd}thead th.sum{background:#eee9f5}tbody tr.group-start td{border-top:1px solid #aebdca}tbody tr:hover td{background:#f1f6fa}.number{font-variant-numeric:tabular-nums}.value{font-weight:650}.ci{font-size:10px;display:block;margin-top:2px}.no-ci .ci{display:none}.na{font-weight:400;color:#6e7b87}.identity-note{display:block;font-size:10px;color:var(--muted)}.empty{padding:24px;text-align:center;white-space:normal}.figure-wrap{overflow:auto}.figure-wrap img{display:block;min-width:800px;width:100%;height:auto}.formula{padding:10px 14px;border-left:3px solid #54788e;background:#f4f8fa}details>summary{cursor:pointer;font-size:15px;font-weight:650}details[open]>summary{margin-bottom:10px}.notes{max-width:1250px;font-size:13px}a{color:#145d82}[hidden]{display:none!important}.export{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:14px}
@media(max-width:700px){main{padding:20px 12px}h1{font-size:23px}.card{padding:13px}}
@media print{body{background:white}main{padding:0}.controls,.export{display:none}.table-wrap{overflow:visible;max-height:none}table{min-width:0;font-size:8px}th,td{padding:4px}thead th{position:static}.card{padding:10px}.figure-wrap img{min-width:0}details:not([open]){display:none}}
</style></head><body><main>
<h1>더미를 추가하면 기존 사실·반사실 문서의 영향이 얼마나 달라지는가?</h1>
<p>같은 문항의 <b>추가 전</b>(기존 문서만)과 <b>추가 후</b>(같은 기존 문서 + 더미)를 비교합니다.<br>
표와 막대그래프 모두 <b>push: 기준 방향으로 미는 힘</b>을 보여줍니다. 기존 문서는 <b>자기 라벨 방향</b>, 더미는 <b>분할의 내재 지식 방향</b>이 기준입니다.</p>
<div id="run-status" class="status" role="status"></div>
<p class="formula"><b>push가 양수:</b> 기준 방향으로 작용 · <b>음수:</b> 기준과 반대 방향으로 작용.<br>
<b>추가 후 막대의 값이 더 낮으면</b> 기준 방향의 영향 감소, <b>더 높으면</b> 증가입니다. 표의 변화량은 후 − 전입니다.</p>

<section class="card">
<h2>더미 추가 전후 push 비교표</h2>
<div class="controls">
<label>내재 지식<select id="split"></select></label>
<label>문서 타입<select id="format"></select></label>
<label>사실 : 반사실 문서 수<select id="ratio"></select></label>
<label>더미 위치<select id="position"></select></label>
<label class="check"><input type="checkbox" id="intervals" checked>95% 신뢰구간</label>
</div>
<p id="count" class="muted" aria-live="polite"></p>
<div class="table-wrap"><table id="push-table"><thead><tr>
<th rowspan="2" class="label">내재 지식 · 문서 타입</th><th rowspan="2">비율</th><th rowspan="2">문항 N</th><th rowspan="2" class="label">문서 그룹</th>
<th colspan="3">문서당 평균 push</th><th colspan="3" class="sum">문항 내 문서 push 합</th>
</tr><tr><th>추가 전</th><th>추가 후</th><th>변화 · 후 − 전</th><th class="sum">추가 전</th><th class="sum">추가 후</th><th class="sum">변화 · 후 − 전</th></tr></thead><tbody></tbody></table></div>
<p class="muted"><b>문항 내 합:</b> 해당 문항의 사실 문서들 또는 반사실 문서들의 개별 push를 더한 뒤 문항 평균을 낸 값입니다. 전체 데이터의 누적 합이 아닙니다.
<b>문서당 평균:</b> 문항 내 합을 해당 그룹의 문서 수로 나눈 값입니다. 더미는 1개이므로 평균과 합이 같습니다.</p>
<p class="muted">더미는 추가 전에는 존재하지 않으므로 추가 전·변화량을 ‘—’로 표시합니다. 사실·반사실 문서가 없는 조건은 ‘문서 없음’으로 표시합니다.
4:4는 기존 문서 8개, 나머지 0:9~9:0은 9개입니다.</p>
<div class="export"><button id="download-table" type="button">선택한 표 CSV 내려받기</button><span class="muted">별도 CSV는 버튼을 누를 때만 생성됩니다.</span></div>
</section>

<section class="card" id="figure-section">
<h2 id="figure-title">막대 높이로 비교하는 추가 전후 push</h2>
<div class="controls">
<label>그래프 단위<select id="unit"><option value="mean">문서당 평균 push</option><option value="sum">문항 내 문서 push 합</option></select></label>
</div>
<p id="figure-formula" class="formula"></p>
<p class="muted" id="figure-note"></p>
<div class="figure-wrap"><img id="push-figure" alt="내재 지식과 문서 비율에 따른 사실·반사실 문서의 추가 전후 push 막대 비교 및 더미의 추가 후 push" loading="lazy"></div>
<p id="figure-empty" class="muted" hidden>선택한 조건의 그래프가 아직 없습니다.</p>
<div class="export"><a id="download-figure" download="push_before_after.png">현재 그래프 PNG 내려받기</a></div>
</section>

<details class="card"><summary>계산 방법과 해석 · 필요한 내용만</summary><div class="notes">
<p><b>S(Score, 점수)</b>는 사실 답의 로그확률 점수에서 반사실 답의 로그확률 점수를 뺀 값이며, 답변 길이 정규화는 측정 실행의 설정을 따릅니다. <b>IE(Indirect Effect, 간접 효과)</b>는 전체 입력의 S에서 대상 문서만 가린 입력의 S를 뺀 값입니다. 대상 문서의 어텐션 마스크만 0으로 바꾸고 나머지 문서의 순서를 유지합니다. 이 마스킹은 영향 측정이며 억제 개입 실험이 아닙니다.</p>
<p><b>push 변환:</b> 사실 문서의 push = IE, 반사실 문서의 push = −IE입니다. 더미는 사실 분할에서 IE, 반사실 분할에서 −IE로 표시해 <b>내재 지식 방향</b>을 기준으로 삼습니다. 이는 더미의 실제 내용이 해당 방향과 일치함을 보장하지 않습니다. S·IE의 계산은 그대로이며, 절댓값을 취하지 않습니다.</p>
<p><b>추가 전후 비교:</b> 같은 문항·문서 타입·문서 비율에서 추가 전후가 모두 완료된 짝만 사용합니다. 막대는 추가 전·후 각각의 push 평균이며, 표의 변화량은 문항별 ‘후 − 전’을 먼저 구한 뒤 평균합니다. 부분 작성 중인 기록과 짝이 없는 결과는 포함하지 않습니다.</p>
<p><b>전체 문서 타입 평균:</b> 해당 비율에서 예정된 모든 문서 타입의 추가 전후 측정이 완료된 문항만 사용합니다. 문항 안에서 타입들을 같은 가중치로 평균한 뒤, 문항들을 같은 가중치로 평균합니다. N은 독립 문항 수이며, 같은 문항이 여러 비율에 등장하므로 N을 행별로 더하지 않습니다.</p>
<p><b>막대 예시:</b> 반사실 문서의 IE가 −0.8 → −0.3이면 push는 +0.8 → +0.3이고, 추가 후 막대가 낮아집니다. 표의 변화량은 −0.5입니다. 사실 문서의 IE가 +0.8 → +0.3이어도 push와 변화량은 같습니다. push가 0을 지나 음수가 되면 기준 반대 방향으로 작용한 것입니다.</p>
<p><b>불확실성:</b> 문항 단위 부트스트랩 __N_BOOT__회, 난수 시드 __SEED__, 95% 신뢰구간입니다. 막대의 세로선은 각 시점의 구간이며, <b>변화의 유의성은 표의 짝지은 변화량 구간</b>으로 확인합니다. 두 막대의 구간이 겹치는지만으로 판단하지 않습니다. 개별 문서·타입을 독립 문항으로 세지 않으며, 비율별 구간은 다중 비교 미보정입니다. 문항 수가 부족하면 구간을 표시하지 않습니다.</p>
<p><b>확증 편향의 징후:</b> 먼저 같은 비율에서 내재 지식과 일치하는 기존 문서가 추가 전에 더 강하게 작용하는지 확인합니다. 그다음 더미 추가 후 그쪽의 push가 더 감소하는지 확인합니다. 문서 설득력 차이와 중복 정보의 영향도 있으므로 이 결과만으로 확증 편향을 확정하거나 감소한 영향이 더미로 그대로 이동했다고 단정하지 않습니다. 개별 문서 push의 합은 문서 그룹을 한꺼번에 제거한 효과와 같지 않습니다.</p>
</div></details>
<noscript><p class="card">포함된 표와 그래프를 표시하려면 JavaScript를 활성화하세요.</p></noscript>
</main>
<script id="push-data" type="application/json">__PUSH_DATA__</script>
<script id="figure-data" type="application/json">__FIGURE_DATA__</script>
<script id="status-data" type="application/json">__STATUS_DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('push-data').textContent);
const figures=JSON.parse(document.getElementById('figure-data').textContent);
const status=JSON.parse(document.getElementById('status-data').textContent);
const $=id=>document.getElementById(id), valid=v=>typeof v==='number'&&Number.isFinite(v);
const unique=key=>[...new Set(data.map(r=>String(r[key])))];
const splitName=s=>s==='fact'?'사실':s==='counter'?'반사실':s;
const formatName=f=>f==='ALL'?'전체 문서 타입 평균':f;
const sideName=(side,split)=>side==='dummy'?'더미 ('+splitName(split)+' 방향)':splitName(side)+' 문서';
const ratioOrder=(a,b)=>{const [af,ac]=a.split(':').map(Number),[bf,bc]=b.split(':').map(Number);return af/(af+ac)-bf/(bf+bc)||af-bf||ac-bc;};
const signed=v=>valid(v)?(v<0?'−':'+')+Math.abs(v).toFixed(3):'—';
function select(id,values,label,all,preferred){if(all){const o=new Option('모두','*');$(id).add(o);}values.forEach(v=>$(id).add(new Option(label(v),v)));if(values.includes(preferred))$(id).value=preferred;$(id).addEventListener('change',render);}
select('split',unique('split').sort((a,b)=>['fact','counter'].indexOf(a)-['fact','counter'].indexOf(b)),splitName,true);
select('format',unique('format').sort((a,b)=>a==='ALL'?-1:b==='ALL'?1:a.localeCompare(b)),formatName,false,'ALL');
select('ratio',unique('ratio').sort(ratioOrder),x=>x,true);
select('position',unique('position'),x=>({last:'맨 뒤',first:'맨 앞',middle:'중간'}[x]||x),false,'last');
$('unit').addEventListener('change',renderFigure);
$('intervals').addEventListener('change',()=>document.body.classList.toggle('no-ci',!$('intervals').checked));
const statusTitle=document.createElement('strong');
statusTitle.textContent=status.preview?'진행 중 결과 · 완료된 추가 전후 짝만 분석':'측정 결과 · 완료된 추가 전후 짝만 분석';
$('run-status').classList.toggle('preview',Boolean(status.preview));$('run-status').append(statusTitle);
const completion=['fact','counter'].map(s=>{const done=status.completed_counts?.[s],total=status.expected_counts?.[s];return valid(done)?splitName(s)+' '+done.toLocaleString()+(valid(total)?' / '+total.toLocaleString():'')+'문항':'';}).filter(Boolean).join(' · ');
const statusText=document.createElement('span');statusText.textContent=[status.run_name,status.captured_at,completion?'저장된 문항: '+completion:'',status.preview?'진행에 따라 표의 N과 결과가 바뀝니다.':''].filter(Boolean).join('  |  ');$('run-status').append(statusText);
function rows(){return data.filter(r=>['split','format','ratio','position'].every(k=>$(k).value==='*'||String(r[k])===$(k).value)).sort((a,b)=>['fact','counter'].indexOf(a.split)-['fact','counter'].indexOf(b.split)||ratioOrder(a.ratio,b.ratio));}
function cell(tr,text,cls=''){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;tr.append(td);return td;}
function metric(tr,row,key,absent,missing,sum){const td=cell(tr,'','number'+(sum?' sum':'')),span=document.createElement('span');span.className='value';if(absent||missing||!valid(row[key])){span.className+=' na';span.textContent=absent?'문서 없음':missing?'—':'미산출';td.append(span);return;}span.textContent=signed(row[key]);td.append(span);const ci=document.createElement('span');ci.className='ci';ci.textContent=valid(row[key+'_lo'])&&valid(row[key+'_hi'])?'['+signed(row[key+'_lo'])+', '+signed(row[key+'_hi'])+']':'구간 미산출';td.append(ci);td.title='push: '+key;}
function render(){
  const selected=rows(),body=$('push-table').querySelector('tbody');body.replaceChildren();
  selected.forEach(row=>{['fact','counter','dummy'].forEach((side,i)=>{
    const tr=document.createElement('tr');if(i===0)tr.className='group-start';
    const identity=cell(tr,splitName(row.split),'label'),note=document.createElement('span');note.className='identity-note';note.textContent=formatName(row.format);identity.append(note);
    cell(tr,row.ratio);cell(tr,valid(row.n_questions)?row.n_questions.toLocaleString():'—');cell(tr,sideName(side,row.split),'label '+side);
    const absent=side!=='dummy'&&row['n_'+side]===0;
    for(const unit of ['mean','sum'])for(const condition of ['A','B','delta'])metric(tr,row,side+'_'+unit+'_'+condition,absent,side==='dummy'&&condition!=='B',unit==='sum');
    body.append(tr);
  });});
  if(!selected.length){const tr=document.createElement('tr');cell(tr,'선택한 조건에서 추가 전후가 모두 완료된 문항이 없습니다.','empty').colSpan=10;body.append(tr);}
  const counts=selected.map(r=>r.n_questions).filter(valid),range=counts.length?(Math.min(...counts)===Math.max(...counts)?Math.min(...counts).toLocaleString():Math.min(...counts).toLocaleString()+'~'+Math.max(...counts).toLocaleString()):'0';
  $('count').textContent=selected.length+'개 조건 · 조건당 '+range+'문항 · N은 각 조건에서 실제 비교에 포함된 문항 수입니다.';$('download-table').disabled=!selected.length;renderFigure();
}
function renderFigure(){
  const position=$('position').value,format=$('format').value,unit=$('unit').value,encoded=figures[position]?.[format]?.[unit],exists=typeof encoded==='string';
  $('push-figure').hidden=!exists;$('figure-empty').hidden=exists;$('download-figure').hidden=!exists;
  $('figure-title').textContent='막대 높이로 비교하는 추가 전후 push · '+formatName(format);
  $('figure-formula').textContent='옅은 빗금 막대: 추가 전 / 진한 막대: 추가 후. 후의 값이 낮아지면 기준 방향 영향 감소, 높아지면 증가입니다. 더미는 추가 후 막대만 표시합니다.';
  $('figure-note').textContent='열은 내재 지식(왼쪽 사실·오른쪽 반사실), 행은 문서 그룹(사실·반사실·더미)입니다. x축은 기존 문서 비율, y축은 '+(unit==='mean'?'문서당 평균 push':'문항 내 문서 push 합')+'입니다. 문서 타입·더미 위치 선택이 적용되며, 표의 내재 지식·비율 필터는 그래프에 적용하지 않습니다. 세로선은 95% 신뢰구간입니다.';
  if(exists){const uri='data:image/png;base64,'+encoded;if($('push-figure').getAttribute('src')!==uri)$('push-figure').src=uri;$('download-figure').href=uri;$('download-figure').download=('push_before_after_'+format+'_'+position+'_'+unit).replace(/[^A-Za-z0-9_.-]/g,'_')+'.png';}
  else{$('push-figure').removeAttribute('src');$('download-figure').removeAttribute('href');}
}
$('download-table').addEventListener('click',()=>{
  const columns=['내재 지식','문서 타입','비율','더미 위치','문항 N','문서 그룹','push 기준 방향'];
  for(const u of ['문서당 평균 push','문항 내 push 합'])for(const c of ['추가 전','추가 후','후−전'])columns.push(u+' '+c,u+' '+c+' 95% 하한',u+' '+c+' 95% 상한');
  const exported=[columns];
  for(const row of rows())for(const side of ['fact','counter','dummy']){
    const direction=side==='dummy'?'분할의 내재 지식 방향: '+splitName(row.split):'문서 라벨 방향: '+splitName(side);
    const record=[splitName(row.split),formatName(row.format),row.ratio,row.position,row.n_questions,sideName(side,row.split),direction];
    for(const unit of ['mean','sum'])for(const condition of ['A','B','delta'])for(const bound of ['','_lo','_hi']){
      const absent=side!=='dummy'&&row['n_'+side]===0,missing=side==='dummy'&&condition!=='B',value=row[side+'_'+unit+'_'+condition+bound];record.push(absent||missing||!valid(value)?'':value);
    }
    exported.push(record);
  }
  const quote=v=>'"'+String(v).replace(/"/g,'""')+'"',blob=new Blob(['\ufeff'+exported.map(r=>r.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='push_table.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></body></html>
'''


def analyze(args, out_dir: str):
    config_path = os.path.join(out_dir, 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as handle:
            config = json.load(handle)
    status = {'preview': args.preview, 'captured_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
              'run_name': os.path.basename(out_dir), 'expected_formats': config.get('formats', args.formats),
              'expected_ratios': config.get('ratios', args.ratios),
              'expected_counts': {split: config.get('n') for split in args.splits}}
    metas, records = load_results(out_dir, snapshot=args.preview, snapshot_info=status)
    selected = set(args.splits)
    excluded = sum(m['split'] not in selected for m in metas)
    metas = [m for m in metas if m['split'] in selected]
    records = [r for r in records if r['split'] in selected]
    completed = Counter(m['split'] for m in metas)
    status['completed_counts'] = {split: completed[split] for split in args.splits}
    print(f'# 분석 대상 분할 : {args.splits} | 선택하지 않은 기존 측정 문항 {excluded}개 제외')
    print(f'# 분석 시점(KST): {status["captured_at"]} | 완료 문항: {status["completed_counts"]} '
          f'| 분할별 목표: {status["expected_counts"]}')
    print(f'# 읽기 범위: {status["files"]}')
    if args.preview:
        print('# 중간 결과: counter 등 미완료 분할은 처리 순서에 따라 표본 구성이 달라지므로 최종 결과로 해석하지 않는다.')
    if not records:
        raise SystemExit(f'선택한 분할의 측정 결과가 없다 : {args.splits}, {out_dir}')

    log_position_balance(records)

    print('# IE 추가 전후 짝 비교 후 push로 표시 (문항 단위 부트스트랩)')
    influence = ie_tables(records, args.n_boot, args.seed, expected_formats=status['expected_formats'])
    if influence.empty:
        raise SystemExit('A/B 짝 측정이 없어 더미 추가 전후 비교표를 생성할 수 없다')
    report_dir = os.path.join(out_dir, 'preview') if args.preview else out_dir
    os.makedirs(report_dir, exist_ok=True)
    display = as_push_table(influence)
    png_paths, figures = plot_push(display, report_dir, args.n_boot, status=status)
    html_paths = write_push_table(display, report_dir, args.n_boot, args.seed, figures=figures, status=status)
    for path in html_paths + png_paths:
        print(f'# {"중간" if args.preview else "분석"} 결과 : {path}')
    print('# HTML에서 문서 형식을 선택하면 표와 그래프가 함께 바뀝니다. CSV는 필요할 때 HTML에서 내려받습니다.')
    print(f'# 집계 진단: {influence.attrs}')


def main():
    out_dir = ARGS.output_dir or os.path.join(OUTPUT_DIR, ARGS.model, ARGS.run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'# 결과 폴더 : {out_dir}')
    if ARGS.analyze_only:
        print(f'# 분석 설정 : n_boot={ARGS.n_boot}, seed={ARGS.seed} (측정 조건은 아래 저장된 설정 참조)')
    else:
        print(f'# 측정 설정 : {vars(ARGS)}')

    with run_session(ARGS, out_dir):
        if not ARGS.analyze_only:
            measure(ARGS, out_dir)
            if ARGS.num_shards > 1:                                             # 다른 GPU 가 아직 돌고 있을 수 있다
                print('# 샤드 측정 완료. 모든 샤드가 끝나면 --analyze-only 로 합쳐서 분석할 것')
                return
        analyze(ARGS, out_dir)


if __name__ == '__main__':
    main()
