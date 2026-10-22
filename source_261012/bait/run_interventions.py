"""독립 개입 실험: calibrate(영향 측정), select(후보 고정), evaluate(답변 저장).

질문과 검색 문서를 읽고, 이 실험에서 제로샷 답변과 더미를 직접 생성한다.
정답으로 후보를 고르지 않는다. 성능 집계·표·그래프는 이 파일에서 만들지 않는다.
"""
from _init import *

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

# 공통 실험 설정. 같은 실행 이름에서 설정을 바꾸면 기존 결과와 섞지 않는다.
SETTINGS = dict(seed=42, dtype='bfloat16', attention='sdpa', max_input_tokens=4096,
                answer_tokens=64, dummy_tokens=192, retention=0.5,
                top_counts={'mlp': 1, 'heads': 4, 'residual': 1})
SEARCH_RATIOS = ['2:7', '4:5', '5:4', '7:2']
METHODS = ('vanilla', 'dummy') + tuple(
    'ours_' + scope + suffix for scope in ('common', 'ratio')
    for suffix in ('', '_no_mlp', '_no_attention', '_no_residual'))
ROOT = Path(__file__).resolve().parents[2]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calibrate', 'select', 'evaluate'))
    parser.add_argument('--model', default='Llama-3.2-3B')
    parser.add_argument('--run-name', default='dummy_intervention')
    parser.add_argument('--n', type=int, default=1000, help='분할별 전체 문항 수')
    parser.add_argument('--calibration-n', type=int, default=1000, help='분할별 후보 탐색/선정 문항 수')
    parser.add_argument('--ratios', nargs='+', default=SEARCH_RATIOS, help='탐색 또는 평가할 사실:반사실 비율')
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=['vanilla', 'dummy', 'ours_common', 'ours_ratio'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 1 <= args.calibration_n <= args.n:
        parser.error('1 <= calibration-n <= n 이어야 합니다.')
    if not 0 <= args.shard < args.num_shards:
        parser.error('샤드 번호와 개수가 올바르지 않습니다.')
    if args.stage == 'select' and (args.shard != 0 or args.num_shards != 1):
        parser.error('select는 모든 샤드가 끝난 뒤 한 번 실행합니다.')
    if len(set(args.ratios)) != len(args.ratios) or len(set(args.methods)) != len(args.methods):
        parser.error('비율과 방법 목록에 중복이 있습니다.')
    return args


# CUDA 장치 탐색을 할 수 있는 라이브러리보다 먼저 GPU를 지정한다.
if __name__ == '__main__':
    ARGS = arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu

import torch
from bait.core.bait_prompts import get_generate_prompt
from bait.core.bait_utils import get_model_name_or_path
from bait.core.intervention_data import (
    Dataset, DummyCache, append_line, digest, freeze, load_lines, result_lock,
)
from bait.core.intervention_modules import Intervention, rank_dummy_sites
from bait.core.intervention_selection import aggregate_selection
from bait.utils import common_utils, context_utils, model_utils, tokenizer_utils


def load_model(args):
    print(f'# GPU={os.environ.get("CUDA_VISIBLE_DEVICES", "all")}, '
          f'UUID={torch.cuda.get_device_properties(0).uuid}', flush=True)
    common_utils.set_seed(SETTINGS['seed'])
    path = get_model_name_or_path(args.model)
    model = model_utils.get_model(path, SETTINGS['dtype'], device='cuda:0',
                                  attn_imp=SETTINGS['attention'], is_eval=True)
    return model, tokenizer_utils.load_tokenizer(path, 'left')


def encode(tokenizer, question, docs):
    text = tokenizer.apply_chat_template(get_generate_prompt(question, docs),
                                        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors='pt')
    if encoded['input_ids'].shape[1] > SETTINGS['max_input_tokens']:
        raise ValueError('입력이 너무 깁니다. 문서를 조용히 잘라서 실험하지 않습니다.')
    positions = context_utils.extract_context_tok_idxs(text, docs, encoded['offset_mapping'][0])
    # 기존 데이터의 빈 문서 슬롯은 유지하되, 내용이 있는 문서의 위치는 반드시 찾는다.
    if any(doc.strip() and not positions[str(i)] for i, doc in enumerate(docs)):
        raise ValueError('문서의 토큰 위치를 찾지 못했습니다.')
    return encoded['input_ids'], [positions[str(i)] for i in range(len(docs))]


@torch.no_grad()
def generate(model, tokenizer, ids):
    ids = ids.to(model.device)
    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                            max_new_tokens=SETTINGS['answer_tokens'], use_cache=True,
                            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def cases(args, dataset):
    limit = args.calibration_n if args.stage == 'calibrate' else None
    for item in dataset.selected(args.shard, args.num_shards, limit):
        for fmt in dataset.formats:
            for ratio in args.ratios:
                docs = dataset.contexts(item, fmt, tuple(map(int, ratio.split(':'))))
                if docs is None:
                    raise ValueError(f'{item["key"]}/{fmt}/{ratio}: 문서가 부족합니다.')
                record = dict(key=item['key'], split=item['split'], sample_rank=item['sample_rank'],
                              format=fmt, ratio=ratio)
                yield item, docs, record


def generated_input(cache, item, model, tokenizer):
    return cache.get(item, model, tokenizer, SETTINGS['max_input_tokens'],
                     SETTINGS['dummy_tokens'], SETTINGS['answer_tokens'])


def calibrate(args, dataset, model, tokenizer, cache, output_dir):
    path = output_dir / f'calibration.shard{args.shard}.jsonl'
    done = {row['case_id'] for row in load_lines(path)}
    for item, docs, record in cases(args, dataset):
        case_id = digest([record['key'], record['format'], record['ratio']])
        if case_id in done:
            continue
        saved = generated_input(cache, item, model, tokenizer)
        ids, spans = encode(tokenizer, item['data']['question'], docs + [saved['dummy']])
        target = tokenizer(saved['zero_shot'], add_special_tokens=False)['input_ids']
        if not target or ids.shape[1] + len(target) - 1 > SETTINGS['max_input_tokens']:
            raise ValueError(f'{item["key"]}: 후보 탐색에 사용할 답변 길이가 올바르지 않습니다.')
        torch.cuda.synchronize()
        started = time.monotonic()
        scores = rank_dummy_sites(model, ids.to(model.device), spans[-1], target)
        torch.cuda.synchronize()
        append_line(path, dict(record, case_id=case_id, ranking=scores,
                               target_tokens=len(target), prompt_tokens=ids.shape[1],
                               seconds=time.monotonic() - started))
        print(f'# 탐색 {record["key"]} | {record["format"]} | {record["ratio"]}', flush=True)


def select(args, dataset, output_dir, config):
    calibration = json.loads((output_dir / 'calibration_config.json').read_text())
    rows = [row for path in sorted(output_dir.glob('calibration.shard*.jsonl')) for row in load_lines(path)]
    selection = aggregate_selection(rows, calibration['ratios'], dataset.formats,
                                    SETTINGS['top_counts'], args.calibration_n)
    if not selection['ready']:
        raise ValueError(f'탐색이 완료되지 않았습니다: {selection["counts"]}')
    selection['config_sha256'] = digest(config)
    path = output_dir / f'selection.n{args.calibration_n}.json'
    freeze(path, selection)
    for name, scope in [*selection['per_ratio'].items(), ('common', selection['common'])]:
        print(name, scope['sites'], scope['unavailable_reason'] or '', flush=True)
    print('# 선정 결과:', path, flush=True)


def read_selection(args, output_dir, config):
    if not any(method.startswith('ours_') for method in args.methods):
        return None
    path = output_dir / f'selection.n{args.calibration_n}.json'
    selection = json.loads(path.read_text())
    if selection['config_sha256'] != digest(config):
        raise ValueError('후보를 선정한 설정과 현재 실험 설정이 다릅니다.')
    for method in args.methods:
        if method.startswith('ours_common'):
            scopes = [selection['common']]
        elif method.startswith('ours_ratio'):
            if not set(args.ratios) <= set(selection['per_ratio']):
                raise ValueError('비율별 개입은 탐색한 비율에서만 가능합니다.')
            scopes = [selection['per_ratio'][ratio] for ratio in args.ratios]
        else:
            continue
        for scope in scopes:
            if not scope['available']:
                raise ValueError(scope['unavailable_reason'])
    return selection


def intervene(model, selection, ratio, dummy_tokens, method):
    if not method.startswith('ours_'):
        return nullcontext()
    sites = (selection['common'] if method.startswith('ours_common') else selection['per_ratio'][ratio])['sites']
    return Intervention(model, dummy_tokens=dummy_tokens, retain=SETTINGS['retention'],
                        mlp_layers=[] if method.endswith('_no_mlp') else sites['mlp_layers'],
                        heads={} if method.endswith('_no_attention') else {int(k): v for k, v in sites['heads'].items()},
                        residual_layers=[] if method.endswith('_no_residual') else sites['residual_layers'])


def evaluate(args, dataset, model, tokenizer, cache, output_dir, selection):
    path = output_dir / f'evaluation.n{args.calibration_n}.shard{args.shard}.jsonl'
    done = {row['case_id'] for row in load_lines(path)}
    for item, docs, record in cases(args, dataset):
        encoded = {}
        for method in args.methods:
            case_id = digest([record['key'], record['format'], record['ratio'], method])
            if case_id in done:
                continue
            add_dummy = method != 'vanilla'
            if add_dummy not in encoded:
                extra = [generated_input(cache, item, model, tokenizer)['dummy']] if add_dummy else []
                encoded[add_dummy] = encode(tokenizer, item['data']['question'], docs + extra)
            ids, spans = encoded[add_dummy]
            with intervene(model, selection, record['ratio'], spans[-1] if add_dummy else [], method):
                response = generate(model, tokenizer, ids)
            append_line(path, dict(record, case_id=case_id, method=method, response=response))
            print(f'# 평가 {record["key"]} | {record["format"]} | {record["ratio"]} | {method}: {response!r}', flush=True)


def main(args):
    output_dir = args.output_dir or ROOT / 'outputs/interventions' / args.model / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    with result_lock(output_dir, directory=True, shared=args.stage != 'select'):
        dataset = Dataset(args.model, args.n, SETTINGS['seed'])
        valid_ratios = {f'{a}:{b}' for a, b in dataset.ratios}
        if not set(args.ratios) <= valid_ratios:
            raise ValueError('문서 비율은 0:9부터 9:0까지 지정할 수 있습니다.')
        config = dict(version=4, model=args.model, n=args.n, settings=SETTINGS,
                      formats=dataset.formats, dataset_sha256=dataset.fingerprint,
                      document_order='main_experiments.ipynb', dummy_generation='own run; one question per batch',
                      calibration='model-generated targets; calibration inputs overlap evaluation inputs')
        with result_lock(output_dir / 'logs', directory=True, blocking=True):
            if (output_dir / 'manifest.json').exists():
                raise ValueError('이전 방식의 결과 폴더입니다. 독립 실험에는 새 run-name을 사용하세요.')
            freeze(output_dir / 'config.json', config)
            if args.stage != 'select':
                freeze(output_dir / 'workers.json', {'num_shards': args.num_shards})
            if args.stage == 'calibrate':
                freeze(output_dir / 'calibration_config.json', {'ratios': args.ratios})
        if args.stage == 'select':
            select(args, dataset, output_dir, config)
            return
        selection = read_selection(args, output_dir, config) if args.stage == 'evaluate' else None
        if args.stage == 'evaluate' and selection is not None:
            with result_lock(output_dir / 'logs', directory=True, blocking=True):
                freeze(output_dir / f'evaluation.n{args.calibration_n}.json', {'selection_sha256': digest(selection)})
        # The same lock prevents simultaneous calibration/evaluation of a shard.
        with result_lock(output_dir / f'dummy.shard{args.shard}.jsonl'):
            cache = DummyCache(output_dir, args.shard)
            model, tokenizer = load_model(args)
            if args.stage == 'calibrate':
                calibrate(args, dataset, model, tokenizer, cache, output_dir)
            else:
                evaluate(args, dataset, model, tokenizer, cache, output_dir, selection)
    print('# 완료:', args.stage, output_dir, flush=True)


if __name__ == '__main__':
    main(ARGS)
