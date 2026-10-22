"""Run one complete vanilla/dummy/ours experiment on one GPU, without sharding.

Keep the notebook's 50 prompts/question and factual-answer accuracy. Reuse
saved dummy documents and rank common candidate scores with configurable budgets.
"""
import argparse
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
GENERATION = dict(seed=42, dtype='bfloat16', attention='sdpa',
                  max_input_tokens=4096, answer_tokens=64)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--method', required=True, choices=('vanilla', 'dummy', 'ours'))
    p.add_argument('--gpu', required=True, help='이 실행을 배치할 GPU 번호 하나')
    p.add_argument('--mlp-layers', type=int, default=1)
    p.add_argument('--mha-layers', type=int, default=4)
    p.add_argument('--heads-per-layer', type=int, default=1)
    p.add_argument('--residual-layers', type=int, default=1)
    p.add_argument('--retain', type=float, default=0.5,
                   help='선택한 MLP/MHA/잔차 경로의 유지율(기본 0.5). 억제율=1−유지율: 1=억제 없음, 0.25=75%% 억제, 0=선택 경로 완전 억제')
    p.add_argument('--attention-renorm', action='store_true',
                   help='저장된 후보는 그대로 사용하고, MHA 더미 기여 억제 후 Attention 가중치를 재정규화')
    p.add_argument('--n', type=int, default=1000, help='분할별 평가 문항 수. 기본 fact/counter 각 1000')
    p.add_argument('--selection-run', default='dummy_intervention', help='개입 후보와 더미를 저장한 실행 이름')
    p.add_argument('--selection-n', type=int, default=1000)
    p.add_argument('--run-name', help='결과 폴더 이름. 생략하면 방법·층/헤드 수·유지율로 자동 생성')
    p.add_argument('--output-dir', type=Path, help='결과 폴더 직접 지정')
    args = p.parse_args()
    if not args.gpu.isdigit():
        p.error('--gpu는 번호 하나입니다. 여러 실험은 서로 다른 명령으로 동시에 실행하세요.')
    if min(args.mlp_layers, args.mha_layers, args.heads_per_layer, args.residual_layers) < 0:
        p.error('층/헤드 수는 0 이상이어야 합니다. 모듈의 층 수가 0이면 해당 개입을 끕니다.')
    if args.method == 'ours' and args.mha_layers and not args.heads_per_layer:
        p.error('MHA를 사용하면 --heads-per-layer는 1 이상이어야 합니다.')
    if args.attention_renorm and (args.method != 'ours' or not args.mha_layers or not args.heads_per_layer):
        p.error('--attention-renorm은 MHA 개입을 사용하는 ours 방법에만 적용할 수 있습니다.')
    if min(args.n, args.selection_n) < 1 or not 0 <= args.retain <= 1:
        p.error('문항 수는 양수, retain은 0~1이어야 합니다.')
    return args


# Set the physical GPU before importing libraries that can initialize CUDA.
if __name__ == '__main__':
    ARGS = arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu

from _init import *
import numpy as np
import torch
from bait.core.bait_prompts import get_generate_prompt
from bait.core.bait_utils import get_model_name_or_path
from bait.core.intervention_data import Dataset, append_line, digest, freeze, load_lines, result_lock
from bait.core.intervention_modules import Intervention
from bait.utils import common_utils, context_utils, model_utils, tokenizer_utils


def select_sites(scores, args):
    """Positive common scores only; ties prefer the lower layer/head index.

    MHA ranks each layer by its top H heads' mean score, then takes L layers.
    Layers need not be consecutive. Increasing L with fixed H nests the sites.
    """
    def top(values, count, name):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f'{name}: 후보 점수가 올바르지 않습니다.')
        indices = [int(i) for i in np.argsort(-values, kind='stable') if values[i] > 0]
        if len(indices) < count:
            raise ValueError(f'{name}: 양수 후보 {len(indices)}개보다 많은 {count}개를 요청했습니다.')
        return indices[:count]

    sites = dict(mlp_layers=top(scores['mlp'], args.mlp_layers, 'MLP'),
                 residual_layers=top(scores['residual'], args.residual_layers, '잔차'), heads={})
    if args.mha_layers:
        heads, layer_scores = {}, []
        for layer, values in enumerate(scores['heads']):
            count = min(args.heads_per_layer, int(np.count_nonzero(np.asarray(values) > 0)))
            heads[layer] = top(values, count, f'MHA 층 {layer}')
            layer_scores.append(float(np.mean(np.asarray(values)[heads[layer]]))
                                if count == args.heads_per_layer else 0.)
        for layer in top(layer_scores, args.mha_layers, 'MHA 층'):
            sites['heads'][str(layer)] = heads[layer]
    return sites


def load_inputs(args):
    config, selection, dummies = None, None, {}
    sites = dict(mlp_layers=[], heads={}, residual_layers=[])
    if args.method != 'vanilla':
        run_dir = ROOT / 'outputs/interventions' / args.model / args.selection_run
        config = json.loads((run_dir / 'config.json').read_text())
        if config['model'] != args.model or any(config['settings'][k] != v for k, v in GENERATION.items()):
            raise ValueError('후보 탐색과 평가의 모델/생성 설정이 다릅니다.')
        rows = [row for path in sorted(run_dir.glob('dummy.shard*.jsonl')) for row in load_lines(path)]
        dummies = {row['case_id']: row for row in rows}
        if len(dummies) != len(rows):
            raise ValueError('더미 캐시에 중복 문항이 있습니다.')
        if args.method == 'ours':
            selection = json.loads((run_dir / f'selection.n{args.selection_n}.json').read_text())
            if selection['config_sha256'] != digest(config) or not selection['ready']:
                raise ValueError('선정 결과의 설정이 다르거나 탐색이 완료되지 않았습니다.')
            sites = select_sites(selection['common']['scores'], args)
    dataset = Dataset(args.model, config['n'] if config else args.n, GENERATION['seed'])
    if config and dataset.fingerprint != config['dataset_sha256']:
        raise ValueError('후보 탐색 이후 원본 데이터 또는 문서 배치가 변경되었습니다.')
    items = [item for split in ('fact', 'counter')
             for item in [x for x in dataset.items if x['split'] == split][:args.n]]
    if len(items) != 2 * args.n:
        raise ValueError('요청한 수만큼의 fact/counter 문항이 없습니다.')
    for item in items:
        if args.method != 'vanilla':
            saved = dummies.get(item['key'], {})
            if saved.get('question') != item['data']['question'] or not saved.get('dummy', '').strip():
                raise ValueError(f'{item["key"]}: 이 문항의 저장된 더미가 없거나 일치하지 않습니다.')
    return dataset, items, dummies, sites, selection


def dummy_tokens(chats, documents, offsets_batch):
    targets = []
    for chat, docs, offsets in zip(chats, documents, offsets_batch):
        if not docs:
            targets.append([])
            continue
        spans = context_utils.extract_context_char_spans(chat, docs)
        last = str(len(docs) - 1)
        tokens = context_utils.mapping_context_tok_idxs(offsets, spans)[last]
        if not tokens or offsets[tokens[-1], 1] < spans[last][1]:
            raise ValueError('더미 문서가 잘렸거나 토큰 위치를 찾지 못했습니다.')
        targets.append(tokens)
    return targets


def evaluate_question(args, dataset, item, dummy, sites, model, tokenizer):
    data = item['data']
    prompts, documents, cells = [], [], []
    for fmt in dataset.formats:
        for fact_n, counter_n in dataset.ratios:
            docs = dataset.contexts(item, fmt, (fact_n, counter_n))
            valid = docs is not None
            docs = (docs + ([dummy] if dummy is not None else [])) if valid else []
            documents.append(docs)
            prompts.append(get_generate_prompt(data['question'], docs) if valid else get_generate_prompt(data['question']))
            cells.append(dict(format=fmt, n_fact=fact_n, n_counter=counter_n, valid=valid))
    chats, inputs = model_utils.make_inputs(tokenizer, 'cpu', prompts, GENERATION['max_input_tokens'],
                                            return_offsets_mapping=args.method == 'ours', return_all=True)
    intervention = nullcontext()
    if args.method == 'ours':
        intervention = Intervention(model, dummy_tokens=dummy_tokens(chats, documents, inputs.pop('offset_mapping')),
                                    retain=args.retain, attention_renorm=args.attention_renorm, **sites)
    inputs = inputs.to('cuda:0')
    with intervention, torch.inference_mode():
        outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                 max_new_tokens=GENERATION['answer_tokens'], pad_token_id=tokenizer.pad_token_id,
                                 do_sample=False, temperature=None, top_k=None, top_p=None)
    # Transfer once, keeping the original batch, decoding and generation settings.
    generated = outputs[:, inputs.input_ids.shape[1]:].cpu().tolist()
    texts = [tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]
    for cell, text in zip(cells, texts):
        cell.update(response=text, correct=bool(cell['valid'] and model_utils.is_correct(text, data['answer_fact'])[1]))
    return dict(case_id=item['key'], split=item['split'], question=data['question'],
                answer_fact=data['answer_fact'], answer_counter=data['answer_counter'], results=cells)


def write_accuracy(rows, path):
    """The notebook's factual-answer accuracy, including its ALL aggregation."""
    counts = {}
    for row in rows:
        for result in row['results']:
            for fmt in (result['format'], 'ALL'):
                key = (row['split'], fmt, result['n_fact'], result['n_counter'])
                values = counts.setdefault(key, [0, 0, 0])
                values[0] += result['correct']
                values[1] += result['valid']
                values[2] += not result['valid']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'format', 'n_fact', 'n_counter', 'correct', 'total', 'skipped', 'accuracy'])
        for key, (correct, total, skipped) in sorted(counts.items()):
            writer.writerow([*key, correct, total, skipped, correct / total if total else ''])


def main(args):
    name = args.method
    if args.method == 'ours':
        name += f'_m{args.mlp_layers}_a{args.mha_layers}_h{args.heads_per_layer}_r{args.residual_layers}_keep{args.retain:g}'
        if args.attention_renorm:
            name += '_renorm'
    output_dir = args.output_dir or ROOT / 'outputs/main_experiments' / args.model / (args.run_name or f'{name}_n{args.n}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with result_lock(output_dir, directory=True):
        dataset, items, dummies, sites, selection = load_inputs(args)
        config = dict(version=1, model=args.model, method=args.method, n=args.n, generation=GENERATION,
                      dataset_sha256=dataset.fingerprint, sites=sites,
                      retain=args.retain if args.method == 'ours' else None,
                      selection_sha256=digest(selection) if selection else None,
                      dummy_sha256=digest({x['key']: dummies[x['key']]['dummy'] for x in items}) if dummies else None)
        # Preserve old run configurations exactly when the option is absent.
        if args.attention_renorm:
            config['attention_renorm'] = True
        freeze(output_dir / 'config.json', config)
        path = output_dir / 'responses.jsonl'
        rows = load_lines(path)
        done = {row['case_id'] for row in rows}
        if not done <= {item['key'] for item in items}:
            raise ValueError('현재 평가 대상 이외의 문항이 결과 파일에 있습니다.')
        todo = [item for item in items if item['key'] not in done]
        print(f'# 결과: {output_dir}\n# 방법: {args.method}, 층/헤드(0부터): {sites}\n# 남은 문항: {len(todo)}/{len(items)}', flush=True)
        if args.method == 'ours':
            print(f'# 선택 경로 유지율={args.retain:g} ({args.retain:.0%}), '
                  f'억제율={1-args.retain:g} ({1-args.retain:.0%})', flush=True)
            if args.attention_renorm:
                print('# MHA: 억제 후 재정규화. 기존 저장 후보를 그대로 사용하며 후보를 다시 탐색하지 않습니다.', flush=True)
                print('# MHA 유지율은 재정규화 전 계수입니다. 최종 더미 가중치의 감소율과는 다릅니다.', flush=True)
        batch_size = len(dataset.formats) * len(dataset.ratios)
        print(f'# 문항 1개 = {batch_size}개 입력을 한 배치로 추론', flush=True)
        if todo:
            common_utils.set_seed(GENERATION['seed'])
            print(f'# GPU={args.gpu}, UUID={torch.cuda.get_device_properties(0).uuid}', flush=True)
            model_path = get_model_name_or_path(args.model)
            model = model_utils.get_model(model_path, GENERATION['dtype'], device='cuda:0',
                                          attn_imp=GENERATION['attention'], is_eval=True)
            tokenizer = tokenizer_utils.load_tokenizer(model_path, 'left')
            started = time.monotonic()
            for index, item in enumerate(todo, 1):
                dummy = dummies[item['key']]['dummy'] if args.method != 'vanilla' else None
                row = evaluate_question(args, dataset, item, dummy, sites, model, tokenizer)
                append_line(path, row)
                rows.append(row)
                if index == 1 or index % 10 == 0 or index == len(todo):
                    elapsed = time.monotonic() - started
                    remaining = elapsed / index * (len(todo) - index)
                    print(f'# {len(rows)}/{len(items)}문항, {len(rows)*batch_size}답변, '
                          f'{elapsed:.0f}초 경과, {elapsed/index:.2f}초/문항({batch_size}답변), '
                          f'남은 시간 약 {remaining/60:.1f}분', flush=True)
        write_accuracy(rows, output_dir / 'accuracy.csv')
        print(f'# 완료: {output_dir / "accuracy.csv"}', flush=True)


if __name__ == '__main__':
    main(ARGS)
