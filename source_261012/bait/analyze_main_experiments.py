"""Compare saved answers: factual accuracy and majority-document faithfulness.

CPU only. Writes one CSV, one HTML report and two PNGs; never changes inference
results. Main scores keep the original is_correct matcher for baseline
comparability. Exact Match (EM) is supplementary. Bootstrap pairs whole questions.
"""
import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import _init  # Register the repository package; no model is loaded.
from bait.utils.model_utils import is_correct

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAPS, SEED = 2000, 42
COLORS = ['#65748b', '#d19c39', '#157eaa', '#b75d91', '#27855c', '#7753ab', '#ca6036', '#167d80']
REVIEW_PER_GROUP = 2
REVIEW_KINDS = {'empty': '빈 응답·기호만 있는 응답', 'dual': '기존 채점에서 양쪽 정답 인정',
                'normalized': '정규화 완전 일치와 기존 규칙 차이', 'single': '단일 후보 포함·완전 일치 아님',
                'partial': '생성 답변이 정답의 일부인 역방향 일치', 'other': '기타 규칙 차이'}


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.sub(r'[\W_]+', ' ', text).split())


def classify(response, fact, counter):
    """Return exact and exclusive-containment labels: fact=1, counter=0, other=-1.

    References must already be normalized and distinct. Both-answer responses
    and empty responses receive no credit. Containment is only a sensitivity
    check: mentioning a reference does not imply affirming it.
    """
    text = normalize(response)
    exact = 1 if text == fact else 0 if text == counter else -1
    if exact >= 0 or not text:
        return exact, exact
    has_fact = f' {fact} ' in f' {text} '
    has_counter = f' {counter} ' in f' {text} '
    loose = int(has_fact) if has_fact != has_counter else -1
    return exact, loose


def record_review(audit, pools, run, row, cell, scores, exact, loose):
    """Count all disagreements; keep a deterministic, stratified inspection sample.

    Scores: legacy accuracy/faithfulness, EM accuracy/faithfulness, containment
    accuracy/faithfulness. Categories describe string rules, not semantic errors.
    """
    differences = [scores[i % 2] != scores[i + 2] for i in range(4)]
    if not any(differences):
        return
    response = cell['response']
    if not normalize(response):
        kind = 'empty'
    elif all(bool(is_correct(response, row[key])[1]) for key in ('answer_fact', 'answer_counter')):
        kind = 'dual'
    elif exact >= 0:
        kind = 'normalized'
    elif loose >= 0:
        kind = 'single'
    elif any(re.search(rf'\b{re.escape(response.lower().strip(" ,.!?~"))}\b', row[key].lower().strip(' ,.!?~'))
             for key in ('answer_fact', 'answer_counter')):
        kind = 'partial'
    else:
        kind = 'other'
    ratio = f'{cell["n_fact"]}:{cell["n_counter"]}'
    key = (run['path'].name, row['split'], cell['format'], ratio, kind)
    counts = audit['counts'].setdefault(key, [0] * 8)
    for i, different in enumerate(differences):
        counts[i] += int(different)
        counts[i + 4] += int(different and scores[i % 2])  # legacy-only credit
    identity = [*key[:4], row['case_id'], row['question'], row['answer_fact'], row['answer_counter'], response]
    case_id = sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]
    pool = pools.setdefault((*key, tuple(differences)), [])
    if len(pool) >= REVIEW_PER_GROUP and case_id >= pool[-1]['id']:
        return
    pool.append(dict(id=case_id, run=key[0], split=key[1], format=key[2], ratio=ratio, kind=kind,
                     case_id=row['case_id'], question=row['question'], fact=row['answer_fact'],
                     counter=row['answer_counter'], response=response, scores=list(map(int, scores))))
    pool.sort(key=lambda case: case['id'])
    del pool[REVIEW_PER_GROUP:]


def load_results(root, n, requested=None):
    folders = [root / name for name in requested] if requested else sorted(root.iterdir())
    runs, pending = [], []
    for folder in folders:
        path = folder / 'config.json'
        if not path.exists():
            if requested:
                raise ValueError(f'Missing configuration: {path}')
            continue
        config = json.loads(path.read_text())
        if config.get('method') in ('vanilla', 'dummy', 'ours') and config.get('n') == n:
            if not requested:
                responses = folder / 'responses.jsonl'
                completed = 0
                if responses.exists():
                    with responses.open('rb') as handle:
                        completed = sum(line.endswith(b'\n') for line in handle)
                if completed < 2 * n:
                    pending.append(dict(run=folder.name, completed=completed, total=2*n))
                    print(f'Skipping unfinished run: {folder.name} ({completed}/{2*n})', flush=True)
                    continue
            runs.append(dict(path=folder, config=config, method=config['method']))
    for method in ('vanilla', 'dummy'):
        if sum(run['method'] == method for run in runs) != 1:
            raise ValueError(f'Exactly one {method} run is required; use --runs to choose runs.')
    if not any(run['method'] == 'ours' for run in runs):
        raise ValueError('No proposed-method results found.')
    runs.sort(key=lambda run: (('vanilla', 'dummy', 'ours').index(run['method']),
                              len(run['config']['sites']['mlp_layers']), len(run['config']['sites']['heads']),
                              sum(map(len, run['config']['sites']['heads'].values())),
                              len(run['config']['sites']['residual_layers']),
                              -(run['config']['retain'] or 0), run['path'].name))
    ours_number = 0
    for index, run in enumerate(runs):
        run['color'] = COLORS[index] if index < len(COLORS) else matplotlib.colors.to_hex(plt.cm.hsv((index * .618) % 1))
        run['retain'] = run['config']['retain'] if run['method'] == 'ours' else None
        run['strength'] = 1 - run['retain'] if run['retain'] is not None else None
        run['budget'] = ''
        if run['method'] == 'ours':
            ours_number += 1
            heads = run['config']['sites']['heads']
            counts = {len(x) for x in heads.values()}
            description = f'{len(heads)}×{next(iter(counts))}' if len(counts) == 1 else f'{len(heads)}층·총{sum(map(len, heads.values()))}헤드'
            sites = run['config']['sites']
            run['budget'] = f'MLP{len(sites["mlp_layers"])}/MHA{description}/잔차{len(sites["residual_layers"])}'
            run['label'] = f'제안 {ours_number} · {run["budget"]} · 억제 {100*run["strength"]:g}%'
            if run['config'].get('attention_renorm', False):
                run['label'] += ' · MHA 재정규화'
        else:
            run['label'] = {'vanilla': '바닐라', 'dummy': '더미'}[run['method']]
    baseline = runs[0]['config']
    for run in runs:
        for key in ('model', 'n', 'generation', 'dataset_sha256'):
            if run['config'][key] != baseline[key]:
                raise ValueError(f'{run["path"].name}: incompatible {key}')
        if run['method'] == 'ours' and run['config']['dummy_sha256'] != runs[1]['config']['dummy_sha256']:
            raise ValueError('Dummy documents differ between methods.')

    reference, formats, cubes = {}, None, {}
    audit = dict(summary=[], counts={}, cases=[], pending=pending)
    pools = {}
    for method_index, run in enumerate(runs):
        data = (run['path'] / 'responses.jsonl').read_bytes()
        if not data.endswith(b'\n'):
            raise ValueError(f'Incomplete final line: {run["path"]}')
        rows = [json.loads(line) for line in data.splitlines()]
        ids = [row['case_id'] for row in rows]
        if len(ids) != len(set(ids)) or len(rows) != 2 * n:
            raise ValueError(f'{run["path"].name}: expected {2*n} unique completed questions.')
        if formats is None:
            formats = list(dict.fromkeys(cell['format'] for cell in rows[0]['results']))
            cubes = {split: np.zeros((len(runs), n, len(formats), 10, 4)) for split in ('fact', 'counter')}
        for split in ('fact', 'counter'):
            subset = sorted((row for row in rows if row['split'] == split), key=lambda row: row['case_id'])
            if len(subset) != n:
                raise ValueError(f'{run["path"].name}/{split}: expected {n} questions.')
            legacy_correct, both_answers, unmatched = 0, 0, 0
            for index, row in enumerate(subset):
                meta = [row[k] for k in ('case_id', 'split', 'question', 'answer_fact', 'answer_counter')]
                if method_index == 0:
                    reference[split, index] = meta
                elif reference[split, index] != meta:
                    raise ValueError('Question IDs or reference answers differ between methods.')
                fact, counter = normalize(row['answer_fact']), normalize(row['answer_counter'])
                if not fact or not counter or fact == counter:
                    raise ValueError(f'{row["case_id"]}: references are empty or indistinguishable.')
                cells = {(cell['format'], cell['n_fact'], cell['n_counter']): cell for cell in row['results']}
                expected = {(fmt, k, 9-k) for fmt in formats for k in range(10)}
                if set(cells) != expected or len(row['results']) != len(expected):
                    raise ValueError(f'{row["case_id"]}: missing/duplicate document conditions.')
                for fi, fmt in enumerate(formats):
                    for ratio in range(10):
                        cell = cells[fmt, ratio, 9-ratio]
                        if not cell['valid']:
                            raise ValueError('Invalid document conditions require an explicit paired exclusion policy.')
                        exact, loose = classify(cell['response'], fact, counter)
                        majority = int(ratio >= 5)
                        fact_match = bool(is_correct(cell['response'], row['answer_fact'])[1])
                        counter_match = bool(is_correct(cell['response'], row['answer_counter'])[1])
                        if fact_match != bool(cell['correct']):
                            raise ValueError(f'{row["case_id"]}: current matcher differs from saved factual grading.')
                        faithful = fact_match if majority else counter_match
                        cubes[split][method_index, index, fi, ratio] = [cell['correct'], faithful,
                                                                       exact == majority, loose == majority]
                        record_review(audit, pools, run, row, cell,
                                      [fact_match, faithful, exact == 1, exact == majority, loose == 1, loose == majority],
                                      exact, loose)
                        both_answers += fact_match and counter_match
                        unmatched += exact == -1
                        legacy_correct += cell['correct']
            denominator = n * len(formats) * 10
            audit['summary'].append(dict(split=split, method=run['label'], run=run['path'].name, legacy_accuracy=100*legacy_correct/denominator,
                                         both_answers=both_answers, exact_unmatched=unmatched))
    audit['counts'] = [dict(run=k[0], split=k[1], format=k[2], ratio=k[3], kind=k[4], counts=v)
                       for k, v in audit['counts'].items()]
    audit['cases'] = sorted((case for pool in pools.values() for case in pool), key=lambda case: case['id'])
    return runs, formats, cubes, audit


def direction(low, high):
    return '증가 확인' if low > 0 else '감소 확인' if high < 0 else '불확실'


def estimates(runs, formats, cubes):
    """Equal format/ratio weights; shared question draws pair every method.

    ALL priors is a 50:50 stratified mean. Confidence intervals are pointwise,
    not multiplicity-adjusted; use overall contrasts for the main conclusion.
    """
    ratio_groups = {-1: range(10), -2: range(5), -3: range(5, 10)}
    ratio_names = {-1: 'ALL', -2: 'LOW', -3: 'HIGH'}
    conditions = [(fmt, ratio) for fmt in ['ALL', *formats] for ratio in [-1, *range(10), -2, -3]]
    means, samples, accuracy_samples, sizes = {}, {}, {}, {}
    rng = np.random.default_rng(SEED)
    for split, cube in cubes.items():
        blocks = []
        for fmt, ratio in conditions:
            fs = range(len(formats)) if fmt == 'ALL' else [formats.index(fmt)]
            rs = ratio_groups.get(ratio, [ratio])
            block = np.take(np.take(cube, list(fs), axis=2), list(rs), axis=3).mean(axis=(2, 3))
            blocks.append(block.transpose(1, 0, 2))
        values = np.stack(blocks, axis=1)  # question, condition, method, metric
        n = values.shape[0]
        weights = rng.multinomial(n, np.full(n, 1/n), size=BOOTSTRAPS) / n
        means[split] = values.mean(axis=0) * 100
        samples[split] = (weights @ values[..., 1].reshape(n, -1)).reshape(BOOTSTRAPS, len(conditions), len(runs)) * 100
        accuracy_samples[split] = (weights @ values[..., 0].reshape(n, -1)).reshape(BOOTSTRAPS, len(conditions), len(runs)) * 100
        sizes[split] = n
    means['ALL'] = (means['fact'] + means['counter']) / 2
    samples['ALL'] = (samples['fact'] + samples['counter']) / 2
    accuracy_samples['ALL'] = (accuracy_samples['fact'] + accuracy_samples['counter']) / 2
    sizes['ALL'] = sizes['fact'] + sizes['counter']
    records = []
    for split in ('ALL', 'fact', 'counter'):
        for ci, (fmt, ratio) in enumerate(conditions):
            for mi, run in enumerate(runs):
                accuracy, faithfulness, exact, loose = means[split][ci, mi]
                row = dict(split=split, format=fmt, ratio=ratio_names.get(ratio, f'{ratio}:{9-ratio}'),
                           method=run['label'], run=run['path'].name, n_questions=sizes[split],
                           module_budget=run['budget'], retain=run['retain'] if run['retain'] is not None else '',
                           suppression_pct=100*run['strength'] if run['strength'] is not None else '',
                           n_answers=sizes[split] * (len(formats) if fmt == 'ALL' else 1) * len(ratio_groups.get(ratio, [ratio])),
                           scoring='legacy_is_correct', accuracy_pct=accuracy, faithfulness_pct=faithfulness,
                           exact_faithfulness_pct=exact, containment_faithfulness_pct=loose)
                for name, base in (('vanilla', 0), ('dummy', 1)):
                    delta = samples[split][:, ci, mi] - samples[split][:, ci, base]
                    low, high = np.quantile(delta, [.025, .975])
                    accuracy_delta = accuracy_samples[split][:, ci, mi] - accuracy_samples[split][:, ci, base]
                    accuracy_low, accuracy_high = np.quantile(accuracy_delta, [.025, .975])
                    row.update({f'delta_accuracy_vs_{name}_pp': accuracy - means[split][ci, base, 0],
                                f'accuracy_vs_{name}_ci_low': accuracy_low, f'accuracy_vs_{name}_ci_high': accuracy_high,
                                f'accuracy_change_vs_{name}': direction(accuracy_low, accuracy_high) if mi != base else '기준',
                                f'delta_faithfulness_vs_{name}_pp': faithfulness - means[split][ci, base, 1],
                                f'faithfulness_vs_{name}_ci_low': low, f'faithfulness_vs_{name}_ci_high': high,
                                f'change_vs_{name}': direction(low, high) if mi != base else '기준'})
                if ratio in (-2, -3):
                    row['condition'] = '저품질: 사실 0~4개 / 9개' if ratio == -2 else '고품질: 사실 5~9개 / 9개'
                elif ratio == -1 or split == 'ALL':
                    row['condition'] = '전체 평균' if ratio == -1 else '내재 지식 전체'
                else:
                    conflict = (split == 'fact') != (ratio >= 5)
                    row['condition'] = '다수 문서≠내재 지식 (핵심 조건)' if conflict else '다수 문서=내재 지식 (유지 확인)'
                if run['method'] != 'ours':
                    row['interpretation'] = '비교 기준' if mi == 0 else '더미 추가 효과'
                elif row['change_vs_dummy'] == '증가 확인':
                    row['interpretation'] = {'증가 확인': '성능 가설 지지', '감소 확인': '회복했지만 바닐라 미달',
                                             '불확실': '더미 대비 개선·바닐라 대비 불확실'}[row['change_vs_vanilla']]
                elif row['change_vs_dummy'] == '감소 확인':
                    row['interpretation'] = '더미 대비 개입 후 악화'
                else:
                    row['interpretation'] = '개입 개선 효과 불확실'
                records.append(row)
    return pd.DataFrame(records)


def charts(table, runs, formats, output):
    """Every plotted number and interval comes from the exported table."""
    family = 'Noto Sans CJK JP'
    if any(font.name == family for font in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = family
    plt.rcParams['axes.unicode_minus'] = False
    colors = [run['color'] for run in runs]
    ratios = [f'{k}:{9-k}' for k in range(10)] + ['ALL']
    for metric, measure in (('faithfulness', '충실도'), ('accuracy', '사실 정확도')):
        fig, axes = plt.subplots(2, 2, figsize=(18, 9), layout='constrained')
        for si, split in enumerate(('fact', 'counter')):
            subset = table[(table['split'] == split) & (table['format'] == 'ALL')]
            for mi, run in enumerate(runs):
                rows = subset[subset['run'] == run['path'].name].set_index('ratio').loc[ratios]
                x = np.arange(11)
                width = .8 / len(runs)
                axes[si, 0].bar(x + (mi-(len(runs)-1)/2)*width, rows[f'{metric}_pct'], width,
                                label=run['label'], color=colors[mi % len(colors)])
                if mi:
                    width = .8 / (len(runs)-1)
                    delta = rows[f'delta_{metric}_vs_vanilla_pp'].to_numpy()
                    lo, hi = rows[f'{metric}_vs_vanilla_ci_low'].to_numpy(), rows[f'{metric}_vs_vanilla_ci_high'].to_numpy()
                    pos = x + (mi-1-(len(runs)-2)/2)*width
                    axes[si, 1].bar(pos, delta, width, label=run['label'], color=colors[mi % len(colors)])
                    axes[si, 1].vlines(pos, lo, hi, color='#293448', linewidth=.8)
            prior = '내재 지식: 사실' if split == 'fact' else '내재 지식: 반사실'
            for col, ax in enumerate(axes[si]):
                ax.set_title(prior + (f' · 방법별 {measure}' if col == 0 else f' · 바닐라 대비 {measure} 변화'))
                ax.set_xticks(np.arange(11), ratios[:-1] + ['평균'])
                ax.set_xlabel('원래 검색 문서의 사실:반사실 비율 (더미 제외)')
                ax.grid(axis='y', alpha=.18)
                ax.set_axisbelow(True)
                ax.axvspan(9.5, 10.5, color='#eef0f4', zorder=0)
                start, end = (-.5, 4.5) if split == 'fact' else (4.5, 9.5)
                ax.axvspan(start, end, color='#fff7e9', zorder=0)
                ax.spines[['top', 'right']].set_visible(False)
            axes[si, 0].set_ylim(0, 100)
            axes[si, 0].set_ylabel('다수 문서 기준 충실도 (%)' if metric == 'faithfulness' else '사실 정확도 (%)')
            axes[si, 1].axhline(0, color='#293448', linewidth=1)
            axes[si, 1].set_ylabel(f'{measure} 차이 (퍼센트포인트, pp)')
        axes[0, 0].legend(fontsize=8, ncol=2)
        axes[0, 1].legend(fontsize=8, ncol=2)
        fig.suptitle(f'모든 문서 타입 평균 | 오른쪽 양수: 바닐라보다 {measure} 증가 / 음수: 감소\n'
                     '연한 주황 배경: 다수 문서와 내재 지식이 충돌하는 핵심 조건 · 검은 선: 문항 단위 짝지은 95% 신뢰구간', fontsize=13)
        fig.savefig(output / f'{metric}.png', dpi=145)
        plt.close(fig)


def write_report(table, runs, formats, audit, output):
    methods = [dict(run=run['path'].name, **{key: run[key] for key in ('label', 'method', 'color', 'budget', 'retain', 'strength')}) for run in runs]
    checkboxes = ''.join(f'<label class="method" data-method-option data-run="{escape(run["path"].name)}"><input type="checkbox" name="method" value="{escape(run["path"].name)}" checked>'
                         f'<span class="swatch" style="background:{run["color"]}"></span>{escape(run["label"])}</label>' for run in runs)
    notes = []
    for run in runs:
        cfg = run['config']
        notes.append(f'<li data-method-detail data-run="{escape(run["path"].name)}">{escape(run["label"])}: {escape(run["path"].name)}; '
                     f'실제 개입 위치(0부터): {escape(json.dumps(cfg["sites"]))}; 유지율 {cfg["retain"]}; '
                     f'MHA 재정규화 {"사용" if cfg.get("attention_renorm", False) else "미사용"}</li>')
    payload = json.dumps(table.to_dict('records'), ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    options = ''.join(f'<option value="{escape(fmt)}">{escape(fmt)}</option>' for fmt in formats)
    ratios = ''.join(f'<option>{k}:{9-k}</option>' for k in range(10))
    budgets = ''.join(f'<option>{escape(budget)}</option>' for budget in dict.fromkeys(run['budget'] for run in runs if run['budget']))
    retentions = ''.join(f'<option value="{retain}">억제 {100*(1-retain):g}% (유지율 {retain:g})</option>'
                         for retain in sorted({run['retain'] for run in runs if run['retain'] is not None}, reverse=True))
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>개입 실험 최종 비교</title>
<style>
body{font:15px/1.65 system-ui,sans-serif;color:#243248;background:#f4f6fa;margin:0}main{max-width:1600px;margin:auto;padding:26px}
h1{font-size:26px;margin:0 0 12px}h2{font-size:19px}section,details{background:white;padding:20px;margin:16px 0;border:1px solid #dde3ec;border-radius:9px}
select{font:inherit;padding:6px;margin:4px 20px 4px 6px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #e1e6ee;white-space:nowrap}
th{background:#eef2f8;position:sticky;top:0}td:first-child,td:nth-child(4),td:last-child{text-align:left}.tablebox{overflow:auto;max-height:650px}
.up{color:#00734f}.down{color:#b02f43}.muted,small{color:#617087}small{display:block;font-size:11px}summary{cursor:pointer;font-weight:650}
.methods{display:flex;flex-wrap:wrap;gap:8px 18px}.method{display:flex;align-items:center;gap:6px;cursor:pointer}.swatch{display:inline-block;width:12px;height:12px;border-radius:2px}
.comparison-controls{background:#f8faff;border:1px solid #dde3ec;border-radius:6px;padding:14px;margin:14px 0}.comparison-controls p{margin:8px 0 0}
button{font:inherit;cursor:pointer;margin:5px 8px 10px 0}#charts,#accuracy-charts,#performance-accuracy-charts{overflow:auto}#charts svg,#accuracy-charts svg,#performance-accuracy-charts svg{display:block;width:100%;min-width:1000px;height:auto}#legend,#accuracy-legend,#performance-accuracy-legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:12px 0}
.notice{border-left:4px solid #d19c39;padding-left:12px}a{color:#157eaa}[hidden]{display:none!important}
.tabs button[aria-selected="true"]{background:#243248;color:white}.tabs button{padding:8px 18px;border:1px solid #dde3ec;border-radius:6px}
.answer{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0;padding:12px;background:#f4f6fa}.review-card{margin:12px 0;padding:14px;border:1px solid #dde3ec;border-radius:6px}
.review-card select{max-width:100%}#review-stats th,#review-stats td{white-space:normal;min-width:75px}#review-stats td:first-child{min-width:220px}
</style><main><h1>개입 실험: 정확도와 다수 문서 기준 충실도</h1>
<p>@@MODEL@@ · fact/counter 각 @@N@@문항 · 문서 타입 @@TYPES@@종 × 혼합 비율 10종 · 동일 문항을 @@METHODS@@개 방법에서 비교</p>
<p><b>주표와 그래프는 기존 베이스라인과 동일한 채점 기준입니다.</b> 완전 일치 채점은 아래 보조 검증에서만 사용합니다.</p>
<p class="muted">@@PENDING@@</p>
<template id="comparison-controls-template">
<b>비교할 방법 선택</b><br>
<label>모듈 구성<select data-filter="budget"><option value="ALL">전체 구성</option>@@BUDGETS@@</select></label>
<label>개입 강도<select data-filter="retention"><option value="ALL">전체 강도</option>@@RETENTIONS@@</select></label><br>
<button data-select-methods="all" type="button">전체 선택</button><button data-select-methods="none" type="button">전체 해제</button>
<div class="methods">@@CHECKBOXES@@</div>
<div data-conditions>
<label>내재 지식<select data-filter="split"><option value="both">fact/counter 각각</option><option value="ALL">전체 평균</option><option value="fact">사실</option><option value="counter">반사실</option></select></label>
<label>문서 타입<select data-filter="format"><option value="ALL">전체 평균</option>@@FORMATS@@</select></label>
<label data-ratio-control>사실:반사실<select data-filter="ratio"><option value="ALL">전체 비율 평균</option><option value="*">모든 비율 + 평균</option>@@RATIOS@@</select></label>
<label data-base-control>전후 비교 기준<select data-filter="accuracy-base"><option value="vanilla">바닐라 — 방법 전체의 추가 효과</option><option value="dummy">더미 — 더미가 있는 상태에서 개입 효과</option></select></label>
</div>
<p class="muted">방법·조건 선택은 다른 표·그래프 및 탭과 동기화됩니다. 구성·강도 필터는 제안 방법에 적용합니다.</p>
</template>
<nav class="tabs" role="tablist" aria-label="분석 탭"><button id="tab-performance" role="tab" aria-selected="true" aria-controls="performance-panel">성능 비교</button>
<button id="tab-accuracy" role="tab" aria-selected="false" aria-controls="accuracy-panel">검색 품질별 정확도</button>
<button id="tab-review" role="tab" aria-selected="false" aria-controls="review-panel">채점 불일치 검수</button></nav>
<div id="performance-panel" role="tabpanel" aria-labelledby="tab-performance">
<section><h2>선택한 방법의 전체 평균</h2>
<div class="comparison-controls" data-mode="methods"></div><div id="conclusions"></div>
<p class="notice">더미 대비 상승은 개입이 더미의 부정적 효과를 줄였다는 근거입니다. <b>최종 성능 가설은 바닐라보다 충실도가 높아지는지도 확인해야 합니다.</b>
이 표만으로 편향이 더미로 이동했다는 내부 작동 원리까지 입증하지는 않습니다.</p></section>
<section><h2>비교표</h2><p>비율과 관계없이 충실도 ↑가 개선입니다. 사실 문서가 다수면 사실 답변, 반사실 문서가 다수면 반사실 답변을 정답으로 채점합니다.
내재 지식은 평가 정답을 바꾸지 않습니다. 정확도는 언제나 사실 정답 기준입니다.</p>
<div class="comparison-controls" data-mode="table"></div>
<div class="tablebox"><table><thead><tr><th>내재 지식</th><th>문서 타입</th><th>비율</th><th>방법</th><th>문항 수</th><th>정확도 %</th><th>충실도 %</th><th id="head-vanilla">충실도 Δ vs 바닐라</th><th id="head-dummy">충실도 Δ vs 더미</th><th id="head-interpretation">해석</th></tr></thead><tbody id="body"></tbody></table></div>
<p class="muted">Δ는 퍼센트포인트(pp). 괄호는 짝지은 95% 신뢰구간입니다. ↑/↓는 관측된 방향이며, 신뢰구간이 0을 포함하면 증감이 확실하다고 판정하지 않습니다.
비율별 판정은 다중 비교 보정이 없는 탐색 결과이고, 주 결론은 전체 평균 비교를 기준으로 합니다.</p></section>
<section><h2>충실도 비교 그래프</h2><p id="chart-note"></p><p class="muted">그래프는 선택한 내재 지식·문서 타입의 모든 비율과 평균을 보여줍니다.
연한 주황 배경은 다수 문서와 내재 지식이 충돌하는 핵심 조건이며, 검은 선은 짝지은 95% 신뢰구간입니다. 막대에 마우스를 올리면 값을 확인할 수 있습니다.</p>
<div class="comparison-controls" data-mode="graph"></div>
<div id="legend"></div><div id="charts"></div></section>
<section><h2>정확도 비교 그래프</h2><p id="performance-accuracy-chart-note"></p>
<p class="muted">정확도는 항상 사실 정답 기준입니다. 위 충실도 그래프와 같은 방법·내재 지식·문서 타입·비교 기준을 사용하며 모든 비율과 평균을 표시합니다.
검은 선은 짝지은 95% 신뢰구간입니다. 정확도 증가·감소가 충실도 증가·감소를 뜻하지는 않습니다. 막대에 마우스를 올리면 값을 확인할 수 있습니다.</p>
<div class="comparison-controls" data-mode="graph"></div>
<div id="performance-accuracy-legend"></div><div id="performance-accuracy-charts"></div></section>
<details><summary>채점 방법 · 가설 판정 · 해석의 한계</summary>
<p>MHA(Multi-Head Attention)는 다중 헤드 어텐션, MLP(Multi-Layer Perceptron)는 다층 퍼셉트론을 뜻합니다.
방법 이름의 층×헤드는 어텐션 개입 범위이며, MLP·잔차 스트림의 실제 개입 위치는 아래 설정에 표시합니다.</p>
<p>억제율=1−유지율이며 선택한 경로에만 적용됩니다. MHA 재정규화 조건에서는 가중치를 먼저 유지율만큼 줄인 뒤 합이 1이 되도록 정규화하므로, 최종 더미 가중치의 감소율은 표시된 억제율과 다릅니다.
유지율 0은 더미 문서 삭제를 뜻하지 않습니다. 방법 선택은 개별 방법의 점수와 평균을 바꾸지 않습니다.</p>
<p><b>기존 채점 함수 is_correct(...)[1]</b>를 그대로 사용합니다. 소문자로 바꾸고 양끝의 공백·쉼표·마침표·느낌표·물음표·물결표를 제거한 뒤,
동일 문자열이거나 정답과 응답 중 하나가 다른 하나에 단어 경계 기준으로 포함되면 정답으로 인정합니다. 이는 완전 일치 채점이 아닙니다.
사실 정확도는 저장된 correct 판정의 평균이며, 현재 채점 함수와 모든 응답의 판정이 같은지도 확인합니다.
다수 문서 기준 충실도는 같은 함수를 원래 검색 문서 다수가 지지하는 정답에 적용한 인정 비율입니다.
더미는 문서 수에서 제외합니다. 이번 실험은 검색 문서 9개로 동률이 없습니다.</p>
<p>이는 동등한 품질·신뢰도의 문서를 가정한 <b>다수 문서 준수의 조작적 지표</b>이며, 일반적인 의미의 모든 문서 충실성을 측정하지는 않습니다.
사실 문서가 다수인 조건에서는 충실도=사실 정확도입니다. 반사실 문서가 다수인 조건에서는 반사실 정답 일치율이며, 다른 답변이 있어 1−정확도로 계산할 수 없습니다.</p>
<p>다수 문서와 내재 지식이 반대인 조건에서의 개선이 핵심입니다. 사실 내재 지식+반사실 다수: 반사실 답변 증가가 기대됩니다.
반사실 내재 지식+사실 다수: 사실 답변 증가가 기대됩니다. 내재 지식과 다수 문서가 일치하는 조건은 성능 유지 여부를 함께 봅니다.
정확도 하락 자체가 충실도 상승은 아닙니다.</p>
<p>제안 방법의 충실도가 더미·바닐라보다 모두 높고, 두 차이의 95% 신뢰구간 하한이 0보다 클 때만 ‘성능 가설 지지’로 표시합니다.
더미보다 좋아도 바닐라보다 낮으면 ‘회복했지만 바닐라 미달’입니다. 이는 성능 가설 판정이며, 편향 이동의 인과적 증명이나 논문 채택 판정이 아닙니다.</p>
<p>평균은 문서 타입과 비율에 동일한 가중치를 줍니다. 전체 내재 지식 평균은 fact/counter를 50:50으로 평균합니다.
95% 신뢰구간은 내재 지식 분할 안에서 문항을 @@BOOT@@회 복원 추출하며, 같은 문항의 모든 타입·비율·방법을 함께 이동시킵니다(시드 @@SEED@@).
따라서 5만 답변을 5만 개의 독립 문항으로 간주하지 않습니다. 이 신뢰구간은 저장된 답변과 선정 모듈을 고정한 조건부 구간이며,
후보를 다시 선정하거나 생성 시드를 바꾸는 변동까지 포함하지 않습니다.</p>
<p><b>사실 정확도는 기존 accuracy.csv와 동일한 기준입니다.</b> 비교의 일관성을 위해 기존의 관대한 채점도 그대로 유지합니다.
이 규칙은 양쪽 정답을 함께 언급해도 각각 정답으로 인정하거나, 부분 답변·빈 답변을 잘못 인정할 수 있습니다.
따라서 의미적으로 올바른 답변을 엄격하게 구분한 점수라고 표현하지 않습니다. 아래 보조 검증으로 주 결론이 채점 기준에 민감한지 확인합니다.</p>
<p>모델·설정 한 종류의 개선만으로 일반성을 입증할 수 없습니다. 개입 후보 선정과 평가 문항이 겹친 설정이므로 독립 평가로 표현해서는 안 되며,
라벨 없는 평가 집합 적응 과정임을 명시해야 합니다. 최종 응답 성능만으로 ‘편향 분산’ 메커니즘을 확정할 수 없습니다.</p>
<ul>@@SETTINGS@@</ul></details>
<details><summary>보조 검증: 엄격한 채점에서도 결론이 유지되는가?</summary>
<p>정규화 완전 일치(Exact Match, EM)는 유니코드 NFKC, 대소문자 통일, 구두점·밑줄의 공백 치환, 연속 공백 정리 후 정답 전체와 비교합니다.
빈 답변·양쪽 답변 동시 제시·다른 답변은 인정하지 않습니다. 뜻이 같은 별칭이나 설명문도 0점이 될 수 있는 엄격한 보조 지표입니다.</p>
<p>단방향 포함 검사는 같은 정규화 후 고유한 정답 전체가 응답에 포함되는지를 봅니다. 정확 일치를 우선하며 두 정답 동시 포함은 0점입니다.
이 검사도 부정문 속 정답 언급을 오판할 수 있습니다. 보조 검증의 절대 수치는 기존 베이스라인 표에 섞지 않습니다.</p>
<div class="comparison-controls" data-mode="methods"></div>
<div class="tablebox"><table><thead><tr><th>내재 지식</th><th>방법</th><th>기존 채점 충실도 %</th><th>완전 일치 충실도 %</th><th>단방향 포함 충실도 %</th></tr></thead><tbody id="sensitivity"></tbody></table></div>
<p>기존 채점 점검(분할·방법별): @@AUDIT@@</p></details>
</div>
<div id="accuracy-panel" role="tabpanel" aria-labelledby="tab-accuracy" hidden>
<section><h2>원래 예상 양상 확인: 검색 품질과 사실 정확도</h2>
<p>검색 품질의 대리값은 <b>사실 문서 수 ÷ 원래 검색 문서 9개</b>입니다. 더미는 제외합니다.
저품질은 0:9~4:5, 고품질은 5:4~9:0으로 나누고 각 구간의 다섯 비율을 동일하게 평균합니다.
정확도 A는 기존 채점으로 사실 정답을 맞힌 비율이며, 변화량은 <b>ΔA = 비교 방법의 정확도 − 기준 방법의 정확도</b>입니다.</p>
<p>원래 예상은 내재 지식이 사실이면 <b>저품질에서 크게 감소·고품질에서 유지 또는 소폭 감소</b>,
반사실이면 <b>저품질에서 크게 증가·고품질에서 유지 또는 소폭 증가</b>입니다.
‘크게·소폭’의 기준은 지정되지 않았으므로 임의의 가설 합격 점수나 별도 충실도 점수를 만들지 않습니다.</p>
<p class="notice">반사실 내재 지식 + 저품질에서 사실 정확도가 증가하는 것은 반사실 내재 답변과 반대 방향의 변화입니다.
그러나 이때 다수 문서도 반사실이므로, 이 증가를 문서 충실도 향상이라고 해석할 수는 없습니다.
사실 정확도 감소 역시 반사실 답변 대신 다른 오답이 늘었을 수 있어, 그 자체로 문서 충실도 증가를 입증하지 않습니다.</p>
<div class="comparison-controls" data-mode="table" data-accuracy-base></div>
<p id="accuracy-note"></p>
<div class="tablebox"><table><thead><tr><th>내재 지식</th><th>검색 품질 / 비율</th><th>비교 방법</th><th>기준 정확도 %</th><th>비교 정확도 %</th><th>정확도 Δ / 95% 신뢰구간</th><th>원래 예상</th></tr></thead><tbody id="accuracy-body"></tbody></table></div>
<p class="muted">이 표 바로 위에서 비율을 선택하세요. 전체 평균 또는 모든 비율이면 저품질·고품질·전체 평균을 표시하고, 특정 비율이면 해당 비율만 표시합니다.
↑/↓는 정확도의 증감이며 충실도 개선/악화 판정이 아닙니다. 신뢰구간이 0을 포함하는 것은 유지가 입증됐다는 뜻이 아닙니다.
구간 신뢰구간은 비율별 구간을 평균하지 않고, 같은 문항의 응답을 묶어 다시 계산합니다.</p></section>
<section><h2>검색 품질별 정확도 곡선과 전후 변화</h2>
<p>왼쪽 x축은 사실:반사실 문서 비율이며 오른쪽으로 갈수록 사실 문서 비율이 높습니다. y축은 사실 정확도(%)입니다.
오른쪽 y축은 선택한 기준 대비 정확도 변화량(%p)입니다. 검은 선은 문항 단위 짝지은 95% 신뢰구간입니다.
두 그래프는 모든 비율을 표시합니다. 연한 주황 배경은 다수 문서와 내재 지식이 충돌하는 조건입니다.
정확도가 검색 품질을 따른다는 가정이 정확도와 사실 문서 비율이 숫자로 같아야 한다는 뜻은 아닙니다.</p>
<div class="comparison-controls" data-mode="graph" data-accuracy-base></div>
<div id="accuracy-legend"></div><div id="accuracy-charts"></div></section></div>
<div id="review-panel" role="tabpanel" aria-labelledby="tab-review" hidden>
<section><h2>채점 불일치의 자동 분류</h2>
<p><b>유형별 통계는 선택한 완료 실행의 전체 응답에서 계산합니다.</b> 아래 유형은 문자열 규칙의 차이이며, 실제 의미상 오류가 확정된 것은 아닙니다.
예를 들어 “정답은 서울이 아니다”도 서울을 포함합니다. 이런 답변의 입장·부정·별칭은 사람이 확인해야 합니다. 검수는 기존 성능 점수를 바꾸지 않습니다.</p>
<div class="comparison-controls" data-mode="table"></div>
<p class="muted">검수에서 전체 평균은 해당 조건의 모든 실제 응답을 뜻합니다.</p>
<label>평가 지표<select id="review-metric"><option value="1">다수 문서 기준 충실도</option><option value="0">사실 정확도</option></select></label>
<label>비교할 채점<select id="review-comparison"><option value="2">기존 ↔ 완전 일치(EM)</option><option value="4">기존 ↔ 단일 후보 포함</option></select></label>
<p id="review-total"></p><div class="tablebox"><table id="review-stats"><thead><tr><th>방법</th><th>전체 응답 수</th><th>불일치 수 / 비율</th>
<th>기존만 인정</th><th>보조만 인정</th>@@KIND_HEADERS@@</tr></thead><tbody id="review-stats-body"></tbody></table></div>
<p class="muted">비율의 분모는 각 방법의 전체 선택 응답입니다. 유형은 빈 응답 → 양쪽 인정 → 정규화 일치 → 단일 후보 포함 → 역방향 일치 → 기타 순서로 하나만 부여합니다.
정확도와 충실도의 불일치 수는 같은 응답을 포함할 수 있으므로 합산하지 않습니다.</p></section>
<section><h2>실제 답변 검수</h2>
<p class="muted">목록은 방법·내재 지식·문서 타입·비율·자동 유형·불일치 조합마다 최대 @@REVIEW_LIMIT@@건을 고른 재현 가능한 표본입니다.
전체 통계는 위 표를 사용하세요. 이 표본의 수동 판정 비율을 전체 오류율로 해석할 수 없습니다.</p>
<div class="comparison-controls" data-mode="table"></div>
<label>자동 유형(아래 표본 목록만 필터)<select id="review-kind"><option value="ALL">전체 유형</option>@@KIND_OPTIONS@@</select></label>
<label><input id="review-blind" type="checkbox" checked> 답변 카드의 방법명 가리기</label>
<button id="review-prev" type="button">이전</button><button id="review-next" type="button">다음</button><span id="review-page"></span>
<p id="review-status" class="muted"></p><button id="review-export" type="button">수동 검수 기록 저장(JSON)</button>
<div id="review-cases"></div>
<p class="muted">수동 판정은 가능한 경우 이 브라우저에 보관합니다. 브라우저가 저장을 막거나 보고서를 다른 경로로 옮기면 유지되지 않을 수 있으므로 JSON으로 저장할 수 있습니다.
방법명 가리기는 카드의 이름을 숨기는 보조 기능이며, 선택한 방법이 하나뿐이면 완전한 익명 검수는 아닙니다.</p></section></div>
<p><a href="summary.csv">모든 방법의 전체 비교표 CSV (LOW/HIGH 구간 평균 포함)</a> · <a href="faithfulness.png">모든 방법·문서 타입 평균 충실도 그래프 PNG</a> · <a href="accuracy.png">모든 방법·문서 타입 평균 정확도 그래프 PNG</a></p>
<script>
const rows=@@DATA@@;
const methods=@@METHOD_DATA@@;
const filters={budget:'ALL',retention:'ALL',split:'both',format:'ALL',ratio:'ALL','accuracy-base':'vanilla'};
const selectedRuns=new Set(methods.map(m=>m.run));
document.querySelectorAll('.comparison-controls').forEach(host=>{
 host.append(document.getElementById('comparison-controls-template').content.cloneNode(true));
 host.querySelector('[data-conditions]').hidden=host.dataset.mode==='methods';
 host.querySelector('[data-ratio-control]').hidden=host.dataset.mode!=='table';
 host.querySelector('[data-base-control]').hidden=!host.hasAttribute('data-accuracy-base');
});
const reviewCounts=@@REVIEW_COUNTS@@,reviewCases=@@REVIEW_CASES@@,reviewKinds=@@REVIEW_KINDS@@;
const reviewLabels={'':'미검수',fact:'사실 답변을 지지',counter:'반사실 답변을 지지',both:'양쪽 제시·입장 불명확',other:'다른 답변·응답 없음',uncertain:'의미 판단 보류'};
const reviewKey=@@REVIEW_KEY@@;
let reviewLabelsSaved={},reviewStorage=true,reviewPage=0;
try{reviewLabelsSaved=JSON.parse(localStorage.getItem(reviewKey)||'{}')||{};}catch(e){reviewStorage=false;}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const splitNames={fact:'사실',counter:'반사실',ALL:'전체 평균'};
const ratios=Array.from({length:10},(_,i)=>`${i}:${9-i}`).concat('ALL');
const index=new Map(rows.map(r=>[JSON.stringify([r.run,r.split,r.format,r.ratio]),r]));
const getRow=(run,split,format,ratio)=>index.get(JSON.stringify([run,split,format,ratio]));
function delta(row,base,metric='faithfulness'){
 if(row['change_vs_'+base]==='기준')return '—';
 const value=row['delta_'+metric+'_vs_'+base+'_pp'],lo=row[metric+'_vs_'+base+'_ci_low'],hi=row[metric+'_vs_'+base+'_ci_high'];
 const sign=value>1e-9?'↑':value< -1e-9?'↓':'→';
 const change=row[(metric==='accuracy'?'accuracy_':'')+'change_vs_'+base];
 return `<span class="${value>0?'up':value<0?'down':''}">${sign} ${value>0?'+':''}${value.toFixed(2)} pp</span><small>[${lo.toFixed(2)}, ${hi.toFixed(2)}] · ${esc(change)}</small>`;
}
function plotPanel(selected,split,format,mode,base,left,top,metric='faithfulness',curve=metric==='accuracy'){
 const isDelta=mode==='delta',series=isDelta?selected.filter(m=>m.run!==base?.run):selected;
 const isAccuracy=metric==='accuracy',isLine=curve&&!isDelta,measure=isAccuracy?'사실 정확도':'충실도',plotRatios=isLine?ratios.slice(0,10):ratios;
 const title=`내재 지식: ${splitNames[split]} · ${isDelta?(base?base.label+' 대비 '+measure+' 변화':measure+' 변화'):'선택한 방법의 '+measure}`;
 let svg=`<g transform="translate(${left},${top})"><text x="395" y="22" text-anchor="middle" font-size="16" font-weight="600">${esc(title)}</text>`;
 if(isDelta&&(!base||!series.length))return svg+`<text x="395" y="175" text-anchor="middle" fill="#617087">${!base?'차이를 보려면 기준 방법(바닐라 또는 더미)의 체크박스를 켜세요.':'비교할 방법을 하나 이상 더 선택하세요.'}</text></g>`;
 const key=isDelta?'delta_'+metric+'_vs_'+base.method+'_pp':metric+'_pct';
 const lowKey=base?metric+'_vs_'+base.method+'_ci_low':'',highKey=base?metric+'_vs_'+base.method+'_ci_high':'';
 const data=series.map(m=>({method:m,cells:plotRatios.map(r=>getRow(m.run,split,format,r))}));
 const bounds=isDelta?[0,...data.flatMap(s=>s.cells.flatMap(r=>[r[key],r[lowKey],r[highKey]]))]:[0,100];
 let low=isDelta?Math.floor(Math.min(...bounds)/5)*5:0,high=isDelta?Math.ceil(Math.max(...bounds)/5)*5:100;
 if(high===low)high=low+5;
 if(isDelta){const pad=(high-low)*.06;low-=pad;high+=pad;}
 const x0=60,y0=45,width=710,height=240,slot=width/plotRatios.length,bar=slot*.8/series.length;
 const y=value=>y0+height-(value-low)/(high-low)*height;
 if(!isLine)svg+=`<rect x="${x0+10*slot}" y="${y0}" width="${slot}" height="${height}" fill="#eef0f4"/>`;
 if(split!=='ALL')svg+=`<rect x="${x0+(split==='fact'?0:5)*slot}" y="${y0}" width="${5*slot}" height="${height}" fill="#fff7e9"/>`;
 for(let tick=0;tick<=4;tick++){
  const value=low+(high-low)*tick/4,pos=y(value);
  svg+=`<line x1="${x0}" x2="${x0+width}" y1="${pos}" y2="${pos}" stroke="#e1e6ee"/><text x="${x0-8}" y="${pos+4}" text-anchor="end" font-size="12">${value.toFixed(isDelta?1:0)}</text>`;
 }
 if(isLine)for(const s of data)svg+=`<polyline points="${s.cells.map((r,i)=>`${x0+(i+.5)*slot},${y(r[key])}`).join(' ')}" fill="none" stroke="${s.method.color}" stroke-width="2"/>`;
 for(let ri=0;ri<plotRatios.length;ri++){
  const center=x0+(ri+.5)*slot;
  svg+=`<text x="${center}" y="${y0+height+23}" text-anchor="middle" font-size="12">${plotRatios[ri]==='ALL'?'평균':plotRatios[ri]}</text>`;
  data.forEach((s,mi)=>{
   const row=s.cells[ri],value=row[key],x=center+(mi-series.length/2)*bar;
   let tip=`${s.method.label} | ${plotRatios[ri]} | ${value.toFixed(2)} ${isDelta?'pp':'%'}`;
   if(isDelta)tip+=` | 95% CI [${row[lowKey].toFixed(2)}, ${row[highKey].toFixed(2)}]`;
   svg+=isLine?`<circle data-run="${esc(s.method.run)}" data-value="${value}" cx="${center}" cy="${y(value)}" r="3" fill="${s.method.color}"><title>${esc(tip)}</title></circle>`
    :`<rect data-run="${esc(s.method.run)}" data-value="${value}" x="${x}" y="${Math.min(y(value),y(0))}" width="${bar*.92}" height="${Math.abs(y(value)-y(0))}" fill="${s.method.color}"><title>${esc(tip)}</title></rect>`;
   if(isDelta)svg+=`<line x1="${x+bar*.46}" x2="${x+bar*.46}" y1="${y(row[lowKey])}" y2="${y(row[highKey])}" stroke="#293448" stroke-width="1"/>`;
  });
 }
 svg+=`<line x1="${x0}" x2="${x0+width}" y1="${y(0)}" y2="${y(0)}" stroke="#293448"/>`;
 svg+=`<text x="${x0+width/2}" y="${y0+height+49}" text-anchor="middle" font-size="13">사실:반사실 문서 비율 (더미 제외)</text>`;
 return svg+`<text transform="translate(16,${y0+height/2}) rotate(-90)" text-anchor="middle" font-size="13">${isDelta?measure+' 차이 (pp)':isAccuracy?'사실 정확도 (%)':'다수 문서 기준 충실도 (%)'}</text></g>`;
}
function renderCharts(selected,split,format,metric='faithfulness'){
 const base=selected.find(m=>m.method==='vanilla')||selected.find(m=>m.method==='dummy');
 const prefix=metric==='accuracy'?'performance-accuracy-':'',measure=metric==='accuracy'?'사실 정확도':'충실도';
 const chart=document.getElementById(prefix+'charts');
 document.getElementById(prefix+'chart-note').textContent=`왼쪽: 선택한 방법의 ${measure}(%). `+(base?'오른쪽: 선택된 '+base.label+' 대비 변화(pp).':'변화량 비교 기준이 선택되지 않았습니다.');
 document.getElementById(prefix+'legend').innerHTML=selected.map(m=>`<span><i class="swatch" style="background:${m.color}"></i> ${esc(m.label)}</span>`).join('');
 if(!selected.length){chart.innerHTML='<p>비교할 방법을 선택하세요.</p>';return;}
 const splits=split==='both'?['fact','counter']:[split];
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 ${splits.length*365}" role="img" aria-label="선택한 방법의 ${measure} 비교" style="font-family:system-ui,sans-serif;fill:#243248">`;
 splits.forEach((s,i)=>{svg+=plotPanel(selected,s,format,'absolute',base,0,i*365,metric,false)+plotPanel(selected,s,format,'delta',base,800,i*365,metric,false);});
 chart.innerHTML=svg+'</svg>';
}
function accuracyExpectation(split,ratio){
 if(split==='ALL'||ratio==='ALL')return '양쪽 내재 지식·품질 구간을 나누어 확인';
 const low=ratio==='LOW'||(ratio!=='HIGH'&&Number(ratio.split(':')[0])<5);
 return split==='fact'?(low?'저품질: 크게 감소':'고품질: 유지 또는 소폭 감소'):(low?'저품질: 크게 증가':'고품질: 유지 또는 소폭 증가');
}
function renderAccuracy(selected,split,format,ratio){
 const base=selected.find(m=>m.method===filters['accuracy-base']);
 const splits=split==='both'?['fact','counter']:[split],groups=(ratio==='ALL'||ratio==='*')?['LOW','HIGH','ALL']:[ratio];
 const labels={LOW:'저품질 0:9~4:5',HIGH:'고품질 5:4~9:0',ALL:'전체 비율 평균'};
 document.getElementById('accuracy-note').textContent=base?`비교 기준: ${base.label}. 문서 타입: ${format==='ALL'?'전체 평균':format}. 아래 예상은 원래 가설의 방향을 제시하며, 충실도 증가 판정은 아닙니다.`:'전후 비교를 보려면 위에서 지정한 기준 방법의 체크박스를 켜세요.';
 let body='';
 if(base)for(const s of splits)for(const group of groups)for(const method of selected){
  const row=getRow(method.run,s,format,group),before=getRow(base.run,s,format,group);
  const expectation=method.run===base.run?'비교 기준':method.method==='ours'?accuracyExpectation(s,group):'제안 방법 예상의 적용 대상 아님';
  body+=`<tr><td>${esc(splitNames[s])}</td><td>${esc(labels[group]||group)}</td><td>${esc(method.label)}</td><td>${before.accuracy_pct.toFixed(2)}</td><td>${row.accuracy_pct.toFixed(2)}</td><td>${delta(row,base.method,'accuracy')}</td><td>${esc(expectation)}</td></tr>`;
 }
 document.getElementById('accuracy-body').innerHTML=body||'<tr><td colspan="7">비교할 방법과 기준 방법을 선택하세요.</td></tr>';
 document.getElementById('accuracy-legend').innerHTML=document.getElementById('legend').innerHTML;
 if(!selected.length){document.getElementById('accuracy-charts').innerHTML='<p>비교할 방법을 선택하세요.</p>';return;}
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 ${splits.length*365}" role="img" aria-label="검색 품질별 사실 정확도와 변화량" style="font-family:system-ui,sans-serif;fill:#243248">`;
 splits.forEach((s,i)=>{svg+=plotPanel(selected,s,format,'absolute',base,0,i*365,'accuracy')+plotPanel(selected,s,format,'delta',base,800,i*365,'accuracy');});
 document.getElementById('accuracy-charts').innerHTML=svg+'</svg>';
}
function inScope(method){
 const budget=filters.budget,retain=filters.retention;
 return method.method!=='ours'||((budget==='ALL'||method.budget===budget)&&(retain==='ALL'||method.retain===Number(retain)));
}
function chosenMethods(){
 return methods.filter(m=>selectedRuns.has(m.run)&&inScope(m));
}
function reviewCondition(row){
 const {split,format,ratio}=filters;
 return ((split==='both'||split==='ALL')||row.split===split)&&(format==='ALL'||row.format===format)&&((ratio==='ALL'||ratio==='*')||row.ratio===ratio);
}
function reviewStatus(){
 const counts={};
 for(const row of reviewCases){const label=reviewLabelsSaved[row.id];if(label&&reviewLabels[label])counts[label]=(counts[label]||0)+1;}
 document.getElementById('review-status').textContent='이 보고서 표본의 수동 판정: '+(Object.entries(counts).map(([k,n])=>reviewLabels[k]+' '+n+'건').join(' · ')||'아직 없음')
  +(reviewStorage?' · 브라우저 보관':' · 브라우저 저장 불가: JSON으로 기록을 저장하세요.');
}
function renderReview(){
 const selected=chosenMethods(),ids=new Set(selected.map(m=>m.run)),metric=Number(document.getElementById('review-metric').value),comparison=Number(document.getElementById('review-comparison').value);
 const position=comparison-2+metric,kind=document.getElementById('review-kind').value,blind=document.getElementById('review-blind').checked;
 const counts=new Map(selected.map(m=>[m.run,{total:0,legacy:0,kinds:Object.fromEntries(Object.keys(reviewKinds).map(k=>[k,0]))}]));
 for(const row of reviewCounts){
  if(!ids.has(row.run)||!reviewCondition(row))continue;
  const current=counts.get(row.run);current.total+=row.counts[position];current.legacy+=row.counts[position+4];current.kinds[row.kind]+=row.counts[position];
 }
 const {split,format,ratio}=filters;
 let total=0,disagree=0;
 document.getElementById('review-stats-body').innerHTML=selected.map(m=>{
  const count=counts.get(m.run),n=getRow(m.run,split==='both'?'ALL':split,format,ratio==='*'?'ALL':ratio).n_answers;
  total+=n;disagree+=count.total;
  return `<tr><td>${esc(m.label)}</td><td>${n.toLocaleString()}</td><td>${count.total.toLocaleString()} / ${(100*count.total/n).toFixed(2)}%</td><td>${count.legacy.toLocaleString()}</td><td>${(count.total-count.legacy).toLocaleString()}</td>${Object.keys(reviewKinds).map(k=>'<td>'+count.kinds[k].toLocaleString()+'</td>').join('')}</tr>`;
 }).join('')||'<tr><td colspan="11">방법을 선택하세요.</td></tr>';
 document.getElementById('review-total').textContent=`선택한 방법의 전체 응답 ${total.toLocaleString()}건 중 채점 불일치 ${disagree.toLocaleString()}건. 같은 문항도 방법·타입·비율마다 별도 응답으로 셉니다.`;
 const cases=reviewCases.filter(c=>ids.has(c.run)&&reviewCondition(c)&&(kind==='ALL'||c.kind===kind)&&c.scores[metric]!==c.scores[metric+comparison]);
 const pageSize=12,pages=Math.max(1,Math.ceil(cases.length/pageSize));reviewPage=Math.min(reviewPage,pages-1);
 document.getElementById('review-page').textContent=` ${reviewPage+1} / ${pages}쪽 · 표시 대상 표본 ${cases.length.toLocaleString()}건`;
 document.getElementById('review-prev').disabled=reviewPage===0;document.getElementById('review-next').disabled=reviewPage===pages-1;
 document.getElementById('review-cases').innerHTML=cases.slice(reviewPage*pageSize,(reviewPage+1)*pageSize).map((c,i)=>{
  const target=metric===0||Number(c.ratio.split(':')[0])>=5?'fact':'counter',method=methods.find(m=>m.run===c.run);
  return `<article class="review-card"><b>${blind?'검수 응답 '+(reviewPage*pageSize+i+1):esc(method.label)}</b> · ${esc(splitNames[c.split])} / ${esc(c.format)} / ${esc(c.ratio)}<br>
  <small>문항 ID: ${esc(c.case_id)} · 자동 유형: ${esc(reviewKinds[c.kind])}</small>
  <p><b>질문:</b> ${esc(c.question)}<br><b>사실 정답:</b> ${esc(c.fact)} · <b>반사실 정답:</b> ${esc(c.counter)}<br><b>현재 채점 대상:</b> ${target==='fact'?'사실':'반사실'} 정답</p>
  <div class="answer">${esc(c.response)||'(빈 응답)'}</div><p>기존 채점: <b>${c.scores[metric]?'인정':'불인정'}</b> / ${comparison===2?'완전 일치':'단일 후보 포함'}: <b>${c.scores[metric+comparison]?'인정':'불인정'}</b></p>
  <label>답변의 실제 입장 <select data-review-id="${c.id}">${Object.entries(reviewLabels).map(([value,label])=>`<option value="${value}"${reviewLabelsSaved[c.id]===value?' selected':''}>${label}</option>`).join('')}</select></label></article>`;
 }).join('')||'<p>선택 조건에 해당하는 불일치 표본이 없습니다.</p>';
 reviewStatus();
}
function render(){
 const {split,format:fmt,ratio}=filters;
 document.querySelectorAll('[data-filter]').forEach(el=>el.value=filters[el.dataset.filter]);
 document.querySelectorAll('input[name="method"]').forEach(el=>el.checked=selectedRuns.has(el.value));
 const selectedMethods=chosenMethods(),ids=new Set(selectedMethods.map(m=>m.run)),showV=selectedMethods.some(m=>m.method==='vanilla'),showD=selectedMethods.some(m=>m.method==='dummy');
 document.querySelectorAll('[data-method-option]').forEach(el=>el.hidden=!inScope(methods.find(m=>m.run===el.dataset.run)));
 document.getElementById('head-vanilla').hidden=!showV;document.getElementById('head-dummy').hidden=!showD;document.getElementById('head-interpretation').hidden=!(showV&&showD);
 const selected=rows.filter(r=>ids.has(r.run)&&(split==='both'?r.split!=='ALL':r.split===split)&&r.format===fmt&&(ratio==='*'?ratios.includes(r.ratio):r.ratio===ratio));
 document.getElementById('body').innerHTML=selected.map(r=>`<tr><td>${esc(splitNames[r.split])}</td><td>${esc(r.format==='ALL'?'전체 평균':r.format)}</td><td title="${esc(r.condition)}">${esc(r.ratio==='ALL'?'전체 평균':r.ratio)}</td><td>${esc(r.method)}</td><td>${r.n_questions}</td><td>${r.accuracy_pct.toFixed(2)}</td><td><b>${r.faithfulness_pct.toFixed(2)}</b></td>${showV?'<td>'+delta(r,'vanilla')+'</td>':''}${showD?'<td>'+delta(r,'dummy')+'</td>':''}${showV&&showD?'<td>'+esc(r.interpretation)+'</td>':''}</tr>`).join('')||`<tr><td colspan="10">비교할 방법을 선택하세요.</td></tr>`;
 const overall=rows.filter(r=>ids.has(r.run)&&r.split==='ALL'&&r.format==='ALL'&&r.ratio==='ALL');
 document.getElementById('conclusions').innerHTML=overall.map(r=>`<p><b>${esc(r.method)}</b>: 정확도 ${r.accuracy_pct.toFixed(2)}%, 충실도 ${r.faithfulness_pct.toFixed(2)}%${showV?' · 바닐라 대비 '+delta(r,'vanilla'):''}${showD?' · 더미 대비 '+delta(r,'dummy'):''}${showV&&showD?' · '+esc(r.interpretation):''}</p>`).join('')||'<p>비교할 방법을 선택하세요.</p>';
 document.getElementById('sensitivity').innerHTML=rows.filter(r=>ids.has(r.run)&&r.split!=='ALL'&&r.format==='ALL'&&r.ratio==='ALL').map(r=>`<tr><td>${esc(splitNames[r.split])}</td><td>${esc(r.method)}</td><td>${r.faithfulness_pct.toFixed(2)}</td><td>${r.exact_faithfulness_pct.toFixed(2)}</td><td>${r.containment_faithfulness_pct.toFixed(2)}</td></tr>`).join('');
 document.querySelectorAll('[data-method-detail]').forEach(el=>el.hidden=!ids.has(el.dataset.run));
 renderCharts(selectedMethods,split,fmt);
 renderCharts(selectedMethods,split,fmt,'accuracy');
 renderAccuracy(selectedMethods,split,fmt,ratio);
 renderReview();
}
document.querySelectorAll('select,input[name="method"]').forEach(el=>el.addEventListener('change',()=>{
 if(el.dataset.filter)filters[el.dataset.filter]=el.value;
 if(el.name==='method'){if(el.checked)selectedRuns.add(el.value);else selectedRuns.delete(el.value);}
 reviewPage=0;render();
}));
document.querySelectorAll('[data-select-methods]').forEach(button=>button.addEventListener('click',()=>{
 selectedRuns.clear();if(button.dataset.selectMethods==='all')methods.forEach(m=>selectedRuns.add(m.run));
 reviewPage=0;render();
}));
for(const tab of ['performance','accuracy','review'])document.getElementById('tab-'+tab).addEventListener('click',()=>{
 for(const name of ['performance','accuracy','review']){document.getElementById(name+'-panel').hidden=name!==tab;document.getElementById('tab-'+name).setAttribute('aria-selected',String(name===tab));}
});
document.getElementById('review-blind').addEventListener('change',renderReview);
document.getElementById('review-prev').addEventListener('click',()=>{reviewPage=Math.max(0,reviewPage-1);renderReview();});
document.getElementById('review-next').addEventListener('click',()=>{reviewPage++;renderReview();});
document.getElementById('review-cases').addEventListener('change',event=>{
 const id=event.target.dataset.reviewId;if(!id)return;
 if(event.target.value)reviewLabelsSaved[id]=event.target.value;else delete reviewLabelsSaved[id];
 try{localStorage.setItem(reviewKey,JSON.stringify(reviewLabelsSaved));}catch(e){reviewStorage=false;}
 reviewStatus();
});
document.getElementById('review-export').addEventListener('click',()=>{
 const cases=reviewCases.filter(c=>reviewLabelsSaved[c.id]&&reviewLabels[reviewLabelsSaved[c.id]]).map(c=>({...c,judgment:reviewLabelsSaved[c.id]}));
 const url=URL.createObjectURL(new Blob([JSON.stringify({scope:'유형별 검수 표본: 전체 오류율 추정용이 아님',cases},null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='scoring_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></main></html>'''
    values = {'MODEL': escape(runs[0]['config']['model']), 'N': str(runs[0]['config']['n']),
              'CHECKBOXES': checkboxes, 'METHOD_DATA': json.dumps(methods, ensure_ascii=False).replace('</', '<\\/'),
              'FORMATS': options, 'RATIOS': ratios, 'SETTINGS': ''.join(notes),
              'BUDGETS': budgets, 'RETENTIONS': retentions,
              'PENDING': '아직 완료되지 않아 비교에서 제외: ' + ' · '.join(f'{escape(row["run"])} ({row["completed"]}/{row["total"]}문항)'
                                                                      for row in audit['pending']) if audit['pending'] else '',
              'AUDIT': ''.join(f'<span data-method-detail data-run="{escape(row["run"])}">{escape(row["split"])} / {escape(row["method"])}: '
                                  f'기존 채점 양쪽 정답 인정 {row["both_answers"]}건, 기존 사실 정확도 {row["legacy_accuracy"]:.2f}%<br></span>'
                                  for row in audit['summary']), 'DATA': payload,
              'REVIEW_COUNTS': json.dumps(audit['counts'], ensure_ascii=False).replace('</', '<\\/'),
              'REVIEW_CASES': json.dumps(audit['cases'], ensure_ascii=False).replace('</', '<\\/'),
              'REVIEW_KINDS': json.dumps(REVIEW_KINDS, ensure_ascii=False),
              'REVIEW_KEY': json.dumps(f'bait-scoring-review:{runs[0]["config"]["model"]}:n{runs[0]["config"]["n"]}'),
              'REVIEW_LIMIT': str(REVIEW_PER_GROUP),
              'KIND_HEADERS': ''.join(f'<th>{escape(label)}</th>' for label in REVIEW_KINDS.values()),
              'KIND_OPTIONS': ''.join(f'<option value="{key}">{escape(label)}</option>' for key, label in REVIEW_KINDS.items()),
              'TYPES': str(len(formats)), 'METHODS': str(len(runs)),
              'BOOT': str(BOOTSTRAPS), 'SEED': str(SEED)}
    for key, value in values.items():
        html = html.replace(f'@@{key}@@', value)
    (output / 'summary.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=1000, help='각 내재 지식 분할의 문항 수')
    parser.add_argument('--runs', nargs='+', help='비교할 결과 폴더 이름. 생략하면 해당 모델·n의 결과를 자동 선택')
    parser.add_argument('--output-dir', type=Path, help='기본: outputs/main_experiments/MODEL/analysis')
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n must be positive.')
    root = ROOT / 'outputs/main_experiments' / args.model
    runs, formats, cubes, audit = load_results(root, args.n, args.runs)
    print(f'Validated {len(runs)} methods, {2*args.n} paired questions each.', flush=True)
    table = estimates(runs, formats, cubes)
    output = args.output_dir or root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'summary.csv', index=False, float_format='%.6f')
    charts(table, runs, formats, output)
    write_report(table, runs, formats, audit, output)
    overall = table[(table['format'] == 'ALL') & (table['ratio'] == 'ALL')]
    print(overall[['split', 'method', 'accuracy_pct', 'faithfulness_pct', 'delta_faithfulness_vs_vanilla_pp',
                   'delta_faithfulness_vs_dummy_pp', 'interpretation']].to_string(index=False))
    print(f'Report: {output / "summary.html"}')


if __name__ == '__main__':
    main()
