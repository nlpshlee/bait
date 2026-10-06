'''
    측정 기록 -> 확증 편향 표. 신뢰구간은 질문 단위 부트스트랩 95%
    (같은 질문의 여러 문서 집합은 서로 상관되므로 질문을 단위로 복원 추출한다. 그룹끼리는 독립)

    문항 그룹과 믿는 편 (history.md 2.1)
        fact            : 사실을 믿는다. 반사실 문서 = 원 데이터 반사실 답
        counter         : 반사실을 믿는다. 일부(dataset)는 원 데이터 반사실 답, 나머지는 반사실 답 = 모델 자기 zero-shot 답
        other           : 반사실을 믿는다. 반사실 답 = 모델 자기 zero-shot 답 (중립 분할이 아니다)
        counter_dataset : counter 중 원 zero-shot 답이 원 데이터 반사실 답이었던 문항 (fact 와 문서 구성이 같다)
        중립 기준선이 없으므로 '사실을 믿을 때' 와 '반사실을 믿을 때' 의 G 차이로 β 를 잰다 (선택지 (a))
        E[G | 사실을 믿음] = Δ_doc + β,  E[G | 반사실을 믿음] = Δ_doc - β  (Δ_doc 이 그룹 간 같다는 전제)

    [β 분해] 조건 A, 셀 = 문서 비율('3:6' 등) 또는 문서 형식('Wikipedia' 등) + 전체('all')
        beta         = (G_fact - G_counter) / 2            주 지표
        beta_own     = (G_fact - G_other) / 2              재현 : 반사실 답이 모델 자기 답인 문항
        beta_dataset = (G_fact - G_counter_dataset) / 2    강건성 : 반사실 문서가 fact 분할과 같은 방식으로 만들어진 문항
        delta_doc    = (G_fact + G_counter) / 2            문서 자체 설득력 차이 (β 가 양방향에서 같다는 전제)
        판정 : beta 가 유의하게 + 이면 확증 편향, - 이면 반확증 (신뢰구간이 0 을 포함하지 않을 때만 부호 인정)

    [더미 효과] 조건 A(더미 없음) vs B(더미 추가), 모든 분할. 같은 문서 집합끼리 짝지어 G_B - G_A 로 β 들을 비교
        beta_diff < 0 이면 더미가 들어가면서 기존 문서에 대한 확증 편향이 줄었다 (더미 쪽으로 옮겨 갔다)

    [대조군] 조건 B(+더미) vs C(+외부 믿는 편 문서), 같은 문서 집합 / 같은 위치끼리 짝지어 비교
        added_diff  = push_added(B) - push_added(C)          > 0 : 모델이 자기 문서를 더 따른다
        absorb_diff = push_believed(C) - push_believed(B)    > 0 : 더미가 기존 믿는 편 문서의 영향을 더 가져간다
        분할은 같은 가중치로 평균한다
'''
from _init import *

import math, warnings

import numpy as np
import pandas as pd

from bait.core.confirmation_bias_measure import INSERT_AT


KEYS = ['split', 'qid', 'format', 'n_fact', 'n_counter']
GROUPS = ['fact', 'counter', 'other', 'counter_dataset']                 # 위 설명의 문항 그룹
ALL = 'all'                     # 모든 비율 / 위치를 합친 셀 이름 (plot 에서도 이 이름을 쓴다)
# beta 부호 -> (한글, 영문). 부호는 신뢰구간이 0 을 포함하지 않을 때만 +1 / -1
VERDICT = {1: ('확증 편향', 'confirmation bias'), -1: ('반확증', 'anti-confirmation'), 0: ('판정 보류', 'inconclusive')}


def _question_matrix(df: pd.DataFrame, value: str, by: str) -> pd.DataFrame:
    '''질문 x 셀 평균 행렬 (같은 질문 / 셀의 값이 여러 개면 평균) + 'all' 열 = 질문별 전체 평균'''
    m = df.pivot_table(index='qid', columns=by, values=value, aggfunc='mean', dropna=False)
    m[ALL] = df.groupby('qid')[value].mean()
    return m


def _bootstrap(m: pd.DataFrame, cells: list, n_boot: int, rng):
    '''질문을 복원 추출해 셀별 평균을 n_boot 번 구한다. 반환 (점추정 (K,), 복제 (n_boot, K))'''
    x = m.reindex(columns=cells).to_numpy(dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                         # 값이 전혀 없는 셀(0:9 등)의 경고
        return np.nanmean(x, axis=0), np.nanmean(x[idx], axis=1)


def _combine(fn, *stats):
    '''(점추정, 복제) 들을 같은 식으로 조합 : 점추정끼리, 복제는 같은 번호끼리'''
    return fn(*[s[0] for s in stats]), fn(*[s[1] for s in stats])


def _to_frame(cells: list, stats: dict) -> pd.DataFrame:
    '''{이름: (점추정, 복제)} -> 열 [이름, 이름_lo, 이름_hi]'''
    table = pd.DataFrame(index=cells)
    for name, (est, boot) in stats.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        table[name], table[f'{name}_lo'], table[f'{name}_hi'] = est, lo, hi
    return table


def _sign(table: pd.DataFrame, name: str):
    '''신뢰구간이 0 을 포함하지 않을 때만 +1 / -1, 아니면 0'''
    return np.where(table[f'{name}_lo'] > 0, 1, np.where(table[f'{name}_hi'] < 0, -1, 0))


def _groups(df: pd.DataFrame, dataset_counter_ids) -> dict:
    '''문항 그룹 이름 -> 행 마스크. 기록이 없는 그룹은 빠진다'''
    masks = {split: df['split'] == split for split in ('fact', 'counter', 'other')}
    masks['counter_dataset'] = masks['counter'] & df['qid'].isin(dataset_counter_ids)
    return {g: m for g, m in masks.items() if m.any()}


def _nan_like(stat):
    return np.full_like(stat[0], np.nan), np.full_like(stat[1], np.nan)


def _beta_stats(G: dict, prefix: str = '') -> dict:
    '''그룹별 (점추정, 복제) -> beta, beta_own, beta_dataset (모듈 설명의 공식). 없는 그룹이 들어간 값은 NaN'''
    f = G['fact']
    half = lambda a, b: (a - b) / 2
    return {f'beta{prefix}': _combine(half, f, G['counter']),
            f'beta_own{prefix}': _combine(half, f, G.get('other', _nan_like(f))),
            f'beta_dataset{prefix}': _combine(half, f, G.get('counter_dataset', _nan_like(f)))}


def beta_table(records: list, n_boot: int, seed: int, by: str = 'ratio', dataset_counter_ids=frozenset()):
    '''
        조건 A 의 β 표. 행 = by 의 셀 + 'all' (by : 'ratio' 문서 비율, 'format' 문서 형식)
        'all' 은 질문별로 모든 기록을 평균한 값이라 by 와 상관없이 같다. fact / counter 분할이 없으면 None
        dataset_counter_ids : counter_dataset 그룹의 문항 id (없으면 beta_dataset 은 NaN)
    '''
    df = pd.DataFrame([r for r in records if r['condition'] == 'A'])
    if df.empty or not {'fact', 'counter'} <= set(df['split']):
        return None

    df['ratio'] = df['n_fact'].astype(str) + ':' + df['n_counter'].astype(str)
    cells = ([f'{a}:{b}' for a, b in sorted(set(zip(df['n_fact'], df['n_counter'])))] if by == 'ratio'
             else list(dict.fromkeys(df[by]))) + [ALL]                     # 형식은 측정한 순서대로
    rng = np.random.default_rng(seed)

    G = {g: _bootstrap(_question_matrix(df[m], 'G', by), cells, n_boot, rng)
         for g, m in _groups(df, dataset_counter_ids).items()}
    table = _to_frame(cells, {**{f'G_{g}': G.get(g, _nan_like(G['fact'])) for g in GROUPS}, **_beta_stats(G),
                              'delta_doc': _combine(lambda a, b: (a + b) / 2, G['fact'], G['counter'])})
    verdicts = [VERDICT[sign] for sign in _sign(table, 'beta')]
    table['verdict'], table['verdict_en'] = [v[0] for v in verdicts], [v[1] for v in verdicts]
    return table


def dummy_effect_table(records: list, n_boot: int, seed: int, dataset_counter_ids=frozenset()):
    '''
        조건 A(더미 없음) vs B(더미 추가) 의 β 들. 행 = 더미 위치 + 'all'
        같은 (문항, 형식, 비율) 의 A 기록을 위치마다의 B 기록과 짝짓고, 짝 차이 G_B - G_A 로 *_diff 를 계산한다
        (beta_diff = beta_B - beta_A 와 같은 값이지만 짝 차이로 계산해 신뢰구간이 더 정확하다)
        fact / counter 분할의 B 기록이 없으면 None
    '''
    df = pd.DataFrame(records)
    if df.empty or not (df['condition'] == 'B').any():
        return None
    a = df[df['condition'] == 'A'][KEYS + ['G']].rename(columns={'G': 'G_A'})
    pairs = df[df['condition'] == 'B'][KEYS + ['position', 'G']].rename(columns={'G': 'G_B'}).merge(a, on=KEYS)
    pairs['G_diff'] = pairs['G_B'] - pairs['G_A']
    if not {'fact', 'counter'} <= set(pairs['split']):
        return None

    cells = [p for p in INSERT_AT if (pairs['position'] == p).any()] + [ALL]
    groups = _groups(pairs, dataset_counter_ids)
    rng = np.random.default_rng(seed)
    stats = {}
    for col, prefix in (('G_A', '_A'), ('G_B', '_B'), ('G_diff', '_diff')):
        G = {g: _bootstrap(_question_matrix(pairs[m], col, 'position'), cells, n_boot, rng) for g, m in groups.items()}
        stats.update(_beta_stats(G, prefix))
    return _to_frame(cells, stats)


def dummy_table(records: list, n_boot: int, seed: int):
    '''조건 B vs C 비교 표 (행 = 위치 + 'all'). 반환 (표, 사용한 분할) 또는 (None, [])'''
    df = pd.DataFrame(records)
    if df.empty or not (df['condition'] == 'B').any():
        return None, []

    # 반대편 기존 문서의 push : 사실을 믿으면(fact) 반사실 문서, 반사실을 믿으면(counter / other) 사실 문서
    df['push_opposing'] = np.where(df['split'] == 'fact', df['push_counter'], df['push_fact'])
    added = lambda cond: df[df['condition'] == cond][KEYS + ['position', 'push_added', 'push_believed', 'push_opposing']]
    base = df[df['condition'] == 'A'][KEYS + ['push_believed', 'push_opposing']].rename(
        columns={'push_believed': 'believed_A', 'push_opposing': 'opposing_A'})
    pairs = added('B').merge(added('C'), on=KEYS + ['position'], suffixes=('_B', '_C')).merge(base, on=KEYS)
    if pairs.empty:                                                             # C 를 측정하지 않았으면(--no-control) 짝이 없다
        return None, []
    pairs['added_diff'] = pairs['push_added_B'] - pairs['push_added_C']
    pairs['absorb_diff'] = pairs['push_believed_C'] - pairs['push_believed_B']

    columns = {'added_dummy': 'push_added_B', 'added_external': 'push_added_C', 'added_diff': 'added_diff',
               'believed_A': 'believed_A', 'believed_B': 'push_believed_B', 'believed_C': 'push_believed_C',
               'opposing_A': 'opposing_A', 'opposing_B': 'push_opposing_B', 'opposing_C': 'push_opposing_C',
               'absorb_diff': 'absorb_diff'}
    cells = [p for p in INSERT_AT if (pairs['position'] == p).any()] + [ALL]      # 위치 목록은 measure 의 정의를 따른다
    splits = [s for s in ('fact', 'counter', 'other') if (pairs['split'] == s).any()]
    rng = np.random.default_rng(seed)

    stats = {}
    for name, col in columns.items():
        per_split = [_bootstrap(_question_matrix(pairs[pairs['split'] == s], col, 'position'), cells, n_boot, rng)
                     for s in splits]
        stats[name] = _combine(lambda *xs: sum(xs) / len(xs), *per_split)     # 분할 같은 가중치 평균
    return _to_frame(cells, stats), splits


def diagnostics(metas: list, records: list, dataset_counter_ids=frozenset()) -> pd.DataFrame:
    '''분할별 문항 수 / 기록 수 / 건너뛴 문서 집합 수 / 사전 믿음 S(∅) 평균 / 더미에 들어 있는 정답 비율 (평가용)'''
    m = pd.DataFrame(metas)
    if 'skipped' not in m:                      # skipped 기록 이전 버전으로 측정한 결과
        m['skipped'] = 0
    table = m.groupby('split').agg(questions=('qid', 'count'), skipped=('skipped', 'sum'), prior_mean=('prior', 'mean'))
    table['records'] = pd.DataFrame(records).groupby('split').size()
    table['dataset_counter'] = m[m['qid'].isin(dataset_counter_ids)].groupby('split').size()   # counter_dataset 문항 수
    table['dataset_counter'] = table['dataset_counter'].fillna(0).astype(int)
    for name in ('zero_shot', 'dummy'):         # [평가용] 다시 생성한 zero-shot 답 / 더미에 들어 있는 정답의 비율
        if f'{name}_answer' in m:
            share = pd.crosstab(m['split'], m[f'{name}_answer'], normalize='index')
            table = table.join(share.add_prefix(f'{name}_has_'))
    return table


# ====================================================================== 출력

def _fmt(v, lo, hi) -> str:
    if math.isnan(v):
        return 'nan'
    star = '*' if lo > 0 or hi < 0 else ''
    return f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}'


def _display(table: pd.DataFrame, names: list, verdict: bool = False) -> str:
    out = pd.DataFrame({n: [_fmt(*v) for v in zip(table[n], table[f'{n}_lo'], table[f'{n}_hi'])] for n in names},
                       index=table.index)
    if verdict:
        out['verdict'] = table['verdict']
    return out.to_string()


def report_text(beta, beta_format, effect, dummy, dummy_splits, diag) -> str:
    '''로그 / report.txt 에 쓸 문자열. 값 [95% 신뢰구간], * = 신뢰구간이 0 을 포함하지 않음'''
    lines = ['', '[진단] 분할별 문항 수, counter_dataset 문항 수, 사전 믿음 S(∅) (fact 는 +, counter / other 는 - 여야 정상), '
                 '다시 생성한 zero-shot 답 / 더미에 들어 있는 정답 비율 (평가용)',
             diag.to_string(float_format=lambda v: f'{v:.3f}')]

    betas = ['beta', 'beta_own', 'beta_dataset', 'delta_doc']
    lines += ['', '[β] 조건 A. 사실을 믿는 fact 와 반사실을 믿는 counter / other / counter_dataset 의 G 비교  (* : 95% 신뢰구간이 0 을 포함하지 않음)',
              '  beta : 주 지표(fact vs counter) / beta_own : 재현(fact vs other) / beta_dataset : 강건성(fact vs counter_dataset)']
    if beta is None:
        lines.append('fact / counter 분할이 모두 있어야 계산할 수 있다.')
    else:
        lines += [_display(beta, [f'G_{g}' for g in GROUPS]), '', _display(beta, betas, verdict=True)]
    if beta_format is not None:
        lines += ['', '[β - 문서 형식별] 조건 A, 모든 비율 평균', _display(beta_format, betas, verdict=True)]

    if effect is not None:
        lines += ['', '[더미 효과] A(더미 없음) vs B(더미 추가), 모든 분할. *_diff = B - A (짝 차이). β 가 줄면(diff < 0) 편향이 더미 쪽으로 옮겨 감']
        for name in betas[:3]:
            lines += [_display(effect, [f'{name}_A', f'{name}_B', f'{name}_diff']), '']

    if dummy is not None:
        lines += ['', f'[대조군] B(+더미) vs C(+외부 같은 편 문서), 사용한 분할 : {dummy_splits}',
                  '  added_diff > 0 : 자기 문서를 더 따름 / absorb_diff > 0 : 더미가 기존 믿는 편 문서 영향을 더 가져감',
                  _display(dummy, ['added_dummy', 'added_external', 'added_diff']), '',
                  _display(dummy, ['believed_A', 'believed_B', 'believed_C', 'absorb_diff'])]
    return '\n'.join(lines) + '\n'
