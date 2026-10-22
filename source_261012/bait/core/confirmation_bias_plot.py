'''
    확증 편향 논문용 그림 (analysis 의 표만 받아서 그린다)

    공통 형식 : 2단 논문 전체 폭(6.9 inch), 8pt, 색각이상 친화 색(Okabe-Ito), 벡터 PDF(TrueType 폰트 내장) + 확인용 PNG
               그림 안에 제목 / 판정 문구를 넣지 않는다 (캡션에서 설명). 점 = 추정치, 선 = 95% 신뢰구간,
               채운 점 = 신뢰구간이 0 을 포함하지 않음

    fig_confirmation_bias
        (a) 문서 비율별 E[G], 믿음(분할)별 선. 확증 편향이면 사실을 믿을 때(파랑)가 반사실을 믿을 때(주황 / 노랑)보다 위에 있다.
            파랑-주황 간격 = 2β. 노랑(other) = 반사실 답이 모델 자기 답인 문항 (중립 기준선이 아니다)
        (b) 전체 비율 평균 β(주 지표), β_own(재현), β_dataset(강건성), Δ_doc
    fig_beta_by_format  문서 형식별 β 들 (형식이 여러 개일 때). 형식마다 같은 방향이면 형식에 강건하다
    fig_dummy_effect  [제안 방법] 더미 없음(A) vs 더미 추가(B), 세 분할 모두
        (a) β 들을 A(회색) / B(보라) 로 나란히. B 에서 0 쪽으로 줄면 기존 문서에 대한 확증 편향이 약해진 것
        (b) 위치별 짝 차이 B - A. 0 보다 왼쪽이면 더미가 들어가면서 편향이 줄었다 (더미 쪽으로 옮겨 갔다)
    fig_dummy_control  [평가용 대조군] 더미(B) vs 외부 같은 편 문서(C), 모든 분할
        (a) 조건 A(추가 없음) -> B(+자가 생성 더미) / C(+외부 문서) 에서 문서 1개당 push 변화
            믿는 편 기존 문서 / 반대편 기존 문서 / 추가 문서. 더미가 영향을 '가져가면' B 에서 믿는 편 문서가 C 보다 더 낮아지고
            더미 자신은 외부 문서보다 높다
        (b) 위치별 짝 차이 (B - C) : 추가 문서 push 차이, 기존 믿는 편 문서에서 더 가져간 양
'''
from _init import *

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bait.core.confirmation_bias_analysis import ALL


STYLE = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'savefig.dpi': 300, 'savefig.bbox': 'tight'}
BLUE, ORANGE, YELLOW, GRAY, PURPLE, GREEN = '#0072B2', '#D55E00', '#E69F00', '#999999', '#CC79A7', '#009E73'  # Okabe-Ito
BETA_ROWS = [('beta', r'$\beta$', 'black'), ('beta_own', r'$\beta_{\mathrm{own}}$', YELLOW),
             ('beta_dataset', r'$\beta_{\mathrm{dataset}}$', ORANGE)]


def _get(table, row, col):
    '''(값, 하한, 상한)'''
    return tuple(float(table.loc[row, c]) for c in (col, f'{col}_lo', f'{col}_hi'))


def _significant(lo, hi) -> bool:
    return lo > 0 or hi < 0


def _forest(ax, items):
    '''items : [(y, 라벨, (값, 하한, 상한), 색)] 를 가로 신뢰구간 + 점으로. 유의하면 채운 점'''
    for y, _, (v, lo, hi), color in items:
        if np.isnan(v):
            continue
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.2)
        ax.plot(v, y, 'o', color=color, markersize=5, markerfacecolor=color if _significant(lo, hi) else 'white')
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))     # 눈금 숫자가 붙지 않게
    ax.set_xlabel('Estimate (95% CI)')


def _err(v, lo, hi):
    '''errorbar 용 (아래, 위) 길이. 부트스트랩 구간이 점추정을 비켜 갈 때 음수가 되지 않게 0 으로 자른다'''
    return [np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)]


def _beta_forest(ax, table, suffix: str = ''):
    '''행 = 표의 셀('all' 이 맨 위), 셀마다 β / β_fact / β_counter 세 점. suffix : 열 이름 뒤에 붙는 '_diff' 등'''
    cells = [ALL] + [c for c in table.index if c != ALL]
    _forest(ax, [(-i + offset, '', _get(table, cell, col + suffix), color)
                 for i, cell in enumerate(cells) for (col, _, color), offset in zip(BETA_ROWS, (0.2, 0.0, -0.2))])
    ax.set_yticks([-i for i in range(len(cells))], cells)
    for _, label, color in BETA_ROWS:
        ax.plot([], [], 'o', color=color, label=label)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)


def _panel_label(ax, text):
    ax.text(-0.14, 1.04, text, transform=ax.transAxes, fontweight='bold', fontsize=9)


def _save(fig, out_dir, name) -> list:
    paths = [os.path.join(out_dir, f'{name}.{ext}') for ext in ('pdf', 'png')]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def plot_confirmation_bias(beta, out_dir: str) -> list:
    ratios = [c for c in beta.index if c != ALL and beta.loc[c, ['G_fact', 'G_counter']].notna().all()]
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={'width_ratios': [1.7, 1]})

        x = np.arange(len(ratios))
        for col, color, marker, style, label in (('G_fact', BLUE, 'o', '-', 'Believes fact'),
                                                  ('G_counter', ORANGE, 's', '-', 'Believes counterfactual'),
                                                  ('G_other', YELLOW, '^', '--', 'Believes own answer (other)')):
            v, lo, hi = (beta.loc[ratios, c].to_numpy(float) for c in (col, f'{col}_lo', f'{col}_hi'))
            if np.isnan(v).all():
                continue
            ax1.plot(x, v, style, color=color, marker=marker, markersize=3.5, linewidth=1.2, label=label)
            ax1.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax1.axhline(0, color='black', linewidth=0.6)
        ax1.set_xticks(x, ratios)
        ax1.set_xlabel('Documents in context (fact : counterfactual)')
        ax1.set_ylabel('Strength gap $G$')
        ax1.legend(frameon=False, loc='best')
        _panel_label(ax1, '(a)')

        rows = BETA_ROWS + [('delta_doc', r'$\Delta_{\mathrm{doc}}$', GRAY)]
        _forest(ax2, [(-i, label, _get(beta, ALL, col), color) for i, (col, label, color) in enumerate(rows)])
        ax2.set_yticks([-i for i in range(len(rows))], [label for _, label, _ in rows])
        ax2.axhline(-2.5, color=GRAY, linewidth=0.5, linestyle=':')            # β 들과 Δ_doc 구분
        _panel_label(ax2, '(b)')

        fig.tight_layout()
        return _save(fig, out_dir, 'fig_confirmation_bias')


def plot_dummy(dummy, splits: list, out_dir: str) -> list:
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={'width_ratios': [1.3, 1]})

        # (a) A -> B / C 에서 문서 1개당 push (모든 위치 평균)
        x = np.array([0, 1, 2])
        for prefix, color, marker, shift, label in (('believed', BLUE, 'o', -0.15, 'Existing, believed side'),
                                                    ('opposing', ORANGE, 's', 0.15, 'Existing, opposing side')):
            vals = [_get(dummy, ALL, f'{prefix}_{c}') for c in 'ABC']
            v, lo, hi = (np.array(t) for t in zip(*vals))
            ax1.errorbar(x + shift, v, yerr=_err(v, lo, hi), color=color, marker=marker, markersize=4, linewidth=1.2,
                         capsize=2, label=label)
        added = [_get(dummy, ALL, 'added_dummy'), _get(dummy, ALL, 'added_external')]
        v, lo, hi = (np.array(t) for t in zip(*added))
        ax1.errorbar(x[1:], v, yerr=_err(v, lo, hi), color=PURPLE, marker='D', markersize=4.5, linestyle='none',
                     capsize=2, label='Added document')
        ax1.axhline(0, color='black', linewidth=0.6)
        ax1.set_xticks(x, ['A: none', 'B: + self-generated', 'C: + external'])
        ax1.set_xlim(-0.45, 2.45)
        ax1.set_ylabel('Push per document')
        ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3,
                   title=f"belief split: {', '.join(splits)}", title_fontsize=6)
        _panel_label(ax1, '(a)')

        # (b) 위치별 짝 차이 B - C
        cells = [ALL] + [c for c in dummy.index if c != ALL]
        items = []
        for i, cell in enumerate(cells):
            items.append((-i + 0.15, cell, _get(dummy, cell, 'added_diff'), PURPLE))
            items.append((-i - 0.15, cell, _get(dummy, cell, 'absorb_diff'), GREEN))
        _forest(ax2, items)
        ax2.set_yticks([-i for i in range(len(cells))], cells)
        ax2.plot([], [], 'o', color=PURPLE, label='Added doc: self $-$ external')
        ax2.plot([], [], 'o', color=GREEN, label='Extra absorption (self $-$ external)')
        ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1)
        _panel_label(ax2, '(b)')

        fig.tight_layout()
        return _save(fig, out_dir, 'fig_dummy_control')


def plot_beta_by_format(beta_format, out_dir: str) -> list:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.3, 2.6))                              # 단 하나 폭
        _beta_forest(ax, beta_format)
        fig.tight_layout()
        return _save(fig, out_dir, 'fig_beta_by_format')


def plot_dummy_effect(effect, out_dir: str) -> list:
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4))

        # (a) 모든 위치 평균 : 더미 없음(A) vs 더미 추가(B)
        items = []
        for i, (col, _, _) in enumerate(BETA_ROWS):
            items.append((-i + 0.15, '', _get(effect, ALL, f'{col}_A'), GRAY))
            items.append((-i - 0.15, '', _get(effect, ALL, f'{col}_B'), PURPLE))
        _forest(ax1, items)
        ax1.set_yticks([-i for i in range(len(BETA_ROWS))], [label for _, label, _ in BETA_ROWS])
        ax1.plot([], [], 'o', color=GRAY, label='A: no dummy')
        ax1.plot([], [], 'o', color=PURPLE, label='B: + self-generated dummy')
        ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        _panel_label(ax1, '(a)')

        # (b) 위치별 짝 차이 B - A
        _beta_forest(ax2, effect, '_diff')
        ax2.set_xlabel('Change with dummy, B $-$ A (95% CI)')
        _panel_label(ax2, '(b)')

        fig.tight_layout()
        return _save(fig, out_dir, 'fig_dummy_effect')


def plot_all(beta, beta_format, effect, dummy, dummy_splits, out_dir: str) -> list:
    '''그릴 수 있는 그림만 out_dir 에 PDF / PNG 로 저장하고 경로 목록을 돌려준다'''
    paths = []
    if beta is not None:
        paths += plot_confirmation_bias(beta, out_dir)
    if beta_format is not None:
        paths += plot_beta_by_format(beta_format, out_dir)
    if effect is not None:
        paths += plot_dummy_effect(effect, out_dir)
    if dummy is not None:
        paths += plot_dummy(dummy, dummy_splits, out_dir)
    return paths
