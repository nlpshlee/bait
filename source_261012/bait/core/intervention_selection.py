"""Select fixed-budget intervention sites from complete calibration measurements.

Each ratio equally weights belief splits and formats. Common scores use the
minimum across ratios, so a common site must score positively in every ratio.
A positive score predicts that suppression lowers the saved zero-shot answer's
log probability; it does not establish bias transfer. Indices are zero based.
"""
from numbers import Integral

import numpy as np


MODULES = ('mlp', 'heads', 'residual')
SPLITS = ('fact', 'counter')


def _positive_int(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be a {"nonnegative" if allow_zero else "positive"} integer.')
    return int(value)


def _selection(scores, top_counts, scope, coverage_ready):
    """Rank sites within each module; never fill a budget with negative sites."""
    modules, candidates = {}, []
    sites = {'mlp_layers': [], 'heads': {}, 'residual_layers': []}
    for module in MODULES:
        count = top_counts[module]
        values = None if scores is None else scores[module]
        selected = []
        positive_count = 0 if values is None else int(np.count_nonzero(values > 0))
        available = values is not None and positive_count >= count
        if available:
            # Stable sorting gives deterministic layer/head order for exact ties.
            order = np.argsort(-values.ravel(), kind='stable')[:count]
            for rank, flat_index in enumerate(order, 1):
                coordinates = np.unravel_index(int(flat_index), values.shape)
                layer = int(coordinates[0])
                head = int(coordinates[1]) if module == 'heads' else None
                record = {'scope': scope, 'module': module, 'rank': rank,
                          'layer': layer, 'head': head, 'score': float(values[coordinates])}
                selected.append(record)
                candidates.append(record)
                if module == 'heads':
                    sites['heads'].setdefault(str(layer), []).append(head)
                else:
                    sites[f'{module}_layers'].append(layer)
        reason = None
        if values is None:
            reason = 'No complete questions in both belief splits.'
        elif positive_count < count:
            reason = f'Only {positive_count} positive sites for the requested budget of {count}.'
        modules[module] = {'available': available, 'requested_count': count,
                           'positive_count': positive_count, 'selected': selected,
                           'unavailable_reason': reason}
    reasons = [f'{name}: {info["unavailable_reason"]}' for name, info in modules.items()
               if not info['available']]
    if not coverage_ready:
        reasons.insert(0, 'The requested paired question prefix is incomplete.')
    return {'available': coverage_ready and all(info['available'] for info in modules.values()),
            'unavailable_reason': ' '.join(reasons) or None,
            'sites': sites, 'modules': modules,
            'scores': None if scores is None else {name: scores[name].tolist() for name in MODULES},
            'candidate_table': candidates}


def _prefix_selection(questions, ratios, formats, top_counts, sample_n, shape):
    expected_cells = {(fmt, ratio) for fmt in formats for ratio in ratios}
    complete = {split: [] for split in SPLITS}
    incomplete = {split: [] for split in SPLITS}
    present_counts = {}
    for split in SPLITS:
        present_counts[split] = sum((split, rank) in questions for rank in range(sample_n))
        for rank in range(sample_n):
            question = questions.get((split, rank))
            usable = {} if question is None else question['usable']
            missing = sorted(expected_cells.difference(usable))
            if missing:
                incomplete[split].append({'sample_rank': rank,
                    'key': None if question is None else question['key'],
                    'missing': [{'format': fmt, 'ratio': ratio} for fmt, ratio in missing]})
            else:
                complete[split].append(question)
    coverage_ready = all(len(complete[split]) == sample_n for split in SPLITS)
    scores = None
    if shape is not None and all(complete.values()):
        scores = {ratio: {module: np.zeros(shape[module], dtype=np.float64) for module in MODULES}
                  for ratio in ratios}
        for split in SPLITS:
            # Equal belief/format weighting even in an incomplete preview.
            weight = 1. / (len(SPLITS) * len(formats) * len(complete[split]))
            for question in complete[split]:
                for (fmt, ratio), ranking in question['usable'].items():
                    for module in MODULES:
                        scores[ratio][module] += ranking[module] * weight
    per_ratio = {ratio: _selection(None if scores is None else scores[ratio], top_counts,
                                   ratio, coverage_ready) for ratio in ratios}
    common_scores = None if scores is None else {
        module: np.minimum.reduce([scores[ratio][module] for ratio in ratios]) for module in MODULES}
    common = _selection(common_scores, top_counts, 'common', coverage_ready)
    return {'sample_n': sample_n, 'ready': coverage_ready,
            'available': coverage_ready and common['available'] and all(s['available'] for s in per_ratio.values()),
            'counts': {'requested_per_split': sample_n, 'present_questions': present_counts,
                       'complete_questions': {split: len(complete[split]) for split in SPLITS}},
            'complete_question_keys': {split: [q['key'] for q in complete[split]] for split in SPLITS},
            'incomplete_questions': incomplete, 'per_ratio': per_ratio, 'common': common,
            'candidate_table': [r for selection in [*per_ratio.values(), common]
                                for r in selection['candidate_table']]}


def aggregate_selection(rows, ratios, formats, top_counts, sample_n):
    """Select per-ratio/common sites from ranks 0..sample_n-1 in each split.

    Rows require case_id, key (split:qid), split, format, ratio, sample_rank,
    and ranking={mlp:[layers], residual:[layers], heads:[layers, heads]}.
    Missing/skipped conditions make ready=False. Duplicates, inconsistent
    question ranks, and nonfinite scores raise ValueError.

    available=True also requires enough positive sites for every module budget;
    a zero budget disables that module. Incomplete previews are never available.
    """
    sample_n = _positive_int(sample_n, 'sample_n')
    ratios, formats = list(ratios), list(formats)
    if not ratios or len(ratios) != len(set(ratios)) or not all(isinstance(r, str) for r in ratios):
        raise ValueError('ratios must be a nonempty list of distinct strings.')
    if not formats or len(formats) != len(set(formats)) or not all(isinstance(f, str) for f in formats):
        raise ValueError('formats must be a nonempty list of distinct strings.')
    if set(top_counts) != set(MODULES):
        raise ValueError(f'top_counts must contain exactly {MODULES}.')
    top_counts = {name: _positive_int(top_counts[name], f'top_counts[{name}]', True) for name in MODULES}
    questions, key_ranks, case_ids, conditions = {}, {}, set(), set()
    shape, analyzed_rows, skipped_rows = None, 0, []
    for row in rows:
        if row.get('ratio') not in ratios or row.get('format') not in formats:
            continue
        rank = _positive_int(row['sample_rank'], 'sample_rank', True)
        if rank >= sample_n:
            continue
        split = row['split']
        if split not in SPLITS:
            raise ValueError(f'Unsupported belief split: {split!r}.')
        key, case_id = row['key'], row['case_id']
        if not isinstance(key, str) or not key.startswith(split + ':') or not isinstance(case_id, str) or not case_id:
            raise ValueError('Each row requires a nonempty case_id and a split:qid key.')
        condition = (split, rank, row['format'], row['ratio'])
        if case_id in case_ids or condition in conditions:
            raise ValueError(f'Duplicate case or question/format/ratio condition: {case_id}.')
        case_ids.add(case_id)
        conditions.add(condition)
        if key in key_ranks and key_ranks[key] != rank:
            raise ValueError(f'Question {key} has inconsistent sample ranks.')
        key_ranks[key] = rank
        question = questions.setdefault((split, rank), {'key': key, 'usable': {}})
        if question['key'] != key:
            raise ValueError(f'{split} rank {rank} refers to more than one question.')
        analyzed_rows += 1
        if row.get('skipped') or row.get('ranking') is None:
            skipped_rows.append({'case_id': case_id, 'reason': str(row.get('skipped') or 'Missing ranking.')})
            continue
        ranking = {name: np.asarray(row['ranking'][name], dtype=np.float64) for name in MODULES}
        if (ranking['mlp'].ndim != 1 or ranking['residual'].ndim != 1 or ranking['heads'].ndim != 2
                or ranking['mlp'].size == 0 or ranking['heads'].shape[1] == 0
                or not ranking['mlp'].shape == ranking['residual'].shape == ranking['heads'].shape[:1]):
            raise ValueError(f'Invalid layer/head score dimensions in {case_id}.')
        row_shape = {name: value.shape for name, value in ranking.items()}
        if shape is not None and row_shape != shape:
            raise ValueError(f'Inconsistent model score dimensions in {case_id}.')
        if not all(np.isfinite(value).all() for value in ranking.values()):
            raise ValueError(f'Nonfinite ranking score in {case_id}.')
        shape = row_shape
        question['usable'][(row['format'], row['ratio'])] = ranking

    result = _prefix_selection(questions, ratios, formats, top_counts, sample_n, shape)
    result.update({'schema_version': 1, 'ratios': ratios, 'formats': formats,
                   'top_counts': top_counts, 'analyzed_rows': analyzed_rows, 'skipped_rows': skipped_rows,
                   'definitions': {
                       'indexing': 'zero-based layer and head indices',
                       'per_ratio_score': 'equal mean over belief splits and formats, then paired questions',
                       'common_score': 'minimum per-ratio mean signed score over all requested ratios',
                       'positive': 'local estimate that suppression lowers the saved zero-shot answer log-probability',
                       'pairing': 'same questions complete in all requested formats and ratios for every scope',
                       'unavailable': 'incomplete requested prefix or fewer positive candidates than requested; no fallback'}})
    return result
