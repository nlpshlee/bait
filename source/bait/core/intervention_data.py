"""Independent intervention inputs, generated-answer cache and small file helpers.

Retrieved documents follow main_experiments.ipynb's seeded sampling order.
Question-only answers and dummy documents are generated in this experiment;
no confirmation-bias measurement output is read.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re

from bait.core.bait_prompts import CONTEXT_SIZE, FILE_FORMATS
from bait.core.bait_utils import generate_dummy_docs, mix_contexts


ROOT = Path(__file__).resolve().parents[3]
# The notebook seeds once before this loop. Replaying earlier models preserves
# its document samples for the 8B model as well as the first, 3B model.
BENCHMARK_MODELS = ('Llama-3.2-3B', 'Llama-3.1-8B', 'Qwen2.5-3B', 'Qwen2.5-7B')


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(path, value):
    """Reject changed settings instead of mixing incompatible cached results."""
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Settings or data changed: {path}. Use a new run name.')
    else:
        write_json(path, value)


def load_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    rows, seen = [], set()
    with path.open('rb') as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete cache line {number}: {path}. Repair the interrupted tail before resuming.')
            try:
                row = json.loads(line)
                case_id = row['case_id']
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Invalid cache line {number}: {path}') from exc
            if case_id in seen:
                raise ValueError(f'Duplicate cached case {case_id}: {path}')
            seen.add(case_id)
            rows.append(row)
    return rows


def append_line(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


@contextmanager
def result_lock(path, *, shared=False, directory=False, blocking=False):
    """Lock the existing directory or append-only output itself; no .lock file."""
    flags = os.O_RDONLY | os.O_DIRECTORY if directory else os.O_RDWR | os.O_CREAT
    descriptor = os.open(path, flags, 0o666)
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    try:
        try:
            fcntl.flock(descriptor, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f'Another worker is using {path}.') from exc
        yield
    finally:
        os.close(descriptor)


class Dataset:
    """Load fact/counter questions and replay the original benchmark mixtures.

    Filtering questions or formats does not change document sampling. Each
    question has a deterministic rank for nested calibration prefixes/shards.
    """

    def __init__(self, model, n, seed=42, formats=None, ratios=None):
        if model not in BENCHMARK_MODELS:
            raise ValueError(f'No main_experiments document order is defined for {model}.')
        if n < 1:
            raise ValueError('The question count must be positive.')
        self.formats = list(FILE_FORMATS if formats is None else formats)
        available_ratios = [(i, CONTEXT_SIZE - 1 - i) for i in range(CONTEXT_SIZE)]
        self.ratios = available_ratios if ratios is None else [tuple(r) for r in ratios]
        if not self.formats or not set(self.formats) <= set(FILE_FORMATS):
            raise ValueError('Unknown or empty document format list.')
        if not self.ratios or not set(self.ratios) <= set(available_ratios):
            raise ValueError('Document ratios must be drawn from 0:9 through 9:0.')
        self.items, self._contexts = [], {}
        self.config = dict(model=model, n=n, seed=seed, formats=self.formats,
                           ratios=self.ratios, document_order='main_experiments.ipynb')
        fingerprints, ordered_hash, text_hashes = {}, hashlib.sha256(), {}
        preceding_models = BENCHMARK_MODELS[:BENCHMARK_MODELS.index(model)]
        state = random.getstate()
        try:
            random.seed(seed)
            for current_model in (*preceding_models, model):
                splits = ('fact', 'counter') if current_model == model else ('fact', 'counter', 'other')
                for split in splits:
                    path = ROOT / 'data/create_contexts' / current_model / f'bait_{current_model}_zero_shot_{split}_created_contexts.json'
                    blob = path.read_bytes()
                    data = json.loads(blob)
                    fingerprints[f'{current_model}/{split}'] = hashlib.sha256(blob).hexdigest()
                    if current_model == model and len(data) < n:
                        raise ValueError(f'{split}: requested {n}, but only {len(data)} questions exist.')
                    if len({str(d['id']) for d in data}) != len(data):
                        raise ValueError(f'Duplicate question IDs: {path}')
                    for index, row in enumerate(data):
                        key = f'{split}:{row["id"]}'
                        keep = current_model == model and index < n
                        if keep:
                            self.items.append(dict(split=split, key=key, data=row))
                        for fmt in FILE_FORMATS:
                            for ratio in available_ratios:
                                docs, _, _ = mix_contexts(row['contexts_fact'][fmt], row['contexts_counter'][fmt], *ratio)
                                if current_model != model:
                                    continue
                                order = None
                                if docs is not None:
                                    order = []
                                    for doc in docs:
                                        if doc not in text_hashes:
                                            text_hashes[doc] = hashlib.sha256(doc.encode()).hexdigest()
                                        order.append(text_hashes[doc])
                                ordered_hash.update(json.dumps([key, fmt, ratio, order], ensure_ascii=False).encode() + b'\n')
                                if keep and fmt in self.formats and ratio in self.ratios:
                                    self._contexts[key, fmt, ratio] = docs
        finally:
            random.setstate(state)
        self.ordered_contexts_sha256 = ordered_hash.hexdigest()
        self.fingerprint = digest(dict(data=fingerprints, config=self.config,
                                       ordered_contexts_sha256=self.ordered_contexts_sha256))
        for split in ('fact', 'counter'):
            def order_key(item):
                question = ' '.join(re.sub(r'[^\w\s]', ' ', item['data']['question'].lower()).split())
                return digest([seed, question, item['key']])
            ordered = sorted((x for x in self.items if x['split'] == split), key=order_key)
            for rank, item in enumerate(ordered):
                item['sample_rank'] = rank

    def selected(self, shard, num_shards, limit=None):
        if not 0 <= shard < num_shards:
            raise ValueError('Invalid shard index/count.')
        return sorted((x for x in self.items if x['sample_rank'] % num_shards == shard
                       and (limit is None or x['sample_rank'] < limit)),
                      key=lambda x: (x['sample_rank'], x['split']))

    def contexts(self, item, file_format, ratio):
        docs = self._contexts[item['key'], file_format, tuple(ratio)]
        return list(docs) if docs is not None else None


class DummyCache:
    """One question-only answer and one dummy per question, in this run only.

    The caller holds result_lock(self.path) for the worker's whole lifetime.
    Calibration/evaluation use the same shard count, so only the owning worker
    reads or extends this file. Generation always uses a batch of one question.
    """

    def __init__(self, output_dir, shard):
        self.path = Path(output_dir) / f'dummy.shard{shard}.jsonl'
        self.rows = {row['case_id']: row for row in load_lines(self.path)}

    def get(self, item, model, tokenizer, max_length=4096, max_new_tokens=192, zero_shot_max_new_tokens=64):
        key, question = item['key'], item['data']['question']
        if key not in self.rows:
            answers, dummies = generate_dummy_docs(model, tokenizer, [question], max_length,
                                                   max_new_tokens, zero_shot_max_new_tokens)
            if not answers[0].strip() or not dummies[0].strip():
                raise ValueError(f'{key}: model generated an empty zero-shot answer or dummy.')
            row = dict(case_id=key, question=question, zero_shot=answers[0], dummy=dummies[0])
            append_line(self.path, row)
            self.rows[key] = row
        row = self.rows[key]
        if row.get('question') != question or not row.get('zero_shot', '').strip() or not row.get('dummy', '').strip():
            raise ValueError(f'{key}: invalid or mismatched generated-answer cache in {self.path}.')
        return row
