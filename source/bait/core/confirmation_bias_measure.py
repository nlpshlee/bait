'''
    확증 편향 측정 : 문서 단위 knockout

        S(D)    = logP(정답_fact | D) - logP(정답_counter | D)          (정답 토큰 로그확률 평균)
        IE(d)   = S(D) - S(D - d)        D - d : 문서 d 의 토큰을 attention_mask 에서 0 으로 (다른 토큰 위치는 그대로)
        push(d) = IE(d) (사실 문서),  -IE(d) (반사실 문서)              부호 반전 기준은 IE 값이 아니라 문서 라벨
        G(D)    = mean push(사실 문서) - mean push(반사실 문서)          한쪽 문서가 없으면(0:9, 9:0) NaN

    분할과 믿는 편 (history.md 2.1) : fact 분할은 사실을, counter / other 분할은 반사실을 믿는다.
        other 분할(과 counter 분할의 대부분)은 answer_counter 를 모델 자신의 zero-shot 답으로 바꿔 만든 문항이라
        '믿음이 어느 쪽도 아닌' 중립 분할이 아니다.

    조건
        A : 사실 + 반사실 문서
        B : A + 모델이 직접 쓴 더미 문서. 모든 분할 / 모든 문항에 항상 넣는다 (제안 방법)
            더미 = 모델이 지금 내는 zero-shot 답을 사실처럼 서술한 문단 (bait_utils.generate_dummy_docs, prepare_dummies).
            라벨 없이 모델 출력만 쓴다.
            더미 내용은 정답 라벨로 검사하지 않는다. 실제 적용 시에는 라벨을 모르므로 측정 조건에 라벨을 쓰면 안 된다.
            zero-shot 답 / 더미에 어느 정답이 들어 있는지는 평가용으로만 기록한다 (meta['zero_shot_answer'], meta['dummy_answer'])
        C : A + 외부 믿는 편 문서 (평가용 대조군. 모든 분할, other 분할의 믿는 편은 counter)
            외부 문서 = 상위 모델이 미리 생성한 믿는 편 문서 중 번호가 가장 큰 것(context_10). 새로 만들지 않는다.
            이 문서는 모든 분할 / 양쪽 편에서 기본 문서 집합 후보에서 빼 두므로 (기본 문서는 context_1 ~ 9 에서만 뽑는다)
            A 에 이미 들어 있을 수 없고, 문항 / 형식마다 하나로 고정된다 (비율 / 위치가 바뀌어도 같은 문서. 더미와 같은 조건)
            같은 편 문서가 하나 늘면 중복 때문에 기존 문서의 IE 는 원래 줄어든다. B 와 비교해 '자가 생성이라서' 생긴 효과만 남긴다.
            '믿는 편' 을 고르려면 라벨이 필요하므로 제안 방법이 아니라 해석을 위한 대조 실험이다.
        B, C 는 같은 문서 집합의 같은 위치(first / middle / last)에 넣어 짝지어 비교한다.
'''
from _init import *

import glob, json, math, os, random, re

import torch
import torch.nn.functional as F

from bait.core.bait_prompts import get_generate_prompt
from bait.core.bait_utils import generate_dummy_docs, mix_contexts
from bait.utils import container_utils, context_utils, model_utils


NAN = float('nan')
BELIEVED = {'fact': 'fact', 'counter': 'counter', 'other': 'counter'}      # 분할 -> 모델이 믿는 편 (평가용)
INSERT_AT = {'first': lambda n: 0, 'middle': lambda n: n // 2, 'last': lambda n: n}


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else NAN


class Scorer:
    '''
        S 와 IE 를 계산한다. 모델은 sdpa 로 로드한다.
        (flash_attention_2 도 가린 토큰을 패딩처럼 빼고 계산해 결과는 같다 : float32 기준 오차가 sdpa 와 같은 수준임을 확인.
         다만 3B 에서 약 5% 빠를 뿐이고 수치가 달라 더미 문장도 바뀌므로, 한 실행 안에서 두 구현을 섞지 말 것)

        정밀도 : lm_head 만 float32 로 계산한다. bf16 로짓은 0.125 단위로 반올림되어 작은 IE 가 묻힌다.
        속도   : 프롬프트는 한 번만 계산(KV 캐시)하고 정답 토큰만 이어서 계산한다.
                 문서 집합 하나(변형 약 11행 x 1000 토큰)로 A100 연산이 이미 포화되어 여러 집합을 묶어도 빨라지지 않는다.
    '''

    def __init__(self, model, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.backbone = model.get_decoder()                                     # 마지막 norm 까지 적용된 은닉 상태
        head = model.get_output_embeddings()
        if getattr(head, 'bias', None) is not None:                             # Llama / Qwen2 는 bias 가 없다
            raise ValueError('Scorer : lm_head 에 bias 가 있는 모델은 지원하지 않는다 (logits = W h 를 가정)')
        self.head = head.weight.detach().float()

    def encode(self, question: str, contexts: list):
        '''
            질문 + 문서 -> (프롬프트 토큰 (P,), 문서별 토큰 인덱스 리스트). 문서가 없으면 zero-shot 프롬프트
            프롬프트가 max_seq_length 이상이면 make_inputs 가 '뒤쪽'(질문, assistant 머리말)부터 잘라버리므로
            (None, None) 을 돌려 측정하지 않게 한다. (문서는 다 남아 있어 겉보기엔 정상이라 조용히 틀린 값이 나온다)
        '''
        chat_prompts, inputs = model_utils.make_inputs(
            self.tokenizer, None, [get_generate_prompt(question, contexts)], self.max_seq_length,
            return_offsets_mapping=True, return_all=True
        )
        input_ids = inputs['input_ids'][0]
        if len(input_ids) >= self.max_seq_length:
            return None, None
        spans = context_utils.extract_context_tok_idxs(chat_prompts[0], contexts, inputs['offset_mapping'][0])
        return input_ids, [spans[f'{i}'] for i in range(len(contexts))]

    def _logp(self, last_hidden, cache, masks, answer: str):
        '''변형마다 logP(정답). 정답이 여러 토큰이면 캐시 뒤에 이어 넣어 계산하고, 끝나면 캐시를 프롬프트 길이로 되돌린다'''
        n_variants, prompt_len = masks.shape
        ids = torch.tensor(self.tokenizer(answer, add_special_tokens=False)['input_ids'], device=masks.device)

        hidden = last_hidden.unsqueeze(1)                                       # (V, 1, H) : 정답 첫 토큰 예측
        if len(ids) > 1:
            cont = ids[:-1].repeat(n_variants, 1)                               # 정답[:-1] 을 넣어 정답[1:] 예측
            out = self.backbone(input_ids=cont, attention_mask=torch.cat([masks, torch.ones_like(cont)], dim=1),
                                past_key_values=cache, use_cache=True)
            hidden = torch.cat([hidden, out.last_hidden_state], dim=1)          # (V, m, H)
            cache.crop(prompt_len)

        logp = F.linear(hidden.float(), self.head).log_softmax(dim=-1)          # (V, m, vocab), float32
        return logp.gather(2, ids.repeat(n_variants, 1).unsqueeze(2)).squeeze(2).mean(dim=1)

    @torch.no_grad()
    def knockout(self, input_ids, doc_idxs: list, answers: tuple):
        '''
            한 배치 = [아무것도 안 가림, 문서 0 가림, 문서 1 가림, ...]
            반환 : (S(D), [IE(d_0), IE(d_1), ...])
            기준 S(D) 를 가림 변형들과 같은 배치에서 계산해야 bf16 배치 잡음이 IE 에 섞이지 않는다.
        '''
        device = self.head.device
        masks = torch.ones(len(doc_idxs) + 1, len(input_ids), dtype=torch.long)
        for i, idxs in enumerate(doc_idxs):
            masks[i + 1, idxs] = 0
        masks = masks.to(device)

        out = self.backbone(input_ids=input_ids.to(device).repeat(len(masks), 1), attention_mask=masks, use_cache=True)
        last, cache = out.last_hidden_state[:, -1], out.past_key_values
        s = (self._logp(last, cache, masks, answers[0]) - self._logp(last, cache, masks, answers[1])).cpu()
        return float(s[0]), (s[0] - s[1:]).tolist()


def prepare_dummies(model, tokenizer, datas: list, qids: set, args) -> dict:
    '''
        분할의 문항(datas = 앞에서부터 n 개) 중 qids 에 든 문항의 더미를 배치로 만든다. 반환 {qid: (zero-shot 답, 더미)}
        배치는 항상 datas 를 처음부터 args.dummy_batch_size 개씩 자른 묶음이다 (샤드 / 이어 돌리기와 무관).
        배치 생성은 함께 묶인 문항에 따라 문장이 달라지므로, 묶음을 고정해 샤드 수나 중단 시점이 바뀌어도 같은 더미가 나오게 한다.
        (--n 이 달라지면 마지막 묶음만 달라질 수 있다.) qids 가 하나도 없는 묶음은 생성하지 않는다.
    '''
    dummies = {}
    for chunk in container_utils.chunks(datas, args.dummy_batch_size):
        if not any(d['id'] in qids for d in chunk):
            continue
        zero_shots, texts = generate_dummy_docs(model, tokenizer, [d['question'] for d in chunk], args.max_seq_length,
                                                args.dummy_max_new_tokens, args.zero_shot_max_new_tokens)
        dummies.update({d['id']: pair for d, pair in zip(chunk, zip(zero_shots, texts)) if d['id'] in qids})
    return dummies


def split_pool(docs_by_key: dict):
    '''
        생성 문서 {'context_N': 본문} -> (기본 문서 후보 {키: 본문}, 대조군 C 용으로 떼어 둔 문서 또는 None)
        번호가 가장 큰 문서(context_10)를 떼어 두고 나머지에서만 기본 문서 집합을 뽑는다. 빈 문서(생성 실패)는 쓰지 않는다.
        키를 번호 순으로 정렬한다 (사전 순이면 context_10 이 context_1 다음에 온다)
    '''
    keys = sorted(docs_by_key, key=lambda k: int(re.search(r'\d+$', k).group()))
    pool = {k: docs_by_key[k] for k in keys[:-1] if docs_by_key[k].strip()}
    held_out = docs_by_key[keys[-1]]
    return pool, (held_out if held_out.strip() else None)


def answers_in(text: str, data: dict) -> str:
    '''[평가용] 문서에 들어 있는 정답 : 'fact' / 'counter' / 'both' / 'none'. 측정 조건을 정하는 데는 쓰지 않는다'''
    has_fact, has_counter = (bool(text) and model_utils.is_correct(text, data[f'answer_{side}'])[1]
                             for side in ('fact', 'counter'))
    return {(True, False): 'fact', (False, True): 'counter', (True, True): 'both'}.get((has_fact, has_counter), 'none')


def gap_stats(ie: list, sides: list, believed=None) -> dict:
    '''
        문서별 IE 와 라벨 -> 지표 (라벨 : 'fact' / 'counter' / 'added' (B, C 의 추가 문서))
            push_fact, push_counter, G : 기존 문서만으로 계산
            ie_added      : 추가 문서의 IE 그대로 (S 축, 사실 쪽이 +). 모든 분할
            push_added    : 추가 문서의 push, 믿는 쪽이 + (believed 를 줄 때)
            push_believed : 믿는 편 기존 문서의 평균 push (believed 를 줄 때)
    '''
    push_fact = mean([v for v, s in zip(ie, sides) if s == 'fact'])
    push_counter = mean([-v for v, s in zip(ie, sides) if s == 'counter'])
    added = [v for v, s in zip(ie, sides) if s == 'added']
    out = {'push_fact': push_fact, 'push_counter': push_counter, 'G': push_fact - push_counter,
           'ie_added': added[0] if added else NAN}

    if believed:
        out['push_added'] = (added[0] if believed == 'fact' else -added[0]) if added else NAN
        out['push_believed'] = push_fact if believed == 'fact' else push_counter
    return out


def measure_question(scorer: Scorer, data: dict, split: str, args, dummy=None):
    '''
        문항 하나 측정. 형식 x 비율마다 조건 A 를, 위치마다 B 와 C 를 측정한다.
        dummy : prepare_dummies() 가 만든 (zero-shot 답, 더미). None 이거나 더미가 비면 B 를 만들지 않는다
        args  : formats, ratios, positions, no_control, seed
        반환 : (문항 메타, 문서 집합 기록 리스트). 메타의 skipped = 건너뛴 문서 집합 수
               (너무 길거나 문서 토큰을 못 찾은 집합 + 빈 문서 때문에 후보가 모자란 비율. 비율이 빠지면 그 비율의 B, C 도 빠진다)
    '''
    qid, question = data['id'], data['question']
    answers = (data['answer_fact'], data['answer_counter'])
    believed = BELIEVED[split]                                                  # C 와 push_believed 에만 쓴다 (평가용)

    prior_ids, _ = scorer.encode(question, [])
    meta = {'split': split, 'qid': qid, 'skipped': 0,
            'prior': scorer.knockout(prior_ids, [], answers)[0] if prior_ids is not None else NAN}

    if dummy is not None:                                                       # 더미는 분할 / 내용과 무관하게 항상
        zero_shot, dummy = dummy
        meta.update(zero_shot=zero_shot, dummy=dummy,                           # *_answer 는 평가용 기록
                    zero_shot_answer=answers_in(zero_shot, data), dummy_answer=answers_in(dummy, data))

    records = []
    for file_format in args.formats:
        # 양쪽 편 모두 context_10 은 대조군 C 용으로 떼어 두고 context_1 ~ 9 에서만 기본 문서를 뽑는다 (분할과 무관한 같은 규칙)
        pools, held_out = {}, {}
        for side in ('fact', 'counter'):
            pools[side], held_out[side] = split_pool(data[f'contexts_{side}'][file_format])

        for n_fact, n_counter in args.ratios:
            random.seed(f'{args.seed}|{split}|{qid}|{file_format}|{n_fact}|{n_counter}')    # 재현 가능한 문서 집합
            docs, fact_idxs, _ = mix_contexts(pools['fact'], pools['counter'], n_fact, n_counter)
            if docs is None:                                                    # 빈 문서 때문에 후보가 모자람
                meta['skipped'] += 1
                continue
            sides = ['fact' if i in fact_idxs else 'counter' for i in range(len(docs))]
            doc_sets = [('A', 'none', docs, sides)]

            # 대조군 C 의 외부 문서 : 믿는 편의 떼어 둔 문서 (문항 / 형식마다 고정). 생성 문서가 우연히 똑같아
            # 이미 집합에 들어 있으면 같은 문서를 두 번 넣게 되므로 뺀다
            external = held_out[believed] if believed and not args.no_control else None
            if external in docs:
                external = None
            if dummy:                                                           # 생성이 빈 문자열이면 넣을 문서가 없다
                for position in args.positions:
                    at = INSERT_AT[position](len(docs))
                    for condition, text in (('B', dummy), ('C', external)):
                        if text:
                            doc_sets.append((condition, position, docs[:at] + [text] + docs[at:],
                                             sides[:at] + ['added'] + sides[at:]))

            for condition, position, contexts, doc_sides in doc_sets:
                input_ids, doc_idxs = scorer.encode(question, contexts)
                if input_ids is None or not all(doc_idxs):                      # 너무 길거나 문서 토큰을 못 찾으면 건너뜀
                    meta['skipped'] += 1
                    continue
                score, ie = scorer.knockout(input_ids, doc_idxs, answers)
                records.append({'split': split, 'qid': qid, 'format': file_format, 'n_fact': n_fact,
                                'n_counter': n_counter, 'condition': condition, 'position': position,
                                'S': score, 'ie': ie, 'sides': doc_sides, **gap_stats(ie, doc_sides, believed)})
    return meta, records


# ====================================================================== 저장 (문항 하나 = 한 줄)

def save_question(path: str, meta: dict, records: list):
    '''한 줄에 문항 하나. 직전 실행이 쓰다 끊겨 줄바꿈 없이 끝났으면 먼저 줄을 바꾼다 (새 줄이 함께 손상되지 않도록)'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    broken = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            broken = f.read(1) != b'\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(('\n' if broken else '') + json.dumps({'meta': meta, 'records': records}, ensure_ascii=False) + '\n')


def load_results(out_dir: str):
    '''
        out_dir 의 measurements*.jsonl 을 모두 읽는다 (GPU 별로 나눠 저장한 파일도 함께).
        반환 (문항 메타 리스트, 기록 리스트). 손상된 줄은 건너뛰고, 같은 문항이 여러 번 있으면 마지막 것을 쓴다.
    '''
    items = {}
    for path in sorted(glob.glob(os.path.join(out_dir, 'measurements*.jsonl'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    items[(item['meta']['split'], item['meta']['qid'])] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    return [i['meta'] for i in items.values()], [r for i in items.values() for r in i['records']]
