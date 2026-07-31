from _init import *

import re


'''
    컨텍스트 내의 공백(띄어쓰기, 줄바꿈 등) 변형은 무시하고, 실제 텍스트의 시작/끝 인덱스 추출
'''
def extract_context_char_spans(prompt: str, contexts: list) -> dict:
    spans = {}

    for i, context in enumerate(contexts):
        words = context.split()

        if not words:
            spans[f'{i}'] = (0, 0)
            continue

        pattern_str = r'\s+'.join([re.escape(w) for w in words])

        # 프롬프트 전체에서 컨텍스트 패턴 검색
        match = re.search(pattern_str, prompt)

        if match:
            spans[f'{i}'] = match.span()
        else:
            spans[f'{i}'] = (0, 0)

    return spans


'''
    토큰의 offset 정보(문자 위치)를 기반으로, 각 문서 구간에 속하는 토큰 인덱스 목록 추출

        - 실제 컨텍스트에 속하는 토큰들을 찾기 위함

    offset_mapping.size(0) == 현재 시퀀스의 토큰 개수 : 922

    # offset_mapping과 context_char_spans의 인덱스는 토큰 위치가 아닌, 실제 char 위치

        - offset_mapping은 토큰의 인덱스와 실제 텍스트의 위치를 맵핑한 것
        - context_char_spans는 실제 텍스트에서 컨텍스트 부분의 위치를 기록한 것
            - key는 컨텍스트 번호

    offset_mapping : tensor([[   0,   17],
                             [  17,   36],
                             [  36,   42],
                             ...,
                             [5055, 5064],
                             [5064, 5081],
                             [5081, 5083]])

    context_char_spans : {'0': (385, 868), '1': (877, 1382), '2': (1391, 1886),
                          '3': (1895, 2376), '4': (2385, 2899), '5': (2908, 3357),
                          '6': (3366, 3865), '7': (3874, 4361), '8': (4370, 4881)}
'''
def mapping_context_tok_idxs(offset_mapping, context_char_spans: dict) -> dict:
    context_tok_idxs = {context_idx: [] for context_idx in context_char_spans.keys()}

    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
        if tok_start == tok_end: # 길이가 0인 특수 토큰
            continue

        for context_idx, (context_start, context_end) in context_char_spans.items():
            # (doc_start <= start) and (end <= doc_end) 이 조건문을 사용하지 않는 이유는 부분 겹침을 고려하기 위함
            if tok_start < context_end and tok_end > context_start:
                context_tok_idxs[context_idx].append(tok_idx)

    return context_tok_idxs


'''
    prompt에서 context들의 토큰 단위 위치를 추출

    offset_mapping
        - prompt(tokenizer.apply_chat_template으로 변환된)로 토크나이징 수행
        - inputs.pop('offset_mapping')으로 가져옴

    context_tok_idxs : {
        '0': [71, ... ,159],
        '1': [164, ... ,250],
        ...
        '8': [800, ... ,884]
    }
'''
def extract_context_tok_idxs(prompt: str, contexts: list, offset_mapping) -> dict:
    context_char_spans = extract_context_char_spans(prompt, contexts)
    context_tok_idxs = mapping_context_tok_idxs(offset_mapping, context_char_spans)

    return context_tok_idxs

