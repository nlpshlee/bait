from _init import *

import torch, re, contextlib
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel

from msnap.utils import model_utils, tokenizer_utils


LOG_PREFIX = '# [LOG] model_utils'
ERR_PREFIX = '# [ERROR] model_utils'
ATTN_IMP = 'flash_attention_2'


def get_model(model_name_or_path, dtype, device=None, device_map=None, is_eval=False):
    if (device is not None) and (device_map is not None):
        if DEBUG.ERROR:
            print(f'\n{ERR_PREFIX}.get_model() : device or device_map is not assigned.\n')
        return None

    if device is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=False,
            attn_implementation=ATTN_IMP
        )
        model = model.to(device)

    elif device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, dtype),
            device_map='auto', # accelerator 사용할 때만
            trust_remote_code=False,
            attn_implementation=ATTN_IMP
        )

    if is_eval:
        model.eval()
    
    if DEBUG.LOG:
        print(f'\n{LOG_PREFIX}.get_model() : {model_name_or_path}\n')

    return model


def merge_and_save(model_name, dtype, adapter_path, save_path, device_map='auto'):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=ATTN_IMP
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path, safe_serialization=True)

    # 패딩 방향(left/right)은 상관 없음
    tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(adapter_path)
    tokenizer.save_pretrained(save_path)

    if DEBUG.LOG:
        print(f'\n{LOG_PREFIX}.merge_and_save() model load : {model_name}')
        print(f'{LOG_PREFIX}.merge_and_save() adapter merge : {adapter_path}')
        print(f'{LOG_PREFIX}.merge_and_save() merged model save : {save_path}\n')


def make_inputs(tokenizer: PreTrainedTokenizerFast, device: str, prompts, max_seq_length):
    chat_prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        chat_prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )

    return inputs.to(device)


def forward(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str,
            prompts, max_seq_length, output_hidden_states=True):

    inputs = model_utils.make_inputs(tokenizer, device, prompts, max_seq_length)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=output_hidden_states)
    
    return outputs


def generate(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str,
             prompts: list, max_seq_length: int, max_new_tokens: int,
             do_sample=False, temperature=None, top_k=None, top_p=None,
             target_layer_idxs: list=None, all_layer_misinfo_vecs: dict=None, alpha=1.0,
             return_all=False):
    
    inputs = model_utils.make_inputs(tokenizer, device, prompts, max_seq_length)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if (target_layer_idxs is not None) and (all_layer_misinfo_vecs is not None):
        intervention_context = apply_null_space_projection(model, device, target_layer_idxs, all_layer_misinfo_vecs, alpha)
    else:
        intervention_context = contextlib.nullcontext()

    with intervention_context:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            # outputs 은 .to(device) 할 필요 없음, input_ids 와 같은 device 에 자동 할당됨
    
    if return_all:
        return input_ids, outputs
    else:
        return outputs


def get_generated_texts(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, device: str, 
                        prompts: list, max_seq_length: int, max_new_tokens: int, 
                        do_sample=False, temperature=None, top_k=None, top_p=None,
                        target_layer_idxs: list=None, all_layer_misinfo_vecs: dict=None, alpha=1.0):

    input_ids, outputs = generate(
        model, tokenizer, device,
        prompts, max_seq_length, max_new_tokens,
        do_sample, temperature, top_k, top_p,
        target_layer_idxs, all_layer_misinfo_vecs, alpha,
        return_all=True
    )

    generated_texts = []
    for i, output in enumerate(outputs):
        prompt_length = input_ids[i].shape[0]
        generated_token_ids = output[prompt_length:]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_texts.append(generated_text.strip())
    
    return generated_texts


def is_correct(generated_text: str, answer: str):
    generated_text = generated_text.lower().strip(' ,.!?~')
    answer = answer.lower().strip(' ,.!?~')

    is_exact = True if generated_text == answer else False

    if is_exact:
        return [True, True]

    check_pat =  re.compile(rf'\b{re.escape(answer)}\b')
    is_contains = bool(check_pat.search(generated_text))

    return [False, is_contains]


def get_null_space_hook(misinfo_vecs: torch.Tensor, alpha: float):
    '''
        히든 벡터에서 취약성 공간(misinfo_vecs) 성분을 깎아내는 포워드 훅 함수
            - misinfo_vecs shape: (k, hidden_size)  (예: PC1 1개만 쓰면 (1, 3072))
    '''
    def hook(module, input, output):
        # h shape: (batch_size, sequence_length, hidden_size)
        hs = output[0]

        '''
            1번째 스텝은 입력이기 때문에 seq_len >>> 1
            2번째 스텝 부터는 디코딩 과정이기 때문에 seq_len == 1

            영공간 투영은 전체 입력의 마지막 토큰 또는 디코딩 토큰에만 적용되어야 함
            즉, 마지막 토큰에만 반영하면 됨
        '''
        hs_target = hs[:, -1:, :]

        # misinfo_vecs를 현재 히든 벡터와 동일한 디바이스 및 데이터 타입으로 맞춤
        misinfo_vecs_proj = misinfo_vecs.to(device=hs_target.device, dtype=hs_target.dtype)

        # 1. 투영 점수(Score) 계산 : hs 내적 vs^T
        # shape: (B, 1, D) @ (D, k) -> (B, 1, k)
        scores = torch.matmul(hs_target, misinfo_vecs_proj.T)

        '''
            적응형(Adaptive) 제어
                - 만약 PC1이 오답 방향을 향하고 있다면, scores가 양수일 때만(오답에 끌려갈 때만) 개입
                    - 무조건 깎아내려면 아래 줄을 주석 처리
                    - 선택적으로 깎으려면 주석 해제
        '''
        scores = torch.relu(scores)

        # 2. 오염된 성분 벡터 복원
        # shape: (B, 1, k) @ (k, D) -> (B, 1, D)
        pollution_vecs = torch.matmul(scores, misinfo_vecs_proj)

        # 3. 영공간 투영 (빼기)
        projected_target_vecs = hs_target - (alpha * pollution_vecs)

        # 4. 수정된 마지막 토큰을 원본 hs 구조에 덮어씌우기
        '''
            inplace 연산 에러를 방지하기 위해 clone() 사용 권장
                - 원본 hs 텐서의 특정 인덱스에 직접 값을 덮어씌우면 모델의 역전파나 캐시 메모리가 꼬이는 In-place operation 에러가 발생할 수 있음
        '''
        hs_new = hs.clone()
        hs_new[:, -1:, :] = projected_target_vecs

        # 수정된 텐서를 다시 튜플로 포장하여 다음 레이어로 전달
        return (hs_new,) + output[1:]

    return hook


@contextlib.contextmanager
def apply_null_space_projection(model, device, target_layer_idxs: list, all_layer_misinfo_vecs: dict, alpha: float):
    handles = []

    try:
        for target_layer_idx in target_layer_idxs:
            target_layer = model.model.layers[target_layer_idx]
            target_misinfo_vecs = torch.tensor(all_layer_misinfo_vecs[target_layer_idx], dtype=torch.float32).to(device)

            # 현재 레이어의 맞춤형 훅 생성 및 등록
            hook_fn = get_null_space_hook(target_misinfo_vecs, alpha)
            handle = target_layer.register_forward_hook(hook_fn)
            handles.append(handle)

        # 추론 시작 : 여기서 함수가 잠시 멈추고, 메인 코드의 with 블록 안으로 들어가 model.generate() 실행
        yield
    finally:
        # 추론이 끝나면 모델 원상 복구 (매우 중요)
        for handle in handles:
            handle.remove()

