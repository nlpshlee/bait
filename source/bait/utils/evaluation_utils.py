from _init import *

import numpy as np

import torch, collections
from transformers import PreTrainedTokenizerFast

from bait.utils import common_utils, tokenizer_utils, model_utils
from bait.core.sft_dataset import SftDataset
from bait.core.sft_trainer import IGNORE_INDEX


def compute_f1(answer: str, predict: str):
    answer_toks = answer.lower().split()
    predict_toks = predict.lower().split()

    common = (collections.Counter(answer_toks) & collections.Counter(predict_toks))
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(predict_toks)
    recall = 1.0 * num_same / len(answer_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def evaluate_sft(model_name_or_path, max_seq_length, max_new_tokens, eval_datas,
                 dtype='bfloat16', device=None, device_map=None, attn_imp=None, ignore_index=IGNORE_INDEX,
                 debug_cnt=-1):

    total_cnt = len(eval_datas)

    if GlobalDebug.EVAL:
        prefix = '# evaluation_utils.evaluate_sft()'
        print(f'{prefix} model_name(or checkpoint_path) : {model_name_or_path}')
        print(f'{prefix} eval_datas size : {total_cnt}')
        print(f'{prefix} start : {common_utils.get_datetime_now()}\n')
        eval_start = common_utils.get_time_ms()

    model = model_utils.get_model(
        model_name_or_path, dtype, device, device_map, attn_imp,
        is_eval=True
    )

    # 학습 시에는 'right', 평가(추론) 시에는 'left'
    tokenizer: PreTrainedTokenizerFast = tokenizer_utils.load_tokenizer(model_name_or_path, 'left')

    # SftDataset 클래스로 변환해줘야 함
    eval_dataset = SftDataset(eval_datas, tokenizer, max_seq_length, ignore_index)

    em_cnt = 0
    start_cnt = 0
    diff_cnt = 0
    f1_scores = []

    with torch.no_grad():
        for i, eval_data in enumerate(eval_dataset):
            input_ids = eval_data['input_ids']
            labels = eval_data['labels']

            source_len = sum(1 for x in labels if x == ignore_index)
            source_ids = input_ids[:source_len]
            target_ids = [x for x in labels if x != ignore_index]

            source_tensor = torch.tensor([source_ids]).to(model.device)
            attention_mask_tensor = torch.tensor([[1] * source_len]).to(model.device)

            outputs = model.generate(
                input_ids=source_tensor,
                attention_mask=attention_mask_tensor,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy 디코딩 (일관된 평가를 위해)
                temperature=None,
                top_p=None
            )

            generated_ids = outputs[0][source_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()

            generated_text = generated_text.lower()
            target_text = target_text.lower()

            if generated_text == target_text:
                em_cnt += 1
            elif generated_text.startswith(target_text):
                start_cnt += 1
            else:
                diff_cnt += 1

                if GlobalDebug.EVAL and diff_cnt <= debug_cnt:
                    print(f'{prefix} gen_text : {generated_text}')
                    print(f'{prefix} tgt_text : {target_text}\n')

            f1_score = compute_f1(target_text, generated_text)
            f1_scores.append(f1_score)

            if GlobalDebug.EVAL:
                if (i+1) % int((total_cnt / 10)) == 0:
                    print(f'{prefix} {i+1} evaluate complet.')

    em_accuracy = (em_cnt / total_cnt) * 100
    start_accuracy = (start_cnt / total_cnt) * 100
    full_cnt = em_cnt + start_cnt
    full_accuracy = (full_cnt / total_cnt) * 100
    f1_avg = np.mean(f1_scores)

    if GlobalDebug.EVAL:
        print(f'\n{model_name_or_path} EM : {em_accuracy:.2f}% ({em_cnt}/{total_cnt})')
        print(f'{model_name_or_path} STARTSWITH : {start_accuracy:.2f}% ({start_cnt}/{total_cnt})')
        print(f'{model_name_or_path} FULL : {full_accuracy:.2f}% ({full_cnt}/{total_cnt})')
        print(f'{model_name_or_path} F1 : {f1_avg:.2f}%\n')

        _, eval_elapsed_str = common_utils.get_elapsed_time_ms(eval_start)
        print(f'{prefix} end : {common_utils.get_datetime_now()}, elapsed : {eval_elapsed_str}\n')

    del model
    del tokenizer
    common_utils.clear_gpu_memory()

    return em_accuracy, start_accuracy, full_accuracy, f1_avg

