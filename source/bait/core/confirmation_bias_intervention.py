from _init import *

from typing import List
from transformers import AutoModelForCausalLM

from bait.utils import common_utils
from bait.core.bait_utils import INTERVENTION_OPTION


class ConfirmationBiasIntervention:
    def __init__(self, model: AutoModelForCausalLM, intervention_option=INTERVENTION_OPTION.OFF, target_layers='all'):
        self._model: AutoModelForCausalLM = model
        self._intervention_option = intervention_option
        self._hooks = []
        self._target_tok_idxs_list = []

        # 모델의 전체 레이어 개수 파악
        num_layers = self._model.config.num_hidden_layers
        if target_layers == 'all':
            self._target_layer_idxs = list(range(num_layers))
        else:
            # 음수 인덱스 지원 (예: [-1, -2])
            self._target_layer_idxs = [(l if l >= 0 else num_layers + l) for l in target_layers]


    def set_target_toks(self, target_tok_idxs_list: List[List[int]]):
        if self._intervention_option != INTERVENTION_OPTION.OFF:
            self._target_tok_idxs_list = target_tok_idxs_list


    '''
        register_hooks() target_layer : LlamaDecoderLayer(
        (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
            (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
            (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
            (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
        )
        (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
            (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
            (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
            (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        )

        [첫번째는 전체 입력, 2~3번째는 토큰 생성 부분]
        _ablation_hook() [Linear(in_features=3072, out_features=3072, bias=False)] hidden_states shape : torch.Size([50, 1041, 3072])
        _ablation_hook() [Linear(in_features=3072, out_features=3072, bias=False)] hidden_states shape : torch.Size([50, 1, 3072])
        _ablation_hook() [Linear(in_features=3072, out_features=3072, bias=False)] hidden_states shape : torch.Size([50, 1, 3072])
        _ablation_hook() [Linear(in_features=3072, out_features=3072, bias=False)] hidden_states shape : torch.Size([50, 1, 3072])
    '''
    def _ablation_hook(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output

        for idx, target_tok_idxs in enumerate(self._target_tok_idxs_list):
            if not target_tok_idxs:
                continue

            valid_target_tok_idxs = [target_tok_idx for target_tok_idx in target_tok_idxs if target_tok_idx < hidden_states.shape[1]]

            if valid_target_tok_idxs:
                hidden_states[idx, valid_target_tok_idxs, :] = 0.0

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


    def register_hooks(self):
        if self._intervention_option != INTERVENTION_OPTION.OFF:
            for target_layer_idx in self._target_layer_idxs:
                target_layer = self._model.model.layers[target_layer_idx]

                if common_utils.check_option(self._intervention_option, INTERVENTION_OPTION.MLP):
                    self._hooks.append(target_layer.mlp.register_forward_hook(self._ablation_hook))

                if common_utils.check_option(self._intervention_option, INTERVENTION_OPTION.ATTN):
                    self._hooks.append(target_layer.self_attn.o_proj.register_forward_hook(self._ablation_hook))


    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()

        self._hooks.clear()


    def __enter__(self):
        self.register_hooks()
        return self


    def __exit__(self, exc_type, exc, tb):
        self.remove_hooks()

