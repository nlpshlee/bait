"""Dummy-specific Llama interventions and one-backward candidate ranking.

Signed retention-gate derivatives rank all MLP layers, attention heads, and
residual bypasses using the model's own answer. They are local influence scores,
not a causal confirmation-bias estimate. Ranking keeps the native attention
backend and never materializes all prompt-by-prompt attention matrices.
"""

from contextlib import contextmanager, nullcontext
import math
from types import MethodType

import torch
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def decoder_layers(model):
    """The tested implementation is transformers 5.16 Llama."""
    unwrap = getattr(model, "get_base_model", None)
    base = unwrap() if callable(unwrap) else model
    if getattr(base.config, "model_type", None) != "llama":
        raise ValueError("Interventions currently support Llama causal language models only.")
    layers = getattr(getattr(base, "model", None), "layers", None)
    if layers is None or not all(isinstance(layer.self_attn, LlamaAttention) for layer in layers):
        raise TypeError("Expected model.model.layers with LlamaAttention modules.")
    return layers


@contextmanager
def _frozen_model(model):
    """Keep the activation graph without allocating parameter gradients."""
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    training = [(module, module.training) for module in model.modules()]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        model.eval()
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, flag in training:
            module.training = flag


def _query_positions(hidden, kwargs, layer_idx):
    # Offsets name physical padded columns, whereas RoPE position_ids omit
    # left padding. Cache columns retain that padding during generation.
    cache = kwargs.get("past_key_values")
    start = cache.get_seq_length(layer_idx) if cache is not None else 0
    return torch.arange(start, start + hidden.shape[1], device=hidden.device)


class Intervention:
    """Temporarily suppress selected dummy-specific paths; restore on exit.

    MLP scales its update at dummy token positions. Residual intervention scales
    the layer-input bypass at those positions, preserving attention/MLP updates.
    MHA scales selected heads' edges from queries after the dummy into dummy
    keys. Optional attention_renorm renormalizes these rows after suppression;
    the default preserves the original unnormalized intervention.
    These paths interact; their scores are not an additive causal decomposition.

    The native attention backend supplies the base output. Only affected query
    rows and selected heads are recomputed to add their changed-edge contribution.
    dummy_tokens is a shared flat list or one list per batch item, always in
    physical padded input columns (the same indices as tokenizer offsets).
    `_gates` is private to identity-point ranking, not an inference option.
    """

    def __init__(self, model, *, dummy_tokens=(),
                 mlp_layers=(), heads=None, residual_layers=(), retain=0.5,
                 attention_renorm=False, _gates=None):
        self.model = model
        self.layers = decoder_layers(model)
        tokens = list(dummy_tokens)
        rows = tokens if tokens and isinstance(tokens[0], (list, tuple)) else [tokens]
        self.dummy_tokens = tuple(tuple(sorted(set(map(int, row)))) for row in rows)
        self._has_dummy = any(self.dummy_tokens)
        self._dummy_mask = self._last_dummy = None
        self.mlp_layers = set(mlp_layers)
        self.heads = {int(layer): tuple(sorted(set(indices)))
                      for layer, indices in (heads or {}).items()}
        self.residual_layers = set(residual_layers)
        self.retain = float(retain)
        self.attention_renorm = bool(attention_renorm)
        self._gates = _gates
        selected = self.mlp_layers | self.residual_layers | set(self.heads)
        if any(index < 0 or index >= len(self.layers) for index in selected):
            raise ValueError("Layer indices must be valid zero-based Llama layer indices.")
        if not 0 <= self.retain <= 1:
            raise ValueError("retain must be in [0, 1].")
        if any(t < 0 for row in self.dummy_tokens for t in row):
            raise ValueError("Token positions must be nonnegative.")
        for layer, indices in self.heads.items():
            attn = self.layers[layer].self_attn
            n_heads = attn.q_proj.out_features // attn.head_dim
            if any(head < 0 or head >= n_heads for head in indices):
                raise ValueError(f"Invalid attention head in layer {layer}.")
        self._positions = {}
        self._layer_inputs = {}
        self._hooks = []
        self._forwards = []
        self._active = False

    def _dummy_masks(self, hidden):
        if len(self.dummy_tokens) not in (1, hidden.shape[0]):
            raise ValueError("dummy_tokens must have one row per batch item or one shared row.")
        if self._dummy_mask is None or self._dummy_mask.device != hidden.device:
            last = [row[-1] if row else -1 for row in self.dummy_tokens]
            self._last_dummy = torch.tensor(last, device=hidden.device)
            self._dummy_mask = torch.zeros((len(last), max(max(last) + 1, 1)),
                                           dtype=torch.bool, device=hidden.device)
            for index, row in enumerate(self.dummy_tokens):
                self._dummy_mask[index, list(row)] = True
        return self._dummy_mask, self._last_dummy

    def _dummy_rows(self, index, hidden):
        positions = self._positions[index]
        mask, _ = self._dummy_masks(hidden)
        return (mask[:, positions.clamp(max=mask.shape[1] - 1)]
                & (positions < mask.shape[1])).unsqueeze(-1)

    def _before_layer(self, index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self._positions[index] = _query_positions(hidden, kwargs, index)
            if index in self.residual_layers:
                self._layer_inputs[index] = hidden
        return hook

    def _after_mlp(self, index):
        def hook(module, args, output):
            if self._gates is not None:
                # At gate=1 the detached term has exactly the same first
                # derivative as output * gate, with no redundant gradient path.
                rows = torch.nonzero(self._dummy_rows(index, output).flatten()).flatten()
                contribution = output.detach().index_select(1, rows).float()
                delta = (self._gates["mlp"][index] - 1) * contribution
                return output + torch.zeros_like(output).index_copy(1, rows, delta.to(output.dtype))
            if self._has_dummy and self.retain != 1:
                return output * (1 - self._dummy_rows(index, output).to(output.dtype)
                                 * (1 - self.retain))
            return output
        return hook

    def _after_layer(self, index):
        def hook(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected transformers 5.16 LlamaDecoderLayer tensor output.")
            layer_input = self._layer_inputs.pop(index)
            if self._gates is not None:
                rows = torch.nonzero(self._dummy_rows(index, output).flatten()).flatten()
                contribution = layer_input.detach().index_select(1, rows).float()
                delta = (self._gates["residual"][index] - 1) * contribution
                return output + torch.zeros_like(output).index_copy(1, rows, delta.to(output.dtype))
            if self._has_dummy and self.retain != 1:
                return output - ((1 - self.retain) * layer_input
                                 * self._dummy_rows(index, output).to(output.dtype))
            return output
        return hook

    def _attention_forward(self, index, original_forward):
        owner = self

        def forward(module, hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
            positions = _query_positions(hidden_states,
                                         dict(kwargs, past_key_values=past_key_values), index)
            dummy_mask, last_dummy = owner._dummy_masks(hidden_states)
            later_queries = ((positions[None, :] > last_dummy[:, None])
                             & (last_dummy[:, None] >= 0))
            active = (owner.heads.get(index) and owner._has_dummy
                      and (owner.retain != 1 or owner._gates is not None)
                      and bool(later_queries.any()))
            if not active:
                return original_forward(hidden_states=hidden_states,
                                        position_embeddings=position_embeddings,
                                        attention_mask=attention_mask,
                                        past_key_values=past_key_values, **kwargs)
            if module.training and module.attention_dropout:
                raise ValueError("Attention interventions require eval mode or zero dropout.")
            input_shape = hidden_states.shape[:-1]
            shape = (*input_shape, -1, module.head_dim)
            query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
            key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
            value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
            if position_embeddings is None:
                raise ValueError("Llama rotary position embeddings were not provided.")
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            if past_key_values is not None:
                key, value = past_key_values.update(key, value, module.layer_idx)
            interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                module.config._attn_implementation, eager_attention_forward)
            output, _ = interface(module, query, key, value, attention_mask,
                                  dropout=0., scaling=module.scaling, **kwargs)
            heads = torch.tensor(owner.heads[index], device=query.device)
            rows = torch.nonzero(later_queries.any(dim=0)).flatten()
            # At identity, d[(gate-1)*contribution]/d(previous activations)=0.
            # Detaching here therefore preserves every first derivative while
            # avoiding a second differentiable attention graph for ranking.
            with torch.no_grad() if owner._gates is not None else nullcontext():
                selected_query = query.index_select(1, heads).index_select(2, rows)
                kv_heads = heads // module.num_key_value_groups
                selected_key = key.index_select(1, kv_heads)
                selected_value = value.index_select(1, kv_heads)
                logits = (selected_query @ selected_key.transpose(2, 3)) * module.scaling
                key_positions = torch.arange(key.shape[-2], device=query.device)
                causal = key_positions.view(1, -1) <= positions[rows].view(-1, 1)
                logits = logits.masked_fill(~causal.view(1, 1, *causal.shape), float("-inf"))
                if attention_mask is not None:
                    mask = attention_mask[..., :key.shape[-2]]
                    if mask.ndim == 2:
                        # FlashAttention receives the original 0/1 padding
                        # mask; SDPA/eager normally receive a 4-D causal mask.
                        mask = mask.to(torch.bool)[:, None, None, :]
                    elif mask.ndim != 4:
                        raise ValueError("Expected a two- or four-dimensional attention mask or None.")
                    if mask.shape[1] != 1:
                        mask = mask.index_select(1, heads)
                    if mask.shape[2] != 1:
                        mask = mask.index_select(2, rows)
                    if mask.dtype == torch.bool:
                        logits = logits.masked_fill(~mask, float("-inf"))
                    else:
                        logits = logits + mask
                if owner.attention_renorm and owner._gates is None:
                    # softmax(z + log(retain) on dummy keys) == gated weights
                    # divided by their sum. Masking at zero avoids log(0), and
                    # working in logit space avoids cancellation when mass≈1.
                    is_dummy = (dummy_mask[:, key_positions.clamp(max=dummy_mask.shape[1] - 1)]
                                & (key_positions < dummy_mask.shape[1]))[:, None, None, :]
                    visible_dummy = (is_dummy & (logits > torch.finfo(logits.dtype).min)).any(-1, keepdim=True)
                    logits = logits.float()
                    logits = (logits.masked_fill(is_dummy, float('-inf')) if owner.retain == 0
                              else logits + is_dummy * math.log(owner.retain))
                    weights = torch.softmax(logits, dim=-1).nan_to_num(nan=0.).to(query.dtype)
                    replacement = weights @ selected_value
                    previous = output.index_select(1, rows).transpose(1, 2).index_select(1, heads)
                    delta = (replacement - previous) * later_queries[:, None, rows, None] * visible_dummy
                else:
                    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
                    weights = torch.nan_to_num(weights, nan=0.)
                    dummy_keys = torch.nonzero(dummy_mask.any(dim=0)).flatten()
                    dummy_keys = dummy_keys[dummy_keys < key.shape[-2]]
                    dummy_weights = (weights.index_select(3, dummy_keys)
                                     * dummy_mask[:, None, None, dummy_keys])
                    contribution = (dummy_weights
                                    @ selected_value.index_select(2, dummy_keys))
                    if owner.attention_renorm:
                        # At gate=1, d[(o+(g-1)c)/(1+(g-1)m)]/dg = c-m*o.
                        previous = output.detach().index_select(1, rows).transpose(1, 2).index_select(1, heads)
                        contribution = contribution.float() - dummy_weights.float().sum(-1, keepdim=True) * previous.float()
                    contribution = contribution * later_queries[:, None, rows, None]
            if owner._gates is not None:
                change = owner._gates["heads"][index].index_select(0, heads) - 1
                delta = (contribution.float() * change.view(1, -1, 1, 1)).to(output.dtype)
            elif not owner.attention_renorm:
                delta = contribution * (owner.retain - 1)
            changed_heads = output.new_zeros((output.shape[0], query.shape[1],
                                             rows.numel(), module.head_dim))
            changed_heads = changed_heads.index_copy(1, heads, delta)
            correction = torch.zeros_like(output).index_copy(1, rows, changed_heads.transpose(1, 2))
            output = module.o_proj((output + correction).reshape(*input_shape, -1).contiguous())
            return output, None
        return forward

    def __enter__(self):
        if self._active or getattr(self.model, "_bait_intervention_active", False):
            raise RuntimeError("Nested Intervention contexts on the same model are unsupported.")
        self._active = True
        self.model._bait_intervention_active = True
        try:
            for index in sorted(self.mlp_layers | self.residual_layers):
                layer = self.layers[index]
                self._hooks.append(layer.register_forward_pre_hook(
                    self._before_layer(index), with_kwargs=True))
                if index in self.mlp_layers:
                    self._hooks.append(layer.mlp.register_forward_hook(self._after_mlp(index)))
                if index in self.residual_layers:
                    self._hooks.append(layer.register_forward_hook(self._after_layer(index)))
            for index, heads in sorted(self.heads.items()):
                if not heads or not self._has_dummy or (self.retain == 1 and self._gates is None):
                    continue
                module = self.layers[index].self_attn
                had_local = "forward" in module.__dict__
                old = module.__dict__.get("forward")
                self._forwards.append((module, had_local, old))
                module.forward = MethodType(self._attention_forward(index, module.forward), module)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in reversed(self._hooks):
            hook.remove()
        for module, had_local, old in reversed(self._forwards):
            if had_local:
                module.forward = old
            else:
                del module.forward
        self._hooks.clear()
        self._forwards.clear()
        self._layer_inputs.clear()
        self._positions.clear()
        self._active = False
        if hasattr(self.model, "_bait_intervention_active"):
            del self.model._bait_intervention_active


def _ids(model, input_ids):
    device = model.get_input_embeddings().weight.device
    result = torch.as_tensor(input_ids, dtype=torch.long, device=device).clone()
    if result.ndim == 1:
        result = result.unsqueeze(0)
    if result.ndim != 2 or result.shape[0] != 1 or result.shape[1] == 0:
        raise ValueError("Expected a nonempty token sequence with batch size one.")
    return result


def rank_dummy_sites(model, input_ids, dummy_tokens, target_ids):
    """One backward returns d(mean answer logP)/d(retention) at retention=1.

    The answer must be the model's saved zero-shot answer, not evaluation gold.
    Positive scores predict that suppression lowers that answer's logprob.
    Attention derivatives concern dummy-key edges from later query positions;
    MLP and residual derivatives concern only dummy-token positions. All paths
    are ranked jointly in the unchanged model, with one scalar per layer/path
    and one scalar per attention head. Actual interventions must validate them.
    """
    layers = decoder_layers(model)
    with torch.inference_mode(False), torch.enable_grad(), _frozen_model(model):
        prefix = _ids(model, input_ids)
        target = _ids(model, target_ids)
        dummy = tuple(sorted(set(int(token) for token in dummy_tokens)))
        if not dummy or dummy[0] < 0 or dummy[-1] >= prefix.shape[1]:
            raise ValueError("dummy_tokens must name positions inside the input prompt.")
        count = len(layers)
        device = prefix.device
        gates = {"mlp": torch.ones(count, device=device, requires_grad=True),
                 "residual": torch.ones(count, device=device, requires_grad=True),
                 "heads": [torch.ones(layer.self_attn.q_proj.out_features // layer.self_attn.head_dim,
                                      device=device, requires_grad=True) for layer in layers]}
        heads = {index: tuple(range(gate.numel())) for index, gate in enumerate(gates["heads"])}
        sequence = torch.cat((prefix, target[:, :-1]), dim=-1)
        with Intervention(model, dummy_tokens=dummy, mlp_layers=range(count),
                          residual_layers=range(count), heads=heads, retain=1, _gates=gates):
            outputs = model(input_ids=sequence, use_cache=False, logits_to_keep=target.shape[1])
            objective = outputs.logits.float().log_softmax(-1).gather(-1, target.unsqueeze(-1)).mean()
            tensors = [gates["mlp"], gates["residual"], *gates["heads"]]
            gradients = torch.autograd.grad(objective, tensors, allow_unused=True)
        gradients = [torch.zeros_like(gate) if gradient is None else gradient
                     for gate, gradient in zip(tensors, gradients)]
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("Nonfinite gate derivatives; no site ranking was produced.")
        return {"mlp": gradients[0].detach().cpu().tolist(),
                "residual": gradients[1].detach().cpu().tolist(),
                "heads": [gradient.detach().cpu().tolist() for gradient in gradients[2:]]}
