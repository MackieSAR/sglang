"""Dense Qwen3 DSpark draft model for fixed-window speculative decoding.

The draft checkpoint contains a small Qwen3-style block model plus a Markov
head. Token embeddings and the LM head are shared with the target model.
"""

from copy import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)


class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def step(
        self,
        logits: torch.Tensor,
        prev_tokens: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ):
        del hidden_states, state
        return logits + self.markov_w2(self.markov_w1(prev_tokens.long())), None


class GatedMarkov(VanillaMarkov):
    def __init__(self, vocab_size: int, rank: int, hidden_size: int):
        super().__init__(vocab_size, rank)
        self.gate_proj = nn.Linear(hidden_size + rank, rank)

    def step(self, logits, prev_tokens, hidden_states, state):
        del state
        prev = self.markov_w1(prev_tokens.long())
        gate = torch.sigmoid(
            self.gate_proj(torch.cat((hidden_states, prev), dim=-1))
        )
        return logits + self.markov_w2(gate * prev), None


class RNNMarkov(VanillaMarkov):
    def __init__(self, vocab_size: int, rank: int, hidden_size: int):
        super().__init__(vocab_size, rank)
        self.joint_proj = nn.Linear(hidden_size + 2 * rank, 3 * rank)

    def step(self, logits, prev_tokens, hidden_states, state):
        prev = self.markov_w1(prev_tokens.long())
        if state is None:
            state = torch.zeros_like(prev)
        gate, candidate, output = self.joint_proj(
            torch.cat((state, prev, hidden_states), dim=-1)
        ).chunk(3, dim=-1)
        gate = torch.sigmoid(gate)
        state = gate * state + (1 - gate) * torch.tanh(candidate)
        return logits + self.markov_w2(torch.tanh(output)), state


def _build_markov(config, dspark_config):
    args = (int(config.vocab_size), dspark_config.markov_rank)
    if dspark_config.markov_head_type == "vanilla":
        return VanillaMarkov(*args)
    if dspark_config.markov_head_type == "gated":
        return GatedMarkov(*args, int(config.hidden_size))
    if dspark_config.markov_head_type == "rnn":
        return RNNMarkov(*args, int(config.hidden_size))
    raise ValueError(
        f"Unsupported DSpark markov_head_type={dspark_config.markov_head_type!r}."
    )


class Qwen3DSparkModel(Qwen3ForCausalLM):
    """Qwen3 block drafter with target-shared embedding and LM head."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        # Qwen3ForCausalLM normally allocates a full embedding and LM head.
        # DSpark shares both with the target, so initialize tiny placeholders
        # and replace the modules before serving.
        init_config = copy(config)
        init_config.vocab_size = 1
        init_config.tie_word_embeddings = False
        super().__init__(init_config, quant_config=quant_config, prefix=prefix)
        self.config = config
        self.model.config = config
        self.model.vocab_size = int(config.vocab_size)
        self.dspark_config = parse_dspark_draft_config(config)
        self.gamma = self.dspark_config.gamma
        self.markov_head = _build_markov(config, self.dspark_config)
        self.fc = nn.Linear(
            len(self.dspark_config.target_layer_ids) * int(config.hidden_size),
            int(config.hidden_size),
            bias=False,
        )
        self.hidden_norm = RMSNorm(
            int(config.hidden_size), eps=float(config.rms_norm_eps)
        )
        for layer in self.model.layers:
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

    def set_runtime_gamma(self, gamma: int) -> None:
        gamma = int(gamma)
        if gamma < 1:
            raise ValueError(f"DSpark runtime gamma must be positive, got {gamma}.")
        self.gamma = gamma

    def attach_target_modules(self, target_model) -> None:
        self.model.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.lm_head

    def _stacked_ctx_kv_params(self) -> Optional[dict]:
        """Build one dense projection containing every draft layer's K/V rows.

        The injected target context is identical for all draft layers.  Computing
        Q for it is unnecessary, and issuing one GEMM per layer is particularly
        expensive for the small speculative batches.  Dense, unquantized QKV
        projections can therefore be sliced and stacked exactly.
        """
        cached = getattr(self, "_stacked_ctx_kv_cache", False)
        if cached is not False:
            return cached

        weights = []
        biases = []
        k_norm_weights = []
        eps = None
        first_attn = None
        for layer in self.model.layers:
            attn = layer.self_attn
            proj = attn.qkv_proj
            weight = getattr(proj, "weight", None)
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 2
                or weight.shape[0] < attn.q_size + 2 * attn.kv_size
            ):
                self._stacked_ctx_kv_cache = None
                return None
            if first_attn is None:
                first_attn = attn
            elif (
                attn.kv_size != first_attn.kv_size
                or attn.head_dim != first_attn.head_dim
                or attn.num_kv_heads != first_attn.num_kv_heads
                or type(attn.rotary_emb) is not type(first_attn.rotary_emb)
            ):
                self._stacked_ctx_kv_cache = None
                return None

            layer_eps = attn.k_norm.variance_epsilon
            if eps is not None and eps != layer_eps:
                self._stacked_ctx_kv_cache = None
                return None
            eps = layer_eps
            kv_slice = slice(attn.q_size, attn.q_size + 2 * attn.kv_size)
            weights.append(weight[kv_slice])
            bias = getattr(proj, "bias", None)
            biases.append(bias[kv_slice] if bias is not None else None)
            k_norm_weights.append(attn.k_norm.weight)

        has_bias = [bias is not None for bias in biases]
        if any(has_bias) and not all(has_bias):
            self._stacked_ctx_kv_cache = None
            return None
        self._stacked_ctx_kv_cache = {
            "weight": torch.cat(weights, dim=0),
            "bias": torch.cat(biases, dim=0) if all(has_bias) else None,
            "k_norm_weight": torch.stack(k_norm_weights, dim=0).float(),
            "eps": eps,
        }
        return self._stacked_ctx_kv_cache

    def _project_ctx_kv_stacked(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        stacked: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn0 = self.model.layers[0].self_attn
        num_layers = len(self.model.layers)
        tokens = context.shape[0]
        kv_size = attn0.kv_size
        num_kv_heads = attn0.num_kv_heads
        head_dim = attn0.head_dim

        kv = F.linear(context, stacked["weight"], stacked["bias"])
        kv = kv.view(tokens, num_layers, 2, kv_size)
        key_fp32 = (
            kv[:, :, 0]
            .reshape(tokens, num_layers, num_kv_heads, head_dim)
            .float()
        )
        variance = key_fp32.square().mean(dim=-1, keepdim=True)
        key_fp32 = key_fp32 * torch.rsqrt(variance + stacked["eps"])
        key_fp32 = key_fp32 * stacked["k_norm_weight"].view(
            1, num_layers, 1, head_dim
        )
        key = key_fp32.to(context.dtype)

        # Qwen3 layers share the same RoPE configuration.  Treat the layer
        # dimension as additional heads and apply RoPE once.
        key_flat = key.reshape(tokens, num_layers * kv_size)
        dummy_q = torch.empty_like(key_flat)
        _, key_flat = attn0.rotary_emb(positions, dummy_q, key_flat)
        key = (
            key_flat.view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        value = (
            kv[:, :, 1]
            .view(tokens, num_layers, num_kv_heads, head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        return key, value

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding
        hidden = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        customized_info = None
        if getattr(self, "_capture_greedy_proposal", False):
            block_ids = input_ids.view(-1, self.gamma)
            customized_info = {
                "dspark_draft_tokens": self.sample_block_greedy(
                    hidden, block_ids[:, 0]
                )
            }
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden,
            customized_info=customized_info,
        )

    def sample_block(
        self,
        hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
        sampling_info,
    ) -> torch.Tensor:
        if sampling_info.is_all_greedy:
            return self.sample_block_greedy(hidden_states, anchor_tokens)

        bs = anchor_tokens.numel()
        hidden = hidden_states.view(bs, self.gamma, -1)
        lm_weight = self.lm_head.weight
        local_logits = torch.matmul(
            hidden_states.to(dtype=lm_weight.dtype), lm_weight.T
        )
        logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        logits = logits[..., : int(self.config.vocab_size)].view(bs, self.gamma, -1)
        temperatures = sampling_info.temperatures.view(-1).clamp_min(1e-5)
        tokens = []
        prev = anchor_tokens.long()
        state = None
        for step in range(self.gamma):
            step_logits, state = self.markov_head.step(
                logits[:, step], prev, hidden[:, step], state
            )
            # Gumbel-max is distribution-equivalent to multinomial(softmax)
            # and avoids the very slow CUDA multinomial path for Qwen's large
            # vocabulary on newer GPUs.
            exp_noise = torch.empty_like(
                step_logits, dtype=torch.float32
            ).exponential_(1)
            sampled = torch.argmax(
                step_logits.float() / temperatures[:, None] - exp_noise.log(),
                dim=-1,
            )
            greedy = torch.argmax(step_logits, dim=-1)
            prev = torch.where(
                sampling_info.top_ks.view(-1) <= 1, greedy, sampled
            )
            tokens.append(prev)
        return torch.stack(tokens, dim=1)

    def sample_block_greedy(
        self,
        hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-capturable LM-head + Markov greedy proposal."""
        bs = anchor_tokens.numel()
        hidden = hidden_states.view(bs, self.gamma, -1)
        lm_weight = self.lm_head.weight
        local_logits = torch.matmul(
            hidden_states.to(dtype=lm_weight.dtype), lm_weight.T
        )
        logits = tensor_model_parallel_all_gather(local_logits, dim=-1)
        logits = logits[..., : int(self.config.vocab_size)].view(bs, self.gamma, -1)
        tokens = []
        prev = anchor_tokens.long()
        state = None
        for step in range(self.gamma):
            step_logits, state = self.markov_head.step(
                logits[:, step], prev, hidden[:, step], state
            )
            prev = torch.argmax(step_logits, dim=-1)
            tokens.append(prev)
        return torch.stack(tokens, dim=1)

    @torch.no_grad()
    def write_target_hidden_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        pool,
    ) -> None:
        expected = self.fc.in_features
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "DSpark target hidden-state width mismatch: "
                f"expected {expected}, got {tuple(target_hidden.shape)}."
            )
        context = self.hidden_norm(self.fc(target_hidden))
        stacked = self._stacked_ctx_kv_params()
        if stacked is not None:
            keys, values = self._project_ctx_kv_stacked(
                context, positions, stacked
            )
        for layer_id, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if stacked is None:
                _, key, value = attn.forward_prepare_native(positions, context)
            else:
                key, value = keys[layer_id], values[layer_id]
            pool.set_kv_buffer(attn.attn, cache_loc, key, value)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        local_names = {
            "fc.weight",
            "hidden_norm.weight",
            *(
                name
                for name, _ in self.named_parameters()
                if name.startswith("markov_head.")
            ),
        }
        params = dict(self.named_parameters())
        backbone = []
        for name, loaded_weight in weights:
            normalized = name[6:] if name.startswith("model.") else name
            if normalized.startswith(("embed_tokens.", "lm_head.", "confidence_head.")):
                continue
            local_name = normalized
            if local_name in local_names:
                param = params[local_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            else:
                backbone.append((name, loaded_weight))
        super().load_weights(backbone)


EntryClass = [Qwen3DSparkModel]
