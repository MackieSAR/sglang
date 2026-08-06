import types
import unittest

import torch

from sglang.srt.models.dspark import Qwen3DSparkModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDSparkOptimizedPaths(unittest.TestCase):
    def test_stacked_kv_matches_per_layer_projection(self):
        tokens, hidden_size = 11, 16
        num_layers, num_kv_heads, head_dim = 5, 2, 4
        kv_size = num_kv_heads * head_dim

        class IdentityRope:
            def __call__(self, positions, query, key):
                return query, key

        layers = []
        for _ in range(num_layers):
            attn = types.SimpleNamespace(
                q_size=hidden_size,
                kv_size=kv_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_emb=IdentityRope(),
            )
            attn.qkv_proj = types.SimpleNamespace(
                weight=torch.randn(
                    hidden_size + 2 * kv_size,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                bias=None,
            )
            attn.k_norm = types.SimpleNamespace(
                weight=torch.randn(
                    head_dim, dtype=torch.bfloat16, device="cuda"
                ),
                variance_epsilon=1e-6,
            )
            layers.append(types.SimpleNamespace(self_attn=attn))

        model = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))
        context = torch.randn(
            tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        positions = torch.arange(tokens, device="cuda")
        stacked = Qwen3DSparkModel._stacked_ctx_kv_params(model)
        keys, values = Qwen3DSparkModel._project_ctx_kv_stacked(
            model, context, positions, stacked
        )

        for layer_id, layer in enumerate(layers):
            attn = layer.self_attn
            weight = attn.qkv_proj.weight[
                attn.q_size : attn.q_size + 2 * attn.kv_size
            ]
            key, value = torch.nn.functional.linear(context, weight).split(
                kv_size, dim=-1
            )
            key = key.view(tokens, num_kv_heads, head_dim).float()
            key = key * torch.rsqrt(
                key.square().mean(dim=-1, keepdim=True)
                + attn.k_norm.variance_epsilon
            )
            key = (key * attn.k_norm.weight.float()).bfloat16()
            value = value.view(tokens, num_kv_heads, head_dim)
            torch.testing.assert_close(keys[layer_id], key, rtol=0, atol=0)
            torch.testing.assert_close(values[layer_id], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
