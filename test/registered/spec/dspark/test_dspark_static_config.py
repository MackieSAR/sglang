import unittest
from types import SimpleNamespace

from sglang.srt.models.dspark import Qwen3DSparkModel
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class TestDSparkStaticConfig(unittest.TestCase):
    def test_public_qwen3_checkpoint_shape(self):
        config = SimpleNamespace(
            block_size=7,
            mask_token_id=151669,
            markov_rank=256,
            markov_head_type="vanilla",
            target_layer_ids=[1, 10, 19, 28, 37],
        )
        parsed = parse_dspark_draft_config(config)
        self.assertEqual(parsed.gamma, 7)
        self.assertEqual(parsed.mask_token_id, 151669)
        self.assertEqual(parsed.target_layer_ids, (1, 10, 19, 28, 37))

    def test_invalid_static_config(self):
        with self.assertRaisesRegex(ValueError, "gamma"):
            parse_dspark_draft_config(
                SimpleNamespace(
                    block_size=0,
                    mask_token_id=1,
                    markov_rank=1,
                    num_hidden_layers=1,
                )
            )
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            parse_dspark_draft_config(
                SimpleNamespace(
                    block_size=7,
                    markov_rank=1,
                    num_hidden_layers=1,
                )
            )

    def test_algorithm_uses_existing_verify_contract(self):
        algorithm = SpeculativeAlgorithm.from_string("dspark")
        self.assertTrue(algorithm.is_dspark())
        self.assertTrue(algorithm.supports_spec_v2())
        self.assertTrue(algorithm.uses_eagle_verify_input())

    def test_runtime_gamma_can_override_checkpoint_default(self):
        model = object.__new__(Qwen3DSparkModel)
        model.gamma = 7
        model.set_runtime_gamma(4)
        self.assertEqual(model.gamma, 4)
        with self.assertRaisesRegex(ValueError, "positive"):
            model.set_runtime_gamma(0)

if __name__ == "__main__":
    unittest.main()
