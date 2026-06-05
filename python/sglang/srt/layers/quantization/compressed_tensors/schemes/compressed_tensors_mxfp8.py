# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)

__all__ = ["CompressedTensorsMxfp8", "CompressedTensorsMxfp8MoE"]

_MXFP8_CONFIG: Optional[Fp8Config] = None


def _get_mxfp8_config() -> Fp8Config:
    global _MXFP8_CONFIG
    if _MXFP8_CONFIG is None:
        _MXFP8_CONFIG = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            use_mxfp8=True,
        )
    return _MXFP8_CONFIG


class CompressedTensorsMxfp8(CompressedTensorsLinearScheme):
    """MXFP8 linear scheme for llm-compressor compressed-tensors checkpoints."""

    def __init__(self):
        from sglang.srt.model_loader.weight_utils import (
            set_mxfp8_weight_scale_remap_enabled,
        )

        set_mxfp8_weight_scale_remap_enabled(True)
        self.fp8_method = Fp8LinearMethod(_get_mxfp8_config())

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self.fp8_method.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def process_weights_after_loading(self, layer) -> None:
        self.fp8_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fp8_method.apply(layer, x, bias=bias)


class CompressedTensorsMxfp8MoE(CompressedTensorsMoEScheme):
    """MXFP8 MoE scheme for llm-compressor compressed-tensors checkpoints."""

    def __init__(self):
        from sglang.srt.model_loader.weight_utils import (
            set_mxfp8_weight_scale_remap_enabled,
        )

        set_mxfp8_weight_scale_remap_enabled(True)
        self.fp8_method = Fp8MoEMethod(_get_mxfp8_config())

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(self, *args, **kwargs):
        return self.fp8_method.create_weights(*args, **kwargs)

    def create_moe_runner(self, layer, moe_runner_config):
        return self.fp8_method.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer) -> None:
        self.fp8_method.process_weights_after_loading(layer)

    def apply_weights(self, layer, dispatch_output):
        return self.fp8_method.apply(layer, dispatch_output)
