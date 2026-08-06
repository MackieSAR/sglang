from dataclasses import dataclass
from typing import Optional


DEFAULT_DSPARK_GAMMA = 7


@dataclass(frozen=True)
class DSparkDraftConfig:
    gamma: int
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    target_layer_ids: tuple[int, ...]


def _first_int(config, names, default=None) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return default


def parse_dspark_draft_config(config) -> DSparkDraftConfig:
    gamma = _first_int(
        config,
        ("dspark_block_size", "block_size", "gamma"),
        DEFAULT_DSPARK_GAMMA,
    )
    mask_token_id = _first_int(
        config,
        ("dspark_noise_token_id", "noise_token_id", "mask_token_id"),
    )
    if mask_token_id is None:
        raise ValueError(
            "DSpark draft config must define dspark_noise_token_id, "
            "noise_token_id, or mask_token_id."
        )
    markov_rank = _first_int(
        config, ("dspark_markov_rank", "markov_rank"), 0
    )
    if gamma < 1:
        raise ValueError(f"DSpark gamma must be positive, got {gamma}.")
    if markov_rank < 1:
        raise ValueError(
            f"DSpark requires a trained Markov head, got markov_rank={markov_rank}."
        )
    layer_ids = getattr(config, "dspark_target_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(config, "target_layer_ids", None)
    if layer_ids is None:
        num_layers = int(getattr(config, "num_hidden_layers", 0))
        layer_ids = (2, num_layers // 2, max(num_layers - 3, 0))
    return DSparkDraftConfig(
        gamma=gamma,
        mask_token_id=mask_token_id,
        markov_rank=markov_rank,
        markov_head_type=str(
            getattr(config, "markov_head_type", "vanilla")
        ).lower(),
        target_layer_ids=tuple(int(x) for x in layer_ids),
    )
