import base64
import json
import mimetypes
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_openai_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
        )


def sample_openai_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[DatasetRow]:
    """
    Load OpenAI-compatible chat completion requests from a JSON or JSONL file.

    Each request should be a JSON object with:
    - "messages": list of {"role": str, "content": str}
    - "max_tokens": int (used as output_len if fixed_output_len not set)
    - "tools": optional list of tool definitions
    - "temperature": optional temperature value
    - "top_p": optional top_p value
    - Other OpenAI API parameters are also extracted and passed through
    """
    dataset_file = Path(dataset_path).expanduser().resolve()
    with dataset_file.open("r", encoding="utf-8") as f:
        raw = f.read()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to the original JSONL format.
        dataset = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        if isinstance(parsed, list):
            dataset = parsed
        elif isinstance(parsed, dict):
            dataset = [parsed]
        else:
            raise ValueError(
                f"OpenAI dataset must contain an object or an array, got "
                f"{type(parsed).__name__}"
            )

    if num_requests > 0:
        dataset = dataset[:num_requests]

    # Fields that should NOT be passed through extra_request_body
    # These are either handled separately or are metadata
    # max_tokens is excluded because it's handled via output_len -> max_completion_tokens
    # max_completion_tokens is also excluded to avoid conflicts
    EXCLUDED_FIELDS = {"messages", "max_tokens", "max_completion_tokens", "model"}

    filtered_dataset: List[DatasetRow] = []
    for data in dataset:
        if not isinstance(data, dict):
            continue

        messages = deepcopy(data.get("messages", []))
        if not messages:
            continue

        # Calibration datasets commonly store the expected response as the final
        # assistant message. It is a completion target, not part of the prompt.
        target_text = ""
        if messages[-1].get("role") == "assistant":
            target = messages.pop().get("content", "")
            if isinstance(target, str):
                target_text = target
            elif isinstance(target, list):
                target_text = "".join(
                    item.get("text", "")
                    for item in target
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        if not messages:
            continue

        # Make relative local image paths portable to the serving process (and
        # remote OpenAI-compatible servers) by embedding them as data URLs.
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    continue
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not url or url.startswith(
                    ("http://", "https://", "data:", "file://")
                ):
                    continue
                image_path = Path(url).expanduser()
                if not image_path.is_absolute():
                    image_path = dataset_file.parent / image_path
                if image_path.is_file():
                    mime_type = (
                        mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream"
                    )
                    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                    image_url["url"] = f"data:{mime_type};base64,{encoded}"

        # Explicit request limits take precedence. Otherwise use the reference
        # answer length when available, matching ShareGPT-style benchmarks.
        if fixed_output_len is not None:
            output_len = fixed_output_len
        elif "max_tokens" in data:
            output_len = data["max_tokens"]
        elif "max_completion_tokens" in data:
            output_len = data["max_completion_tokens"]
        elif target_text:
            output_len = len(tokenizer.encode(target_text))
        else:
            output_len = 256

        # Extract extra request body parameters (tools, temperature, top_p, etc.)
        extra_body = {k: v for k, v in data.items() if k not in EXCLUDED_FIELDS}

        # Calculate prompt length by applying chat template
        # This includes the messages but not the tools
        prompt_len = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

        # If tools are present, we need to add their token count
        # Tools are sent as part of the request and count toward input tokens
        if "tools" in extra_body:
            # Encode tools as JSON string to estimate token count
            tools_str = json.dumps(extra_body["tools"])
            tools_tokens = len(tokenizer.encode(tools_str))
            prompt_len += tools_tokens

        # Pass messages list directly - bench_serving handles List[Dict] prompts
        filtered_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
                extra_request_body=extra_body,  # Store per-request parameters
            )
        )

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(
        "#Input tokens (estimated, before vision expansion): "
        f"{np.sum([x.prompt_len for x in filtered_dataset])}"
    )
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
