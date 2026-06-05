#!/usr/bin/env python3
"""Quantize Qwen3-30B-A3B to MXFP8 with llmcompressor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = "/data/home/cgxu2/models/Qwen3-14B-modelscope"
SAVE_DIR = "/data/home/cgxu2/models/Qwen3-14B-modelscope-DYNAMIC_FP8"
SHAREGPT_PATH = "/data/home/cgxu2/data/sharegpt_CN.json"
MAX_NEW_TOKENS = 128
TRUST_REMOTE_CODE = True
RUN_GENERATION_CHECK = True


def load_json_or_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from (x for x in json.load(f) if isinstance(x, dict))
        else:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item


def get_text(item: dict[str, Any]) -> str:
    conversations = item.get("conversations") or item.get("messages") or []
    if not isinstance(conversations, list):
        return ""

    for msg in conversations:
        role = str(msg.get("from", msg.get("role", ""))).lower()
        if role not in {"human", "user", "prompter"}:
            continue
        text = str(msg.get("value", msg.get("content", ""))).strip()
        if 10 <= len(text) <= 500 and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return text
    return ""


def pick_sharegpt_prompt(path: str) -> str:
    sharegpt_path = Path(path)
    if not sharegpt_path.exists():
        return "请用三句话解释一下大语言模型量化的作用。"

    for item in load_json_or_jsonl(sharegpt_path):
        text = get_text(item)
        if text:
            return text
    return "请用三句话解释一下大语言模型量化的作用。"


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*gate$",
            "re:.*router$",
        ],
    )

    oneshot(model=model, recipe=recipe)

    if RUN_GENERATION_CHECK:
        print("========== SAMPLE GENERATION ==============")
        dispatch_model(model)
        prompt = pick_sharegpt_prompt(SHAREGPT_PATH)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
        print("==========================================")

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved MXFP8 model to {SAVE_DIR}")


if __name__ == "__main__":
    main()
