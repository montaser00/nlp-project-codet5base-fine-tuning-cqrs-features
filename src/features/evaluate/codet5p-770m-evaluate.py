"""
CodeT5 C# CQRS inference script.

Loads a fine-tuned CodeT5+ model with an optional LoRA adapter and generates
C# CQRS boilerplate files from natural-language prompts.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import TypedDict

import torch
import transformers
from codebleu import calc_codebleu
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# ANSI colour helpers
# ──────────────────────────────────────────────────────────────

_ANSI_ENABLED: bool = sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))


def _colorize(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI_ENABLED else text


def cyan(text: str) -> str:     return _colorize(text, "96")
def green(text: str) -> str:    return _colorize(text, "92")
def yellow(text: str) -> str:   return _colorize(text, "93")
def red(text: str) -> str:      return _colorize(text, "91")
def bold(text: str) -> str:     return _colorize(text, "1")
def dim(text: str) -> str:      return _colorize(text, "2")
def magenta(text: str) -> str:  return _colorize(text, "95")


# ──────────────────────────────────────────────────────────────
# Tokenizer compatibility patch  (transformers ≥ 4.48 Rust backend)
# ──────────────────────────────────────────────────────────────

def _patch_tokenizer_backend() -> None:
    """Fix AddedToken compatibility between the Python and Rust tokenizer backends."""
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        from tokenizers import AddedToken as RustAddedToken

        def _add_tokens_fixed(self, new_tokens: list, special_tokens: bool = False) -> int:
            converted = []
            for token in new_tokens:
                if isinstance(token, (str, RustAddedToken)):
                    converted.append(token)
                elif hasattr(token, "content"):
                    kwargs = dict(
                        single_word=getattr(token, "single_word", False),
                        lstrip=getattr(token, "lstrip", False),
                        rstrip=getattr(token, "rstrip", False),
                        normalized=getattr(token, "normalized", True),
                    )
                    try:
                        converted.append(RustAddedToken(str(token.content), special=special_tokens, **kwargs))
                    except Exception:
                        converted.append(RustAddedToken(str(token.content), **kwargs))
                else:
                    converted.append(str(token))

            return (
                self._tokenizer.add_special_tokens(converted)
                if special_tokens
                else self._tokenizer.add_tokens(converted)
            )

        TokenizersBackend._add_tokens = _add_tokens_fixed

    except ImportError:
        pass


_patch_tokenizer_backend()

# ──────────────────────────────────────────────────────────────
# Loading Testing Data
# ──────────────────────────────────────────────────────────────

with open("./content/test_data.json", "r", encoding="utf-8") as f:
    TEST_DATA = json.load(f)

SAMPLE_DATA = {
    "test": TEST_DATA
}

# ──────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────

class GenerationConfig(TypedDict):
    max_new_tokens: int
    num_beams: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_MODEL_PATH   = os.path.abspath("./content/codet5p-770m")
_ADAPTER_PATH = "./content/output/lora_adapter"
_OUTPUT_PATH  = os.path.abspath("./content/output")

_DEFAULT_GENERATION_CONFIG: GenerationConfig = {
    "max_new_tokens":     2048,
    "num_beams":          4,
    "do_sample":          False,
    "temperature":        1.0,
    "top_p":              1.0,
    "repetition_penalty": 1.2,
}

# ──────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────

def load_model(model_name: str, adapter_path: str, is_base: bool) -> tuple:
    print(bold("\n  Loading tokenizer ..."))
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(bold("  Loading base model ..."))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if not os.path.isdir(adapter_path) or is_base:
        print(yellow(f"\n  [WARN] Adapter not found at: {adapter_path}"))
        print(yellow("         Continuing with base model (output quality will be low).\n"))
    else:
        print(bold(f"  Loading LoRA adapter: {adapter_path}"))
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(green("  Adapter merged — fine-tuned model ready.\n"))

    if device == "cpu":
        model = model.to(device)

    model.eval()
    return model, tokenizer, device


# ──────────────────────────────────────────────────────────────
# Generation — generates until EOS
# ──────────────────────────────────────────────────────────────

def generate(model, tokenizer, device: str, prompt: str, config: GenerationConfig) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=12000,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            num_beams=config["num_beams"],
            do_sample=config["do_sample"],
            temperature=config["temperature"] if config["do_sample"] else 1.0,
            top_p=config["top_p"]             if config["do_sample"] else 1.0,
            repetition_penalty=config["repetition_penalty"],
            early_stopping=False,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_output


# ──────────────────────────────────────────────────────────────
# Quantitative Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_codebleu(predictions: list[str], actuals: list[str], lang: str = "c_sharp") -> dict:
    result = calc_codebleu(actuals, predictions, lang=lang, weights=(0.25, 0.25, 0.25, 0.25))
    return result

# ──────────────────────────────────────────────────────────────
# Test prompts runner
# ──────────────────────────────────────────────────────────────

def run_test_prompts(model, tokenizer, device: str, is_base: bool) -> None:
    print("\n" + "=" * 62)
    print(bold(cyan("   CodeT5 C# CQRS Evaluation  ") + dim("(fine-tuned)") if not is_base else ''))
    print(f"   Model      : {_MODEL_PATH}")
    print(f"   Adapter    : {os.path.abspath(_ADAPTER_PATH)}")
    print(f"   Output dir : {_OUTPUT_PATH}")
    print(f"   Device     : {device}")
    print(f"   transformers {transformers.__version__}")
    print("=" * 62)

    print(dim("  Generating ..."))
    predictions = list()
    sample_index = 1

    for prompt in SAMPLE_DATA['test']:

        try:
            print(prompt['input'])
            output = generate(model, tokenizer, device, prompt['input'], _DEFAULT_GENERATION_CONFIG)
            predictions.append(output)
            print(sample_index, " is Done.")
            sample_index += 1
        except Exception as e:
            print(red(f"  Error: {e}\n"))

        if not output.strip():
            print(yellow(
                "  [WARN] Output was empty after cleaning.\n"
                "  Make sure the adapter path is correct and the model finished training.\n"
            ))

        if sample_index % 3 == 0:
            print('Hit Enter To Continue.')
            enter = input()

    print('Base: ' if is_base else 'Fine-Tuned', evaluate_codebleu(predictions, SAMPLE_DATA['validation']['output'], lang="C#"))

# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    if False:
        model, tokenizer, device = load_model(_MODEL_PATH, _ADAPTER_PATH, is_base = True)
        run_test_prompts(model, tokenizer, device, is_base= True)

        model, tokenizer, device = load_model(_MODEL_PATH, _ADAPTER_PATH, is_base = False)
        run_test_prompts(model, tokenizer, device, is_base= False)

    actuals = SAMPLE_DATA['test']
    actuals = [item['output'] for item in actuals]

    with open('./content/base_predictions.json', 'r') as f:
        predictions = json.load(f)
        print(evaluate_codebleu(predictions, [[a] for a in actuals], lang="c_sharp"))

    with open('./content/fine-tuned_predictions.json', 'r') as f:
        predictions = json.load(f)
        print(evaluate_codebleu(predictions, [[a] for a in actuals], lang="c_sharp"))
if __name__ == "__main__":
    main()
