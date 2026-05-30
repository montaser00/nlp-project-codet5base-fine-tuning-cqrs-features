"""
CodeT5 C# CQRS inference script.

Loads a fine-tuned CodeT5+ model with an optional LoRA adapter and generates
C# CQRS boilerplate files from natural-language prompts.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import TypedDict

import torch
import transformers
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

def load_model(model_name: str, adapter_path: str) -> tuple:
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

    if not os.path.isdir(adapter_path):
        print(red(f"\n  [WARN] Adapter not found at: {adapter_path}"))
        print(red("         Run codet5_csharp_cqrs_finetune.py first, then retry."))
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
# Print / save helpers
# ──────────────────────────────────────────────────────────────

def _print_output(text: str) -> None:
    print()
    print(bold(cyan("Model")) + bold(" >"))
    print("  " + "─" * 60)
    for line in text.splitlines():
        print("  " + line)
    print("  " + "─" * 60)
    print()


def _save_output(output: str, file_name: str) -> str:
    output_dir = os.path.abspath(os.path.join("..", "..", "content", "generated", "inference"))
    os.makedirs(output_dir, exist_ok=True)

    if not file_name.endswith(".cs"):
        file_name += ".cs"

    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(output)

    return file_path


# ──────────────────────────────────────────────────────────────
# Test prompts runner
# ──────────────────────────────────────────────────────────────

def run_test_prompts(model, tokenizer, device: str) -> None:
    print("\n" + "=" * 62)
    print(bold(cyan("   CodeT5 C# CQRS Chat  ") + dim("(fine-tuned)")))
    print(f"   Model      : {_MODEL_PATH}")
    print(f"   Adapter    : {os.path.abspath(_ADAPTER_PATH)}")
    print(f"   Output dir : {_OUTPUT_PATH}")
    print(f"   Device     : {device}")
    print(f"   transformers {transformers.__version__}")
    print("=" * 62)
    print(dim("   Type /help for commands and prompt format.  /quit to exit.\n"))

    print(dim("  Generating ..."))

    while True:
        print("Enter Component Name: ")
        prompt_text = input()
        print("\nEnter Save File Path:")
        file_name = os.path.abspath(input())

        try:
            output = generate(model, tokenizer, device, prompt_text, _DEFAULT_GENERATION_CONFIG)
        except Exception as e:
            print(red(f"  Error: {e}\n"))

        if not output.strip():
            print(yellow(
                "  [WARN] Output was empty after cleaning.\n"
                "  Make sure the adapter path is correct and the model finished training.\n"
            ))

        _print_output(output)
        _save_output(output, file_name)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    model, tokenizer, device = load_model(_MODEL_PATH, _ADAPTER_PATH)
    run_test_prompts(model, tokenizer, device)


if __name__ == "__main__":
    main()
