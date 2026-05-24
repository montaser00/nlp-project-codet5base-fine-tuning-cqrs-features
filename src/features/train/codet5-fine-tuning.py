import os
import inspect
import warnings
import torch
import transformers
import json

warnings.filterwarnings("ignore")

# ── Disable HF safetensors auto-conversion (prevents OSError) ─────────
os.environ["DISABLE_TELEMETRY"]              = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]  = "0"
os.environ["TOKENIZERS_PARALLELISM"]         = "false"

import evaluate
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,          # ← FIXED: was AutoModelForSeq2SeqLM
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ──────────────────────────────────────────────────────────────────────
# Tokenizer compatibility patch
# ──────────────────────────────────────────────────────────────────────

def _fix_tokenizers_compatibility() -> None:
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        from tokenizers import AddedToken as _Rust

        def _patched(self, new_tokens, special_tokens=False):
            fixed = []
            for t in new_tokens:
                if isinstance(t, (str, _Rust)):
                    fixed.append(t)
                elif hasattr(t, "content"):
                    kw = dict(single_word=getattr(t, "single_word", False),
                              lstrip=getattr(t, "lstrip", False),
                              rstrip=getattr(t, "rstrip", False),
                              normalized=getattr(t, "normalized", True))
                    try:    fixed.append(_Rust(str(t.content), special=special_tokens, **kw))
                    except: fixed.append(_Rust(str(t.content), **kw))
                else:
                    fixed.append(str(t))
            return (self._tokenizer.add_special_tokens(fixed)
                    if special_tokens else self._tokenizer.add_tokens(fixed))

        TokenizersBackend._add_tokens = _patched
        print("[INFO] Tokenizers compatibility patch applied.")
    except ImportError:
        pass

_fix_tokenizers_compatibility()


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_name: str = os.path.abspath('./content/codet5p-770m')
    output_dir: str = "./content/output/"

    max_input_length:  int = 512
    max_target_length: int = 2048        # ← FIXED: was 12000 (causing 13s/it slowness)

    # QLoRA
    lora_r:              int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q", "v", "k", "o"]
    )

    # Training
    num_train_epochs:              int   = 5
    per_device_train_batch_size:   int   = 4     # ← FIXED: was 2 (too small, GPU starving)
    per_device_eval_batch_size:    int   = 4     # ← FIXED: was 2
    learning_rate:                 float = 2e-4
    warmup_steps:                  int   = 50
    logging_steps:                 int   = 10
    eval_steps:                    int   = 50
    save_steps:                    int   = 100
    gradient_accumulation_steps:   int   = 4
    bf16:                          bool  = True  # ← FIXED: use bf16 on A100, not fp16
    fp16:                          bool  = False # ← FIXED: disabled, bf16 is better on A100

    # Quantization
    use_4bit:               bool  = True
    bnb_4bit_compute_dtype: str   = "bfloat16"  # ← FIXED: was float16, use bfloat16 on A100
    bnb_4bit_quant_type:    str   = "nf4"
    use_nested_quant:       bool  = True


# ──────────────────────────────────────────────────────────────────────
# C# CQRS training dataset
# ──────────────────────────────────────────────────────────────────────

with open("./content/training_data.json", "r", encoding="utf-8") as f:
    SAMPLE_DATA = json.load(f)

with open("./content/validation_data.json", "r", encoding="utf-8") as f:
    VALIDATION_DATA = json.load(f)

SAMPLE_DATA = {
    "train":      SAMPLE_DATA,
    "validation": VALIDATION_DATA
}


# ──────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────

def build_dataset(raw: Dict, tokenizer, config: TrainingConfig) -> DatasetDict:
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=config.max_input_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["output"],
            max_length=config.max_target_length,
            padding="max_length",
            truncation=True,
        )
        label_ids = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    splits = {}
    for split, examples in raw.items():
        ds   = Dataset.from_list(examples)
        drop = [c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")]
        ds   = ds.map(tokenize, batched=True, remove_columns=drop)
        splits[split] = ds
    return DatasetDict(splits)


# ──────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────

def load_base_model(config: TrainingConfig, quantize: bool = False):
    """
    FIXED:
      - Use T5ForConditionalGeneration instead of AutoModelForSeq2SeqLM
      - Removed trust_remote_code (not needed for 770m, causes issues)
      - Use bfloat16 instead of float16 on A100
      - Removed device_map=None which was blocking GPU usage
    """
    if quantize and config.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,   # ← FIXED: bfloat16 for A100
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        model = T5ForConditionalGeneration.from_pretrained(  # ← FIXED
            config.model_name,
            quantization_config=bnb,
            device_map="auto",                               # ← FIXED: was None
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = T5ForConditionalGeneration.from_pretrained(  # ← FIXED
            config.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,  # ← FIXED
        )

    model.config.use_cache = False
    return model


def apply_qlora(model, config: TrainingConfig):
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _device(model) -> torch.device:
    try:    return next(model.parameters()).device
    except: return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────

def generate_predictions(model, tokenizer, examples: List[Dict],
                         config: TrainingConfig) -> List[str]:
    model.eval()
    dev   = _device(model)
    preds = []
    with torch.no_grad():
        for ex in examples:
            inputs = tokenizer(
                ex["input"],
                return_tensors="pt",
                max_length=config.max_input_length,
                truncation=True,
            ).to(dev)
            ids = model.generate(
                **inputs,
                max_new_tokens      = config.max_target_length,
                num_beams           = 4,
                early_stopping      = True,              # ← FIXED: was False with num_beams=4 (contradictory)
                eos_token_id        = tokenizer.eos_token_id,
                forced_eos_token_id = tokenizer.eos_token_id,
                repetition_penalty  = 1.2,
            )
            preds.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return preds


# ──────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(predictions: List[str], references: List[str], stage: str) -> Dict:
    bleu   = evaluate.load("sacrebleu")
    exact  = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return {
        "stage":       stage,
        "bleu":        round(result["score"], 2),
        "exact_match": round(exact / len(references) * 100, 2),
        "samples":     len(references),
    }


def print_comparison(before: Dict, after: Dict):
    print("\n" + "=" * 60)
    print("         EVALUATION RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<22} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 57)
    for k, label in [("bleu", "BLEU Score"), ("exact_match", "Exact Match (%)")]:
        b, a  = before[k], after[k]
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        print(f"{label:<22} {b:>10.2f} {a:>10.2f} {sign}{delta:>9.2f}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset: DatasetDict, config: TrainingConfig):
    collator    = DataCollatorForSeq2Seq(tokenizer, model=model,
                                         label_pad_token_id=-100,
                                         pad_to_multiple_of=8)

    args_params    = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters

    eval_kwarg = ({"eval_strategy": "steps"}
                  if "eval_strategy" in args_params
                  else {"evaluation_strategy": "steps"})
    tok_kwarg  = ({"processing_class": tokenizer}
                  if "processing_class" in trainer_params
                  else {"tokenizer": tokenizer})

    # ← FIXED: use bf16 instead of fp16 on A100
    use_bf16 = config.bf16 and torch.cuda.is_available()
    use_fp16 = config.fp16 and torch.cuda.is_available() and not use_bf16

    train_args = Seq2SeqTrainingArguments(
        output_dir                  = config.output_dir,
        num_train_epochs            = config.num_train_epochs,
        per_device_train_batch_size = config.per_device_train_batch_size,
        per_device_eval_batch_size  = config.per_device_eval_batch_size,
        learning_rate               = config.learning_rate,
        warmup_steps                = config.warmup_steps,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
        bf16                        = use_bf16,          # ← FIXED: A100 optimized
        fp16                        = use_fp16,
        logging_dir                 = os.path.join(config.output_dir, "logs"),
        logging_steps               = config.logging_steps,
        **eval_kwarg,
        eval_steps                  = config.eval_steps,
        save_steps                  = config.save_steps,
        save_total_limit            = 2,
        predict_with_generate       = True,
        generation_max_length       = config.max_target_length,  # ← FIXED: added
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        dataloader_num_workers      = 4,                 # ← FIXED: added, speeds up data loading
        dataloader_pin_memory       = True,              # ← FIXED: added, faster CPU→GPU transfer
        report_to                   = "none",
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = train_args,
        train_dataset = dataset["train"],
        eval_dataset  = dataset["validation"],
        data_collator = collator,
        **tok_kwarg,
    )

    print("\n[INFO] Starting QLoRA fine-tuning ...")
    trainer.train()
    print("[INFO] Training complete.")

    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[INFO] Adapter saved → {adapter_path}")
    return trainer


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Model       : {config.model_name}")
    print(f"[INFO] transformers: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────────
    # FIXED: use AutoTokenizer directly — codet5p-770m uses standard tokenizer
    # RobertaTokenizer is not needed and causes confusion
    print("\n[INFO] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print(f"[INFO] Tokenizer loaded  (vocab size: {tokenizer.vocab_size})")

    # ── Dataset ────────────────────────────────────────────────────────
    print("[INFO] Building dataset ...")
    dataset      = build_dataset(SAMPLE_DATA, tokenizer, config)
    val_examples = SAMPLE_DATA['validation']
    val_refs     = [ex["output"] for ex in val_examples]
    print(f"[INFO] Train: {len(SAMPLE_DATA['train'])} examples  "
          f"Val: {len(val_examples)} examples")

    # ── BEFORE evaluation ──────────────────────────────────────────────
    print("\n[INFO] Loading base model for BEFORE evaluation ...")
    base_model    = load_base_model(config, quantize=False)
    before_preds  = generate_predictions(base_model, tokenizer, val_examples, config)
    before_result = compute_metrics(before_preds, val_refs, "before")
    print(f"[BEFORE] BLEU: {before_result['bleu']:.2f}  "
          f"Exact Match: {before_result['exact_match']:.2f}%")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Fine-tune ──────────────────────────────────────────────────────
    print("\n[INFO] Loading model in 4-bit for QLoRA ...")
    qlora_model = load_base_model(config, quantize=True)
    qlora_model = apply_qlora(qlora_model, config)
    train(qlora_model, tokenizer, dataset, config)

    # ── AFTER evaluation ───────────────────────────────────────────────
    print("\n[INFO] AFTER evaluation ...")
    after_preds  = generate_predictions(qlora_model, tokenizer, val_examples, config)
    after_result = compute_metrics(after_preds, val_refs, "after")
    print(f"[AFTER]  BLEU: {after_result['bleu']:.2f}  "
          f"Exact Match: {after_result['exact_match']:.2f}%")

    # ── Comparison ────────────────────────────────────────────────────
    print_comparison(before_result, after_result)

    # ── Print sample ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SAMPLE  —  Validation example 0")
    print("=" * 60)
    print(f"\nINPUT:\n{val_examples[0]['input']}")
    print(f"\nTARGET (first 300 chars):\n{val_examples[0]['output'][:300]} ...")
    print(f"\nBEFORE (first 300 chars):\n{before_preds[0][:300]} ...")
    print(f"\nAFTER  (first 300 chars):\n{after_preds[0][:300]} ...")
    print("=" * 60)

    # ── Save metrics JSON ─────────────────────────────────────────────
    results_path = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({"before": before_result, "after": after_result}, f, indent=2)
    print(f"[INFO] Metrics saved → {results_path}")


if __name__ == "__main__":
    main()
