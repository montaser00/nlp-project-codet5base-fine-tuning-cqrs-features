# Model Card — CodeT5+ 770M (QLoRA Fine-Tuned for Code Generation)

---

## Model Summary

| Field | Details |
|---|---|
| **Base Model** | `Salesforce/codet5p-770m` |
| **Model Type** | Encoder-Decoder Transformer |
| **Fine-Tuning Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Task** | Code Generation / Completion |
| **Parameters** | ~770M (base) |
| **Precision** | 4-bit quantized (NF4) + BFloat16 compute |
| **Hardware Target** | NVIDIA A100 (BF16-optimized) |

---

## Model Description

This model is a parameter-efficient fine-tune of Salesforce's **CodeT5+ 770M** — an encoder-decoder model pre-trained on large-scale code corpora across multiple programming languages. The fine-tuning was performed using **QLoRA**, which quantizes the base model to 4-bit precision and trains only a small set of low-rank adapter weights, making it feasible to fine-tune a 770M-parameter model on a single GPU without sacrificing output quality.

The model is trained to generate or complete code given a natural language prompt or a partial code snippet.

---

## Intended Use

### Primary Use
- **Code generation from natural language:** Given a description or docstring, the model produces corresponding code.
- **Code completion:** Given a partial code snippet, the model continues or fills in the remaining logic.

### Out-of-Scope Use
- This model is not intended for general-purpose natural language tasks.
- Should not be used for generating malicious or unsafe code.
- Not validated for production use without further evaluation and testing.

---

## Training Configuration

### Data

| Parameter | Value |
|---|---|
| Max Input Length | 512 tokens |
| Max Target Length | 2048 tokens |

> **Note:** The max target length was fixed from an earlier setting of 12,000 tokens, which caused severe training slowness (~13 seconds/iteration).

---

### QLoRA Adapter Settings

| Parameter | Value |
|---|---|
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | `q`, `v`, `k`, `o` (all attention projections) |

**Effective scaling factor** (`lora_alpha / lora_r`): **2.0**

---

### Quantization Settings

| Parameter | Value |
|---|---|
| Load in 4-bit | ✅ Yes |
| Quantization Type | `nf4` (Normal Float 4) |
| Compute Dtype | `bfloat16` |
| Nested Quantization | ✅ Yes (double quantization for memory savings) |

> **Note:** `nf4` is the recommended quantization type for QLoRA as it was specifically designed for normally-distributed weights (as in pre-trained LLMs). Nested quantization further reduces the memory footprint of the quantization constants themselves.

---

### Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 5 | |
| Train Batch Size (per device) | 4 | Fixed from 2 — previous size was starving the GPU |
| Eval Batch Size (per device) | 4 | Fixed from 2 |
| Learning Rate | `2e-4` | |
| Warmup Steps | 50 | |
| Gradient Accumulation Steps | 4 | Effective batch size = **16** |
| Precision | BF16 | Fixed: BF16 is preferred over FP16 on A100 |
| FP16 | Disabled | |
| Logging Steps | 10 | |
| Eval Steps | 50 | |
| Save Steps | 100 | |

**Effective Batch Size Calculation:**
```
per_device_train_batch_size × gradient_accumulation_steps = 4 × 4 = 16
```

---

## Training Infrastructure

| Item | Detail                                  |
|---|-----------------------------------------|
| GPU | B200 + VRAM: 250GB + RAM 280GB (RunPod) |
| Mixed Precision | BFloat16 (`bf16=True`)                  |
| Output Directory | `./content/output/`                     |

---

## Architecture & Adapter Details

```
Base:     Salesforce/codet5p-770m  (frozen, 4-bit NF4)
Adapters: LoRA on [q, k, v, o] projections
          r=16, alpha=32, dropout=0.05
Trainable params: ~10M parameters
```

The LoRA adapters are injected into all four attention projection matrices (`query`, `key`, `value`, `output`), covering the full self-attention mechanism in each transformer layer.

---

## Known Fixes & Design Decisions

| Issue | Old Value | Fixed Value | Reason |
|---|---|---|---|
| `max_target_length` too large | 12,000 | 2,048 | Caused ~13s/iteration training slowness |
| Batch size too small | 2 | 4 | GPU underutilized ("GPU starving") |
| Wrong precision type | `fp16=True` | `bf16=True` | BF16 is hardware-accelerated  |
| Wrong compute dtype | `float16` | `bfloat16` | Consistency with bf16 training |

---

## Evaluation

> ⚠️ Evaluation metrics are not yet specified. It is recommended to evaluate on standard code generation benchmarks such as:
> - **Qualitative** — functional correctness.
> - **sacrebleu** — for generated code model accuracy. 

---

## Limitations

- Fine-tuned with QLoRA adapters only — the base model weights are frozen and quantized to 4-bit, which may introduce minor quality degradation compared to full fine-tuning.
- The model's maximum input context is **512 tokens**, which may be insufficient for very long code files or complex multi-function prompts.
- Performance is bounded by the capabilities of the `codet5p-770m` base model and the quality/diversity of the fine-tuning dataset.
- No bias or fairness evaluation has been conducted on this checkpoint.

---

