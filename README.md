# NLP Project — CodeT5+ 770M Fine-Tuning for Generating CQRS Features in .NET

> **Author:** Montaser Mohtaseb

Fine-tuning [Salesforce/codet5p-770m](https://huggingface.co/Salesforce/codet5p-770m) using QLoRA to generate CQRS feature code in .NET from natural language prompts.

📄 [See Model Card](./MODEL_CARD.md)

---

## Quick Start

Run the setup script to prepare the environment and execute the notebooks automatically:

```bash
bash setup.sh
```

---

## Manual Setup

### 1. Prepare the Environment

```bash
python -m venv venv
./venv/Scripts/activate
python -m pip install -r requirements.txt
pip install -U huggingface_hub
hf download Salesforce/codet5p-770m --local-dir ./content/codet5p-770m
```

### 2. Run Model Fine-Tuning

```bash
python .\src\features\train\codet5-fine-tuning.py
```

### 3. Run the Notebooks

Pre-trained model evaluation:

```bash
.\src\notebooks\codet5p-770m.ipynb
```

Fine-tuned model evaluation:

```bash
.\src\notebooks\codet5p-770m-fine-tuned.ipynb
```

---

## Project Structure

```
nlp-project-codet5base-fine-tuning-cqrs-features/
├── content/
│   ├── codet5p-770m/
│   ├── generated/
│   │   ├── base/
│   │   └── fine-tuned/
│   ├── output/
│   ├── training_data.json
│   └── validation_data.json
├── src/
│   ├── features/
│   │   └── train/
│   │       └── codet5-fine-tuning.py
│   └── notebooks/
│       ├── codet5p-770m.ipynb
│       └── codet5p-770m-fine-tuned.ipynb
├── .gitignore
├── MODEL_CARD.md
├── README.md
├── requirements.txt
└── setup.sh
```

### Directory Reference

| Path | Description |
|---|---|
| `content/training_data.json` | Training data |
| `content/validation_data.json` | Validation data |
| `content/generated/` | Generated artifacts (base & fine-tuned outputs) |
| `content/output/` | QLoRA adapter weights |
| `content/codet5p-770m/` | Local base model |
