# Medical Triage Agent — Fine-Tuned LLM for CHSA (POC)

> Proof-of-concept of a bilingual medical triage assistant based on Qwen3-1.7B,
> fine-tuned with QLoRA (SFT) and aligned with human preferences (DPO),
> served via a vLLM + FastAPI endpoint with a full CI/CD pipeline.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-FF6F00.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.67+-945DD6.svg?logo=dvc&logoColor=white)
![GCS](https://img.shields.io/badge/GCS-Artifact_Store-4285F4.svg?logo=googlecloud&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-70_passed-brightgreen.svg?logo=pytest)

---

## Project status

| Phase | Content | Status |
|---|---|---|
| Week 1 | Data pipeline — ingestion, cleaning, anonymization, SFT/DPO datasets | ✅ Done |
| Week 2 | SFT — QLoRA fine-tuning of Qwen3-1.7B + MLflow + GCS | ✅ Done |
| Week 3 | DPO — preference alignment on UltraMedical-Preference | 🔄 Code ready, run pending |
| Week 4 | Deployment — vLLM endpoint + FastAPI + CI/CD + evaluation | 🔄 In progress |

**Deliverables:**
- ✅ Bilingual medical corpus (Parquet, DVC-versioned, Presidio-anonymized)
- ✅ SFT model (LoRA adapters — train loss 1.112 / eval loss 1.189)
- 🔄 DPO-aligned model (pending run)
- 🔄 Inference endpoint (API ready, deployment to finalize)
- 🔄 Technical report (sections 1–4 written)

---

## What this project does

The Centre Hospitalier Saint-Aurélien (CHSA) needs an AI assistant capable of triaging
patient descriptions and classifying their urgency level (immediate / moderate / deferred).

This repository implements the full ML pipeline for that POC:

1. **Data** — Four public medical QA datasets (French + English) are cleaned, anonymized,
   and assembled into a 5,000-pair SFT corpus and a 5,000-triplet DPO corpus.

2. **Training** — Qwen3-1.7B-Base is fine-tuned with QLoRA (4-bit NF4, LoRA rank 16)
   on the SFT corpus, then aligned on human preferences via DPO.

3. **Deployment** — The merged model is served via vLLM behind a FastAPI REST API,
   containerized with Docker, and deployed to a GCP VM through a GitHub Actions CI/CD pipeline.

---

## Quick start

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- [DVC](https://dvc.org/) 3.67+
- Google Cloud credentials: `gcloud auth application-default login`
- A GCS bucket named `p14-medical-data` (or update `config/paths.py`)
- For training: NVIDIA GPU with ≥ 16 GB VRAM (T4, L4, A100) + CUDA

### Setup

```bash
git clone https://github.com/RandomFab/FINE-TUNING_MEDICAL.git
cd FINE-TUNING_MEDICAL

uv sync --all-groups
source .venv/bin/activate

# Required for the anonymization module
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md
```

### Run the data pipeline

```bash
dvc repro
```

This runs all 6 stages in order (clean → generate SFT/DPO datasets) and caches
outputs in GCS. Any change to `params.yaml` automatically invalidates affected stages.

### Run SFT training

```bash
MLFLOW_TRACKING_URI=http://<your-mlflow-server>:5000 \
MLFLOW_EXPERIMENT_NAME=sft-qwen3-medical \
python -m src.training.train_sft
```

### Run DPO alignment (after SFT completes)

```bash
MLFLOW_TRACKING_URI=http://<your-mlflow-server>:5000 \
MLFLOW_EXPERIMENT_NAME=dpo-qwen3-medical \
python -m src.training.train_dpo
```

The DPO script automatically retrieves the latest SFT champion from MLflow
(tagged `model_status=champion`, `stage=sft`) — no manual path configuration needed.

### Run the API locally (no GPU)

```bash
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Run the test suite

```bash
pytest tests/ -v
# Expected: 70 passed
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE (DVC)                      │
│                                                                 │
│  HuggingFace Hub                                                │
│  ├── MediQAL MCQU (FR) ──┐                                      │
│  ├── FrenchMedMCQA (FR) ─┤                                      │
│  ├── MedQuAD (EN) ───────┼─► clean ─► anonymize ─► tokenize    │
│  └── UltraMedical (EN) ──┘             │                        │
│                                        ▼                        │
│                          sft_dataset/  +  dpo_dataset/          │
│                          (Parquet, versioned on GCS)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      TRAINING (Colab / GCP VM)                  │
│                                                                 │
│  Qwen3-1.7B-Base ──► QLoRA SFT ──► LoRA adapters                │
│                            │         (MLflow + GCS)             │
│                            ▼                                    │
│                  DPO alignment ──► aligned adapters             │
│                            │         (MLflow + GCS)             │
│                            ▼                                    │
│              merge_and_unload() ──► merged model (GCS)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    DEPLOYMENT (GCP VM)                          │
│                                                                 │
│  GitHub Actions CI/CD                                           │
│  ├── Job 1: ruff + pytest (70 tests, GPU-free via mock)         │
│  ├── Job 2: docker build + push → GHCR                          │
│  └── Job 3: SSH deploy → docker run --gpus all                  │
│                                                                 │
│  FastAPI (/health, /generate)                                   │
│  └── vLLM (continuous batching, PagedAttention)                 │
│      └── merged Qwen3 model (volume-mounted from GCS)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
FINE-TUNING_MEDICAL/
│
├── config/
│   ├── logger.py                   # Centralized structured logging
│   └── paths.py                    # GCS + local path constants (env-overridable)
│
├── data/processed/                 # DVC-tracked Parquet outputs
│   ├── mediqal_dataset/
│   ├── frenchmedmcqa_dataset/
│   ├── medquad_dataset/
│   ├── ultramed_dataset/
│   ├── sft_dataset/                # sft_train / sft_val / sft_test
│   └── dpo_dataset/                # dpo_train / dpo_val / dpo_test
│
├── src/
│   ├── processing/
│   │   ├── mediqal_cleaning.py
│   │   ├── frenchmedmcqa_cleaning.py
│   │   ├── medquad_cleaning.py
│   │   ├── ultramed_cleaning.py
│   │   ├── anonymisation.py        # Presidio + spaCy, FR + EN
│   │   ├── utils_cleaning.py       # Shared utilities (sampling, splitting, token counts)
│   │   ├── sft_dataset/generate_sft_dataset.py
│   │   └── dpo_dataset/generate_dpo_dataset.py
│   │
│   ├── training/
│   │   ├── train_sft.py            # QLoRA SFT with HF Trainer
│   │   ├── train_dpo.py            # DPO alignment with TRL DPOTrainer
│   │   ├── utils_training.py       # Chat templates, MLflow helpers, training args
│   │   └── generate_model_for_deployment.py  # Merge LoRA + push to GCS
│   │
│   └── api/
│       ├── main.py                 # FastAPI app (lifespan, middleware, error handler)
│       ├── schemas.py              # Pydantic request/response models
│       └── services/inference.py  # VLLMEngine wrapper (AsyncLLMEngine)
│
├── tests/
│   ├── unit/                       # Schema validation, path config, logger
│   ├── integration/                # API behaviour with mocked vLLM
│   └── smoke/                      # Dockerfile and CI workflow structure
│
├── models/
│   ├── sft_checkpoints/            # Auto-managed (max 3 kept)
│   └── lora_trained_model/         # Final LoRA adapter weights
│
├── notebooks/                      # EDA + HuggingFace import notebooks
├── dvc.yaml                        # 6-stage pipeline definition
├── params.yaml                     # All tunable parameters (DVC surface)
├── Dockerfile
├── .dockerignore
└── .github/workflows/cicd.yml      # 3-job CI/CD pipeline
```

---

## Data

### Sources

| Dataset | Language | Type | Rows (raw) | Usage |
|---|---|---|---|---|
| MediQAL MCQU | FR | Clinical MCQ | 10 113 (train) | SFT |
| FrenchMedMCQA | FR | Medical MCQ | 595 (train) | SFT |
| MedQuAD | EN | Open QA | 16 407 | SFT |
| UltraMedical-Preference | EN | Preference pairs | 109 353 | SFT + DPO |

### Output datasets

| Dataset | Schema | Size | Split |
|---|---|---|---|
| SFT | `question`, `answer`, `language`, `question_type`, `confidence_level`, `dataset_name`, `token_count_*` | 5 000 pairs | 70 / 20 / 10 |
| DPO | `question`, `chosen`, `rejected`, `language`, `confidence_level`, `dataset_name`, `token_count_*` | 5 000 triplets | 70 / 20 / 10 |

**Key design decisions:**
- PII anonymization via Microsoft Presidio (5 entity types: `PERSON`, `EMAIL`, `PHONE`, `DATE`, `LOCATION`)
- Token counts computed with the Qwen3 tokenizer before training — no re-tokenization needed downstream
- Stratified splits on `dataset_name` — source distribution is preserved in train/val/test
- All parameters (sample counts, split ratios, source files) are in [`params.yaml`](params.yaml)

---

## Training

### SFT — Supervised Fine-Tuning

Qwen3-1.7B-Base fine-tuned on 3,500 medical QA pairs with QLoRA.
All hyperparameters are in [`params.yaml`](params.yaml) under `lora_config`,
`sft_model`, `quantization_config`, and `training_arguments.sft`.

| Metric | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Quantization | 4-bit NF4 (QLoRA, double quant) |
| LoRA rank / alpha | 16 / 32 |
| Target modules | 7 (q/k/v/o\_proj + gate/up/down\_proj) |
| Effective batch size | 32 (1 × 32 grad. accum.) |
| Peak learning rate | 2e-4 (cosine schedule, 30 warmup steps) |
| Epochs | 2 |
| Train loss | 1.112 |
| Eval loss | 1.189 |
| Duration | ~2h41 on NVIDIA T4 (16 GB) |

MLflow experiment: `sft-qwen3-medical`. Best run tagged `model_status=champion`, `stage=sft`.

### DPO — Direct Preference Optimization

Aligns the SFT champion on 3,500 `(question, chosen, rejected)` triplets from UltraMedical-Preference.
All hyperparameters are in [`params.yaml`](params.yaml) under `training_arguments.dpo`.

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 5e-6 | Fine-grained adjustment — 40× lower than SFT |
| Beta | 0.1 | Conservative KL penalty — preserves medical knowledge |
| Epochs | 2 | Small dataset, overfitting risk beyond 2 |

> Full metrics (`rewards/chosen`, `rewards/rejected`, `rewards/margins`) will be updated
> after run completion.

---

## API

The inference endpoint exposes two routes.

### `GET /health`

Returns `200` when the vLLM engine is loaded and ready, `503` otherwise.

```json
{ "status": "healthy", "model": "merged_qwen3_medical" }
```

### `POST /generate`

```bash
curl -X POST http://<IP_VM>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Patient, 58 ans, douleurs thoraciques irradiant dans le bras gauche, diaphorèse.",
    "max_tokens": 512,
    "temperature": 0.1
  }'
```

```json
{ "response": "Niveau de priorité : URGENCE MAXIMALE. ..." }
```

**Input constraints:** `prompt` 10–4096 chars · `max_tokens` 1–2048 · `temperature` 0.0–2.0

Interactive documentation available at `http://<IP_VM>:8000/docs`.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/cicd.yml`) runs three sequential jobs on every push to `main`:

1. **`code-quality-and-tests`** — `ruff check .` + `pytest tests/` (GPU-free, vLLM mocked via `sys.modules`)
2. **`build-and-push-docker`** — Docker Buildx build + push to GHCR with commit SHA tag
3. **`deploy`** — SSH into GCP VM, pull new image, restart container with `--gpus all`

The data pipeline (DVC) and the deployment pipeline (GitHub Actions) are intentionally decoupled:
the Docker image contains no datasets or model weights. Model weights are volume-mounted from GCS
at container startup, so a new model can be deployed without rebuilding the image, and vice versa.

---

## Configuration

All tunable parameters are centralized in [`params.yaml`](params.yaml).
DVC stages declare explicit `params:` dependencies, so any change triggers only
the affected downstream stages — not the full pipeline.

---

## License

This project was developed as part of an AI engineering programme. Datasets are subject
to their respective original licenses (see HuggingFace dataset cards).
