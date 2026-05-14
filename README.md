# 🩺 Fine-Tuning Medical — POC Agent IA de Triage Médical (CHSA)

A machine learning project that ingests, cleans, anonymizes, and prepares French-language medical QA
datasets, then fine-tunes Qwen3-1.7B-Base via QLoRA for a medical triage assistant. Targets multiple
Hugging Face sources, normalizes them into unified SFT `(question, answer)` and DPO
`(question, chosen, rejected)` schemas, tokenizes with Qwen3 chat template, and trains with LoRA
adapters — all orchestrated through reproducible DVC pipelines and tracked with MLflow.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458.svg?logo=pandas&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-FF6F00.svg)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4bit-green.svg)
![GCS](https://img.shields.io/badge/Google_Cloud_Storage-3.10.1-4285F4.svg?logo=googlecloud&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.67+-945DD6.svg?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg?logo=githubactions&logoColor=white)
![vLLM](https://img.shields.io/badge/Inference-vLLM-blueviolet.svg)
![pytest](https://img.shields.io/badge/Tests-70_passed-brightgreen.svg?logo=pytest)
---

## 🎯 Objective

Medical LLMs require high-quality, domain-specific training data — but raw datasets from the
community are heterogeneous, contain duplicates, irrelevant columns, inconsistent answer encodings,
and potentially identifiable information. This project standardizes four French and bilingual medical
QA corpora into clean SFT and DPO formats, then fine-tunes a compact language model to serve as a
medical triage assistant for the Centre Hospitalier Saint-Aurélien (CHSA).

**Data pipeline (Step 1 — complete):**
- 🗂️ Ingest datasets directly from Hugging Face Hub
- 🧹 Clean, deduplicate, and normalize each source independently
- 🔄 Resolve answer indices to their textual form and create a ground-truth `answer` column
- 🕵️ Anonymize PII (names, emails, phones, dates, locations) across FR and EN text
- 🏷️ Tag every row with source metadata (language, question type, confidence level, dataset name)
- 🔢 Count tokens using the Qwen3 tokenizer before any fine-tuning stage
- ⚖️ Sample a balanced, multi-source SFT dataset and a DPO preference dataset
- 🔁 Reproduce the full pipeline deterministically with a single `dvc repro`

**SFT training (Step 2 — complete):**
- 🧠 Load Qwen3-1.7B-Base in 4-bit precision (QLoRA with NF4 quantization)
- 🔧 Apply LoRA adapters to all attention + MLP projections (rank 16, alpha 32)
- 💬 Format data with Qwen3 chat template (system prompt + user + assistant) and label masking
- 🏋️ Train with gradient checkpointing, paged AdamW 8-bit, and cosine LR schedule
- 📊 Track experiments with MLflow, save checkpoints, and auto-resume after interruption
- ✅ Achieved train loss 1.112 / eval loss 1.189 after 3 epochs (~2h41 on T4 GPU)

**DPO alignment (Step 3 — in progress):**
- 🎯 Align the SFT model on human preferences using Direct Preference Optimization
- 📐 Fine-tune with `(question, chosen, rejected)` triplets from UltraMedical-Preference
- 🔗 Load the SFT champion automatically from MLflow via `model_status=champion` tag
- 📊 Monitor `rewards/chosen`, `rewards/rejected`, and `rewards/margins` in MLflow
- ⚖️ Conservative `beta=0.1` to preserve medical knowledge acquired during SFT

**Deployment & CI/CD (Step 4 — in progress):**
- ⚡ Serve the merged model (base + LoRA) via **vLLM** for optimized inference (continuous batching, PagedAttention)
- 🌐 Expose a REST API with **FastAPI** (lifespan pattern, Pydantic validation, latency middleware)
- 🐳 Containerize with **Docker** and deploy on a GCP VM via SSH
- 🔄 Automate tests → build → push → deploy with **GitHub Actions** (3-job pipeline)
- 🧪 70 automated tests across unit / integration / smoke layers — GPU-free CI via vLLM mocking
---

## ✨ Features

### Data preparation
- ✅ Dedicated cleaning pipeline for **MediQAL** (MCQU subset, French clinical MCQ)
- ✅ Dedicated cleaning pipeline for **FrenchMedMCQA** (595 / 164 / 321 split)
- ✅ Dedicated cleaning pipeline for **MedQuad** (16 407 English open-QA pairs)
- ✅ Dedicated cleaning pipeline for **UltraMedical-Preference** — outputs both SFT and DPO variants
- ✅ Shared utility layer: `drop_columns`, `drop_duplicates`, `transform_correct_answers_to_text`, `create_ground_truth_answer_column`, `add_metadata`, `add_token_counts`, `collect_balanced_samples`, `split_dataset`
- ✅ PII anonymization via **Presidio + spaCy** (FR + EN), 5 entity types: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `DATE_TIME`, `LOCATION` — auto-detected language with `langdetect` fallback
- ✅ Token counting with **Qwen3-1.7B-Base tokenizer** — adds `token_count_<col>` columns before splitting
- ✅ Balanced multi-source sampling with adaptive shortfall redistribution (`collect_balanced_samples`)
- ✅ Stratified train / val / test split on `dataset_name` via scikit-learn (`split_dataset`)
- ✅ Metadata tagging: `language`, `question_type`, `confidence_level`, `dataset_name` on every row
- ✅ Full **SFT dataset** generation: 5 000 balanced samples across 4 sources → train / val / test Parquet
- ✅ Full **DPO dataset** generation: 5 000 preference pairs from UltraMedical → train / val / test Parquet
- ✅ Clinical-case-aware filtering for MediQAL (retains only rows with an associated clinical case)
- ✅ Structured logging at every pipeline step via a centralized `config/logger.py`
- ✅ Reproducible **DVC pipeline** with 6 stages, `params.yaml` surface, and GCS remote tracking
- ✅ EDA notebooks for MediQAL, FrenchMedMCQA, MedQuad, UltraMedical, and the final SFT dataset
- ✅ Import notebook covering all four source datasets from Hugging Face Hub

### SFT training
- ✅ **QLoRA** training: 4-bit NF4 quantization with double quantization via BitsAndBytes
- ✅ **LoRA adapters** on 7 target modules (q/k/v/o_proj + gate/up/down_proj), rank 16, alpha 32
- ✅ **Qwen3 chat template** formatting with system prompt, user question, and assistant answer
- ✅ **Label masking**: prompt tokens masked with -100, loss computed only on assistant response
- ✅ **Memory-efficient training**: gradient checkpointing, paged AdamW 8-bit optimizer, FP16 mixed precision
- ✅ **Cosine LR schedule** with warmup (30 steps), peak LR 2e-4, effective batch size 32
- ✅ **Checkpoint management**: auto-save every 50 steps, auto-resume from last checkpoint, best model selection on eval_loss
- ✅ **MLflow integration** for experiment tracking (loss curves, hyperparameters, model artifacts)
- ✅ Externalized configuration: all hyperparameters in `params.yaml` (LoRA, quantization, training arguments)

### DPO alignment
- ✅ `train_dpo.py` — full DPO training pipeline with `DPOTrainer` (TRL)
- ✅ Automatic SFT champion retrieval from MLflow (tag-based: `model_status=champion`, `stage=sft`)
- ✅ `format_dpo_chat()` — formats `(question, chosen, rejected)` into Qwen3 conversational template
- ✅ `beta=0.1` — conservative KL penalty preserving medical knowledge from SFT
- ✅ Separate MLflow experiment `dpo-qwen3-medical` with `rewards/*` metrics tracking
- ✅ LoRA adapters merged into base model post-training for deployment (`generate_model_for_deployment.py`)

### Deployment & API
- ✅ `src/api/main.py` — FastAPI app with `lifespan` pattern, latency middleware, global error handler
- ✅ `src/api/services/inference.py` — `VLLMEngine` wrapping `AsyncLLMEngine` with `AsyncMock` for CI
- ✅ `src/api/schemas.py` — `GenerationRequest` / `GenerationResponse` with strict Pydantic validation
- ✅ `Dockerfile` — multi-stage build, `HEALTHCHECK`, `ENV PYTHONPATH=/app`, `ENV LANG=C.UTF-8`
- ✅ `.dockerignore` — excludes `data/`, `models/`, `.git`, `.env` from build context
- ✅ GitHub Actions CI/CD — 3 jobs: `code-quality-and-tests` → `build-and-push-docker` → `deploy`
- ✅ GHCR image registry with commit SHA tagging
- ✅ SSH-based deployment to GCP VM with `docker run --gpus all`

### Testing
- ✅ **70 tests, 3 layers**: unit (Pydantic schemas, path config, logger) + integration (API with mocked vLLM) + smoke (Dockerfile & CI structure)
- ✅ `tests/conftest.py` — PYTHONPATH fix for CI runners
- ✅ Integration fixtures: vLLM injected via `sys.modules` mock + `AsyncMock` for GPU-free CI
- ✅ API contract test: enforces exact `{"response": str}` schema — explicit guard against unintended field changes
- ✅ `ruff check .` enforced in CI
---

## 📊 Architecture

```mermaid
graph TB
    subgraph Sources ["☁️ Hugging Face Hub"]
        A["🗄️ ANR-MALADES/MediQAl (mcqu / mcqm / oeq)"]
        B["🗄️ nthngdy/frenchmedmcqa"]
        C["🗄️ keivalya/MedQuad"]
        D["🗄️ TsinghuaC3I/UltraMedical-Preference"]
    end

    subgraph Ingestion ["📥 notebooks/"]
        E["📓 Import_data_from_HuggingFace.ipynb"]
    end

    subgraph GCS_Raw ["🪣 GCS — raw_data/"]
        F["mediqal_datasets/ mcqu / mcqm / oeq"]
        G["frenchmedmcqa_dataset/"]
        H["MedQuad_dataset/"]
        I["UltraMedical_dataset/"]
    end

    subgraph Params ["⚙️ params.yaml"]
        P["sft / dpo sampling config
        lora_config / quantization_config
        training_arguments / sft_model"]
    end

    subgraph Cleaning ["🧹 src/processing/ — Stage 1-4"]
        J["mediqal_cleaning.py clean_mediqal()"]
        K["frenchmedmcqa_cleaning.py clean_frenchmedmcqa()"]
        L["medquad_cleaning.py clean_medquad()"]
        M["ultramed_cleaning.py clean_ultramed_for_SFT() clean_ultramed_for_DPO()"]
        U["utils_cleaning.py drop_columns / drop_duplicates transform_answers / create_ground_truth add_metadata / merge_raw_data_splits"]
    end

    subgraph Local_Processed ["💾 data/processed/ — intermediate"]
        N["mediqal_dataset/ mediqal.parquet"]
        O["frenchmedmcqa_dataset/ frenchmedmcqa.parquet"]
        Q["medquad_dataset/ medquad.parquet"]
        R["ultramed_dataset/ ultramed_sft.parquet ultramed_dpo.parquet"]
    end

    subgraph Generation ["🏗️ src/processing/ — Stage 5-6"]
        ANON["🕵️ anonymisation.py Presidio + spaCy FR + EN · 5 entity types"]
        TOK["🔢 add_token_counts() Qwen3 tokenizer"]
        S["sft_dataset/generate_sft_dataset.py collect_balanced_samples() split_dataset()"]
        T["dpo_dataset/generate_dpo_dataset.py collect_balanced_samples() split_dataset()"]
    end

    subgraph Final ["📦 data/processed/ — final outputs"]
        SF["sft_dataset/ sft_train / val / test.parquet"]
        DF["dpo_dataset/ dpo_train / val / test.parquet"]
    end

    subgraph Training ["🧠 src/training/ — SFT"]
        UT["utils_training.py chat template / tokenization / label masking"]
        TR["train_sft.py QLoRA model + LoRA adapters + Trainer"]
    end

    subgraph Models ["🏆 models/"]
        LM["lora_trained_model/ adapter_config.json adapter_model.safetensors"]
    end

    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    E --> G
    E --> H
    E --> I
    F --> J
    G --> K
    H --> L
    I --> M
    J --> U
    K --> U
    L --> U
    M --> U
    U --> N
    U --> O
    U --> Q
    U --> R
    N --> S
    O --> S
    Q --> S
    R --> S
    R --> T
    P --> S
    P --> T
    P --> TR
    S --> ANON
    T --> ANON
    ANON --> TOK
    TOK --> SF
    TOK --> DF
    SF --> UT
    UT --> TR
    TR --> LM
```

---

## 📁 Project Structure

```
FINE-TUNING_MEDICAL/
│
├── 📂 config/
│   ├── __init__.py
│   ├── logger.py                         # Centralized logging configuration
│   └── paths.py                          # GCS + local path constants
│
├── 📂 data/
│   └── 📂 processed/                     # DVC-tracked outputs
│       ├── mediqal_dataset/
│       │   └── mediqal.parquet
│       ├── frenchmedmcqa_dataset/
│       │   └── frenchmedmcqa.parquet
│       ├── medquad_dataset/
│       │   └── medquad.parquet
│       ├── ultramed_dataset/
│       │   ├── ultramed_sft.parquet
│       │   └── ultramed_dpo.parquet
│       ├── sft_dataset/
│       │   ├── sft_dataset.parquet       # Full merged dataset
│       │   ├── sft_train.parquet
│       │   ├── sft_val.parquet
│       │   └── sft_test.parquet
│       └── dpo_dataset/
│           ├── dpo_dataset.parquet       # Full merged dataset
│           ├── dpo_train.parquet
│           ├── dpo_val.parquet
│           └── dpo_test.parquet
│
├── 📂 models/
│   ├── 📂 sft_checkpoints/               # Training checkpoints (auto-managed, max 3)
│   └── 📂 lora_trained_model/            # Final LoRA adapter weights
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
├── 📂 notebooks/
│   ├── __init__.py
│   ├── Import_data_from_HuggingFace.ipynb    # Ingest all sources → GCS
│   └── 📂 EDA/
│       ├── frenchmedmcqa_analysis.ipynb
│       ├── mediqal_analysis.ipynb
│       ├── medquad_analysis.ipynb
│       ├── sft_dataset_analysis.ipynb
│       └── ultramedical_analysis.ipynb
│
├── 📂 src/
│   ├── 📂 api/ 
│   │   ├── main.py                       # FastAPI app (lifespan, middleware, error handler)
│   │   ├── schemas.py                    # Pydantic request/response models
│   │   └── 📂 services/
│   │       └── inference.py              # VLLMEngine (AsyncLLMEngine wrapper)
│   └── 📂 training/
│       ├── train_sft.py
│       ├── train_dpo.py
│       ├── utils_training.py
│       └── generate_model_for_deployment.py (merge LoRA + push GCS)
│
├── 📂 tests/                       
│   ├── conftest.py                       # PYTHONPATH fix for CI
│   ├── 📂 unit/
│   │   ├── test_schemas.py
│   │   ├── test_paths.py
│   │   └── test_logger.py
│   ├── 📂 integration/
│   │   ├── conftest.py                   # Mock vLLM + TestClient fixtures
│   │   ├── test_health.py
│   │   ├── test_generate.py
│   │   └── test_middleware.py
│   └── 📂 smoke/
│       └── test_docker_build.py
│
├── Dockerfile                     
├── .dockerignore                         
├── .github/
│   └── workflows/
│       └── cicd.yml                      # 3-job CI/CD pipeline
│
├── 🗂️ dvc.yaml                               # 6-stage DVC pipeline definition
├── 🗂️ dvc.lock                               # Locked hashes for reproducibility
├── 📦 params.yaml                            # All parameters (data, LoRA, quantization, training)
├── 📦 pyproject.toml                         # Project metadata & dependencies (uv)
├── 📦 uv.lock                                # Locked dependency graph
├── 📄 RAPPORT_TECHNIQUE.md                   # Technical report (20 pages max)
└── .python-version                           # Python 3.13
```

---

## 🔁 DVC Pipeline

The full data preparation pipeline is defined in `dvc.yaml` and driven by `params.yaml`. Running
`dvc repro` executes all six stages in dependency order, skipping any stage whose inputs have not
changed since the last run.

### Stages (in execution order)

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | `clean_mediqal` | `python -m src.processing.mediqal_cleaning` | `data/processed/mediqal_dataset/` |
| 2 | `clean_frenchmedmcqa` | `python -m src.processing.frenchmedmcqa_cleaning` | `data/processed/frenchmedmcqa_dataset/` |
| 3 | `clean_medquad` | `python -m src.processing.medquad_cleaning` | `data/processed/medquad_dataset/` |
| 4 | `clean_ultramed` | `python -m src.processing.ultramed_cleaning` | `data/processed/ultramed_dataset/` |
| 5 | `generate_sft` | `python -m src.processing.sft_dataset.generate_sft_dataset` | `data/processed/sft_dataset/` |
| 6 | `generate_dpo` | `python -m src.processing.dpo_dataset.generate_dpo_dataset` | `data/processed/dpo_dataset/` |

### Running the pipeline

```bash
# Reproduce all stages (only re-runs stages with changed inputs)
dvc repro

# Force re-run of all stages regardless of cache
dvc repro --force

# Check which stages are outdated without running them
dvc status
```

### Configuration surface — `params.yaml`

All tunable parameters live in `params.yaml`. Changing any value and re-running `dvc repro`
automatically invalidates the downstream stages.


#### Data sampling parameters

All sampling parameters (target size, split ratios, source datasets) are defined in
[`params.yaml`](params.yaml) under the `sft` and `dpo` keys, and are tracked by DVC
as explicit stage dependencies — meaning any change to these values automatically
invalidates the downstream `generate_sft` and `generate_dpo` stages.


#### LoRA & model configuration

LoRA rank, target modules, quantization settings, and the system prompt are all defined
in [`params.yaml`](params.yaml) under `lora_config`, `sft_model`, and `quantization_config`.
Key values at a glance: rank 16, alpha 32, 4-bit NF4 quantization, 7 target modules
(all attention + MLP projections), max sequence length 512 tokens.

#### Training arguments

All training hyperparameters for both SFT and DPO are centralized in
[`params.yaml`](params.yaml) under `training_arguments.sft` and `training_arguments.dpo`.
The key difference between the two stages: SFT uses `lr=2e-4` (LoRA adapts aggressively
on new domain), while DPO uses `lr=5e-6` (fine-grained preference alignment without
destabilizing acquired medical knowledge). Both run for 2 epochs on a T4 GPU (FP16,
`paged_adamw_8bit`, effective batch size 32 via gradient accumulation).

> 💡 Stages 5 and 6 declare their `params:` dependencies explicitly in `dvc.yaml`, so DVC detects
> parameter changes and only reruns those stages — not the cleaning stages.

---

## 🧠 SFT Training

### Overview

The SFT training fine-tunes Qwen3-1.7B-Base on 3,500 medical QA pairs using QLoRA (4-bit quantized
base model + LoRA adapters). The training pipeline formats each `(question, answer)` pair with the
Qwen3 chat template, masks prompt tokens in the labels, and trains only the LoRA adapter weights.

### Training results

| Metric | Value |
|---|---|
| Total duration | ~2h41 on NVIDIA T4 (16 GB VRAM) |
| Total steps | 330 (3 epochs × ~110 steps/epoch) |
| Train loss (average) | 1.112 |
| Eval loss (step 300) | 1.189 |
| Throughput | 1.083 samples/sec |

### Running the training

```bash
# Run SFT training (requires a GPU — T4 16GB minimum)
python -m src.training.train_sft
```

The script will:
1. Tokenize train and validation datasets using the Qwen3 chat template
2. Load the base model in 4-bit precision and apply LoRA adapters
3. Train for 3 epochs with checkpoints every 50 steps
4. Save the final LoRA adapter to `models/lora_trained_model/`

If a previous checkpoint exists in `models/sft_checkpoints/`, training automatically resumes from it.

---

## 🧠 Pipeline Logic

### Answer normalization

Both MCQ datasets encode correct answers differently. The shared utility
`transform_correct_answers_to_text` resolves the `correct_answers` column using a per-dataset
mapping dict, then `create_ground_truth_answer_column` looks up the actual answer text from the row:

```python
df["answer"] = df.apply(lambda row: row[row["correct_answer_text"]], axis=1)
```

This produces a clean `answer` column regardless of the original encoding scheme (letter string
vs. integer index).

### MediQAL-specific filtering

MediQAL MCQU rows without a linked clinical case are dropped before any other transformation.
Only rows where `clinical_case` is not null are retained, ensuring the cleaned dataset is
anchored to concrete clinical context.

### PII anonymization

`anonymisation.py` wraps Microsoft Presidio with a bilingual spaCy NLP engine. Language is
auto-detected per row via `langdetect` (fallback: `"en"`). Five entity types are replaced by
structured tags:

| Entity type | Replacement tag |
|---|---|
| `PERSON` | `<PERSON>` |
| `EMAIL_ADDRESS` | `<EMAIL>` |
| `PHONE_NUMBER` | `<PHONE>` |
| `DATE_TIME` | `<DATE>` |
| `LOCATION` | `<LOCATION>` |

For SFT, anonymization is applied to `question` and `answer`. For DPO, it covers `question`,
`chosen`, and `rejected`.

### Token counting

`add_token_counts()` loads the `Qwen/Qwen3-1.7B-Base` tokenizer once per process (via
`@lru_cache`) and adds a `token_count_<col>` column for each specified text column. This enables
length-aware filtering at the fine-tuning stage without re-tokenizing.

### Balanced multi-source sampling

`collect_balanced_samples()` distributes the `target_samples` quota evenly across all source
Parquet files. When a file has fewer rows than its share, the shortfall is redistributed to the
remaining files in the same pass — ensuring the final count stays as close to the target as
possible without oversampling any single source.

### Stratified splitting

`split_dataset()` uses `sklearn.model_selection.train_test_split` with `stratify=df['dataset_name']`
at both split levels (train/rest, then val/test), preserving source proportions across all three
splits.

### Dataset sizes

| Dataset | Format | Split | Raw rows | Processed rows |
|---|---|---|---|---|
| MediQAL MCQU | MCQ (FR) | train | 10 113 | — |
| MediQAL MCQU | MCQ (FR) | validation | 2 561 | — |
| MediQAL MCQU | MCQ (FR) | test | 4 343 | — |
| MediQAL MCQM | MCQ (FR) | train | 5 767 | — |
| MediQAL MCQM | MCQ (FR) | validation | 1 466 | — |
| MediQAL MCQM | MCQ (FR) | test | 3 384 | — |
| MediQAL OEQ | Open QA (FR) | test | 4 969 | — |
| FrenchMedMCQA | MCQ (FR) | train | 595 | — |
| FrenchMedMCQA | MCQ (FR) | validation | 164 | — |
| FrenchMedMCQA | MCQ (FR) | test | 321 | — |
| MedQuad | Open QA (EN) | train | 16 407 | `medquad.parquet` |
| UltraMedical-Preference | Conversational (EN) | train | 109 353 | `ultramed_sft.parquet` + `ultramed_dpo.parquet` |
| **SFT dataset** | `(question, answer)` | full | — | **5 000** (train 3 500 / val 1 000 / test 500) |
| **DPO dataset** | `(question, chosen, rejected)` | full | — | **5 000** (train 3 500 / val 1 000 / test 500) |

> 🔹 SFT schema: `question`, `answer`, `language`, `question_type`, `confidence_level`,
> `dataset_name`, `token_count_question`, `token_count_answer`

> 🔹 DPO schema: `question`, `chosen`, `rejected`, `language`, `question_type`,
> `confidence_level`, `dataset_name`, `token_count_question`, `token_count_chosen`,
> `token_count_rejected`

---

## 🚀 Installation

### Prerequisites

- Python **3.11+** (3.10 minimum pour la CI, 3.11 recommandé en local)
- [uv](https://docs.astral.sh/uv/) package manager
- [DVC](https://dvc.org/) **3.67+** (included in the `dev` dependency group)
- A Google Cloud project with a GCS bucket named `p14-medical-data` (or equivalent)
- Application Default Credentials configured: `gcloud auth application-default login`
- **For training**: NVIDIA GPU with ≥16 GB VRAM (T4, L4, A100) and CUDA toolkit

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/RandomFab/FINE-TUNING_MEDICAL.git
cd FINE-TUNING_MEDICAL

# 2. Create and activate the virtual environment (includes dev dependencies)
uv sync --all-groups

# On Linux / macOS
source .venv/bin/activate

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Download spaCy language models (required by the anonymization module)
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md
```
### Running the API locally (without GPU)

```bash
# Install API dependencies
uv pip install fastapi uvicorn pydantic httpx

# Run with mocked vLLM engine (no GPU required)
uvicorn src.api.main:app --reload --port 8000
# → Swagger UI available at http://localhost:8000/docs
```

### Running the API in Docker (with GPU)

```bash
docker build -t medical-qwen-api .
docker run -d --name medical-api --gpus all -p 8000:8000 \
  -v /path/to/models:/app/models \
  -e GCS_MERGED_MODEL_PATH=/app/models/merged_model_for_deployment \
  medical-qwen-api
```

### Running tests

```bash
# Unit + integration + smoke (no GPU required)
pytest tests/ -v

# Expected output: 70 passed
```

### Launching DPO training (after SFT champion is available in MLflow)

```bash
MLFLOW_TRACKING_URI=http://<your-mlflow-server>:5000 \
MLFLOW_EXPERIMENT_NAME=dpo-qwen3-medical \
python -m src.training.train_dpo
```

### Running the ingestion notebook

```bash
jupyter notebook notebooks/Import_data_from_HuggingFace.ipynb
```

> 💡 A Hugging Face token is optional but recommended to avoid rate-limiting on large datasets
> (e.g. UltraMedical-Preference at ~994 MB). Set it via `export HF_TOKEN=<your_token>` before
> launching the notebook.

### Running the full data pipeline (Step 1)

```bash
# Reproduce all 6 stages in order, using DVC cache
dvc repro
```

### Running individual data stages

```bash
# Stage 1 — Clean MediQAL
python -m src.processing.mediqal_cleaning

# Stage 2 — Clean FrenchMedMCQA
python -m src.processing.frenchmedmcqa_cleaning

# Stage 3 — Clean MedQuad
python -m src.processing.medquad_cleaning

# Stage 4 — Clean UltraMedical (outputs both SFT and DPO variants)
python -m src.processing.ultramed_cleaning

# Stage 5 — Generate SFT dataset (reads params.yaml)
python -m src.processing.sft_dataset.generate_sft_dataset

# Stage 6 — Generate DPO dataset (reads params.yaml)
python -m src.processing.dpo_dataset.generate_dpo_dataset
```

All scripts read from `gs://p14-medical-data/raw_data/` (stages 1–4) or from
`data/processed/` (stages 5–6), and write Parquet files to `data/processed/`.

> ⚠️ Make sure the GCS bucket exists and your service account has `Storage Object Admin` rights
> before running stages 1–4.

### Running SFT training (Step 2)

```bash
# Requires GPU — run on Colab, GCP VM, or local GPU machine
python -m src.training.train_sft
```

Training outputs:
- Checkpoints: `models/sft_checkpoints/` (auto-managed, max 3 kept)
- Final model: `models/lora_trained_model/` (LoRA adapter weights only)
- Metrics: logged to MLflow (configure via `MLFLOW_TRACKING_URI`)

> 💡 On Google Colab, upload the project files and run the training script in a notebook cell.
> The checkpoint resume mechanism handles session timeouts automatically.

---

## 🌐 API Reference

The deployed endpoint exposes two routes:

### `GET /health`
Returns `200` if the vLLM engine is loaded and ready, `503` otherwise.

```json
{"status": "healthy", "model": "merged_qwen3_medical"}
```

### `POST /generate`
Generates a medical triage response.

**Request body:**
```json
{
  "prompt": "Patient se plaint de douleurs thoraciques irradiant dans le bras gauche depuis 20 minutes.",
  "max_tokens": 512,
  "temperature": 0.1
}
```

**Response:**
```json
{
  "response": "Niveau de priorité : URGENCE MAXIMALE. ..."
}
```

**Validation constraints:**
- `prompt`: 10–4096 characters
- `max_tokens`: 1–2048
- `temperature`: 0.0–2.0

Swagger UI is available at `http://<IP_VM>:8000/docs`.

---

## 🔭 Roadmap

This project follows a 4-week mission plan for the CHSA medical triage POC:

- [x] **Week 1** — Data preparation: corpus ingestion, cleaning, anonymization, SFT/DPO dataset generation
- [x] **Week 2** — SFT training: QLoRA fine-tuning of Qwen3-1.7B-Base with LoRA adapters
- [ ] **Week 3** — DPO alignment: preference optimization using UltraMedical chosen/rejected pairs
- [ ] **Week 4** — Deployment: vLLM endpoint, FastAPI API, CI/CD pipeline, final evaluation & report

---

## 👤 Author

**RandomFab - Fabien BARDOUIL**