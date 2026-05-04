# 🩺 Fine-Tuning Medical — French Medical QA: Data Pipeline & SFT Training

A machine learning project that ingests, cleans, anonymizes, and prepares French-language medical QA
datasets, then fine-tunes Qwen3-1.7B-Base via QLoRA for a medical triage assistant. Targets multiple
Hugging Face sources, normalizes them into unified SFT `(question, answer)` and DPO
`(question, chosen, rejected)` schemas, tokenizes with Qwen3 chat template, and trains with LoRA
adapters — all orchestrated through reproducible DVC pipelines and tracked with MLflow.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-150458.svg?logo=pandas&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-FF6F00.svg)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-4bit-green.svg)
![GCS](https://img.shields.io/badge/Google_Cloud_Storage-3.10.1-4285F4.svg?logo=googlecloud&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.67+-945DD6.svg?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg?logo=mlflow&logoColor=white)

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
│   ├── 📂 processing/
│   │   ├── __init__.py
│   │   ├── mediqal_cleaning.py               # MediQAL cleaning pipeline
│   │   ├── frenchmedmcqa_cleaning.py         # FrenchMedMCQA cleaning pipeline
│   │   ├── medquad_cleaning.py               # MedQuad cleaning pipeline
│   │   ├── ultramed_cleaning.py              # UltraMedical SFT + DPO cleaning
│   │   ├── anonymisation.py                  # Presidio + spaCy PII anonymizer (FR/EN)
│   │   ├── utils_cleaning.py                 # Shared utilities (sampling, splitting, tokens…)
│   │   ├── 📂 sft_dataset/
│   │   │   └── generate_sft_dataset.py       # Stage 5 — balanced SFT assembly
│   │   └── 📂 dpo_dataset/
│   │       └── generate_dpo_dataset.py       # Stage 6 — DPO preference assembly
│   └── 📂 training/
│       ├── __init__.py
│       ├── utils_training.py                 # Chat template, tokenization, label masking
│       └── train_sft.py                      # SFT training: QLoRA + LoRA + Trainer
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

```yaml
sft:
  target_samples: 5000     # Total rows in the final SFT dataset
  random_state: 42          # Reproducibility seed
  val_size: 0.2             # Fraction allocated to validation
  test_size: 0.1            # Fraction allocated to test
  source_datasets:          # Parquet files relative to data/processed/
    - mediqal_dataset/mediqal.parquet
    - frenchmedmcqa_dataset/frenchmedmcqa.parquet
    - medquad_dataset/medquad.parquet
    - ultramed_dataset/ultramed_SFT.parquet

dpo:
  target_samples: 5000
  random_state: 42
  val_size: 0.2
  test_size: 0.1
  source_datasets:
    - ultramed_dataset/ultramed_DPO.parquet
```

#### LoRA & model configuration

```yaml
lora_config:
  r: 16                    # Rank of LoRA adapters
  lora_alpha: 32            # Scaling factor (alpha/r = 2)
  target_modules:           # All linear projections in Qwen3
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  task_type: "CAUSAL_LM"

sft_model:
  model_name: "Qwen/Qwen3-1.7B-Base"
  system_prompt: "Tu es un agent de triage médical aux urgences..."
  max_length: 512           # Max sequence length (tokens)

quantization_config:
  load_in_4bit: True
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: True
  bnb_4bit_quant_type: "nf4"
```

#### Training arguments

```yaml
training_arguments:
  per_device_train_batch_size: 1    # Physical batch (T4 constraint)
  gradient_accumulation_steps: 32   # Effective batch = 1 × 32 = 32
  learning_rate: 2e-4               # Higher LR for LoRA (frozen base)
  num_train_epochs: 3
  warmup_steps: 30                  # ~10% of total steps
  lr_scheduler_type: "cosine"
  eval_strategy: "steps"
  eval_steps: 50
  save_strategy: "steps"
  save_steps: 50
  save_total_limit: 3
  logging_steps: 10
  load_best_model_at_end: True
  bf16: False                       # T4 does not support bf16
  fp16: True
  gradient_checkpointing: True
  optim: "paged_adamw_8bit"
  metric_for_best_model: "eval_loss"
  greater_is_better: False
  report_to: "mlflow"
```

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

- Python **3.13** (enforced via `.python-version`)
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

## 🔭 Roadmap

This project follows a 4-week mission plan for the CHSA medical triage POC:

- [x] **Week 1** — Data preparation: corpus ingestion, cleaning, anonymization, SFT/DPO dataset generation
- [x] **Week 2** — SFT training: QLoRA fine-tuning of Qwen3-1.7B-Base with LoRA adapters
- [ ] **Week 3** — DPO alignment: preference optimization using UltraMedical chosen/rejected pairs
- [ ] **Week 4** — Deployment: vLLM endpoint, FastAPI API, CI/CD pipeline, final evaluation & report

---

## 👤 Author

**RandomFab - Fabien BARDOUIL**