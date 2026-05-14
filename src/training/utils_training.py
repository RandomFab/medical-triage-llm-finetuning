import pandas as pd
import yaml
import re
import mlflow
import torch
from pathlib import Path
from typing import Literal
from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOConfig

from config.logger import logger
from config.paths import PARAMS_PATH
import os
from functools import lru_cache

# Configuration explicite de MLflow via les variables d'environnement
if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
if "MLFLOW_EXPERIMENT_NAME" in os.environ:
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])


# === load dataset functions ===
def load_dataset(dataset_path: str):
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = pd.read_parquet(dataset_path)
    logger.info(f"Dataset loaded successfully from {dataset_path}")
    return dataset


def transform_ds_from_pandas_to_hf(dataset: pd.DataFrame) -> Dataset:
    logger.info("Transforming dataset from pandas DataFrame to Hugging Face Dataset...")
    hf_dataset = Dataset.from_pandas(dataset)
    logger.info("Dataset transformed successfully to Hugging Face Dataset")
    return hf_dataset

# === formatting functions for Qwen ===



@lru_cache(maxsize=1)
def _get_qwen_tokenizer():
    """Chargé une seule fois par process (lru_cache)."""

    logger.info("Loading Qwen tokenizer...")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")

    logger.info("Qwen tokenizer loaded successfully")
    return tokenizer


@lru_cache(maxsize=1)
def _load_params():
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

@lru_cache(maxsize=1)
def _get_system_prompt():
    """Chargé une seule fois par process (lru_cache)."""

    logger.info("Loading system prompt...")

    params = _load_params()
    system_prompt = params["sft_model"]["system_prompt"]

    logger.info(f"System prompt loaded successfully: {system_prompt}")

    return system_prompt

@lru_cache(maxsize=1)
def _get_max_length():
    """Chargé une seule fois par process (lru_cache)."""

    logger.info("Loading max length...")

    params = _load_params()
    max_length = params["sft_model"]["max_length"]

    logger.info(f"Max length loaded successfully: {max_length}")

    return max_length

@lru_cache(maxsize=1)
def _get_model_name():
    params = _load_params()
    return params["sft_model"]["model_name"]


@lru_cache(maxsize=1)
def _get_quantization_config():
    params = _load_params()
    return params["quantization_config"]

def _clean_thinking_markers(text: str) -> str:
    return re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL).strip()


def _apply_chat_template(chat: list[dict], add_generation_prompt: bool, tokenize: bool) -> str | list[int]:
    tokenizer = _get_qwen_tokenizer()
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=add_generation_prompt, tokenize=tokenize
    )


def format_qwen_chat(question: str, answer: str) -> str: # → fonction de debuggage : permet de voir le formatage pour l'utilisateur
    system_prompt = _get_system_prompt()
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return _clean_thinking_markers(
        _apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
    )


def format_qwen_prompt(question: str) -> str: # → fonction de debuggage : permet de voir le formatage pour l'utilisateur
    system_prompt = _get_system_prompt()
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return  _clean_thinking_markers(
        _apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    )

def format_dpo_chat(question: str, chosen_answer: str, rejected_answer: str) -> dict:
    system_prompt = _get_system_prompt()
    chat = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "chosen": [
            {"role": "assistant", "content": chosen_answer}
        ],
        "rejected": [
            {"role": "assistant", "content": rejected_answer}
        ]
    }
    return chat


def decode_token(token_id: int) -> str: # → fonction de debuggage : permet de décoder un token pour l'utilisateur
    tokenizer = _get_qwen_tokenizer()

    decoded_text = tokenizer.decode([token_id])

    return decoded_text


# === training arguments ===

@lru_cache(maxsize=2)
def _get_config_training_arguments(stage: Literal["sft", "dpo"]):
    params = _load_params()
    return params["training_arguments"][stage]


def define_training_arguments(
    stage: Literal["sft", "dpo"],
    checkpoint_output_dir: Path,
) -> TrainingArguments | DPOConfig:

    logger.info(f"loading {stage} training arguments from params.yaml")
    params_training_args = _get_config_training_arguments(stage)
    
    kwargs = dict(
        output_dir=checkpoint_output_dir,
        per_device_train_batch_size=params_training_args["per_device_train_batch_size"],
        per_device_eval_batch_size=params_training_args["per_device_eval_batch_size"],
        gradient_accumulation_steps=params_training_args["gradient_accumulation_steps"],
        learning_rate=float(params_training_args["learning_rate"]),
        num_train_epochs=params_training_args["num_train_epochs"],
        warmup_steps=params_training_args["warmup_steps"],
        lr_scheduler_type=params_training_args["lr_scheduler_type"],
        eval_strategy=params_training_args["eval_strategy"],
        eval_steps=params_training_args["eval_steps"],
        save_strategy=params_training_args["save_strategy"],
        save_steps=params_training_args["save_steps"],
        save_total_limit=params_training_args["save_total_limit"],
        logging_steps=params_training_args["logging_steps"],
        load_best_model_at_end=params_training_args["load_best_model_at_end"],
        bf16=params_training_args["bf16"],
        fp16=params_training_args["fp16"],
        gradient_checkpointing=params_training_args["gradient_checkpointing"],
        gradient_checkpointing_kwargs=params_training_args["gradient_checkpointing_kwargs"],
        optim=params_training_args["optim"],
        metric_for_best_model=params_training_args["metric_for_best_model"],
        greater_is_better=params_training_args["greater_is_better"],
        report_to=params_training_args["report_to"],
    )

    if stage == "dpo":
        training_args = DPOConfig(beta=0.1,max_length=512,  **kwargs)
    else:
        training_args = TrainingArguments(**kwargs)

    logger.info(f"{stage} training arguments defined successfully")

    return training_args


# === MLflow helpers ===

def _get_gpu_name() -> str:
    if torch.cuda.is_available():
        raw = torch.cuda.get_device_name(0)
        # "Tesla T4" → "T4", "NVIDIA A100-SXM4-40GB" → "A100"
        for known in ("T4", "A100", "A10G", "V100", "L4", "H100", "3090", "4090"):
            if known in raw:
                return known
        return raw.split()[-1]
    return "cpu"


def setup_mlflow_run(stage: str) -> mlflow.ActiveRun:
    """Démarre un run MLflow nommé et taggé à partir de params.yaml.

    À appeler en début de main() avant le Trainer (qui réutilisera le run actif).
    stage : "sft" | "dpo"
    """
    params = _load_params()
    lora = params["lora_config"]
    quant = params["quantization_config"]
    train_args = params["training_arguments"][stage]
    model_name: str = params["sft_model"]["model_name"]   # "Qwen/Qwen3-1.7B-Base"

    short_model = model_name.split("/")[-1].lower()       # "qwen3-1.7b-base"
    dtype = "fp16" if train_args.get("fp16") else "bf16"
    gpu = _get_gpu_name()

    run_name = f"{stage}_{short_model}_qlora_r{lora['r']}_{dtype}_{gpu}"

    tags = {
        "stage":         stage,
        "model":         model_name,
        "method":        "QLoRA",
        "hardware":      gpu,
        "quantization":  f"4bit-{quant['bnb_4bit_quant_type']}-{dtype}",
        "lora_r":        str(lora["r"]),
        "lora_alpha":    str(lora["lora_alpha"]),
        "epochs":        str(train_args["num_train_epochs"]),
        "learning_rate": str(train_args["learning_rate"]),
        "dataset_size":  str(params[stage]["target_samples"]),
    }

    logger.info(f"Starting MLflow run: {run_name}")
    return mlflow.start_run(run_name=run_name, tags=tags)

def _load_sft_lora_adapter():

    logger.info("Loading SFT LoRA adapter from the latest champion run...")
    experiment = mlflow.get_experiment_by_name("sft-qwen3-medical")

    runs = mlflow.search_runs(
        filter_string='tags.model_status = "champion" and tags.stage = "sft"',
        order_by=["start_time DESC"],
        max_results=1,
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        raise ValueError(
            "No champion run found for SFT stage. Please train an SFT model first."
        )

    run_id = runs.iloc[0].run_id

    logger.info(f"Found champion run with ID: {run_id}")

    model = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/lora_trained_model")

    return model

def _load_dpo_lora_adapter() -> str:
    """Récupère les adaptateurs LoRA du meilleur run DPO depuis MLflow.
    
    Cherche dans l'expérience 'sft-qwen3-medical' le run le plus récent
    taggé model_status=champion et stage=dpo.
    """
    logger.info("Loading DPO LoRA adapter from the latest champion run...")
    experiment = mlflow.get_experiment_by_name("sft-qwen3-medical")

    if experiment is None:
        raise ValueError(
            "Experiment 'sft-qwen3-medical' introuvable. "
            "Vérifie que le run DPO a bien été lancé avec MLFLOW_EXPERIMENT_NAME=sft-qwen3-medical."
        )

    runs = mlflow.search_runs(
        filter_string='tags.model_status = "champion" and tags.stage = "dpo"',
        order_by=["start_time DESC"],
        max_results=1,
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        raise ValueError(
            "No champion run found for DPO stage. "
            "Please train a DPO model first (train_dpo.py)."
        )

    run_id = runs.iloc[0].run_id
    logger.info(f"Found DPO champion run with ID: {run_id}")

    # Le DPOTrainer sauvegarde dans 'dpo_model_trained' (voir train_dpo.py)
    adapter_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/dpo_model_trained"
    )
    return adapter_path
