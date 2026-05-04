import pandas as pd
import yaml
import re
import mlflow
import torch

from config.logger import logger
from config.paths import PARAMS_PATH, PROJECT_ROOT


# === load dataset functions ===
def load_dataset(dataset_path: str):
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = pd.read_parquet(dataset_path)
    logger.info(f"Dataset loaded successfully from {dataset_path}")
    return dataset


# === formatting functions for Qwen ===
from functools import lru_cache


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


# === tokenization functions for Qwen ===
def tokenize_chat(question: str, answer: str) -> dict:
    tokenizer = _get_qwen_tokenizer()
    max_length = _get_max_length()

    chat_text = format_qwen_chat(question, answer)
    prompt_text = format_qwen_prompt(question)

    input_ids = tokenizer.encode(chat_text, truncation=True, max_length=max_length)
    prompt_ids = tokenizer.encode(prompt_text)

    idx = min(len(prompt_ids), len(input_ids))

    labels = input_ids.copy()
    labels[:idx] = [-100] * idx

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def decode_token(token_id: int) -> str: # → fonction de debuggage : permet de décoder un token pour l'utilisateur
    tokenizer = _get_qwen_tokenizer()

    decoded_text = tokenizer.decode([token_id])

    return decoded_text


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
    train_args = params["training_arguments"]
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


if __name__ == "__main__":
    dataset = load_dataset(
        PROJECT_ROOT / "data/processed/sft_dataset/sft_train.parquet"
    )
    print(dataset.iloc[0]["question"])
    print(dataset.iloc[0]["answer"])
    token = tokenize_chat(dataset.iloc[0]["question"], dataset.iloc[0]["answer"])

    tokenizer = _get_qwen_tokenizer()
    real_labels = [t for t in token["labels"] if t != -100]
    print(tokenizer.decode(real_labels))
