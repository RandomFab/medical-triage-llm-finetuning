import pandas as pd
import yaml
import re

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

    chat_text = format_qwen_chat(question, answer)
    prompt_text = format_qwen_prompt(question)

    input_ids = tokenizer.encode(chat_text)
    prompt_ids = tokenizer.encode(prompt_text)

    idx = len(prompt_ids)

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
