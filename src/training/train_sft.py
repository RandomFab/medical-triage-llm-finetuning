from functools import lru_cache
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
import os
import mlflow

from datasets import Dataset
import pandas as pd
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

from config.logger import logger
from config.paths import (
    ROOT_MODEL_DIR,
    SFT_TRAIN_DATASET_PATH,
    SFT_VAL_DATASET_PATH,
    SFT_TEST_DATASET_PATH,
)
from src.training.utils_training import (
    _get_qwen_tokenizer,
    _get_max_length,
    load_dataset,
    _load_params,
    setup_mlflow_run,
    format_qwen_chat,
    format_qwen_prompt,
    define_training_arguments,
)


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


def transform_ds_from_pandas_to_hf(dataset: pd.DataFrame) -> Dataset:
    logger.info("Transforming dataset from pandas DataFrame to Hugging Face Dataset...")
    hf_dataset = Dataset.from_pandas(dataset)
    logger.info("Dataset transformed successfully to Hugging Face Dataset")
    return hf_dataset


def apply_tokenisation(dataset_hf: Dataset) -> Dataset:
    logger.info("Applying tokenization to the dataset...")
    tokenized_dataset = dataset_hf.map(
        lambda x: tokenize_chat(x["question"], x["answer"]),
        batched=False,
        remove_columns=dataset_hf.column_names,
    )
    logger.info("Tokenization applied successfully to the dataset")
    return tokenized_dataset


def tokenize_flow(pd_dataset_path: Path) -> Dataset:
    ds_name = pd_dataset_path.stem
    logger.info(f"===== Tokenizing {ds_name} =====")

    pd_dataset = load_dataset(pd_dataset_path)
    hf_dataset = transform_ds_from_pandas_to_hf(pd_dataset)
    tokenized_dataset = apply_tokenisation(hf_dataset)

    logger.info(f"===== {ds_name} tokenized successfully =====")
    return tokenized_dataset


def get_data_collator(tokenizer):
    logger.info(f"Generating data collator for sequence-to-sequence...")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    logger.info("Data collator generated successfully")
    return data_collator


@lru_cache(maxsize=1)
def _get_lora_config():
    params = _load_params()
    return params["lora_config"]


@lru_cache(maxsize=1)
def _get_model_name():
    params = _load_params()
    return params["sft_model"]["model_name"]


@lru_cache(maxsize=1)
def _get_quantization_config():
    params = _load_params()
    return params["quantization_config"]


def define_model():

    quantization_config = _get_quantization_config()
    quantization = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_quant_type=quantization_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quantization_config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(
            torch, quantization_config["bnb_4bit_compute_dtype"], None
        ),
    )

    model_name = _get_model_name()
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
    )

    # ← ajouté : cast des layer norms en fp32, désactivation du KV cache,
    #   activation de input_require_grads pour le gradient checkpointing
    model_4bit = prepare_model_for_kbit_training(
        model_4bit,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    config = _get_lora_config()
    lora_config = LoraConfig(
        task_type=config["task_type"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )

    model = get_peft_model(model_4bit, lora_config)
    return model


def train_model(
    training_args: TrainingArguments,
    model: PeftModel | PeftMixedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: DataCollatorForSeq2Seq,
) -> Trainer:
    logger.info("Training model...")
    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        logger.info(f"Resuming training from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("Starting training from scratch...")
        trainer.train()

    trainer.save_model(ROOT_MODEL_DIR / "lora_trained_model")

    mlflow.log_artifacts(
        str(ROOT_MODEL_DIR / "lora_trained_model"),
        artifact_path="lora_trained_model",
    )
    mlflow.set_tag("model_status", "champion")
    logger.info("Model trained successfully")
    return trainer


def main():
    logger.info("Starting SFT training...")

    with setup_mlflow_run(stage="sft"):

        # === tokenization ===
        tokenized_train_ds = tokenize_flow(SFT_TRAIN_DATASET_PATH)
        tokenized_eval_ds = tokenize_flow(SFT_VAL_DATASET_PATH)

        # === data collator ===
        tokenizer = _get_qwen_tokenizer()
        data_collator = get_data_collator(tokenizer)

        # === training arguments ===
        training_args = define_training_arguments(
            stage="sft",
            checkpoint_output_dir=ROOT_MODEL_DIR / "sft_checkpoints",
        )

        # === model definition ===
        model = define_model()

        # === training ===
        train_model(
            training_args=training_args,
            model=model,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_eval_ds,
            data_collator=data_collator,
        )

    logger.info("SFT training completed successfully")


if __name__ == "__main__":
    main()
