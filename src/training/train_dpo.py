import pandas as pd
import os
import mlflow.artifacts
from peft import PeftModel
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from config.logger import logger
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from config.paths import ROOT_MODEL_DIR, DPO_TRAIN_DATASET_PATH, DPO_VAL_DATASET_PATH
from src.training.utils_training import (
    define_training_arguments,
    _load_sft_lora_adapter,
    _get_quantization_config,
    _get_model_name,
    setup_mlflow_run,
    transform_ds_from_pandas_to_hf,
    _get_qwen_tokenizer,
    format_dpo_chat,
)
from peft import prepare_model_for_kbit_training
from datasets import Dataset
import mlflow


def define_model() -> PeftModel:
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
        torch_dtype=torch.float16,
    )

    # ← ajouté : cast des layer norms en fp32, désactivation du KV cache,
    #   activation de input_require_grads pour le gradient checkpointing
    model_4bit = prepare_model_for_kbit_training(
        model_4bit,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    sft_model = _load_sft_lora_adapter()
    model = PeftModel.from_pretrained(
        model_4bit,
        sft_model,
        device_map="auto",
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    model = model.to(torch.float16) 
    
    return model


def prepare_dpo_dataset(dataset: pd.DataFrame) -> Dataset:

    dataset_hf = transform_ds_from_pandas_to_hf(dataset)

    formated_dataset = dataset_hf.map(
        lambda x: format_dpo_chat(x["question"], x["chosen"], x["rejected"]),
        batched=False,
        remove_columns=dataset_hf.column_names,
    )

    return formated_dataset


def train_dpo_model(
    model: PeftModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: DPOConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DPOTrainer:

    trainer = DPOTrainer(
        model=model,
        args=training_args,  # DPOConfig
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # le tokenizer Qwen3
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

    trainer.save_model(ROOT_MODEL_DIR / "dpo_model_trained")

    mlflow.log_artifacts(
        str(ROOT_MODEL_DIR / "dpo_model_trained"),
        artifact_path="dpo_model_trained",
    )
    mlflow.set_tag("model_status", "champion")
    logger.info("Model trained successfully")

    return trainer


def main():

    print("LANCEMENT DPO VERSION 2 AVEC FLOAT32")
    logger.info("Starting DPO training...")

    with setup_mlflow_run(stage="dpo"):

        # === Read parquet files ===
        dpo_train_dataset = pd.read_parquet(DPO_TRAIN_DATASET_PATH)
        dpo_val_dataset = pd.read_parquet(DPO_VAL_DATASET_PATH)

        # === Prepare datasets ===
        dpo_train_dataset_prepared = prepare_dpo_dataset(dpo_train_dataset)
        dpo_val_dataset_prepared = prepare_dpo_dataset(dpo_val_dataset)

        # === training arguments ===
        training_args = define_training_arguments(
            stage="dpo", checkpoint_output_dir=ROOT_MODEL_DIR / "dpo_checkpoints"
        )

        # === model definition ===
        model = define_model()

        # === training ===

        train_dpo_model(
            model=model,
            train_dataset=dpo_train_dataset_prepared,
            eval_dataset=dpo_val_dataset_prepared,
            training_args=training_args,
            tokenizer=_get_qwen_tokenizer(),
        )

        logger.info("DPO training completed successfully")


if __name__ == "__main__":
    main()
