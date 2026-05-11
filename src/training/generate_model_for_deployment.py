import os

from transformers import AutoModelForCausalLM
from config.paths import GCS_MERGED_MODEL_PATH, LOCAL_MERGED_MODEL_PATH
from src.training.utils_training import (
    _get_model_name,
    _load_sft_lora_adapter,
    _get_qwen_tokenizer,
)
from config.logger import logger

import mlflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

def load_base_model():
    """Load base model in full precision for merge and deployment."""
    logger.info("Chargement du modèle de base en précision standard...")
    model = AutoModelForCausalLM.from_pretrained(_get_model_name(), device_map="cpu")
    logger.info("Modèle de base chargé avec succès.")
    return model


def merge_base_model_with_lora_adapter(base_model, lora_adapter_path):
    """Load the LoRA adapter and merge it with the base model."""
    from peft import PeftModel

    logger.info(
        f"Chargement de l'adaptateur LoRA depuis {lora_adapter_path} et fusion avec le modèle de base..."
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path, is_trainable=False)
    model = model.merge_and_unload()
    logger.info("Fusion effectuée avec succès.")
    return model


def save_model_for_deployment(model, output_path):
    """Save the merged model for deployment."""
    logger.info(
        f"Sauvegarde du modèle fusionné pour le déploiement vers {output_path}..."
    )
    model.save_pretrained(output_path)
    logger.info("Modèle sauvegardé avec succès.")


def main():
    logger.info("Début de la génération du modèle pour le déploiement.")
    base_model = load_base_model()
    lora_adapter_model = _load_sft_lora_adapter()
    merged_model = merge_base_model_with_lora_adapter(base_model, lora_adapter_model)

    save_model_for_deployment(merged_model, LOCAL_MERGED_MODEL_PATH)
    logger.info("Génération du modèle terminée.")

    logger.info("Chargement du tokenizer...")
    tokenizer = _get_qwen_tokenizer()
    logger.info("Tokenizer chargé avec succès.")
    logger.info(f"Sauvegarde du tokenizer vers {LOCAL_MERGED_MODEL_PATH}...")
    tokenizer.save_pretrained(LOCAL_MERGED_MODEL_PATH)
    logger.info("Tokenizer sauvegardé avec succès.")
    with mlflow.start_run(run_name="merge_model_for_deployment", tags={"stage": "deployment"}):
        
        mlflow.log_artifacts(
            str(LOCAL_MERGED_MODEL_PATH),
            artifact_path="merged_model_for_deployment",
        )


if __name__ == "__main__":
    main()
