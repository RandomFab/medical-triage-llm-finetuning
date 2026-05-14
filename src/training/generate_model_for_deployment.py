import os

import mlflow
from transformers import AutoModelForCausalLM

from config.logger import logger
from config.paths import LOCAL_MERGED_MODEL_PATH
from src.training.utils_training import (
    _get_model_name,
    _get_qwen_tokenizer,
    _load_dpo_lora_adapter,  # ← DPO, pas SFT
)

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))


def load_base_model():
    """Charge le modèle de base en précision complète (float32) pour le merge.
    
    On charge en float32 et non en 4-bit : le merge LoRA doit s'effectuer
    sur les poids originaux non quantifiés. La quantification est uniquement
    un outil d'entraînement, pas une propriété permanente du modèle.
    """
    logger.info("Chargement du modèle de base en précision standard (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        _get_model_name(),
        device_map="cpu",  # CPU suffit pour le merge — pas d'inférence ici
    )
    logger.info("Modèle de base chargé avec succès.")
    return model


def merge_base_model_with_lora_adapter(base_model, lora_adapter_path: str):
    """Fusionne les adaptateurs LoRA avec le modèle de base.
    
    merge_and_unload() intègre mathématiquement les matrices LoRA (A·B)
    dans les poids originaux W, puis supprime les adaptateurs.
    Résultat : un modèle standard sans dépendance PEFT, 
    directement chargeable par vLLM.
    """
    from peft import PeftModel

    logger.info(
        f"Chargement de l'adaptateur DPO depuis {lora_adapter_path} "
        "et fusion avec le modèle de base..."
    )
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        is_trainable=False,  # mode inférence, pas d'entraînement
    )
    model = model.merge_and_unload()
    logger.info("Fusion effectuée avec succès.")
    return model


def save_model_for_deployment(model, output_path):
    """Sauvegarde le modèle mergé localement avant push vers GCS."""
    logger.info(f"Sauvegarde du modèle fusionné vers {output_path}...")
    model.save_pretrained(output_path)
    logger.info("Modèle sauvegardé avec succès.")


def main():
    logger.info("=" * 60)
    logger.info("Génération du modèle DPO mergé pour le déploiement")
    logger.info("=" * 60)

    # 1. Chargement du modèle de base (float32, CPU)
    base_model = load_base_model()

    # 2. Récupération des adaptateurs DPO depuis MLflow/GCS
    lora_adapter_path = _load_dpo_lora_adapter()

    # 3. Merge LoRA → modèle monolithique
    merged_model = merge_base_model_with_lora_adapter(base_model, lora_adapter_path)

    # 4. Sauvegarde locale
    save_model_for_deployment(merged_model, LOCAL_MERGED_MODEL_PATH)

    # 5. Tokenizer (toujours celui de Qwen3-1.7B-Base, inchangé par le fine-tuning)
    logger.info("Sauvegarde du tokenizer...")
    tokenizer = _get_qwen_tokenizer()
    tokenizer.save_pretrained(LOCAL_MERGED_MODEL_PATH)
    logger.info("Tokenizer sauvegardé.")

    # 6. Push vers GCS via MLflow
    logger.info("Push du modèle mergé vers GCS via MLflow...")
    with mlflow.start_run(
        run_name="merge_dpo_model_for_deployment",
        tags={"stage": "deployment", "source": "dpo"},
    ):
        mlflow.log_artifacts(
            str(LOCAL_MERGED_MODEL_PATH),
            artifact_path="merged_model_for_deployment",
        )
    logger.info("=" * 60)
    logger.info("✅ Modèle DPO mergé et pushé sur GCS avec succès.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()