import os

import mlflow
from transformers import AutoModelForCausalLM

from config.logger import logger
from config.paths import LOCAL_MERGED_DPO_MODEL_PATH,LOCAL_MERGED_SFT_MODEL_PATH
from src.training.utils_training import (
    _get_model_name,
    _get_qwen_tokenizer,
    _load_sft_lora_adapter,
    _load_dpo_lora_adapter,
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


def save_models_for_deployment(model, output_path):
    """Sauvegarde le modèle mergé localement avant push vers GCS."""
    logger.info(f"Sauvegarde du modèle fusionné vers {output_path}...")
    model.save_pretrained(output_path)
    logger.info("Modèle sauvegardé avec succès.")


def main():
    logger.info("=" * 60)
    logger.info("Génération des modèles SFT et DPO mergés pour le déploiement")
    logger.info("=" * 60)

    # 1. Chargement du modèle de base (float32, CPU) - partagé
    base_model = load_base_model()
    tokenizer = _get_qwen_tokenizer()

    # Chemins locaux de sauvegarde
    sft_output_path = LOCAL_MERGED_SFT_MODEL_PATH
    dpo_output_path = LOCAL_MERGED_DPO_MODEL_PATH

    # ==========================
    # TRAITEMENT DU MODELE SFT
    # ==========================
    logger.info("--- Début du traitement SFT ---")
    sft_adapter_path = _load_sft_lora_adapter()
    sft_merged_model = merge_base_model_with_lora_adapter(base_model, sft_adapter_path)
    
    save_models_for_deployment(sft_merged_model, sft_output_path)
    tokenizer.save_pretrained(sft_output_path)
    
    logger.info("Push du modèle SFT mergé vers GCS via MLflow...")
    with mlflow.start_run(
        run_name="merge_sft_model_for_deployment",
        tags={"stage": "deployment", "source": "sft"},
    ):
        mlflow.log_artifacts(
            str(sft_output_path),
            artifact_path="merged_model_sft_for_deployment",
        )

    # Libération de la mémoire
    import gc
    del sft_merged_model
    gc.collect()

    # ==========================
    # TRAITEMENT DU MODELE DPO
    # ==========================
    logger.info("--- Début du traitement DPO ---")
    # On recharge le modèle de base, car un merge est irréversible en mémoire sur l'objet Pytorch. 
    # Pour faire propre, recharger une nouvelle instance clean (évite les chevauchements)
    base_model = load_base_model()

    dpo_adapter_path = _load_dpo_lora_adapter()
    dpo_merged_model = merge_base_model_with_lora_adapter(base_model, dpo_adapter_path)
    
    save_models_for_deployment(dpo_merged_model, dpo_output_path)
    tokenizer.save_pretrained(dpo_output_path)
    
    logger.info("Push du modèle DPO mergé vers GCS via MLflow...")
    with mlflow.start_run(
        run_name="merge_dpo_model_for_deployment",
        tags={"stage": "deployment", "source": "dpo"},
    ):
        mlflow.log_artifacts(
            str(dpo_output_path),
            artifact_path="merged_model_dpo_for_deployment",
        )

    logger.info("=" * 60)
    logger.info("✅ Modèles SFT et DPO mergés et pushés sur GCS avec succès.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()