"""
triage_augmentation.py
─────────────────────────────────────────────────────────────────────────────
Reformate TOUTES les réponses du dataset SFT au format triage structuré
via l'API Mistral (free tier — 1 Md tokens/mois).

Pipeline :
  1. Charge sft_dataset.parquet (5000 lignes, déjà filtrées cliniquement)
  2. Envoie chaque (question, answer) à Mistral pour reformatage triage
  3. Valide le format structuré de chaque réponse
  4. Sauvegarde sft_dataset_augmented.parquet (réponses reformatées)
     + sft_triage_failures.parquet (échecs, pour audit)

Usage :
    export MISTRAL_API_KEY="..."
    python -m src.processing.triage_augmentation

Coût estimé : ~2.5M tokens → 0.25% du free tier Mistral (0€)
Durée estimée : ~45-60 min (5000 appels × ~0.5s)
"""

import os
import re
import time
import yaml
import pandas as pd
from pathlib import Path
from functools import lru_cache

from config.logger import logger
from config.paths import PROJECT_ROOT, SFT_DATASET_DIR

# ── Chemins ──────────────────────────────────────────────────────────────────
SFT_DATASET_PATH = SFT_DATASET_DIR / "sft_dataset.parquet"
SFT_AUGMENTED_PATH = SFT_DATASET_DIR / "sft_dataset_augmented.parquet"
SFT_FAILURES_PATH = SFT_DATASET_DIR / "sft_triage_failures.parquet"


# ── Paramètres ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_triage_params() -> dict:
    params = _load_params()
    return params.get("triage_augmentation", {
        "max_retries": 2,
        "model": "mistral-small-latest",
        "batch_log_interval": 100,
    })


# ── Prompt de reformulation triage ────────────────────────────────────────────
REFORMULATION_SYSTEM_PROMPT = """\
Tu es un expert en médecine d'urgence chargé de reformater des réponses \
médicales au format standardisé du protocole de triage du CHSA.

À partir d'une question clinique et d'une réponse médicale brute, tu dois \
produire UNIQUEMENT une réponse structurée respectant EXACTEMENT ce format :

Bilan de triage :
- Symptômes relevés : [liste des symptômes principaux identifiés]
- Niveau d'urgence : [EXACTEMENT "maximale" ou "modérée" ou "différée"]
- Orientation recommandée : Service [nom du service médical]
- Hypothèses diagnostiques : [1 à 3 hypothèses séparées par " / "]
- Action immédiate : [action si urgence maximale, sinon "Consultation programmée"]

Règles strictes :
1. Le niveau d'urgence DOIT être EXACTEMENT l'un des trois termes : maximale, modérée, différée.
2. Le service doit être un service hospitalier réel (Urgences, Cardiologie, ORL, Neurologie, \
Pneumologie, Gastro-entérologie, Pédiatrie, Chirurgie orthopédique, Psychiatrie, Dermatologie, \
Ophtalmologie, Gynécologie, Rhumatologie, Médecine interne, etc.).
3. Ne produis AUCUN texte avant ou après le bilan.
4. Réponds TOUJOURS en français, même si la question est en anglais.
5. Base-toi sur la réponse médicale fournie pour déterminer le niveau d'urgence et les hypothèses."""


# ── Validation du format ──────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    "Bilan de triage",
    "Symptômes relevés",
    "Niveau d'urgence",
    "Orientation recommandée",
    "Hypothèses diagnostiques",
    "Action immédiate",
]

VALID_URGENCY_LEVELS = {"maximale", "modérée", "différée"}


def _validate_triage_response(response: str) -> bool:
    """Valide le format triage : champs présents + urgence correcte."""
    if not all(field in response for field in REQUIRED_FIELDS):
        return False
    match = re.search(r"Niveau d'urgence\s*:\s*(.+)", response)
    if not match:
        return False
    urgency = match.group(1).strip().lower().rstrip(".")
    return urgency in VALID_URGENCY_LEVELS


# ── Appel Mistral ─────────────────────────────────────────────────────────────
def _call_mistral(
    client,
    question: str,
    answer: str,
    model: str,
    max_retries: int,
) -> str | None:
    """
    Envoie une paire (question, answer) à Mistral pour reformulation triage.
    Retourne la réponse si valide, None si tous les essais échouent.
    """
    user_message = f"""Question clinique :
{question}

Réponse médicale brute :
{answer}

Reformate cette réponse au format triage standardisé du CHSA."""

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.complete(
                model=model,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            text = response.choices[0].message.content.strip()

            if _validate_triage_response(text):
                return text
            else:
                logger.debug(
                    f"Attempt {attempt+1}: invalid format for question "
                    f"'{question[:50]}...' — retrying"
                )

        except Exception as e:
            wait = 10 * (attempt + 1)
            logger.warning(f"API error (attempt {attempt+1}): {e} — waiting {wait}s")
            time.sleep(wait)

    return None


# ── Pipeline principal ────────────────────────────────────────────────────────
def main():
    from mistralai.client import Mistral

    params = _get_triage_params()
    model = params.get("model", "mistral-small-latest")
    max_retries = params.get("max_retries", 2)
    batch_log_interval = params.get("batch_log_interval", 100)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY non définie. "
            "Crée un compte sur console.mistral.ai (gratuit) et exporte ta clé : "
            "export MISTRAL_API_KEY='...'"
        )
    client = Mistral(api_key=api_key)

    # 1. Chargement
    if not SFT_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset SFT introuvable : {SFT_DATASET_PATH}\n"
            "Lance d'abord : dvc repro generate_sft"
        )
    sft_df = pd.read_parquet(SFT_DATASET_PATH)
    total = len(sft_df)

    logger.info("=" * 60)
    logger.info(f"Triage augmentation — {total} examples via Mistral ({model})")
    logger.info("=" * 60)

    # 2. Reformulation de TOUTES les lignes
    successes = []
    failures = []

    for i, (idx, row) in enumerate(sft_df.iterrows()):
        if i % batch_log_interval == 0:
            logger.info(
                f"Progress: {i}/{total} — "
                f"successes: {len(successes)} | failures: {len(failures)}"
            )

        triage_answer = _call_mistral(
            client=client,
            question=row["question"],
            answer=row["answer"],
            model=model,
            max_retries=max_retries,
        )

        if triage_answer is not None:
            new_row = row.to_dict()
            new_row["answer_original"] = new_row["answer"]  # garder l'original pour audit
            new_row["answer"] = triage_answer
            successes.append(new_row)
        else:
            failures.append(row.to_dict())

        # Pause légère entre les appels (rate limiting free tier)
        time.sleep(0.3)

    logger.info("=" * 60)
    logger.info(f"Augmentation complete — successes: {len(successes)} | failures: {len(failures)}")

    # 3. Sauvegarde
    SFT_AUGMENTED_PATH.parent.mkdir(parents=True, exist_ok=True)

    if successes:
        augmented_df = pd.DataFrame(successes)

        # Recalcul des token_count_answer (les bilans triage sont plus courts)
        from src.processing.utils_cleaning import add_token_counts
        augmented_df = add_token_counts(augmented_df, columns=["answer"])

        augmented_df.to_parquet(SFT_AUGMENTED_PATH, index=False)
        logger.info(f"Augmented dataset saved: {len(augmented_df)} rows → {SFT_AUGMENTED_PATH}")
    else:
        raise RuntimeError("Aucun exemple reformaté avec succès. Vérifie ta clé Mistral.")

    if failures:
        failures_df = pd.DataFrame(failures)
        failures_df.to_parquet(SFT_FAILURES_PATH, index=False)
        logger.info(f"Failures saved for audit: {len(failures_df)} rows → {SFT_FAILURES_PATH}")

    logger.info(
        f"Success rate: {len(successes)/total*100:.1f}% "
        f"({len(successes)}/{total})"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()