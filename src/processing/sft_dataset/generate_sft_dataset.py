"""
generate_sft_dataset.py (v2)
─────────────────────────────────────────────────────────────────────────────
Génère le dataset SFT brut (5000 paires Q&A) à partir des 4 sources nettoyées.

Pipeline v2 :
  1. Charger chaque source nettoyée
  2. Filtrer les questions cliniquement pertinentes (mots-clés + longueur)
  3. Sampler de manière équilibrée pour atteindre target_samples
  4. Calculer les token counts
  5. Sauvegarder sft_dataset.parquet (pas de split ici)

Le split train/val/test est réalisé APRÈS l'augmentation triage
(split_sft_dataset.py) pour que les exemples reformatés soient répartis
dans tous les splits.
"""

import yaml
import pandas as pd

from config.logger import logger
from config.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, SFT_DATASET_DIR
from src.processing.utils_cleaning import (
    add_token_counts,
    filter_clinical_questions,
)


def collect_clinical_samples(
    parquet_files: list[str],
    base_dir,
    target_samples: int,
    random_state: int,
    min_question_tokens: int = 15,
) -> pd.DataFrame:
    """
    Charge chaque source, applique le filtre clinique, puis sample de manière
    équilibrée pour atteindre exactement target_samples.

    L'algorithme de redistribution du surplus est identique au v1
    (collect_balanced_samples), mais appliqué APRÈS le filtre clinique :
    si une source a moins de lignes que sa part après filtrage,
    le surplus est redistribué aux sources suivantes.
    """
    n_sources = len(parquet_files)
    remaining = target_samples
    frames = []

    for i, pq_file in enumerate(parquet_files):
        path = base_dir / pq_file
        logger.info(f"Loading source: {path}")
        df = pd.read_parquet(path)
        n_raw = len(df)

        # ── Filtre clinique ──────────────────────────────────────────────
        df = filter_clinical_questions(df, min_question_tokens=min_question_tokens)
        n_clinical = len(df)
        logger.info(f"  {pq_file}: {n_raw} raw → {n_clinical} clinical")

        # ── Sampling équilibré avec redistribution ───────────────────────
        sources_left = n_sources - i
        share = remaining // sources_left

        if len(df) <= share:
            logger.warning(
                f"  {pq_file}: only {len(df)} clinical rows (share={share}). "
                f"Taking all — surplus redistributed."
            )
            sampled = df
        else:
            sampled = df.sample(n=share, random_state=random_state)

        frames.append(sampled)
        remaining -= len(sampled)
        logger.info(f"  Sampled {len(sampled)} from {pq_file} — {remaining} remaining")

    result = pd.concat(frames, ignore_index=True)

    if len(result) < target_samples:
        logger.warning(
            f"Only {len(result)} clinical samples available "
            f"(target: {target_samples}). Proceeding with {len(result)}."
        )
    else:
        logger.info(f"Target reached: {len(result)} samples")

    return result


def main():
    with (PROJECT_ROOT / "params.yaml").open() as f:
        params = yaml.safe_load(f)["sft"]

    target_samples: int = params["target_samples"]
    random_state: int = params["random_state"]
    parquet_files: list[str] = params["source_datasets"]
    min_question_tokens: int = params.get("min_question_tokens", 15)

    logger.info("=" * 60)
    logger.info("Starting SFT dataset generation (v2 — clinical filter)")
    logger.info(f"Target: {target_samples} clinical samples from {len(parquet_files)} datasets")
    logger.info("=" * 60)

    sft_dataset = collect_clinical_samples(
        parquet_files=parquet_files,
        base_dir=PROCESSED_DATA_DIR,
        target_samples=target_samples,
        random_state=random_state,
        min_question_tokens=min_question_tokens,
    )

    sft_dataset = add_token_counts(sft_dataset, columns=["question", "answer"])

    output_path = SFT_DATASET_DIR / "sft_dataset.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_dataset.to_parquet(output_path, index=False)

    logger.info("=" * 60)
    logger.info(f"SFT dataset saved: {len(sft_dataset)} rows → {output_path}")
    for name, count in sft_dataset["dataset_name"].value_counts().items():
        logger.info(f"  {name}: {count} ({count/len(sft_dataset)*100:.1f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()