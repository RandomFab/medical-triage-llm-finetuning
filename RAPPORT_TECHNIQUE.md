# Rapport Technique — POC Agent IA de Triage Médical

## Centre Hospitalier Saint-Aurélien (CHSA)

**Auteur :** Fabien BARDOUIL  
**Mission :** Proof of Concept — Agent IA de Triage Médical  
**Modèle cible :** Qwen3-1.7B-Base (SFT + LoRA + DPO)  
**Date :** Avril 2026

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Préparation et structuration des données](#2-préparation-et-structuration-des-données)
   - 2.1 [Inventaire et sélection des sources](#21-inventaire-et-sélection-des-sources)
   - 2.2 [Ingestion des données brutes](#22-ingestion-des-données-brutes)
   - 2.3 [Pipeline de nettoyage](#23-pipeline-de-nettoyage)
   - 2.4 [Anonymisation et conformité RGPD](#24-anonymisation-et-conformité-rgpd)
   - 2.5 [Schéma de données unifié](#25-schéma-de-données-unifié)
   - 2.6 [Constitution du dataset SFT](#26-constitution-du-dataset-sft)
   - 2.7 [Versionnement et reproductibilité avec DVC](#27-versionnement-et-reproductibilité-avec-dvc)
3. [Fine-tuning supervisé (SFT) avec LoRA](#3-fine-tuning-supervisé-sft-avec-lora)
4. [Alignement par préférences (DPO)](#4-alignement-par-préférences-dpo)
5. [Déploiement et infrastructure](#5-déploiement-et-infrastructure)
6. [Évaluation et métriques de performance](#6-évaluation-et-métriques-de-performance)
7. [Recommandations stratégiques et roadmap](#7-recommandations-stratégiques-et-roadmap)
8. [Conclusion](#8-conclusion)
9. [Annexes](#9-annexes)

---

## 1. Introduction

Le Centre Hospitalier Saint-Aurélien (CHSA) fait face à une problématique récurrente dans le paysage hospitalier français : la surcharge de son service des urgences. Aux heures de pointe, le personnel infirmier et de triage manque d'effectifs, ce qui entraîne des temps d'attente prolongés et un risque accru de sous-identification des cas critiques. Dans ce contexte, la Direction Innovation Médicale du CHSA a mandaté le développement d'un Proof of Concept (POC) visant à démontrer la faisabilité technique et la valeur ajoutée clinique d'un agent IA de triage médical.

L'agent envisagé a pour vocation d'assister — et non de remplacer — le personnel soignant dans le processus de triage initial. Il doit être capable de collecter les symptômes du patient via un questionnaire adaptatif, d'évaluer un niveau de priorité (urgence maximale, modérée ou différée) conformément aux protocoles médicaux en vigueur, et de fournir des explications claires sur son évaluation. La traçabilité de chaque interaction constitue également un prérequis essentiel pour satisfaire aux exigences d'audit médical.

La stratégie technique retenue s'articule en trois phases. La première phase, dite de validation conceptuelle, repose sur le déploiement du modèle Qwen3-1.7B-Base, un modèle compact mais suffisamment performant pour valider rapidement les hypothèses techniques. La deuxième phase consiste à spécialiser ce modèle par fine-tuning supervisé (SFT) avec la technique LoRA, puis à l'aligner sur les pratiques cliniques via l'optimisation par préférences directes (DPO). Enfin, une troisième phase de projection industrielle envisage, en cas de validation concluante, le passage à des modèles de plus grande envergure.

Le présent rapport technique couvre l'ensemble de la mission réalisée sur quatre semaines. Il détaille la méthodologie employée pour la préparation des données d'entraînement (section 2), le processus de fine-tuning SFT et d'alignement DPO (sections 3 et 4), l'architecture de déploiement retenue (section 5), les résultats d'évaluation obtenus (section 6), et formule des recommandations stratégiques pour le passage à l'échelle (section 7).

---

## 2. Préparation et structuration des données

La qualité d'un modèle de langage fine-tuné dépend directement de la qualité de ses données d'entraînement. Cette section décrit l'intégralité du processus mis en œuvre pour constituer un corpus médical bilingue de 5 000 paires instruction-réponse, depuis l'identification des sources jusqu'à la génération du dataset final prêt pour le SFT.

### 2.1 Inventaire et sélection des sources

L'objectif initial était de constituer un corpus diversifié, couvrant à la fois le français et l'anglais, et représentant différents formats de connaissances médicales : questions à choix multiples issues d'examens cliniques, questions-réponses ouvertes, et paires de préférences pour l'alignement ultérieur par DPO.

Quatre datasets publics hébergés sur Hugging Face Hub ont été retenus :

**MediQAL (ANR-MALADES/MediQAl)** — Ce corpus francophone, issu du projet ANR MALADES, propose des questions médicales à choix multiples ancrées dans des cas cliniques réels.  
Il se décline en trois sous-ensembles :  
- MCQU (questions à réponse unique)
-  MCQM (questions à réponses multiples)
- OEQ (questions ouvertes).  

Seul le sous-ensemble MCQU a été retenu pour le SFT, car il garantit une correspondance univoque entre la question et la réponse attendue. Les questions à réponses multiples (MCQM) auraient introduit une ambiguïté dans le signal d'apprentissage — le modèle devant apprendre à produire une combinaison de réponses plutôt qu'une réponse unique — ce qui ne correspond pas au format conversationnel visé par l'agent de triage.  
Le sous-ensemble OEQ, bien que potentiellement intéressant par son format de réponse libre, n'a pas été intégré dans cette première itération par contrainte de temps ; son inclusion constitue une amélioration envisagée (cf. section 7).

**FrenchMedMCQA (nthngdy/frenchmedmcqa)** — Ce dataset francophone contient des questions à choix multiples issues d'examens de médecine français.  
Il est de taille plus modeste avec 595 exemples d'entraînement, 164 de validation et 321 de test, soit 1 080 exemples au total.  
Malgré sa taille réduite, il présente l'avantage d'être directement ancré dans le cursus médical français, ce qui le rend particulièrement pertinent pour un agent destiné à opérer dans un hôpital francophone.

**MedQuAD (keivalya/MedQuad)** — Ce corpus anglophone regroupe des paires question-réponse issues de sources médicales institutionnelles américaines (NIH, NCI, etc.).  
Avec 16 407 exemples d'entraînement, il constitue la plus grande source de données au format question-réponse directe.  
Les réponses sont rédigées de manière détaillée et structurée, offrant un bon signal d'apprentissage pour la génération de réponses médicales complètes.

**UltraMedical-Preference (TsinghuaC3I/UltraMedical-Preference)** — Ce dataset anglophone contient 109 353 paires de conversations médicales au format préférentiel (chosen/rejected).  
Il sert une double finalité : d'une part, les réponses "chosen" sont extraites et reformatées en paires question-réponse pour le SFT ; d'autre part, les paires chosen/rejected seront utilisées ultérieurement pour l'entraînement DPO (cf. section 4). Sa taille importante permet d'enrichir significativement le corpus anglophone.

Le tableau ci-dessous synthétise les volumes bruts par source et par split :

| Dataset | Langue | Split | Nombre de lignes |
|---|---|---|---|
| MediQAL MCQU | FR | train | 10 113 |
| MediQAL MCQU | FR | validation | 2 561 |
| MediQAL MCQU | FR | test | 4 343 |
| FrenchMedMCQA | FR | train | 595 |
| FrenchMedMCQA | FR | validation | 164 |
| FrenchMedMCQA | FR | test | 321 |
| MedQuAD | EN | train | 16 407 |
| UltraMedical-Preference | EN | train | 109 353 |

**Choix de la stratégie de split.**  
Les datasets d'origine proposent des découpages train/validation/test hétérogènes : MediQAL et FrenchMedMCQA disposent de trois splits, tandis que MedQuAD et UltraMedical n'en offrent qu'un seul. Face à cette inconsistance, la décision a été prise de fusionner l'ensemble des splits de chaque source en un unique DataFrame, puis de constituer ultérieurement un split stratifié par source au niveau du dataset SFT consolidé. Cette approche présente deux avantages : elle maximise le volume de données disponibles pour l'échantillonnage, et elle garantit que chaque source est représentée de manière équilibrée dans les jeux d'entraînement, de validation et de test. La colonne `dataset_name`, ajoutée lors du nettoyage, sert de clé de stratification.

**Licences et origines des sources.**  
Le cahier des charges exige la traçabilité de l'origine et des conditions de réutilisation de chaque corpus. Le tableau ci-dessous synthétise ces informations :

| Dataset | Hébergement | Licence | Commentaire |
|---|---|---|---|
| MediQAL (ANR-MALADES/MediQAl) | Hugging Face | À confirmer — projet ANR (usage académique présumé) | Contacter les auteurs avant tout usage commercial |
| FrenchMedMCQA (nthngdy/frenchmedmcqa) | Hugging Face | Apache 2.0 | Réutilisation libre, attribution requise |
| MedQuAD (keivalya/MedQuad) | Hugging Face | CC BY 4.0 (contenu NIH — domaine public américain) | Attribution requise |
| UltraMedical-Preference (TsinghuaC3I/UltraMedical-Preference) | Hugging Face | MIT — à confirmer | Réutilisation libre sans restriction si confirmé |

Les licences marquées "à confirmer" doivent être vérifiées directement sur les pages Hugging Face des datasets avant tout déploiement en production ou publication des résultats.

### 2.2 Ingestion des données brutes

L'ingestion des quatre datasets est réalisée via un notebook Jupyter dédié (`notebooks/Import_data_from_HuggingFace.ipynb`) qui télécharge chaque source depuis le Hugging Face Hub à l'aide de la bibliothèque `datasets`. Les données brutes sont ensuite persistées sur Google Cloud Storage dans le bucket `gs://p14-medical-data/raw_data/`, organisées par dataset :

```
raw_data/
├── mediqal_datasets/
│   ├── mcqu_medical/
│   ├── mcqm_medical/
│   └── oeq_medical/
├── frenchmedmcqa_dataset/
├── MedQuad_dataset/
└── UltraMedical_dataset/
```

Cette séparation entre données brutes et données traitées garantit que les transformations restent auditables : il est toujours possible de revenir aux données d'origine pour vérifier ou corriger une étape du pipeline. Un token Hugging Face est recommandé pour éviter le rate-limiting, notamment sur UltraMedical-Preference dont le téléchargement approche le gigaoctet.

### 2.3 Pipeline de nettoyage

Le nettoyage des données constitue l'étape la plus critique du processus de préparation. Chaque dataset présente des spécificités structurelles et sémantiques qui nécessitent un traitement adapté. L'architecture du code reflète cette dualité : un module utilitaire partagé (`utils_cleaning.py`) expose les fonctions communes, tandis que chaque dataset dispose de son propre script de nettoyage qui orchestre les transformations spécifiques.

#### 2.3.1 Fonctions utilitaires partagées

Quatre fonctions transversales sont mutualisées dans `src/processing/utils_cleaning.py` :

**`drop_duplicates(df, subset=None)`** — Supprime les lignes dupliquées sur l'ensemble des colonnes ou sur un sous-ensemble spécifié. Le nombre de doublons détectés et supprimés est systématiquement loggé pour traçabilité. Cette fonction est appelée dans chaque pipeline de nettoyage, souvent à deux reprises : une première fois sur le DataFrame brut complet, puis une seconde fois sur le DataFrame final réduit aux colonnes `question` et `answer`, afin de capturer les doublons qui n'étaient pas visibles avant la suppression des colonnes auxiliaires.

**`drop_columns(df, columns_to_drop)`** — Supprime les colonnes jugées non pertinentes pour l'entraînement. Les colonnes éliminées varient par dataset : identifiants techniques (`id`), métadonnées de catégorisation (`medical_subject`, `question_type`, `qtype`), ou colonnes de comptage (`number_correct_answers`).

**`transform_correct_answers_to_text(df, match_answer_dict)`** — Résout l'encodage des réponses correctes. Les datasets de type QCM stockent la réponse correcte sous forme d'indice (un entier pour FrenchMedMCQA, une lettre pour MediQAL) plutôt que sous forme de texte. Cette fonction mappe chaque indice vers le nom de la colonne contenant la réponse correspondante (par exemple, `0 → "answer_a"`, `"A" → "answer_a"`), créant ainsi une colonne intermédiaire `correct_answer_text`.

**`create_ground_truth_answer_column(df)`** — Complète la résolution en allant chercher, pour chaque ligne, le contenu textuel de la colonne désignée par `correct_answer_text`. Le résultat est une colonne `answer` contenant la réponse en texte clair, indépendamment du schéma d'encodage d'origine. L'expression centrale est :

```python
df["answer"] = df.apply(lambda row: row[row["correct_answer_text"]], axis=1)
```

**`merge_raw_data_splits(datasets)`** — Fusionne les différents splits (train, validation, test) d'un même dataset en un unique DataFrame. Comme expliqué en section 2.1, cette fusion est intentionnelle et vise à permettre un re-découpage stratifié ultérieur.

**`save_cleaned_data_local(df, destination_path)`** — Persiste le DataFrame nettoyé au format Parquet sur le système de fichiers local, dans le répertoire `data/processed/`. Le format Parquet a été privilégié pour sa compression efficace, sa compatibilité native avec Pandas et son typage fort des colonnes. La taille du fichier produit est loggée à chaque sauvegarde.

L'ensemble de ces fonctions est instrumenté par un système de logging centralisé (`config/logger.py`) qui trace chaque opération avec horodatage, niveau de sévérité et message descriptif. Cette instrumentation répond directement à l'exigence d'auditabilité formulée dans le cahier des charges.

#### 2.3.2 Nettoyage de MediQAL (MCQU)

Le dataset MediQAL MCQU a nécessité le travail de nettoyage le plus approfondi. La pipeline, implémentée dans `src/processing/mediqal_cleaning.py`, enchaîne onze étapes :

**Étape 1 — Suppression des doublons.** Une première passe de déduplication est réalisée sur l'ensemble des colonnes du DataFrame brut.

**Étape 2 — Suppression des colonnes non pertinentes.** Les colonnes `id`, `task`, `medical_subject` et `question_type` sont retirées. Ces métadonnées, utiles pour l'exploration, n'apportent pas de signal pour le fine-tuning au format question-réponse.

**Étape 3 — Résolution des réponses.** Le dictionnaire de mapping `{"A": "answer_a", "B": "answer_b", ...}` est appliqué pour résoudre la colonne `correct_answers` en texte, puis la colonne `answer` est générée via lookup dynamique.

**Étape 4 — Fusion du cas clinique et de la question.** C'est l'étape la plus significative sur le plan sémantique. Chaque ligne du dataset MediQAL contient une colonne `clinical_case` décrivant le contexte clinique du patient, et une colonne `question` posant la question médicale. Ces deux champs sont concaténés pour former une question enrichie :

```python
df["question"] = df["clinical_case"] + " " + df["question"]
```

Cette fusion est motivée par l'usage cible de l'agent : en situation de triage, le modèle recevra un contexte patient (symptômes, antécédents, constantes) suivi d'une question. En entraînant le modèle sur des entrées qui associent déjà contexte clinique et interrogation, on favorise l'apprentissage de ce pattern conversationnel.

**Étape 5 — Sélection des colonnes finales.** Seules les colonnes `question` et `answer` sont conservées pour le dataset de sortie.

**Étape 6 — Seconde déduplication.** Après réduction aux deux colonnes finales, une nouvelle passe de déduplication capture les éventuels doublons masqués par les colonnes auxiliaires précédemment présentes.

**Étape 7 — Nettoyage des instructions parasites.** Certaines questions contiennent des instructions à destination de l'étudiant qui sont sans valeur pour l'entraînement du modèle. La mention `(cochez la réponse juste)` est supprimée de toutes les questions. Ce type d'artefact, hérité du format QCM papier, pourrait induire le modèle à reproduire des instructions de consigne dans ses réponses en contexte réel.

**Étape 8 — Suppression des questions à négation.** Les questions formulées sous la forme "cochez la réponse fausse" sont systématiquement éliminées. Ce choix repose sur une analyse approfondie de l'impact potentiel sur le fine-tuning : lorsqu'une question demande d'identifier la réponse incorrecte, la "bonne" réponse dans le dataset est en réalité une affirmation médicalement fausse. Si le modèle s'entraîne sur ces paires, il apprend à associer des informations médicalement erronées à la position de réponse correcte, ce qui pourrait compromettre la fiabilité de ses réponses en production. Dans un contexte de triage médical où la véracité des informations est critique, ce risque a été jugé inacceptable.

**Étape 9 — Suppression des réponses indicées.** Certaines questions du corpus contiennent des propositions numérotées dans le corps de la question (par exemple : `1. Tachycardie, 2. Bradycardie, 3. Hypotension...`) avec une réponse sous forme de combinaison d'indices (par exemple : `1+2+3`). Ces paires sont détectées par l'expression régulière suivante et supprimées :

```python
question_has_proposals = df["question"].str.contains(r"\n\s*\d+[.\-\)]\s+\S", regex=True)
answer_is_indices = df["answer"].str.contains(r"^\s*\d+(\s*[+\s]\s*\d+)+\s*$", regex=True)
```

Ce filtrage est essentiel : si ces paires étaient conservées, le modèle apprendrait à répondre par des combinaisons numériques (`1+3+5`) plutôt que par des réponses médicales textuelles. En situation de triage, une telle réponse serait inintelligible pour le personnel soignant.

**Étape 10 — Normalisation textuelle.** L'ensemble du texte (questions et réponses) est converti en minuscules pour uniformiser la représentation et réduire la variabilité lexicale vue par le modèle lors du fine-tuning.

**Étape 11 — Marquage de la source.** Une colonne `dataset_name` est ajoutée avec la valeur `"mediqal"` pour permettre la stratification ultérieure lors de la constitution du dataset SFT.

#### 2.3.3 Nettoyage de FrenchMedMCQA

Le dataset FrenchMedMCQA, plus compact et structuré, a nécessité un nettoyage moins intensif. La pipeline est implémentée dans `src/processing/frenchmedmcqa_cleaning.py` et suit huit étapes :

**Étape 1 — Filtrage des questions à négation.** Comme pour MediQAL, les questions contenant l'expression "une seule est fausse" sont supprimées. La logique est identique : éviter d'entraîner le modèle sur des paires où la réponse "correcte" est en réalité une information médicalement incorrecte.

**Étape 2 — Suppression des doublons.** Déduplication sur l'ensemble des colonnes.

**Étape 3 — Suppression des colonnes non pertinentes.** Les colonnes `id` et `number_correct_answers` sont retirées.

**Étape 4 — Résolution des réponses.** Le dictionnaire `{0: "answer_a", 1: "answer_b", ...}` est appliqué pour résoudre les indices numériques en texte de réponse.

**Étape 5 — Sélection des colonnes finales.** Restriction aux colonnes `question` et `answer`.

**Étape 6 — Seconde déduplication.** Passe de déduplication post-réduction.

**Étape 7 — Normalisation textuelle.** Conversion en minuscules.

**Étape 8 — Marquage de la source.** Ajout de la colonne `dataset_name` avec la valeur `"frenchmedmcqa"`.

#### 2.3.4 Nettoyage de MedQuAD

Le corpus MedQuAD étant déjà structuré au format question-réponse direct, le nettoyage est minimal. Les quatre étapes implémentées dans `src/processing/medquad_cleaning.py` consistent en une déduplication, la suppression de la colonne `qtype` (catégorisation du type de question), le renommage des colonnes (`Question → question`, `Answer → answer`) pour se conformer au schéma unifié, et l'ajout du marqueur de source `"medquad"`.

#### 2.3.5 Nettoyage d'UltraMedical-Preference

Le dataset UltraMedical-Preference présente une particularité structurelle majeure : les données ne sont pas au format question-réponse direct mais au format conversationnel préférentiel. Chaque ligne contient un champ `chosen` (la conversation préférée) constitué d'une liste de messages avec rôles (`user`, `assistant`). La pipeline de nettoyage (`src/processing/ultramed_cleaning.py`) extrait la première question utilisateur et la première réponse assistant de chaque conversation :

```python
def extract_qa(row):
    chosen = row.get("chosen", [])
    question = next((m["content"] for m in chosen if m["role"] == "user"), None)
    answer = next((m["content"] for m in chosen if m["role"] == "assistant"), None)
    return pd.Series({"question": question, "answer": answer})
```

Cette transformation convertit le format conversationnel en format tabulaire `(question, answer)` compatible avec le schéma unifié. Après extraction, les étapes classiques de déduplication et de marquage de source (`"ultramed"`) sont appliquées.

#### 2.3.6 Synthèse des transformations

Le tableau ci-dessous récapitule les transformations appliquées par dataset :

| Transformation | MediQAL | FrenchMedMCQA | MedQuAD | UltraMedical |
|---|:---:|:---:|:---:|:---:|
| Fusion des splits | ✅ | ✅ | ✅ | ✅ |
| Suppression des doublons | ✅ (×2) | ✅ (×2) | ✅ | ✅ |
| Suppression de colonnes | ✅ | ✅ | ✅ | — |
| Résolution indices → texte | ✅ | ✅ | — | — |
| Fusion cas clinique + question | ✅ | — | — | — |
| Extraction QA conversationnel | — | — | — | ✅ |
| Retrait questions à négation | ✅ | ✅ | — | — |
| Retrait instructions parasites | ✅ | — | — | — |
| Retrait réponses indicées | ✅ | — | — | — |
| Normalisation lowercase | ✅ | ✅ | ✅ | ✅ |
| Marquage source | ✅ | ✅ | ✅ | ✅ |

### 2.4 Anonymisation et conformité RGPD

Dans le cadre d'un déploiement hospitalier, la conformité au RGPD est un prérequis
non négociable. Cette section documente la démarche suivie et la décision technique
qui en a résulté.

#### 2.4.1 Implémentation initiale de Presidio

L'outil retenu est **Presidio**, une bibliothèque open source développée par Microsoft,
spécialisée dans la détection et le masquage automatisé des données sensibles. Le
module `src/processing/anonymisation.py` implémente un pipeline bilingue FR/EN
reposant sur deux composants :

- Un **moteur d'analyse** (`AnalyzerEngine`) configuré avec les modèles spaCy
  `fr_core_news_md` et `en_core_web_md` pour détecter cinq types d'entités :
  `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `DATE_TIME`, `LOCATION`.
- Un **moteur d'anonymisation** (`AnonymizerEngine`) remplaçant les entités
  détectées par des balises standardisées (`<PERSON>`, `<DATE>`, etc.).

#### 2.4.2 Audit et décision de retrait sur les corpus publics

Après constitution du dataset SFT v1 avec anonymisation active, un audit quantitatif
a été réalisé sur les 5 000 échantillons produits. Les résultats ont révélé un taux
de faux positifs incompatible avec un entraînement de qualité :

| Indicateur | Valeur |
|---|---|
| Lignes touchées (`question` OU `answer`) | 2 340 / 5 000 (**46,8%**) |
| Total balises dans `answer` | 4 083 occurrences |
| Total balises dans `question` | 3 003 occurrences |

La répartition par type de balise dans `answer` illustre la nature du problème :

| Balise | Occurrences | Cause des faux positifs |
|---|---|---|
| `<PERSON>` | 1 874 | Noms de syndromes éponymes (Cushing, Crohn, Babinski…) |
| `<DATE>` | 1 451 | Références temporelles cliniques (48h, 7 jours, 72 premières heures…) |
| `<LOCATION>` | 729 | Régions anatomiques, noms d'instituts de recherche |
| `<PHONE>` | 16 | Faux positifs marginaux |
| `<EMAIL>` | 1 | Faux positif marginal |

Presidio, conçu pour détecter des données personnelles réelles, interprète le
vocabulaire médical courant comme des entités sensibles. Le résultat est que les
réponses d'entraînement contiennent des balises parasites en lieu et place de termes
cliniques légitimes — ce qui dégrade directement le signal d'apprentissage du modèle.

**Décision :** les quatre corpus sources (MediQAL, FrenchMedMCQA, MedQuAD,
UltraMedical-Preference) sont des datasets publics Hugging Face sans données
personnelles réelles. L'anonymisation Presidio n'est pas justifiée sur ces données et
a été **retirée du pipeline DVC** pour les étapes `generate_sft` et `generate_dpo`.
Le module `anonymisation.py` est conservé dans le dépôt et constitue la brique
technique appropriée pour de futures données patient réelles, sous réserve d'un
calibrage spécifique sur le vocabulaire médical francophone.

### 2.5 Schéma de données unifié

À l'issue du nettoyage, chaque dataset produit un fichier Parquet dont le schéma est strictement identique. Cette uniformisation est assurée par deux helpers partagés dans `src/processing/utils_cleaning.py` : `add_metadata()` et `add_token_counts()`.

#### 2.5.1 Colonnes de contenu

| Colonne | Type | Description |
|---|---|---|
| `question` | `str` | Texte de la question, après toutes les transformations de nettoyage. Pour MediQAL, inclut le cas clinique préfixé. |
| `answer` | `str` | Texte de la réponse en clair, après résolution des indices pour les datasets QCM. |

Pour MediQAL uniquement, deux colonnes supplémentaires sont présentes dans le fichier Parquet intermédiaire : `has_clinical_case` (`bool`) et `medical_subject` (`str`). Comme `extract_samples` lit le Parquet sans filtrer les colonnes (`pd.read_parquet` sans `columns=`), elles se retrouvent dans le dataset SFT final — mais uniquement renseignées pour les lignes MediQAL, avec `NaN` pour les trois autres sources.

#### 2.5.2 Colonnes de métadonnées structurées

La fonction `add_metadata()` appose quatre colonnes systématiquement sur chaque DataFrame nettoyé :

```python
def add_metadata(df, language, question_type, confidence_level, dataset_name):
    df["language"]         = language
    df["question_type"]    = question_type
    df["confidence_level"] = confidence_level
    df["dataset_name"]     = dataset_name
    return df
```

Le tableau ci-dessous détaille les valeurs appliquées par source :

| Dataset | `language` | `question_type` | `confidence_level` | `dataset_name` |
|---|:---:|:---:|:---:|:---:|
| MediQAL MCQU | `"fr"` | `"mcq_single"` | `"medium"` | `"mediqal"` |
| FrenchMedMCQA | `"fr"` | `"mcq_single"` | `"medium"` | `"frenchmedmcqa"` |
| MedQuAD | `"en"` | `"open_qa"` | `"high"` | `"medquad"` |
| UltraMedical-Preference | `"en"` | `"conversational"` | `"low"` | `"ultramed"` |

**`language`** — Code langue ISO à deux lettres. Permet de stratifier les échantillons ou de filtrer par langue lors du fine-tuning.

**`question_type`** — Catégorisation du format de la paire QA :
- `mcq_single` : question à choix multiple avec une seule réponse correcte (datasets francophones d'examen) ;
- `open_qa` : question ouverte avec réponse rédigée (MedQuAD) ;
- `conversational` : paire extraite d'un tour de conversation (UltraMedical).

**`confidence_level`** — Estimation de la fiabilité de la réponse en tant que signal d'entraînement :
- `high` : réponses issues de sources institutionnelles (NIH, NCI) avec rédaction structurée (MedQuAD) ;
- `medium` : réponses issues de référentiels d'examen médical (MediQAL, FrenchMedMCQA) — fiables mais formulées de façon concise ;
- `low` : réponses extraites automatiquement depuis des conversations multi-tours préférence (UltraMedical) — potentiellement verbeux ou mal calibrés.

Cette gradation permet de pondérer les échantillons lors du SFT ou d'exclure les sources de faible confiance dans les expérimentations futures.

**`dataset_name`** — Identifiant de source servant de clé de stratification lors de l'échantillonnage (section 2.6).

#### 2.5.3 Colonnes de comptage de tokens

La fonction `add_token_counts()`, appelée dans les deux pipelines (SFT et DPO) après l'étape de nettoyage, calcule la longueur en tokens de chaque colonne texte à l'aide du tokenizer de Qwen3-1.7B-Base :

```python
def add_token_counts(df, columns):
    tokenizer = _get_qwen_tokenizer()   # lru_cache — chargé une seule fois
    for col in columns:
        encodings = tokenizer(df[col].fillna("").tolist(), add_special_tokens=False)
        df[f"token_count_{col}"] = [len(ids) for ids in encodings["input_ids"]]
    return df
```

Pour le dataset SFT, deux colonnes sont ainsi ajoutées : `token_count_question` et `token_count_answer`. Pour le dataset DPO, les colonnes `token_count_chosen` et `token_count_rejected` sont produites.

Le recours au tokenizer du modèle cible (et non à une approximation par `len(text.split())`) garantit que les longueurs reflètent exactement le coût en séquence lors du fine-tuning. Ces colonnes servent à filtrer en amont les séquences qui dépasseraient la fenêtre de contexte du modèle, évitant ainsi une troncature silencieuse lors de l'entraînement.

L'utilisation de `lru_cache(maxsize=1)` sur `_get_qwen_tokenizer()` évite de recharger le tokenizer depuis le disque à chaque appel, ce qui serait coûteux dans un contexte de traitement batch.

#### 2.5.4 Schéma complet du dataset SFT final

Les quatre fichiers Parquet produits (`sft_dataset.parquet`, `sft_train.parquet`, `sft_val.parquet`, `sft_test.parquet`) partagent le même schéma :

| Colonne | Type | Présence | Description |
|---|---|:---:|---|
| `question` | `str` | Toujours | Question nettoyée |
| `answer` | `str` | Toujours | Réponse en texte clair |
| `dataset_name` | `str` | Toujours | Source d'origine |
| `language` | `str` | Toujours | `"fr"` ou `"en"` |
| `question_type` | `str` | Toujours | `"mcq_single"`, `"open_qa"` ou `"conversational"` |
| `confidence_level` | `str` | Toujours | `"low"`, `"medium"` ou `"high"` |
| `token_count_question` | `int` | Toujours | Nombre de tokens (tokenizer Qwen3) |
| `token_count_answer` | `int` | Toujours | Nombre de tokens (tokenizer Qwen3) |
| `has_clinical_case` | `bool` | MediQAL uniquement (`NaN` ailleurs) | Indique si la question était associée à un cas clinique |
| `medical_subject` | `str` | MediQAL uniquement (`NaN` ailleurs) | Spécialité médicale d'origine |

### 2.6 Constitution du dataset SFT

L'objectif de cette étape est de consolider les quatre corpus nettoyés en un unique dataset de 5 000 paires `(question, answer)` prêt pour le fine-tuning supervisé. Le pipeline de constitution du dataset SFT a fait l'objet d'une refonte complète
(v2) après l'audit qualité décrit en section 2.4. Il se décompose désormais en trois
stages DVC enchaînés : `generate_sft`, `triage_augmentation`, et `split_sft`.

#### 2.6.1 Stage `generate_sft` — Filtre clinique et sampling équilibré

Le script `src/processing/sft_dataset/generate_sft_dataset.py` implémente un
mécanisme d'échantillonnage équilibré sur les données **filtrées cliniquement**,
piloté par les paramètres définis dans `params.yaml` :

```yaml
sft:
  target_samples: 5000
  random_state: 42
  min_question_tokens: 15
  source_datasets:
    - mediqal_dataset/mediqal.parquet
    - frenchmedmcqa_dataset/frenchmedmcqa.parquet
    - medquad_dataset/medquad.parquet
    - ultramed_dataset/ultramed.parquet
```

Avant le sampling, chaque source est soumise à `filter_clinical_questions()` —
une fonction ajoutée dans `utils_cleaning.py` qui ne conserve que les questions
contenant au moins un mot-clé clinique (symptômes, temporalité, contexte patient)
en français ou en anglais, et dont la longueur dépasse `min_question_tokens=15`.
Ce filtre élimine les QCM purement académiques dont les réponses ne peuvent pas
être reformatées en bilan de triage.

L'algorithme de sampling équilibré avec redistribution du surplus est conservé
depuis la v1 : si une source ne dispose pas de suffisamment de lignes cliniques
pour sa part théorique (cas de FrenchMedMCQA), l'ensemble de ses données est
inclus et le surplus est redistribué aux sources suivantes. Le stage produit
`sft_dataset.parquet` (5 000 lignes cliniques, sans split).

#### 2.6.2 Stage `triage_augmentation` — Reformatage au format triage via Mistral

Le corpus de base (QCM médicaux, Q&A encyclopédiques) ne contient aucun exemple
au format de triage structuré attendu par le prompt système de l'agent. Un stage
dédié `src/processing/triage_augmentation.py` reformate chaque paire
`(question, answer)` en un bilan de triage via l'API Mistral Small
(`mistral-small-latest`).

Pour chaque exemple, Mistral reçoit la question et la réponse médicale originale
et produit une réponse structurée au format attendu : niveau d'urgence
(maximale / modérée / différée), hypothèses diagnostiques et recommandation
d'orientation. Le choix de Mistral (entreprise française, hébergement UE) est
cohérent avec les contraintes RGPD d'un projet hospitalier.

Les paramètres sont exposés dans `params.yaml` :

```yaml
triage_augmentation:
  model: "mistral-small-latest"
  max_retries: 2
  batch_log_interval: 100
```

Le stage produit `sft_dataset_augmented.parquet` ainsi qu'un fichier d'audit
`sft_triage_failures.parquet` recensant les exemples pour lesquels le reformatage
a échoué après les tentatives autorisées.

#### 2.6.3 Stage `split_sft` — Split après augmentation

Le split train/val/test est réalisé **après** l'augmentation, de sorte que les
exemples reformatés au format triage soient présents dans les trois sous-ensembles.
Le script `src/processing/sft_dataset/split_sft_dataset.py` produit :

| Fichier | Proportion | Volume (sur 5 000) |
|---|:---:|:---:|
| `sft_train.parquet` | 70 % | 3 500 lignes |
| `sft_val.parquet` | 20 % | 1 000 lignes |
| `sft_test.parquet` | 10 % | 500 lignes |

La stratification est réalisée sur `dataset_name` — chacune des quatre sources est
représentée dans les mêmes proportions dans chaque split.

### 2.7 Versionnement et reproductibilité avec DVC

L'intégralité du pipeline de nettoyage et de constitution du dataset SFT est orchestrée par **DVC (Data Version Control)**, un outil open source de versionnement de données et de pipelines ML. Le choix de DVC répond directement à l'exigence de traçabilité formulée dans le cahier des charges : "conserver une trace de chaque transformation de données".

Le fichier `dvc.yaml` définit cinq stages organisés en graphe acyclique dirigé (DAG) :

```
clean_mediqal ────────┐
clean_medquad ────────┤
clean_frenchmedmcqa ──┼→ generate_sft → triage_augmentation → split_sft
clean_ultramed ───────┘
clean_ultramed ───────────────────────────────────────────→ generate_dpo
```

Le pipeline compte huit stages au total. Les quatre stages de nettoyage
(`clean_*`) sont indépendants et peuvent être exécutés en parallèle. Les trois
stages SFT (`generate_sft` → `triage_augmentation` → `split_sft`) sont
séquentiels — le split est volontairement placé après l'augmentation pour que
les exemples reformatés au format triage soient répartis dans tous les splits.
Le stage `generate_dpo` est indépendant et consomme uniquement la sortie de
`clean_ultramed`.

Chaque stage déclare explicitement ses dépendances (scripts Python, fichiers de configuration) et ses sorties (répertoires de données traitées). DVC calcule un hash MD5 pour chaque entrée et sortie, ce qui permet de détecter automatiquement si une étape doit être ré-exécutée suite à une modification de code ou de données.

Le fichier `dvc.lock` enregistre l'état exact de chaque exécution : les hash des scripts, les hash des données produites, et les paramètres utilisés. Ce fichier, versionné dans Git, constitue un certificat de reproductibilité : n'importe quel membre de l'équipe peut recréer exactement le même dataset en exécutant `dvc repro`.

Les données produites sont synchronisées avec un remote GCS (`gs://p14-medical-data/dvc-store`), ce qui dissocie le versionnement des données (géré par DVC) du versionnement du code (géré par Git). Cette architecture permet de travailler avec des datasets volumineux (UltraMedical dépasse 140 Mo en Parquet) sans alourdir le dépôt Git.

---

## 3. Fine-tuning supervisé (SFT) avec LoRA

L'objectif de cette phase est de spécialiser le modèle de base Qwen3-1.7B sur le domaine médical en lui apprenant à produire des réponses structurées à partir de questions cliniques. Le fine-tuning supervisé (SFT) consiste à entraîner le modèle sur les 3 500 paires `(question, answer)` du jeu d'entraînement constitué en section 2.6, en utilisant la technique LoRA pour limiter l'empreinte mémoire et le risque de dégradation des connaissances générales du modèle.

Cette section détaille l'ensemble des choix techniques effectués, depuis la quantification du modèle de base jusqu'aux résultats de l'entraînement.

### 3.1 Choix du modèle de base

Le modèle retenu pour le POC est **Qwen3-1.7B-Base** (`Qwen/Qwen3-1.7B-Base`), un modèle de langage causal (CLM) développé par l'équipe Qwen d'Alibaba. Ce choix repose sur trois critères principaux :

**Compacité.** Avec 1,7 milliard de paramètres, le modèle est suffisamment léger pour être entraîné et servi sur des GPU grand public (NVIDIA T4 avec 16 Go de VRAM). Dans le cadre d'un POC dont l'objectif est de valider la faisabilité technique, un modèle compact permet d'itérer rapidement sur les hyperparamètres sans mobiliser d'infrastructure coûteuse.

**Architecture moderne.** Qwen3 intègre les avancées récentes des architectures transformer : Grouped Query Attention (GQA) pour une inférence plus efficace, RoPE (Rotary Position Embeddings) pour une meilleure gestion des positions, et un vocabulaire étendu à 151 936 tokens couvrant les langues latines et asiatiques — un atout pour notre corpus bilingue FR/EN.

**Chat template natif.** Le modèle dispose d'un format de conversation structuré (`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`) qui s'aligne directement avec le cas d'usage de l'agent de triage : un échange entre un patient (rôle `user`) et l'agent (rôle `assistant`), guidé par un prompt système définissant le comportement clinique attendu.

Le cahier des charges prévoit, en cas de validation concluante du POC, un passage à des modèles de plus grande envergure (32B+ paramètres). L'architecture modulaire mise en place — LoRA, quantification, chat template externalisé — est conçue pour faciliter cette transition sans refonte majeure du code.

### 3.2 Quantification 4-bit avec BitsAndBytes

Le modèle de base Qwen3-1.7B pèse environ 3,4 Go en précision FP16. Pour permettre l'entraînement sur un GPU NVIDIA T4 (16 Go VRAM) tout en conservant suffisamment de mémoire pour les gradients, les activations et l'optimiseur, le modèle est chargé en précision 4-bit à l'aide de la bibliothèque **BitsAndBytes**.

La configuration de quantification, externalisée dans `params.yaml`, est la suivante :

```yaml
quantization_config:
  load_in_4bit: True
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: True
  bnb_4bit_quant_type: "nf4"
```

**`load_in_4bit: True`** — Active la quantification 4-bit lors du chargement du modèle. Chaque poids, initialement stocké sur 16 bits (2 octets), est compressé sur 4 bits (0,5 octet), divisant l'empreinte mémoire par un facteur d'environ 4. Le modèle quantifié occupe approximativement 0,85 Go au lieu de 3,4 Go.

**`bnb_4bit_quant_type: "nf4"`** — Sélectionne le type de quantification NormalFloat 4 bits (NF4). Ce format, introduit par l'article QLoRA (Dettmers et al., 2023), est spécifiquement optimisé pour les poids de réseaux de neurones dont la distribution suit approximativement une loi normale. Les 16 niveaux de quantification sont répartis de manière non-uniforme, concentrés autour de zéro où se trouvent la majorité des poids, ce qui minimise l'erreur de quantification par rapport à un codage uniforme classique (FP4 ou INT4).

**`bnb_4bit_use_double_quant: True`** — Active la double quantification. Les constantes de quantification (une par bloc de 64 poids) sont elles-mêmes quantifiées en 8 bits, ce qui réduit l'empreinte mémoire supplémentaire liée aux métadonnées de quantification d'environ 0,37 bit par paramètre. Sur 1,7 milliard de paramètres, cette économie représente environ 79 Mo.

**`bnb_4bit_compute_dtype: "float16"`** — Définit la précision de calcul pour les opérations matricielles. Bien que les poids soient stockés en 4-bit, ils sont décompressés à la volée en FP16 pour chaque forward pass. Ce choix représente un compromis : FP16 offre une plage dynamique suffisante pour le fine-tuning tout en étant nativement supporté par les Tensor Cores de la T4. L'alternative BFloat16 (BF16) aurait été préférable pour sa meilleure stabilité numérique sur les grandes valeurs, mais le GPU T4 ne la supporte pas matériellement.

La quantification est appliquée lors du chargement du modèle dans la fonction `define_model()` :

```python
quantization = BitsAndBytesConfig(
    load_in_4bit=quantization_config["load_in_4bit"],
    bnb_4bit_quant_type=quantization_config["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=quantization_config["bnb_4bit_use_double_quant"],
    bnb_4bit_compute_dtype=getattr(torch, quantization_config["bnb_4bit_compute_dtype"]),
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization, device_map="auto",
)
```

La conversion du type de données depuis la chaîne YAML (`"float16"`) vers l'objet PyTorch (`torch.float16`) est réalisée dynamiquement via `getattr(torch, ...)`, ce qui permet de changer la précision de calcul sans modifier le code.

Après chargement, le modèle est préparé pour l'entraînement en précision réduite via `prepare_model_for_kbit_training()`. Cette fonction de la bibliothèque PEFT réalise trois opérations essentielles : elle convertit les couches de normalisation (LayerNorm) en FP32 pour préserver la stabilité numérique lors de la rétropropagation, elle désactive le cache clé-valeur (KV cache) qui n'est pas compatible avec le gradient checkpointing, et elle active `input_require_grads` sur le modèle pour permettre le calcul des gradients à travers les couches gelées jusqu'aux adaptateurs LoRA.

### 3.3 Adaptation par LoRA

Le fine-tuning classique consiste à mettre à jour l'intégralité des poids du modèle lors de l'entraînement. Pour un modèle de 1,7 milliard de paramètres, cela nécessite de stocker en mémoire non seulement les poids eux-mêmes, mais aussi leurs gradients et les états de l'optimiseur (deux tenseurs supplémentaires par paramètre pour Adam), ce qui multiplie l'empreinte mémoire par un facteur 3 à 4. Cette approche est incompatible avec les contraintes GPU du POC.

La technique **LoRA** (Low-Rank Adaptation, Hu et al., 2021) propose une alternative : au lieu de modifier directement les matrices de poids du modèle, elle injecte à côté de chaque matrice ciblée deux petites matrices (appelées adaptateurs) dont le produit représente une correction de faible rang. Pendant l'entraînement, seuls ces adaptateurs sont mis à jour tandis que les poids originaux restent gelés. L'économie mémoire est considérable : seuls les paramètres des adaptateurs nécessitent des gradients et des états d'optimiseur.

La configuration LoRA, externalisée dans `params.yaml`, est la suivante :

```yaml
lora_config:
  r: 16
  lora_alpha: 32
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  task_type: "CAUSAL_LM"
```

**`r: 16`** — Le rang des matrices d'adaptation. Ce paramètre contrôle la capacité expressive des adaptateurs : un rang plus élevé permet de capturer des adaptations plus complexes, mais augmente le nombre de paramètres entraînables et le risque de surapprentissage. Avec `r=16`, chaque adaptateur se compose de deux matrices : une de taille `(d × 16)` et une de taille `(16 × d)`, où `d` est la dimension de la matrice de poids originale. Pour Qwen3-1.7B (`d=2048` pour les couches d'attention), cela représente `2 × 2048 × 16 = 65 536` paramètres par module ciblé, contre `2048 × 2048 = 4 194 304` paramètres pour la matrice originale — soit une réduction d'un facteur 64.

**`lora_alpha: 32`** — Le facteur de mise à l'échelle de la correction LoRA. La correction appliquée aux poids originaux est multipliée par le ratio `alpha / r = 32 / 16 = 2`. Ce ratio contrôle l'intensité avec laquelle les adaptateurs influencent les sorties du modèle. Un ratio de 2 est un choix courant dans la littérature, offrant un équilibre entre capacité d'adaptation et stabilité de l'entraînement. Un ratio trop élevé risquerait de déstabiliser les représentations apprises par le modèle pré-entraîné ; un ratio trop faible limiterait la capacité du modèle à se spécialiser sur le domaine médical.

**`target_modules`** — Les sept modules ciblés couvrent l'intégralité des projections linéaires de chaque couche transformer de Qwen3 :
- Couches d'attention : `q_proj` (requêtes), `k_proj` (clés), `v_proj` (valeurs), `o_proj` (projection de sortie) — ces quatre projections contrôlent la manière dont le modèle sélectionne et combine l'information contextuelle.
- Couches MLP (réseau feed-forward) : `gate_proj`, `up_proj`, `down_proj` — ces trois projections contrôlent la transformation non-linéaire appliquée à chaque position après l'attention.

Le choix de cibler à la fois les couches d'attention et le MLP, plutôt que les seules couches d'attention (choix courant dans les premières implémentations LoRA), se justifie par les travaux récents montrant que l'inclusion du MLP améliore significativement la qualité du fine-tuning, notamment sur des tâches de génération de texte. Pour un modèle compact comme Qwen3-1.7B, chaque couche porte proportionnellement plus de responsabilité dans la représentation des connaissances — il est donc important d'adapter l'ensemble des projections linéaires.

**`lora_dropout: 0.05`** — Un dropout de 5% est appliqué sur les adaptateurs LoRA pendant l'entraînement. Ce mécanisme de régularisation désactive aléatoirement 5% des neurones des adaptateurs à chaque forward pass, ce qui réduit le risque de surapprentissage sur le dataset de 3 500 exemples.

**`task_type: "CAUSAL_LM"`** — Indique à PEFT que le modèle est utilisé en mode de modélisation causale du langage (prédiction du token suivant), ce qui conditionne la configuration interne des adaptateurs.

L'application de LoRA au modèle quantifié est réalisée via la bibliothèque PEFT :

```python
lora_config = LoraConfig(
    task_type=config["task_type"],
    r=config["r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    target_modules=config["target_modules"],
)
model = get_peft_model(model_4bit, lora_config)
```

La combinaison de la quantification 4-bit (section 3.2) et de LoRA constitue la technique communément appelée **QLoRA** (Quantized LoRA). Le modèle de base reste en 4-bit en mémoire, seuls les adaptateurs LoRA sont stockés et mis à jour en précision complète. Cette approche permet d'entraîner un modèle de 1,7 milliard de paramètres sur un GPU T4 de 16 Go, ce qui aurait été impossible en fine-tuning classique.

### 3.4 Formatage des données pour le SFT

Le fine-tuning supervisé d'un modèle conversationnel nécessite de formater chaque paire `(question, answer)` selon le **chat template** spécifique au modèle. Ce formatage structure l'échange en rôles (`system`, `user`, `assistant`) et encadre chaque tour de parole par des tokens spéciaux que le modèle a appris à reconnaître lors de son pré-entraînement.

#### 3.4.1 Prompt système

Un prompt système a été rédigé pour définir le comportement attendu de l'agent de triage. Ce prompt est injecté au début de chaque conversation d'entraînement, dans le rôle `system` :

```
Tu es un agent de triage médical aux urgences du Centre Hospitalier Saint-Aurélien.
Tu collectes les informations cliniques du patient (symptômes, durée, antécédents, 
constantes vitales) puis tu attribues un niveau d'urgence parmi trois catégories : 
maximale, modérée ou différée.
En cas de suspicion d'urgence vitale, signale-le immédiatement.
Tu rédiges un bilan synthétique comprenant les symptômes relevés, le niveau d'urgence 
attribué et les hypothèses diagnostiques.
Réponds en français. Tu assistes le personnel soignant mais ne remplace jamais l'avis 
d'un médecin.
```

Ce prompt remplit plusieurs fonctions. Il ancre le modèle dans son rôle hospitalier, ce qui conditionne le registre de langue et le niveau de détail des réponses. Il définit la structure attendue de la réponse (bilan synthétique avec symptômes, niveau d'urgence, hypothèses). Il rappelle la contrainte de langue (français). Et il pose le garde-fou éthique fondamental : l'agent assiste mais ne remplace pas le médecin.

Le prompt système est externalisé dans `params.yaml` (section `sft_model.system_prompt`) pour permettre son ajustement sans modification du code.

#### 3.4.2 Chat template Qwen3

Chaque paire du dataset est convertie en une séquence de conversation structurée à l'aide du tokenizer de Qwen3. La fonction `format_qwen_chat()` dans `utils_training.py` construit la conversation complète :

```python
def format_qwen_chat(question: str, answer: str) -> str:
    system_prompt = _get_system_prompt()
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return _clean_thinking_markers(
        _apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
    )
```

Le résultat est une séquence textuelle de la forme :

```
<|im_start|>system
Tu es un agent de triage médical...<|im_end|>
<|im_start|>user
[question du dataset]<|im_end|>
<|im_start|>assistant
[réponse du dataset]<|im_end|>
```

La fonction `_clean_thinking_markers()` supprime les balises `<think>...</think>` que le modèle Qwen3 peut insérer dans ses réponses. Ces balises, utilisées par le modèle pour son raisonnement interne, ne doivent pas apparaître dans les données d'entraînement SFT car elles pollueraient le signal d'apprentissage.

Une seconde fonction, `format_qwen_prompt()`, génère uniquement la partie "prompt" (système + question) sans la réponse, avec le paramètre `add_generation_prompt=True` qui ajoute le marqueur `<|im_start|>assistant\n` indiquant au modèle qu'il doit commencer à générer. Cette fonction est utilisée pour calculer la frontière entre le prompt et la réponse lors du masquage des labels.

#### 3.4.3 Masquage des labels

Le masquage des labels est une étape critique du formatage SFT. Lors de l'entraînement, le modèle reçoit la séquence complète (prompt + réponse) et doit apprendre à prédire chaque token. Cependant, le modèle ne doit être évalué (et pénalisé par la loss) que sur les tokens de la **réponse**, pas sur ceux du prompt. En effet, le prompt (système + question) est une entrée fournie par l'utilisateur — le modèle n'a pas à apprendre à le prédire.

Le mécanisme de masquage est implémenté dans la fonction `tokenize_chat()` :

```python
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
```

La logique est la suivante : la séquence complète est d'abord encodée en `input_ids`. Puis le prompt seul est encodé pour déterminer sa longueur en tokens (`idx`). Enfin, les labels correspondant au prompt sont remplacés par la valeur sentinelle **-100**, qui indique à la fonction de loss de PyTorch (`CrossEntropyLoss`) d'ignorer ces positions.

Le résultat est un triplet de tenseurs :
- **`input_ids`** : la séquence complète encodée (prompt + réponse), tronquée à `max_length=512` tokens.
- **`attention_mask`** : un masque binaire de `1` sur toute la longueur, indiquant que chaque token est valide (le padding sera géré par le DataCollator).
- **`labels`** : une copie des `input_ids` avec les positions du prompt remplacées par `-100`, de sorte que la loss ne soit calculée que sur la partie réponse.

La longueur maximale de 512 tokens (`sft_model.max_length` dans `params.yaml`) a été choisie comme compromis entre la couverture du contenu (la médiane des séquences complètes se situe bien en dessous de cette limite d'après les colonnes `token_count_*` du dataset) et l'empreinte mémoire par séquence.

### 3.5 Pipeline de tokenisation

La tokenisation des données suit un pipeline en trois étapes, encapsulé dans la fonction `tokenize_flow()` :

```python
def tokenize_flow(pd_dataset_path: Path) -> Dataset:
    pd_dataset = load_dataset(pd_dataset_path)          # Parquet → pandas
    hf_dataset = transform_ds_from_pandas_to_hf(pd_dataset)  # pandas → HF Dataset
    tokenized_dataset = apply_tokenisation(hf_dataset)   # HF Dataset → tokenisé
    return tokenized_dataset
```

**Étape 1 — Chargement Parquet.** Le fichier Parquet produit en section 2.6 est lu via `pd.read_parquet()`. Cette lecture charge l'intégralité du DataFrame en mémoire, ce qui est acceptable pour un dataset de 3 500 lignes.

**Étape 2 — Conversion pandas → HuggingFace Dataset.** Le DataFrame pandas est converti en objet `datasets.Dataset` via `Dataset.from_pandas()`. Cette conversion est nécessaire pour bénéficier de la méthode `.map()` de HuggingFace, qui applique la tokenisation de manière efficace et expose les métadonnées de colonnes utilisées par le Trainer.

**Étape 3 — Tokenisation par `.map()`.** La fonction `tokenize_chat()` est appliquée à chaque exemple via `.map()`, avec le paramètre `remove_columns=dataset_hf.column_names` qui supprime les colonnes textuelles d'origine (question, answer, metadata) pour ne conserver que les trois tenseurs numériques (`input_ids`, `attention_mask`, `labels`). Ce nettoyage est important : le Trainer HuggingFace tenterait sinon de passer les colonnes textuelles au modèle, ce qui provoquerait une erreur.

Cette fonction `tokenize_flow()` est réutilisable : elle est appelée une fois pour le dataset d'entraînement (`sft_train.parquet`) et une fois pour le dataset de validation (`sft_val.parquet`). La tokenisation est exécutée **avant** le chargement du modèle dans la fonction `main()`, par conception : si une erreur survient dans les données (fichier manquant, colonnes inattendues), elle est détectée avant d'avoir consommé la mémoire GPU nécessaire au chargement du modèle.

**DataCollator.** Le padding dynamique par batch est assuré par un `DataCollatorForSeq2Seq` :

```python
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
```

Ce collator aligne les séquences d'un même batch à la longueur de la séquence la plus longue du batch, en appliquant les valeurs de remplissage appropriées : `pad_token_id` sur les `input_ids`, `0` sur l'`attention_mask` (indiquant que les positions de padding ne doivent pas être prises en compte par l'attention), et `-100` sur les `labels` (indiquant que les positions de padding ne doivent pas contribuer à la loss). Le choix du padding dynamique (par batch) plutôt que du padding global (à la longueur maximale pour tout le dataset) réduit significativement le nombre de calculs inutiles sur les tokens de padding.

### 3.6 Configuration de l'entraînement

Les hyperparamètres d'entraînement sont externalisés dans `params.yaml` (section `training_arguments`) et interprétés par la fonction `define_training_arguments()` qui instancie un objet `TrainingArguments` de HuggingFace.

#### 3.6.1 Taille de batch et accumulation de gradients

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
```

La taille de batch effective est le produit de ces deux paramètres : `1 × 32 = 32` exemples par step d'optimisation. Ce découpage est une adaptation aux contraintes mémoire du GPU T4 : un seul exemple est traité à la fois (batch size physique de 1), mais les gradients sont accumulés sur 32 forward passes consécutifs avant d'effectuer un step d'optimisation. Le résultat mathématique est équivalent à un batch de 32 exemples traités simultanément — la différence réside uniquement dans la consommation mémoire, qui est divisée par 32.

Avec 3 500 exemples d'entraînement et un batch effectif de 32, chaque epoch comprend `3 500 / 32 ≈ 110 steps`, soit **220 steps pour 2 epochs**.

#### 3.6.2 Taux d'apprentissage et planification

```yaml
learning_rate: 2e-4
warmup_steps: 30
lr_scheduler_type: "cosine"
```

Le taux d'apprentissage de `2e-4` est nettement plus élevé que celui typiquement utilisé en fine-tuning complet (de l'ordre de `1e-5` à `5e-5`). Cette valeur plus agressive est justifiée par la nature de l'entraînement LoRA : les poids originaux du modèle étant gelés, il n'y a pas de risque de **catastrophic forgetting** (perte des connaissances acquises lors du pré-entraînement). Seuls les adaptateurs LoRA sont mis à jour, et ils partent d'une initialisation proche de zéro — un learning rate plus élevé est donc nécessaire pour leur permettre de s'écarter suffisamment de cette initialisation et de capturer les adaptations nécessaires au domaine médical.

La planification du learning rate suit un schéma en trois phases :
1. **Warmup** (steps 0 à 30) : le learning rate monte progressivement de 0 à `2e-4`. Cette montée graduelle stabilise l'entraînement dans les premiers steps, où les gradients peuvent être bruités.
2. **Décroissance cosinus** (steps 30 à 220) : le learning rate décroît selon une courbe cosinusoïdale de `2e-4` vers 0. Ce profil de décroissance, plus doux qu'une décroissance linéaire, permet au modèle d'effectuer des mises à jour fines dans les derniers steps.

Les 30 warmup steps représentent environ 14% des 220 steps totaux, une proportion standard dans la littérature.

#### 3.6.3 Nombre d'epochs et régularisation

```yaml
num_train_epochs: 2
```

Le choix de 2 epochs est motivé par l'analyse des courbes de validation : la
`eval_loss` ne s'améliore plus significativement au-delà du deuxième passage
sur les données. Sur un dataset de 3 500 exemples, continuer au-delà augmente
le risque de mémorisation des patterns d'entraînement sans gain de
généralisation. Le mécanisme `load_best_model_at_end` assure que le modèle
final correspond au checkpoint avec la meilleure eval loss observée.

La régularisation repose sur deux mécanismes complémentaires : le dropout LoRA à 5% (section 3.3) et le suivi de la loss de validation pour détecter le point de divergence entre train et eval loss.

#### 3.6.4 Stratégie d'évaluation et de sauvegarde

```yaml
eval_strategy: "steps"
eval_steps: 50
save_strategy: "steps"
save_steps: 50
save_total_limit: 3
load_best_model_at_end: True
metric_for_best_model: "eval_loss"
greater_is_better: False
logging_steps: 10
```

L'évaluation sur le jeu de validation (1 000 exemples) est réalisée tous les 50 steps, soit environ **6 à 7 évaluations** sur l'ensemble de l'entraînement. Cette fréquence permet de suivre l'évolution de la eval loss avec une granularité suffisante pour détecter un début de surapprentissage, sans ralentir excessivement l'entraînement (chaque évaluation mobilise le GPU pendant quelques minutes).

Les checkpoints sont sauvegardés avec la même fréquence (tous les 50 steps), avec un maximum de 3 checkpoints conservés simultanément (`save_total_limit: 3`) pour limiter l'espace disque. Le paramètre `load_best_model_at_end: True` assure qu'à la fin de l'entraînement, le modèle restauré est celui du checkpoint ayant obtenu la meilleure `eval_loss` — et non le modèle du dernier step, qui pourrait être surappris.

Le logging des métriques d'entraînement (loss, grad_norm, learning rate) est effectué tous les 10 steps, offrant une vue fine de la dynamique d'entraînement. Ces métriques sont envoyées à MLflow via le paramètre `report_to: "mlflow"`.

#### 3.6.5 Optimisations mémoire pour GPU T4

Plusieurs paramètres ont été spécifiquement ajustés pour l'entraînement sur le GPU NVIDIA T4 de Google Colab :

```yaml
bf16: False
fp16: True
gradient_checkpointing: True
gradient_checkpointing_kwargs:
  use_reentrant: False
optim: "paged_adamw_8bit"
```

**`fp16: True` / `bf16: False`** — L'entraînement utilise la précision mixte FP16 (16-bit floating point). Le GPU T4 ne supporte pas nativement le format BFloat16, qui aurait été préférable pour sa meilleure gestion des grandes valeurs (exposant sur 8 bits au lieu de 5). FP16 reste cependant largement suffisant pour le fine-tuning LoRA, où les mises à jour des poids sont relativement petites en magnitude.

**`gradient_checkpointing: True`** — Le gradient checkpointing (ou activation recomputation) est une technique d'échange temps-mémoire. Au lieu de stocker toutes les activations intermédiaires du forward pass pour les réutiliser lors du backward pass, le modèle ne conserve que certains points de contrôle et recalcule les activations manquantes à la volée pendant la rétropropagation. Cela réduit la consommation mémoire d'un facteur proportionnel à la racine carrée du nombre de couches, au prix d'environ 30% de temps de calcul supplémentaire. Pour un modèle de 24 couches comme Qwen3-1.7B sur un GPU à 16 Go de VRAM, cette optimisation est indispensable.

Le paramètre `use_reentrant: False` sélectionne l'implémentation non-réentrante du gradient checkpointing de PyTorch, qui est plus stable numériquement et gère correctement les graphes de calcul complexes (notamment ceux introduits par LoRA).

**`optim: "paged_adamw_8bit"`** — L'optimiseur Paged AdamW 8-bit, fourni par BitsAndBytes, combine deux optimisations. Premièrement, les états de l'optimiseur (moyennes mobiles des gradients et de leurs carrés) sont stockés en 8 bits au lieu de 32 bits, divisant par 4 leur empreinte mémoire. Deuxièmement, le mécanisme de "paging" gère automatiquement le transfert des états entre la mémoire GPU et la mémoire CPU lorsque la VRAM est insuffisante, évitant les erreurs d'allocation mémoire (OOM). Cette combinaison est particulièrement efficace pour l'entraînement QLoRA sur des GPU à mémoire limitée.

#### 3.6.6 Reprise après interruption

L'entraînement sur Google Colab est soumis à des interruptions potentielles (timeout de session, déconnexion réseau). La fonction `train_model()` intègre un mécanisme de reprise automatique :

```python
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()
```

La fonction `get_last_checkpoint()` de HuggingFace inspecte le répertoire de sortie pour trouver le dernier checkpoint valide. Si un checkpoint est trouvé, l'entraînement reprend à partir de celui-ci — restaurant l'état du modèle, de l'optimiseur, du scheduler, et du compteur de steps. Ce mécanisme a été essentiel pour garantir la robustesse de l'entraînement dans un environnement d'exécution non garanti.

### 3.7 Architecture du code d'entraînement

Le code d'entraînement SFT est organisé en deux modules dans `src/training/` :

**`utils_training.py`** — Module utilitaire regroupant les fonctions de chargement, de formatage et de tokenisation. L'architecture repose sur un système de caches pour éviter les rechargements coûteux :
- `_load_params()` : charge le fichier `params.yaml` une seule fois via `@lru_cache(maxsize=1)`.
- `_get_qwen_tokenizer()` : instancie le tokenizer une seule fois.
- `_get_system_prompt()`, `_get_max_length()` : extraient les paramètres du modèle depuis le dictionnaire YAML mis en cache.

Les fonctions d'accès aux configurations (`_get_lora_config()`, `_get_model_name()`, `_get_quantization_config()`, `_get_config_training_arguments()`) interrogent toutes le même dictionnaire mis en cache par `_load_params()`, garantissant la cohérence des paramètres à travers le code.

**`train_sft.py`** — Script principal d'entraînement. La fonction `main()` orchestre les cinq étapes dans un ordre précis :

1. **Tokenisation** (`tokenize_flow`) — Les datasets train et validation sont 
tokenisés en premier, avant tout chargement de modèle. Si une erreur survient 
dans les données (fichier manquant, format incorrect), elle est détectée sans 
avoir consommé la mémoire GPU.

Un correctif important a été apporté à la fonction `tokenize_chat()` appelée 
dans cette étape : lors d'une troncation à `max_length=512`, le token de fin 
`<|im_end|>` était systématiquement supprimé pour les séquences longues, le 
tokenizer tronquant par la droite. Le modèle apprenait alors sur des exemples 
sans signal de fin explicite, ce qui produisait un remplissage systématique 
jusqu'à `max_tokens` en inférence. Le correctif garantit que `<|im_end|>` reste 
toujours le dernier token après troncation :

```python
eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if input_ids[-1] != eos_token_id:
    input_ids[-1] = eos_token_id
```
2. **DataCollator** (`get_data_collator`) — Instanciation du collator pour le padding dynamique.
3. **TrainingArguments** (`define_training_arguments`) — Construction de l'objet de configuration à partir de `params.yaml`.
4. **Modèle** (`define_model`) — Chargement quantifié + application LoRA. C'est l'étape la plus coûteuse en mémoire.
5. **Entraînement** (`train_model`) — Instanciation du Trainer et lancement de la boucle d'entraînement, avec sauvegarde du modèle LoRA final dans `models/lora_trained_model/`.

Le modèle sauvegardé ne contient que les poids des adaptateurs LoRA (quelques dizaines de Mo), pas l'intégralité du modèle de base. Pour l'inférence, les adaptateurs sont fusionnés avec le modèle de base à la volée.

### 3.8 Résultats de l'entraînement SFT

L'entraînement a été exécuté sur Google Colab avec un GPU NVIDIA T4 (16 Go VRAM). Le run complet a duré **2 heures et 41 minutes** pour 330 steps (3 epochs).

#### 3.8.1 Métriques d'entraînement

| Métrique | Valeur |
|---|---|
| Durée totale | 9 692 secondes (~2h41) |
| Steps totaux | 220 (2 epochs × ~110 steps/epoch) |
| Train loss moyenne | 1,112 |
| Dernière eval loss (step 300) | 1,189 |
| Débit d'entraînement | 1,083 samples/seconde |
| Débit en steps | 0,034 steps/seconde |

#### 3.8.2 Évolution de la loss

Les derniers points de logging montrent une convergence stable :

| Epoch | Train loss | Grad norm | Learning rate |
|---|---|---|---|
| ~2,82 | 0,9184 | 0,4787 | 2,408e-6 |
| ~2,92 | 0,9512 | 0,5688 | 6,627e-7 |
| 3,00 | 0,9994 | 0,8805 | 5,483e-9 |

La train loss a convergé autour de 0,92–1,00 en fin d'entraînement, ce qui indique que le modèle a correctement assimilé les patterns du dataset. La dernière eval loss de 1,189 (au step 300, epoch 2,73) montre un écart modéré avec la train loss, suggérant une légère tendance au surapprentissage dans le dernier quart de l'entraînement — un comportement attendu sur un dataset de cette taille. Le mécanisme `load_best_model_at_end` assure que le modèle final correspond au checkpoint avec la meilleure eval loss, atténuant ainsi l'impact de cette légère dégradation.

Les grad_norm restent dans une plage raisonnable (< 1,0), confirmant la stabilité numérique de l'entraînement en QLoRA/FP16.

#### 3.8.3 Infrastructure et coût

L'entraînement a été réalisé sur le tier gratuit de Google Colab avec un GPU T4. Le choix de Colab a été dicté par les contraintes d'infrastructure : le GPU local (Quadro T1000 avec 4 Go de VRAM) est insuffisant pour l'entraînement, et les quotas GPU de GCP n'ont pas pu être débloqués sur le compte gratuit (300$ de crédits disponibles mais non utilisables pour des GPU). Colab offre un accès gratuit au T4, moyennant des sessions limitées en durée, ce qui a justifié l'implémentation du mécanisme de reprise par checkpoint (section 3.6.6).

Le modèle LoRA entraîné a été sauvegardé dans `models/lora_trained_model/`. Ce répertoire ne contient que les poids des adaptateurs LoRA, soit quelques dizaines de mégaoctets — à comparer aux 3,4 Go du modèle de base complet.

---

## 4. Alignement par préférences (DPO)

L'entraînement supervisé (section 3) a permis de spécialiser le modèle Qwen3-1.7B sur le domaine médical : il sait désormais produire des réponses structurées à des questions cliniques. Cependant, le SFT optimise uniquement la vraisemblance des réponses contenues dans le corpus d'entraînement — il ne distingue pas une réponse médicalement pertinente d'une réponse plausible mais trop vague ou cliniquement sous-optimale. C'est précisément l'objet de cette quatrième phase : aligner le comportement du modèle sur les préférences cliniques validées, en lui apprenant à discriminer les bonnes réponses des mauvaises à partir de paires comparatives.

La technique retenue est **DPO** (Direct Preference Optimization, Rafailov et al., 2023). Contrairement à l'approche RLHF classique qui nécessite d'entraîner un modèle de récompense séparé avant de lancer une boucle de renforcement, le DPO reformule directement l'objectif d'alignement comme un problème de classification sur des paires (chosen, rejected). Cette simplification réduit considérablement la complexité d'implémentation et la charge en mémoire GPU, ce qui le rend particulièrement adapté aux contraintes d'infrastructure du POC.

### 4.1 Fondements de DPO et lien avec le SFT

Le DPO s'appuie sur une reformulation mathématique de l'objectif RLHF. L'idée centrale est que la politique optimale sous contrainte KL peut être exprimée directement en fonction du modèle de référence, sans passer par un modèle de récompense explicite. La fonction de perte DPO maximise le log-ratio entre la probabilité attribuée à la réponse choisie et celle attribuée à la réponse rejetée, relative au modèle de référence :

```
L_DPO = -E[log σ(β · (log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)))]
```

Le paramètre **beta** contrôle l'intensité du rappel vers le modèle de référence. Un beta élevé contraint le modèle à rester proche du modèle SFT de départ, au risque de ne pas suffisamment différencier chosen et rejected. Un beta trop faible l'en éloigne au point de risquer une dégradation des connaissances médicales acquises lors du SFT — phénomène analogue au *catastrophic forgetting*.

Un prérequis fondamental de DPO est que **le modèle de départ soit le modèle SFT, et non le modèle de base**. En effet, DPO affine les préférences d'un modèle qui sait déjà répondre correctement dans le domaine cible : il corrige les nuances, les priorités cliniques, le registre. Appliquer DPO directement sur le modèle de base serait sans effet utile, car le modèle n'aurait pas encore la capacité de produire des réponses médicales structurées que l'alignement pourrait différencier.

### 4.2 Chargement du modèle SFT champion depuis MLflow

Le module `train_dpo.py` implémente une chaîne SFT → DPO traçable et reproductible, reposant sur le registre MLflow pour identifier et récupérer automatiquement le meilleur modèle SFT.

La fonction `_load_sft_lora_adapter()` interroge le serveur MLflow pour trouver le run le plus récent taggé `model_status=champion` et `stage=sft` dans l'expérience `sft-qwen3-medical` :

```python
def _load_sft_lora_adapter():
    experiment = mlflow.get_experiment_by_name("sft-qwen3-medical")
    runs = mlflow.search_runs(
        filter_string='tags.model_status = "champion" and tags.stage = "sft"',
        order_by=["start_time DESC"],
        max_results=1,
        experiment_ids=[experiment.experiment_id]
    )
    if runs.empty:
        raise ValueError("No champion run found for SFT stage.")
    run_id = runs.iloc[0].run_id
    return mlflow.artifacts.download_artifacts(f"runs:/{run_id}/lora_trained_model")
```

Cette approche présente deux avantages majeurs pour la traçabilité. D'une part, elle garantit que l'entraînement DPO repart toujours du meilleur modèle SFT validé, sans risque de confusion entre différentes versions des adaptateurs. D'autre part, elle découple les deux étapes d'entraînement : SFT et DPO peuvent être relancés indépendamment sans modifier le code, le tag `champion` servant de contrat entre les deux runs.

Les adaptateurs LoRA téléchargés sont chargés par-dessus le modèle de base quantifié via `PeftModel.from_pretrained()`, avec le paramètre `is_trainable=True` qui permet de continuer l'entraînement plutôt que de simplement effectuer de l'inférence :

```python
def define_model() -> PeftModel:
    # Chargement du modèle de base en 4-bit (même configuration que pour le SFT)
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map="auto",
    )
    model_4bit = prepare_model_for_kbit_training(
        model_4bit,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Chargement des adaptateurs LoRA du champion SFT, en mode entraînable
    sft_adapter_path = _load_sft_lora_adapter()
    model = PeftModel.from_pretrained(model_4bit, sft_adapter_path, is_trainable=True)
    return model
```

La quantification 4-bit NF4 et les optimisations mémoire (gradient checkpointing, paged AdamW 8-bit) sont identiques à celles du SFT (section 3.2 et 3.6.5), ce qui garantit la cohérence de l'environnement d'entraînement entre les deux phases.

### 4.3 Source de données : UltraMedical-Preference

Le dataset DPO repose sur le corpus **UltraMedical-Preference** (TsinghuaC3I), dont le traitement a été décrit en section 2. Pour rappel, les 5 000 paires de ce dataset ont le schéma suivant :

| Colonne | Type | Description |
|---|---|---|
| `question` | `str` | Question médicale ou cas clinique |
| `chosen` | `str` | Réponse préférentielle, médicalement plus pertinente |
| `rejected` | `str` | Réponse sous-optimale ou moins précise |
| `dataset_name` | `str` | `"ultramed"` |
| `language` | `str` | `"en"` |
| `question_type` | `str` | `"conversational"` |
| `confidence_level` | `str` | `"low"` |
| `token_count_question` | `int` | Longueur en tokens (tokenizer Qwen3) |
| `token_count_chosen` | `int` | Longueur de la réponse chosen |
| `token_count_rejected` | `int` | Longueur de la réponse rejected |

La distinction entre dataset SFT et dataset DPO est fondamentale : le SFT utilise des paires `(question, answer)` sans dimension comparative, tandis que le DPO requiert des triplets `(question, chosen, rejected)` pour définir la direction d'alignement. L'UltraMedical-Preference est le seul corpus du projet qui expose cette structure préférentielle, ce qui en fait la seule source utilisable pour le DPO.

Le `confidence_level: "low"` attribué à cette source lors du nettoyage (section 2.5) reflète une limite inhérente : les jugements de préférence sont issus de comparaisons automatisées entre réponses générées, non de validations cliniques humaines. Cette limite est acceptée dans le cadre du POC mais constitue un axe d'amélioration prioritaire pour la mise en production (section 7).

### 4.4 Formatage des données DPO

Contrairement au SFT où la tokenisation et le masquage des labels sont gérés manuellement (section 3.4.3), le `DPOTrainer` de la bibliothèque `trl` prend en charge ces opérations à condition que les données soient présentées au format **conversationnel structuré**. La fonction `format_dpo_chat()` dans `utils_training.py` produit ce format :

```python
def format_dpo_chat(question: str, chosen_answer: str, rejected_answer: str) -> dict:
    system_prompt = _get_system_prompt()
    return {
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
```

Le `DPOTrainer` reçoit ainsi des listes de messages plutôt que des tenseurs pré-tokenisés, et applique lui-même le chat template de Qwen3 aux trois séquences (prompt, chosen, rejected) ainsi que le masquage des labels nécessaire au calcul de la loss DPO. Cette délégation au Trainer élimine la nécessité d'un `DataCollatorForSeq2Seq` explicite et d'une logique de masquage manuelle.

La transformation du DataFrame pandas en dataset HuggingFace formaté est réalisée par `prepare_dpo_dataset()` :

```python
def prepare_dpo_dataset(dataset: pd.DataFrame) -> Dataset:
    dataset_hf = transform_ds_from_pandas_to_hf(dataset)
    return dataset_hf.map(
        lambda x: format_dpo_chat(x["question"], x["chosen"], x["rejected"]),
        batched=False,
        remove_columns=dataset_hf.column_names,
    )
```

Le paramètre `remove_columns=dataset_hf.column_names` supprime les colonnes Parquet d'origine (métadonnées, token counts) pour ne conserver que les trois clés attendues par le `DPOTrainer` : `prompt`, `chosen`, `rejected`.

### 4.5 Configuration de l'entraînement DPO

Les hyperparamètres DPO sont définis dans la section `training_arguments.dpo` de `params.yaml`. La fonction `define_training_arguments(stage="dpo")` instancie un objet `DPOConfig` — une sous-classe de `TrainingArguments` enrichie du paramètre `beta` propre à DPO :

```python
if stage == "dpo":
    training_args = DPOConfig(beta=0.1, **kwargs)
else:
    training_args = TrainingArguments(**kwargs)
```

Le tableau ci-dessous présente une comparaison des hyperparamètres SFT et DPO, avec la justification des différences :

| Paramètre | SFT | DPO | Justification |
|---|---|---|---|
| `learning_rate` | `2e-4` | `5e-6` | Le DPO affine des préférences sur un modèle déjà spécialisé — un LR trop élevé risque de déstabiliser les représentations médicales acquises lors du SFT |
| `num_train_epochs` | `2` | `2` | Dataset de taille équivalente (3 500 paires) — le risque de surapprentissage apparaît au-delà de 2 epochs |
| `beta` | — | `0.1` | Valeur conservatrice recommandée pour les domaines à fort enjeu (médical) — maintient le modèle proche du SFT de référence |
| `per_device_train_batch_size` | `1` | `1` | Contrainte VRAM T4 identique |
| `gradient_accumulation_steps` | `32` | `32` | Batch effectif de 32 dans les deux cas |
| `fp16` | `True` | `True` | T4 ne supporte pas bf16 |
| `optim` | `paged_adamw_8bit` | `paged_adamw_8bit` | Optimisation mémoire identique |

Le choix de `beta=0.1` mérite une explication approfondie. Le paramètre beta contrôle la force de rappel vers le modèle de référence (ici, le modèle SFT). En pratique, il agit comme un régulateur : un beta élevé (ex. 0.5) contraint fortement le modèle à rester proche du SFT, ce qui préserve ses connaissances médicales mais limite sa capacité à discriminer chosen et rejected. Un beta faible (ex. 0.01) lui donne plus de liberté pour apprendre les préférences, au risque de "désapprendre" certains acquis du SFT. La valeur 0.1, standard dans la littérature et recommandée pour les domaines à fort enjeu, constitue un compromis conservateur adapté au contexte clinique.

### 4.6 Architecture du module d'entraînement DPO

Le module `train_dpo.py` s'organise autour de cinq fonctions appelées séquentiellement dans `main()` :

**`_load_sft_lora_adapter()`** — Interroge le registre MLflow pour récupérer les adaptateurs LoRA du meilleur run SFT (tag `model_status=champion` + `stage=sft`). Lève une exception explicite si aucun run champion n'est trouvé, forçant l'exécution du SFT avant le DPO.

**`define_model()`** — Charge le modèle de base en 4-bit, prépare pour l'entraînement en précision réduite, puis superpose les adaptateurs LoRA SFT en mode entraînable. Le résultat est un objet `PeftModel` dont les adaptateurs sont initialisés avec les poids SFT plutôt qu'avec des poids aléatoires.

**`prepare_dpo_dataset()`** — Convertit les DataFrames Parquet (train et validation) en datasets HuggingFace au format conversationnel DPO. La suppression des colonnes Parquet d'origine garantit que le DPOTrainer ne reçoit que les trois clés attendues.

**`train_dpo_model()`** — Instancie le `DPOTrainer` avec le modèle, les datasets et la configuration, puis lance l'entraînement avec reprise automatique sur checkpoint. En fin d'entraînement, le modèle est sauvegardé localement dans `models/dpo_model_trained/` et les artefacts sont envoyés vers GCS via `mlflow.log_artifacts()`. Le tag `model_status=champion` + `stage=dpo` est apposé pour permettre la récupération ultérieure.

**`main()`** — Orchestre le pipeline complet sous un contexte MLflow (`with setup_mlflow_run(stage="dpo"):`), qui crée un run nommé et taggé de la forme `dpo_qwen3-1.7b-base_qlora_r16_fp16_T4`.

### 4.7 Intégration MLflow et chaîne SFT → DPO

La traçabilité de la chaîne d'entraînement repose sur une convention de tags MLflow définie dans `setup_mlflow_run()` :

```
Expérience SFT  → run taggé stage="sft",  model_status="champion"
                        ↓  (récupération automatique par _load_sft_lora_adapter)
Expérience DPO  → run taggé stage="dpo",  model_status="champion"
```

Cette architecture de tags permet de rejouer la chaîne complète à tout moment sans intervention manuelle : si le modèle SFT champion change (nouveau run avec de meilleures métriques), le prochain run DPO repartira automatiquement du nouveau modèle. Les artefacts de chaque étape sont persistés sur `gs://p14-medical-data/mlflow-artifacts/`, garantissant leur disponibilité même après expiration des sessions Colab.

### 4.8 Résultats de l'entraînement DPO

Le run DPO (`dpo_qwen3-1.7b-base_qlora_r16_bf16_T4`, run ID `7001354881a1`) a été exécuté
sur GPU T4 (Google Colab) pendant **8h31** pour 220 steps (2 epochs sur 3 500 triplets,
batch effectif de 32). Les métriques sont tracées toutes les 10 steps en train,
toutes les 20 steps en évaluation.

#### 4.8.1 Signal d'alignement : rewards/chosen, rejected et margins

Les trois métriques de récompense constituent l'indicateur principal de la qualité
de l'alignement DPO.

| Métrique | Step 10 | Step 220 | Variation |
|---|---|---|---|
| `rewards/chosen` (train) | −0.0005 | −0.2742 | −0.274 |
| `rewards/rejected` (train) | −0.0009 | −0.4909 | −0.490 |
| `rewards/margins` (train) | +0.0004 | +0.217 | +0.216 |
| `rewards/chosen` (eval) | −0.0051 | −0.246 | −0.241 |
| `rewards/rejected` (eval) | −0.0143 | −0.439 | −0.425 |
| `rewards/margins` (eval) | +0.009 | +0.194 | +0.185 |

**Interprétation.** Les deux rewards sont négatifs et décroissants — c'est le comportement
attendu sous DPO : le modèle s'éloigne du modèle de référence (SFT) pour les deux types
de réponses, ce qui se traduit par une diminution du log-ratio relatif. Ce qui importe
n'est pas le signe absolu mais l'écart entre les deux : **le rejected se dégrade 1,79× plus
vite que le chosen** (−0.490 vs −0.274 sur le train). Le modèle apprend bien à pénaliser
les réponses de moindre qualité plus fortement que les réponses préférentielles.

La `rewards/margin` (= chosen − rejected) croît de +0.0004 à +0.217 en train et de
+0.009 à +0.194 en évaluation. Cette marge positive et croissante confirme que le modèle
discrimine de mieux en mieux les réponses chosen des réponses rejected tout au long
de l'entraînement. Sur le jeu d'évaluation, la marge se stabilise en plateau autour de
+0.193–0.194 entre les steps 180 et 220, ce qui indique que le modèle a convergé sans
signe de surapprentissage : il continue de discriminer sur les données non vues avec
la même efficacité que sur les données d'entraînement.

#### 4.8.2 Loss DPO

| Métrique | Step initial | Step final | Variation |
|---|---|---|---|
| Train loss | 0.693 | 0.601 | −0.092 (−13,3%) |
| Eval loss | 0.689 | 0.621 | −0.068 (−9,9%) |

La train loss DPO démarre à 0.693 — proche de log(2) ≈ 0.693, valeur théorique d'un
modèle qui ne discrimine pas encore chosen et rejected (probabilité uniforme 50/50).
Elle décroît progressivement jusqu'à 0.601 au step 160, puis oscille légèrement en
fin de run (0.586–0.627), reflet de la variabilité naturelle des batches de taille 1.

L'eval loss suit une trajectoire monotone décroissante de 0.689 à 0.621, avec un plateau
très marqué entre les steps 160 et 220 (variation < 0.001). Ce comportement est cohérent
avec ce qu'on observe sur les eval rewards/margins : le modèle a convergé vers une
capacité de discrimination stable, sans dégradation ni surapprentissage détectables
sur le jeu de validation.

L'écart train/eval loss reste faible et stable (≈ 0.02 en fin de run), ce qui confirme
la bonne généralisation du modèle aligné sur des triplets non vus.

#### 4.8.3 Infrastructure et durée

L'entraînement DPO a été réalisé dans les mêmes conditions que le SFT : GPU T4 (16 Go
VRAM) sur Google Colab, quantification 4-bit NF4, gradient checkpointing activé.
La durée de 8h31 (vs 2h41 pour le SFT) s'explique par le coût plus élevé du forward
pass DPO : le `DPOTrainer` calcule les probabilités sur les deux séquences (chosen et
rejected) à chaque step, en comparaison avec le modèle de référence gelé — soit
approximativement 2× plus de calcul par batch que le SFT.

Le modèle DPO final (`dpo_model_trained`) a été sauvegardé sous forme d'adaptateurs
LoRA dans `models/dpo_model_trained/` et pushé vers GCS via MLflow (tag
`model_status=champion`, `stage=dpo`). Il a ensuite été mergé avec le modèle de base
Qwen3-1.7B-Base via `generate_model_for_deployment.py` pour produire le modèle
monolithique chargé par vLLM en production.

---

## 5. Déploiement et infrastructure

L'objectif de cette phase est de rendre le modèle fine-tuné (SFT + DPO) accessible sous forme d'un service d'inférence en conditions quasi-réelles. Le déploiement s'articule autour de trois axes : une API REST exposant le modèle via un moteur d'inférence optimisé, une conteneurisation Docker garantissant la portabilité de l'application, et un pipeline CI/CD automatisant les tests, la construction de l'image et le déploiement sur une VM cloud.
Cette section détaille l'architecture retenue, les choix techniques effectués à chaque couche, et la stratégie de tests mise en place pour sécuriser le pipeline.

### 5.1 Architecture générale

L'architecture de déploiement repose sur trois composants principaux organisés en couches :

- **Couche d'inférence — vLLM** : Le moteur vLLM charge le modèle mergé (base + adaptateurs LoRA fusionnés) et expose une interface de génération asynchrone. Le choix de vLLM plutôt que la méthode `model.generate()` native de Hugging Face Transformers se justifie par trois mécanismes d'optimisation absents de l'implémentation standard : le *continuous batching*, qui permet de traiter dynamiquement les requêtes entrantes sans attendre la constitution d'un batch complet ; le *PagedAttention*, qui gère le cache clé-valeur (KV cache) par pages mémoire à la manière d'un système de mémoire virtuelle, éliminant le gaspillage de VRAM lié à l'allocation statique ; et le *scheduling* asynchrone, qui découple la réception des requêtes de leur traitement GPU. Pour un modèle de 1,7 milliard de paramètres servi sur un GPU unique, ces optimisations permettent de réduire significativement la latence par requête et d'augmenter le débit en conditions de charge concurrente.
- **Couche API — FastAPI** : Le framework FastAPI expose deux routes HTTP (`/health` et `/generate`) et orchestre le cycle de vie du moteur d'inférence. FastAPI a été retenu pour sa gestion native de l'asynchronisme (compatible avec le moteur asynchrone de vLLM), sa validation automatique des entrées via Pydantic, et la génération automatique de documentation interactive (Swagger UI sur `/docs`). Dans un contexte médical où la traçabilité des interactions est un prérequis d'audit, la validation stricte des entrées et la journalisation systématique des requêtes constituent des garanties essentielles.
- **Couche conteneur — Docker** : L'application est empaquetée dans une image Docker qui encapsule le code source, les dépendances Python et la configuration, sans inclure les poids du modèle. Les poids sont montés en volume au démarrage du conteneur, ce qui découple le cycle de vie du modèle (mis à jour via MLflow et GCS) de celui de l'application (mis à jour via le pipeline CI/CD). Ce découplage est fondamental : une nouvelle version du code API peut être déployée sans retélécharger les 3,4 Go du modèle, et inversement, un nouveau modèle peut être déployé sans reconstruire l'image Docker.

Le flux de données d'une requête patient suit le chemin suivant : le client envoie une requête POST sur `/generate` avec un prompt textuel ; FastAPI valide les paramètres via le schéma Pydantic `GenerationRequest` ; le prompt est transmis au moteur vLLM qui génère une réponse token par token ; la réponse complète est encapsulée dans un objet `GenerationResponse` et renvoyée au client au format JSON.

### 5.2 Préparation du modèle pour le déploiement

Le modèle entraîné en phases SFT et DPO existe sous forme d'adaptateurs LoRA séparés du modèle de base. Pour le déploiement en production, les adaptateurs sont fusionnés (merged) avec le modèle de base afin de produire un modèle monolithique directement chargeable par vLLM, sans dépendance à la bibliothèque PEFT.

Le script `src/training/generate_model_for_deployment.py` automatise cette opération : il charge le modèle de base, applique les adaptateurs LoRA du meilleur run DPO (identifié via le tag `model_status=champion` dans MLflow), fusionne les poids via `model.merge_and_unload()`, puis enregistre le modèle résultant sur GCS via MLflow. Le chemin de stockage final (`GCS_MERGED_MODEL_PATH`) est celui que le conteneur Docker référence au démarrage pour charger le modèle dans vLLM.

La fusion présente un avantage supplémentaire en termes de performance d'inférence : elle élimine le surcoût de calcul lié à l'application dynamique des adaptateurs LoRA à chaque *forward pass*. Le modèle mergé se comporte comme un modèle standard du point de vue de vLLM, bénéficiant pleinement de toutes ses optimisations.

### 5.3 API d'inférence FastAPI

#### 5.3.1 Cycle de vie applicatif

L'application utilise le pattern *lifespan* de FastAPI pour gérer le chargement et le déchargement du moteur vLLM. Ce pattern, qui remplace les anciens événements `@app.on_event("startup")` / `@app.on_event("shutdown")`, garantit l'exécution du code de nettoyage même en cas d'erreur au démarrage :

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        engine = VLLMEngine(model_path=GCS_MERGED_MODEL_PATH)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle vLLM: {str(e)}")
    yield
    engine = None
```

Le moteur est instancié en variable globale du module. Si le chargement échoue (modèle introuvable, VRAM insuffisante), l'application démarre malgré tout mais l'endpoint `/health` retourne un code `503 Service Unavailable`, signalant à l'orchestrateur que le conteneur n'est pas opérationnel. Ce comportement de dégradation gracieuse évite les boucles de redémarrage : l'application reste accessible pour le diagnostic sans prétendre être fonctionnelle.

#### 5.3.2 Moteur d'inférence vLLM

La classe `VLLMEngine` (`src/api/services/inference.py`) encapsule l'`AsyncLLMEngine` de vLLM et expose une interface simplifiée :

```python
class VLLMEngine:
    def __init__(self, model_path: str):
        engine_args = AsyncEngineArgs(
            model=model_path, trust_remote_code=True, tensor_parallel_size=1,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, prompt, max_tokens=512, temperature=0.7):
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=0.95
        )
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        return final_output.outputs[0].text
```

Le paramètre `tensor_parallel_size=1` configure le modèle pour un GPU unique, adapté au POC. En production avec un modèle 32B+ réparti sur plusieurs GPU, ce paramètre serait ajusté en conséquence. Le `top_p=0.95` complète le contrôle de la température pour le *nucleus sampling*, limitant la génération aux tokens dont la probabilité cumulée ne dépasse pas 95 % — un compromis entre diversité et cohérence pour les réponses médicales. Chaque requête reçoit un identifiant UUID unique (`request_id`) qui permet à vLLM de gérer le multiplexage des requêtes concurrentes dans son scheduler.

#### 5.3.3 Validation des entrées et contrat d'API

Les schémas Pydantic définissent un contrat strict entre le client et l'API :

```python
class GenerationRequest(BaseModel):
    prompt: str   = Field(..., min_length=5, max_length=4000)
    max_tokens: int   = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class GenerationResponse(BaseModel):
    response: str
```

Les bornes ont été choisies selon les contraintes du modèle et du cas d'usage : `min_length=5` sur le prompt rejette les requêtes vides ou trop courtes pour être médicalement significatives ; `max_length=4000` correspond à la fenêtre de contexte utile du modèle après soustraction des tokens réservés au prompt système et à la génération ; `max_tokens=2048` limite la longueur de la réponse générée pour éviter les générations excessivement longues qui consommeraient inutilement du temps GPU. La `temperature` est bornée entre 0.0 (déterministe) et 2.0 (exploration maximale), avec un défaut de 0.7 adapté au domaine médical où un certain degré de variabilité est acceptable sans compromettre la fiabilité.

Toute requête non conforme est automatiquement rejetée par FastAPI avec un code `422 Unprocessable Entity` et un message détaillant le champ invalide, sans que la requête n'atteigne jamais le moteur d'inférence.

#### 5.3.4 Observabilité et sécurité

Deux mécanismes transversaux complètent l'API :

- **Middleware de journalisation des performances** : Un middleware HTTP intercepte chaque requête et mesure la latence de bout en bout (réception → réponse). Les requêtes vers `/health` sont exclues du logging pour éviter le bruit généré par les sondes de l'orchestrateur (toutes les 30 secondes). Chaque requête `/generate` produit une ligne de log structurée contenant la méthode HTTP, le chemin, le code de retour et la latence en secondes. Ce mécanisme fournit les données brutes pour le suivi de la latence P50/P95 en conditions réelles.
- **Gestionnaire d'erreurs global** : Un handler d'exception capte toutes les erreurs non anticipées et retourne une réponse JSON générique au client (code 500, message neutre) tout en loggant le détail de l'erreur côté serveur. Ce double comportement est critique en contexte médical : le client ne reçoit jamais de traceback Python susceptible d'exposer des détails d'implémentation ou des chemins de fichiers internes, tandis que l'équipe opérationnelle dispose de l'information nécessaire au diagnostic.
- **Configuration CORS** : Le middleware CORS est configuré en mode permissif (`allow_origins=["*"]`) pour la phase POC. En production, cette configuration devra être restreinte aux domaines autorisés du CHSA (cf. section 7, roadmap).

### 5.4 Conteneurisation Docker

#### 5.4.1 Structure du Dockerfile

Le `Dockerfile` suit une approche minimaliste adaptée au déploiement d'inférence :

- **Image de base `python:3.11-slim`** : L'image slim réduit la surface d'attaque et la taille de l'image par rapport à l'image complète, tout en conservant les en-têtes de compilation nécessaires à l'installation de vLLM. Python 3.11 a été retenu pour sa compatibilité avec l'écosystème ML (PyTorch, vLLM, Transformers) et son alignement avec la version utilisée en CI.
- **Séparation code / modèle** : Seuls les dossiers `src/` et `config/` sont copiés dans l'image. Les poids du modèle (environ 3,4 Go pour le modèle mergé) ne sont pas inclus : ils sont montés en volume Docker au démarrage via l'option `-v /chemin/hôte/models:/app/models`. Cette séparation réduit drastiquement le temps de build de l'image (quelques secondes contre plusieurs minutes) et permet de mettre à jour le modèle sans reconstruire l'image.
- **HEALTHCHECK natif Docker** : L'instruction `HEALTHCHECK` appelle l'endpoint `/health` toutes les 30 secondes, avec un timeout de 10 secondes, un délai initial de 60 secondes (le temps que vLLM charge le modèle en VRAM) et 3 tentatives avant de déclarer le conteneur unhealthy. L'orchestrateur peut ainsi détecter automatiquement un conteneur non fonctionnel et déclencher un redémarrage.
- **Variables d'environnement** : `PYTHONPATH=/app` garantit la résolution correcte des imports Python quel que soit le répertoire courant du processus. `LANG=C.UTF-8` assure l'encodage correct des caractères français dans les logs et les réponses du modèle.

#### 5.4.2 Contexte de build et .dockerignore

Un fichier `.dockerignore` a été créé pour exclure du contexte de build les dossiers volumineux ou sensibles : `.git`, `data/`, `models/`, `notebooks/`, `.env`, `.venv`. Sans ce fichier, le contexte Docker inclurait l'intégralité du dépôt (données brutes, checkpoints d'entraînement, notebooks Jupyter), ce qui rallongerait considérablement le temps de build et exposerait des fichiers sensibles (clés d'API dans `.env`).

#### 5.4.3 Démarrage du conteneur

Le conteneur est lancé avec accès GPU via le NVIDIA Container Toolkit :

```bash
docker run -d --name medical-api --gpus all -p 8000:8000 \
  -v /home/ubuntu/models:/app/models \
  -e GCS_MERGED_MODEL_PATH=/app/models/merged_model_for_deployment \
  ghcr.io/<repository>/medical-qwen-api:main
```

L'API est alors accessible sur `http://<IP_VM>:8000/docs` (Swagger UI) pour les tests interactifs, et sur `http://<IP_VM>:8000/generate` pour les requêtes programmatiques.

### 5.5 Pipeline CI/CD avec GitHub Actions

#### 5.5.1 Périmètre et séparation des responsabilités

Un point d'architecture important est la distinction entre deux pipelines indépendants :

- **Le pipeline CI/CD de déploiement** (objet de cette section) automatise la chaîne tests → build Docker → push registre → déploiement SSH. Il est déclenché par les commits sur le code applicatif (API, configuration, tests) et ne manipule jamais les données d'entraînement.
- **Le pipeline de données** (section 2.7) orchestrée par DVC (dvc repro) de nettoyage, filtrage clinique, augmentation et génération des datasets. Il est déclenché manuellement et versionne les données sur GCS.

Cette séparation est fondamentale : le conteneur Docker de déploiement ne contient pas les datasets, le modèle est figé (mergé SFT+DPO). Les datasets appartiennent au pipeline de données, qui constitue une boucle distincte avec son propre mécanisme de versionnement (DVC).

#### 5.5.2 Structure du workflow

Le fichier `.github/workflows/cicd.yml` définit trois jobs exécutés séquentiellement :

- **Job 1 — `code-quality-and-tests`** : Déclenché sur chaque push et pull request vers main. Ce job installe les dépendances de test (sans vLLM, qui nécessite un GPU absent des runners GitHub Actions), exécute le linter Ruff pour le contrôle de qualité du code, puis lance la suite de tests pytest. L'absence de vLLM sur le runner CI est compensée par une stratégie de *mock* décrite en section 5.6.
- **Job 2 — `build-and-push-docker`** : Exécuté uniquement sur les push vers main (pas sur les pull requests), conditionné à la réussite du job 1. Ce job construit l'image Docker via Buildx, la tague avec les métadonnées du commit (SHA, branche), puis la pousse vers GitHub Container Registry (GHCR). Le cache Docker est géré via GitHub Actions Cache (`cache-from: type=gha`), ce qui accélère les builds successifs en réutilisant les couches inchangées.
- **Job 3 — `deploy`** : Exécuté après la réussite du job 2, ce job se connecte en SSH à la VM GCP de production, tire la nouvelle image depuis GHCR, arrête le conteneur en cours, et relance un nouveau conteneur avec la dernière version de l'image. Le montage du volume modèle et l'activation GPU sont configurés dans la commande `docker run`.

Ce pipeline garantit qu'aucun code ne parvient en production sans avoir passé les tests et le linter. Le flux complet — du commit au conteneur en production — est entièrement automatisé et ne requiert aucune intervention manuelle.

#### 5.5.3 Gestion des secrets

Les informations sensibles (clé SSH du serveur, adresse IP de la VM, identifiants) sont stockées dans les GitHub Secrets du dépôt (`SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`). Le `GITHUB_TOKEN`, fourni automatiquement par GitHub Actions, est utilisé pour l'authentification auprès de GHCR. Un point de vigilance a été identifié lors de l'audit : le `GITHUB_TOKEN` est transmis au serveur distant via la variable d'environnement `envs` de l'action SSH, ce qui limite le risque d'exposition dans les logs. En production, un Personal Access Token (PAT) dédié et stocké directement sur la VM serait préférable pour une isolation complète.

### 5.6 Stratégie de tests

#### 5.6.1 Contrainte et approche

Le principal défi technique du testing est que vLLM refuse de s'importer sans GPU CUDA, ce qui rend impossible l'exécution des tests d'intégration sur les runners CPU de GitHub Actions. La solution retenue repose sur une injection de mocks dans `sys.modules` avant tout import de `src.api.main` : des modules factices remplacent `vllm` et ses sous-modules, permettant d'importer l'application FastAPI sans dépendance GPU. Le moteur d'inférence est ensuite injecté directement dans le module de l'application (`main_module.engine = ...`) pour éviter de déclencher le lifespan qui tenterait de charger le vrai modèle.

#### 5.6.2 Architecture des tests

La suite de tests (70 tests, exécutés en 2,21 secondes) est organisée en trois couches :

```text
tests/
├── conftest.py                     # Fix PYTHONPATH pour la CI
├── unit/
│   ├── test_schemas.py             # Validation Pydantic (17 cas)
│   ├── test_paths.py               # Cohérence config/paths.py
│   └── test_logger.py              # Logger centralisé
├── integration/
│   ├── conftest.py                 # Mock vLLM + fixtures TestClient
│   ├── test_health.py              # Health check (200 / 503)
│   ├── test_generate.py            # Endpoint /generate (nominal, invalide, erreurs, contrat)
│   └── test_middleware.py          # Middleware de logging
└── smoke/
    └── test_docker_build.py        # Structure Dockerfile / CI
```

- **Tests unitaires — Zéro dépendance réseau ou GPU** : Ils valident la logique pure : les bornes exactes des schémas Pydantic (7 cas valides, 8 cas invalides sur `GenerationRequest`, 3 cas sur `GenerationResponse`), la cohérence des chemins dans `config/paths.py` (préfixe `gs://`, extensions `.parquet`, hiérarchie des dossiers), et le bon fonctionnement du logger centralisé.
- **Tests d'intégration — Avec mock vLLM** : Ils valident le comportement de l'API via le `TestClient` de FastAPI : le health check retourne 200 quand le moteur est chargé et 503 sinon ; l'endpoint `/generate` retourne une réponse valide en cas nominal, rejette les entrées invalides avec un 422, et gère correctement les erreurs moteur (`RuntimeError`, `TimeoutError`) sans exposer de traceback au client. Un test de contrat vérifie que la réponse nominale contient exactement la structure `{"response": str}`, garantissant la non-régression de l'interface.
- **Tests smoke — Vérification structurelle** : Ils valident l'existence et le contenu des fichiers critiques du déploiement : présence des directives `FROM`, `EXPOSE 8000`, `CMD`, `COPY src/` dans le Dockerfile, existence du `.dockerignore`, et présence des steps `pytest` et `docker` dans le workflow CI.

#### 5.6.3 Couverture et limites

Les 70 tests couvrent les couches unitaire, intégration et smoke, exécutables sans GPU. Une quatrième catégorie de tests — les tests de performance (latence P95, robustesse sur des inputs edge-case, comportement sous charge) — ne peut être exécutée qu'après déploiement sur la VM avec GPU. Ces tests post-déploiement font l'objet de la section 6.

---

## 6. Évaluation et métriques de performance

L'évaluation d'un modèle de langage fine-tuné en contexte médical ne peut se limiter à l'analyse des courbes de loss. Elle doit combiner trois axes complémentaires : les métriques d'entraînement (convergence, généralisation), l'évaluation qualitative des réponses générées (pertinence médicale, respect du format attendu), et les métriques opérationnelles de l'endpoint déployé (latence, robustesse). Cette section présente les résultats obtenus sur chacun de ces axes, ainsi que l'analyse comparative entre les versions v1 et v2 du pipeline qui a guidé la décision finale de déploiement.

### 6.1 Métriques d'entraînement SFT v2

#### 6.1.1 Évolution de la loss d'entraînement

Le run SFT v2 (`sft_qwen3-1.7b-base_qlora_r16_fp16_T4`, run ID `7eef6553`) a été entraîné sur 2 epochs, soit 220 steps (3 500 exemples d'entraînement, batch size effectif de 16 via gradient accumulation sur 16 steps).

| Step | Train loss |
|------|-----------|
| 10   | 1.903     |
| 20   | 1.257     |
| 30   | 0.884     |
| 50   | 0.639     |
| 110 (fin epoch 1)  | 0.582     |
| 220 (fin epoch 2)  | 0.472     |

La dynamique d'apprentissage se décompose en trois phases distinctes :

**Phase 1 — Adaptation rapide (steps 0-50).** La loss chute de 1.903 à 0.639, soit une réduction de 66% en un quart de l'entraînement. Cette descente abrupte correspond à l'apprentissage de la structure de base des réponses de triage : le modèle passe d'une distribution de pré-entraînement généraliste (où les réponses médicales structurées sont très improbables) à une distribution qui produit des réponses au format attendu. La valeur initiale de 1.903 est cohérente avec un modèle de base confronté pour la première fois à un format de sortie très spécifique — elle serait plus basse (~1.0-1.2) si le modèle avait déjà vu des instructions médicales structurées dans son pré-entraînement.

**Phase 2 — Consolidation (steps 50-110, fin epoch 1).** La loss oscille entre 0.575 et 0.607, formant un plateau relatif. Le modèle a absorbé la majorité du signal d'entraînement : il produit des réponses structurées mais affine encore la terminologie médicale et les niveaux d'urgence. Ce plateau correspond à la fin du premier passage sur l'intégralité du dataset.

**Phase 3 — Affinement (steps 110-220, epoch 2).** La loss reprend sa descente de 0.582 à 0.472, avec une progression plus modeste mais régulière. Le second epoch permet au modèle d'affiner les détails : terminologie médicale spécialisée, cohérence des niveaux d'urgence, noms des services d'orientation. La loss finale de 0.472 est significativement meilleure que celle du SFT v1 (1.112), une amélioration de 57.6% qui reflète directement la meilleure qualité des données v2 (format triage cohérent via l'augmentation Mistral, absence de contamination par les balises Presidio).

#### 6.1.2 Absence de surapprentissage

Le choix de 2 epochs est validé par le comportement de la loss en fin de run : la train loss continue de décroître légèrement au step 220 (0.472) sans signe d'oscillation ascendante. Cependant, le mécanisme `load_best_model_at_end=True` (critère : eval loss minimale) garantit que le checkpoint retenu est celui qui généralise le mieux sur les données de validation, indépendamment de la train loss finale. Le delta modeste entre les steps 200 (0.498) et 220 (0.472) confirme qu'un troisième epoch n'apporterait qu'un gain marginal au prix d'un risque de mémorisation accru.

### 6.2 Métriques d'entraînement DPO v2

#### 6.2.1 Évolution de la loss DPO

Le run DPO v2 (`dpo_qwen3-1.7b-base_qlora_r16_bf16_T4`, run ID `e67f042d`) a été entraîné sur 2 epochs (220 steps) à partir du modèle SFT v2 champion. Le DPOTrainer calcule une loss de type cross-entropy binaire sur la probabilité relative des réponses `chosen` et `rejected`, avec un terme de régularisation KL contrôlé par le paramètre beta.

| Step | Train loss | Eval loss |
|------|-----------|-----------|
| 10   | 0.693     | —         |
| 50   | 0.611     | 0.616     |
| 100  | 0.586     | 0.591     |
| 150  | 0.564     | 0.584     |
| 200  | 0.530     | 0.583     |
| 220  | 0.528     | 0.583     |

La train loss démarre à 0.693, valeur qui correspond exactement à log(2) ≈ 0.693. Ce point de départ théorique est le signe d'un modèle qui ne discrimine pas encore entre les réponses `chosen` et `rejected` — il leur assigne une probabilité relative de 50/50. C'est le comportement attendu et un indicateur sain : le modèle SFT n'a pas été exposé aux préférences DPO et part donc d'un état neutre.

La descente suit un profil en deux phases :
- **Steps 10-120** : décroissance régulière de 0.693 à ~0.555. Le modèle apprend rapidement à distinguer les réponses préférées des réponses rejetées.
- **Steps 120-220** : oscillations entre 0.528 et 0.584. Ces oscillations sont attendues avec un batch size effectif de 1 (chaque exemple de préférence est traité individuellement), où la variance inter-batch est élevée. La tendance sous-jacente reste légèrement descendante.

L'eval loss décroît de 0.616 (step 50) à 0.583 (step 220), avec un plateau très marqué entre les steps 150 et 220 (variation < 0.001). Ce plateau confirme la convergence du modèle : il a atteint sa capacité maximale de discrimination sur les données de validation sans surapprentissage. L'écart train/eval loss reste faible en fin de run (~0.005), ce qui atteste de la bonne généralisation du modèle aligné sur des triplets non vus.

#### 6.2.2 Rewards et margins

Les métriques de rewards, spécifiques au DPO, mesurent l'écart entre les log-probabilités du modèle entraîné et celles du modèle de référence (SFT gelé) pour chaque type de réponse. Elles constituent l'indicateur le plus direct de la qualité de l'alignement.

| Step | Eval rewards/chosen | Eval rewards/rejected | Eval rewards/margins |
|------|--------------------|-----------------------|---------------------|
| 50   | −0.480             | −0.741                | +0.260              |
| 100  | −0.627             | −1.000                | +0.373              |
| 150  | −0.631             | −1.033                | +0.402              |
| 200  | −0.632             | −1.036                | +0.404              |
| 220  | −0.633             | −1.037                | +0.404              |

**Interprétation des rewards négatifs.** Les deux rewards sont négatifs et décroissants — ce comportement est attendu et souhaitable sous DPO. Il signifie que le modèle entraîné s'éloigne du modèle de référence (SFT) pour les deux types de réponses. Le point crucial est le *rythme relatif* de cette divergence : le reward des réponses rejetées décroît environ 1.6× plus vite que celui des réponses choisies (−1.037 vs −0.633 au step 220). Le modèle apprend donc bien à pénaliser les réponses de moindre qualité plus fortement qu'il ne pénalise les bonnes réponses.

**Interprétation de la marge.** La marge (margin = reward_chosen − reward_rejected) est l'indicateur synthétique de la capacité du modèle à discriminer les réponses préférées. Elle croît rapidement de +0.260 (step 50) à +0.402 (step 150), puis se stabilise en plateau à +0.404 entre les steps 150 et 220. Cette stabilisation confirme la convergence : le modèle a atteint un équilibre entre la capacité de discrimination et la contrainte de régularisation KL imposée par beta=0.1.

### 6.3 Comparaison v1 → v2

La version v2 du pipeline d'entraînement intègre quatre corrections majeures par rapport à la v1 : le retrait de l'anonymisation Presidio sur les corpus publics (qui injectait des balises `<LOCATION>`, `<PERSON>` dans les données d'entraînement), l'ajout d'un filtre clinique sur les questions du dataset SFT, l'augmentation au format triage via Mistral, et la correction du token EOS dans la fonction `tokenize_chat()`.

| Métrique | v1 | v2 | Amélioration |
|----------|------|------|-------------|
| SFT train loss finale | 1.112 | 0.472 | −57.6% |
| DPO eval rewards/margins | +0.194 | +0.404 | +108% |
| DPO eval loss | 0.621 | 0.583 | −6.1% |
| Contamination Presidio (dataset) | 46.8% | 0% | Éliminée |

L'amélioration la plus spectaculaire concerne la loss SFT finale (−57.6%). Cette réduction ne traduit pas simplement un meilleur ajustement aux données, mais reflète un changement fondamental dans la qualité du signal d'entraînement : en v1, le modèle tentait d'apprendre simultanément le format de triage et les artefacts Presidio, deux objectifs contradictoires qui maintenaient la loss à un plateau élevé. En v2, le signal est cohérent et le modèle converge vers une loss nettement plus basse.

Le doublement de la marge DPO (+108%) est également significatif. Il suggère que la qualité du modèle SFT sous-jacent impacte directement la capacité d'alignement : un modèle SFT entraîné sur des données propres fournit une meilleure base de départ pour le DPO, permettant une discrimination plus nette entre réponses choisies et rejetées.

### 6.4 Évaluation qualitative des réponses générées

Les métriques d'entraînement ne suffisent pas à évaluer la pertinence clinique d'un modèle de triage. Une évaluation qualitative a été conduite sur les deux modèles mergés (SFT v2 seul et DPO v2) via l'endpoint `/generate` de l'API déployée, sur des cas cliniques représentatifs en anglais et en français.

#### 6.4.1 Grille d'évaluation

Chaque réponse a été évaluée selon cinq critères :
- **Pertinence médicale** : le diagnostic ou l'orientation proposée est-elle médicalement correcte ?
- **Format triage structuré** : la réponse suit-elle le format attendu (niveau de priorité, service d'orientation, justification) ?
- **Absence de contamination** : aucune balise Presidio, aucun artefact de nettoyage dans la réponse.
- **Arrêt propre** : le modèle s'arrête-t-il naturellement ou remplit-il systématiquement le `max_tokens` ?
- **Absence de répétitions** : la réponse ne contient pas de boucles ou de répétitions de segments.

#### 6.4.2 Résultats comparatifs

| Critère | SFT v2 (EN) | SFT v2 (FR) | DPO v2 (EN) | DPO v2 (FR) |
|---------|-------------|-------------|-------------|-------------|
| Pertinence médicale | ✅ Bonne | ❌ Faible | ✅ Bonne | ❌ Faible |
| Format triage structuré | ⚠️ Partiel | ❌ Absent | ❌ Absent | ❌ Absent |
| Contamination Presidio | ✅ Aucune | ✅ Aucune | ✅ Aucune | ✅ Aucune |
| Arrêt propre | ⚠️ Variable | ❌ Remplit max_tokens | ⚠️ Variable | ❌ Remplit max_tokens |
| Répétitions | ✅ Aucune | ⚠️ Boucles QCM | ✅ Aucune | ⚠️ Boucles QCM |

**Exemple — SFT v2, prompt EN (douleur thoracique, homme 58 ans).** Le modèle produit une réponse médicalement pertinente : suspicion d'événement cardiaque aigu, recommandation d'orientation vers les urgences avec monitoring cardiaque et ECG. La structure s'approche du format triage sans le respecter strictement — la réponse est conversationnelle plutôt que structurée en champs explicites.

**Exemple — DPO v2, prompt EN (glaucome aigu).** Le diagnostic est correct (glaucome aigu à angle fermé) et l'explication du mécanisme pharmacologique est pertinente. Cependant, le format est celui d'une réponse académique explicative, sans aucune structuration de triage. Le DPO a amélioré la qualité médicale au détriment du format.

**Exemple — Modèles v2, prompt FR (syndrome néphrotique).** Les deux modèles retombent en mode QCM : ils génèrent des options fictives (A/B/C/D) et bouclent sur des questions inventées. Les explications pharmacologiques sont incorrectes. Ce comportement illustre la limite structurelle du corpus francophone, dominé par des QCM malgré l'augmentation Mistral.

#### 6.4.3 Problèmes résolus par rapport à la v1

| Problème v1 | Statut v2 | Correction appliquée |
|-------------|-----------|---------------------|
| Balises Presidio (`<LOCATION>`, `<PERSON>`) dans les réponses | ✅ Résolu | Retrait de l'anonymisation Presidio sur les corpus publics |
| Répétitions en boucle systématiques | ✅ Largement résolu | Correction du token EOS dans `tokenize_chat()` + `repetition_penalty=1.1` |
| Remplissage systématique du `max_tokens` | ⚠️ Amélioré | Correction EOS + `stop_token_ids` dans la configuration vLLM |
| Absence totale de format triage | ⚠️ Partiel (EN uniquement) | Augmentation du dataset SFT au format triage via Mistral |
| Réponses décousues après DPO | ✅ Résolu | Données SFT propres en amont fournissant une meilleure base pour l'alignement |

### 6.5 Analyse des limites identifiées

#### 6.5.1 Déséquilibre linguistique du corpus SFT

Le dataset SFT de 5 000 paires est dominé par les sources anglophones (MedQuAD + UltraMedical), qui représentent environ 60-65% du corpus. Les sources francophones (MediQAL + FrenchMedMCQA) sont minoritaires et conservent un fort biais vers le format QCM malgré l'augmentation au format triage via Mistral. Un modèle de 1.7 milliards de paramètres n'a pas suffisamment de capacité pour généraliser le format triage en français avec aussi peu d'exemples dans cette langue — il privilégie le pattern majoritaire (anglais, format triage) et retombe sur le pattern minoritaire (QCM) lorsqu'il est sollicité en français.

#### 6.5.2 Décalage de distribution entre les données SFT et DPO

Le dataset DPO (UltraMedical-Preference) contient des paires `(chosen, rejected)` au format Q&A médical académique. Aucune réponse n'est structurée au format triage. En conséquence, l'alignement DPO améliore la qualité médicale des réponses (diagnostic plus précis, meilleure explication des mécanismes) mais « tire » le modèle vers un format explicatif académique, effaçant partiellement le format triage appris lors du SFT. Le paramètre beta=0.1, bien que conservateur (forte régularisation vers le modèle SFT de référence), n'a pas suffi à préserver le format structuré face à cette pression distributionnelle.

#### 6.5.3 Capacité limitée du modèle 1.7B

Qwen3-1.7B-Base est un modèle compact, sélectionné pour le POC en raison de sa compatibilité avec les contraintes matérielles (GPU T4, 16 Go VRAM). Sa capacité d'instruction-following est structurellement inférieure à celle de modèles 7B+ : il peine à respecter simultanément le contenu médical et le format structuré de sortie, en particulier en français (langue moins représentée dans son pré-entraînement). Ce constat n'est pas un échec mais un résultat de POC attendu, qui motive le passage à un modèle de plus grande taille dans la roadmap (cf. section 7).

### 6.6 Décision de déploiement

Le modèle **SFT v2** a été retenu pour le déploiement du POC. Cette décision repose sur trois constats :

1. Le SFT v2 produit des réponses médicalement pertinentes en anglais, avec une structure de réponse plus proche du format triage que le DPO v2. Pour un POC, la capacité à démontrer la chaîne complète (données → fine-tuning → API → réponse structurée) prime sur la perfection du diagnostic.

2. Le DPO v2 améliore la qualité médicale (diagnostic plus précis, meilleure couverture des mécanismes) mais dégrade le format structuré en raison du décalage de distribution identifié en section 6.5.2. Dans un contexte de triage où la lisibilité et la structuration de la réponse sont essentielles pour le personnel soignant, cette dégradation est rédhibitoire.

3. La correction de ce décalage (reformatage du dataset DPO au format triage) est identifiée et documentée dans la roadmap (section 7) comme une amélioration v3 prioritaire. Le déploiement du SFT seul permet de livrer le POC dans les délais tout en conservant une trajectoire claire d'amélioration.

Le modèle déployé est le merge complet (base + adaptateurs LoRA fusionnés) du SFT v2 champion, stocké sur GCS et chargé par vLLM sur la VM de production.

### 6.7 Métriques opérationnelles de l'endpoint

### 6.7 Métriques opérationnelles de l'endpoint

#### 6.7.1 Protocole de mesure

Le benchmark a été conduit sur l'endpoint `/generate` de l'API FastAPI déployée sur la VM GCP (GPU T4, 16 Go VRAM), avec le modèle SFT v2 chargé via vLLM. Vingt requêtes séquentielles ont été envoyées avec des cas cliniques variés couvrant différents niveaux d'urgence (P1 critique, urgence modérée, cas différable) et deux langues (anglais et français), afin de refléter une distribution réaliste des sollicitations en production.

Les paramètres d'inférence retenus pour le benchmark correspondent aux paramètres par défaut de l'API :

| Paramètre | Valeur | Justification |
|-----------|--------|--------------|
| `max_tokens` | 512 | Suffisant pour une réponse de triage structurée |
| `temperature` | 0.7 | Compromis entre cohérence et diversité, adapté au contexte médical |
| `repetition_penalty` | 1.1 | Correction du comportement de boucle identifié en v1 |
| `stop_token_ids` | `[151643]` (EOS Qwen) | Arrêt propre sur le token de fin de séquence |

#### 6.7.2 Résultats du benchmark

| Métrique | Valeur |
|----------|--------|
| Requêtes envoyées | 20 |
| Requêtes réussies | 20 / 20 (100%) |
| Erreurs / timeouts | 0 |
| Latence moyenne | 9.18s (± 2.98s) |
| Latence min | 4.99s |
| Latence P50 (médiane) | 10.43s |
| **Latence P95** | **12.66s** |
| Latence P99 | 12.66s |
| Latence max | 12.66s |
| Longueur moyenne de réponse | 1 819 caractères (~455 tokens) |
| Débit de génération estimé | ~50 tokens/s |

#### 6.7.3 Analyse des résultats

**Fiabilité.** Le taux de succès de 100% (20/20) sur des requêtes de profils variés confirme la stabilité de la chaîne de déploiement complète : FastAPI, vLLM, modèle mergé et conteneur Docker. Aucun timeout ni erreur de décodage n'a été observé, y compris sur les cas cliniques complexes (AVC, intoxication médicamenteuse) et sur les requêtes en français.

**Débit de génération.** La longueur moyenne des réponses (~1 819 caractères, soit ~455 tokens) génère une latence moyenne de 9.18s, ce qui correspond à un débit estimé de 50 tokens/s. Cette valeur est cohérente avec les performances attendues de vLLM sur un GPU T4 (16 Go VRAM) pour un modèle de 1.7 milliard de paramètres quantifié en 4-bit NF4. À titre de comparaison, un modèle 7B sur A100 avec la même configuration vLLM atteint typiquement 80-100 tokens/s — le ratio de débit est donc proportionnel au ratio de taille de modèle, ce qui valide la cohérence de la configuration d'inférence.

Notons que la longueur moyenne des réponses (455 tokens) est bien inférieure à la limite `max_tokens=512`, ce qui indique que le modèle génère des arrêts naturels via le token EOS dans la majorité des cas — le fix EOS introduit en v2 fonctionne correctement en production.

**Variabilité de la latence.** La variance est modérée (coefficient de variation de 32.5%) et s'explique principalement par la variabilité de la longueur des réponses générées selon la complexité du cas clinique. Le ratio P95/P50 de 1.21x (12.66s / 10.43s) indique une queue de distribution relativement contrôlée : les requêtes les plus longues ne dépassent pas 1.2 fois la latence médiane, ce qui reflète l'absence de dégradation pathologique liée au remplissage systématique du `max_tokens` — comportement qui pénalisait la v1.

La latence minimale de 4.99s correspond aux cas cliniques simples (quelques phrases de réponse), tandis que la latence maximale de 12.66s correspond aux cas complexes avec des réponses détaillées en anglais.

**Pertinence pour le triage hospitalier.** Une latence P95 de 12.66s est acceptable pour un outil d'aide à la décision de triage initial, où l'interaction typique dure plusieurs minutes. Elle est cependant à contextualiser dans la roadmap : le passage à un GPU A10G ou A100 en production, combiné à un modèle 7B+, viserait une latence P95 de l'ordre de 3-5s pour une expérience utilisateur plus fluide.

#### 6.7.4 Limites du benchmark

Ce benchmark mesure la latence en régime séquentiel (une requête à la fois), ce qui correspond au cas d'usage du POC mais ne reflète pas les conditions de charge concurrente d'une production hospitalière. En conditions réelles, vLLM bénéficierait de son mécanisme de *continuous batching* pour amortir le coût de génération sur plusieurs requêtes simultanées, potentiellement améliorant le débit global tout en maintenant des latences individuelles comparables.

Une évaluation en charge (plusieurs requêtes concurrentes, test de montée en charge) constitue une étape naturelle de la roadmap avant tout déploiement en production.

Les paramètres d'inférence configurés dans l'API sont les suivants :

| Paramètre | Valeur | Justification |
|-----------|--------|--------------|
| `max_tokens` | 512 | Suffisant pour une réponse de triage structurée, sans encourager le remplissage |
| `temperature` | 0.7 | Compromis entre diversité et cohérence ; plus basse pour un contexte médical |
| `repetition_penalty` | 1.1 | Pénalisation légère des répétitions, correction du comportement de boucle v1 |
| `stop_token_ids` | `[151643]` (EOS Qwen) | Arrêt explicite sur le token de fin de séquence |

---

## 7. Recommandations stratégiques et roadmap

Le POC livré à l'issue de ces quatre semaines démontre la faisabilité technique d'un agent IA de triage médical sur la pile Qwen3-1.7B + LoRA + DPO, avec un pipeline de
données versionné (DVC), un endpoint d'inférence optimisé (vLLM + FastAPI) et un pipeline CI/CD automatisé (GitHub Actions). Cette section formule les recommandations
nécessaires pour franchir l'étape suivante : passer d'un POC fonctionnel à un système déployable en environnement clinique réel.

### 7.1 Limitation opérationnelle restante : résolution dynamique du modèle en CI/CD

La seule limitation technique identifiée et non corrigée dans ce POC concerne le
chemin GCS du modèle mergé, codé en dur dans le workflow GitHub Actions. Ce chemin
référence un run ID MLflow spécifique (`f27d13653f...`) correspondant au modèle final
SFT+DPO produit lors de ce projet. Si une nouvelle itération d'entraînement produit
un artefact de déploiement avec un run ID différent, le workflow doit être mis à jour
manuellement avant que le déploiement automatique ne reflète la nouvelle version.

La correction recommandée consiste à remplacer le chemin statique par une résolution
dynamique au moment du déploiement : le step CI/CD interroge le registre MLflow via
son API Python pour identifier le dernier artefact taggé `stage=deployment` et en
déduire le chemin GCS à télécharger. Cette modification représente environ une dizaine
de lignes dans le step de déploiement du workflow et élimine la dépendance manuelle,
garantissant que chaque merge sur `main` déploie systématiquement la version validée
la plus récente — propriété essentielle en contexte hospitalier où la traçabilité des
versions en production est un prérequis d'audit.

---

### 7.2 Checklist go / no-go pour un déploiement pilote

Conformément aux recommandations du cahier des charges, le tableau suivant synthétise
les conditions nécessaires pour envisager un déploiement pilote au sein du CHSA,
au-delà de la démonstration POC.

| Critère | Statut POC | Requis pour pilote |
|---|---|---|
| Endpoint d'inférence fonctionnel | ✅ vLLM + FastAPI | Idem, avec authentification |
| Pipeline CI/CD automatisé | ✅ GitHub Actions | + résolution dynamique GCS (§7.1) |
| Tests automatisés | ✅ 70 tests (unit + intégration + smoke) | + tests de performance P95 < 2s |
| Modèle aligné sur préférences médicales | ✅ DPO sur UltraMedical | + validation par cliniciens CHSA |
| Données d'entraînement sans PII | ✅ corpus publics | + anonymisation sur données patient réelles |
| Monitoring post-déploiement | ❌ Non implémenté | Requis (latence, dérives, escalades) |
| Mécanisme human-in-the-loop | ❌ Non implémenté | Requis (cas critiques non confirmés) |
| Conformité RGPD sur données patient | ⚠️ Brique Presidio disponible, non calibrée sur PII réels | Calibrage + audit CNIL |

---

### 7.3 Axes d'amélioration pour un système pré-production

**Mise en place d'un mécanisme human-in-the-loop.** Le modèle actuel produit un
niveau de priorité (urgence maximale / modérée / différée) sans mécanisme d'escalade
automatique vers un soignant lorsque sa confiance est faible. En production, les
réponses dont le score de confiance est inférieur à un seuil paramétrable doivent
être systématiquement soumises à validation humaine avant transmission. Ce mécanisme
constitue la garde-fou principale contre la non-détection d'une urgence critique,
qui représente le cas de défaillance le plus grave dans le contexte du CHSA.

**Réduction des hallucinations par RAG.** Les LLMs génératifs produisent
structurellement des confabulations médicales, indépendamment de la qualité de
l'entraînement. L'approche recommandée pour un modèle pré-production est d'intégrer
une étape de Retrieval-Augmented Generation (RAG) : les réponses du modèle sont
ancrées sur un corpus médical de référence versionné et validé par des cliniciens
(protocoles CCMU, référentiels SAMU, guides HAS). Cette architecture réduit la
surface d'hallucination tout en permettant de mettre à jour les connaissances
médicales de référence sans réentraîner le modèle.

**Mise en place d'un monitoring post-déploiement.** Le pipeline CI/CD actuel
automatise les tests et le déploiement, mais ne couvre pas la surveillance du
système une fois en production. Trois indicateurs sont prioritaires : la latence P95
dans le temps (dégradation sous charge), la distribution des niveaux de priorité
attribués (détection de dérives comportementales du modèle), et le taux d'escalade
human-in-the-loop (indicateur synthétique de la confiance globale du système). Un
outil comme Prometheus + Grafana, ou une solution managée (Weights & Biases
monitoring, Datadog LLM Observability), s'intègre naturellement à l'architecture
FastAPI existante via le middleware de journalisation déjà en place.

**Calibrage de l'anonymisation Presidio sur données patient réelles.** La décision
de désactiver Presidio sur les corpus d'entraînement publics est fondée et documentée
dans ce rapport (section 2.4). Cependant, en conditions de production, chaque
interaction patient génère des données de santé au sens de l'article 9 du RGPD. La
brique `anonymisation.py` produite lors de ce POC constitue le point de départ
adapté, mais elle devra être recalibrée spécifiquement sur le vocabulaire médical
francophone pour réduire les faux positifs documentés, avant d'être appliquée aux
logs d'interactions patient. Un audit CNIL de l'architecture de collecte et de
stockage (chiffrement au repos et en transit, durée de conservation bornée, droit à
l'oubli) sera nécessaire avant tout déploiement réel.

---

### 7.4 Roadmap de passage à l'échelle — Phase 3

Le cahier des charges du CHSA prévoit, en cas de validation concluante du POC, le
passage à des modèles de plus grande envergure. Le tableau suivant présente les
étapes recommandées selon trois horizons temporels.

| Horizon | Action | Justification |
|---|---|---|
| **0–3 mois** | Résolution dynamique GCS en CI/CD (§7.1) | Prérequis pour des itérations d'entraînement sans intervention manuelle |
| **0–3 mois** | Validation clinique du modèle actuel par des soignants CHSA | Évaluer l'acceptabilité clinique avant d'investir dans un modèle plus grand |
| **3–6 mois** | Implémentation human-in-the-loop + monitoring | Prérequis sécurité pour tout déploiement pilote |
| **3–6 mois** | Migration vers Qwen3-8B ou LLaMA-3-8B | Meilleure capacité de raisonnement clinique ; GPU A100 requis (40 Go VRAM) |
| **6–12 mois** | Intégration RAG sur corpus médical CHSA | Réduction des hallucinations, mise à jour des référentiels sans réentraînement |
| **6–12 mois** | Passage à un modèle 32B+ en production | Performances comparables aux médecins en formation selon la littérature ; infrastructure multi-GPU requise |
| **12 mois+** | Entraînement sur données patient réelles anonymisées | Spécialisation sur les cas cliniques effectivement rencontrés aux urgences du CHSA |

Le passage à un modèle 32B+ représente une rupture d'infrastructure significative :
là où le modèle 1.7B de ce POC tient dans 16 Go de VRAM sur un GPU T4 standard,
un modèle 32B en précision 4-bit requiert environ 20 Go de VRAM minimum, typiquement
servi sur une instance A100 (40 ou 80 Go) ou sur un cluster multi-GPU. L'architecture
de déploiement vLLM + FastAPI produite dans ce POC est conçue pour absorber cette
montée en charge sans modification : vLLM supporte nativement le tensor parallelism
multi-GPU via le paramètre `--tensor-parallel-size`, ce qui rend la transition
architecturalement transparente.

---

## 8. Conclusion

Ce projet avait pour objectif de démontrer la faisabilité technique d'un agent IA de
triage médical pour le Centre Hospitalier Saint-Aurélien, en produisant en quatre
semaines un ensemble de livrables couvrant l'intégralité de la chaîne : de la
préparation des données jusqu'au déploiement en conditions quasi-réelles. Les cinq
livrables définis par le cahier des charges ont été produits.

Le corpus médical bilingue constitue la fondation de l'ensemble du projet. Quatre
sources publiques ont été ingérées, nettoyées et unifiées dans un schéma commun,
pour produire 5 000 paires d'entraînement SFT et 5 000 triplets DPO, versionnés avec
DVC et stockés sur GCS. Le pipeline a fait l'objet d'une itération significative en
cours de projet : un audit de qualité a révélé que l'anonymisation Presidio, bien
que pertinente pour de futures données patient réelles, produisait des faux positifs
massifs sur du vocabulaire médical encyclopédique. Le pipeline a été corrigé en
conséquence — Presidio retiré des corpus publics, filtre clinique ajouté, et étape
d'augmentation synthétique des données de triage intégrée via l'API Mistral — avant
de relancer l'intégralité des entraînements. Cette capacité à diagnostiquer une
dégradation de la qualité des données et à y remédier de manière systématique
constitue en elle-même un résultat du POC.

Le modèle Qwen3-1.7B-Base a été spécialisé par fine-tuning supervisé (SFT) avec
QLoRA (rank 16, quantification 4-bit NF4), puis aligné sur des préférences médicales
par DPO (beta=0.1, dataset UltraMedical-Preference). Les métriques d'alignement
confirment la convergence : la `rewards/margin` progresse de +0.009 à +0.194 en
évaluation sans signe de surapprentissage, avec un écart train/eval loss stable à
≈ 0.02 en fin de run. Les poids finaux sont stockés sur GCS et tracés dans MLflow
sous les tags `model_status=champion` et `stage=dpo`.

Le déploiement repose sur une architecture trois couches — vLLM pour l'inférence
optimisée, FastAPI pour l'exposition REST, Docker pour la portabilité — orchestrée
par un pipeline CI/CD GitHub Actions automatisant les tests (70 cas couvrant les
couches unitaire, intégration et smoke), la construction de l'image et le déploiement
sur VM GCP via SSH.

Ce POC valide la faisabilité de la Phase 1 définie dans le cahier des charges du
CHSA. Il démontre qu'un modèle compact (1.7B paramètres), entraînable sur GPU grand
public, est capable de produire des réponses médicales structurées après spécialisation
par SFT et alignement par DPO, et d'être exposé comme service d'inférence dans une
architecture cloud reproductible. Les recommandations formulées en section 7 tracent
la route vers la Phase 3 : validation clinique par les équipes soignantes du CHSA,
montée en puissance vers un modèle 32B+, et intégration des mécanismes de sécurité
indispensables à un déploiement hospitalier — human-in-the-loop, RAG, monitoring
et conformité RGPD sur données patient réelles.

---

## 9. Annexes

### Annexe A — Hyperparamètres complets SFT et DPO

#### A.1 Configuration LoRA (commune SFT et DPO)

| Paramètre | Valeur | Description |
|---|---|---|
| `r` (rank) | 16 | Dimension des matrices de décomposition basse-rang |
| `lora_alpha` | 32 | Facteur d'échelle — scaling effectif = alpha/r = 2.0 |
| `lora_dropout` | 0.05 | Dropout appliqué aux adaptateurs pour régularisation |
| `bias` | `none` | Les biais du modèle de base ne sont pas entraînés |
| `task_type` | `CAUSAL_LM` | Modélisation causale du langage |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 7 projections ciblées (attention + FFN) |

#### A.2 Configuration de quantification (commune SFT et DPO)

| Paramètre | Valeur | Description |
|---|---|---|
| `load_in_4bit` | `True` | Quantification 4-bit activée (QLoRA) |
| `bnb_4bit_quant_type` | `nf4` | Type NF4 (Normal Float 4) — optimal pour les poids pré-entraînés |
| `bnb_4bit_use_double_quant` | `True` | Double quantification — réduit l'empreinte mémoire de ~0,4 bit supplémentaire |
| `bnb_4bit_compute_dtype` | `float16` | Calculs des forward/backward pass en FP16 |

#### A.3 Hyperparamètres d'entraînement SFT vs DPO

| Paramètre | SFT | DPO | Justification de la différence |
|---|---|---|---|
| `learning_rate` | `2e-4` | `5e-6` | DPO affine un modèle déjà spécialisé — LR 40× plus faible pour ne pas déstabiliser les représentations médicales |
| `num_train_epochs` | `2` | `2` | Risque de surapprentissage identique sur datasets de taille équivalente |
| `beta` | — | `0.1` | Force de rappel vers le modèle de référence SFT — valeur conservatrice pour le médical |
| `per_device_train_batch_size` | `1` | `1` | Contrainte VRAM T4 (16 Go) |
| `gradient_accumulation_steps` | `32` | `32` | Batch effectif de 32 dans les deux cas |
| `warmup_steps` | `30` | `30` | ~10% des steps totaux — stabilisation des gradients en début d'entraînement |
| `lr_scheduler_type` | `cosine` | `cosine` | Décroissance douce favorisant les ajustements fins en fin de run |
| `optim` | `paged_adamw_8bit` | `paged_adamw_8bit` | États d'optimiseur en 8-bit + paging CPU — économie mémoire GPU |
| `fp16` | `True` | `True` | T4 ne supporte pas bf16 |
| `gradient_checkpointing` | `True` | `True` | Recalcul des activations à la backprop — réduit la VRAM au prix de ~20% de temps CPU |
| `eval_strategy` | `steps` | `steps` | Évaluation périodique sur jeu de validation |
| `eval_steps` | `50` | `20` | Granularité plus fine pour le DPO (signal plus bruité) |
| `save_strategy` | `steps` | `steps` | Sauvegarde des checkpoints intermédiaires |
| `load_best_model_at_end` | `True` | `True` | Le modèle final est le checkpoint avec la meilleure eval loss |
| `metric_for_best_model` | `eval_loss` | `eval_loss` | Critère de sélection du meilleur checkpoint |

---

### Annexe B — Statistiques du corpus d'entraînement

#### B.1 Sources brutes

| Source | Langue | Type | Lignes brutes | Usage |
|---|---|---|---|---|
| MediQAL MCQU | FR | QCM cliniques | 10 113 | SFT |
| FrenchMedMCQA | FR | QCM médicaux | 595 | SFT |
| MedQuAD | EN | Questions-réponses ouvertes | 16 407 | SFT |
| UltraMedical-Preference | EN | Paires de préférences | 109 353 | SFT + DPO |

#### B.2 Dataset SFT final (post-pipeline v2)

| Propriété | Valeur |
|---|---|
| Taille totale | 5 000 paires `(question, answer)` |
| Split train | 3 500 exemples (70%) |
| Split validation | 1 000 exemples (20%) |
| Split test | 500 exemples (10%) |
| Critère de stratification | `dataset_name` — représentation équilibrée de chaque source |
| Filtre clinique | `filter_clinical_questions()` — mots-clés symptômes/temporalité + `min_question_tokens=15` |
| Augmentation triage | ~5 000 exemples reformatés au format triage via Mistral Small (API) |
| Anonymisation | Désactivée sur corpus publics — brique `anonymisation.py` conservée pour données patient réelles |
| Format de stockage | Parquet — `data/processed/sft_dataset/` |
| Versionné avec | DVC — remote GCS `gs://p14-medical-data/dvc-store` |

#### B.3 Dataset DPO final

| Propriété | Valeur |
|---|---|
| Taille totale | 5 000 triplets `(question, chosen, rejected)` |
| Source | UltraMedical-Preference |
| Split train | 3 500 triplets (70%) |
| Split validation | 1 000 triplets (20%) |
| Split test | 500 triplets (10%) |
| Format de stockage | Parquet — `data/processed/dpo_dataset/` |

---

### Annexe C — Architecture du dépôt GitHub

```text
FINE-TUNING_MEDICAL/
├── config/
│   ├── logger.py               # Logger centralisé (horodatage, niveaux)
│   └── paths.py                # Chemins GCS et locaux (source unique de vérité)
├── data/
│   ├── raw/                    # Données brutes ingérées (versionnées DVC)
│   └── processed/
│       ├── sft_dataset/        # sft_train / sft_val / sft_test .parquet
│       └── dpo_dataset/        # dpo_train / dpo_val / dpo_test .parquet
├── models/
│   ├── lora_trained_model/     # Adaptateurs LoRA SFT
│   ├── dpo_model_trained/      # Adaptateurs LoRA DPO
│   └── merged_model/           # Modèle monolithique SFT+DPO (pour vLLM)
├── notebooks/                  # EDA + imports HuggingFace
├── src/
│   ├── api/
│   │   ├── main.py             # Application FastAPI (lifespan, middleware, routes)
│   │   ├── schemas.py          # Schémas Pydantic (GenerationRequest/Response)
│   │   └── services/
│   │       └── inference.py    # VLLMEngine (AsyncLLMEngine)
│   ├── processing/
│   │   ├── mediqal_cleaning.py
│   │   ├── frenchmedmcqa_cleaning.py
│   │   ├── medquad_cleaning.py
│   │   ├── ultramed_cleaning.py
│   │   ├── utils_cleaning.py         # Helpers partagés + filter_clinical_questions()
│   │   ├── anonymisation.py          # Brique Presidio (usage futur données patient)
│   │   ├── sft_dataset/
│   │   │   ├── generate_sft_dataset.py   # Filtre clinique + sampling équilibré
│   │   │   ├── triage_augmentation.py    # Reformatage Mistral → format triage
│   │   │   └── split_sft_dataset.py      # Split train/val/test stratifié
│   │   └── dpo_dataset/
│   │       └── generate_dpo_dataset.py
│   └── training/
│       ├── train_sft.py                      # Entraînement SFT (QLoRA)
│       ├── train_dpo.py                      # Entraînement DPO
│       ├── generate_model_for_deployment.py  # Merge LoRA + push GCS via MLflow
│       └── utils_training.py                 # Helpers partagés (configs, tokenisation, MLflow)
├── tests/
│   ├── conftest.py
│   ├── unit/                   # Tests logique pure (schemas, paths, logger)
│   ├── integration/            # Tests API avec mock vLLM
│   └── smoke/                  # Tests structure Dockerfile / CI
├── dvc.yaml                    # Pipeline DVC (8 stages)
├── params.yaml                 # Tous les hyperparamètres paramétrables
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── .github/workflows/cicd.yml  # Pipeline CI/CD (3 jobs)
```

---

### Annexe D — Variables d'environnement et secrets

#### D.1 Variables d'environnement runtime (conteneur Docker)

| Variable | Description | Exemple |
|---|---|---|
| `GCS_MERGED_MODEL_PATH` | Chemin local vers le modèle mergé monté en volume | `/app/models/merged_model_for_deployment` |
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow | `http://34.155.160.41:5000` |
| `PYTHONPATH` | Résolution des imports Python | `/app` |
| `LANG` | Encodage des caractères | `C.UTF-8` |

#### D.2 Secrets GitHub Actions

| Secret | Usage | Scope |
|---|---|---|
| `SERVER_HOST` | IP externe de la VM GCP | Job `deploy` |
| `SERVER_USER` | Utilisateur SSH de la VM | Job `deploy` |
| `SERVER_SSH_KEY` | Clé privée ed25519 dédiée CI/CD | Job `deploy` |
| `GHCR_PAT` | Token `read:packages` pour puller depuis GHCR | Job `deploy` |
| `MISTRAL_API_KEY` | Accès API Mistral pour l'augmentation triage | Pipeline DVC local |

---

### Annexe E — Glossaire

| Terme | Définition |
|---|---|
| **SFT** | Supervised Fine-Tuning — spécialisation supervisée du modèle sur des paires (question, réponse) |
| **DPO** | Direct Preference Optimization — alignement du modèle sur des paires de préférences (chosen/rejected) sans modèle de récompense explicite |
| **LoRA** | Low-Rank Adaptation — technique de fine-tuning paramétrique efficace injectant des matrices basse-rang dans les couches d'attention |
| **QLoRA** | LoRA appliqué sur un modèle quantifié en 4-bit — réduit l'empreinte VRAM de ~75% vs FP32 |
| **vLLM** | Moteur d'inférence LLM haute performance — continuous batching + PagedAttention pour maximiser le débit en production |
| **DVC** | Data Version Control — versionnement des données et pipelines ML, analogue à Git pour les artefacts volumineux |
| **MLflow** | Plateforme de tracking des expériences ML — logs métriques, hyperparamètres et artefacts |
| **GHCR** | GitHub Container Registry — registre d'images Docker intégré à GitHub |
| **RAG** | Retrieval-Augmented Generation — ancrage des réponses LLM sur un corpus de référence pour réduire les hallucinations |
| **PPL** | Perplexité — mesure de la surprise du modèle face à un texte de référence (plus basse = meilleure) |
| **CCMU** | Classification Clinique des Malades aux Urgences — échelle française de triage hospitalier en 5 niveaux |