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

Dans le cadre d'un déploiement hospitalier, la conformité au Règlement Général sur la Protection des Données (RGPD) est un prérequis non négociable. L'anonymisation des données d'entraînement vise à garantir qu'aucune donnée personnelle identifiable (PII) ne subsiste dans le corpus utilisé pour le fine-tuning.

L'outil retenu est **Presidio**, une bibliothèque open source développée par Microsoft, spécialisée dans la détection et le masquage automatisé des données sensibles. Le processus d'anonymisation s'insère dans le pipeline après l'étape de nettoyage et avant la constitution du dataset SFT. Cette séquence est intentionnelle : anonymiser sur des données déjà nettoyées évite de traiter des lignes qui seront de toute façon éliminées, optimisant ainsi le temps de calcul.

Le pipeline d'anonymisation repose sur deux composants :

- Un **moteur d'analyse** (`AnalyzerEngine`) configuré avec le modèle linguistique français `fr_core_news_md` pour détecter les entités de type nom, prénom, numéro de téléphone, adresse et email dans les textes francophones.
- Un **moteur d'anonymisation** (`AnonymizerEngine`) qui applique une stratégie de masquage sur les entités détectées.

Il convient de noter que les quatre datasets utilisés sont des corpus publics accessibles sur Hugging Face, composés principalement de questions d'examen ou de contenu médical encyclopédique. Le risque réel de présence de PII y est faible. Néanmoins, la mise en place du mécanisme d'anonymisation démontre la prise en compte de la conformité dès la phase de POC, et constitue une brique réutilisable pour de futurs corpus contenant potentiellement des données patient réelles.

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

La fonction `add_token_counts()`, ajoutée après l'étape d'anonymisation dans les deux pipelines (SFT et DPO), calcule la longueur en tokens de chaque colonne texte à l'aide du tokenizer de Qwen3-1.7B-Base :

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

L'objectif de cette étape est de consolider les quatre corpus nettoyés en un unique dataset de 5 000 paires `(question, answer)` prêt pour le fine-tuning supervisé. Le script `src/processing/sft_dataset/generate_sft_dataset.py` implémente un mécanisme d'échantillonnage équilibré piloté par les paramètres définis dans `params.yaml` :

```yaml
sft:
  target_samples: 5000
  random_state: 42
  source_datasets:
    - mediqal_dataset/mediqal.parquet
    - frenchmedmcqa_dataset/frenchmedmcqa.parquet
    - medquad_dataset/medquad.parquet
    - ultramed_dataset/ultramed.parquet
```

L'algorithme répartit le quota de 5 000 échantillons de manière équitable entre les quatre sources. Pour chaque dataset, le nombre de lignes à prélever est calculé dynamiquement en divisant le quota restant par le nombre de sources restantes à traiter. Si un dataset contient moins de lignes que sa part théorique (cas de FrenchMedMCQA avec ses quelques centaines de lignes nettoyées), l'ensemble de ses données est inclus et le surplus est redistribué aux datasets suivants. Le seed de randomisation (`random_state=42`) assure la reproductibilité de l'échantillonnage.

Le dataset SFT final est d'abord sauvegardé en intégralité sous `data/processed/sft_dataset/sft_dataset.parquet` (5 000 lignes). Un split stratifié est ensuite appliqué pour produire trois sous-ensembles prêts à l'emploi :

| Fichier | Proportion | Volume (sur 5 000) |
|---|:---:|:---:|
| `sft_train.parquet` | 70 % | 3 500 lignes |
| `sft_val.parquet` | 20 % | 1 000 lignes |
| `sft_test.parquet` | 10 % | 500 lignes |

La stratification est réalisée sur la colonne `dataset_name` à l'aide de `sklearn.model_selection.train_test_split` — ce qui garantit que chacune des quatre sources (MediQAL, FrenchMedMCQA, MedQuAD, UltraMedical) est représentée dans les mêmes proportions dans chaque split. Les proportions sont pilotées par les paramètres `val_size: 0.2` et `test_size: 0.1` dans `params.yaml`. Le schéma complet de ces quatre fichiers est décrit en section 2.5.

### 2.7 Versionnement et reproductibilité avec DVC

L'intégralité du pipeline de nettoyage et de constitution du dataset SFT est orchestrée par **DVC (Data Version Control)**, un outil open source de versionnement de données et de pipelines ML. Le choix de DVC répond directement à l'exigence de traçabilité formulée dans le cahier des charges : "conserver une trace de chaque transformation de données".

Le fichier `dvc.yaml` définit cinq stages organisés en graphe acyclique dirigé (DAG) :

```
clean_mediqal ────────┐
clean_medquad ────────┤
clean_frenchmedmcqa ──┤→ generate_sft
clean_ultramed ───────┘
```

Chaque stage déclare explicitement ses dépendances (scripts Python, fichiers de configuration) et ses sorties (répertoires de données traitées). DVC calcule un hash MD5 pour chaque entrée et sortie, ce qui permet de détecter automatiquement si une étape doit être ré-exécutée suite à une modification de code ou de données.

Le fichier `dvc.lock` enregistre l'état exact de chaque exécution : les hash des scripts, les hash des données produites, et les paramètres utilisés. Ce fichier, versionné dans Git, constitue un certificat de reproductibilité : n'importe quel membre de l'équipe peut recréer exactement le même dataset en exécutant `dvc repro`.

Les données produites sont synchronisées avec un remote GCS (`gs://p14-medical-data/dvc-store`), ce qui dissocie le versionnement des données (géré par DVC) du versionnement du code (géré par Git). Cette architecture permet de travailler avec des datasets volumineux (UltraMedical dépasse 140 Mo en Parquet) sans alourdir le dépôt Git.

---

## 3. Fine-tuning supervisé (SFT) avec LoRA

*[À compléter — Semaine 2]*

---

## 4. Alignement par préférences (DPO)

*[À compléter — Semaine 3]*

---

## 5. Déploiement et infrastructure

*[À compléter — Semaine 4]*

---

## 6. Évaluation et métriques de performance

*[À compléter]*

---

## 7. Recommandations stratégiques et roadmap

*[À compléter]*

---

## 8. Conclusion

*[À compléter]*

---

## 9. Annexes

*[À compléter]*