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

Avec 3 500 exemples d'entraînement et un batch effectif de 32, chaque epoch comprend `3 500 / 32 ≈ 110 steps`, soit **330 steps pour 3 epochs**.

#### 3.6.2 Taux d'apprentissage et planification

```yaml
learning_rate: 2e-4
warmup_steps: 30
lr_scheduler_type: "cosine"
```

Le taux d'apprentissage de `2e-4` est nettement plus élevé que celui typiquement utilisé en fine-tuning complet (de l'ordre de `1e-5` à `5e-5`). Cette valeur plus agressive est justifiée par la nature de l'entraînement LoRA : les poids originaux du modèle étant gelés, il n'y a pas de risque de **catastrophic forgetting** (perte des connaissances acquises lors du pré-entraînement). Seuls les adaptateurs LoRA sont mis à jour, et ils partent d'une initialisation proche de zéro — un learning rate plus élevé est donc nécessaire pour leur permettre de s'écarter suffisamment de cette initialisation et de capturer les adaptations nécessaires au domaine médical.

La planification du learning rate suit un schéma en trois phases :
1. **Warmup** (steps 0 à 30) : le learning rate monte progressivement de 0 à `2e-4`. Cette montée graduelle stabilise l'entraînement dans les premiers steps, où les gradients peuvent être bruités.
2. **Décroissance cosinus** (steps 30 à 330) : le learning rate décroît selon une courbe cosinusoïdale de `2e-4` vers 0. Ce profil de décroissance, plus doux qu'une décroissance linéaire, permet au modèle d'effectuer des mises à jour fines dans les derniers steps.

Les 30 warmup steps représentent environ 10% des 330 steps totaux, une proportion standard dans la littérature.

#### 3.6.3 Nombre d'epochs et régularisation

```yaml
num_train_epochs: 3
```

Le choix de 3 epochs est motivé par la taille modeste du dataset (3 500 exemples). Au-delà de 3 passages complets sur les données, le risque de surapprentissage augmente significativement : le modèle commence à mémoriser les exemples individuels plutôt qu'à en extraire des patterns généralisables. Ce phénomène est particulièrement prononcé dans le contexte médical où les formulations peuvent être stéréotypées (questions d'examen suivant des patterns récurrents).

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

1. **Tokenisation** (`tokenize_flow`) — Les datasets train et validation sont tokenisés en premier, avant tout chargement de modèle. Si une erreur survient dans les données (fichier manquant, format incorrect), elle est détectée sans avoir consommé la mémoire GPU.
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
| Steps totaux | 330 (3 epochs × ~110 steps/epoch) |
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

*[À compléter après exécution du run DPO]*

L'analyse des résultats portera sur les axes suivants :

**Métriques quantitatives :**
- Évolution de la `rewards/chosen` et `rewards/rejected` au fil des steps (indicateurs directs de la qualité de l'alignement)
- `rewards/margins` = différence entre les deux — à surveiller : une marge trop faible indique que le modèle ne discrimine pas suffisamment chosen et rejected
- Train loss et eval loss DPO
- Perplexité sur le jeu de test SFT avant et après DPO (pour s'assurer que le DPO n'a pas dégradé les capacités acquises)

**Analyse qualitative :**
- Comparaison de réponses du modèle SFT vs DPO sur une sélection de questions cliniques du jeu de test
- Vérification que les niveaux d'urgence (maximale / modérée / différée) sont mieux calibrés après alignement
- Détection d'éventuelles régressions (cas où le modèle SFT répondait correctement mais le modèle DPO produit une réponse dégradée)

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