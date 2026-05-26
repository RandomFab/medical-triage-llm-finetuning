# Rapport Technique — POC Agent IA de Triage Médical
## Centre Hospitalier Saint-Aurélien (CHSA)

**Auteur :** Fabien BARDOUIL
**Mission :** Proof of Concept — Agent IA de Triage Médical
**Modèle cible :** Qwen3-1.7B-Base (SFT + LoRA + DPO)
**Date :** Avril 2026

---

## Table des matières

1. Introduction
2. Préparation et structuration des données
3. Fine-tuning supervisé (SFT) avec QLoRA
4. Alignement par préférences (DPO)
5. Déploiement et infrastructure
6. Évaluation et métriques de performance
7. Recommandations stratégiques et roadmap
8. Conclusion
9. Annexes

---

## 1. Introduction

Le Centre Hospitalier Saint-Aurélien (CHSA) fait face à une surcharge récurrente de son service des urgences : aux heures de pointe, le manque d'effectifs infirmiers entraîne des temps d'attente prolongés et un risque accru de sous-identification des cas critiques. La Direction Innovation Médicale a mandaté le développement d'un Proof of Concept (POC) visant à démontrer la faisabilité d'un agent IA de triage médical.

L'agent a pour vocation d'**assister — et non de remplacer** — le personnel soignant dans le triage initial. Il collecte les symptômes du patient via un questionnaire adaptatif, évalue un niveau de priorité (urgence maximale, modérée ou différée) selon les protocoles en vigueur, fournit un bilan structuré, et trace chaque interaction à des fins d'audit médical.

La stratégie technique s'articule en trois phases :

- **Phase 1 (POC)** — Déploiement et spécialisation de Qwen3-1.7B-Base pour valider les hypothèses techniques sur infrastructure légère (GPU T4).
- **Phase 2 (fine-tuning)** — Spécialisation par SFT/LoRA sur corpus médical bilingue, puis alignement sur les pratiques cliniques par DPO.
- **Phase 3 (industrialisation)** — Passage à des modèles 32B+ et intégration dans les systèmes du CHSA, conditionné à la validation clinique de la Phase 1.

Ce rapport couvre l'intégralité de la mission réalisée sur quatre semaines : préparation des données (§2), fine-tuning SFT (§3), alignement DPO (§4), déploiement (§5), évaluation (§6) et recommandations stratégiques (§7).

---

## 2. Préparation et structuration des données

La qualité d'un modèle fine-tuné dépend directement de la qualité de ses données d'entraînement. Cette section décrit l'intégralité du processus : identification des sources, nettoyage, anonymisation, augmentation synthétique au format triage, et versionnement.

### 2.1 Sources de données

Quatre datasets publics hébergés sur Hugging Face ont été retenus pour constituer un corpus médical bilingue FR/EN couvrant différents formats de connaissances médicales.

| Dataset | Langue | Type | Volume brut | Usage | Licence |
|---|---|---|---|---|---|
| MediQAL MCQU (ANR-MALADES) | FR | QCM cliniques | ~17 000 lignes | SFT | À confirmer (académique) |
| FrenchMedMCQA (nthngdy) | FR | QCM médicaux | 1 080 lignes | SFT | Apache 2.0 |
| MedQuAD (keivalya) | EN | Q&A ouvertes | 16 407 lignes | SFT | CC BY 4.0 |
| UltraMedical-Preference (TsinghuaC3I) | EN | Paires préférentielles | 109 353 lignes | SFT + DPO | MIT (à confirmer) |

**Choix de MediQAL MCQU uniquement.** Le sous-ensemble MCQM (réponses multiples) a été écarté : il introduit une ambiguïté dans le signal d'apprentissage, le modèle devant apprendre à produire une combinaison de réponses plutôt qu'une réponse unique. Le sous-ensemble OEQ (réponses libres) constitue une amélioration envisagée pour les itérations futures.

**Stratégie de split.** Les splits d'origine (train/val/test) ont été fusionnés par dataset, puis re-découpés de manière stratifiée par source (clé `dataset_name`) au niveau du dataset SFT consolidé. Cette approche maximise le volume disponible et garantit la représentation équilibrée de chaque source dans chaque split.

### 2.2 Pipeline de nettoyage

Chaque dataset dispose de son propre script de nettoyage. Des fonctions utilitaires partagées (`utils_cleaning.py`) assurent la cohérence entre sources : déduplication (appliquée deux fois — avant et après réduction aux colonnes finales, pour capturer les doublons masqués par les colonnes auxiliaires), normalisation en minuscules, marquage de source (`dataset_name`), et comptage de tokens via le tokenizer Qwen3 (et non une approximation par `len(text.split())`).

Le tableau ci-dessous synthétise les transformations par source :

| Transformation | MediQAL | FrenchMedMCQA | MedQuAD | UltraMedical |
|---|:---:|:---:|:---:|:---:|
| Fusion des splits | ✅ | ✅ | ✅ | ✅ |
| Déduplication (×2) | ✅ | ✅ | ✅ | ✅ |
| Résolution indices → texte | ✅ | ✅ | — | — |
| Fusion cas clinique + question | ✅ | — | — | — |
| Extraction QA depuis format conversationnel | — | — | — | ✅ |
| Retrait questions à négation | ✅ | ✅ | — | — |
| Retrait instructions parasites ("cochez la réponse juste") | ✅ | — | — | — |
| Retrait réponses indicées (regex) | ✅ | — | — | — |
| Normalisation lowercase | ✅ | ✅ | ✅ | ✅ |

**Justification des choix critiques :**

**Fusion cas clinique + question (MediQAL).** MediQAL dissocie dans deux colonnes le contexte clinique du patient et la question posée. Ces deux champs sont concaténés pour former une entrée unifiée. La motivation est directement liée au cas d'usage : en situation de triage, le modèle recevra un contexte patient suivi d'une question — l'entraîner sur des inputs déjà fusionnés favorise l'apprentissage de ce pattern conversationnel.

**Retrait des questions à négation.** Les questions formulées comme "cochez la réponse fausse" ont pour "bonne réponse" dans le dataset une affirmation médicalement incorrecte. Les intégrer au corpus entraînerait le modèle à associer des informations erronées à la position de réponse correcte — risque inacceptable en contexte de triage.

**Retrait des réponses indicées.** Certaines questions MediQAL contiennent des propositions numérotées avec une réponse sous forme de combinaison d'indices (ex. `1+2+3`). Ces paires sont détectées par regex et supprimées : une telle réponse serait inintelligible pour le personnel soignant en situation réelle.

**Extraction QA depuis UltraMedical.** Ce dataset est au format conversationnel préférentiel. La première question utilisateur et la première réponse assistant de chaque conversation `chosen` sont extraites pour constituer les paires SFT. Le reste de la conversation est ignoré à ce stade — il sera utilisé dans sa forme complète pour le DPO.

### 2.3 Schéma de données unifié

À l'issue du nettoyage, chaque dataset produit un fichier Parquet au schéma identique, enrichi de quatre colonnes de métadonnées :

| Colonne | Type | Valeurs |
|---|---|---|
| `question` | str | Texte nettoyé (cas clinique + question pour MediQAL) |
| `answer` | str | Réponse en texte clair |
| `language` | str | `"fr"` ou `"en"` |
| `question_type` | str | `"mcq_single"`, `"open_qa"`, `"conversational"` |
| `confidence_level` | str | `"high"` (MedQuAD), `"medium"` (MediQAL, FrenchMedMCQA), `"low"` (UltraMedical) |
| `dataset_name` | str | Clé de stratification pour le split |
| `token_count_question` | int | Longueur en tokens (tokenizer Qwen3) |
| `token_count_answer` | int | Longueur en tokens (tokenizer Qwen3) |

Le `confidence_level` reflète la fiabilité estimée du signal d'entraînement : `high` pour les sources institutionnelles NIH/NCI (MedQuAD), `medium` pour les référentiels d'examen médical, `low` pour les réponses extraites automatiquement de conversations. Cette gradation permet de pondérer les sources dans les expérimentations futures.

### 2.4 Anonymisation et conformité RGPD

L'outil retenu est **Presidio** (Microsoft), déployé dans un pipeline bilingue FR/EN détectant cinq types d'entités : `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `DATE_TIME`, `LOCATION`. Le choix de Presidio répond à l'exigence de conformité RGPD pour un déploiement hospitalier.

**Audit sur le dataset SFT v1.** Après constitution du premier dataset avec anonymisation active, un audit quantitatif sur les 5 000 échantillons a révélé un taux de faux positifs incompatible avec l'entraînement :

| Balise | Occurrences dans `answer` | Cause principale |
|---|---|---|
| `<PERSON>` | 1 874 | Syndromes éponymes (Cushing, Crohn, Babinski…) |
| `<DATE>` | 1 451 | Références temporelles cliniques (48h, 7 jours, 72 premières heures…) |
| `<LOCATION>` | 729 | Régions anatomiques, noms d'instituts de recherche |
| `<PHONE>` | 16 | Faux positifs marginaux |
| **Total lignes affectées** | **2 340 / 5 000 (46,8%)** | — |

Presidio, conçu pour détecter des données personnelles réelles, interprète le vocabulaire médical courant comme des entités sensibles. Les réponses d'entraînement contenaient des balises parasites en lieu et place de termes cliniques légitimes, dégradant directement le signal d'apprentissage.

**Décision :** les quatre sources étant des datasets publics Hugging Face sans données personnelles réelles, Presidio a été **retiré du pipeline DVC** pour les étapes `generate_sft` et `generate_dpo`. Le module `anonymisation.py` est conservé dans le dépôt comme brique prête à l'emploi pour les futures données patient réelles, sous réserve d'un calibrage spécifique sur le vocabulaire médical francophone.

### 2.5 Constitution du dataset SFT et augmentation au format triage

Le pipeline de constitution (v2) se décompose en trois stages DVC enchaînés.

#### Stage `generate_sft` — Filtre clinique et sampling équilibré

Avant l'échantillonnage, chaque source est filtrée par `filter_clinical_questions()` : seules les questions contenant au moins un mot-clé clinique (symptômes, temporalité, contexte patient) en français ou en anglais, et dépassant 15 tokens, sont conservées. Ce filtre élimine les QCM purement académiques non reformatables en bilan de triage. L'algorithme de sampling équilibré redistribue le surplus des sources insuffisantes (FrenchMedMCQA, trop petit) vers les sources plus volumineuses. Résultat : `sft_dataset.parquet` — 5 000 paires cliniques sans split.

#### Stage `triage_augmentation` — Reformatage au format triage via Mistral

C'est l'étape la plus structurante de la préparation v2. Le corpus de base (QCM médicaux, Q&A encyclopédiques) ne contient aucun exemple au format de triage structuré attendu par le prompt système de l'agent. Un stage dédié reformate chaque paire `(question, answer)` en un bilan de triage via l'API **Mistral Small** (`mistral-small-latest`).

Pour chaque exemple, Mistral reçoit la question et la réponse médicale originale, et produit une réponse structurée au format suivant :

```
Bilan de triage :
- Symptômes relevés : Douleur épigastrique aggravée par les repas, douleur rétrosternale,
  satiété précoce, ballonnements, sensibilité épigastrique légère
- Niveau d'urgence : différée
- Orientation recommandée : Service Gastro-entérologie
- Hypothèses diagnostiques : Gastrite / Reflux gastro-œsophagien (RGO) / Ulcère gastroduodénal
- Action immédiate : Consultation programmée
```

Ce format structure la réponse en cinq champs explicites : symptômes relevés, niveau d'urgence (maximale / modérée / différée), orientation recommandée, hypothèses diagnostiques, et action immédiate. C'est ce format que le modèle fine-tuné devra reproduire en production.

Le choix de Mistral (entreprise française, hébergement UE) est cohérent avec les contraintes RGPD d'un projet hospitalier. Le stage gère les échecs via un mécanisme de retry (`max_retries: 2`) et produit un fichier d'audit `sft_triage_failures.parquet` recensant les exemples pour lesquels le reformatage a échoué, permettant une analyse post-hoc.

#### Stage `split_sft` — Split stratifié après augmentation

Le split est réalisé **après** l'augmentation, de sorte que les exemples reformatés au format triage soient présents dans les trois sous-ensembles :

| Fichier | Proportion | Volume |
|---|:---:|:---:|
| `sft_train.parquet` | 70% | 3 500 lignes |
| `sft_val.parquet` | 20% | 1 000 lignes |
| `sft_test.parquet` | 10% | 500 lignes |

La stratification est réalisée sur `dataset_name` — chaque source est représentée dans les mêmes proportions dans chaque split.

### 2.6 Versionnement et reproductibilité (DVC)

L'intégralité du pipeline est orchestrée par **DVC**. Le `dvc.yaml` définit un DAG de huit stages :

```
clean_mediqal ────────┐
clean_medquad ────────┤
clean_frenchmedmcqa ──┼→ generate_sft → triage_augmentation → split_sft
clean_ultramed ───────┘
clean_ultramed ───────────────────────────────────────────→ generate_dpo
```

DVC calcule un hash MD5 pour chaque entrée et sortie, détectant automatiquement si un stage doit être ré-exécuté suite à une modification de code ou de données. Le fichier `dvc.lock`, versionné dans Git, enregistre l'état exact de chaque exécution et constitue un certificat de reproductibilité : n'importe quel membre de l'équipe peut recréer exactement le même dataset via `dvc repro`. Les données sont synchronisées sur GCS (`gs://p14-medical-data/dvc-store`), dissociant le versionnement des données de celui du code.

---

## 3. Fine-tuning supervisé (SFT) avec QLoRA

### 3.1 Modèle de base

Le modèle retenu est **Qwen3-1.7B-Base** — compact (compatible GPU T4 16 Go VRAM), doté d'une architecture moderne (Grouped Query Attention, RoPE, vocabulaire 151 936 tokens couvrant FR/EN), et d'un chat template natif structuré (`<|im_start|>system/user/assistant`) directement aligné avec le cas d'usage conversationnel de triage. En cas de validation concluante du POC, l'architecture modulaire LoRA mise en place facilite le passage à des modèles plus grands sans refonte majeure.

### 3.2 Configuration QLoRA

La technique **QLoRA** combine quantification 4-bit et adaptateurs LoRA pour permettre l'entraînement sur GPU T4 — impossible en fine-tuning classique (les états d'optimiseur Adam multiplieraient l'empreinte mémoire par 3 à 4).

**Quantification 4-bit NF4 (BitsAndBytes) :**
Le modèle passe de 3,4 Go (FP16) à ~0,85 Go en mémoire. Le format NF4 (NormalFloat 4-bit), optimisé pour les poids de réseaux neuronaux dont la distribution est approximativement normale, minimise l'erreur de quantification. La double quantification (`bnb_4bit_use_double_quant: True`) réduit l'empreinte des métadonnées d'environ 79 Mo supplémentaires. Les calculs forward/backward restent en FP16 (le T4 ne supporte pas BF16).

**Adaptateurs LoRA :**

| Paramètre | Valeur | Justification |
|---|---|---|
| rank `r` | 16 | Facteur 64× moins de paramètres vs matrice pleine pour d=2048 |
| `lora_alpha` | 32 | Scaling effectif alpha/r = 2 — équilibre adaptation/stabilité |
| `lora_dropout` | 0.05 | Régularisation sur dataset de 3 500 exemples |
| Modules ciblés | q, k, v, o, gate, up, down proj | Attention + MLP — meilleure qualité qu'attention seule |

Les sept modules ciblés couvrent l'intégralité des projections linéaires de chaque couche transformer : les quatre projections d'attention (requêtes, clés, valeurs, sortie) et les trois projections MLP (gate, up, down). L'inclusion du MLP, parfois omise dans les premières implémentations LoRA, améliore significativement la qualité sur les tâches de génération, particulièrement pour un modèle compact où chaque couche porte proportionnellement plus de responsabilité.

**Optimisations mémoire complémentaires :**
- Gradient checkpointing activé (`use_reentrant: False`) — réduit la VRAM au prix de ~30% de temps de calcul supplémentaire, indispensable sur T4.
- Optimiseur Paged AdamW 8-bit — états d'optimiseur en 8-bit avec paging CPU pour éviter les erreurs OOM.

### 3.3 Formatage des données et prompt système

Le fine-tuning d'un modèle conversationnel nécessite de formater chaque paire `(question, answer)` selon le chat template du modèle. Un **prompt système** a été rédigé pour ancrer le modèle dans son rôle hospitalier et définir la structure attendue des réponses :

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

Ce prompt est externalisé dans `params.yaml` pour permettre son ajustement sans modification du code. Il est injecté au début de chaque conversation d'entraînement, produisant des séquences de la forme :

```
<|im_start|>system
Tu es un agent de triage médical...<|im_end|>
<|im_start|>user
[question du dataset]<|im_end|>
<|im_start|>assistant
[réponse du dataset — bilan de triage formaté]<|im_end|>
```

**Masquage des labels.** La loss ne doit être calculée que sur les tokens de la réponse, pas sur ceux du prompt (système + question). Les labels correspondant au prompt sont remplacés par la valeur `-100`, qui indique à PyTorch d'ignorer ces positions. Ce masquage est implémenté dans `tokenize_chat()` en calculant la frontière exacte entre prompt et réponse via le tokenizer.

**Correctif EOS critique (v2).** La troncature à `max_length=512` supprimait le token de fin `<|im_end|>` pour les séquences longues. Le modèle apprenait alors sur des exemples sans signal de fin explicite, produisant en inférence un remplissage systématique jusqu'à `max_tokens`. Le correctif v2 force `<|im_end|>` comme dernier token après toute troncature, éliminant ce comportement en production.

### 3.4 Configuration d'entraînement

| Paramètre | Valeur | Justification |
|---|---|---|
| `learning_rate` | 2e-4 | Élevé car poids gelés — pas de risque de catastrophic forgetting |
| `num_train_epochs` | 2 | Eval loss ne s'améliore plus au-delà ; 3 epochs → risque de mémorisation |
| Batch effectif | 32 (batch=1, accumulation=32) | Contrainte VRAM T4 — équivalent mathématique à un vrai batch de 32 |
| `lr_scheduler_type` | cosine | Décroissance douce favorisant les ajustements fins en fin de run |
| `warmup_steps` | 30 | ~14% des steps totaux — stabilisation des gradients en début d'entraînement |
| `eval_steps` / `save_steps` | 50 | ~6-7 évaluations sur l'entraînement — granularité suffisante pour détecter le surapprentissage |
| `load_best_model_at_end` | True | Modèle final = checkpoint avec meilleure eval loss, pas dernier step |

**Reprise après interruption.** L'entraînement sur Google Colab est soumis à des interruptions (timeout, déconnexion). La fonction `train_model()` détecte automatiquement le dernier checkpoint valide et reprend l'entraînement depuis celui-ci — restaurant l'état du modèle, de l'optimiseur, du scheduler et du compteur de steps.

### 3.5 Résultats de l'entraînement SFT

**Run SFT v1** (référence de base, avant corrections) : 220 steps, 2h41 sur GPU T4.

| Métrique | SFT v1 |
|---|---|
| Train loss finale | 1.112 |
| Eval loss finale | 1.189 |
| Contamination Presidio | 46,8% des lignes |
| Comportement en inférence | Remplissage systématique max_tokens, balises `<PERSON>`, `<DATE>` dans les réponses |

**Run SFT v2** (après corrections pipeline) : mêmes hyperparamètres, données corrigées.

| Step | Train loss | Phase |
|---|---|---|
| 10 | 1.903 | — |
| 50 | 0.639 | Adaptation rapide (−66% en 1/4 du run) |
| 110 (fin epoch 1) | 0.582 | Plateau de consolidation |
| 220 (fin epoch 2) | 0.472 | Affinement |

La dynamique en trois phases confirme un apprentissage sain. La loss finale de 0.472 représente une amélioration de **−57,6%** par rapport à la v1 (1.112). Cette réduction ne traduit pas uniquement un meilleur ajustement : en v1, le modèle tentait d'apprendre simultanément le format de triage et les artefacts Presidio — deux objectifs contradictoires maintenant la loss à un plateau élevé. En v2, le signal est cohérent.

---

## 4. Alignement par préférences (DPO)

### 4.1 Principe et lien avec le SFT

Le SFT optimise la vraisemblance des réponses du corpus d'entraînement mais ne distingue pas une réponse médicalement pertinente d'une réponse plausible mais cliniquement sous-optimale. La technique **DPO** (Direct Preference Optimization, Rafailov et al., 2023) reformule l'objectif d'alignement comme un problème de classification sur des paires `(chosen, rejected)`, sans nécessiter de modèle de récompense explicite — réduisant considérablement la complexité par rapport à l'approche RLHF classique.

La loss DPO maximise le log-ratio entre la probabilité attribuée à la réponse choisie et celle attribuée à la réponse rejetée, relative au modèle de référence (SFT gelé). Le paramètre **beta** contrôle l'intensité du rappel vers ce modèle de référence : un beta élevé préserve les connaissances médicales mais limite la discrimination ; un beta faible permet plus de liberté au risque de "désapprendre" les acquis du SFT.

**Prérequis fondamental :** le DPO s'applique sur le **modèle SFT**, non sur le modèle de base. Il affine les préférences d'un modèle qui sait déjà répondre dans le domaine cible — appliquer DPO sur le modèle de base serait sans effet utile.

### 4.2 Chaîne SFT → DPO et traçabilité MLflow

La chaîne d'entraînement repose sur une convention de tags MLflow garantissant la traçabilité et la reproductibilité :

```
Expérience SFT  → run taggé stage="sft",  model_status="champion"
                        ↓ (récupération automatique)
Expérience DPO  → run taggé stage="dpo",  model_status="champion"
```

Le module `train_dpo.py` interroge automatiquement le registre MLflow pour récupérer les adaptateurs LoRA du meilleur run SFT avant de lancer l'alignement. Si aucun run SFT champion n'est trouvé, une exception explicite est levée — forçant l'exécution du SFT avant le DPO. Ce mécanisme garantit que tout nouveau run DPO repart systématiquement du meilleur modèle SFT validé sans intervention manuelle. Les artefacts sont persistés sur `gs://p14-medical-data/mlflow-artifacts/`.

### 4.3 Dataset DPO

Le dataset DPO provient d'**UltraMedical-Preference** — 5 000 triplets `(question, chosen, rejected)` issus de comparaisons automatisées entre réponses générées (non de validations cliniques humaines, d'où `confidence_level: "low"`). C'est la seule source du projet exposant la structure préférentielle requise par DPO.

Split identique au SFT : 3 500 train / 1 000 val / 500 test, stratifié sur `dataset_name`.

Le `DPOTrainer` de la bibliothèque `trl` reçoit les données au format conversationnel structuré (`prompt / chosen / rejected` en listes de messages) et gère lui-même le masquage des labels et l'application du chat template — contrairement au SFT où ces opérations sont gérées manuellement dans `tokenize_chat()`.

### 4.4 Configuration et hyperparamètres

| Paramètre | SFT | DPO | Justification |
|---|---|---|---|
| `learning_rate` | 2e-4 | 5e-6 | DPO affine un modèle déjà spécialisé — LR 40× plus faible pour ne pas déstabiliser les représentations médicales |
| `beta` | — | 0.1 | Régularisation KL conservatrice — valeur standard pour les domaines à fort enjeu |
| `num_train_epochs` | 2 | 2 | Risque de surapprentissage identique |
| Batch effectif | 32 | 32 | Identique |
| `eval_steps` | 50 | 20 | Signal DPO plus bruité — granularité plus fine nécessaire |
| `fp16` | True | True | T4 ne supporte pas BF16 |
| `optim` | paged_adamw_8bit | paged_adamw_8bit | Optimisation mémoire identique |

---

## 5. Déploiement et infrastructure

### 5.1 Architecture générale

Le déploiement s'articule autour de trois couches organisées en pipeline :

**Couche d'inférence — vLLM.** Le moteur vLLM charge le modèle mergé (base + adaptateurs LoRA fusionnés via `merge_and_unload()`, éliminant le surcoût de calcul LoRA à chaque forward pass) et expose une interface de génération asynchrone. Trois mécanismes d'optimisation le distinguent de l'implémentation `model.generate()` native de Hugging Face : le *continuous batching* (traitement dynamique des requêtes sans attendre un batch complet), le *PagedAttention* (gestion du KV cache par pages mémoire, éliminant le gaspillage lié à l'allocation statique), et le *scheduling* asynchrone (découplage réception / traitement GPU). Pour un modèle 1.7B sur GPU unique, ces optimisations réduisent la latence et augmentent le débit en charge concurrente.

**Couche API — FastAPI.** Deux routes sont exposées : `/health` et `/generate`. La validation des entrées est assurée par des schémas Pydantic stricts :

| Paramètre | Bornes | Justification |
|---|---|---|
| `prompt` | 5–4 000 chars | Rejette les requêtes vides ; 4 000 = fenêtre utile après prompt système |
| `max_tokens` | 1–2 048 | Limite la génération pour éviter la consommation excessive de GPU |
| `temperature` | 0.0–2.0 (défaut 0.7) | Compromis diversité/cohérence adapté au médical |

Le pattern *lifespan* de FastAPI gère le chargement et déchargement du moteur vLLM. En cas d'échec au démarrage, l'application reste accessible mais `/health` retourne `503 Service Unavailable` — signalant à l'orchestrateur que le conteneur n'est pas opérationnel sans déclencher de boucle de redémarrage (*dégradation gracieuse*). Un middleware HTTP journalise la latence de bout en bout pour chaque requête `/generate`. Les erreurs internes retournent un message générique au client (pas de traceback exposé), tandis que le détail est loggé côté serveur.

**Couche conteneur — Docker.** Seuls `src/` et `config/` sont inclus dans l'image. Les poids du modèle (~3,4 Go) sont montés en volume au démarrage, découplant le cycle de vie du code de celui du modèle : une nouvelle version de l'API se déploie sans retélécharger le modèle, et inversement. Un `HEALTHCHECK` natif appelle `/health` toutes les 30s (timeout 10s, délai initial 60s pour laisser vLLM charger le modèle en VRAM, 3 tentatives avant de déclarer le conteneur unhealthy).

### 5.2 Pipeline CI/CD (GitHub Actions)

Deux pipelines indépendants coexistent :
- **Pipeline de données** (DVC) — nettoyage, augmentation, génération des datasets. Déclenché manuellement.
- **Pipeline de déploiement** (GitHub Actions) — tests → build Docker → déploiement. Déclenché par chaque commit sur `main`.

Le workflow `.github/workflows/cicd.yml` définit trois jobs séquentiels :

1. **`code-quality-and-tests`** — Linter Ruff + 70 tests pytest. vLLM, qui refuse de s'importer sans GPU CUDA, est mocké via injection dans `sys.modules` avant tout import de l'application — permettant l'exécution complète sur les runners CPU de GitHub Actions.
2. **`build-and-push-docker`** — Construction de l'image (Buildx + cache GitHub Actions) et push vers GHCR, conditionné à la réussite du job 1.
3. **`deploy`** — Connexion SSH à la VM GCP, pull de la nouvelle image, redémarrage du conteneur avec accès GPU.

### 5.3 Stratégie de tests

La suite de 70 tests (exécutés en 2,21 secondes) couvre trois couches :

- **Unitaire** — Validation Pydantic (7 cas valides, 8 cas invalides sur `GenerationRequest`), cohérence des chemins GCS dans `config/paths.py` (préfixe `gs://`, extensions `.parquet`, hiérarchie des dossiers), fonctionnement du logger.
- **Intégration** — Comportement de l'API via `TestClient` : health check (200 moteur chargé / 503 moteur absent), endpoint `/generate` (nominal, 422 sur entrées invalides, gestion des `RuntimeError` et `TimeoutError` sans exposition de traceback), contrat de réponse (`{"response": str}`).
- **Smoke** — Structure du Dockerfile (directives `FROM`, `EXPOSE 8000`, `CMD`, `COPY src/`), existence du `.dockerignore`, présence des steps `pytest` et `docker` dans le workflow CI.

Les tests de performance (latence P95, robustesse sous charge) ne peuvent être exécutés qu'après déploiement sur VM avec GPU — ils font l'objet du benchmark en section 6.7.

---

## 6. Évaluation et métriques de performance

### 6.1 Métriques d'entraînement SFT v2

Run `sft_qwen3-1.7b-base_qlora_r16_fp16_T4` — 2 epochs, 220 steps, 2h41 sur GPU T4.

| Step | Train loss |
|---|---|
| 10 | 1.903 |
| 30 | 0.884 |
| 50 | 0.639 |
| 110 (fin epoch 1) | 0.582 |
| 220 (fin epoch 2) | 0.472 |

La dynamique se décompose en trois phases : adaptation rapide (steps 0–50, −66% de loss), plateau de consolidation (steps 50–110, le modèle assimile la structure de base des réponses), puis affinement (steps 110–220, terminologie médicale spécialisée et cohérence des niveaux d'urgence). L'absence de remontée de la eval loss valide le choix de 2 epochs ; `load_best_model_at_end=True` garantit que le checkpoint retenu est celui qui généralise le mieux.

### 6.2 Métriques d'entraînement DPO v2

Run `dpo_qwen3-1.7b-base_qlora_r16_bf16_T4` — 2 epochs, 220 steps, **8h31** sur GPU T4 (durée ~3× supérieure au SFT : le DPOTrainer calcule les probabilités sur chosen et rejected en comparaison avec le modèle de référence gelé — approximativement 2× plus de calcul par batch).

| Step | Train loss | Eval loss | Eval rewards/margin |
|---|---|---|---|
| 10 | 0.693 (= log 2 : état neutre) | — | — |
| 50 | 0.611 | 0.616 | +0.260 |
| 100 | 0.586 | 0.591 | +0.373 |
| 150 | 0.564 | 0.584 | +0.402 |
| 220 | 0.528 | 0.583 | +0.404 |

La loss DPO démarre à log(2) ≈ 0.693 — valeur théorique d'un modèle ne discriminant pas encore chosen et rejected (probabilité relative 50/50). C'est le comportement attendu et un indicateur sain. La marge (`rewards/margin` = reward_chosen − reward_rejected) croît de +0.260 à +0.404 en évaluation, confirmant l'apprentissage de la discrimination. Le plateau entre les steps 150 et 220 (variation < 0.001) atteste de la convergence sans surapprentissage. L'écart train/eval loss stable à ~0.005 en fin de run confirme la bonne généralisation sur des triplets non vus.

### 6.3 Comparaison v1 → v2

| Métrique | v1 | v2 | Amélioration |
|---|---|---|---|
| SFT train loss finale | 1.112 | 0.472 | **−57,6%** |
| DPO eval rewards/margins | +0.194 | +0.404 | **+108%** |
| DPO eval loss | 0.621 | 0.583 | −6,1% |
| Contamination Presidio | 46,8% des lignes | 0% | **Éliminée** |

Le doublement de la marge DPO (+108%) confirme qu'un meilleur modèle SFT fournit une meilleure base d'alignement : la qualité des données en amont impacte directement la capacité de discrimination du DPO.

### 6.4 Évaluation qualitative

Une évaluation sur des cas cliniques représentatifs a été conduite sur les deux modèles mergés (SFT v2 et DPO v2) via l'endpoint `/generate`, en anglais et en français.

| Critère | SFT v2 (EN) | SFT v2 (FR) | DPO v2 (EN) | DPO v2 (FR) |
|---|---|---|---|---|
| Pertinence médicale | ✅ Bonne | ❌ Faible | ✅ Bonne | ❌ Faible |
| Format triage structuré | ⚠️ Partiel | ❌ Absent | ❌ Absent | ❌ Absent |
| Contamination Presidio | ✅ Aucune | ✅ Aucune | ✅ Aucune | ✅ Aucune |
| Arrêt propre (EOS) | ⚠️ Variable | ❌ Remplit max_tokens | ⚠️ Variable | ❌ Remplit max_tokens |
| Répétitions | ✅ Aucune | ⚠️ Boucles QCM | ✅ Aucune | ⚠️ Boucles QCM |

**Exemple SFT v2 — prompt EN (douleur thoracique, homme 58 ans).** Réponse médicalement pertinente : suspicion d'événement cardiaque aigu, orientation urgences avec monitoring cardiaque et ECG. La structure s'approche du format triage sans le respecter strictement — réponse conversationnelle plutôt que structurée en champs explicites.

**Exemple DPO v2 — prompt EN (glaucome aigu).** Diagnostic correct (glaucome aigu à angle fermé) et explication mécanistique pertinente. Cependant le format est celui d'une réponse académique explicative — aucune structuration de triage. Le DPO a amélioré la qualité médicale au détriment du format.

**Exemple FR (syndrome néphrotique) — SFT et DPO v2.** Les deux modèles retombent en mode QCM : ils génèrent des options fictives (A/B/C/D) et bouclent sur des questions inventées. Ce comportement illustre la limite structurelle du corpus francophone.

**Problèmes résolus entre v1 et v2 :**

| Problème v1 | Statut v2 | Correction |
|---|---|---|
| Balises Presidio dans les réponses | ✅ Résolu | Retrait de l'anonymisation sur corpus publics |
| Répétitions en boucle systématiques | ✅ Résolu | Correctif EOS + `repetition_penalty=1.1` |
| Remplissage systématique max_tokens | ⚠️ Amélioré | Correctif EOS + `stop_token_ids` vLLM |
| Absence totale de format triage | ⚠️ Partiel (EN uniquement) | Augmentation dataset via Mistral |
| Réponses décousues après DPO | ✅ Résolu | Données SFT propres = meilleure base d'alignement |

### 6.5 Limites identifiées

**Déséquilibre linguistique.** Les sources anglophones représentent ~60–65% du corpus SFT. Les sources francophones conservent un fort biais QCM malgré l'augmentation Mistral. Un modèle 1.7B n'a pas la capacité de généraliser le format triage en français avec aussi peu d'exemples dans cette langue — il privilégie le pattern majoritaire (anglais, format triage) et retombe sur le pattern minoritaire (QCM) en français.

**Décalage de distribution SFT/DPO.** UltraMedical-Preference contient des paires au format Q&A académique, non au format triage. L'alignement DPO améliore la qualité médicale mais tire le modèle vers un format explicatif, effaçant partiellement le format structuré appris lors du SFT. Le beta=0.1, pourtant conservateur, n'a pas suffi à compenser cette pression distributionnelle.

**Capacité du modèle 1.7B.** La capacité d'instruction-following est structurellement inférieure à celle des modèles 7B+ : le modèle peine à respecter simultanément le contenu médical et le format de sortie structuré, particulièrement en français. Ce constat est un résultat attendu de POC qui motive directement la roadmap.

### 6.6 Décision de déploiement

Le modèle **SFT v2** a été retenu, pour trois raisons :

1. Il produit des réponses médicalement pertinentes en anglais avec une structure plus proche du format triage que le DPO v2.
2. Le DPO v2, bien que meilleur médicalement, dégrade le format structuré — rédhibitoire pour la lisibilité par le personnel soignant.
3. La correction identifiée (reformatage du dataset DPO au format triage, comme pour le SFT) est documentée en roadmap comme amélioration v3 prioritaire.

### 6.7 Métriques opérationnelles

Benchmark sur 20 requêtes séquentielles — VM GCP, GPU T4, modèle SFT v2 via vLLM, cas cliniques variés (P1 critique, urgence modérée, cas différable) en anglais et en français.

| Paramètre d'inférence | Valeur | Justification |
|---|---|---|
| `max_tokens` | 512 | Suffisant pour un bilan de triage structuré |
| `temperature` | 0.7 | Compromis diversité/cohérence |
| `repetition_penalty` | 1.1 | Correction du comportement de boucle v1 |
| `stop_token_ids` | [151643] (EOS Qwen) | Arrêt propre sur token de fin de séquence |

| Métrique | Valeur |
|---|---|
| Taux de succès | 20/20 (100%) |
| Latence moyenne | 9,18s (± 2,98s) |
| Latence P50 | 10,43s |
| **Latence P95** | **12,66s** |
| Longueur moyenne de réponse | ~455 tokens |
| Débit de génération | ~50 tokens/s |

**Analyse :** Le taux de succès de 100% valide la stabilité de la chaîne complète. La longueur moyenne (455 tokens < limite 512) confirme que le correctif EOS produit des arrêts naturels. Le ratio P95/P50 de 1.21× indique une queue de distribution contrôlée — absence de dégradation pathologique liée au remplissage systématique. La latence P95 de 12,66s est acceptable pour un outil d'aide à la décision de triage (interaction typique de plusieurs minutes) ; elle serait ramenée à 3–5s avec un GPU A10G/A100 en production.

Le benchmark mesure la latence en régime séquentiel — une évaluation en charge (requêtes concurrentes, montée en charge) constitue une étape naturelle avant tout déploiement pilote.

---

## 7. Recommandations stratégiques et roadmap

### 7.1 Limitation opérationnelle restante

Le chemin GCS du modèle est codé en dur dans le workflow GitHub Actions (run ID MLflow spécifique). Lors d'un nouvel entraînement, le workflow doit être mis à jour manuellement. La correction — résolution dynamique via l'API MLflow au moment du déploiement — représente ~10 lignes dans le step CI/CD et garantit que chaque merge sur `main` déploie systématiquement la version validée la plus récente, propriété essentielle pour la traçabilité des versions en production hospitalière.

### 7.2 Checklist go / no-go pour un déploiement pilote

| Critère | Statut POC | Requis pour pilote |
|---|---|---|
| Endpoint d'inférence fonctionnel | ✅ vLLM + FastAPI | + authentification |
| Pipeline CI/CD automatisé | ✅ GitHub Actions | + résolution dynamique GCS |
| Tests automatisés | ✅ 70 tests (unit + intégration + smoke) | + tests de performance P95 < 2s |
| Modèle aligné sur préférences médicales | ✅ DPO sur UltraMedical | + validation par cliniciens CHSA |
| Données sans PII | ✅ corpus publics | + anonymisation données patient réelles |
| Monitoring post-déploiement | ❌ Non implémenté | **Requis** |
| Human-in-the-loop | ❌ Non implémenté | **Requis** |
| Conformité RGPD données patient | ⚠️ Presidio disponible, non calibré | Calibrage + audit CNIL |

### 7.3 Axes d'amélioration pré-production

**Human-in-the-loop.** Le modèle actuel produit un niveau de priorité sans mécanisme d'escalade vers un soignant lorsque sa confiance est faible. En production, les réponses dont le score de confiance est sous un seuil paramétrable doivent être soumises à validation humaine avant transmission — garde-fou principal contre la non-détection d'une urgence critique, cas de défaillance le plus grave dans le contexte du CHSA.

**RAG pour réduire les hallucinations.** Ancrer les réponses sur un corpus médical de référence versionné et validé (protocoles CCMU, référentiels SAMU, guides HAS) réduit la surface d'hallucination et permet de mettre à jour les connaissances médicales sans réentraîner le modèle.

**Monitoring post-déploiement.** Trois indicateurs prioritaires : latence P95 dans le temps (dégradation sous charge), distribution des niveaux de priorité attribués (détection de dérives comportementales), taux d'escalade human-in-the-loop (indicateur synthétique de la confiance globale du système). Prometheus + Grafana ou une solution LLM Observability s'intègre naturellement au middleware FastAPI existant.

**Calibrage Presidio sur données patient réelles.** Le module `anonymisation.py` est prêt à l'emploi mais devra être recalibré sur le vocabulaire médical francophone pour réduire les faux positifs documentés en §2.4. Un audit CNIL de l'architecture de stockage (chiffrement au repos et en transit, durée de conservation bornée, droit à l'oubli) sera nécessaire avant tout déploiement réel.

**Dataset DPO au format triage (v3).** Reformater le dataset UltraMedical-Preference au format triage via Mistral — comme réalisé pour le SFT — permettrait au DPO d'aligner les préférences sans dégrader la structure des réponses, corrigeant le décalage de distribution identifié en §6.5.

### 7.4 Roadmap Phase 3

| Horizon | Action | Justification |
|---|---|---|
| **0–3 mois** | Résolution dynamique GCS en CI/CD | Prérequis pour itérations sans intervention manuelle |
| **0–3 mois** | Validation clinique par soignants CHSA | Évaluer l'acceptabilité avant d'investir dans un modèle plus grand |
| **3–6 mois** | Human-in-the-loop + monitoring | Prérequis sécurité pour tout déploiement pilote |
| **3–6 mois** | Migration vers Qwen3-8B ou LLaMA-3-8B | Meilleur raisonnement clinique — GPU A100 requis (40 Go VRAM) |
| **6–12 mois** | Intégration RAG sur corpus médical CHSA | Réduction hallucinations, mise à jour sans réentraînement |
| **6–12 mois** | Passage à un modèle 32B+ | Performances cliniques significativement supérieures — multi-GPU requis |
| **12 mois+** | Entraînement sur données patient réelles anonymisées | Spécialisation sur les cas effectivement rencontrés aux urgences du CHSA |

L'architecture vLLM + FastAPI produite dans ce POC absorbe cette montée en charge sans refonte : vLLM supporte nativement le tensor parallelism multi-GPU via `--tensor-parallel-size`, rendant la transition architecturalement transparente.

---

## 8. Conclusion

Ce POC démontre la faisabilité technique d'un agent IA de triage médical sur la pile Qwen3-1.7B + QLoRA + DPO, avec un pipeline de données versionné (DVC), un endpoint d'inférence optimisé (vLLM + FastAPI) et un pipeline CI/CD automatisé (GitHub Actions). Les cinq livrables du cahier des charges ont été produits en quatre semaines.

La démarche itérative v1 → v2 constitue un résultat structurant en soi : l'audit qualité a révélé que l'anonymisation Presidio produisait des faux positifs massifs sur le vocabulaire médical, dégradant directement le signal d'apprentissage. La correction systématique — retrait de Presidio sur les corpus publics, filtre clinique, augmentation synthétique au format triage structuré via Mistral, correctif EOS — a produit une amélioration de 57,6% de la loss SFT finale et un doublement de la marge DPO. La capacité à diagnostiquer une dégradation de la qualité des données et à y remédier de manière systématique et traçable constitue en elle-même une validation de la maturité du pipeline.

Les limites sont claires et documentées : le modèle 1.7B fonctionne correctement en anglais mais reste insuffisant en français, et le décalage de distribution entre les datasets SFT et DPO dégrade le format structuré après alignement. Ces constats tracent précisément la roadmap : validation clinique par les soignants du CHSA, montée vers un modèle 8B puis 32B+, intégration RAG, et human-in-the-loop comme prérequis de sécurité non négociable avant tout déploiement hospitalier réel.

---

## 9. Annexes

### Annexe A — Architecture du dépôt GitHub

```
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
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI (lifespan, middleware, routes)
│   │   ├── schemas.py          # Schémas Pydantic
│   │   └── services/inference.py  # VLLMEngine (AsyncLLMEngine)
│   ├── processing/
│   │   ├── mediqal_cleaning.py
│   │   ├── frenchmedmcqa_cleaning.py
│   │   ├── medquad_cleaning.py
│   │   ├── ultramed_cleaning.py
│   │   ├── utils_cleaning.py         # Helpers + filter_clinical_questions()
│   │   ├── anonymisation.py          # Brique Presidio (usage futur)
│   │   ├── sft_dataset/
│   │   │   ├── generate_sft_dataset.py
│   │   │   ├── triage_augmentation.py   # Reformatage Mistral → format triage
│   │   │   └── split_sft_dataset.py
│   │   └── dpo_dataset/
│   │       └── generate_dpo_dataset.py
│   └── training/
│       ├── train_sft.py
│       ├── train_dpo.py
│       ├── generate_model_for_deployment.py  # Merge LoRA + push GCS via MLflow
│       └── utils_training.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── dvc.yaml                    # Pipeline DVC (8 stages)
├── params.yaml                 # Tous les hyperparamètres
├── Dockerfile
└── .github/workflows/cicd.yml  # Pipeline CI/CD (3 jobs)
```

### Annexe B — Hyperparamètres complets SFT et DPO

| Paramètre | SFT | DPO |
|---|---|---|
| `learning_rate` | 2e-4 | 5e-6 |
| `num_train_epochs` | 2 | 2 |
| `beta` | — | 0.1 |
| `per_device_train_batch_size` | 1 | 1 |
| `gradient_accumulation_steps` | 32 | 32 |
| `warmup_steps` | 30 | 30 |
| `lr_scheduler_type` | cosine | cosine |
| `optim` | paged_adamw_8bit | paged_adamw_8bit |
| `fp16` | True | True |
| `gradient_checkpointing` | True | True |
| `eval_steps` | 50 | 20 |
| `load_best_model_at_end` | True | True |
| LoRA rank `r` | 16 | 16 |
| `lora_alpha` | 32 | 32 |
| `lora_dropout` | 0.05 | 0.05 |
| Modules ciblés | q,k,v,o,gate,up,down proj | q,k,v,o,gate,up,down proj |
| Quantification | 4-bit NF4 | 4-bit NF4 |

### Annexe C — Glossaire

| Terme | Définition |
|---|---|
| **SFT** | Supervised Fine-Tuning — spécialisation supervisée sur des paires (question, réponse) |
| **DPO** | Direct Preference Optimization — alignement sur des paires (chosen/rejected) sans modèle de récompense explicite |
| **LoRA** | Low-Rank Adaptation — matrices basse-rang injectées dans les couches d'attention et MLP |
| **QLoRA** | LoRA sur modèle quantifié 4-bit — réduit l'empreinte VRAM de ~75% vs FP32 |
| **vLLM** | Moteur d'inférence haute performance — continuous batching + PagedAttention |
| **DVC** | Data Version Control — versionnement des données et pipelines ML |
| **MLflow** | Plateforme de tracking des expériences ML — métriques, hyperparamètres, artefacts |
| **RAG** | Retrieval-Augmented Generation — ancrage des réponses sur un corpus de référence |
| **GHCR** | GitHub Container Registry — registre d'images Docker intégré à GitHub |
| **CCMU** | Classification Clinique des Malades aux Urgences — échelle française de triage en 5 niveaux |
| **PPL** | Perplexité — mesure de la surprise du modèle face à un texte de référence (plus basse = meilleure) |