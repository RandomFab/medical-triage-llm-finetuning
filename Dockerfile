# Utilisation d'une image Python officielle (Python 3.10 est recommandé pour vLLM)
FROM python:3.10-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Ajout explicite du chemin principal au PYTHONPATH pour rendre les imports robustes
ENV PYTHONPATH=/app

# Installation des paquets systèmes requis
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier de configuration des dépendances
COPY pyproject.toml ./

# Installation des dépendances (on installe explicitement fastapi, uvicorn, vllm, etc.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn vllm pydantic python-dotenv

# Copier les dossiers nécessaires pour faire tourner l'API
COPY src/ ./src/
COPY config/ ./config/

# EXPLICATION : On ne copie PAS le dossier 'models/' directement dans l'image car il est trop lourd.
# Il est préférable de le monter en tant que volume lors du 'docker run'.

# Informe Docker de l'état de santé du conteneur en appelant l'endpoint /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Exposer le port de l'API
EXPOSE 8000

# Commande pour démarrer l'application avec Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Commandes pour l'utiliser
#1. Construire l'image Docker (à faire à la racine du projet) :
# 'docker build -t medical-qwen-api .'
# 2. Lancer le conteneur (en montant le dossier de modèles et en activant le GPU) :
# Pour que vLLM puisse faire son travail, vous devez lancer le conteneur avec l'option --gpus all (nécessite le NVIDIA Container Toolkit sur la machine hôte) et monter votre dossier models pour que le code puisse accéder aux poids vLLM (GCS_MERGED_MODEL_PATH).
#  'docker run --gpus all -p 8000:8000 -v "$(pwd)/models:/app/models" medical-qwen-api'
# L'API sera alors accessible sur http://localhost:8000/docs (Swagger UI fourni nativement par FastAPI permettant de tester la route /generate).