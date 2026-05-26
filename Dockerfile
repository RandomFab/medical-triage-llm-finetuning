# Image officielle vLLM — inclut Python 3.11, CUDA, PyTorch et vLLM
FROM vllm/vllm-openai:latest

WORKDIR /app

ENV PYTHONPATH=/app
ENV LANG=C.UTF-8

# Seules les dépendances API manquantes — vLLM est déjà installé
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv

COPY src/ ./src/
COPY config/ ./config/
COPY params.yaml /app/params.yaml

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT []

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Commandes pour l'utiliser
#1. Construire l'image Docker (à faire à la racine du projet) :
# 'docker build -t medical-qwen-api .'
# 2. Lancer le conteneur (en montant le dossier de modèles et en activant le GPU) :
# Pour que vLLM puisse faire son travail, vous devez lancer le conteneur avec l'option --gpus all (nécessite le NVIDIA Container Toolkit sur la machine hôte) et monter votre dossier models pour que le code puisse accéder aux poids vLLM (GCS_MERGED_MODEL_PATH).
#  'docker run --gpus all -p 8000:8000 -v "$(pwd)/models:/app/models" medical-qwen-api'
# L'API sera alors accessible sur http://localhost:8000/docs (Swagger UI fourni nativement par FastAPI permettant de tester la route /generate).