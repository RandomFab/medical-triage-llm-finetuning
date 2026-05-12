from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from contextlib import asynccontextmanager
from config.paths import ROOT_MODEL_DIR, DPO_TRAIN_DATASET_PATH, DPO_VAL_DATASET_PATH,GCS_MERGED_MODEL_PATH
from config.logger import logger
from .services.inference import VLLMEngine

# Initialisation du moteur
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (startup/shutdown)."""
    global engine
    
    logger.info("🚀 Démarrage de l'application...")
    logger.info("🤖 Préchargement du modèle vLLM...")
    try:
        engine = VLLMEngine(model_path=GCS_MERGED_MODEL_PATH)
        logger.info("✅ Modèle vLLM chargé et prêt pour l'inférence")
    except Exception as e:
        logger.error(f"⚠️ Erreur lors du chargement du modèle vLLM: {str(e)}")
    
    yield
    
    logger.info("🛑 Arrêt de l'application...")
    engine = None

app = FastAPI(
    title="API Médicale Qwen fine-tuné",
    description="API d'inférence via vLLM pour le modèle médical fine-tuné",
    version="1.0.0",
    lifespan=lifespan
)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    response: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    if not engine:
        raise HTTPException(status_code=500, detail="Moteur vLLM non initialisé")
    
    try:
        output_text = await engine.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return GenerationResponse(response=output_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API opérationnelle"}