from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from contextlib import asynccontextmanager
from config.paths import GCS_MERGED_MODEL_PATH
from config.logger import logger
from .services.inference import VLLMEngine
from .schemas import GenerationRequest, GenerationResponse

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
# 1. Sécurité : Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # À restreindre en production ! (ex: ["https://mon-app.fr"])
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 2. Observabilité : Middleware pour mesurer les performances
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Ne pas logger les healthchecks pour éviter le bruit
    if request.url.path != "/health":
        logger.info(
            f"Requête: {request.method} {request.url.path} "
            f"| Status: {response.status_code} "
            f"| Latence: {process_time:.4f}s"
        )
    return response

# 1. Robustesse : Gestion globale des erreurs non prédites
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur critique inattendue : {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Une erreur interne est survenue. Veuillez réessayer plus tard."},
    )


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
    # Vérification avancée : le moteur d'inférence est-il bien en ligne ?
    model_status = "loaded" if engine else "not_loaded"
    
    if not engine:
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
        
    return {
        "status": "ok", 
        "message": "API opérationnelle",
        "model_status": model_status
    }