from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    # Validation stricte des entrées pour éviter les inputs aberrants / l'explosion mémoire
    prompt: str = Field(..., min_length=5, max_length=4000, description="Le prompt utilisateur (max 4000 caractères)")
    max_tokens: int = Field(512, ge=1, le=2048, description="Le nombre max de tokens à générer")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="La créativité du modèle (0.0 = déterministe, 2.0 = aléatoire)")

class GenerationResponse(BaseModel):
    response: str
