"""Test 1 — Validation des schémas Pydantic (GenerationRequest / GenerationResponse)."""
import pytest
from pydantic import ValidationError
from src.api.schemas import GenerationRequest, GenerationResponse


# ──────────────────────────────────────────────────
# GenerationRequest — cas valides
# ──────────────────────────────────────────────────

class TestGenerationRequestValid:

    def test_request_with_defaults(self):
        """Un prompt seul doit fonctionner avec les valeurs par défaut."""
        req = GenerationRequest(prompt="Quels sont les symptômes de la grippe ?")
        assert req.prompt == "Quels sont les symptômes de la grippe ?"
        assert req.max_tokens == 512
        assert req.temperature == 0.7

    def test_request_with_custom_params(self):
        req = GenerationRequest(
            prompt="Diagnostic différentiel douleur thoracique aiguë",
            max_tokens=1024,
            temperature=0.3,
        )
        assert req.max_tokens == 1024
        assert req.temperature == 0.3

    def test_temperature_zero_deterministic(self):
        """temperature=0.0 est valide — mode déterministe, utile pour les tests de reproductibilité."""
        req = GenerationRequest(prompt="Test déterministe", temperature=0.0)
        assert req.temperature == 0.0

    def test_temperature_max_allowed(self):
        req = GenerationRequest(prompt="Test créativité maximale", temperature=2.0)
        assert req.temperature == 2.0

    def test_max_tokens_boundaries(self):
        """Teste les bornes min et max de max_tokens."""
        req_min = GenerationRequest(prompt="Test tokens min", max_tokens=1)
        assert req_min.max_tokens == 1
        req_max = GenerationRequest(prompt="Test tokens max", max_tokens=2048)
        assert req_max.max_tokens == 2048

    def test_prompt_at_min_length(self):
        """5 caractères = exactement la limite min_length."""
        req = GenerationRequest(prompt="Abcde")
        assert len(req.prompt) == 5

    def test_prompt_at_max_length(self):
        """4000 caractères = exactement la limite max_length."""
        req = GenerationRequest(prompt="x" * 4000)
        assert len(req.prompt) == 4000


# ──────────────────────────────────────────────────
# GenerationRequest — cas invalides (doivent lever ValidationError)
# ──────────────────────────────────────────────────

class TestGenerationRequestInvalid:

    def test_prompt_missing(self):
        with pytest.raises(ValidationError):
            GenerationRequest()

    def test_prompt_too_short(self):
        """'Hi' = 2 caractères, en dessous du min_length=5."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Hi")

    def test_prompt_too_long(self):
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="x" * 4001)

    def test_max_tokens_zero(self):
        """ge=1 → 0 est rejeté."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test valide", max_tokens=0)

    def test_max_tokens_negative(self):
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test valide", max_tokens=-10)

    def test_max_tokens_above_limit(self):
        """le=2048 → 9999 est rejeté."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test valide", max_tokens=9999)

    def test_temperature_negative(self):
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test valide", temperature=-0.1)

    def test_temperature_above_limit(self):
        """le=2.0 → 2.1 est rejeté."""
        with pytest.raises(ValidationError):
            GenerationRequest(prompt="Test valide", temperature=2.1)


# ──────────────────────────────────────────────────
# GenerationResponse
# ──────────────────────────────────────────────────

class TestGenerationResponse:

    def test_valid_response(self):
        resp = GenerationResponse(response="Urgence modérée — consulter sous 24h.")
        assert resp.response == "Urgence modérée — consulter sous 24h."

    def test_empty_response_is_valid(self):
        """Une réponse vide est techniquement valide (le modèle peut ne rien générer)."""
        resp = GenerationResponse(response="")
        assert resp.response == ""

    def test_response_serialization(self):
        """Vérifie que la sérialisation JSON produit le bon format."""
        resp = GenerationResponse(response="Test de sérialisation")
        data = resp.model_dump()
        assert data == {"response": "Test de sérialisation"}