"""Endpoint /generate (cas nominal, invalide, erreur, contrat)."""

import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, ANY

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# /generate avec moteur mocké et input valide
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGenerateNominal:

    def test_returns_200(self, client_with_engine):
        """Vérifie que l'endpoint retourne un code HTTP 200 pour une requête nominale valide."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Quels sont les symptômes de la méningite ?",
                "max_tokens": 256,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200

    def test_returns_non_empty_response(self, client_with_engine):
        """Vérifie que la réponse formatée renvoyée contient bien la clé "response" avec du texte."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Diagnostic différentiel douleur thoracique aiguë",
            },
        )
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_works_with_defaults_only(self, client_with_engine):
        """Seul le prompt est obligatoire — max_tokens et temperature ont des défauts."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Traitement de première intention pour l'hypertension artérielle",
            },
        )
        assert response.status_code == 200

    def test_engine_called_with_correct_params(self, client_with_engine, mock_engine):
        """Vérifie que le moteur reçoit bien les paramètres transmis par l'utilisateur (avec prompt formaté)."""
        client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question médicale test",
                "max_tokens": 100,
                "temperature": 0.2,
            },
        )
        mock_engine.generate.assert_called_once_with(
            prompt=ANY,
            max_tokens=100,
            temperature=0.2,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# /generate avec inputs invalides → 422
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGenerateInvalidInput:

    def test_missing_prompt_returns_422(self, client_with_engine):
        """Vérifie de renvoyer une erreur 422 si le champ 'prompt', qui est obligatoire, est manquant."""
        response = client_with_engine.post("/generate", json={})
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client_with_engine):
        """Vérifie que l'envoi d'un corps de requête textuel vide entraîne une erreur Pydantic (422)."""
        response = client_with_engine.post(
            "/generate",
            content="",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_prompt_too_short_returns_422(self, client_with_engine):
        """Vérifie qu'un prompt ne respectant pas la longueur minimum échoue avec une 422."""
        response = client_with_engine.post("/generate", json={"prompt": "Hi"})
        assert response.status_code == 422

    def test_prompt_too_long_returns_422(self, client_with_engine):
        """Vérifie qu'un prompt dépassant la limite de caractères autorisée échoue avec une 422."""
        response = client_with_engine.post("/generate", json={"prompt": "x" * 4001})
        assert response.status_code == 422

    def test_temperature_out_of_range_returns_422(self, client_with_engine):
        """Vérifie qu'une température définie au-delà des bornes valides lève une erreur 422."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question médicale valide",
                "temperature": 5.0,
            },
        )
        assert response.status_code == 422

    def test_max_tokens_negative_returns_422(self, client_with_engine):
        """Vérifie qu'un nombre négatif de tokens n'est pas permis (422)."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question médicale valide",
                "max_tokens": -1,
            },
        )
        assert response.status_code == 422

    def test_max_tokens_above_limit_returns_422(self, client_with_engine):
        """Vérifie qu'un nombre trop élevé de tokens max (hors limite) lève une erreur 422."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question médicale valide",
                "max_tokens": 9999,
            },
        )
        assert response.status_code == 422


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# /generate avec gestion des erreurs (moteur absent / exception)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGenerateErrorHandling:

    def test_no_engine_returns_500(self, client_without_engine):
        """Si le moteur n'a pas pu se charger au startup, l'API retourne 500."""
        response = client_without_engine.post(
            "/generate",
            json={
                "prompt": "Question qui ne devrait pas atteindre le moteur",
            },
        )
        assert response.status_code == 500

    def test_engine_exception_returns_500(self, client_with_engine, mock_engine):
        """Si le moteur lève une exception (ex: GPU OOM), l'API retourne 500 proprement."""
        mock_engine.generate = AsyncMock(side_effect=RuntimeError("CUDA out of memory"))
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question qui provoque un crash moteur",
            },
        )
        assert response.status_code == 500
        assert "CUDA out of memory" in response.json()["detail"]

    def test_engine_timeout_returns_500(self, client_with_engine, mock_engine):
        """Simule un timeout du moteur."""
        mock_engine.generate = AsyncMock(
            side_effect=TimeoutError("Inference timed out after 30s")
        )
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question très complexe qui timeout",
            },
        )
        assert response.status_code == 500

    def test_error_response_never_leaks_stacktrace(
        self, client_with_engine, mock_engine
    ):
        """En contexte médical, les erreurs ne doivent jamais exposer de détails internes
        au-delà du message d'erreur contrôlé."""
        mock_engine.generate = AsyncMock(
            side_effect=ValueError("Unexpected tensor shape [2, 3, 512]")
        )
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Question de test sécurité",
            },
        )
        body = response.text
        # Pas de traceback Python dans la réponse
        assert "Traceback" not in body
        assert "File " not in body

    def test_global_exception_handler_returns_500_json(self):
        """Test direct du gestionnaire d'erreurs global (indépendant des endpoints).

        Ce handler est un filet de sécurité pour les exceptions non catchées —
        on vérifie qu'il retourne bien un JSON structuré avec un code 500.
        """
        from src.api.main import global_exception_handler

        async def _run():
            mock_request = MagicMock()
            return await global_exception_handler(
                mock_request, RuntimeError("Erreur inattendue")
            )

        response = asyncio.run(_run())
        assert response.status_code == 500
        body = json.loads(response.body)
        assert "message" in body


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Contrat de réponse API (schéma exact)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestResponseContract:

    def test_generate_response_has_exact_fields(self, client_with_engine):
        """Le contrat API : exactement un champ 'response' de type str.
        Si demain on ajoute un champ (latence, confiance, tokens_used),
        ce test oblige à mettre à jour le schéma explicitement."""
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Quels sont les signes d'alerte d'un AVC ?",
            },
        )
        data = response.json()
        assert set(data.keys()) == {"response"}
        assert isinstance(data["response"], str)

    def test_422_error_has_detail_field(self, client_with_engine):
        """Vérifie que la réponse renvoyée par le serveur indique bien JSON comme Content-Type."""
        response = client_with_engine.post("/generate", json={"prompt": "Hi"})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_content_type_is_json(self, client_with_engine):
        response = client_with_engine.post(
            "/generate",
            json={
                "prompt": "Test du content-type de la réponse",
            },
        )
        assert response.headers["content-type"] == "application/json"
