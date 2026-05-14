""" Middleware de logging des requêtes."""
from unittest.mock import patch


class TestRequestLoggingMiddleware:

    def test_generate_request_is_logged(self, client_with_engine):
        """Une requête sur /generate DOIT être tracée dans les logs (latence, status)."""
        with patch("src.api.main.logger") as mock_logger:
            client_with_engine.post("/generate", json={
                "prompt": "Test de traçabilité du middleware",
            })

            # Vérifie qu'au moins un appel logger.info mentionne /generate
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("/generate" in call for call in info_calls), (
                f"Le middleware n'a pas loggé la requête /generate. Appels : {info_calls}"
            )

    def test_health_request_is_not_logged(self, client_with_engine):
        """Une requête sur /health ne DOIT PAS être loggée par le middleware.

        Pourquoi : en production, le health check est appelé toutes les N secondes
        par l'orchestrateur. Logger chaque ping noierait les vrais logs d'inférence.
        """
        with patch("src.api.main.logger") as mock_logger:
            client_with_engine.get("/health")

            # Aucun appel logger.info ne doit mentionner /health
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert not any("/health" in call for call in info_calls), (
                f"Le middleware a loggé un health check — il ne devrait pas. Appels : {info_calls}"
            )

    def test_log_contains_latency(self, client_with_engine):
        """Le log doit inclure la latence pour le monitoring de performance."""
        with patch("src.api.main.logger") as mock_logger:
            client_with_engine.post("/generate", json={
                "prompt": "Test de mesure de latence",
            })

            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            # Le middleware log "Latence: X.XXXXs"
            assert any("Latence" in call or "latence" in call.lower() for call in info_calls)

    def test_log_contains_status_code(self, client_with_engine):
        """Le log doit inclure le code HTTP pour détecter les erreurs en production."""
        with patch("src.api.main.logger") as mock_logger:
            client_with_engine.post("/generate", json={
                "prompt": "Test de status code dans le log",
            })

            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("200" in call for call in info_calls)