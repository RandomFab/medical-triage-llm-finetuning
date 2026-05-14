"""Test 5 — Health check endpoint (/health)."""


class TestHealthWithEngine:
    """Quand le moteur est chargé → l'API est opérationnelle."""

    def test_health_returns_200(self, client_with_engine):
        response = client_with_engine.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client_with_engine):
        data = client_with_engine.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_status"] == "loaded"
        assert "message" in data


class TestHealthWithoutEngine:
    """Quand le moteur n'a pas pu se charger → l'API signale qu'elle n'est pas prête."""

    def test_health_returns_503(self, client_without_engine):
        response = client_without_engine.get("/health")
        assert response.status_code == 503

    def test_health_503_has_detail(self, client_without_engine):
        data = client_without_engine.get("/health").json()
        assert "detail" in data