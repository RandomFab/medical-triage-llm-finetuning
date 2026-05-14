"""Fixtures partagées pour les tests d'intégration API.

Le mock de vLLM au niveau sys.modules est nécessaire car :
1. Les runners CI (GitHub Actions) n'ont pas de GPU
2. vLLM refuse de s'importer sans CUDA
3. On veut tester la logique API indépendamment du moteur d'inférence
"""
import sys
from unittest.mock import MagicMock, AsyncMock

import pytest

# ──────────────────────────────────────────────────────────
# Mock de vLLM AVANT tout import de src.api
# L'ordre est critique : si src.api.main est importé avant
# ce mock, Python essaiera de charger vllm → ImportError.
# ──────────────────────────────────────────────────────────
_vllm_mock = MagicMock()
sys.modules["vllm"] = _vllm_mock
sys.modules["vllm.engine"] = _vllm_mock.engine
sys.modules["vllm.engine.arg_utils"] = _vllm_mock.engine.arg_utils
sys.modules["vllm.engine.async_llm_engine"] = _vllm_mock.engine.async_llm_engine
sys.modules["vllm.sampling_params"] = _vllm_mock.sampling_params

# Maintenant on peut importer l'app en toute sécurité
from fastapi.testclient import TestClient # noqa: E402
from src.api.main import app # noqa: E402
import src.api.main as main_module # noqa: E402


@pytest.fixture
def mock_engine():
    """Crée un faux moteur vLLM qui retourne une réponse médicale fixe.

    AsyncMock est indispensable car engine.generate() est une coroutine
    (async def) — un MagicMock standard ne serait pas awaitable.
    """
    engine = MagicMock()
    engine.generate = AsyncMock(
        return_value="Le patient présente des symptômes compatibles avec une urgence modérée. "
        "Recommandation : consultation dans les 24 heures."
    )
    return engine


@pytest.fixture
def client_with_engine(mock_engine):
    """TestClient avec un moteur chargé — simule le fonctionnement normal.

    On ne passe PAS par le context manager (with TestClient(app) as c:)
    car cela déclencherait le lifespan, qui essaierait de charger le vrai
    modèle vLLM. On injecte le mock_engine directement dans le module.
    """
    main_module.engine = mock_engine
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    main_module.engine = None


@pytest.fixture
def client_without_engine():
    """TestClient SANS moteur — simule un échec de chargement au démarrage."""
    main_module.engine = None
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    main_module.engine = None