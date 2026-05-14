"""Test 4 — Vérification du logger centralisé (config/logger.py)."""
import logging
from config.logger import logger


class TestLogger:

    def test_logger_is_not_none(self):
        assert logger is not None

    def test_logger_is_logging_instance(self):
        """Le logger doit être une instance standard de logging.Logger."""
        assert isinstance(logger, logging.Logger)

    def test_logger_has_a_name(self):
        """Un logger sans nom utilise le root logger, ce qui pollue les logs d'autres modules."""
        assert logger.name != "root"

    def test_logger_can_log_without_error(self):
        """Vérifie que les 3 niveaux utilisés dans l'API ne lèvent pas d'exception."""
        logger.debug("Test unitaire — debug")
        logger.info("Test unitaire — info")
        logger.error("Test unitaire — error")