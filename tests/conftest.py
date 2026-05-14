import sys
from pathlib import Path

# Ajoute la racine du projet au PYTHONPATH pour que les imports
# comme `from config.paths import ...` et `from src.api.main import ...`
# fonctionnent dans l'environnement de test CI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))