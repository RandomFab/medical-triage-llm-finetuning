from langdetect import detect, LangDetectException
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- Configuration multilingue (FR + EN) ---
# Presidio supporte plusieurs modèles simultanément : on les déclare tous dans "models".
# Prérequis :
#   python -m spacy download fr_core_news_md
#   python -m spacy download en_core_web_md
#   pip install langdetect
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_md"},
    ],
        "ner_model_configuration": {
        "labels_to_ignore": ["CARDINAL", "QUANTITY", "MISC", "ORDINAL"]
        }
}
provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
nlp_engine = provider.create_engine()

# Initialisation une seule fois (pas à chaque appel)
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])
anonymizer = AnonymizerEngine()

ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "LOCATION"]

OPERATORS = {
    "PERSON":        OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "DATE_TIME":     OperatorConfig("replace", {"new_value": "<DATE>"}),
    "LOCATION":      OperatorConfig("replace", {"new_value": "<LOCATION>"}),
}

# Langues supportées par notre moteur NLP
_SUPPORTED_LANGUAGES = {"fr", "en"}


def _detect_language(text: str) -> str:
    """Détecte la langue du texte. Retourne 'en' par défaut si incertain."""
    try:
        lang = detect(text)
        return lang if lang in _SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"


def anonymize_text(text: str, language: str | None = None) -> str:
    """Anonymise un texte en remplaçant les entités sensibles par des tags.

    Si `language` n'est pas fourni, la langue est détectée automatiquement.
    Langues supportées : "fr", "en". Toute autre langue tombe en fallback "en".
    """
    if not isinstance(text, str) or not text.strip():
        return text

    lang = language if language in _SUPPORTED_LANGUAGES else _detect_language(text)

    results = analyzer.analyze(text=text, language=lang, entities=ENTITIES)
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=OPERATORS,
    )
    return anonymized.text
