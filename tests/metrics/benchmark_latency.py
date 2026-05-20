"""
Benchmark de latence P95 — Endpoint /generate
===============================================
Script à exécuter depuis la VM GCP ou depuis n'importe quelle machine
ayant accès à l'endpoint.

Usage :
    python benchmark_latency.py
    python benchmark_latency.py --url http://localhost:8000
    python benchmark_latency.py --url http://35.214.207.229:8000 --runs 20

Ce script :
1. Vérifie que l'endpoint est opérationnel (/health)
2. Envoie N requêtes séquentielles sur /generate avec des cas cliniques variés
3. Calcule P50, P95, P99, moyenne, min, max
4. Affiche un résumé prêt à copier-coller dans la section 6.7 du rapport
"""

import argparse
import time
import statistics
import sys

import requests

# ── Cas cliniques de test ──────────────────────────────────────────────
# Mélange de complexité et de langues pour refléter un usage réaliste

TEST_PROMPTS = [
    # EN — Urgence cardiaque (cas classique de triage P1)
    {
        "prompt": "A 58-year-old male presents with sudden onset crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath. He has a history of hypertension and type 2 diabetes. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Pédiatrie (cas modéré)
    {
        "prompt": "A 3-year-old child is brought in by parents with a fever of 39.2°C for the past 2 days, mild cough, runny nose, and decreased appetite. The child is alert and playful. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Ophtalmologie (cas aigu)
    {
        "prompt": "A 72-year-old woman presents with sudden painless vision loss in her right eye that started 30 minutes ago. She describes it as a curtain coming down over her vision. She has a history of atrial fibrillation. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Cas psychiatrique
    {
        "prompt": "A 25-year-old male presents with severe agitation, paranoid ideation, and auditory hallucinations. He has not slept for 3 days and reports stopping his medication 2 weeks ago. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Traumatologie
    {
        "prompt": "A 40-year-old construction worker fell from a 3-meter ladder. He is conscious but reports severe lower back pain and tingling in both legs. He cannot move his toes. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # FR — Douleur abdominale (test limite linguistique)
    {
        "prompt": "Une femme de 35 ans se présente avec une douleur abdominale intense dans la fosse iliaque droite depuis 6 heures, accompagnée de nausées et d'une fièvre à 38.5°C. Quel est votre bilan de triage ?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Allergologie (cas urgent)
    {
        "prompt": "A 28-year-old woman presents 15 minutes after eating shellfish with lip swelling, urticaria spreading across her trunk, and difficulty breathing. She has an EpiPen but has not used it. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Cas simple / différable
    {
        "prompt": "A 22-year-old male presents with a sore throat, mild fever of 37.8°C, and body aches for 2 days. No difficulty breathing or swallowing. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Neurologie (AVC)
    {
        "prompt": "A 65-year-old woman presents with sudden right-sided facial droop, slurred speech, and weakness in her right arm. Symptoms started approximately 45 minutes ago. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    # EN — Intoxication médicamenteuse
    {
        "prompt": "A 19-year-old female is brought in by friends after reportedly ingesting 20 tablets of acetaminophen 500mg approximately 2 hours ago. She is drowsy but responsive. What is your triage assessment?",
        "max_tokens": 512,
        "temperature": 0.7,
    },
]


def check_health(base_url: str) -> bool:
    """Vérifie que l'endpoint est opérationnel."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        if r.status_code == 200:
            print(f"✅ Endpoint opérationnel ({base_url}/health → 200)")
            return True
        else:
            print(f"❌ Endpoint non prêt ({base_url}/health → {r.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Connexion refusée sur {base_url}")
        return False


def run_benchmark(base_url: str, num_runs: int) -> list[dict]:
    """Envoie les requêtes et mesure les latences."""
    results = []
    url = f"{base_url}/generate"

    for i in range(num_runs):
        prompt_data = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        prompt_label = prompt_data["prompt"][:60] + "..."

        print(f"\n  [{i+1}/{num_runs}] {prompt_label}")

        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=prompt_data, timeout=120)
            latency = time.perf_counter() - t0

            if r.status_code == 200:
                data = r.json()
                response_text = data.get("response", "")
                response_len = len(response_text)
                print(f"    → {latency:.2f}s | {response_len} chars | HTTP 200")
                results.append({
                    "index": i + 1,
                    "latency_s": latency,
                    "status": 200,
                    "response_length": response_len,
                    "prompt_excerpt": prompt_label,
                })
            else:
                latency = time.perf_counter() - t0
                print(f"    → {latency:.2f}s | HTTP {r.status_code} ⚠️")
                results.append({
                    "index": i + 1,
                    "latency_s": latency,
                    "status": r.status_code,
                    "response_length": 0,
                    "prompt_excerpt": prompt_label,
                })

        except requests.exceptions.Timeout:
            latency = time.perf_counter() - t0
            print(f"    → TIMEOUT après {latency:.2f}s ⚠️")
            results.append({
                "index": i + 1,
                "latency_s": latency,
                "status": "timeout",
                "response_length": 0,
                "prompt_excerpt": prompt_label,
            })

    return results


def print_report(results: list[dict]):
    """Affiche le résumé des métriques, prêt pour le rapport."""
    successful = [r for r in results if r["status"] == 200]

    if not successful:
        print("\n❌ Aucune requête réussie — impossible de calculer les métriques.")
        return

    latencies = [r["latency_s"] for r in successful]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    p50 = latencies_sorted[int(0.50 * n)]
    p95 = latencies_sorted[min(int(0.95 * n), n - 1)]
    p99 = latencies_sorted[min(int(0.99 * n), n - 1)]
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if n > 1 else 0.0
    min_lat = min(latencies)
    max_lat = max(latencies)

    response_lengths = [r["response_length"] for r in successful]
    avg_response_len = statistics.mean(response_lengths)

    print("\n" + "=" * 65)
    print("  RÉSUMÉ — Benchmark latence endpoint /generate")
    print("=" * 65)
    print(f"  Requêtes envoyées : {len(results)}")
    print(f"  Requêtes réussies : {n} / {len(results)}")
    print(f"  Erreurs / timeouts : {len(results) - n}")
    print("-" * 65)
    print(f"  Latence moyenne    : {mean:.2f}s (± {std:.2f}s)")
    print(f"  Latence min        : {min_lat:.2f}s")
    print(f"  Latence max        : {max_lat:.2f}s")
    print(f"  P50 (médiane)      : {p50:.2f}s")
    print(f"  P95                : {p95:.2f}s")
    print(f"  P99                : {p99:.2f}s")
    print("-" * 65)
    print(f"  Longueur moyenne réponse : {avg_response_len:.0f} caractères")
    print("=" * 65)

    # Tableau Markdown prêt pour le rapport
    print("\n📋 Tableau prêt à copier dans la section 6.7 du rapport :\n")
    print("| Métrique | Valeur |")
    print("|----------|--------|")
    print(f"| Requêtes réussies | {n} / {len(results)} |")
    print(f"| Latence moyenne | {mean:.2f}s (± {std:.2f}s) |")
    print(f"| Latence P50 (médiane) | {p50:.2f}s |")
    print(f"| **Latence P95** | **{p95:.2f}s** |")
    print(f"| Latence P99 | {p99:.2f}s |")
    print(f"| Latence min / max | {min_lat:.2f}s / {max_lat:.2f}s |")
    print(f"| Longueur moyenne de réponse | {avg_response_len:.0f} caractères |")


def main():
    parser = argparse.ArgumentParser(description="Benchmark latence P95 — endpoint /generate")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="URL de base de l'API (défaut: http://localhost:8000)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Nombre de requêtes à envoyer (défaut: 10)")
    args = parser.parse_args()

    print(f"\n🔬 Benchmark latence — {args.runs} requêtes sur {args.url}")
    print("-" * 65)

    if not check_health(args.url):
        print("\nArrêt : l'endpoint n'est pas prêt.")
        sys.exit(1)

    print(f"\n🚀 Lancement du benchmark ({args.runs} requêtes séquentielles)...")
    results = run_benchmark(args.url, args.runs)
    print_report(results)


if __name__ == "__main__":
    main()
