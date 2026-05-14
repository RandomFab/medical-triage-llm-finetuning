"""Smoke test : structure du Dockerfile et fichiers de déploiement."""
from pathlib import Path

# Remonte de tests/smoke/ vers la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDockerfileStructure:

    def test_dockerfile_exists(self):
        assert (PROJECT_ROOT / "Dockerfile").exists(), (
            "Dockerfile manquant à la racine du projet"
        )

    def test_dockerfile_has_from(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "FROM" in content, "Dockerfile sans instruction FROM"

    def test_dockerfile_uses_gpu_image(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "vllm" in content.lower(), (
            "Le Dockerfile devrait utiliser une image de base vLLM (avec CUDA)"
    )

    def test_dockerfile_exposes_port_8000(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content, (
            "Le Dockerfile doit exposer le port 8000 (port de l'API FastAPI)"
        )

    def test_dockerfile_has_cmd(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "CMD" in content, "Dockerfile sans instruction CMD"

    def test_dockerfile_runs_uvicorn(self):
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "uvicorn" in content, (
            "Le CMD devrait lancer uvicorn pour servir l'API FastAPI"
        )

    def test_dockerfile_copies_src_and_config(self):
        """Vérifie que le Dockerfile copie les deux dossiers nécessaires à l'API."""
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "COPY src/" in content, "Le Dockerfile ne copie pas src/"
        assert "COPY config/" in content, "Le Dockerfile ne copie pas config/"


class TestDockerIgnore:

    def test_dockerignore_exists(self):
        assert (PROJECT_ROOT / ".dockerignore").exists(), (
            ".dockerignore manquant — Docker copiera .git/, data/, notebooks/ dans l'image"
        )


class TestCICDWorkflow:

    def test_github_workflow_exists(self):
        workflow = PROJECT_ROOT / ".github" / "workflows" / "cicd.yml"
        assert workflow.exists(), (
            "Workflow GitHub Actions manquant (.github/workflows/cicd.yml)"
        )

    def test_workflow_has_test_job(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "cicd.yml").read_text()
        assert "pytest" in content, (
            "Le workflow CI/CD ne contient pas de step pytest"
        )

    def test_workflow_has_docker_build(self):
        content = (PROJECT_ROOT / ".github" / "workflows" / "cicd.yml").read_text()
        assert "docker" in content.lower(), (
            "Le workflow CI/CD ne contient pas de step Docker build"
        )