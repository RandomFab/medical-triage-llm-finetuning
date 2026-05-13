"""Test 2 — Vérification de la configuration des chemins (config/paths.py)."""
from pathlib import Path
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    ROOT_MODEL_DIR,
    PARAMS_PATH,
    SFT_DATASET_DIR,
    SFT_TRAIN_DATASET_PATH,
    SFT_VAL_DATASET_PATH,
    SFT_TEST_DATASET_PATH,
    DPO_DATASET_DIR,
    DPO_TRAIN_DATASET_PATH,
    DPO_VAL_DATASET_PATH,
    DPO_TEST_DATASET_PATH,
    RAW_DATA_GCS_URL,
    GCS_MODEL_PATH,
    GCS_MERGED_MODEL_PATH,
)


class TestLocalPaths:
    """Vérifie que les chemins locaux sont bien des Path et cohérents entre eux."""

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists(), "PROJECT_ROOT n'existe pas sur le filesystem"
        assert PROJECT_ROOT.is_dir()

    def test_local_paths_are_path_objects(self):
        """Tous les chemins locaux doivent être des pathlib.Path, pas des strings."""
        for path in [DATA_DIR, PROCESSED_DATA_DIR, ROOT_MODEL_DIR, PARAMS_PATH]:
            assert isinstance(path, Path), f"{path} devrait être un Path, got {type(path)}"

    def test_data_dir_is_child_of_project(self):
        assert DATA_DIR.parent == PROJECT_ROOT

    def test_processed_is_child_of_data(self):
        assert PROCESSED_DATA_DIR.parent == DATA_DIR

    def test_sft_paths_under_sft_dir(self):
        """Les 3 splits SFT doivent être dans le même dossier."""
        assert SFT_TRAIN_DATASET_PATH.parent == SFT_DATASET_DIR
        assert SFT_VAL_DATASET_PATH.parent == SFT_DATASET_DIR
        assert SFT_TEST_DATASET_PATH.parent == SFT_DATASET_DIR

    def test_dpo_paths_under_dpo_dir(self):
        """Les 3 splits DPO doivent être dans le même dossier."""
        assert DPO_TRAIN_DATASET_PATH.parent == DPO_DATASET_DIR
        assert DPO_VAL_DATASET_PATH.parent == DPO_DATASET_DIR
        assert DPO_TEST_DATASET_PATH.parent == DPO_DATASET_DIR

    def test_parquet_extensions(self):
        """Tous les fichiers dataset doivent avoir l'extension .parquet."""
        for path in [
            SFT_TRAIN_DATASET_PATH, SFT_VAL_DATASET_PATH, SFT_TEST_DATASET_PATH,
            DPO_TRAIN_DATASET_PATH, DPO_VAL_DATASET_PATH, DPO_TEST_DATASET_PATH,
        ]:
            assert path.suffix == ".parquet", f"{path.name} n'a pas l'extension .parquet"


class TestGCSPaths:
    """Vérifie que les chemins GCS sont bien formés."""

    def test_gcs_paths_have_gs_prefix(self):
        """Tous les chemins GCS doivent commencer par gs:// — sinon vLLM/GCS ne les résoudra pas."""
        assert RAW_DATA_GCS_URL.startswith("gs://"), f"RAW_DATA_GCS_URL invalide : {RAW_DATA_GCS_URL}"
        assert GCS_MODEL_PATH.startswith("gs://"), f"GCS_MODEL_PATH invalide : {GCS_MODEL_PATH}"
        assert GCS_MERGED_MODEL_PATH.startswith("gs://"), f"GCS_MERGED_MODEL_PATH invalide : {GCS_MERGED_MODEL_PATH}"

    def test_gcs_paths_are_strings(self):
        """Les chemins GCS viennent de os.environ.get() → toujours des str, jamais des Path."""
        assert isinstance(RAW_DATA_GCS_URL, str)
        assert isinstance(GCS_MODEL_PATH, str)
        assert isinstance(GCS_MERGED_MODEL_PATH, str)

    def test_gcs_bucket_name_is_consistent(self):
        """Les 3 chemins GCS doivent pointer vers le même bucket."""
        bucket = "gs://p14-medical-data/"
        assert RAW_DATA_GCS_URL.startswith(bucket)
        assert GCS_MODEL_PATH.startswith(bucket)
        assert GCS_MERGED_MODEL_PATH.startswith(bucket)