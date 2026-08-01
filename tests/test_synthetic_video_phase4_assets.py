import json
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "phase4_assets", Path("scripts/run_synthetic_video_phase4_assets.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
canonical = MODULE.canonical
digest = MODULE.digest


def test_phase4_asset_config_freezes_shared_assets_and_split():
    cfg = json.loads(Path("configs/synthetic_video_phase4_assets.json").read_text())
    assert cfg["allocation"]["counts"] == {"training": 28, "evaluation": 8, "validation": 4}
    assert cfg["lexical"]["styles"] == ["realistic", "cartoon"]
    assert cfg["temporal"]["candidate_count"] == 8
    assert cfg["sealing"]["all_later_arms"] == ["Real-full", "Synthetic-full", "Real-small", "Mixed"]
    assert cfg["sealing"]["test_assets_may_steer_later_work"] is False


def test_canonical_digest_is_order_independent():
    assert canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_language_model_preparation_is_public_cpu_only_and_offline_validated():
    batch = Path("scripts/phase4_prepare_language_models.sbatch").read_text()
    assert "#SBATCH --partition=dev" in batch
    assert "#SBATCH --gpus" not in batch
    assert "openai-whisper==20250625" in batch
    assert "1a922f3b32a8e809e17a47d4b32142d8105924e5" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "local_files_only=True" in batch
    assert "LANGUAGE_ENVIRONMENT_READY" in batch


def test_governed_build_freezes_final_topology_and_seals_common_assets():
    batch = Path("scripts/build_synthetic_video_phase4_assets.sbatch").read_text()
    source = Path("scripts/build_synthetic_video_phase4_assets.py").read_text()
    assert "#SBATCH --partition=h200" in batch
    assert "#SBATCH --ntasks-per-node=2" in batch
    assert "#SBATCH --gpus-per-node=2" in batch
    assert "#SBATCH --time=12:00:00" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert 'dist.init_process_group("nccl"' in source
    assert "calibration_C_only" in source
    assert '"Recall@1","MRR"' in source
    assert "common_asset_references" in source
    assert "test_assets_may_steer_later_work" in source
