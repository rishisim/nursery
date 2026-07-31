from pathlib import Path


def test_preflight_freezes_approved_final_topology_and_offline_controls():
    batch = Path("scripts/phase4_flux_topology_preflight.sbatch").read_text()
    source = Path("scripts/phase4_flux_topology_preflight.py").read_text()
    assert "#SBATCH --partition=h200" in batch
    assert "#SBATCH --nodes=1" in batch
    assert "#SBATCH --ntasks-per-node=2" in batch
    assert "#SBATCH --gpus-per-node=2" in batch
    assert "#SBATCH --time=00:30:00" in batch
    assert "singularity exec --nv" in batch
    assert "pytorch-2.8.0-cu126.sif" in batch
    assert "phase4-pydeps.tar" in batch
    assert 'export PYTHONPATH="$local_tmp/pydeps"' in batch
    assert "world_size != 2" in source
    assert "local_files_only=True" in source
    assert 'dist.init_process_group("nccl"' in source
    assert "public_dummy_only" in source


def test_public_container_preparation_is_cpu_only_and_bounded():
    batch = Path("scripts/phase4_prepare_public_container.sbatch").read_text()
    assert "#SBATCH --partition=dev" in batch
    assert "#SBATCH --mem=64G" in batch
    assert "#SBATCH --time=00:30:00" in batch
    assert "#SBATCH --gpus" not in batch
    assert "SINGULARITY_TMPDIR" in batch
    assert "pytorch:2.8.0-cuda12.6-cudnn9-runtime" in batch
    assert "--no-deps" in batch
    assert "phase4-pydeps.tar" in batch
    assert "CONTAINER_ENVIRONMENT_READY" in batch
