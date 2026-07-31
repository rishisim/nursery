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
    assert "world_size != 2" in source
    assert "local_files_only=True" in source
    assert 'dist.init_process_group("nccl"' in source
    assert "public_dummy_only" in source
