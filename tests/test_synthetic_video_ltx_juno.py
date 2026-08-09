from argparse import Namespace
import json

import pytest

from scripts.run_synthetic_video_ltx_juno import JunoLTXError, PROMPT_COMMITMENT, run


def test_juno_runner_fails_before_model_import_on_wrong_prompt_commitment(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"commitment_sha256": "wrong", "prompt": "public prompt"}))
    args = Namespace(
        plan=plan,
        checkpoint=tmp_path / "checkpoint",
        upsampler=tmp_path / "upsampler",
        gemma_root=tmp_path / "gemma",
        output=tmp_path / "output.mp4",
        record=tmp_path / "record.json",
        source_revision="9377758131b1ffde4b7f766804590a6617bf2ab9",
        weights_revision="4229404625088d21c4f112eb640fb04a0900ee25",
        gemma_revision="68f7ee4fbd59087436ada77ed2d62f373fdd4482",
        seed=314159,
        height=640,
        width=1152,
        num_frames=241,
        frame_rate=24.0,
    )
    with pytest.raises(JunoLTXError, match="E_PROMPT_COMMITMENT"):
        run(args)
    assert PROMPT_COMMITMENT not in plan.read_text()


def test_juno_wrapper_freezes_one_full_h100_and_suppresses_slurm_logs():
    source = open("scripts/run_synthetic_video_ltx_juno.sbatch").read()
    assert "#SBATCH --gres=gpu:nvidia_h100_nvl:1" in source
    assert "#SBATCH --output=/dev/null" in source
    assert "#SBATCH --error=/dev/null" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "--num-frames" not in source
