from pathlib import Path

from scripts.childlens_viewer_view_preflight import evaluate_host


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_viewer_view_preflight_v1.json"


def test_apple_silicon_without_cuda_stops_before_media() -> None:
    result = evaluate_host(
        CONFIG, which=lambda _: None, system="Darwin", machine="arm64"
    )
    assert result["decision"] == "STOP_HARDWARE_PRIVACY_NO_LOCAL_CUDA"
    assert result["next_stage_authorized"] is False
    assert result["media_opened"] is False
    assert result["network_private_data_transfer"] is False


def test_cuda_tools_authorize_instrument_stage() -> None:
    result = evaluate_host(
        CONFIG,
        which=lambda name: f"/opt/cuda/bin/{name}",
        system="Linux",
        machine="x86_64",
    )
    assert result["decision"] == "PASS_LOCAL_CUDA_EXECUTION_PATH"
    assert result["next_stage_authorized"] is True
