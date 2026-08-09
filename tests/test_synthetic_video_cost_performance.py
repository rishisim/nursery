from scripts.render_synthetic_video_cost_performance import ARMS, PROBES
import json


def test_cost_projection_keeps_viewpoint_as_failed_guardrail():
    assert PROBES[-1] == ("Viewpoint*", "viewpoint_guardrail")


def test_cost_projection_keeps_api_and_local_ltx_distinct():
    assert ARMS == (("Real", "real"), ("Hailuo", "hailuo"), ("LTX API", "ltx_api"), ("LTX Juno", "ltx_juno"))


def test_cost_projection_declares_unpriced_categories_and_no_lower_cost_claim():
    source = open("scripts/render_synthetic_video_cost_performance.py").read()
    assert '"real_fully_loaded_cost_observed": False' in source
    assert '"lower_cost_claim_supported": False' in source
    assert '"unpriced_synthetic_categories"' in source
    assert "shadow rate" in source


def test_juno_shadow_projection_uses_observed_allocated_gpu_time():
    result = json.load(open("results/synthetic_video_public_cost_projection.json"))
    assert result["juno_compute_projection"]["allocated_GPU_hours_per_generated_hour_at_100_percent_yield"] == 14.2
    assert result["juno_shadow_cost_sensitivity"]["shadow_cost_USD_per_generated_hour_at_100_percent_yield"]["4"] == 56.8
    assert result["real_fully_loaded_cost_observed"] is False
    assert result["lower_fully_loaded_cost_claim_supported"] is False
