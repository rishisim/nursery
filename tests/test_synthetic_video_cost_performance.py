from scripts.render_synthetic_video_cost_performance import PROBES


def test_cost_projection_keeps_viewpoint_as_failed_guardrail():
    assert PROBES[-1] == ("Viewpoint*", "viewpoint_guardrail")


def test_cost_projection_declares_unpriced_categories_and_no_lower_cost_claim():
    source = open("scripts/render_synthetic_video_cost_performance.py").read()
    assert '"real_fully_loaded_cost_observed": False' in source
    assert '"lower_cost_claim_supported": False' in source
    assert '"unpriced_synthetic_categories"' in source
