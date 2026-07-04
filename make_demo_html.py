from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ARM_LABELS = {
    "vision": "vision",
    "vision_proprio": "vision + proprio",
    "vision_proprio_touch": "vision + proprio + touch",
    "oracle_full_state": "oracle full state",
}

SPLIT_LABELS = {
    "held_out_material": "held-out material",
    "held_out_impulse_mass": "held-out high impulse/mass",
    "held_out_composition": "held-out object/action",
    "random": "random",
}

HONEST_ARMS = ("vision", "vision_proprio", "vision_proprio_touch", "oracle_full_state")


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_existing_dir(path: Path) -> Path:
    if path.exists():
        return path
    nested = Path("babyworld-lite") / path
    if nested.exists():
        return nested
    return path


def format_mae_cell(split: dict, arm: str) -> str:
    metric = split["arms"][arm]["regression"]["target_displacement"]["mae"]
    point = metric["point"]
    lo, hi = metric["ci95"]
    return f"{point:.2f} <span class=ci>[{lo:.2f}, {hi:.2f}]</span>"


def honest_eval_section(eval_dir: Path) -> str:
    results_path = eval_dir / "results.json"
    if not results_path.exists():
        return f"""
<h2>Honest forward-prediction eval</h2>
<p class=note>Honest eval outputs were not found at <code>{eval_dir}</code>; run <code>python3 run_honest_eval.py --episodes data/honest/episodes.jsonl --out data/honest/eval</code> before regenerating this demo.</p>
"""

    results = json.loads(results_path.read_text())
    splits = results.get("splits", {})
    rows = []
    for split_name, split in splits.items():
        cells = "".join(f"<td>{format_mae_cell(split, arm)}</td>" for arm in HONEST_ARMS if arm in split["arms"])
        rows.append(
            f"<tr><td>{SPLIT_LABELS.get(split_name, split_name)}<br><span class=muted>n_test={split['n_test']}</span></td>{cells}</tr>"
        )

    rules = results.get("decision_rules", {})
    touch_rule = rules.get("touch_helps_held_out_displacement", {})
    negative_rule = rules.get("honest_negative_escalation", {})
    touch_status = touch_rule.get("status", "unknown")
    if negative_rule.get("triggered") is True:
        negative_text = "triggered"
    elif negative_rule.get("triggered") is False:
        negative_text = "did not trigger"
    else:
        negative_text = "unknown"
    distinction = (
        "Pre-registered vs exploratory: decision_rules.yaml registered the touch-help and honest-negative checks "
        f"(touch-help {touch_status}; honest-negative escalation {negative_text}), while the k-sweep and this demo's chart selection are exploratory diagnostics."
    )

    headline_plot = eval_dir / "held_out_material_displacement_mae.png"
    if not headline_plot.exists():
        headline_candidates = sorted(eval_dir.glob("*_displacement_mae.png"))
        headline_plot = headline_candidates[0] if headline_candidates else headline_plot
    k_sweep_plot = eval_dir / "k_sweep_error_vs_k.png"

    headline_img = (
        f'<figure><img class="plot" alt="Held-out material displacement MAE chart" src="data:image/png;base64,{b64(headline_plot)}" />'
        "<figcaption>Headline mechanism split: held-out material displacement MAE with 95% CI. Lower is better.</figcaption></figure>"
        if headline_plot.exists()
        else ""
    )
    k_sweep_img = (
        f'<figure><img class="plot" alt="K-sweep displacement error plot" src="data:image/png;base64,{b64(k_sweep_plot)}" />'
        "<figcaption>Exploratory k-sweep: displacement MAE as the prediction frame moves after contact. Lower is better.</figcaption></figure>"
        if k_sweep_plot.exists()
        else ""
    )

    table_header = "".join(f"<th>{ARM_LABELS[arm]}</th>" for arm in HONEST_ARMS)
    return f"""
<h2>Honest forward-prediction eval</h2>
<p>Models predict continuous post-contact rollout targets from frames [0,k]; event labels are used for coverage reporting, not as prediction targets.</p>
<p class=note>{distinction}</p>
<div class=table-wrap><table class=metrics><tr><th>Mechanism split</th>{table_header}</tr>{''.join(rows)}</table></div>
{headline_img}
{k_sweep_img}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained BabyWorld-Lite demo HTML page.")
    parser.add_argument("--data", type=str, default="data/honest")
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default="demo.html")
    parser.add_argument("--max-episodes", type=int, default=6)
    args = parser.parse_args()

    data = resolve_existing_dir(Path(args.data))
    eval_dir = resolve_existing_dir(Path(args.eval_dir)) if args.eval_dir else data / "eval"
    records = []
    with (data / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if "gif_path" in rec:
                gif_path = data / rec["gif_path"]
                rec["gif_b64"] = b64(gif_path)
                records.append(rec)
            if len(records) >= args.max_episodes:
                break

    honest_section = honest_eval_section(eval_dir)

    cards = []
    for rec in records:
        obj = rec["object"]
        tactile = rec["tactile_summary"]
        cf = rec["counterfactuals"]
        cards.append(f"""
        <section class=card>
          <div class=left><img src="data:image/gif;base64,{rec['gif_b64']}" /></div>
          <div class=right>
            <h3>Episode {rec['episode_id']:05d}: {rec['action']} → {rec['event_label']}</h3>
            <p><b>Caregiver before:</b> {rec['utterance_pre']}</p>
            <p><b>Caregiver after:</b> {rec['utterance_post']}</p>
            <p><b>Object:</b> {obj['color_name']} {obj['material']} {obj['shape']} &nbsp; | &nbsp; mass={obj['mass']:.2f}, friction={obj['friction']:.2f}, bounce={obj['bounciness']:.2f}</p>
            <p><b>Tactile/action summary:</b> first_force={tactile['first_contact_force']:.2f}, contact_count={tactile['contact_count']:.0f}, slip={tactile['slip_estimate']:.2f}, vibration={tactile['vibration_energy']:.2f}</p>
            <p><b>Causal graph:</b> action/contact/impulse/object/material → event</p>
            <p><b>Counterfactuals:</b> {json.dumps(cf)}</p>
          </div>
        </section>
        """)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>BabyWorld-Lite: Synthetic developmental multimodal pilot</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; background:#fafafa; color:#222; overflow-x:hidden; }}
h1 {{ margin-bottom: 0; }}
.subtitle {{ color:#555; max-width: 900px; }}
.card {{ display:flex; gap:24px; background:white; border:1px solid #ddd; border-radius:14px; padding:16px; margin:18px 0; box-shadow:0 2px 6px rgba(0,0,0,0.05); }}
.left img {{ width:224px; height:224px; border-radius:10px; border:1px solid #ddd; }}
.right {{ flex:1; }}
pre {{ background:#f1f1f1; padding:12px; border-radius:10px; overflow:auto; }}
table {{ border-collapse:collapse; background:white; margin-top:12px; }}
td, th {{ border:1px solid #ddd; padding:8px 12px; }}
.table-wrap {{ display:block; max-width:100%; overflow-x:auto; overflow-y:hidden; -webkit-overflow-scrolling:touch; }}
.metrics td, .metrics th {{ text-align:left; vertical-align:top; }}
.ci {{ color:#666; white-space:nowrap; }}
.muted {{ color:#666; font-size:0.92em; }}
.note {{ color:#444; max-width:900px; }}
figure {{ margin:18px 0; }}
figcaption {{ color:#555; font-size:0.95em; margin-top:6px; max-width:900px; }}
.plot {{ max-width:900px; width:100%; border:1px solid #ddd; border-radius:12px; background:white; padding:8px; box-sizing:border-box; }}
.badge {{ display:inline-block; background:#eef; padding:4px 8px; border-radius:10px; margin-right:8px; }}
@media (max-width: 700px) {{
  body {{ margin: 32px; }}
  .card {{ flex-direction:column; gap:12px; }}
  .left img {{ width:min(224px, 100%); height:auto; aspect-ratio:1 / 1; }}
}}
</style>
</head>
<body>
<h1>BabyWorld-Lite</h1>
<p class=subtitle>A tiny proof-of-concept for synthetic developmental multimodal data: egocentric frames, child-directed language, action/proprioception traces, touch/contact signals, object state, causal event labels, and counterfactuals.</p>
<p><span class=badge>video</span><span class=badge>speech text</span><span class=badge>action</span><span class=badge>proprioception</span><span class=badge>touch/contact</span><span class=badge>causal graph</span><span class=badge>counterfactuals</span></p>
{honest_section}
<h2>Example generated episodes</h2>
{''.join(cards)}
<h2>Core research claim this prototype demonstrates</h2>
<p>Instead of trying to curate impossible real baby data with every modality, we can instrument a synthetic world where the model receives the modalities babies use and where the dataset includes causal ground truth. Then we can run DataComp-style controlled tests: hold the model and compute fixed, vary the data, and measure transfer/usefulness per dollar.</p>
</body>
</html>
"""
    Path(args.out).write_text(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
