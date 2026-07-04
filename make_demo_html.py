from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained BabyWorld-Lite demo HTML page.")
    parser.add_argument("--data", type=str, default="data/sample")
    parser.add_argument("--out", type=str, default="demo.html")
    parser.add_argument("--max-episodes", type=int, default=6)
    args = parser.parse_args()

    data = Path(args.data)
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

    bench = {}
    bench_path = data / "benchmark_results.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
    plot_b64 = ""
    plot_path = data / "benchmark_plot.png"
    if plot_path.exists():
        plot_b64 = b64(plot_path)

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

    bench_rows = "".join([
        f"<tr><td>{name}</td><td>{vals['accuracy']:.3f}</td><td>{vals['macro_f1']:.3f}</td></tr>"
        for name, vals in bench.items()
    ])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>BabyWorld-Lite: Synthetic developmental multimodal pilot</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; background:#fafafa; color:#222; }}
h1 {{ margin-bottom: 0; }}
.subtitle {{ color:#555; max-width: 900px; }}
.card {{ display:flex; gap:24px; background:white; border:1px solid #ddd; border-radius:14px; padding:16px; margin:18px 0; box-shadow:0 2px 6px rgba(0,0,0,0.05); }}
.left img {{ width:224px; height:224px; border-radius:10px; border:1px solid #ddd; }}
.right {{ flex:1; }}
pre {{ background:#f1f1f1; padding:12px; border-radius:10px; overflow:auto; }}
table {{ border-collapse:collapse; background:white; margin-top:12px; }}
td, th {{ border:1px solid #ddd; padding:8px 12px; }}
.plot {{ max-width:720px; border:1px solid #ddd; border-radius:12px; background:white; padding:8px; }}
.badge {{ display:inline-block; background:#eef; padding:4px 8px; border-radius:10px; margin-right:8px; }}
</style>
</head>
<body>
<h1>BabyWorld-Lite</h1>
<p class=subtitle>A tiny proof-of-concept for synthetic developmental multimodal data: egocentric frames, child-directed language, action/proprioception traces, touch/contact signals, object state, causal event labels, and counterfactuals.</p>
<p><span class=badge>video</span><span class=badge>speech text</span><span class=badge>action</span><span class=badge>proprioception</span><span class=badge>touch/contact</span><span class=badge>causal graph</span><span class=badge>counterfactuals</span></p>
<h2>Data-usability mini-benchmark</h2>
<p>Same model class and train/test split. Only the available modalities change. This is not a scientific result yet; it is a sanity-check that the benchmark can measure whether richer synthetic modalities help action-effect prediction.</p>
<table><tr><th>Input condition</th><th>Accuracy</th><th>Macro F1</th></tr>{bench_rows}</table>
{f'<p><img class="plot" src="data:image/png;base64,{plot_b64}" /></p>' if plot_b64 else ''}
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
