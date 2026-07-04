# BabyWorld-Lite

A tiny working prototype for **synthetic developmental multimodal data**.

The point is not photorealism. The point is to show that a synthetic infant-like world can be deeply instrumented in ways real baby datasets usually cannot be: synchronized egocentric frames, child-directed language, action/proprioception traces, touch/contact signals, object state, causal event labels, and counterfactuals.

## What this demonstrates

Each generated episode contains:

- **RGB frames / GIF** from a toy egocentric scene.
- **Caregiver language** before and after the action.
- **Action command**: `tap`, `push`, `poke`, or `grasp`.
- **Proprioception**: hand position and velocity per frame.
- **Touch/contact**: contact binary, force, normal, slip/vibration proxies.
- **Object states**: position, velocity, angle per frame.
- **Causal event label**: e.g. `rolls_far`, `topples`, `slides`, `grasp_success`.
- **Causal graph and counterfactuals**: e.g. action/contact/material/impulse -> event.

It also includes a mini-benchmark that trains the same classifier under different input conditions:

1. vision + language proxy
2. plus action
3. plus sensorimotor/touch
4. oracle physical state

This gives a quick **data-usability** story: richer synthetic modalities can be evaluated by whether they improve action-effect prediction under fixed model/training conditions.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_dataset.py --n 1200 --out data/sample --render-first 12
python train_benchmark.py --data data/sample/manifest.csv --out data/sample
python make_demo_html.py --data data/sample --out demo.html
open demo.html  # or double-click it
```

## Leakage-aware forward-prediction eval

The honest eval reads `episodes.jsonl` directly and recomputes all features from
frames `[0, k]`, where `k` is the contact/prediction frame. It does not use the
legacy manifest summaries (`max_force`, `contact_count`, `slip_estimate`,
`vibration_energy`) or `hardness_proxy` as inputs. The training objective is
continuous rollout prediction: post-contact displacement, x/y delta, final x/y
position, and topple angle. Event labels are used only for coverage/diversity
reporting, not as model targets.

```bash
python3 generate_dataset.py --n 6000 --out data/honest --render-first 12
python3 run_honest_eval.py --episodes data/honest/episodes.jsonl --out data/honest/eval
```

By default the full run evaluates mechanism splits: held-out material,
held-out high impulse x high mass region, and held-out object/action
composition. A random split is still available explicitly as a contrast.

Outputs:

- `results.json`
- `forward_prediction_report.txt`
- `coverage_report.txt`
- `coverage_report.json`
- `{split}_displacement_mae.png`
- `k_sweep_error_vs_k.png`

For quick smoke tests on the bundled 800-episode sample, lower the exploratory
support threshold or run a subset of splits:

```bash
python3 run_honest_eval.py --episodes data/sample/episodes.jsonl --out data/sample/eval --splits random held_out_material --min-test-cell-size 1 --bootstrap 100 --skip-k-sweep
python3 -m pytest
```

## Why this is deliberately small

The full idea is huge. This pilot isolates the core claim:

> Synthetic developmental data can be cheap, scalable, perfectly instrumented, and benchmarked for usefulness even before it is photorealistic.

Next extensions:

- replace 2D physics with PyBullet, TDW, Habitat, or OmniGibson;
- add depth, segmentation masks, richer audio, and object IDs;
- train a small VLM/world-model instead of a classifier;
- test transfer to real egocentric data or held-out simulated worlds;
- add a cost-normalized leaderboard: performance per generated hour, per dollar, or per causal event.

 