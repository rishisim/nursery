# Pre-access evidence summary

Status: **infrastructure validated; no language-effect claim.**

## Dataset gate

- Profile: `provisional_not_babyview_matched`.
- Scale: 96 base episodes and 1152 factorial examples.
- Audit valid: `true`; failures: `[]`.
- Language arms: strong, weak, shuffled.
- Motor arms: null, synchronized, split-local episode-shuffled, time-shifted.
- Model-visible inputs passed the allowlist; oracle state is stored separately.
- No shuffled self-matches, cross-split donors, split hash overlap, or split composition overlap.
- Realized utterance rate: 53.75/minute (provisional target 52.00).
- Audit includes utterance count/length/timing, silent-frame fraction, target-word frequency, activity duration, visibility/occlusion, camera motion, distractor count, and object/action/material marginals.

## Pilot gate

- Source/alignment: `grounding-v0` / `weak`.
- Paired runs: seeds [11, 22, 33]; n_train=83; n_test=13.
- Primary test uses rendered RGB and text only; the motor encoder is omitted.
- Synchronized - shuffled: -20.37 percentage points (seed-bootstrap 95% CI [-33.80, -1.39]).
- Synchronized - null: -24.23 percentage points (seed-bootstrap 95% CI [-31.48, -15.74]).
- Synchronized - time-shifted: -6.33 percentage points (seed-bootstrap 95% CI [-14.81, +0.00]).

Synchronized does not beat the shuffled or null controls, so the prespecified positive-effect gate fails. The result is not evidence of a motor-cue benefit.

## Official Machine-DevBench gate

- Official EgoBabyVLM commit: `224621caf0628270b6115845ac75a65b984234a3`.
- Public benchmark installed: 20 manifests / 3721 trials across realistic and cartoon styles.
- Nursery checkpoint implements the official `MultiModalFeatureExtractor` boundary and completed every trial.
- Diagnostic Nursery result: lexical 51.6, grammatical 17.1, overall 34.4.
- Exact target-vocabulary coverage: 13/1414 (0.9%).

The Nursery score is not a substantive benchmark result: the vocabulary audit shows extensive UNK collisions. The completed run validates the integration and fixes the next requirement—expand/calibrate the learner's linguistic experience before interpreting Machine-DevBench.

The off-the-shelf OpenCLIP reference produced lexical 86.3, grammatical 70.3, and overall 78.3; the published CLIP-L leaderboard row is 87.3 / 70.4 / 78.8.

## Reproduce

```bash
source .venv/bin/activate
python -m babyworld_lite.grounding --config configs/grounding_provisional.yaml --out data/grounding_provisional
python train_grounding_pilot.py --episodes data/grounding_provisional/examples.jsonl --out data/grounding_pilot_weak --seeds 11 22 33 --holdout cup:push --holdout plush:grasp --alignment weak --epochs 8 --batch-size 24 --frame-count 6 --image-size 48 --hidden-dim 48 --embedding-dim 32 --device auto
python scripts/train_nursery_checkpoint.py --episodes data/grounding_provisional/examples.jsonl --out output/machine_devbench/nursery_sync_seed11.pt --arm synchronized --seed 11 --alignment weak --epochs 8 --device auto
python scripts/run_machine_devbench.py --model nursery --checkpoint output/machine_devbench/nursery_sync_seed11.pt --out output/machine_devbench/nursery_full.json --device auto
python scripts/audit_machine_devbench_coverage.py --checkpoint output/machine_devbench/nursery_sync_seed11.pt
python scripts/run_machine_devbench.py --model openclip --openclip-model ViT-L-14-quickgelu --openclip-pretrained openai --out output/machine_devbench/openclip_l_full.json --device auto
python scripts/build_pre_access_evidence.py
```
