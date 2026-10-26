# Learner: how the body is used

Goal (AGENTS.md): body inputs should raise the word-learning ceiling beyond
video alone. Measure both understanding with real vision + body available and
transfer to later picture + word testing without the body. A reliable gain in
either advances the goal; picture-only retention is an additional achievement,
not a requirement for recognizing a body-present benefit. Oracle inputs remain
diagnostics. See the [current plan](../PLAN.md) and the
[archived comparisons](../archive/language_learning/experiments.md).

## Status and scope

This is the implementation reference, including diagnostic flags retained for
reproduction. Their presence does not mean they are recommended next runs.
The additive linear learner is the reference baseline; the joint learner is a
completed comparison. Results and superseded design rationales belong to the
[experiment archive](../archive/language_learning/experiments.md) and
[cue/gate history](../archive/language_learning/cue_experiments.md).

Frozen DINOv2 features feed a trainable visual projection and trainable word
vectors. The direct-label diagnostic demonstrated visual headroom; earlier
claims that perfect naming capped improvement at about two points must not be
treated as a general limit. Weak naming remains part of the scientific task.

## Visual side (identical in every condition)

Linear visual projection and word vectors (upstream `RegionMIL`), soft-MIL pooling (`--pool soft`, tau_p 0.1).
The video-only condition is this model alone. Subject s8 is excluded from training in every condition.

## Body input (per second, from `body/<video>.npz`)

- Contact: fraction of each anatomical patch touching, per hand: 5 fingertips, 5 finger segments, palm = 22.
  Patch map: `nursery/learner/hand_patches.json` (derivation in `body.py`). Vertices are the 778 SMPL-X hand
  vertices per hand (left, then right) packed in `hand_contact`; each goes to the finger joint with the largest
  skinning weight (distal joint = tip, proximal and middle = segment, wrist = palm).
- Finger configuration: local (palm-relative) finger rotations, both hands (90).
- Arm: elbows and wrists in the head camera frame (full camera rotation), their 1-step differences, and the
  gravity direction in that frame (27). Differences are zero at the first second of a clip.
- Missing mask for absent seconds. Excluded: object pose, translation, global orientation, body pose, labels,
  asset ids. Normalisation is fit on training sessions only.
- Camera: camera-to-world, -Z forward, 320x240, 100 deg HFOV. The DINO centre crop covers the central square.

## Body encoder

One MLP per group (contact, fingers, arm), then a bidirectional GRU (2 x 64 = 128-d) over +-5 s. The state is reset
at clip boundaries in both directions, so nothing from another clip reaches the centre. Clip boundaries come
from the `clip` column of `frames.parquet` (one entry per second).

## Training

1. Body pathway (required, on with `--body`). The language loss scores a pair as picture score +
   beta * cos(projected body latent, word vector). Each pair drops its body with p = 0.5 (`--body-dropout`), so the
   picture path must carry the word on its own. Gradients reach the regions, words and every body sub-encoder.
   Dropout 1.0 equals video-only.
2. Vision -> body prediction (`--w-pred`, on by default). The projected regions that are used for word matching
   (17 per frame) predict that second's per-patch contact pattern, its change to the next second, grip aperture
   and hand-in-view. The gradient goes into the shared projection.
3. Optional, off by default, kept behind flags: contact-episode bags and episode consistency
   (`--episodes contact`, `--w-episode`, `--w-consist`; `--episodes temporal` is the length-matched control),
   masked body prediction (`--w-mask`).

Weights are in `nursery/learner/config.json`; every term has a CLI override and 0 switches it off. Defaults were
not tuned (no dev sweep): pred 0.5, dropout 0.5.

## Controls and diagnostics (flags, not part of the two-run comparison)

- `--body shuffled`: whole clip-length body sequences swapped between clips (equal lengths), marginals kept.
- `--body oracle`: true held-object identity through the same body slot.
- Body-present 4AFC on s8 named utterances with a visible referent, and a body probe (held word, grasp family,
  next-second contact; split by subject and by source grasp family; persistence and mean baselines).

## Tests

- Body-present understanding: compare real vision + body with the matched video-only learner on identical
  held-out trials. Report all, held-referent and non-held-referent trials; removing the body from the multimodal
  learner is an additional ablation, not a substitute for the video-only baseline.
- Picture-only 4AFC: rendered test (held / seen-only / extras), Konkle-60. Epoch picked on rendered dev.
- Unit tests (`tests/test_learner.py`): the language loss changes with the body; gradients reach every
  sub-encoder; dropout 1.0 equals video-only; picture-only eval never calls the body; clip-boundary reset and
  masking; episode segmentation; shuffle keeps marginals; each loss switches off.

## Conditions

Video-only (soft-MIL) vs video + body. Seed 0, same data, speech, dev selection. Stage further controls from the result.

## As built

- Encoder is bidirectional (2 x 64), not a single 128 GRU, so the centre of a window sees both sides.
- `held_category` is never an input. Episodes end at clip changes (which include any held-object change),
  gaps, and runs shorter than 2 s; held_category only checks episodes in tests.
- Mask for windows that cross a boundary: steps from other clips are reset out, not dropped; losses on
  masked/predicted steps use only the centre's clip.
- `--pool max` stays as the default so existing hard-max runs and the upstream path reproduce; every body
  condition requires `--pool soft`.
- Late fusion uses the picture score during training with body dropout (the pathway), instead of a
  detached test-only term.
- `--w-bodymatch` (default 0): body-only matching, weighted InfoNCE between `fuse(latent)` cosine scores and the batch's
  word bags, with the same pair weights as the language loss, on every pair (body dropout does not apply).

## Oracle-to-picture distillation (ORACLE DIAGNOSTIC)

Flags: `--distill-from <oracle run dir>`, `--distill-weight` (lambda, default 1), `--distill-temp` (T, default 2),
`--inspect-epoch N`. Off by default; the default path is unchanged. Needs a picture-only soft-pooling student.

- Teacher: a finished `--body oracle` run, frozen, eval mode, always given its oracle body (no dropout). Its target for
  each training batch is the detached scaled training score matrix (soft-MIL picture score + beta * body cosine, the
  same code as the language loss, tau_p of the run). Candidates are the batch's word bags, as in training.
- Loss: language loss + lambda * T^2 * KL(teacher || student), both matching directions, with the student's own
  picture-only scaled score. The KL updates the visual projection and word vectors. Lambda 0 is the plain run.
- Building the teacher does not touch the student's CPU RNG stream. Check a new control against the stored
  video-only history before reading a comparison.
- Body-present diagnostic: `picture` / `fused` use hard max over regions (old); `picture_soft` / `fused_soft` use the
  training pooling (soft-MIL, tau_p). Read the soft ones. `body` is unchanged.
- Agreement (metrics `checkpoints`, at epoch N and at the dev-selected epoch): KL at T and top-1 agreement with the
  teacher on training pairs and on s8 named visible-referent pairs (random fixed batches of 128). s8 pairs are used
  because dev here is a picture-probe split with no teacher body.

## Clip split (`--split clips`)

Default `--split subject` is unchanged. `--split clips` (needs `--pool soft`; no distill/labels/shuffled/oracle) also
holds out 15% dev / 15% test of each of s1-s7's source clips (config `body.clip_split`, seed 0; all replays of a clip
stay together). Pairs whose image window touches another partition are dropped; body normalisation uses train frames only.
Every epoch logs body-present 4AFC (foils from the same subject) on `clips_dev`, `clips_test` and `s8` as `bp`. A second
epoch is selected on `clips_dev` (`fused_soft`, or `picture_soft` for `--body none`) and saved as `model_bodypresent.pt`;
`model.pt` stays the picture-dev epoch. Metrics: `split`, `selection_body_present`, `body_present_selected`,
`body_present_at_picture_epoch` (with per-trial 0/1 for paired bootstrap).

## Joint learner (`--arch joint`)

Default `--arch additive` is the path above, unchanged. `joint` needs `--pool soft`, `--body none|real`, `--episodes none`,
no `--distill-from`/`--labels`; every auxiliary loss is off (a positive `--w-*` is an error). Works with both splits.
Model `JointMIL` (subclass of upstream `RegionMIL`, width 128); the transformer settings are bounded pilot settings.

- Tokens: the window's frames x 17 regions, `vproj(v - center_mu)` (768 -> 128), and with `--body real` all 11 states of the
  body encoder (unchanged) through a Linear(128, 128). Learned modality embedding (visual / body) plus a learned
  temporal embedding shared by both (11 rows, seconds from the pair's centre frame; visual frames use their true offset).
- Transformer: shared, 2 layers, 4 heads, feed-forward 512, dropout 0 (so body tokens never change RNG use), pre-norm,
  final LayerNorm. Only the contextualised visual tokens (unit norm) are kept.
- Masks: repeated padding frames are masked. Body tokens are masked when the pair's body is dropped (p = 0.5 in training),
  the step is missing or outside the session, or the step is in another clip than the centre (so under `--split clips`
  no other partition reaches the tokens).
- Score (`JointMIL.score`, used by training and every evaluation): soft-MIL at tau_p over valid visual tokens, uniform
  prior, against the word vectors. No body-word term, no beta. Loss: the usual weighted symmetric InfoNCE.
- Control: the same model with no body modules. Common parameters get the same init as the body run for the same seed;
  body-only modules are built in a forked RNG, so the default stream and data order match.
- Evaluation: picture probe uses upstream's exact trials scored with `score` (single frame, no body); upstream's hard-max
  numbers are logged as `hard_dev/hard_test/hard_konkle`. Body-present 4AFC: `picture_soft` = no body tokens,
  `fused_soft` = the candidate's own body tokens (no `body` score). Epoch selection rules are unchanged.
- Metrics: `arch`, `joint` (settings, parameter counts total/common/body_only), `readout_epoch_30` (history record of
  epoch 30 when epochs >= 30). No `body_probe`.

## Direct-label check (LABEL DIAGNOSTIC)

Flag: `--labels` (picture-only, `--pool soft`, no `--body`/`--distill-from`). Off by default.

- Same frozen features, linear projection, soft-MIL and vocabulary (built from the speech). Only the text changes.
- Each speech pair (training sessions only, s8 excluded) keeps its frame window and gets the word of one object whose
  `visible_px > 0` in the centre frame of `frames.parquet`, drawn once, uniformly, seeded. Pairs with no visible object
  are dropped (226 of 29,960), so the pair count stays matched. A frame shows about 5 objects, so one label is one of
  several correct words and the loss cannot go to zero (same word several times in a batch).
- `metrics.json["labels"]`: pair counts and `fit` on a fixed 4096 training pairs: `top1` (drawn label among all 78
  object words, soft pooling) and `top1_visible` (any visible object).
