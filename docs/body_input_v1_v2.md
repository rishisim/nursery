# Step 3: body input for the word learner (design)

Goal: show that body signals given as inputs during training raise the ceiling on word learning, tested vision-only (picture + word) as with children.
Learner today: frozen DINOv2 regions (CLS + 4x4 = 17) -> `RegionMIL` (upstream, pinned) -> bag-of-words text tower; an utterance scores a frame by its best region, over a +-2 frame window (85 regions per pair, `FramePairs`). Eval is `eval_4afc_region` on image regions and words only.

## 1. Signal inventory (all in `<episodes>/body/<video_id>.npz`, 1 Hz, 120 rows per session, row = second)

Rule: a signal may carry what a child's own body would tell it (grasp, reach, touch). It must not carry object identity or a per-clip fingerprint.

| Signal | Shape | Leakage risk | Verdict / fix |
|---|---|---|---|
| hand_contact (bits) | 120x195 | Which vertices touch reflects object shape (rim vs handle); count reflects size | Safe if coarsened. Use per hand: any-contact and 4-level contact area. Drop the raw vertex pattern. |
| left/right_hand_pose | 120x45 each | Grasp aperture partly encodes object size. Legit proprioception, but a size cue | Moderate. Keep both hands; 8-d bottleneck + Gaussian noise in training; gate by the leakage probe (section 5). |
| joints | 120x55x3 | World coordinates fingerprint the room and clip; hand height hints at table vs floor | Moderate. Use only body + wrist joints (first 22), pelvis-centred and yaw-aligned; drop face/jaw/eye joints and absolute position. Hand joints are already in hand pose. |
| camera | 120x4x4 | Room/clip fingerprint if fed in | Geometry only: used to project joints to image regions, never an encoder input. |
| object_pose | 120x4x4 | Height and rotation of a canonical mesh reveal size/category | Risky. Not an input. Use only offline for sanity checks. |
| transl, global_orient | 120x3 each | Absolute position and heading: clip fingerprint | Risky. Excluded. |
| body_pose | 120x63 | Whole-body posture tracks the action (drink, throw), so partly identity; redundant with joints | Excluded in v1. |
| frame_idx, hand_contact_vertices | - | Time index / constant | Excluded. |

Open check before building: confirm `camera` is world-to-camera or camera-to-world and get intrinsics from the render config, by projecting wrists onto 10 frames.
Note: `frames.parquet` may skip seconds, but body has all 120. Index body by `frame_idx`.

## 2. Where the body input enters

Constraint: the body is absent at test, so it must change what the image-word mapping learns, not add a test-time feature. A body token added to the score (or a body-conditioned text tower) cannot do this: the body path can absorb the loss, and the image path may learn nothing new.

**Primary: body-guided region selection (training only).**
- `RegionMIL` already has a `region_prior` hook: a bias on which region wins the max, while the score stays the raw similarity. Extend it to a per-sample bias over the 85 window regions.
- Geometric part (fixed, not learned): project the active hand (mean of that hand's joints; the hand with contact) with `camera` + intrinsics, and set `d_r` = distance from the projection to region r's centre.
- Body part (learned): three sub-encoders, each a small MLP.
  - contact (4 scalars) -> e_c
  - hand pose (90 -> 8-d bottleneck) -> e_h
  - joints (22x3, normalised) -> e_j
  - The sum e_b = e_c + e_h + e_j gives a trust gate g in [0, g_max] (g_max ~ 3 in cosine units).
- Bias: b_r = -g * d_r / sigma. Selection is argmax(sim + b); the loss uses the raw sim at the selected region.
- Effect: in training, words are matched to the region the hand is at, so the shared `vproj` learns the object-word link from held objects. At test there is no bias and nothing changes in `eval_4afc_region`.
- The learned part is only a trust gate, so it cannot just learn to pick the best-matching region. If g collapses to 0, the cue is useless, and we see that directly.

**Alternative: auxiliary region-grounding loss.**
- A small head on the region embeddings predicts which region the hand is over (target = projected hand region, trained only on contact frames); cross-entropy added to the contrastive loss.
- Cheaper and has no selection bias, but it links to words only indirectly through the shared `vproj`, so expect a weaker effect.

## 3. Training details

- **Dropout:** per pair, drop the whole body with p=0.5 (bias = 0, i.e. plain MIL). When present, drop each sub-signal independently with p=0.3. Fixed schedule in v1. If gains are small, sweep p in {0.25, 0.75}.
- **Conditions** (3 seeds each, same epochs and data as the current runs, uniform pair weights, so this is orthogonal to the cue weights):
  1. video-only (existing baseline)
  2. body (primary)
  3. shuffled-body control: same architecture, body features taken from a random other pair in the epoch; keeps capacity, params and marginals, and removes alignment
  4. oracle: replace the projected hand point with the true held-object mask centroid from `frames.parquet` (`objects`); upper bound for any hand-derived region cue
- **Per-signal ablation:** train with a signal mask, `{contact, hand, joints, all}` and leave-one-out. A masked signal is zeroed at train time, and its sub-encoder is removed so params differ only by the dropped part. The hand-location geometry stays in all cases, so ablating contact/hand/joints tests their effect on trust only. Add a "geometry only, g fixed to 1" run to separate "where the hand is" from "what the body says".
- **Reading results:** body > shuffled ≈ video-only means the alignment is what helps. Report the same groups as now (held / seen-only / extras on dev, test, Konkle).

## 4. Minimal code change list

Upstream is pinned; subclass, do not edit it.
- New `nursery/learner/body.py` (~200 lines)
  - `load_body(episodes, video_ids)`: unpack bits, derive the 4 contact scalars, normalise joints
  - `project_hand(joints, camera, intrinsics) -> region distances [N, 17]`
  - `BodyEncoders` (three MLPs + gate)
  - `BodyRegionMIL(RegionMIL)`: override `forward_loss` and `sims` to take a bias; `region_prior` stays None at eval
- `nursery/learner/train.py::train_run` (~40 lines)
  - Precompute `body_feats[idx]` and `region_dist[idx]` aligned to `ds`, reproducing `FramePairs` window logic (frames `fi-2..fi+2` present in `lut`, padded with the first). Index them by `idx` exactly like `weights[idx]`, so dataset and collate are unchanged.
  - Apply drop masks, shuffle or oracle controls, add the new `RegionMIL` constructor, save the gate stats in `metrics.json`
- `nursery/learner/__main__.py`: flags `--body signals`, `--body-control {none,shuffle,oracle}`, `--body-drop`
- New tests in `tests/` (the project's existing test location): projection, window alignment, drop masks, eval unchanged
- Effort: ~1.5 days including the projection sanity check. Runs are small (cached embeddings, tiny head), about the same as current runs.

## 5. Risks and checks

- **Identity leakage probe (required before any result):** train a classifier from body features alone (per signal, and all, 5-frame window) to predict the held word, with cross-validation held out by clip and by subject. Pass if accuracy is within about 2x chance (1/K) for each signal. If a signal exceeds that, coarsen or add noise, or drop it. Also probe for room/clip fingerprints (predict room from body).
- **Projection sanity:** on contact frames, the projected hand region should contain the held-object mask centroid far above chance (random ≈ 1/17). Overlay 10 frames by eye.
- **Null control:** with body drop = 1.0 the model must match the video-only baseline within seed noise.
- **Order check:** oracle ≥ body > shuffled ≈ video-only. If shuffled ≈ body, the gain is capacity or noise, not alignment.
- **Gate collapse:** if mean g ≈ 0, hand location is not informative here; this is a design signal (try the alternative or richer cues), not a conclusion.
- **Contact frames only ~67% in the example session:** the bias applies only when the hand holds something; otherwise it falls back to plain MIL.
- **Train/test shift:** none by construction (eval is unbiased), but verify `eval_4afc_region` is byte-identical on the baseline path.

## 6. As built (changes forced by checks)

- **Camera:** `camera` is camera-to-world (Blender `matrix_world`: -Z forward, +Y up); intrinsics from the render config (320x240, horizontal FOV 100 deg). Hand point = mean of that hand's wrist and 15 finger joints, for the hand with more contact vertices. On 5,625 contact frames with a visible held object, the hand lands in a held-object cell 87% of the time (chance 15%). Overlay: `outputs/legacy/step3/projection_check.jpg`.
- **Region grid is the centre crop:** the pinned embedder uses DINOv2's processor (short side to 256, centre crop 224), so the 4x4 grid covers only x in [55, 265], y in [15, 225] of the render. Region centres and distances use that crop. `objects[].cells` in `frames.parquet` is on the full-image grid, so the oracle converts those cells to pixel boxes.
- **Distances:** cell units, clipped at 3. CLS gets the mean grid distance; frames with no hand in contact get a neutral 1.5 for all regions.
- **Gate gradient:** hard argmax gives the gate no gradient. Training uses a straight-through soft selection (tau 0.05) that feeds the gate only; the forward value is still the raw sim at the argmax region.
- **Leakage probe** (MLP on 5-frame window means, 56,782 contact windows, 50 held words; 1/K = 2.0%, majority class 6.9%; room chance 14.3%). Pass = no better than about 2x the majority rate.

| Signal | Word, held out by clip | Word, held out by subject | Room, by subject | Action |
|---|---|---|---|---|
| contact (4 scalars) | 6.6% | 8.0% | 14.8% | keep |
| hand pose, full 90 | 24.4% | 18.0% | 14.4% | fail: replaced |
| hand pose, PCA-8 | 11.4% | 9.5% | - | fail: replaced |
| hand pose, grip closure (2 scalars) | 6.2% | 6.6% | 14.4% | keep |
| joints, 22 body joints | 9.1% | 6.6% | 14.2% | replaced |
| joints, 2 wrists only | 4.8% | 5.1% | 14.3% | keep |
| hand location (geometry) | 5.4% | 6.0% | 14.2% | keep |
| contact + closure + wrists | 14.7% | 13.7% | 14.4% | fail: add noise |
| same, plus N(0,1) noise on standardised inputs | 8.2% | 8.8% | - | keep |

  Built: contact 4 scalars, hand pose reduced to 2 grip-closure scalars (the 8-d bottleneck of section 2 still leaked), joints reduced to the two wrists (pelvis-centred, yaw-aligned), N(0,1) input noise in training. Room is at chance for every signal.
- **Null control:** with `--body-drop 1.0` the body encoders are built under a forked RNG, so the default stream matches video-only training; the first attempt without this differed from `none_s0` by 2.5 points from RNG shift alone (a rough seed-noise measure).
- **Smoke result (seed 0, 20 epochs):** see the run table in the report; the gate collapsed to about 0 in the body and shuffled runs.
