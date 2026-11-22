# Learner implementation reference

The scientific question and next studies live in [PLAN.md](../PLAN.md).
This document describes existing code, not a new experiment protocol.
Completed comparisons and historical settings are in
[Past experiments](../archive/language_learning/past_experiments.md) and the
[experiment record](../archive/language_learning/experiments.md).

## Which files implement which learner?

| File | Role |
|---|---|
| [video_only.py](../nursery/learner/video_only.py) | Frozen image features → trainable visual projection → word matching. |
| [additive.py](../nursery/learner/additive.py) | Inherits the visual baseline; adds a learned body–word score. |
| [body.py](../nursery/learner/body.py), [windows.py](../nursery/learner/windows.py) | Body features, normalization, sequence encoding and bounded context. |
| [gaze.py](../nursery/learner/gaze.py), [gaze_data.py](../nursery/learner/gaze_data.py) | Step 2 gaze readouts and frame-aligned point/compact close-up inputs. |
| [losses.py](../nursery/learner/losses.py) | Shared pooling and loss functions. |
| [embed.py](../nursery/learner/embed.py), [train.py](../nursery/learner/train.py), [evaluate.py](../nursery/learner/evaluate.py) | Shared feature extraction, training and evaluation. |

Frozen DINOv2 region features feed a trainable visual projection and word
vectors from the pinned upstream `RegionMIL`. Soft pooling (`--pool soft`,
temperature 0.1) combines region evidence. The additive score is:

`visual–word score + learned weight × body–word score`

New embeddings retain the full 16×16 patch grid plus CLS (257 tokens per
frame), using the same image crop and frozen encoder. `embed --grid 4`
reproduces the historical 17-token caches. Use a fresh ignored output root
when changing grids; existing caches are never converted in place. Large
caches use upstream's deterministic shards and a disk-backed merge, preserving
frame order without holding the whole cache in RAM. Training and picture
caches must use the same grid. The archived width-128 joint pilot remains
restricted to its original 4×4 caches.

The preserved acquisition runner is integrated here. Its `prepare` command
accepts `--fractions` and `--conditions` to freeze a selected comparison;
`timing`, `run` and `report` require that complete selection. Without those
options, the original five fractions and three learner conditions remain the
default. [Step 1](step_1_visual_baseline.md#execution-protocol) uses full-data video
only with the original three seeds.

The language objective trains the visual projection, word vectors, body encoder
and body projection together. Body input is dropped on half the training pairs
by default. This is simultaneous language learning with late score fusion.
The proposed joint interaction module is not implemented here. The old
[width-128 joint pilot](../archive/code/learner/joint_pilot.py) remains reachable
through the explicitly historical `--arch joint` option for compatibility.

## Child gaze implementation (selection pending)

Step 2 has two implemented, unselected readouts in `gaze.py`. `additive`
adds a separate point/close-up word score with a learned scalar; `steering`
uses the point as a Gaussian prior over video patches and adds one sharp-view
candidate per frame through the shared visual projection. CLS keeps its
relative mass within the video bank, and a point outside the DINO crop does
not steer visible patches.
The initial weight and spatial width are implementation defaults, not a
development-selected protocol.

`GazeData` reads frame keys, normalized image points and frozen image features.
It maps points through the pinned 256-pixel resize and 224-pixel centre crop,
aligns them to the existing five-frame visual window, and masks padding and
missing gaze. A close-up is one 768-value mean of DINO's 256 patch features,
excluding CLS, computed before the trainable projection. The adapter can read
Phase A's full cache without modifying it; only selected compact features are
retained in memory. Point-only input does not load a close-up cache.

The shuffled control swaps point and close-up windows together with another
centre moment from the same partition and selected experience. Donor rows
are deterministic and recorded. Neither semantic object IDs nor gaze/held
categories enter the adapter or model. With gaze removed (`ctx=None`), either
readout uses exactly its visual score. Fixed-panel evaluation records both
modes at the same selected checkpoint; picture transfer always removes gaze.

`acquisition.train_one` has explicit `video_gaze`, `video_shuffled_gaze` and
`video_gaze_point` hooks, alongside `video_only`. They require explicit gaze
option, point and partition-map paths, dropout and shuffle seed; full gaze also
requires the close-up cache. Scientific execution requires those inputs to be
hashed in the frozen run. Historical preparation, timing and matrix defaults
are unchanged. `child_gaze.py` now provides new-world preparation, a fixed development-only
selection rule, a four-hour pilot gate, protocol freezing and matrix timing.
It never launches rentals. Phase A outputs are validated; readout selection
remains pending the pilot. Child-gaze training gathers bounded float16 batches
and casts to the original float32 inputs on the accelerator, preserving frame
order, padding, sampler order and optimizer settings. Historical loaders are
unchanged. See [the Step 2 protocol](step_2_child_gaze.md#phase-b-execution-protocol).

## Body input and timing

- Contact: 22 anatomical hand patches, defined by `hand_patches.json`.
- Fingers: 90 local rotation features, both hands.
- Arms: 27 camera-relative position, movement and gravity features.
- With `--physical <directory>`: additional load and skin groups from saved
  modeled physical signals. These are optional inputs, not inferred force measurements.

Each group has an MLP. A bidirectional GRU summarizes an approximately
11-second window (five seconds before and after the utterance). It resets at
clip boundaries in both directions. Missing seconds are masked; normalization
uses training data only. Object identity, asset IDs and held-category labels
are not ordinary body inputs. There is no persistent state across examples.

Speech remains intentionally weakly aligned, including delayed and nonheld
names. Rounded Gaussian timing jitter is clamped to the contiguous source clip;
it cannot cross a cut. Source time, raw jitter and effective jitter are recorded.
Held status is recomputed at the aligned frame. A held-channel name need not
still be supported by a holding event inside the aligned body window: that
residual uncertainty is retained rather than filtered away.

## Shared evaluation and splits

Body-present understanding and picture-only transfer are separate measurements.
Removing body input from a multimodal model does not replace training a matched
video-only baseline.

Body-present four-choice trials require a visible target at the aligned frame.
Foils come from other sessions of the same subject and evaluation partition,
have a different referent, and exclude the target from both their accessible
visual and contiguous-clip body windows. There is no looser fallback. Rows
without three eligible foils are skipped; an empty requested split is rejected.
Correct slots are seeded and randomized, and controls share the same trials.
Renderer visibility does not guarantee recognizability after image cropping;
this remains a situation-matching diagnostic, not broad sentence comprehension.

The default subject split excludes s8 from training. `--split clips` additionally
holds out 15% dev and 15% test source clips of training subjects; all replays of
a source clip stay together. Cross-partition visual windows are dropped, and
body windows stop at clip boundaries. `model_bodypresent.pt` is selected on
body-present dev; `model.pt` remains selected on picture-only dev. Test scores
do not choose checkpoints. Read `picture_soft` / `fused_soft` for soft-pooling
comparisons; older hard-pooling and pre-repair trial metrics are not interchangeable.

## Historical options retained in the shared runner

The cleanup preserves numerical behavior, checkpoint parameter names and
defaults. It does not retune the learner or silently change old run settings.

- `--pool max` remains the historical default; body input requires `--pool soft`.
- `--w-pred` defaults to 0.5: an auxiliary visual-to-body prediction loss.
  Use `--w-pred 0` for the direct-language-only comparison in the study design.
- Episode/temporal bags, consistency, masked-body and body-only matching losses
  remain optional, off by default. `--body shuffled`, `--body oracle`,
  `--labels` and oracle distillation are historical diagnostics, not study defaults.
- Completed heavy/light and sensory-pretraining runners are archived and no
  longer exposed as current learner subcommands. Their original runnable package
  layout is preserved in Git; see [the archive](../archive/README.md).

The [speech implementation](../nursery/speech/transcripts.py), shared evaluator
and training tests cover weak alignment, strict foil selection, gradient flow,
body dropout, clip resets, split isolation and checkpoint selection. The
[README](../README.md#local-data-and-verification) lists the validation commands.
