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
| [interaction.py](../nursery/learner/interaction.py) | Extends additive matching with the zero-initialized visual–body interaction. |
| [body.py](../nursery/learner/body.py), [windows.py](../nursery/learner/windows.py) | Body features, normalization, sequence encoding and bounded context. |
| [losses.py](../nursery/learner/losses.py) | Shared pooling and loss functions. |
| [embed.py](../nursery/learner/embed.py), [train.py](../nursery/learner/train.py), [evaluate.py](../nursery/learner/evaluate.py) | Shared feature extraction, training and evaluation. |

Frozen DINOv2 region features feed a trainable visual projection and word
vectors from the pinned upstream `RegionMIL`. Soft pooling (`--pool soft`,
temperature 0.1) combines region evidence. The additive score is:

`visual–word score + learned weight × body–word score`

The language objective trains the visual projection, word vectors, body encoder
and body projection together. Body input is dropped on half the training pairs
by default. This is simultaneous language learning with late score fusion.
The joint interaction preserves both direct pathways and adds a learned
body-to-visual attention term, as specified in the
[acquisition protocol](acquisition_curves.md#frozen-execution-protocol). All three
learners expose `score_encoded` so acquisition training and fixed-panel evaluation
use the same scoring arithmetic. The `acquisition` CLI prepares subsets, runs
the matrix and reports the results with separate dev-selected checkpoints.
The old
[width-128 joint pilot](../archive/code/learner/joint_pilot.py) remains reachable
through the explicitly historical `--arch joint` option for compatibility.

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
