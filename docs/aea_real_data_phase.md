# AEA real-data phase: protocol and operations

## Scientific scope

Aria Everyday Activities (AEA) is used here because it combines head-mounted
RGB, speech-to-text, and synchronized inertial sensing in naturalistic indoor
activity recordings. The participants are adults and the scenarios are partly
scripted. AEA is therefore a sensor-format analogue for the Nursery/BabyWorld
method, not developmental evidence, an infant sample, or a BabyView match.

The locked question is:

> Does correctly synchronized six-axis head IMU during training improve
> video-language action grounding when IMU is withheld at test?

The primary estimand is action-balanced 2AFC accuracy for synchronized training
minus split-local, whole-performed-event-shuffled IMU training. Null and time-shifted
arms are additional controls. Every arm uses the same architecture,
initialization, seeded batch order, optimizer schedule, and number of updates.
The exact architecture, sample counts, batch size, learning rate, motor weight,
time shift, and eight-epoch schedule are locked in `configs/aea_real.yaml`; a
run with any mismatch cannot be labeled a primary finding.

## Official resources

- [AEA overview](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset)
- [Getting started](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_getting_started)
- [Dataset download](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_download_dataset)
- [Data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_everyday_activities_dataset/aea_data_format)

The official release reports 143 recordings in five locations and more than
7.5 accumulated hours. Its link manifest expires after 14 days. The complete
release is too large for the current disk budget, and raw VRS alone is about
287.7 GiB in the supplied manifest.

## Acquisition policy

The user must personally accept the AEA license and provide the expiring link
JSON. Repository tooling does not submit forms, accept terms, serialize signed
links, or print URL-bearing exceptions. The link JSON remains outside the
repository.

`configs/aea_subset_40.yaml` locks a 40-recording subset with eight recordings
per location and eight per script. Where concurrent pairs are present, they are
kept together. Selection uses only:

1. predeclared action-form counts from the tiny `speech.csv` annotations;
2. location/script/recording identifiers;
3. declared component sizes.

No VRS pixels or inertial values are inspected during selection. The locked
subset contains 347 predeclared action anchors and has a raw VRS plus annotation
budget of about 106.3 GiB. MPS eye gaze, trajectories, point clouds,
calibration, and artifacts are excluded initially. The downloader keeps a 40
GiB free-space reserve, verifies declared size and SHA-1, supports partial-file
resume, and suppresses signed URLs on errors.

Safe annotation-first workflow:

```bash
python scripts/download_aea_subset.py \
  --links "$HOME/Downloads/AriaEverydayActivities_download_urls.json" \
  --plan configs/aea_subset_40.yaml \
  --out data/aea_raw \
  --components annotations \
  --all-annotations \
  --confirm-license-accepted-by-user

python scripts/plan_aea_subset.py \
  --links "$HOME/Downloads/AriaEverydayActivities_download_urls.json" \
  --annotations-root data/aea_raw \
  --out configs/aea_subset_40.yaml
```

Then acquire only the locked raw VRS recordings:

```bash
python scripts/download_aea_subset.py \
  --links "$HOME/Downloads/AriaEverydayActivities_download_urls.json" \
  --plan configs/aea_subset_40.yaml \
  --out data/aea_raw \
  --components main_vrs \
  --confirm-license-accepted-by-user
```

## Window and label construction

Each example is a locked six-second window centered on an ASR action-form
anchor. Up to eight upright RGB frames are queried in device time. Both the
accelerometer and gyroscope from `imu-left` are read in their native SI units
and jointly linearly resampled to a fixed 50 Hz device-time grid. A window is
rejected if finite six-axis data do not cover at least 98% of the grid or if a
raw gap exceeds the locked tolerance.

The text input is the ASR transcript overlapping the same window. The action
target is the canonical form of the anchor token. The optional object target is
the nearest predeclared object token within five seconds. These are noisy
lexical labels—not human action annotations. This limitation is persisted in
every example, audit, and report.

## Leakage controls and evaluations

- All windows from one recording stay together, and concurrent recordings of
  the same performed activity share one donor group.
- Concurrent views sharing location+script+sequence are grouped or the partner
  view is purged.
- Shuffled IMU is a permutation of whole trajectories inside the training
  split, with no self donor and no donor from the same performed-event group
  (including concurrent partner recordings).
- The primary evaluator constructs batches with only RGB and text. It neither
  loads an IMU array nor calls the motor encoder.
- Held-out location uses leave-one-location-out folds. Location 5 is marked as
  limited to scripts 4 and 5.
- Held-out wearer-session uses the release-visible
  location+script+recording field. AEA does not expose a persistent identity,
  so reports call this a proxy and purge concurrent partner events.
- Held-out composition removes every event containing a locked action-object
  pair, verifies that the exact pair is absent from training, and requires the
  action and object to occur separately in training.

Locked composition probes are `get coffee`, `cook egg`, and `put dish`, chosen
from annotation-only support before VRS inspection.

## Positive controls and manipulation checks

1. Annotation/timestamp integrity: the action surface form must occur in its
   own transcript window. This is not visual-grounding evidence.
2. Optimization sanity: first- and last-epoch losses are retained per arm.
3. Motor manipulation: synchronized, shuffled, shifted, and null trajectory
   statistics and motor-only nearest-centroid action accuracy are retained.
4. Donor integrity: the shuffled self-match and same-episode-group match rates
   must both be zero.
5. Shortcut diagnostic: an object/metadata-only action baseline is reported
   next to the learner.
6. Pairing: initialization digest, optimizer steps, and batch-order digest are
   compared across all arms for every seed.

## Inference and honest reporting

The primary interval uses a hierarchical bootstrap over held-out-location fold
units and then paired seeds within folds. Secondary wearer-session and
composition families receive separate intervals. The claim gate requires:

- primary synchronized-minus-shuffled mean of at least +3 absolute accuracy
  points;
- hierarchical 95% CI above zero;
- every leakage audit passing;
- zero same-performed-event-group shuffled donors;
- a non-smoke run;
- the complete locked fold, seed, and training configuration.

Null, mixed, or negative results are reported without changing the endpoint.
The generated `aea_report.md` is organized for an eventual update to Professor
Frank and always distinguishes:

- implemented infrastructure;
- synthetic schema smoke results;
- preprocessed real-data coverage;
- real-data effect estimates.

`scripts/build_aea_report_artifact.py` also converts a result and preprocessing
audit into a canonical bounded report payload with a held-out-location lift
chart, exact family/audit tables, source notes, and explicit technical-report
section mapping. It can be packaged as one self-contained HTML file by the
Codex portable report builder.

The current smoke fixture is synthetic and exists only to exercise the AEA
schema, six-axis encoder, four arms, paired bootstrap, report writer, and hard
motor omission at test.
