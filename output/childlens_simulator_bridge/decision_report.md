# ChildLens-calibrated simulator bridge decision

**Decision: `SIMULATOR_CALIBRATION_READY`**

The frozen contract was evaluated with 180 one-minute synthetic episodes per
alignment regime (60 per each of seeds 2026072501, 2026072502, and 2026072503).
All six authorized empirical inputs matched their frozen SHA-256 values. No
locked participant, media, transcript, external child corpus, empirical
vocabulary, learner, or private row-level score was accessed.

## Aggregate distribution match

The model-independent generator is identical across regimes except for the
predeclared weak audiovisual timing process, so the following pooled results
apply separately to `weak_lower`, `weak_central`, and `weak_upper`; each regime
passed every frozen metric gate. “Synthetic” is pooled mean (episode SD), and
SMD is the absolute discrepancy standardized by the empirical 90% interval.

| Metric | ChildLens target (90% interval) | Synthetic | SMD | Gate |
|---|---:|---:|---:|---|
| Speech bout duration, s | 11.1865 [7.4267, 15.3812] | 11.0776 (1.0854) | 0.0450 | PASS |
| Speech seconds / observation min | 38.9448 [33.8388, 43.8644] | 38.9444 (1.3721) | 0.0001 | PASS |
| Speech bouts / min | 5.5000 [4.2963, 6.7222] | 5.4556 (0.4980) | 0.0603 | PASS |
| Speech gap, s | 4.5655 [3.2214, 6.3762] | 4.5751 (0.4757) | 0.0100 | PASS |
| Candidate events / min | 11.2926 [11.2481, 11.3333] | 11.3000 (0.4583) | 0.2858 | PASS |
| Motion | 0.1185 [0.1095, 0.1273] | 0.1198 (0.0027) | 0.2414 | PASS |
| Adjacent-frame persistence | 0.7124 [0.6912, 0.7330] | 0.7123 (0.0059) | 0.0073 | PASS |
| Scene-change rate | 0.1310 [0.1069, 0.1558] | 0.1317 (0.0072) | 0.0495 | PASS |
| Audio log RMS | -3.4167 [-3.6035, -3.2362] | -3.4021 (0.0492) | 0.1306 | PASS |
| Audio non-silent fraction | 0.9231 [0.9026, 0.9427] | 0.9231 (0.0055) | 0.0024 | PASS |
| Audio clipped fraction | 0 [0, 0] | 0 (0) | 0 | PASS |

All episode values stayed inside frozen supports. Seed-mean ranges passed the
frozen stability rule. The representative natural activity mixture passed with
total variation 0.002883 (gate 0.08); it was not replaced by the
grounding-enriched conditional stratum. Motion was nonconstant, scene-change
states both occurred, and mean lag-1 motion autocorrelation was 0.995981.

The separately generated grounding-enriched conditional stratum also passed:
adjacent-frame persistence was 0.809506 versus 0.8091 [0.7885, 0.8301], audio
log RMS was -3.759845 versus -3.7568 [-4.1160, -3.4335], motion was 0.093243
versus 0.0925 [0.0782, 0.1072], and released speech support was 0.692241 versus
0.6904 [0.5882, 0.7802]. These results remain conditional and do not substitute
for the representative natural mixture.

## Alignment sensitivity and modality integrity

The process contrasts were exactly 0, 0.00177, and 0.00620 for `weak_lower`,
`weak_central`, and `weak_upper`. Negative sampling noise was truncated at
physical zero. Their observable mean speech-timing shifts from `weak_lower`
were ordered at 0, 0.00487743, and 0.01708479 seconds. These are sensitivity
conditions derived from the repaired primary contrast interval, not easy /
medium / hard conditions and not passed naturalistic alignment estimates. The
unresolved 0.03713 lag-curve-amplitude upper bound remains a sensitivity cap;
it still exceeds the immutable 0.02 weak/flat gate.

RGB, simulator-native speech/audio, activity/context, action, proprioception,
contact, IMU, and synchronization schemas were present and reproducible.
Evaluation-only word/referent relations were separately withholdable. Automated
checks found no lexical target or referent in side streams, no episode-ID or
filename encoding, no constant side-channel encoding, and no clock mismatch.
No raw or restricted empirical content appears in the synthetic schema or
curated artifacts.

Exact owner-private V5 scorer weights were not available through the authorized
tracked inputs. Consequently, real/time-shifted/shuffled synthetic score
matching is **unsupported**, not passed, and apples-to-apples model-score
equivalence is not claimed.

## Claim boundary and next task

Readiness supports only an exploratory synthetic cue-lift prototype calibrated
to selected privacy-safe aggregate properties of ChildLens ages 3–5. It is not
infant calibration, validated infant learning, naturalistic German lexical
grounding, causal cue lift, or successful real-world audiovisual calibration;
the immutable V1–V5 and support-repair terminal records remain unchanged.

The next separately authorized task should freeze and run one apples-to-apples
learner across absent, synchronized, shuffled, time-shifted, and uninformative
side-cue conditions with matched data and compute, withholding every side cue
at final spoken-word/object grounding evaluation.
