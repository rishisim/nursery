# ChildLens feasibility v1: storage, compute, and annotation projection

Date: 2026-07-21  
Status: `NONSCIENTIFIC_CAPACITY_SCENARIOS`  
Scope: feasibility planning only; no learner training or acquisition outcome was run

## Bottom line

A metadata-first, selectively acquired ChildLens pilot can fit on the current host if exact file bytes are preflighted and capped before download. The base planning allocation peaks at about 69.4 GiB and leaves about 52.9 GiB of the 122.3 GiB currently free. The high allocation peaks at about 90.0 GiB, exceeds the 73 GiB namespace cap, and would leave only about 32.3 GiB; it is therefore not admissible on the current disk without reducing scratch duplication, shrinking the acquired shard, or adding separately authorized storage. The full live package must not be downloaded.

These are capacity scenarios, not ChildLens measurements. Video duration, byte size, preprocessing throughput, model shape, update budget, and annotation speed must be replaced by frozen, measured values before a pilot or later outcome launch. No result in this document establishes lexical coverage, annotatability, or scientific feasibility.

## Read-only host audit

The following checks read system state only. They did not enumerate or open restricted media.

| Item | Observed value | Source and limitation |
| --- | ---: | --- |
| Free space on data volume | 128,260,468 KiB = 122.3 GiB | `df -k` on the workspace path, 2026-07-21; shared volume and therefore time-varying |
| Unified memory | 34,359,738,368 bytes = 32 GiB | `sysctl -n hw.memsize` |
| Logical / physical CPUs | 10 / 10 | `sysctl`; concurrency is capped below this hardware count |
| PyTorch | 2.13.0 | local environment probe |
| MPS | built and available | local environment probe; one trainer globally |
| CUDA | unavailable, zero devices | local environment probe |

The existing construction readiness report's tiny fixture throughput is not used to predict scientific runtime: it excluded backward/optimizer/data loading and used arbitrary 64-pixel, four-frame construction shapes.

## Sanitized live-package receipt and count uncertainty

The authenticated sanitized Keeper receipt reports 192 MP4 files in the current live UI. Displayed sizes are available for 191 files and sum to approximately 201.33 decimal GB; one MP4 has no displayed size. The share/package is not pinned to a version identifier. The missing-size file cannot be selected or downloaded until its exact byte size is resolved through read-only metadata.

The public paper describes the full corpus as 354 video files and 108.58 hours. The 192-file live package may be an annotated subset, but that is an inference, not a versioned release fact. Its relationship to the published full corpus remains unresolved. The retired preliminary count is not used. Public hours and live-package bytes are not divided to estimate pilot size because they may describe different inventories. The governance manifest must pin or locally fingerprint the accessible package and reconcile the difference before acquisition.

Selection is governed by the exact displayed byte sum for the predeclared metadata-stratified files. Count averages, duration averages, and the public-paper total are not download estimators.

## Pilot capacity scenarios

The 12--18-video range follows the predeclared feasibility-pilot target. Minutes per video and raw storage allocations below are independent planning brackets chosen before content inspection, not observed ChildLens durations or file-size estimates. Exact selected bytes supersede the allocation and must remain within the applicable scenario and hard caps.

| Scenario | Videos | Planning minutes/video | Planning media hours | Raw storage allocation |
| --- | ---: | ---: | ---: | ---: |
| Low | 12 | 12 | 2.4 | 8 GiB |
| Base | 15 | 18 | 4.5 | 15 GiB |
| High | 18 | 30 | 9.0 | 20 GiB |

Before any byte transfer, use the release manifest to sum the exact selected file sizes. Reject a selection containing the one file whose size is unresolved. Selection remains metadata-stratified and outcome-blind. Enforce all of these controls:

1. confirmed data-use permission for local processing and restricted derivatives;
2. maximum 20 GiB of raw pilot payload;
3. maximum 73 GiB peak ChildLens-feasibility namespace footprint;
4. at least 50 GiB free on the shared volume after projected peak allocation;
5. fail closed if the exact selected bytes or temporary expansion factor are unknown; and
6. acquire one resumable shard at a time, verify its checksum, then release only disposable temporary material before the next shard.

No workflow should materialize all decoded frames. Decode bounded windows on demand; keep raw media and every restricted derivative in the restricted corpus-instance namespace.

## Peak storage projection

The estimates include a fixed 8 GiB allowance for versioned measurement-instrument weights/environments, the nonempirical raw allocation above, restricted demux/proxy derivatives at 25% of raw, scratch at 2x raw, annotation/manifests, and a 20% contingency on the subtotal. Instrument weights are quarantined measurement tools only; they are not learner ancestors.

| Component (GiB) | Low | Base | High |
| --- | ---: | ---: | ---: |
| Fixed measurement tools/environment | 8.0 | 8.0 | 8.0 |
| Raw selective shard | 8.0 | 15.0 | 20.0 |
| Restricted audio/proxy/ASR derivatives | 2.0 | 3.8 | 5.0 |
| Bounded scratch and temporary decode | 16.0 | 30.0 | 40.0 |
| Manifests, annotations, QA, logs | 0.5 | 1.0 | 2.0 |
| Contingency (20%) | 6.9 | 11.6 | 15.0 |
| **Projected peak** | **41.4** | **69.4** | **90.0** |
| Free space remaining from current 122.3 GiB | 80.9 | 52.9 | 32.3 |
| Current-host admission | PASS | PASS, 2.9 GiB headroom above floor | **FAIL: namespace cap and free-space floor** |

The 191 displayed live-package sizes already sum to approximately 201.33 decimal GB, with one additional unknown-size MP4. The full package cannot fit and is prohibited by scope regardless of nominal capacity. The base allocation has little capacity headroom and therefore requires a fresh preflight immediately before transfer. The high case may be recovered by streaming bounded decodes so scratch is at most 1x raw, reducing the raw subset from metadata before content inspection, or provisioning authorized storage; it must be recalculated rather than silently overrunning either cap.

## Preprocessing and measurement-instrument compute

These worker-hour brackets cover checksumming, demux/resampling, bounded window extraction, instrument inference, packaging, and validation. They are deliberately not throughput claims. A fixed, content-blind benchmark shard must measure wall time and memory before scaling; benchmark configuration and tool versions freeze before processing the rest of the pilot.

| Scenario | CPU worker-hours assumption | Wall time with 2--4 CPU workers | ASR/diarization instrument accelerator-hours | Visual measurement accelerator-hours | Total instrument accelerator-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | 1x media = 2.4 | 0.6--1.2 h plus I/O | 0.5x media = 1.2 | 0.25x media = 0.6 | 1.8 |
| Base | 3x media = 13.5 | 3.4--6.8 h plus I/O | 1.5x media = 6.8 | 1x media = 4.5 | 11.3 |
| High | 6x media = 54.0 | 13.5--27.0 h plus I/O | 4x media = 36.0 | 3x media = 27.0 | 63.0 |

Only the controller may create CPU workers, with an initial limit of two and a hard maximum of four after measuring resident memory and I/O pressure. Each worker and every numeric library gets one thread. Instrument inference and later learner training must not contend for MPS; the local host runs at most one accelerator process at a time.

ASR, diarization, and vision outputs are candidate annotations, never gold. Their model/version/checksum and settings must be recorded. Humans fluent in the established corpus language must validate or correct utterance timing, text, and child-versus-other speaker role; humans must also validate visible candidate/null/ambiguity/action labels. Instrument embeddings or features never enter the learner.

## Manual annotation labor

Labor is projected from media duration so it can be replaced after the frozen pilot measures real annotation speed. The estimate assumes a usable annotation interface and annotators fluent in the observed language. It includes protocol training, transcript/timing/speaker correction, referential object/action/null/ambiguity and lag coding, mandatory reliability duplication, and adjudication.

| Scenario | Protocol setup/training | Transcript, timing, speaker correction | Referential/event annotation | Reliability duplication + adjudication | Total person-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | 16.0 | 3x media = 7.2 | 5x media = 12.0 | 30% of coding = 5.8 | **41.0** |
| Base | 24.0 | 5x media = 22.5 | 8x media = 36.0 | 40% of coding = 23.4 | **105.9** |
| High | 40.0 | 8x media = 72.0 | 12x media = 108.0 | 50% of coding = 90.0 | **310.0** |

Reliability work is not an optional reserve. The frozen annotation protocol must specify the double-coded fraction, unitization/timing tolerance, chance-corrected categorical agreement, continuous timing agreement, adjudication rules, and acceptance thresholds before judging the pilot. If qualified language-matched annotators are unavailable, machine-only transcripts cannot substitute; labor and feasibility must be revised.

## Later learner training and evaluation capacity

No learner run was performed. The scenarios below are transparent capacity envelopes for the proposed compact temporal CLIP+ family. They do not authorize update counts, seed counts, stopping rules, or model selection. A later protocol must freeze those values before any outcome is exposed.

Every bundle contains all five arms and executes them serially on one device. `total_updates = bundles * 5 * updates_per_arm`. The provisional time model is `device_hours = total_updates * seconds_per_update / 3600`; local elapsed time includes a 20% allowance for evaluation, checkpoint verification, and orchestration.

| Scenario | Complete paired bundles | Updates/arm | Total arm runs | Total updates | Planning sec/update | Accelerator device-hours | Local elapsed with 20% overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 4 | 10,000 | 20 | 200,000 | 0.25 | 13.9 | 16.7 h |
| Base | 8 | 25,000 | 40 | 1,000,000 | 0.75 | 208.3 | 250.0 h (10.4 d) |
| High | 12 | 50,000 | 60 | 3,000,000 | 1.50 | 1,250.0 | 1,500.0 h (62.5 d) |

Seconds per update are planning assumptions, not a benchmark. Before outcome launch, run a fixed-shape, fixed-step engineering benchmark that does not compute or expose an arm contrast. Freeze measured throughput, peak memory, batch/clip shape, checkpoint size, and evaluation cost. If the frozen base design resembles the base envelope, one local MPS trainer is operationally possible but slow; identical CUDA GPUs running whole bundles are the appropriate wall-time reduction. GPU parallelism changes scheduling only, never the paired causal unit.

Evaluation is separately provisioned below. The inventory (noun/object, verb/action, strong-ceiling and weak-baseline checks) must be frozen before launch; these figures are allowances rather than an inferred number of ChildLens trials.

| Scenario | Evaluation CPU worker-hours | CPU wall time with 2--4 workers | Evaluation accelerator-hours | Checkpoint/report scratch allowance |
| --- | ---: | ---: | ---: | ---: |
| Low | 8 | 2--4 h | 1 | 2 GiB |
| Base | 32 | 8--16 h | 4 | 8 GiB |
| High | 96 | 24--48 h | 12 | 20 GiB |

Evaluation scratch is a later outcome-task allowance and is not added to the feasibility-pilot storage table. The later storage preflight must add it to measured learner checkpoint sizes and reapply the 50 GiB free-space floor. Evaluation parallelism operates across immutable examples/checkpoints only; canonical scoring is deterministic and merges a complete expected inventory.

For `B` bundles on `G` identical GPUs, each with `U` updates per arm and measured `s` seconds/update, the no-contention elapsed estimate is `ceil(B/G) * 5 * U * s / 3600`, plus frozen overhead. Evaluation uses immutable checkpoints and may use two to four CPU workers; it must not trigger outcome-dependent retries, extra seeds, or budget changes.

## Unresolved quantities required before launch

- exact accessible release identity, file inventory, and selected-shard bytes;
- terms governing local derivatives, transcripts, model-instrument use, retention, and aggregate export;
- actual pilot media duration and audio/video codec properties;
- language-matched annotator availability and measured correction/annotation rates;
- instrument model versions, memory, throughput, cache size, and failure rate;
- final compact learner shape, clip/window policy, batch size, checkpoint size, and seconds/update;
- frozen bundle/seed count, update budget, stopping rule, and evaluation inventory; and
- authorized storage location, encryption/access controls, retention, and deletion procedure.

No AEA or BabyView data, measurements, distributions, vocabularies, tokenizers, weights, results, or other empirical artifacts informed this projection.
