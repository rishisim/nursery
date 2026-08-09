# ChildLens v1.1 frozen-feasibility rubric application

Date applied: 2026-07-21  
Controlling rubric: `childlens-feasibility-rubric-v1.0.0`  
Controlling pilot: `childlens-lexical-feasibility-pilot-v1.0.0`  
Scientific outcome authorized: no

## Immutability statement

Version 1.1 does not replace, edit, or loosen the frozen v1 rubric or pilot. This document is a release-specific application receipt. The original decision logic, thresholds, one-corpus boundary, privacy constraints, and STOP/REVISE/GO meanings remain controlling. The v1 artifact baseline contains 23 artifacts with sorted path-and-SHA-256 set digest `35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf`.

Decision logic remains:

- `GO`: every essential gate passes on release-specific evidence; promises, public-paper aggregates, unvalidated ASR, or simulator substitution cannot pass a gate.
- `REVISE`: no fundamental STOP condition is established, but at least one essential gate is conditional or unresolved and has a bounded ChildLens-only correction.
- `STOP`: an essential gate fails irreparably without corpus mixing, prohibited ancestry, material reframing, or a terms/privacy violation.

No threshold, sample rule, endpoint, or status was changed after content inspection. No media content or learner result was inspected in this continuation.

## Frozen gate application

| Gate | Frozen pass requirement | v1.1 status | Fixed next evidence |
| --- | --- | --- | --- |
| G1 terms | Release-specific terms support local read-only processing, restricted transcripts/annotations, and nonidentifying aggregate export. | PASS WITH CONTROLLING CONSTRAINTS | Continue only inside the signed project's secure, noncommercial, nonshared, aggregate-only, citation, and July 2027 time limits. |
| G2 release/grouping | Pin the accessible release with library/share identity plus a canonical nonidentifying manifest; retain participant/session grouping internally. | CONDITIONAL | Replace the placeholder sizes with exact selected remote bytes, complete media/annotation linkage and checksums, and emit the final restricted-manifest digest. Do not infer live/snapshot byte equivalence. |
| G3 audio/timing | At least 80% of sampled annotated-video duration decodes; at least 70% of speech-bearing sampled windows have usable manually corrected utterance boundaries. | UNRESOLVED | Run the frozen authorized media pilot and genuine human timing validation. Synthetic decode readiness is not empirical evidence. |
| G4 transcript/role | Establish actual language; on the first 300 utterances or 30 speech minutes, require five-way role validation with macro-F1 at least 0.80, bounded corrected text error, provenance, and at least 20% blinded double annotation. | UNRESOLVED | Obtain language-matched human text/timing/role judgments. Raw ASR is never gold; blocked automated role tools remain excluded. |
| G5 input lexicon | At least 8 noun/object and 6 verb/action types, each meeting frozen train/visible/held-out support across participant groups. | UNRESOLVED | Audit only fully human-dispositioned non-child ChildLens-local text after the splits freeze; export counts/bands, not token strings. |
| G6 referential annotation | At least 15% of validated non-child utterances have visible candidates; retain null/ambiguity; status agreement alpha or equivalent at least 0.67; double-code at least 20%. | UNRESOLVED | Apply the six-state ontology and ±5-second window with genuine blinded humans, clustering-aware uncertainty, and the single allowed bounded remediation cycle. |
| G7 held-out evaluation | Participant-disjoint primary and session-disjoint sensitivity splits preserve frozen support for candidate types, including visible support in at least 3 held-out participant groups. | UNRESOLVED | Build the deterministic split/support receipts inside quarantine after human disposition; do not move groups in response to lexical support or outcomes. |
| G8 baseline/ceiling | One canonical ChildLens RGB/text inventory instantiates weak and strong controls with equal empirical exposure and only a frozen alignment-target/readout difference. | CONDITIONAL | Validate timing/referential target reliability and paired-manifest equality. Protocol construction is frozen, but release-specific proof is incomplete. |
| G9 simulator calibration | Terms permit nonidentifying video/audio aggregates; absent physical streams remain unavailable and simulator-defined only. | PASS WITH LIMITATIONS | Use only permitted ChildLens video/audio aggregates. Never represent IMU, touch/contact, proprioception, or motor variables as ChildLens-measured. |
| G10 resources | Selective acquisition leaves at least 30 GiB reserve; annotation is bounded/staffable; compute follows one-MPS or identical-GPU paired bundles. | CONDITIONAL | Revalidate the admitted quarantine, resolve exact selected media bytes, enforce v1.1's stricter 20 GiB raw, 73 GiB peak, and 50 GiB post-peak-free limits, and measure human labor before launch. |

## Frozen pilot binding

The target remains 15 videos in the allowable 12-18 range. The v1.1 preselection contains 15 videos from 15 distinct participant groups and a 192-item annotated source pool. Selection used only coarse activity, speech-presence, an annotation-derived duration-proxy tertile, location when available, and the frozen release/protocol-bound hash order. Lexical content, ASR output, visual content, and learner results were not used.

The procedural receipts are:

| Receipt | SHA-256 |
| --- | --- |
| Restricted preselection input | `81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a` |
| Placeholder manifest | `cc2b669e811ff78a045565be031537d5a93bfece40f27de3482768350adc1deb` |
| Restricted download plan | `bb39c32139310e30ac15095a15b3d4b95508ab217694f591845aac38e00364b8` |
| Human-validation packet | `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5` |

The placeholder manifest cannot satisfy G2 or acquisition admission. Its video sizes are placeholders, exact remote bytes remain unresolved, and the final canonical manifest is incomplete. The preselection cannot be replaced or expanded based on language, speech yield, lexical richness, visibility, annotation agreement, or any learner result.

## Decision produced by the unchanged rubric

G1 and G9 pass within explicit limits. G2, G8, and G10 are conditional. G3-G7 are unresolved. Because essential gates remain conditional/unresolved and each has a bounded ChildLens-only correction, while no unrepairable essential failure is established, the unchanged rubric yields exactly:

`CHILDLENS_FEASIBILITY_REVISE`
