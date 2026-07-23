# ChildLens bridge v4 — development-only go/no-go

**Decision: NO_GO**
Status: `TERMINAL_DEVELOPMENT_NO_GO`

## Design

Training used 12 participants and 112 real audiovisual windows. Evaluation used 6 participant-disjoint participants and 61 windows, with zero participant overlap.

The sole route used fixed wav2vec2-base raw-speech and DINOv2-small vision frontends. Only two 128-dimensional linear projection heads were learned. No transcript, ASR decoder, language identification, translation, categorical referent label, or generative VLM judgment was used.

## Ordered gates

- G0_governance_binding_and_immutability: **PASS**
- G1_split_windows_and_support: **PASS**
- G2_preprocessing_and_perturbation_stability: **PASS**
- G3_primary_within_recording_alignment: **FAIL**
- G4_secondary_cross_participant_alignment: **FAIL**
- G5_privacy_safe_reporting: **PASS**

## Participant-disjoint controls

- Real minus within-recording shift: mean lift 0.0051, 90% participant-cluster interval [-0.0209, 0.0286], one-sided sign-flip p=0.3906.
- Real minus cross-participant shuffle: mean lift 0.0418, 90% participant-cluster interval [-0.0478, 0.1441], one-sided sign-flip p=0.3438.
- Shortcut audit: `NO_CREDIBLE_WITHIN_RECORDING_ALIGNMENT`.

## Scope and interpretation

V4 measures privacy-safe temporal, speech-audio, visual-motion, and audiovisual distributions in ChildLens; freezes those aspects and an apples-to-apples contrastive learner for eventual simulation with otherwise unavailable embodied side information; and stops here at the development anchor. The only eventual allowed claim is measured cue lift in simulation under temporal, visual, and audiovisual distributions provisionally calibrated to ChildLens ages 3-5, not validated naturalistic German lexical grounding.

All 22 locked participants remained sealed. No locked evaluation, new recording acquisition, AEA/external-volume use, BabyView use, simulator generation, or physical side-cue condition training occurred.

## Limitations

- Only six participant-disjoint development participants are in the reportable evaluation cell.
- The fixed audio frontend was self-supervised on adult English read speech and may transfer imperfectly to German child-centered natural audio.
- Released speech-like timing supplies candidate support, not word boundaries, transcripts, language identity, or lexical truth.
- Recording-level coarse activity is available consistently; timed activity subevents are not uniformly bound across both immutable cohorts.
- A development pass would still require a separately authorized locked evaluation before any simulator or cue-lift study.
