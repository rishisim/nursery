# ChildLens v1.3.1 author-language blocker disposition

Date: 2026-07-22  
Terminal state: `CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES`

## Disposition

The attempted author pass is invalid for scientific comparison. Two independent conditions are disqualifying:

1. The v1.3 interface passed the full source media object to the browser and used start/end values as player hints. It did not physically confine playback to the frozen audit window, so the audit scope was ambiguous.
2. The available author reports no German competence. The pre-frozen aggregate diagnostic from the completed offline Whisper instrument identifies German as dominant under an 80% rule applied jointly to item, window, and duration shares. That receipt is explicitly `MODEL_DIAGNOSTIC_NOT_HUMAN_VALIDATED`; it does not claim every item is German. Together, the diagnostic and the author's competence report establish that the sole available annotator cannot correct the encountered source-language transcript or validate German lexical semantics.

No partial author row is used in any gate. At disposition time the database contained no persisted item, utterance, or mention row, but the workflow was nevertheless sealed: future label writes, author lock, and prediction attachment are blocked. Predictions remained hidden, the author pass was never globally locked, and no model–human comparison was run. The app is stopped and was not relaunched.

Machine translation cannot repair this problem. A non-German speaker comparing a local German ASR hypothesis with its machine translation would validate neither the source words nor their morphology, omissions, timing, speaker role, or reference. Agreement between models measures instrument consistency, not human correctness. GPT-5.6 Luna, Codex, and every hosted model remain prohibited from receiving restricted content and would not create source-language ground truth even if that boundary did not exist.

## UI defect and correction

Synthetic source inspection confirmed the defect in `childlens_author_audit_app_v1_3.py`: the full media path was passed to `st.video` with `start_time` and `end_time`. The v1.3.1 correction materializes each frozen interval as a physically separate, metadata-stripped clip, verifies its duration with `ffprobe`, and renders that clip without seek offsets. The future view states the exact audit-window number and duration and the completed/total 15-minute progress.

This correction is tested only with generated audiovisual fixtures. It is not launched for the current author. A future qualified, authorized German annotator would receive a clean workflow; the invalid v1.3 attempt would not be reopened.

## Scientific consequence

Under current resources, ChildLens cannot serve as the sole empirical linguistic-acquisition corpus for the proposed lexical-grounding feasibility claim. The missing evidence is not merely extra compute or a stronger pseudo-labeler; it is a qualified source-language human audit that can establish transcript, timing, role, and referential validity. The corpus may still support a separately scoped descriptive visual/activity/speech-presence study, but that study must not be represented as validating linguistic content.

The only bounded ChildLens remedy is written authorization for a qualified German annotator to access the restricted audit and completion of a fresh, corrected, blinded 15-minute pass. Until both authorization and the annotator exist, the lexical role is stopped rather than revised by model consensus.

## Evidence receipts

- `output/childlens_feasibility_v1_3_1/author_attempt_invalidation_receipt.json`
- `output/childlens_feasibility_v1_3_1/language_diagnostic_receipt.json`
- `output/childlens_feasibility_v1_3_1/decision_record.json`
- `output/childlens_feasibility_v1_3_1/immutability_receipt.json`

No restricted filename, identifier, transcript, exact timestamp, frame, item-level language label, or small cell is exported in these records.
