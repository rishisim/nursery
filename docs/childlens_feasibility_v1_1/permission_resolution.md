# ChildLens v1.1 clause-level permission resolution

Date reviewed: 2026-07-21  
Gate disposition: `G1_TERMS = PASS_WITH_CONTROLLING_CONSTRAINTS`  
Legal action taken by this task: none

## Evidence chain

1. **Signed document:** *Dataset Access Request and Agreement Form* for *ChildLens - An Egocentric Video Dataset for Activity Analysis in Children*, signed 2026-07-15. Source: the user's UTD Exchange Sent Items attachment, with a matching already-downloaded local PDF. SHA-256: `332d638f1b1f27e10e3b65d9d4342a246b488fe0978cd5130ba75c04a552b066`.
2. **Approval correspondence:** the ChildLens dataset manager acknowledged receipt of the completed form on 2026-07-16 and approved access, subject to Keeper account activation.
3. **Access completion correspondence:** the dataset manager sent the Keeper library invitation on 2026-07-21; the user confirmed successful read-only access the same day.

No signature, personal address, account identifier, message identifier, phone number, email address, or verbatim private correspondence is reproduced here. The task did not sign, accept, reply to, move, flag, or alter any message or agreement.

## Controlling approved purpose

The signed request identifies noncommercial academic action-language grounding research using ChildLens videos and annotations. It describes evaluation and calibration of models comparing synchronized sensorimotor learning signals with matched controls, with results reported only in aggregate. The expected use period is July 2026 through July 2027.

The agreement limits the data to scientific, academic, noncommercial use; prohibits commercial machine-learning training; prohibits sharing, redistribution, or third-party access; requires secure access/storage under institutional data-protection rules; and requires proper ChildLens citation in publications or presentations.

This is permission for the approved project, not an unrestricted data license. It does not authorize cloud processing, data redistribution, public checkpoints, a new purpose, a different institution, commercial activity, or retention beyond the approved period.

## Required-action matrix

| Required action | Status | Evidence-based interpretation | Controlling constraint |
| --- | --- | --- | --- |
| Selective local copy | PASS | The agreement expressly contemplates secure dataset storage during the approved use period; the approved project uses the videos and annotations. | Store only in the user's institutionally controlled restricted workspace; no third-party access; no full-archive acquisition. |
| Local decode | PASS | Scientific use of the videos and model evaluation/calibration within the named project covers local decoding as an internal processing step. | Local-only; no remote media service; restricted intermediates. |
| Restricted transcripts and referential annotations | PASS WITH SCOPE LIMIT | Internal derivatives are a necessary analysis product of the approved action-language grounding purpose and remain subject to the same secure-storage/non-sharing terms. | Treat every derivative as restricted dataset material; no transcript/example export; delete by the retention deadline unless permission is renewed. |
| Fixed local measurement instruments | PASS WITH SCOPE LIMIT | The signed project expressly includes model evaluation/calibration, and the agreement prohibits only commercial model training rather than internal noncommercial model use. | Each instrument must be separately licensed, local, fixed, quarantined, and unable to transmit data; its features and vocabulary never enter learner ancestry. |
| Learner and checkpoint treatment | PASS WITH SCOPE LIMIT | Internal noncommercial model evaluation/calibration is the approved purpose; commercial training is expressly prohibited. | No scientific outcome in this continuation; any later checkpoint is restricted, noncommercial, nonshared, ChildLens-only, and expires with the approved use unless renewed. |
| Retention and deletion | PASS WITH CONSERVATIVE DEADLINE | The signed request states July 2026–July 2027 and requires secure storage. | Use 2027-07-31 as the fail-closed deletion/review deadline for local data and restricted derivatives; earlier deletion is allowed; extension requires new written permission. |
| Nonidentifying aggregate export | PASS | Aggregate-only reporting is part of the approved project description, and the agreement anticipates cited publications/presentations. | Cell suppression, no examples/text/identifiers/timestamps, proper DOI citation, and no release before the later scientific protocol authorizes reporting. |

## Adjudication rule

`G1_TERMS` passes for the bounded v1.1 feasibility pilot because every required action is either expressly stated or is an internal, necessary processing step inside the specifically approved noncommercial action-language/model-calibration project. The scope-limited rows do not expand the agreement: they inherit its secure-storage, no-sharing, institutional, noncommercial, aggregate-only, citation, and time limits.

If any planned operation would transmit data to a vendor, publish a derivative/checkpoint, continue after July 2027, change the project purpose, or involve a person outside the approved institutional boundary, this receipt no longer suffices and processing must stop for new written permission.

This is a research-governance interpretation for the frozen pilot, not legal advice and not a new acceptance of terms.
