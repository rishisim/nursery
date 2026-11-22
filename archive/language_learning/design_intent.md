# Why weak speech alignment was intentional

Historical clarification, 26 October 2026. This records intent and evidence;
it does not change data, the generator, the learner or evaluation.

## The distinction to preserve

Weak, delayed, unseen and nonheld naming are part of Nursery's scientific
question. They were not all accidental defects. The intended question is
whether bodily information helps learn transferable language despite this
ambiguity. Both arms receive the same speech, and ordinary language training
must not gain hidden object labels or privileged example selection.

The more specific unresolved issue is the interaction between two separately
reasonable decisions: speech can refer to holding within a ±5-second session
window, while the body encoder resets at hard source-clip boundaries. A name
may therefore be associated with contact in a neighboring clip that the model
cannot access. The audit measures that condition; it does not measure how much
of the failed language result it causes.

Do not turn “make the connection trustworthy” into “make every utterance name
the current held object.” Any later intervention must distinguish natural
within-episode ambiguity from artificial clip concatenation, preserve matched
exposure, and retain nonheld/unseen naming. A clean or oracle condition may be
a labelled diagnostic; it is not a replacement for the intended weak task.

## Direct historical evidence

1. **Codex, 15 July 2026, 19:22 UTC.** The user explicitly asked to carry forward
   the premise: “weak alignment is part of BabyView/ChildLens—not a flaw we can
   simply filter away.” The same message called for broader episodes, possible
   no-referent outcomes, cross-situational learning, synchronized versus shuffled
   side-information controls, and independent generalization evaluation.
   Source: local Codex session `019f62dd-e1d1-7af3-9084-0ae78fec05c9`, record 543
   in the archived/compressed rollout. This was a requested research direction,
   not evidence that all proposed alignment mechanisms were implemented.
2. **Claude Code, 14 October 2026, 22:43 UTC.** The user clarified that contact
   and proprioception should be inputs and that the goal is a gain in language
   acquisition, rather than treating contact weighting as the final question.
   Source: session `224c9975-5cd4-488a-b576-f7ec9795b864.jsonl`, record 69.
3. **Claude Code, 15 October 2026, 05:08 UTC.** After the user authorized step 2
   at 05:07, the assistant's implementation brief specified `[t−5, t+5]`,
   “clipped to the session,” to keep naming before and after a grasp symmetric.
   This was an assistant-selected implementation of the accepted goal, not an
   exact window size dictated by the user. Source: record 1 of subagent
   `agent-a1a9f786c0975375f.jsonl` under session `224c9975-…`; implementation
   commit `cea4af3d`. [speech_coupling.md](../../docs/speech_coupling.md) preserves
   the literature evidence, age differences, sampling limits and parameter choices.
4. **Claude Code, 16 October 2026, 00:38 UTC.** For the body-only language-loss
   pilot the user supplied pasted instructions: “Use exactly the caregiver speech available to the
   baseline. Do not select or relabel examples using the hidden held-object
   identity or whether the spoken referent is held.” Source: the same session,
   record 4940. A failed noisy-language pilot does not authorize silently
   replacing its supervision with oracle labels.

5. **Claude Code, 15 October 2026, 16:04 UTC.** Codex reasoning pasted by the
   user said a recurrent encoder must reset or mask across recorded clips,
   “or it will learn artificial transitions.” Source: the same session, record
   3404. The reset therefore had a deliberate purpose; simply removing it is
   not an established repair for the speech-window mismatch.

At the user's request, Claude Code reviewed the original session history on
October 26 using read-only tools. It located the window and reset rationales
above; these excerpts were also checked directly. Neither review found an
earlier explicit rationale or completed test for naming across a hard cut while
denying the learner the relevant body history. This is a bounded historical
finding, not proof that no unexamined conversation ever discussed it. Claude
also read the archive while it was being written; its agreement is not an
independent scientific validation. The July Codex message was verified directly
from the Codex rollout, not by Claude's search of Claude sessions.

Session records may contain duplicated history after continuation. Timestamps
and quoted user messages identify the evidence; raw private transcripts are
not copied into Git. The local Claude Code history review is retained under
ignored `outputs/archive_review/`.

Earlier proposals restricted evaluation to picture-only transfer. That is not
the current scope: commit `1868b328` and the current plan explicitly recognize
body-present comprehension and picture-only transfer as separate outcomes.

## What the new checks establish

- The current training split contains 1,593 named held-channel events; 397
  lack recorded holding of the named object in the aligned, same-clip body
  window. This is 24.9% of held-channel naming, not of all training data.
- The current four-choice sampler excludes a distractor's named referent but
  does not exclude other visible target objects in that scene. On 472
  familiar-subject test trials, 114 have at least 100 renderer-labelled target
  pixels in a distractor's supplied image window. This is not a human visual
  recognizability or post-crop annotation.
- On the exploratory 322-trial subset with no renderer-labelled target in any
  distractor window, existing video-only predictions still score 50.3% versus
  44.4% for the full body-plus-physical model. The ambiguity does not explain
  away the latest null/regression.

Local reproducibility: `outputs/episode_review/review.py` and `audit.json`;
independent alignment detail in `outputs/research/body_debate/`. These are
read-only audits of saved data/predictions, not new training or preregistered
confirmation. The implementation/protocol has not yet been repaired.
