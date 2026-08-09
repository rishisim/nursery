# BabyView future lane: access and fresh-initialization checklist

Status: **future, separate, and not initialized**  
Public-source review date: 2026-07-22  
Required before the immediate synthetic lane: **no**

## Scope

The immediate synthetic lane may proceed without BabyView access, authorization,
data, metadata, models, or results. This document reserves a later BabyView-only
study. It does not authorize or initiate that study, and it must not delay or
alter the already-frozen synthetic protocol.

This review used only public BabyView and Databrary policy pages. No one logged
in, accepted terms, requested access, contacted a custodian, downloaded a file,
or viewed restricted BabyView content. Public documentation establishes
prerequisites; it does not establish that the author, a supervisor, or UTD is
authorized.

## Minimal user-action checklist

Do not acquire or process BabyView material until every applicable item below is
documented in the future lane.

1. **Confirm the institutional route.** Ask UTD's research or sponsored-programs
   office whether UTD has an executed Databrary Access Agreement and identify
   its Authorized Organizational Representative and any current Authorized
   Investigator relevant to the work. A public volume page or ordinary account
   is not evidence of institutional authorization.
2. **Choose the correct personal role.** A researcher eligible to conduct
   independent research or lead proposals should seek Authorized Investigator
   status. The first investigator at an institution uses the full Access
   Agreement; an additional investigator uses Annex II. A student or staff
   researcher who is not independently eligible needs Affiliate status under an
   Authorized Investigator, who assumes supervision and responsibility. Never
   borrow or share an account.
3. **Obtain institutional ethics disposition.** Confirm current institution-
   required human-subjects training and obtain a written UTD IRB or research-
   ethics determination for the proposed secondary scientific reuse. Databrary
   notes that limited pre-research activity may be treated differently, but
   research uses almost always require institutional review; the project must
   not classify itself.
4. **Review before accepting.** Only after institutional review should the
   researcher personally register with an institutional email address, accept
   the applicable Databrary terms, and enable required TOTP two-factor
   authentication. Credentials and recovery material remain private.
5. **Verify the dataset entitlement.** Once authorized, verify access to the
   main BabyView volume (`1882`) and record a clause-level receipt for the exact
   accessible release: release name/date, citation, sharing level, dataset-
   specific restrictions, permitted local copying and processing, derivative
   treatment, retention/deletion, and aggregate export. Databrary authorization
   is necessary but is not by itself proof that every planned BabyView action is
   permitted.
6. **Approve security before download.** Establish named least-privilege users,
   institution-approved sensitive-data storage, retention/deletion rules, and a
   breach response. Prohibit reidentification, recontact, credential sharing,
   public excerpts not expressly permitted by a file's release level, commercial
   use, cloud/third-party processing not expressly approved, and redistribution.
7. **Resolve derivatives before creating them.** Databrary's Terms section 11
   prohibits further copying or redistribution of downloaded data and requires
   new products based on downloaded content to be shared through Databrary or a
   similarly appropriate institutional repository. UTD must reconcile that
   provision with participant release levels, IRB terms, checkpoint privacy,
   transcripts/annotations, retention, and any aggregate-export plan before
   derived processing begins.
8. **Bind an exact release before acquisition.** The public BabyView page
   describes an evolving, multi-release dataset. Start the future lane with a
   new read-only terms receipt and immutable restricted manifest. Use selective,
   capacity-bounded acquisition; never treat a live volume as an immutable
   snapshot without evidence.

Any unresolved item is a fail-closed blocker for the BabyView lane, not for the
immediate synthetic lane.

## Strict BabyView-only boundary

The future lane must start with a new goal, governance contract, release
receipt, quarantine, manifest, sampler, tokenizer, learner initialization,
checkpoint namespace, and decision record. BabyView is its sole empirical child
corpus.

The lane must not read, import, derive from, calibrate with, or reuse ChildLens
or AEA raw data, derivatives, aggregates, empirical priors, distributions,
lexical material, vocabularies, tokenizers, splits, identifiers, timestamps,
embeddings, features, checkpoints, weights, model outputs, outcome-selected
seeds/hyperparameters, results, or code/configuration paths carrying those
values. It must not inherit another lane's access receipt or claim that one
corpus validates another.

BabyView-trained public checkpoints, embeddings, tokenizers, and derived model
artifacts are also excluded as learner ancestors. The scientific learner starts
from random weights, and any corpus tokenizer is trained from the independently
authorized BabyView training partition only. Public papers and generic method or
code interfaces may inform a prospectively frozen design only when they carry no
empirical values or trained artifacts.

A generic pretrained model may be proposed only as a separately licensed,
fixed, local/offline measurement instrument under a predeclared amendment that
the Databrary/UTD permissions allow. Instrument features, embeddings, weights,
tokenizers, vocabularies, confidences, or scores may not enter learner ancestry,
and machine hypotheses are not evaluation truth.

There is no joint training, shared vocabulary, cross-corpus calibration, record
linkage, pooled evaluation, or outcome-driven protocol transfer. A later
cross-study synthesis may be considered only after each study is independently
frozen and complete, and only with separately authorized, nonidentifying
aggregate summaries. It may not retroactively tune either study.

## Authoritative public sources

- [BabyView dataset and access overview](https://babyview-project.github.io/dataset/)
- [BabyView main Databrary volume 1882](https://www.databrary.org/volume/1882)
- [Databrary authorized-access overview](https://databrary.org/about/agreement)
- [Databrary Access Agreement](https://databrary.org/about/agreement/agreement)
- [Annex I: rights and responsibilities](https://databrary.org/about/agreement/agreement-annex-I)
- [Annex II: additional Authorized Investigators](https://databrary.org/about/agreement/agreement-annex-II)
- [Annex III: Access Guide](https://databrary.org/about/agreement/agreement-annex-III)
- [Databrary Terms and Conditions of Use](https://databrary.org/about/policies/terms)
- [Databrary guidance for using shared data](https://databrary.org/support/using-shared)
- [Databrary secondary-reuse IRB template](https://databrary.org/support/irb/irb-application)
- [Databrary release levels](https://databrary.org/support/irb/release-levels)
- [Databrary two-factor authentication guide](https://databrary.org/support/authenticator-guide)

These links should be rechecked when the future lane begins because access
guidance and the BabyView release can change.
