# ChildLens v1.1 provenance and privacy audit

Version: `childlens-provenance-privacy-audit-v1.1.0`  
Date: 2026-07-21  
Scope: permission resolution, release binding, metadata-only acquisition, and pilot preparation  
Scientific learner or causal outcome run: no

## Disposition

The v1.1 continuation preserved the empirical boundary: ChildLens is the only child corpus from which restricted empirical material was acquired. The only acquired ChildLens payload classes were the participant table and final annotation objects, held in the restricted quarantine. No ChildLens video, audio, decoded waveform, transcript, human label, ASR output, learner input, checkpoint, or scientific result was acquired or produced.

No AEA or BabyView empirical data, aggregates, priors, vocabulary, tokenizer, checkpoint, weight, result, or other empirical artifact entered this work. Public methodological references do not create empirical ancestry. No external service received ChildLens material.

The prior v1 artifacts remain immutable. Their recorded artifact-set SHA-256 is `35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf`; v1.1 writes only in its new documentation/output namespace and an untracked restricted quarantine.

## Ancestry ledger

| Source or artifact class | Access/use in v1.1 | Scientific ancestry | Disposition |
| --- | --- | --- | --- |
| ChildLens permission instrument and approval correspondence | Read only to resolve G1 | Governance only | Clause-level paraphrase receipt; no private message text or personal fields exported. |
| Public ChildLens DOI/library/snapshot identifiers | Observed through the redacted release extractor | Release identity only | Public identifiers may appear in receipts. |
| ChildLens participant table | Acquired locally | Restricted metadata only | Quarantine only; no row, identifier, date, or small cell exported. |
| ChildLens final annotations | Acquired locally | Restricted feasibility metadata only | Quarantine only; no source record, label sequence, timestamp, or identifier exported. |
| ChildLens video/audio/transcripts | Not acquired | None | Content-level and human gates remain unresolved. |
| AEA | One procedural scope incident described below; no empirical material used | None | Strict no-read exception recorded and contained. |
| BabyView | No empirical access or use | None | No data, measurement, distribution, vocabulary, model artifact, or result used. |
| Fixed pretrained measurement instruments | Preflight/provenance review only | None | No restricted inference was run; no feature, embedding, vocabulary, score, or weight entered a learner. |
| Learner or checkpoint | Not created or run | None | No training, evaluation, causal arm, or acquisition effect. |

## Fail-closed authenticated extraction

Authenticated Keeper inspection occurred only after local synthetic validation of `childlens-keeper-redacted-extractor-v1.1.7`. The extractor returns a fixed allowlist of public release literals, booleans, safe aggregate integers, broad schema booleans, and constant error codes. It has no raw DOM, table, CSV, response-body, filename, identifier, exact timestamp, exception-message, or fallback serialization path.

The sentinel suite passed 9 of 9 tests. It exercised synthetic secrets in row text, path-like labels, schema labels, configuration, exceptions, and throwing DOM getters; wrong identity, missing evidence, invalid configuration, and oversized output all failed closed. No broad DOM/table/CSV capture was an authorized fallback.

## Quarantine controls and acquisition disposition

The restricted quarantine is outside the Git repository and is not a worktree. It is owner-only: directories use mode `0700`, files use mode `0600`, and creation uses umask `077`. Symlinks, group/world access, unapproved ACLs, cloud synchronization, and Git tracking are prohibited. The exact quarantine root was excluded from content indexing and Time Machine before restricted payload was retained.

Only one participant-table object and the final-annotation class were retained. Attempts to stage content through endpoints that returned HTML rather than the expected file type were detected as invalid. Those staging files were not parsed as ChildLens data, were not admitted to a manifest, and were unlinked. They do not count as acquired media.

No video or audio shard was admitted. Consequently, there was no media decode, frame extraction, transcription, diarization, referential annotation, or local measurement-instrument inference. This is an important privacy success but also leaves the empirical audio, language, speaker-role, visible-referent, lexical-recurrence, and human-reliability gates unresolved.

## Provenance exception: overbroad AEA path search

One resource-preflight agent issued an overbroad `rg` search that matched two nonempirical environment fields in an AEA provenance receipt. This violated the task's strict AEA no-read scope even though it did not open or reveal AEA data, aggregates, identifiers, result codes, tokenizers, checkpoints, or weights.

The incident was contained immediately: the two matched values were not used, host facts were re-probed independently, and subsequent searches were restricted to fixed ChildLens-relevant paths. It creates a procedural compliance exception, but it does not create AEA empirical ancestry or contaminate any feasibility inference. The exception must remain visible in the terminal report rather than being silently reclassified away.

## Export and repository boundary

Reportable artifacts may contain only public release identifiers, digests, fixed status codes, broad schema/coverage booleans, and permitted cell-suppressed nonidentifying aggregates. They may not contain participant/session values, restricted filenames or locators, exact recording dates or media timestamps, annotation rows, transcript text, examples, or small identifiable cells.

Restricted manifests, HMAC keys, row-level hashes, source locators, acquired annotations, staging material, future decoded audio, transcripts, and human judgments remain outside the repository. Invalid staging material is not a source artifact. No ChildLens payload was uploaded, emailed, committed, pushed, or published. No Keeper share or license was modified or accepted by this task.

## Unresolved privacy-relevant blockers

1. Exact byte equivalence between the accessible live view and public DOI snapshot is not proven. The live observation is receipted, but a library identifier and aggregate count are not a checksum-complete equivalence proof.
2. No selected media bytes were acquired or hash verified. A future selective shard must bind each admitted object to the frozen restricted manifest and verify size plus local SHA-256 inside quarantine.
3. No genuine human auditory or referential validation exists. Codex-generated judgments cannot substitute for qualified, language-matched human ratings.
4. Because language is still empirically unestablished, no ASR/aligner route may be promoted from instrument preflight to scientific measurement.
5. The AEA overbroad-search exception requires explicit acknowledgement in any final provenance claim, even though no empirical ancestry resulted.

## Audit conclusion

The v1.1 repository boundary is clean with respect to ChildLens restricted payload and cross-corpus empirical ancestry. The metadata-only acquisition does not establish lexical-grounding feasibility. Exact-byte binding and genuine human content validation remain necessary before the frozen gates can be adjudicated as passed.
