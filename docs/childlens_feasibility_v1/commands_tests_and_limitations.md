# ChildLens feasibility v1 commands, tests, and unresolved limitations

## Actions performed

- Created and pursued an explicit audit goal with no token budget.
- Read the governing child-only contract, ChildLens measurement specification, construction readiness report, parallel execution design, and local EgoBabyVLM method-interface README before workstream execution.
- Froze the feasibility rubric and 12-18-video pilot protocol before inspecting media content. No media content was inspected and the pilot was not run.
- Ran six bounded parallel workstreams for governance/release, annotation schema, transcription/diarization, lexical/referential feasibility, resources/parallel execution, and provenance/privacy validation.
- Inspected the authenticated Keeper release read-only for aggregate inventory, metadata fields, folder structure, and a visible annotation-schema prefix. No library object was downloaded; no share, permission, or license was changed or accepted.
- Consulted primary public release, paper, repository, model-card, and license metadata sources as documented in the workstream reports.
- Wrote only to the new feasibility report namespaces plus the new validator and its test file.

## Validation commands and results

The final validation set used the repository virtual environment where the existing construction tests require PyTorch:

```text
.venv/bin/python -m pytest -q tests/test_childlens_feasibility_v1.py tests/test_child_only_prototype_v1.py
39 passed in 1.95s
```

The new validator unit suite also passed independently under the system interpreter:

```text
python3 -m pytest -q tests/test_childlens_feasibility_v1.py
17 passed in 0.13s
```

Compilation and JSON parsing passed:

```text
python3 -m py_compile scripts/validate_childlens_feasibility_v1.py
python3 -m json.tool <each JSON in docs/childlens_feasibility_v1 and output/childlens_feasibility_v1>
```

The official local EgoBabyVLM method-interface checkout remained clean at commit `224621caf0628270b6115845ac75a65b984234a3`:

```text
git -C .external/egobabyvlm rev-parse HEAD
git -C .external/egobabyvlm status --short
```

A focused restricted-pattern scan returned no matches in the report namespaces:

```text
rg -n -i '<numeric media filename or direct numeric participant/session assignment patterns>' docs/childlens_feasibility_v1 output/childlens_feasibility_v1
```

The structural/privacy/ancestry/decision validator was run during assembly. Its pre-final pass correctly failed only for the then-missing command report, missing evidence index, and one overconservative machine-key name; these assembly findings were corrected. The final command passed:

```text
python3 scripts/validate_childlens_feasibility_v1.py --json
status: PASS; issue_count: 0
```

One attempted combined test under the system Python failed during collection because that interpreter lacks PyTorch. The same new and existing test files then passed under `.venv/bin/python`; the failed command is retained here rather than hidden.

## Explicit non-actions

- No full archive or pilot shard was downloaded.
- No video/audio stream was decoded and no frame, clip, transcript, identifier, exact media timestamp, or small cell was written to the report namespaces.
- No ASR, diarization, voice-type, temporal-proposal, or vision model weight was downloaded or executed on ChildLens.
- No tokenizer or learner was trained, no causal arm ran, and no scientific acquisition outcome was produced or inspected.
- No other child-corpus or adult-corpus empirical data, aggregate, vocabulary, tokenizer, weight, checkpoint, prior, result, or artifact entered the feasibility ancestry.
- No upload, email, contact, commit, push, publication, license acceptance, or Keeper share change occurred.

## Tooling privacy incident

The authenticated inspection was not a zero-leakage tool run. An initial broad DOM response and a malformed CSV/schema extraction transiently emitted restricted metadata values into browser tool responses. No value is repeated here. The values were not downloaded as corpus files, saved to disk, copied into these artifacts, or sent to an external service. Subsequent extraction was aggregate/redacted and the browser session was finalized.

This incident makes the privacy disposition conditional and fail-closed. Before any further authenticated ChildLens inspection, the workflow must use page-local allowlisted aggregation/redaction, constant error responses, synthetic sentinel tests, and no broad DOM/table/CSV serialization. The validator can demonstrate that the on-disk report namespaces lack detected restricted payload patterns; it cannot retroactively prove that no tool response ever contained one.

## Unresolved limitations

1. The accessible release's archive License field is blank. No recipient-specific instrument reviewed here grants local copying, decoding, restricted derivative creation, model processing, checkpoint treatment, retention/deletion, or aggregate export.
2. The live library matches the public DOI library identity but exposes no immutable live revision. Byte-for-byte equality to the DOI snapshot is unknown.
3. The live view contains 192 video objects rather than the paper's 354-file full-corpus snapshot. An annotated-subset interpretation is plausible but not proven. One video also lacks a displayed size.
4. A restricted canonical manifest, media/annotation hash linkage, participant/date linkage coverage, duplicate/corruption audit, and container-duration union were not produced because terms did not permit derived processing.
5. Audio presence is documented, but stream/codec coverage, audibility, speech timing, and utterance boundaries were not validated.
6. Recorded language, multilingual/code-switching prevalence, and availability of qualified language-matched annotators remain unknown.
7. No validated timed lexical transcript or child/non-child role pipeline exists. Candidate voice-type code/weights lack clear permission and disagree on version identity; the defensible fallback is human-only role annotation.
8. Non-child transcript volume, lexical recurrence, noun/object and verb/action support, visible-referent rate, null/ambiguity distribution, event lag, and inter-annotator reliability are unmeasured.
9. Participant/date split fields exist, but complete linkage and lexical/action support after leakage-resistant splits are unmeasured.
10. Low/base selective storage allocations fit the current host, but the base case has limited floor headroom; exact selected bytes, correction labor, instrument throughput, learner update budget, and GPU runtime remain to be measured.
11. The annotation-instrument amendment is proposed but not adopted. It cannot authorize processing by itself.
12. All conclusions are feasibility/governance judgments. They provide no evidence for or against the later synchronized-cue acquisition hypothesis.
