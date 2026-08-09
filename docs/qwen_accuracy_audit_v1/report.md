# Qwen3-VL ChildLens accuracy diagnostic v1

## Bottom line

The existing Qwen3-VL outputs did **not** look accurate enough to justify
reopening the ChildLens branch. In a deterministic, prediction-blind audit of
27 pre-existing windows, only **7/27 complete answers were defensible** and
only **7/27 exact lag bins had visible support**. The 95% Wilson interval for
either rate is 0.132–0.447.

This is deliberately labeled **provisional visual/temporal plausibility**, not
human gold. No qualified German interpreter participated, and ASR,
utterance-window alignment, sparse-frame evidence, and Qwen judgment errors
cannot be separated here.

## What was actually tested

- Governing v1.7/v1.8 contracts, frozen protocols, amendments, calibration
  receipts, and the terminal causal-stop record were inspected first.
- The audit reused only the 135 existing windows in the retained 15-video
  extension subset of the completed 30-video calibration.
- Twenty-seven windows were selected without reading content or predictions:
  one hash-ranked window from every retained item, then a second from 12
  items.
- The existing Qwen predictions, existing local ASR hypotheses, and the same
  five ordered frames were reviewed locally. Qwen was not rerun.
- No media was acquired, no causal endpoint was opened, and no scientific
  acquisition outcome was run.
- Restricted frames, text, timing, identifiers, paths, and per-window labels
  remain outside the repository. Only K=5-safe aggregate counts were emitted.

## Aggregate results

| Provisional audit metric | Positive | Other | Wilson 95% |
|---|---:|---:|---:|
| Speech referent identifiable | 16 | 11 | 0.407–0.755 |
| Visible related object/action candidate present | 13 | 14 | 0.307–0.660 |
| Clear candidate rather than ambiguity/null/undecidable | 8 | 19 | 0.159–0.485 |
| Qwen exact lag bin visibly supported | 7 | 20 | 0.132–0.447 |
| Qwen complete answer defensible | 7 | 20 | 0.132–0.447 |

The audited Qwen sample itself was heavily affirmative: 22/27 outputs called a
visible candidate and 20/27 selected overlap. That pattern, combined with the
much lower defensibility count, is consistent with over-alignment or
over-confident lag selection. It is not proof that the Qwen model alone caused
the errors, because the fixed ASR and five-frame packaging are part of the
instrument path.

## Grounded-VideoLLM second-instrument result

The one selected independent instrument was
**Grounded-VideoLLM Phi-3.5-Vision-Instruct-4B (exact released checkpoint)**.

It is the best methodological complement here because it uses a non-Qwen base
and was built around an explicit temporal stream and timestamp tokens. The
authors report 36.8 mIoU on Charades-STA, 36.1 on ActivityNet grounding,
59.4 on MVBench, and 47.7 on Video-MME without subtitles for the general
checkpoint. The model and base are MIT-licensed.

The exact authors' implementation is CUDA-oriented. The initial storage
preflight failed, but that condition was superseded after storage was cleared:
the weight set was admitted, downloaded, and verified. The official code has
no MPS runner, and the one-shot public Mac canary subsequently failed because
its InternVideo2 temporal stream requires `Conv3D`, which PyTorch 2.1.2 does
not support on MPS. The authors recommend one NVIDIA GPU with 24 GB VRAM.

A prospective PyTorch 2.13 follow-up showed that the exact production-shaped
`Conv3D` now works on this M5. Two full-MPS smoke paths still failed on dtype
limitations before generating an answer. A final split path—InternVideo2 on
CPU float32, spatial/projector modules on MPS float32, and Phi on MPS
float16—did complete end-to-end generation. Its frozen runner then crashed on
a receipt-only Python boolean typo; the single unscored response was neither
printed nor retained, so schema admission was not recoverable and the run was
not repeated.

After explicit user authorization, a prospective v5 corrected only the
receipt boolean while preserving the model, runtime split, prompt, fixtures,
parser, generation parameters, and thresholds. The smoke passed. The unchanged
eight-case German public canary then scored **8/8 schema valid but only 4/8
referential-plus-lag correct**, failing the frozen 6/8 minimum. Grounded-
VideoLLM labeled all eight cases present—including the null and ambiguous
controls—and selected lead in six cases. This is the same over-alignment risk
the independent instrument was supposed to resolve.

The model card distinguishes the general Phi result from an asterisked variant
trained with subsets of Charades-STA and ActivityNet, but the public weight
repository exposes only one generically named Phi SFT checkpoint. The test was
prospectively bound to that exact released artifact while explicitly
withholding any benchmark-cleanliness claim.

TimeLens2-4B is the stronger compact July 2026 temporal specialist on repaired
benchmarks, but it is built on Qwen3-VL-4B. It is therefore a useful future
ablation, not the independent second instrument for this test.

## Decision

**Do not reopen a larger ChildLens human-grounded calibration study now.** The
predeclared screen required at least 19/27 defensible answers and a Wilson lower
bound of at least 0.50; the observed result was 7/27 and 0.132.

After storage was cleared, the resource gate passed and the exact released
checkpoint was downloaded and verified. A prospective clarification bound the
single published Phi checkpoint while labeling its general-versus-benchmark
training status undocumented. The one allowed public canary then failed before
its first answer: the model loaded, but InternVideo2 required `Conv3D`, which
PyTorch 2.1.2 does not support on MPS.

Later, separately frozen Mac-runtime versions established that PyTorch 2.13
can execute the exact Conv3D and that the model can generate end to end on the
M5 with whole-InternVideo2 CPU execution. They did not produce an admissible
smoke score: the successful generation's schema result was lost to a
post-generation receipt typo, and the one-attempt rule forbids reconstructing
it by rerun.

The explicitly authorized v5 did produce a model-performance result: perfect
formatting but only 4/8 correct public decisions, with no successful null or
ambiguous control. Because the predeclared canary required 6/8, the 27
restricted windows were correctly not opened. Grounded-VideoLLM did not
demonstrate that it performs better than Qwen, and it does not justify
investing in a larger human-grounded ChildLens calibration study.

## Files and sources

- Frozen diagnostic: `docs/qwen_accuracy_audit_v1/frozen_protocol.json`
- Aggregate receipt: `output/qwen_accuracy_audit_v1/audit_aggregate_receipt.json`
- Model comparison: `docs/qwen_accuracy_audit_v1/model_landscape.md`
- Frozen second-instrument test: `docs/qwen_accuracy_audit_v1/frozen_second_instrument_minitest.json`
- Execution status: `docs/qwen_accuracy_audit_v1/grounded_videollm_execution_status.md`
- Execution preflight receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_execution_preflight_receipt.json`
- Public canary execution receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_public_canary_execution_receipt.json`
- Mac runtime v2 receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_mac_runtime_v2_execution_receipt.json`
- Mac runtime v3 receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_mac_runtime_v3_execution_receipt.json`
- Mac runtime v4 receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_mac_runtime_v4_execution_receipt.json`
- Frozen Mac runtime v5: `docs/qwen_accuracy_audit_v1/frozen_grounded_videollm_mac_runtime_v5.json`
- Mac runtime v5 smoke receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_mac_runtime_v5_smoke_receipt.json`
- Mac runtime v5 canary receipt: `output/qwen_accuracy_audit_v1/grounded_videollm_mac_runtime_v5_canary_receipt.json`
- Primary model sources are linked in the model comparison.
