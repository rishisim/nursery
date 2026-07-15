# AEA terminal coarse-action and IMU diagnostic v3

## Formal decision: STOP

Scientific conclusion: **STOP_AEA_ROUTE**. one or more frozen coarse annotation, split/donor, capacity, or IMU gates failed.

Language conclusion: **STOP_LANGUAGE_ROUTE**. Sensor conclusion: **STOP_AEA_ROUTE**.

AEA is an adult, partly scripted sensor-format analogue. These are development diagnostics, not developmental findings, BabyView-like evidence, human annotations, or a confirmatory effect test.

## Fixed coarse annotation

The iteration reused exactly 72 development windows from all 18 development event groups and the existing 31-frame evidence. V3 made zero new RGB queries and accessed zero reserve RGB/IMU files.

Coarse visible-action agreement was 79.2% (kappa 0.6781883194278903). Modeled consensus was 43/72 (59.7%); the frozen 60% gate required 44/72. The coarse annotation/support gate failed.

Consensus support: gross_body_motion=15 windows/10 groups, object_or_body_interaction=28 windows/12 groups.

## Natural-speech alignment

Consensus language-aligned anchors were 7/72 (9.7%); the frozen gate required 18/72 plus at least 15 wearer-action labels in each pass and reliable simplified-referent agreement. The language gate failed.

A failed language gate is terminal for AEA language grounding and is not rescued by sensor performance.

## Split, video capacity, and IMU

The one-shot constrained split/donor status was `not_run_coarse_annotation_gate_failed` (gate False); no relaxation or retry was performed.

The transcript-free same-row video action-head status was `not_run_stage_gate_failed` (gate False; mean training balanced accuracy not run).

The conditional IMU diagnostic status was `not_run_stage_gate_failed` (gate False). No development IMU array was opened after the stage gate failed.

## Storage

Recommendation: **no additional acquisition**. The frozen viability and sole group-support/power bottleneck conjunction did not pass; storage use is not justified.

Current filesystem free space was 191.348 GiB. Free space is a ceiling, not a reason to spend storage.

## Limitations

- The two context-isolated passes are model-assisted and do not estimate human inter-rater reliability.
- Thirty-one ordered frames are dense evidence but are not continuous video or audio.
- The binary repair deliberately discards fine action distinctions and cannot support subtype claims.
- Capacity is a same-row optimization check; only the IMU endpoint uses held-out event groups.
- Handcrafted 129-dimensional IMU features and one logistic family bound the diagnostic.
- The v1 reserve is prospective from v1 onward, not pristine relative to the earlier smoke run.
- Threshold proximity cannot override the frozen terminal decisions.
