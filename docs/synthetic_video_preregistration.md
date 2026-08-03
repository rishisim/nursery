# Synthetic-video governance and preregistration

**Phase 4 status:** **CORRECTED COMMON ASSETS PASS — ACTIVITY CHECKPOINT PUBLIC-DEVELOPMENT NO-GO**

**Evidence cut-off:** 2026-08-02

**Authority boundary:** this record preserves every frozen Phase 3 decision
and records the public-language PASS plus the reopened common-asset gate. It
does not itself execute or authorize generator work, generation, TTS, score
opening, or scientific evaluation. Confirmatory Phase 5 is not authorized.
BabyView remains unavailable,
out of scope, and supplies no empirical ancestry.

The public/dummy single-NVIDIA-L4 engineering preflight at Nursery commit
`1e8e94dbec4ad1ed701298f2c0ebb58cca26c877` is a PASS for the pinned
EgoBabyVLM bridge/runtime only. It is neither scientific performance nor
restricted-data readiness. Its 12.75 GB peak is one smoke-test observation,
not a training-memory estimate. DDP is untested.

## Registered question and claims

> At equal \(H\) video hours, does EgoBabyVLM CLIP+ `triple` trained on
> `Synthetic-full` achieve lexical-grounding performance equivalent to
> `Real-full`, while the synthetic corpus has lower prospectively measured
> production cost?

The primary performance contrast is `Synthetic-full - Real-full`, with both
arms at \(H\) credited hours. Secondary contrasts are `Mixed - Real-full` and
`Mixed - Real-small`, where `Mixed` contains the same nested \(r\)-hour real
subset as `Real-small` plus \(H-r\) synthetic hours. Equivalent performance is
not equivalent linguistic acquisition. Machine-DevBench Lexical uses generated
images and is not held-out-real ChildLens evaluation. A separately reported
held-out-real model-derived temporal frame–utterance transfer safeguard is
mandatory.

The learner is fixed to
`facebookresearch/egobabyvlm@224621caf0628270b6115845ac75a65b984234a3`,
CLIP+ `triple`, with the 4:1:1 contrastive/MLM/DINO-iBOT schedule and the
public BERT/DINO pins and bridge recorded by the preflight. Every arm uses the
same offline German ASR → English translation path, fixed public
`bert-base-uncased` tokenizer, public BERT/DINO priors, training-step rule,
seed schedule, and one shared Machine-DevBench Lexical asset. Synthetic oracle
text is QA-only.

## Evidence and decision matrix

Status meanings are `FROZEN`, `REQUIRES CONFIRMATION`, and `NO-GO`.

| Topic | Status | Evidence or frozen decision | Owner | Exact unblock action |
|---|---|---|---|---|
| Learner/runtime | `FROZEN` | Public/dummy single-L4 bridge/runtime PASS; no scientific score; DDP untested | Technical lead | Preserve immutable pins and rerun acceptance inside the approved governed environment |
| ChildLens academic use and aggregate reporting | `FROZEN` | The signed request explicitly covers ChildLens videos/annotations for non-commercial model calibration/evaluation in grounded learning, with aggregate-only reporting through July 2027. Paul Grohmann accepted the form and granted access | Authorized applicant | Keep every use within that scope, applicant-only, and cite DOI `10.17617/4.fe` |
| ChildLens local storage | `FROZEN` | The corpus was migrated to an AES-256 encrypted sparsebundle with an applicant-only Keychain secret. A read-only checksum comparison matched 67,087 regular files / 47,917,156,217 bytes, 13 symlinks, and 6,353 directories before the unencrypted source was removed | Authorized applicant | Keep the image encrypted and unmounted except for applicant-only governed work |
| ChildLens inventory | `FROZEN` | Aggregate inventory found 58 catalog children / 192 recordings / complete durations. The preexisting development-only allowlist fixes \(C\) at 18 children / 58 recordings / 14.374241 source hours, leaving 40 children / 134 recordings / 40.362056 source hours for confirmatory allocation with zero child overlap | Authorized applicant | Apply the frozen keyed split and \(H/r\) rules only after governed compute and the identical ASR/translation pipeline are qualified |
| Governed CUDA | `FROZEN` | Institution-owned Juno is qualified under the signed agreement and UTD's risk-based policy: applicant-only account, UTD VPN plus SSH, private `0700` home/work/scratch targets, group-restricted `yding` storage, SLURM isolation/accounting, and UTD-managed execution. Open public egress is permitted for pinned public dependency ingress and is not itself a failure; transmission of ChildLens or derived restricted artifacts to APIs, hosted services, Git, cloud storage, telemetry, or other third parties remains prohibited | Authorized applicant | Enforce the frozen restricted-job controls below; keep all restricted files/caches/logs/checkpoints on UTD storage and clean temporary scratch data |
| SLURM fair-share mapping | `REQUIRES CONFIRMATION` *(non-blocking)* | The applicant is in the `yding` Unix group, but `sacctmgr` lists `compsci` and one unrelated PI association rather than `yding`; CPU test job `310661` was charged to `compsci` | Authorized applicant / UTD HPC administration | Ask HPC to add or affirm the `yding` SLURM association and review/remove the unrelated association; this is an accounting/fair-share correction, not a ChildLens security gate |
| DDP/scaling | `FROZEN` | Final-topology public/dummy 1-node/2-H100/2-process preflight passed before the corrected restricted build | Technical lead | Preserve the tested topology and frozen 12-hour ceiling for the corrected Phase 4 build record |
| ASR/translation | `FROZEN` | Public gate selected immutable Whisper small plus OPUS-MT German→English; offline reload, timestamps, confidence, round trip, and telemetry controls passed | Language/technical lead | Use only the selected identical pipeline for later real and synthetic audio |
| German human validation | `FROZEN` | No German-speaking human annotator is available, and the agreement prohibits making the dataset accessible to third parties | Authorized applicant | Use no human rater and retain only explicitly model-derived claims; a future rater requires separate MPI authorization |
| \(C,H,r\), margins, seeds | `REQUIRES CONFIRMATION` | Blind rules and bounds are frozen below; numeric values require permitted aggregate inventory and real-only variance | Authorized applicant in locked statistician stage | Apply the registered algorithms without synthetic results and sign/hash the completed config amendment |
| Common benchmark | `FROZEN` | Exactly one corrected lexical asset and one corrected held-out-real temporal safeguard are hashed, sealed, and referenced identically by every later arm | Evaluation custodian | Preserve commitments `3798fafc…17b4` and `3cc29f32…e46e`; prohibit test steering |
| Score sealing/unblinding | `FROZEN` | Synthetic-arm scores remain inaccessible until the real-only gate passes; this is a disclosed single-operator protocol | Authorized applicant | Use separate procedural roles, coded outputs, append-only commitments, and the ordered unblinding script below |
| Cost comparison | `FROZEN` | Prospective like-for-like marginal and fully loaded ledgers; sunk ChildLens collection is not zero | Authorized applicant in locked cost stage | Insert pre-generation unit prices and distributions, then hash/sign the ledger |

Phase 3 is PASS. ChildLens permission, encrypted local storage, complete
aggregate inventory, calibration lineage, source-duration inventory,
child-level split feasibility, governed Juno handling, cost rules, sealing,
and unblinding are established. Exact credited \(H/r\) remains governed by the
frozen blind post-ASR yield rule rather than an invented source-hour value.
The bounded public-only ASR/translation selection is the sole remaining
technical substage permitted by the Phase 3 pass rule. DDP remains a mandatory
public/dummy gate only if blind sizing requires multiple processes. A confirmed
consent/license incompatibility, inability to form an independent evaluation
split under the rules below, or loss of every compliant institutional compute
path changes the status to `NO-GO`.

## Governance and permission matrix

“Governed boundary” means the encrypted applicant-only local sparsebundle and
institution-owned Juno storage/execution under UTD policy. It excludes this
repository, unencrypted personal-device paths, ordinary cloud storage, Hugging
Face Jobs, external APIs, and hosted experiment trackers. ChildLens permission
is established by the accepted signed agreement. The following rules
operationalize its secure, applicant-only, non-commercial handling conditions.

Established agreement evidence is affirmative and project-specific. The signed
request names “Sensorimotor Cues for Grounded Action-Language Learning,” UTD,
Dr. Yi Ding, and July 2026–July 2027. It specifies ChildLens videos and
annotations for non-commercial action-language-grounding research, model
calibration/evaluation with matched controls, and aggregate-only reporting. On
2026-07-16 Paul Grohmann confirmed that the completed form was accepted and
access would be granted; on 2026-07-21 he sent the Keeper library invitation.
The agreement prohibits commercial use, commercial-model training, sharing,
redistribution, and third-party access; requires secure storage under
institutional data-protection rules and institutional-email correspondence;
and requires citation of DOI `10.17617/4.fe`. It does not certify a particular
drive, workstation, CUDA host, or third-party rater. Dr. Yi Ding is the named
supervisor/UTD research sponsor; the applicant is the sole current data user
and pipeline operator. Membership in the `yding` Unix research group records
that sponsorship, consistent with
[UTD's PI-sponsored Juno account process](https://hpc.utdallas.edu/getting-started-with-hpc/).
SLURM fair-share accounting does not define ChildLens authorization.

| Material class | Permitted storage/execution | Network and egress | Access roles and logging | Retention and disposition | Git |
|---|---|---|---|---|---|
| ChildLens raw video/audio | Encrypted applicant-only local storage or applicant-only UTD-managed Juno storage/compute | Public dependency downloads are allowed; dataset content may never be transmitted to hosted services, APIs, cloud storage, Git, telemetry, or other third parties | Authorized applicant only; Unix permissions and SLURM job/provenance records retained | Source retention follows the agreement/project period; applicant removes task-local and scratch copies | Never |
| Transcripts, ASR, translations, frames, embeddings, ChildLens-derived prompts/statistics | Same boundary and controls as raw unless a named aggregate passes disclosure review | Open network availability grants no export permission; external logging/telemetry disabled and model loading is local/offline during restricted processing | Authorized applicant only; artifact creation and disposition recorded in the governed run manifest | Derived retention schedule fixed before creation; applicant deletes temporaries and expired outputs | Never, except a specifically approved compact non-identifying aggregate |
| Participant/session identifiers and split ledgers | Separate applicant-only governed namespace; direct identifiers never enter learner-facing manifests | No third-party transfer | Authorized applicant sees the identity map; learner pipeline receives opaque keyed IDs; mapping operations recorded | Identity map follows the agreement/project period; analysis ledger archived as authorized | Never |
| Public/dummy media and public model weights | Repository-external public cache or hosted/public CUDA | Ordinary network allowed only for declared public sources | Technical staff; provenance and hashes logged | Reproducible caches may be deleted; immutable pins retained | Source/config/hash metadata only; not weights/media |
| Synthetic outputs, failed generations, QA labels, manifests | Public-only development outputs may use approved public compute; study outputs are governed when plans/statistics derive from \(C\) | Restricted study artifacts may not be sent to third parties; public-only outputs may use declared public services | Generator/QA roles; every attempt, label, and disposition recorded | Retain all attempts through locked analysis in governed run storage, then archive/delete per approved schedule | No media, attempts, prompts derived from \(C\), or row-level manifests; compact approved aggregates/config only |
| Compact permitted aggregates | Governed staging until disclosure review; then approved institutional/repository location | Export only after documented disclosure review | Data steward approves; exporter and exact fields logged | Curated decision records retained with provenance | Allowed only if non-identifying, authorized, compact, and no reconstructive content |
| Human-rater materials | Governed rater interface only; no local downloads/screenshots | No outbound transfer | Named authorized raters, rater manager, and auditor; item access and decisions logged | Rater caches deleted at session end; labels follow approved retention | Aggregate agreement/decision only if approved |

Deletion/archival responsibility is explicit: the authorized applicant owns
raw and identity records and removes expired Juno work/scratch copies,
pipeline owners delete their derived temporaries and caches, the generator
lead accounts for every synthetic attempt, the rater manager clears any later
authorized rater workspace, and the repository maintainer admits only approved
compact records. Administrator-witnessed deletion is not required by the
agreement or UTD policy. “De-identified,” a keyed hash, or an embedding does
not itself permit third-party transfer.

## Governed CUDA handling contract

The signed ChildLens agreement requires secure access and storage following
institutional data-protection guidelines. It does not require internet-disabled
compute, dedicated nodes, one-way ingress, immutable file/network audit logs,
or administrator-witnessed deletion. [UTDBP3096](https://policy.utdallas.edu/utdbp3096)
applies risk-based controls, prefers UT-managed servers, requires encryption
for confidential personal-device copies and remote access over unsecured
networks, and prohibits unapproved third-party storage. The prospective
qualification therefore requires:

1. institution-owned or institution-approved execution; Juno satisfies this
   ownership requirement;
2. access through the applicant's individual UTD account over GlobalProtect
   VPN and SSH, with no shared credentials;
3. raw and derived restricted material only in applicant-only `0700` Juno
   paths, or a deliberately approved `yding` path when collaboration is later
   authorized; encrypted applicant-only storage for any personal-Mac copy;
4. public packages and weights may be downloaded, but must be pinned by
   revision/license/SHA-256 and cached before restricted processing;
5. restricted jobs load models from local files, disable telemetry and
   external experiment tracking, and never send ChildLens media, transcripts,
   translations, frames, embeddings, prompts/statistics, identifiers, logs,
   checkpoints, or other derived restricted artifacts to an API, hosted GPU,
   cloud store, Git, paste/upload endpoint, or other third party;
6. input, temporary frames/audio, model caches, compiler caches, checkpoints,
   logs, and crash outputs resolve only to declared UTD-managed locations; no
   restricted value may enter a filename, URL, scheduler job name, environment
   variable, or outward error report;
7. scratch follows copy-in, process, copy-out, and cleanup. The applicant
   records completion in the governed manifest and removes task temporaries,
   failed-job remnants, and expired outputs under the project retention plan;
8. scheduler/job records plus a compact governed provenance manifest record
   execution and disposition. Agreement compliance does not depend on
   immutable file/network audit coverage not required by UTD policy;
9. the immutable upstream commit, patch/config hashes, public weight hashes,
   resolved environment lock, driver/CUDA/GPU inventory, container hash,
   deterministic settings, and fail-closed path/telemetry assertions are
   recorded; and
10. capacity qualification resolves single-process versus DDP. If multiple
    processes are required, the registered public/dummy DDP preflight remains
    mandatory before restricted execution.

### Juno qualification evidence

On 2026-07-30 UTD HPC confirmed account activation by institutional email.
The applicant connected over UTD GlobalProtect to
`juno.hpcre.utdallas.edu`; both private login addresses presented the same
ED25519 host fingerprint
`SHA256:ylbFsrAnLBJNCg9IF2mBfjE6hTg0l32Th8CRIPuihQE`. A dedicated
passphrase-protected Ed25519 key is stored only on the applicant's Mac and
loaded through the macOS Keychain. Khalid Warraich directed the applicant to
the [official Juno orientation materials](https://hpc.utdallas.edu/systems-resources/juno/)
as the quick-start path. The current
88-page PDF, `Juno_Orientation_v14` (2026-07-15), was reviewed on 2026-07-30
and identified by SHA-256
`f1f585a8a5871a091a19c5b0ecc2d75bcb9493206d378a44fb3fcb1ae617a2be`;
the PDF itself remains untracked. The deck has no quiz, acknowledgement, or
completion-certificate mechanism. Sreshtha's formal welcome email remains
pending, but no further local orientation step is identified. Read-only checks
and one minimal public/dummy CPU job established:

- Rocky Linux 9.5 and SLURM 23.11.10 on login node `juno-l-02`;
- applicant-only mode `0700` on `/home/dal503972`,
  `/work/dal503972`, and `/scratch/juno/dal503972`, with advertised quotas of
  50 GB, 1 TB, and 30 TB respectively; `/groups/yding` is mode `2770` and
  restricted to the `yding` Unix group;
- active A30, H100, and H200 partitions; no GPU allocation or job was launched;
- `auth/munge`, `proctrack/cgroup`, `task/cgroup,task/affinity`,
  `jobacct_gather/cgroup`, and `accounting_storage/slurmdbd`;
- SLURM associations for `compsci` plus one unrelated PI account, but no
  `yding` SLURM association (the unrelated name is intentionally omitted from
  Git), while the applicant's research Unix group is `yding`;
- successful public DNS and HTTPS access to Hugging Face and PyPI from the
  login node; and
- successful public DNS plus HTTP 200 responses from Hugging Face and PyPI
  inside public/dummy CPU-only SLURM job `310661` on `dev` (one task, one CPU,
  512 MiB, two-minute cap); the job completed in 25 seconds and was charged to
  `compsci`. SingularityCE 4.2.2 also exposes optional network namespaces.

These checks establish institution-managed compute, individual access,
private filesystem targets, scheduler isolation/accounting, orientation, and
GPU availability. Open public egress is documented but is not a failure under
the signed agreement or UTDBP3096. It permits pinned public dependency ingress;
it does not authorize restricted-data egress. Juno is qualified for
applicant-only ChildLens processing under the handling contract above.
Adding/affirming `yding` as the SLURM fair-share account and removing the
unrelated PI association remain non-blocking administrative corrections.

Ordinary Hugging Face Jobs or other hosted services are allowed only for
public/dummy work. They may never receive ChildLens media, audio, transcripts,
translations, frames, embeddings, derived prompts/statistics, identifiers, or
split metadata.

### Conservative sizing and distributed gate

The sizing operator must use public/dummy tensors with the frozen architecture,
resolution, objective mix, precision candidate, and representative upper-bound
utterance/token/frame shapes. For each objective, measure allocated and
reserved GPU memory through forward, backward, optimizer step, post-SSL sync,
checkpoint save/resume, and evaluation. Repeat after allocator warm-up and use
the maximum. Add measured model/optimizer/checkpoint staging, then require at
least `max(25% of measured peak, 8 GiB)` free headroom per GPU and a successful
two-cycle fragmentation test. Estimate wall time from at least 100 steady-state
objective cycles and apply a 1.5× scheduling contingency. The 12.75 GB L4 smoke
peak may be reported beside, but never substituted for, this measurement.

Choose single GPU only if the complete registered batch/accumulation contract,
headroom, wall-time ceiling, and checkpoint behavior pass without semantic
changes. If sizing requires multiple GPUs/processes, a public/dummy DDP
preflight is mandatory before governed work. It must use the final world size
and topology class and pass: one 4:1:1 cycle per rank, sharding/sampler
disjointness, synchronized backbone equality, gradient/update equivalence to a
declared single-process reference within frozen tolerances, checkpoint/resume
across ranks, failure propagation, and evaluator aggregation. No billable job
is authorized here. A later request must name provider, GPU count/type,
wall-time cap, and cost ceiling and receive explicit approval before launch.

## Data lineage, allocation, and credited hours

### Partitions and opaque identifiers

The only real-data roles are:

- \(C\): calibration/generator-development material, excluded from learner
  training, validation, and evaluation;
- eligible learner-training pool: supplies `Real-full`, its nested budgets,
  `Real-small`, and the real component of `Mixed`;
- validation: used only for training-health diagnostics under the common fixed
  checkpoint rule, never arm-specific stopping; and
- independent evaluation: used only for the held-out-real safeguard.

Inside the governed boundary, the custodian maps direct identifiers to
`HMAC-SHA-256(secret_split_key, namespace || canonical_source_id)`. Keys are
versioned, access-controlled, never placed in Git, and distinct from content
hashes. Allocation operates on opaque IDs.

The preexisting development-only allowlist is now the complete \(C\): 18
children, 58 source recordings, and 14.374241 source hours. Those children have
already influenced prior development and can never enter learner training,
validation, or evaluation. The catalog contains 58 children / 192 recordings;
the remaining eligible confirmatory catalog therefore contains 40 children /
134 recordings with zero child overlap with \(C\).

After technical eligibility is complete, allocate the 40 remaining children by
the largest-remainder method toward 70% training, 20%
evaluation, and 10% validation, with minima 3/2/1 and tie order training,
evaluation, validation. If all remain eligible, the fixed counts are 28
training, 8 evaluation, and 4 validation children. Sort the 40 children by the
HMAC of `study_id || child_id`, then deal them in repeating role order
evaluation, validation, training, skipping a role once its count is filled.
This makes assignment deterministic without content or outcomes.
Within a child, all sessions stay in that child's partition. If unequal
duration leaves the training pool too small, \(H\) decreases under the frozen
rule; children are not reassigned after duration is known. If policy forces one
child across operational roles, confirmatory independence fails; session-only
allocation is not an automatic substitute. The independent statistician may
authorize a preregistered leave-one-child-out design only before outcomes and
must narrow the claim accordingly; otherwise this is `NO-GO`.

The aggregate inventory returned complete child, recording, file, date, and
duration fields. A governed join between the frozen development allowlist and
the preselection metadata found 14.374241 source hours in \(C\) and 40.362056
source hours in the 40-child eligible catalog, with no missing durations and no
cross-role child. These are source hours, not credited hours. Direct IDs,
filenames, text, frames, media hashes, and participant attributes remain inside
the encrypted boundary.

### Overlap and near-duplicate audit

Before role lock, audit exact source identifiers, recording time intervals,
session provenance, and approved local content hashes. After extraction, audit
frame perceptual hashes and frozen public embedding distances only inside the
boundary. Thresholds are selected on public/dummy transforms plus \(C\), never
test. Any exact/temporal overlap is removed from all but its earliest governed
source role; any cross-partition near-duplicate is quarantined for blind
adjudication without outcomes. Test contamination that cannot be repaired
without changing the frozen evaluation unit is `NO-GO`.

Evaluation children, sessions, frames, utterances, neighbors, vocabulary,
exposure counts, and derived statistics cannot influence \(C\), generator
development, episode plans/prompts, benchmark construction, QA rules,
training, stopping, checkpoint choice, or analysis changes. Validation cannot
influence benchmark/generator choices or arm-specific stopping.

### Nested real subsets and insufficient inventory

Order eligible training sessions by
`HMAC-SHA-256(secret_budget_key, "train-budget" || opaque_session_id)`;
within a session, order eligible utterance windows by the corresponding keyed
hash. Create nested real budgets by prefixes of this one order. Whole sessions
are preferred; only the final boundary session may contribute a prefix of
non-overlapping chronological accepted windows, and the same prefix is reused
everywhere. `Real-small` and the real component of `Mixed` are byte-identical
at \(r\). `Real-full` extends that prefix to \(H\). No statistics from the
remainder of `Real-full` may reach the \(r\) arms.

Choose \(H\) as the largest prespecified candidate not exceeding the lower
95% one-sided confidence bound on credited eligible training hours after
reserving \(C\), validation, evaluation, and a 10% operational reserve.
Candidate hours are powers-of-two multiples of the smallest meaningful budget
supported by at least three independent training children and at least two
sessions per included child. Choose \(r\) as the largest candidate below
\(H\) that leaves at least \(H-r\) synthetic hours and is itself one of the
three-or-more nested readiness budgets. If fewer than three training children,
fewer than two independent evaluation children, or fewer than three feasible
nested budgets remain, the statistician considers the already specified
leave-one-child-out alternative; if no independent design with usable power is
possible, status is `NO-GO`. No actual IDs are assigned in Phase 3.

### Credited hour

One credited second is one unique source-timeline second in an accepted learner
window that has a decodable eligible frame under the frozen sampler and
non-silent rendered/recorded German audio yielding a non-abstained,
timestamp-valid ASR segment and non-abstained translation. Overlapping windows
credit their union once. Silence, corruption, out-of-scope content, failed or
rejected generation, padding, repeated epochs, duplicated frames/utterances,
and QA-only oracle text receive zero credit. For real data, credit follows the
original recording timeline; for synthetic, the final accepted rendered
timeline. The same thresholds and code apply to both. Failed material remains
in the cost/yield denominator even when it earns zero hours.

Every use of \(C\)—inventory, transcription/translation, vocabulary proposal,
prompt/statistic extraction, generator calibration, benchmark construction,
QA threshold selection, fidelity reference, labor, storage, and compute—is
recorded in lineage and cost. \(C\) never counts toward \(H\) or \(r\).

## Statistical analysis and score sealing

### Endpoints and fixed training rule

The primary endpoint is the arithmetic mean of the common realistic-style
Machine-DevBench Lexical noun macro-accuracy and adjective macro-accuracy.
Both components, cartoon diagnostics, per-concept results, and exposure strata
are reported. The held-out-real safeguard reports child-clustered temporal
frame-to-utterance Recall@1 and mean reciprocal rank, with reciprocal retrieval
secondary; it is model-derived temporal alignment transfer, not referent truth.

Training uses the same fixed number of 4:1:1 cycles for every arm/seed. The
cycle count is set from public/dummy throughput and the real-only smallest
budget so every arm receives the same objective-step counts; it is signed
before any test evaluation. The checkpoint at that exact step is primary.
Validation loss and earlier checkpoints are diagnostic only and cannot select
an arm-specific checkpoint.

### Blind numeric design completion

An independent statistician, with no synthetic outcomes, completes numeric
fields after the authorized inventory and real-only pilot:

- \(H\) and \(r\) follow the rules above.
- Use at least three common seeds. Increase to five or ten only if the blinded
  power calculation shows the lower count cannot attain 80% power under the
  resource ceiling; never reduce below three.
- The practical equivalence margin \(\Delta\) is the smallest of: 0.05 absolute
  macro-accuracy; half the real-only improvement from the smallest readiness
  budget to \(H\); and 0.5 times the child-cluster bootstrap SD of the
  `Real-full` seed aggregate. It must lie in `[0.02, 0.05]`. If the blind value
  is below 0.02 or power is below 80% even at 0.05 with the maximum approved
  seeds, the confirmatory equivalence study does not launch.
- The meaningful readiness margin \(\delta_R\) is
  `max(0.02, 0.25 × real-only improvement from smallest budget to H)`, capped
  at 0.05.
- Freeze the seed list by public randomness beacon plus study identifier
  before training. Synthetic results cannot affect any value.

The real-only gate passes only if the hierarchical two-sided 95% seed/concept
interval for `Real-full - 0.50` lies wholly above \(\delta_R\), and a weighted
regression over at least three nested log-hour budgets has a positive slope
whose one-sided 95% lower bound is above zero. Failure seals all synthetic
scores and permits only real-only diagnosis under a new prospective amendment.

### Inference

Primary equivalence uses two one-sided tests at familywise alpha 0.05, or
equivalently the paired 90% confidence interval for
`Synthetic-full - Real-full`, and passes only if the whole interval is within
`[-Δ, +Δ]`. Pair common seeds; within each seed, resample evaluation concepts
for Machine-DevBench and evaluation children for the held-out-real safeguard.
The primary bootstrap is hierarchical: resample seeds, then noun/adjective
concepts within task; report a percentile interval with at least 10,000
deterministically seeded replicates. A mixed-effects sensitivity analysis uses
task/concept and seed effects.

Secondary tests are hierarchical and do not rescue the primary:

1. only after primary equivalence passes, test `Mixed - Real-full` for
   equivalence with the same \(\Delta\);
2. only after that passes, test one-sided superiority of
   `Mixed - Real-small` at alpha 0.05.

If reported regardless of gate, non-reached tests are descriptive with
Holm-adjusted 95% intervals across the two secondary contrasts. The safeguard
is mandatory but not part of the lexical equivalence claim; report its paired
95% child-cluster interval and do not relabel a lexical pass as real-domain
equivalence.

Aggregate across all completed registered seeds equally. A run is failed if it
violates config/data hashes, produces non-finite state, misses the fixed
checkpoint, or fails the registered runtime assertions. Do not replace seeds.
If one seed fails for an arm, exclude that common seed from every arm's paired
primary analysis, report it as failed, and run worst-case sensitivity imputing
the failed synthetic observation at chance and the failed real observation at
the best observed registered real seed (reversed when conservative for a
contrast). Fewer than three complete paired seeds makes inference
non-confirmatory.

Report training exposure per benchmark concept as `0`, `1–k`, and `>k`, with
`k` fixed as the pooled nonzero median exposure computed from sealed training
manifests before scores. Zero-exposure concepts remain in the primary.
Sensitivities include noun/adjective separately, cartoon style, equal-unique-
pair subsampling, alternate permitted duplicate threshold, per-child
safeguard, and marginal-versus-fully-loaded cost.

### Sealing and unblinding

This is a disclosed single-researcher study. “Operator,” “custodian,”
“statistician,” and “cost lead” below are separate procedural roles performed
at different locked stages by the authorized applicant, not separate people.
This does not create personnel independence and must be reported as a
limitation. The evaluation stage uses randomized arm codes and writes no
human-readable score summary. The code key stays outside the active analysis
workspace until the scripted real-only gate requests the permitted mapping.
Each encrypted score bundle is committed by SHA-256 plus
config, checkpoint, asset, manifest, seed, timestamp, and failure-status hashes
to an append-only institutional store. Git may receive only the permitted
commitment hash.

Unblinding order is fixed:

1. lock config, assets, manifests, cost ledger, code, and analysis script;
2. commit all coded score bundles;
3. disclose only real-only budget codes and decide readiness;
4. if readiness fails, retain synthetic codes and scores sealed;
5. if it passes, sign the gate, disclose all arm codes once, run the primary
   analysis, then the ordered secondary analyses, safeguard, sensitivities,
   fidelity, and cost analyses.

No post-hoc change to \(\Delta\), endpoints, benchmark, seeds, failures,
thresholds, checkpoint, or primary claim is allowed. Amendments before
unblinding must state reason and impact, retain Git history, and be signed by
the PI, statistician, and custodian.

## Cost and fidelity independence

Use USD in the calendar year in which generation begins; convert other
currencies with the preregistered monthly central-bank rate and inflate prior
costs with the applicable national research/consumer index. Freeze quantities,
unit prices, uncertainty distributions, allocation rules, and the prospective
real comparator before generation.

| Ledger category | Real prospective \(H\)-hour comparator | Synthetic \(H\)-hour corpus |
|---|---|---|
| Development/collection labor | protocol, recruitment, consent, scheduling, collection, troubleshooting | protocol-specific generator/TTS integration, episode planning, prompt/QA development |
| Participants/equipment | participant compensation, staff, cameras/audio, maintenance, loss/failure | compute hardware rental/depreciation and required capture/render tooling |
| \(C\) | fully loaded acquisition/access/preparation share, amortized by registered expected number of studies and also reported with all cost assigned to this study | same single \(C\) cost allocation, never double-counted; calibration labor/use separately visible |
| Attempts/yield | failed visits/recordings and unusable/silent/rejected portions | every failed/retried generation, rejected audio/video, and zero-credit attempt |
| Processing | secure transfer/storage, ASR, translation, frame extraction, QA, human review | generation GPU/runtime/energy, storage, secure transfer, TTS, ASR, translation, QA, human review |
| Fees | licenses, data access, equipment/service overhead | model/API/license fees (external APIs prohibited for restricted inputs), compute overhead |

Track quantity, unit, unit price, labor role and loaded hourly rate, GPU/CPU
seconds, energy where measurable, storage byte-days, attempt ID, accepted
credited seconds, source quote, and low/base/high uncertainty. Report marginal
cost (additional production operations after the frozen system exists) and
fully loaded cost (all attributable development, \(C\), overhead, failures,
equipment depreciation, and labor), total and per credited usable hour. Existing
ChildLens cost is reconstructed prospectively; it is never zero because sunk.

“Lower cost” passes only if performance equivalence passes and the upper bound
of the preregistered 95% uncertainty interval for
`synthetic fully-loaded cost / prospective real fully-loaded cost` is below
1.00. Marginal ratio is secondary. Use paired Monte Carlo propagation with at
least 10,000 fixed-seed draws from frozen input distributions and report
low/base/high deterministic sensitivity. Missing categories fail closed rather
than being set to zero.

Cost and fidelity staff cannot access learner test assets, uncoded arm results,
or checkpoint decisions. Their samples, ratings, yields, and cost judgments
cannot change benchmark contents, QA acceptance after lock, learner training,
stopping, score sealing, or unblinding.

## Offline ASR/translation and human-access gate

Exact ASR and translation models are selected in one bounded public-language
substage, not with ChildLens. Candidate models must: run fully offline from
locally cached weights during restricted processing; have immutable
revision/file SHA-256 hashes and licenses permitting the research and
restricted local processing; expose no telemetry; support German ASR with word
timestamps and German→English translation; preserve episode/utterance IDs,
word order, start/end timestamps, punctuation-normalized text, and abstention
flags; and fit the sized governed resource.

Use a fixed, redistributable public German speech set plus self-authored German
audio/text covering child-directed vocabulary, overlap, silence, noise, and
long utterances. No large weights are downloaded and no experiment runs in
Phase 3. The later substage passes only if local-files-only execution succeeds
with telemetry and external tracking disabled, timestamps are monotonic and
within audio duration for 100% of non-abstained items, ID/word/timestamp
round-trip tests pass, license/hash manifests are complete, no crashes or
silent truncation occur, and blind selection minimizes public-set word error
rate then translation adequacy under a frozen resource tie-break. Before
launch, the language lead must set public-only numeric WER, translation,
timestamp-error, confidence, and maximum-abstention thresholds without
ChildLens. Confidence below threshold, missing/nonmonotonic timestamps, empty
output, language mismatch, or translation failure causes abstention and zero
credited time.

The chosen immutable pipeline and thresholds process real audio and rendered
synthetic audio identically. Synthetic oracle text may measure QA error but may
not replace ASR/translation output.

No German-speaking human annotator is currently available. Model-derived
German adequacy, visibility, or alignment judgments are not human validation.
The current MPI agreement prohibits making the dataset available to third
parties, so no later human rater may access real ChildLens material without
separate written MPI authorization plus a governed no-download interface,
least-privilege item assignment, access logging, and retention/deletion
approval. Without that authorization, the safeguard remains explicitly
model-derived and no human-validated German claim is made.

## Phase 3 pass and post-pass gates

Phase 3 is PASS. Dataset access, project-specific model
calibration/evaluation, aggregate reporting, governed Juno handling, and every
safe protocol decision are established. The following frozen gates still
control later execution:

1. **Public-language selection — PASS:** the bounded public/self-authored gate
   selected `openai-whisper==20250625` `small.pt` at SHA-256
   `9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`
   plus `Helsinki-NLP/opus-mt-de-en` at immutable revision
   `1a922f3b32a8e809e17a47d4b32142d8105924e5`. This exact local-files-only,
   no-telemetry pipeline is now mandatory and identical for real and synthetic
   audio.
2. **Resource sizing/DDP — PASS:** the final 1-node/2-H100/2-process topology
   passed public/dummy qualification before corrected restricted execution.
3. **Restricted common assets — CORRECTED PASS:** literal keyed-HMAC allocation,
   shared abstention adapter, blocking exact/temporal overlap audit, pinned
   upstream filtering, official loader/evaluator smoke, and corrected resealing
   all passed. The lexical commitment is `3798fafc…17b4`; the held-out-real
   temporal commitment is `3cc29f32…e46e`. The prior commitments remain
   explicitly provisional and superseded.
4. **SLURM administration (non-blocking):** ask HPC to add or affirm `yding`
   fair-share association and review/remove the unrelated PI association. This
   does not block Phase 4 or compliant applicant-only ChildLens work.

The prior Stage B commitments are provisional and superseded by the corrected
seals. The frozen one-seed lean equal-duration pilot evaluated initialized 0h,
Real-1h, and the sole permitted nested Real-3h extension. Both trained budgets
failed the realistic lexical learnability gate, so the stop rule fired before
synthetic generation. Confirmatory multi-seed Phase 5, automatic 6h acquisition,
equivalence testing, and inferential scientific claims remain blocked.

### Prospective lean generator amendment

After the frozen Real-1h gate result but before any matched-corpus generation
or synthetic learner score, the authorized applicant selected MiniMax H3 for
the lean pilot. This latest prospective choice supersedes the intervening
Gemini and local-LTX selections while preserving them in Git history. It is not
a reinterpretation of the completed blinded bakeoff: Gemini led that screen at
27/28 primary checks, followed by MiniMax at 25/28, Seedance at 24/28, and LTX
at 19/28. MiniMax must not be described as the bakeoff winner.

The ChildLens boundary remains strict for source and reconstructive material.
Only after governed measurement, disclosure review, sparse-cell suppression,
and commitment may compact non-identifying aggregate targets from \(C\)
condition a public episode-plan distribution. The target table, plans, prompts,
attempts, outputs, logs, and QA remain on applicant-governed Juno storage. No
hosted generator receives ChildLens media, transcripts, exact vocabulary,
identifiers, embeddings, row-level or per-child/session values, evaluation
concepts, or reconstructive combinations.

The selected hosted route is `minimax/hailuo-3` through OpenRouter using the
exact request frozen by the completed bakeoff. Video generation is asynchronous
and ineligible for ZDR, and hosted outputs remain blocked for learner training
without written provider and institutional clearance. As of 2026-08-01, no
official immutable Hailuo 3 video-weight release with a verified compatible
license had been established for governed local execution. No corpus was
generated because Real-3h failed the learnability gate; a later generator route
cannot revive this stopped pilot without a new prospective protocol.

### Coverage-based one-hour exploratory redesign (prospective)

On 2026-08-02, before any redesigned learner outcome, new generator outcome,
synthetic corpus, or synthetic learner score, the applicant authorized a
separate exploratory amendment. It preserves the prior 570-step stop and its
commitments, but does not treat that run as evidence that one hour is
intrinsically insufficient: roughly 380 contrastive updates at batch size two
provided only about 760 example exposures for 1,244 Real-1h records.

Both active arms contain exactly one accepted credited hour and exactly 1,244
learner records. Batch size remains two. Five contrastive-equivalent passes
require 3,110 contrastive batches; rounding upward to complete 4:1:1 cycles
freezes 778 cycles, 4,668 total scheduler steps, 3,112 contrastive updates, 778
MLM updates, and 778 DINOv2 updates. Contrastive and DINO streams use separate
deterministic iterators so auxiliary steps do not silently consume
contrastive coverage. The three public hash-derived seeds are `436034264`,
`1285938051`, and `151347827`; Real-1h and Synthetic-1h use matched seeds,
byte-identical per-seed initialization, steps, optimizers, and checkpoints.

The Real-1h positive control passes only if its realistic noun/adjective macro
averages at least 0.52, its mean seed-matched gain over initialization is at
least 0.02, at least two of three seed gains are positive, and every seed's
temporal Recall@1 and MRR are finite and decline by no more than 0.05. Failure
is inconclusive about corpus quality and stops C calibration and generation.
No grammar or overall aggregate is introduced.

The latest prospective generator choice is locally governed LTX-2.3 using
`Lightricks/LTX-2@9377758`, LTX-2.3 weights revision `4229404`, and Gemma
text-encoder revision `68f7ee4`. It is a practical local-execution choice, not
a reinterpretation of the blinded screen: Gemini scored 27/28, MiniMax 25/28,
Seedance 24/28, and LTX 19/28. All C aggregates, aggregate-conditioned prompts,
plans, attempts, media, QA, logs, and artifacts remain on applicant-governed
Juno storage; no hosted provider receives ChildLens material or derivatives.
Modular German TTS and the repaired common language adapter remain mandatory.

Only after the Real positive control passes may governed C calibration and a
public/dummy LTX topology preflight occur. Before generation, that preflight
must replace the provisional ceiling with exact Juno GPU type/count, wall-time,
aggregate GPU-hours, storage, attempt/retry, and cost-accounting ceilings.
Generation stops at exactly 3,600 accepted credited seconds and may create at
most 5,399.625 raw seconds (1,071 attempts of 121 frames at 24 fps, at most 357
retries and two attempts per plan), within 72 hours and 100 GiB. The same accepted synthetic
hour is reused for all learner seeds. More than one accepted synthetic hour,
confirmatory equivalence claims, and evaluation-driven generator tuning are
prohibited.

### Coverage-redesign gate result

All three frozen Real-1h seeds completed at exactly 4,668 steps with the
registered mode counts and finite runtime-health diagnostics. The seed-mean
realistic macro was `0.5183627698` at initialization and `0.5360246706` after
training, a mean gain of `0.0176619008`. Two of three seed gains were positive;
all temporal safeguards were non-catastrophic; and the trained mean exceeded
`0.52`. The gate nevertheless failed because the mean gain was below the frozen
`0.02` requirement. Per the prospective rule, this is inconclusive about data
quality and stops before governed C calibration, LTX topology preflight,
generation, or synthetic learner training. No evaluation-driven threshold,
budget, seed, or generator change is permitted inside this amendment.

### User-authorized post-gate descriptive extension (prospective)

After the coverage-redesign Real-1h failure above was sealed, and before any
governed C calibration, new LTX outcome, synthetic corpus, or synthetic learner
score, the user explicitly authorized one scientifically weaker descriptive
Synthetic-1h extension. The failed gate is not reclassified: trained mean
realistic macro remains `0.5360246706`, mean gain remains `0.0176619008` against
the frozen `0.02` requirement, two of three seeds improved, and all temporal
safeguards were non-catastrophic.

The extension asks only whether one particular locally governed LTX-2.3 corpus
of exactly 3,600 accepted credited seconds produces a directionally similar
training effect under the already frozen three-seed learner procedure. It has
no binary success gate. Results will be point estimates and seed dispersion;
a positive synthetic effect resembling the observed real effect may be called
only encouraging descriptive evidence. Because the formal real gate failed,
“directionally competitive,” “equivalent,” “noninferior,” “same quality,” and
confirmatory Phase 5 language are prohibited. Similar flat or declining arms
are inconclusive.

The execution order is frozen: first complete the bounded governed eight-axis
C calibration and commit the aggregate-conditioned public episode-plan
distribution; then run a public/dummy LTX final-topology preflight; then freeze
and, where required, obtain approval for exact Juno GPU, wall-time, GPU-hour,
storage, attempt/retry, and cost ceilings. Only after those gates pass may one
accepted synthetic hour be produced and reused across the same three public
learner seeds. Corrected evaluation assets, language-adapter abstention rules,
4:1:1 schedule, 4,668 steps, byte-identical initialization, and checkpoint rule
remain unchanged. There is no automatic 3h arm, more than one accepted
synthetic hour, extra seed, hosted provider, or outcome-driven tuning.

### Governance pause during calibration schema inspection

Before any C feature outcome was computed, one schema-inspection command
unintentionally expanded a dictionary and printed 94 opaque asset-level
SHA-256 keys into the Codex task output. The keys are unkeyed hashes over a
restricted source-object key and clip ordinal. No media, audio, text, filename,
path, child/session key, direct participant identifier, embedding, prompt, or
outcome was printed, and no key was copied into Git or a local artifact.
Restricted execution stopped immediately. The authorized applicant/data-steward
subsequently reviewed the incident, determined that no additional institutional
reporting or remediation was required, and explicitly authorized governed
processing to resume. Aggregate-only containment checks found zero restricted
asset-key matches in all tracked Git content and in permitted compact records;
the experiment manifest, split, scores, seeds, thresholds, calibration target,
and seals were unchanged. The prior rendered task output is not claimed to have
been reverted or deleted. Future governed inspection uses a flat explicit-field
whitelist that rejects lists, nested values, arbitrary strings, and hashes
outside designated commitment fields.

### Governed C-calibration result — frozen no-go

The bounded calibration completed on one H100 in `00:05:03`, sampling 3,193
frames and 653 grounding events. Five of eight axes exceeded the prospectively
frozen 20% missingness ceiling. Activity/context was 64.83% missing, scene
complexity 54.98%, hand-object action 76.61%, audiovisual grounding 41.25%, and
diversity/heterogeneity 27.29%. Egocentric visual regime (17.89%), temporal
continuity/recurrence (1.01%), and language environment (0%) were measured
within the ceiling.

Audiovisual grounding is a frozen critical axis, and three measured axes cannot
support the later requirement that at least six of eight axes be within
calibration tolerance. The calibration commitment is `28ef267c…e3d5`; a
1,428-plan governed episode-plan commitment `57803d35…dc54` is retained as
provisional and non-executable. Lowering the 0.02 public-vision abstention
margin, changing the estimator, or weakening missingness after observing this
result would be post hoc and is prohibited. LTX final-topology preflight,
generation, and Synthetic-1h learner training therefore did not run.

### Prospective calibration-extractor repair

After the original no-go and commitments above were sealed, the user explicitly
requested one better extraction attempt. This is a new, coverage-focused
measurement repair, not a reinterpretation of the original result. Before any
repaired C output, LTX run, synthetic corpus, or synthetic learner score, one
candidate was frozen: the existing immutable PE-Core model with three-prompt
activity prototypes plus the Apache-2.0
`google/owlv2-base-patch16-ensemble@cfd3195ba4ea9592eec887ded089f4c08eff231d`
open-vocabulary detector. OWLv2 supplies public-category object, hand,
visibility, contact, framing, distractor, and occlusion proxies; PE-Core emits
the coarse activity class and retains its margin as a low/medium/high
uncertainty band. The official Transformers image processor uses pinned
`scipy==1.16.1` from the CPython 3.11 manylinux wheel with SHA-256
`adccd93a…e5b7` under BSD-3-Clause.

The scientific gates are not relaxed. The maximum axis missing fraction stays
at `0.20`, all critical axes must remain within that ceiling, and at least six
of eight axes must be measured. A valid finite detector inference with no boxes
is the measured `none`/`no_hand` condition, not imputation; decode or inference
failure remains missing. The earlier margin-abstention output and its governed
files remain untouched under their original commitment. Repaired artifacts are
written to the semantically distinct governed `calibration_repair` root and
explicitly supersede the provisional plan only if the repaired gate passes.

Before C may be reopened, this exact combined extractor must pass one bounded
public-only qualification using eight frozen COCO 2017 validation images with
retained Creative Commons licenses and immutable image hashes. The frozen gate
requires at least 4/8 coarse activity matches, 5/8 expected public-object hits,
2 positive hand detections, 2 correct hand-negative fixtures, complete proxy
records for all eight images, and zero invalid boxes. Model preparation may use
public network ingress; qualification must reload every weight and fixture
locally with telemetry disabled and no restricted mount. Failure is a no-go:
no new model, threshold change, candidate cycling, or C-driven tuning is
permitted in this repair.

### Extractor-repair public result — frozen no-go

The public-only qualification completed on one H100 in `00:00:20` with no
restricted mount and zero direct monetary cost. The combined extractor passed
the coarse activity check (7/8 against 4 required), expected-object check (8/8
against 5), positive-hand check (4/5 with 2 hits required), proxy completeness
(8/8), and box validity (zero invalid). It failed the frozen hand-negative
check: only 1/3 negative fixtures remained negative, below the required 2/3.
The extractor-repair commitment is `fd454059…a332d` and the public-result
commitment is `042e9ba4…fa7f`.

This is a detector-specificity no-go, not evidence about C and not a new C
calibration result. No ChildLens material was reopened, and no repaired C
targets or episode plans were created. The observed fixtures, labels, score
thresholds, and pass criteria are not changed; no second detector is tried.
Governed C rerun, LTX preflight, generation, and synthetic learner training
remain stopped.

### Prospective domain-appropriate extractor redesign

After preserving both failures above, the user authorized one new extractor
amendment before any further model inference, C reopening, LTX outcome, or
synthetic learner score. It asks only whether one fixed public stack can pass
a disjoint egocentric-video qualification and then measure C under the
unchanged gate: every passing axis has missingness at most `0.20`, every
critical axis passes, and at least six of eight axes are measured. The earlier
commitments and no-gos are not reinterpreted or overwritten.

The single stack is frozen to EgoVLPv2 temporal video-language scoring for the
existing eight activity/context labels; EgoHOS for left/right hands, contact,
and first-order interacting objects; Grounding DINO with a public coarse
ontology followed by SAM 2.1 mask propagation for scene and referent timing;
and DINOv2 for governed-only dispersion and near-duplicate embeddings. The
already functioning deterministic visual, temporal-continuity, and shared
language modules remain. SAM 3, DINOv3, alternate checkpoints, and post-hoc
candidate cycling are prohibited.

Public qualification is prospectively split into threshold-development and
sealed holdout partitions using first-person Charades-Ego, VISOR validation,
and fixed self-authored German audiovisual micro-clips. Partitions are
source-participant/video disjoint and use public seed `20260802`. All fixtures,
annotations, licenses, and hashes remain outside Git and must be committed as
one manifest hash before holdout inference. C is never used for model or
threshold selection. The exact metric floors are frozen in canonical proof
schema 9: per-module coverage at least `0.80`; activity macro-F1 at least
`0.60`; hand sensitivity and specificity each at least `0.80`; contact and
scene-complexity macro-F1 at least `0.60`; tracked-category presence F1 at
least `0.70` with median mask IoU above `0.50`; audiovisual timing macro-F1
at least `0.65`, no-referent specificity above `0.80`, and event coverage
at least `0.80`; near-duplicate balanced accuracy at least `0.90`; and zero
crashes, silent truncations, invalid records, external calls, or unaccounted
failures.

Before adapters, fixtures, or model inference, every repository and checkpoint
must have an immutable revision and resolved byte hash, explicit code and
weight terms compatible with local academic processing, feasible pinned
dependencies and Juno resource expectations, and a successful local-files-only
reload plan. A failure of any component is a no-go for this single stack; no
replacement model or threshold relaxation is permitted.

### Domain-appropriate extractor pre-model result — frozen no-go

The official repositories were pinned before inference: EgoVLPv2 at
`550c0596…c84`, EgoHOS at `cd9bdf42…3e7`, Grounding DINO at
`856dde20…e44`, SAM 2.1 at `2b90b9f5…1a4`, and DINOv2 at
`7764ea0f…fc8`. All five code-license records were resolved. SAM 2.1 and
DINOv2 explicitly cover their code and model checkpoints under Apache-2.0.
No model inference or public fixture selection occurred.

The pre-model gate failed on the first validity-critical component. The
official EgoVLPv2 zero-shot Charades-Ego checkpoint URL returned HTTP `403`
for both HEAD and ranged GET requests, including three user-agent variants.
Its bytes and SHA-256 therefore could not be resolved. The official project
record applies MIT to EgoVLPv2 but does not separately state terms for the
externally hosted checkpoint artifact, so the required weight-terms record is
also unresolved. A Juno-side cross-check was not available because SSH
public-key authentication was unavailable in this session; that does not
change the local 403 or missing checkpoint-specific terms.

The single-stack feasibility status is `NO-GO`. Per the frozen rule, no
alternate checkpoint or model is tried. Dependency integration, VRAM/storage
sizing, local-files-only reload, adapters, public development, sealed holdout,
and the governed C rerun were not reached. Both earlier no-gos and commitments
remain intact; LTX preflight, generation, and synthetic learner training remain
stopped. Resumption requires the same official EgoVLPv2 checkpoint to become
locally obtainable with an immutable byte hash and explicit academic-local
weight terms. The compact feasibility record commitment is
`b5dd6c03…a49d0`.

### Prospective activity/context checkpoint selection amendment

The preceding pre-model no-go remains final for its frozen EgoVLPv2 stack. It
is the third preserved failure in sequence, after the original governed-C
missingness no-go (`28ef267c…e3d5`) and the PE-Core + OWLv2 public specificity
no-go (`042e9ba4…fa7f`). On 2026-08-02 the user prospectively authorized a new,
bounded selection study that may replace only the unavailable activity/context
component. It does not reopen those results, alter the other extractor
families, or change any public-holdout or governed-C threshold.

Primary-source research kept unlike evidence separate. In particular,
zero-shot, public-probe, fine-tuned, retrieval, classification, and
anticipation results are not treated as one leaderboard:

| Family | Relevant published evidence | Temporal/text recipe | Artifact and terms decision |
| --- | --- | --- | --- |
| EVA02-AT B/L | strongest reviewed egocentric video-text evidence; L reports Charades-Ego zero-shot mAP 30.9 and substantially stronger fine-tuned and EK100 results | joint spatial-temporal attention; 4-frame pretraining and 16-frame downstream settings | ineligible: official delivery is Baidu-only here; bytes, SHA-256, and separate checkpoint terms were not resolved |
| EgoHOD EgoVideo-L | zero-shot EK100 MIR mAP 41.8, EK100 action top-1 24.0, EGTEA mean/top-1 47.1/51.7, EgoMCQ intra/inter 65.5/95.9; no Charades-Ego result | 16-frame egocentric video-text encoder with motion adapter | eligible; official Apache-2.0 code/model record and complete 5,130,482,632-byte checkpoint, SHA-256 `71faa0b6…880da` |
| LaViLa TSF-L | zero-shot Charades-Ego mAP 28.9; EK100 MIR mAP 35.0/36.1 for 4/16 frames | Ego4D video-text dual encoder, 16-frame Charades evaluation | ineligible: MIT code, but no separately stated terms for the external checkpoint |
| V-JEPA 2 / 2.1 | V-JEPA 2 ViT-L reports public attentive-probe SSv2 73.7 and Diving48 89.0, plus EK100 anticipation action R@5 32.7; these are not zero-shot | 64-frame self-supervised temporal encoder; no text alignment, so this study permits one frozen public-only probe | V-JEPA 2 ViT-L eligible under its official Meta model card; V-JEPA 2.1 excluded because its direct checkpoint record lacks separate weight terms |
| EgoVLP / HierVL | older direct Charades-Ego zero-shot mAP 25.0/26.0; HierVL adds long-term hierarchy | short-term video-text transfer, with hierarchical aggregation only in HierVL | not shortlisted: weaker directly relevant evidence and/or less explicit checkpoint provenance than the bounded candidates |
| EgoVideo 2024 / InternVideo2-CLIP | predecessor reports strong EK100 MIR; InternVideo2-CLIP reports general zero-shot Charades 32.9 | four-frame predecessor; eight-frame 1B general video-text model | predecessor is superseded by EgoHOD; relevant InternVideo2 1B bytes returned unauthorized behind its access gate |
| VideoPrism-LvT-L | public zero-shot Charades mAP 32.4 and ActivityNet retrieval 49.1/51.3; frozen-backbone 53.2 Charades and 64.6 SSv2 are explicitly probe results | released 8-frame factorized temporal video-text encoder | eligible; official Apache-2.0 code, CC-BY-4.0 non-software materials statement, and complete 2,319,542,188-byte checkpoint, SHA-256 `eb951c65…0936b` |

The bounded candidate set is exactly EgoHOD EgoVideo-L zero-shot,
VideoPrism-LvT-L zero-shot, and V-JEPA 2 ViT-L with one frozen public-only
multilabel probe. Their exact official repository commits, model revisions,
local file SHA-256 values, preprocessing, frame counts, prompts/probe recipe,
expected VRAM, and limitations are in schema 10 of the canonical proof config.
No fourth candidate may be added after any result.

The public activity fixture is now fixed without opening model outcomes. It
uses 96 verified 5–60-second first-person Charades-Ego videos: 48 development
items from 24 public subjects and 48 untouched holdout items from 19 different
public subjects. Subject and video overlap are both zero. The frozen public
action-code mapping supplies multilabel versions of the existing eight coarse
contexts; the least represented label has 9 development positives and 8
holdout positives. The external row/file manifest—including public IDs and
per-file hashes—remains outside Git. Its commitment is
`7a44e6cd…2441f8`; the official video and annotation archive hashes are
`3df448ab…b1110` and `ee54787f…e66e`.

Candidate selection uses four subject-group cross-fitted development folds.
Label thresholds and explicit abstention margins are learned only from the
other folds; V-JEPA's eight independent L2 logistic probes are also trained
only on other-fold subjects. A candidate is eligible to win only with macro-F1
at least `0.60`, worst-class recall at least `0.40`, coverage at least `0.80`,
zero unaccounted failures/external calls, and the frozen positive directional
tests on ordered versus shuffled and repeated-center motion clips. Eligible
candidates are ranked once by macro-F1, worst-class recall, coverage, temporal
sensitivity, peak VRAM, runtime, then candidate ID. Exactly one winner is
refit/calibrated on all development data and sealed before the untouched
holdout. Winning development is not a public-gate pass.

The run is single-process on one H100 80GB: at most 1.5 hours per candidate,
4.5 candidate GPU-hours total, 6 additional hours for the complete public
holdout, and—only after a complete holdout pass—8 hours for governed C. Public
and governed incremental storage ceilings are 160 and 80 GiB; direct monetary
cost is zero and institutional GPU time is accounted. This exact narrowing is
covered by the user's existing broad GPU authorization. No DDP or other GPU
topology is authorized. All media, annotations, weights, predictions,
embeddings, probes, caches, and logs stay outside Git; no restricted mount is
permitted during public work, and inference is network-disabled after cache
preparation.

At amendment freeze, no candidate inference, public holdout inference,
governed-C reopening, LTX work, generation, or synthetic learner score has
occurred. The next authorized operation is implementation qualification and
the one frozen public-development comparison. The complete extractor must
still pass every unchanged strengthened holdout gate before C can reopen.

Before candidate inference, implementation qualification resolved two local
runtime details prospectively. EgoHOD EgoVideo-L uses the official ViT-L/14
input size of 336 pixels with the already frozen 16 uniform frames (the earlier
224-pixel prose was incompatible with that checkpoint architecture); its
complete state dict must load strictly without fetching an OpenAI
initialization checkpoint. VideoPrism's public `c4_en` SentencePiece artifact
is pinned at SHA-256 `1e5036be…f8ec`, and V-JEPA 2's local model and processor
configs are separately hashed. The CUDA container is pinned at
`f274f1ac…5792`. Exact isolated dependency versions and the requirement for a
hashed install report before inference are recorded in the canonical config.
No model output had been observed when these compatibility details were
committed.

The CPU dependency preflight then exposed that isolated `pip --target`
resolution would otherwise pull a future Torch/CUDA stack through Timm. Before
any model load, the EgoHOD environment was narrowed to its actual inference
imports. The adapter supplies Torch-native equivalents for the three Timm
layer utilities and two MMEngine initialization helpers used by the official
model, plus no-op modules for unused `ipdb`/OpenCV debug visualization imports.
The checkpoint still must load every tensor with no missing or unexpected
keys, and the forward implementation remains official EgoHOD code. A runtime
guard rejects any dependency target that shadows the pinned Torch or
Torchvision container versions.

The repaired CPU-only preparation then passed in 438 seconds without loading
any model. It verified three checkpoint files, four pinned repositories and 96
public fixture items, and sealed 88 installed public distributions under
dependency-manifest commitment `20bc4ad8…bca`. The public run had no restricted
mount and recorded `model_inference_executed=false`. Candidate loading remains
gated on that exact manifest. The next operation is blind single-item resource
sizing for each frozen candidate, followed by the one frozen development
comparison; no holdout or governed-C outcome has been opened.

The sizing command is itself frozen before model loading. For each candidate it
uses only manifest ordinal zero from the public development partition, runs the
ordered clip once, and discards the numerical output immediately after finite
width validation. It cannot retain a score, prediction, public ID or label and
cannot compute a scientific metric. Each single-process one-H100 job is capped
at 30 minutes, for 1.5 aggregate sizing GPU-hours, 160 GiB within the existing
public storage ceiling and zero direct monetary cost. This adds at most 1.5
GPU-hours to the previously frozen path, making the through-C ceiling 20.0
GPU-hours. The development, holdout and C ceilings themselves are unchanged.

The first EgoHOD sizing attempt stopped before model-state loading or inference:
Torch's `weights_only=true` loader found an OmegaConf `ListConfig` in the
official checkpoint and refused to deserialize it because that class was not
allowlisted. No score or prediction was produced. The prior dependency manifest
`20bc4ad8…bca` remains preserved but is not valid for candidate inference. A
new engineering-only repair is frozen before retry: retain `weights_only=true`,
install pinned OmegaConf/PyYAML/ANTLR runtimes, ask the pinned Torch scanner for
the checkpoint's unsafe globals, require an exact match to the 13 names recorded
in the canonical config, and allow only those names during deserialization. Any
new global stops the run before loading. The repaired environment must receive
a new dependency-manifest commitment before sizing resumes.

That CPU-only re-preparation passed in 435 seconds without model inference. It
sealed 91 installed distributions at commitment `5fb4a9d3…e81d`, explicitly
superseding the unusable environment commitment while preserving its record.
The three checkpoints, four repositories and 96 public items were reverified;
no restricted mount was present. Blind sizing may now restart under the
unchanged one-item, one-H100 controls.

On the restarted EgoHOD sizing load, all 13 statically reported globals matched
exactly. Torch's safe builder then stopped before state loading on the one
dynamically constructed type that its scanner cannot enumerate,
`numpy.dtypes.Float64DType`; the checkpoint pickle contains only one NumPy dtype
constructor. The repair therefore adds only that exact runtime type to the safe
context. It does not broaden the static-name gate or permit
`weights_only=false`, and no inference output was produced by either attempt.

The third safe-load attempt passed and completed the label-blind EgoHOD sizing
forward pass. Its committed record reports width 8/8, 12.71 seconds total and
2.432 GiB peak device use, with no external call, retained score/prediction or
scientific metric. The batch job then exited only because the terminal-safe
serializer rejected the lowercase public candidate ID after the record was
atomically sealed; the record commitment `fa6a684a…1280` verifies. The terminal
schema is therefore narrowed to omit candidate and partition strings for all
sizing, development and selection summaries. This output-only repair does not
rerun or alter the valid EgoHOD sizing record.

The first VideoPrism sizing load then stopped before model construction because
the pinned official `models.py` imports its TensorFlow-backed tokenizer module.
That module is needed there only for the `Tokenizer` annotation and an unused
`SentencePieceTokenizer` helper; this adapter already freezes the same public
SentencePiece bytes, canonicalization and 64-token padding locally. The
prospective import-only repair supplies exactly those two namespace attributes,
makes the hosted/GCS helper raise if invoked, and verifies from the pinned source
that no other tokenizer attribute is referenced. It does not modify VideoPrism
parameters, preprocessing, text IDs or forward computation and adds no new
dependency.
The next import check exposed the same narrow issue in official `utils.py`,
where `gfile.GFile` is used only if a checkpoint path is nonlocal; the frozen
checkpoint is an existing local file and follows the preceding NumPy branch.
The compatibility namespace therefore adds exactly that one attribute and
makes it raise if reached. A pinned-source guard requires the complete gfile
reference surface to remain exactly `{GFile}`. TensorFlow is still absent, no
hosted path can execute, and model/preprocessing behavior is unchanged.

VideoPrism and its local checkpoint then loaded, but Torch's module discovery
encountered the temporary TensorFlow namespace during preprocessing and stopped
before the numerical forward. The import adapter now marks that namespace and
removes it immediately after official VideoPrism import. The already imported
utility retains only its local raising `gfile` object, so hosted fallback stays
impossible while later libraries correctly observe that TensorFlow is absent.
No score or prediction was produced by this attempt.

The repaired VideoPrism sizing then passed: width 8/8, 7.72-second load,
43.92-second one-item forward, 51.64 seconds total and 6.629 GiB peak use versus
the 24 GiB ceiling. It made zero external calls and retained no score,
prediction or scientific metric. Its sealed record commitment
`6619aa2e…e1e7` verifies. Two of three blind sizing checks now pass; V-JEPA 2
remains unopened.

V-JEPA 2 also passed blind sizing: width 1024/1024, 26.00-second load,
2.35-second one-item forward, 28.37 seconds total and 2.498 GiB peak use versus
the 32 GiB ceiling, with commitment `89b03f90…a617`. All three candidates
passed with zero external calls, crashes, invalid output widths or retained
scores; aggregate measured runtime was 92.72 seconds. The next authorized gate
is the already frozen 48-item public-development comparison, run sequentially
on one H100. Holdout and C remain unopened.

The one frozen public-development comparison then ran once. All three
candidates completed all 48 items with zero crashes, truncations, invalid
records or external calls. EgoHOD had macro-F1 0.6680, worst-class recall
0.6471 and coverage 1.0, but every temporal-control floor failed. VideoPrism
had macro-F1 0.6410, worst-class recall 0.5556 and coverage 1.0; its ordered
score exceeded shuffled on average, but it missed the repeated-frame mean and
both positive-fraction floors. V-JEPA 2 had macro-F1 0.5733, worst-class recall
0.2222 and coverage 1.0; it missed both classification floors plus the
shuffled-order mean and positive-fraction floors. Thus `0/3` candidates were
eligible. Selection commitment `e11727dd…c29d` verifies. The frozen action is
`NO_GO_NO_ELIGIBLE_CANDIDATE`: no winner, holdout opening, governed-C rerun,
episode-plan rebuild, LTX work or synthetic learner work is permitted, and no
candidate or threshold may be added or relaxed in this task.

### Prospective mechanistic training-tuple calibration amendment

The activity-checkpoint result above remains a valid fourth no-go, under
commitment `e11727dd…c29d`. On 2026-08-02 the user authorized a new scientific
construct before any further public result, C measurement, generator outcome,
synthetic corpus, or synthetic learner score. The question is now whether the
pipeline can measure and match the lexical-grounding opportunities that the
pinned learner actually consumes. This does not call the old temporal test
wrong after the fact: broad settings can often be recognized from static
objects or backgrounds, so frame order is a poor validity test for labels such
as meal or reading. Broad activity/context is therefore retained only as a
descriptive nonblocking quantity. Temporal sensitivity is blocking only on
localized, genuinely order-dependent public actions: open/close, take/put,
sit-down/stand-up, and turn-on/turn-off.

The learner-effective unit is one tuple whose German audio passes the frozen
shared adapter and whose exact accepted English text and accepted ASR segment
bounds are paired with an in-bounds utterance-centered video window. OPUS-MT
does not align individual English translation tokens back to German word
timestamps, so all noun/adjective mentions in a translation share the accepted
segment midpoint and before/during/after interval; no English-token timestamp
is fabricated. This executable clarification was committed before any new
fixture, C, generator, or synthetic learner outcome. An adapter abstention
contributes no tuple, lexical exposure, recurrence, or credited duration. The
seven frozen axes are adapter-qualified yield; noun/adjective exposure;
utterance-centered referent visibility, dominance, and ambiguity;
cross-episode recurrence; adjective–attribute contrast; hand/action coupling;
and egocentric sensor regime. The first five are validity-critical. Activity
mixture is not an eighth gate, and no omnibus fidelity score is permitted.

This construct follows task-relevant evidence rather than global visual
similarity. Natural naming events are distinguished by target size,
dominance, competitors, and temporal stability around naming
([Yu and Smith, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3829203/));
child-view object exposure is long-tailed and recurrent across different
timescales
([Clerkin et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5124080/),
[Clerkin and Smith, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9170168/));
and adjective generalization requires visible contrasts across noun categories
([Leonard et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7201330/)).
These sources motivate the frozen features but do not supply ChildLens labels
or thresholds.

One fixed modular stack is registered, with no post-outcome substitution:
the repaired shared language adapter; NLTK/WordNet plus public `wordfreq`
frequency norms; Grounding DINO plus SAM 2.1 for public-category grounding and
tracking; EgoHOS for egocentric hands/contact/interacting objects; DINOv2 for
governed recurrence embeddings; PE-Core plus deterministic mask measurements
for visible attributes; deterministic sensor metrics; and the already pinned
EgoHOD checkpoint for the localized inverse-action control. Exact repository
commits, model identities, gates, threshold-development grids, and licenses are
in schema 11 of the canonical proof config. Every unresolved checkpoint byte
hash, weight term, offline reload, or dependency is a pre-model no-go; another
model cannot be substituted.

Public qualification is modular and task matched. It uses the already frozen
language fixtures; VISOR hand/contact/active-object annotations; fixed
self-authored audiovisual composites with exact referent timing, masks,
attribute contrasts, ambiguity, and null cases; fixed recurrence pairs;
programmatically controlled motion/blur/brightness/cut clips; and
subject-disjoint Charades-Ego localized inverse-action clips. Public
development may select only values from the frozen numeric grids. Fixture and
threshold commitments must be committed before the sealed holdout opens. Every
one of the seven module gates and the inverse-action control must pass; broad
context cannot cause either pass or failure.

If the complete public gate passes, governance permits a small C-domain audit
only by the sole authorized applicant inside applicant-governed storage. It is
limited to visual labels and the exact accepted English input the learner sees;
it is not human German ASR or translation validation. The frozen sample has 96
utterance-centered events and 24 recurrence pairs across all 18 C children,
selected with a new governed HMAC key. The coding interface hides extractor
predictions and all evaluation, learner, generator, and synthetic outcomes;
reference labels are sealed before predictions open. No external rater receives
access and only compact aggregate metrics and commitments may leave. Because
there is one authorized applicant, no inter-rater reliability claim is
possible; concealed duplicate coding measures only intra-rater consistency.

The new governed gate is explicit rather than a retroactive relaxation of the
old eight-axis gate. All five critical tuple axes must pass, at least six of
seven axes must be measured, per-axis missingness must not exceed `0.20`, and
the eligible public-ontology mapping must cover at least `0.60` of all accepted
noun/adjective mentions. Synthetic plans may then match only approved aggregate
bins using public words. Generated attempts and the accepted corpus are
measured separately; all critical axes and at least six axes total must meet
the frozen TV/Wasserstein tolerances, with no severe registered joint miss.

The active amendment commitment is `c9a48206…adaf`; the pre-clarification
commitment `fed6a3dc…4222` remains in Git history. Its bounded qualification route
uses one H100 80GB process: at most 1 GPU-hour for blind sizing, 6 GPU-hours for
public development plus holdout, and—only after public and human-transfer
passes—8 GPU-hours for governed C, for a 15 GPU-hour aggregate ceiling, 200 GiB
public storage, 100 GiB governed incremental storage, and zero direct monetary
cost. This is within the user's existing broad GPU authorization. DDP is not
authorized. LTX retains its separate final-topology/resource gate and the
synthetic corpus remains capped at exactly one accepted credited hour.

The public artifact subgate subsequently passed under dependency-manifest
commitment `8c787a01…b527`: all four repository archives and ten public
weight/resource files were resolved to immutable SHA-256 values (14.62 GB),
with no model inference and no restricted mount. Three preceding preparation
attempts were engineering-only failures caused by missing container transport
dependencies and opened no scientific outcome. The official MIT-licensed
EgoHOS project distributes its checkpoint bundle, but that archive contains no
separate license file; the eligibility record is limited to academic local
prototype processing and makes no broader commercial-use claim. At that point,
exact local dependency installation, network-disabled reload, and label-blind
sizing remained; public development and holdout were unopened.

The exact 53-package overlay was then prepared successfully in 8 minutes 45
seconds under runtime-manifest commitment `968f2570…1f10`. The preparation
mounted no restricted data and executed no model. This is a resource-only
runtime PASS, not extractor qualification; the next gate remains the
network-disabled one-H100 reload of all seven axes plus the separate action
control, with no labels, retained scores, or scientific metrics.

Before that reload, the exact runtime overlay and compatibility surface were
frozen and clarified before inference. A second label-blind sizing attempt
then exposed that the official NLTK resource archives had been extracted as
top-level resource directories while NLTK 3.9.1 resolves them under
`taggers/` and `corpora/`. The attempt stopped before any model loaded. The
active runtime commitment `ee70ae31…b41b` therefore adds one narrow adapter:
after verifying every resource file against the already sealed manifest, it
creates scratch-only namespace symlinks without modifying source bytes. The
prior runtime commitments `adcac3fc…b285`, `6c3c76e3…dec1`, and
`c59a81f4…0acc` remain preserved. Earlier clarifications record
that seven calibration axes plus the separate blocking order-action control
require eight sizing modules; the other pins the PE-Core loader's required
`einops==0.8.0`. They retain
the seven registered models. Grounding DINO uses its own commit's official
pure-PyTorch deformable-attention fallback because the base container has no
compiler; SAM 2 disables its officially optional CUDA extension; and the
EgoHOS shim supplies only unused `mmcv.ops` registry symbols and must fail if
the selected Swin/UPerNet path calls one. The pinned public BERT snapshot and
the EgoBabyVLM PE-Core loader are additional hashed dependencies. The sizing
run remains label blind, retains no scores, uses one item per component, and is
capped at one H100 GPU-hour.

The active overlay was resealed successfully in 8 minutes 38 seconds under
runtime-manifest commitment `9810a618…48f9`, again with no restricted mount or
model inference. It authorizes the unchanged label-blind sizing gate only; it
is not an extractor qualification result.

That sizing retry cleared the adapter, lexical, and sensor checks, then stopped
before Grounding DINO construction because the pinned model source imports an
unused visualization class whose module alone requires unpinned Matplotlib.
The class is not otherwise referenced in the selected model file. Before any
model inference, active runtime commitment `623225bf…09e4` therefore froze a
single-line source repair: verify the original file at
`cdfb48d5…a5b8`, remove exactly the unused import, and require patched hash
`0da7cea7…c671`. Model computation, weights, fixtures, thresholds, and gates do
not change. The `ee70ae31…b41b` runtime remains preserved in the prior list;
the active overlay was resealed in 8 minutes 39 seconds at runtime-manifest
commitment `03c15506…2c15`, again without model inference or a restricted
mount. Label-blind sizing may now resume.

Before that sizing run, the task-matched public-fixture implementation was
made exact at commitment `506a1f41…251d`, without opening a model outcome. The
blocking action module uses the pinned EgoHOD checkpoint with three fixed
prompts for each of eight directions: open/close, take/put, sit down/stand up,
and turn on/turn off. Official Charades-Ego action codes are paired by object;
eligible localized intervals must be first-person, verified, 1–12 seconds,
free of an overlapping opposite action, and disjoint by subject and video from
the prior 96-item broad-context fixture. Development and holdout each contain
six clips per direction under the frozen public-seed ordering. Exact action
margin, reversal, and repeated-center semantics are registered in the
canonical config. The remaining task-matched fixture manifests and byte hashes
must still be prepared and committed before development inference.

The ensuing label-blind sizing job reached the pinned Grounding DINO forward
pass and stopped on `E_TUPLE_GROUNDING_NONFINITE`. Inspection of the immutable
official implementation established that `ContrastiveEmbed` deliberately masks
inactive text tokens and pads unused `max_text_len` positions with negative
infinity; the official sigmoid maps those padding sentinels to finite zero. The
run used no fixture labels and retained no prediction, score, scientific metric,
governed C value, generator outcome, or synthetic learner outcome. Before any
retry, sizing-validation commitment `afc936f7…a2d5` freezes a stronger exact
check: active positions defined by the pinned tokenizer attention mask must be
finite, every complementary padded position must be negative infinity, all
post-sigmoid scores must be finite, and every normalized predicted-box
coordinate must be finite within `[0,1]`. This clarification changes no model,
weight, prompt, fixture, threshold, or scientific gate.

The sizing retry under that rule passed the adapter, lexical, sensor, Grounding
DINO, SAM 2.1, and DINOv2 reloads. It then stopped during import of the pinned
EgoBabyVLM alignment package because its initializer imports `submitit`; this
occurred before PE-Core construction. The approved base container also lacks
`cloudpickle`, Submitit's required runtime dependency. No fixture label,
prediction, score, or scientific metric was retained. Before another model
attempt, active runtime commitment `eb878d8c…fbea` freezes exactly
`submitit==1.5.3` (MIT wheel `ccc35100…1795`) and `cloudpickle==3.1.1`
(BSD-3-Clause wheel `c8c5a442…50e`) and verifies both hashes before install.
All earlier runtime commitments and the 53-dependency manifest
`03c15506…2c15` remain preserved. The new 55-dependency overlay must be prepared
and sealed before label-blind sizing resumes; no model, prompt, fixture,
threshold, or scientific gate changes.
