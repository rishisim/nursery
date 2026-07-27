# Synthetic-video governance and preregistration

**Phase 3 status:** **INFRASTRUCTURE/PERMISSION GATE**

**Evidence cut-off:** 2026-07-27

**Authority boundary:** this record freezes every decision that can be made
without examining restricted ChildLens content or participant-identifying
records. It does not authorize ChildLens access, common evaluation assets,
training, generator work, generation, TTS, or scientific evaluation. BabyView
is unavailable, out of scope, and supplies no empirical ancestry.

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

Status meanings are `FROZEN`, `REQUIRES CONFIRMATION`,
`INFRASTRUCTURE/PERMISSION GATE`, and `NO-GO`.

| Topic | Status | Evidence or frozen decision | Owner | Exact unblock action |
|---|---|---|---|---|
| Learner/runtime | `FROZEN` | Public/dummy single-L4 bridge/runtime PASS; no scientific score; DDP untested | Technical lead | Preserve immutable pins and rerun acceptance inside the approved governed environment |
| ChildLens academic use and aggregate reporting | `FROZEN` | The signed request explicitly covers ChildLens videos/annotations for non-commercial model calibration/evaluation in grounded learning, with aggregate-only reporting through July 2027. Paul Grohmann accepted the form and granted access | Authorized applicant | Keep every use within that scope, applicant-only, and cite DOI `10.17617/4.fe` |
| ChildLens local storage | `INFRASTRUCTURE/PERMISSION GATE` | Non-content inspection found the current ChildLens sparsebundle and its APFS volume report no encryption. Permission is established, but this storage does not meet the protocol's secure-storage rule | Authorized applicant | Move the corpus to an encrypted APFS volume or encrypted sparsebundle under applicant-only access, verify encryption at both image/volume level, then securely retire the unencrypted copy |
| ChildLens inventory | `REQUIRES CONFIRMATION` | The user authorizes ChildLens access and reports downloaded videos, but no content, drive, or identifying records were inspected; \(H\), \(r\), split feasibility, and power inputs remain unknown | Authorized applicant | After storage qualification, run the read-only aggregate inventory in this document inside the governed boundary |
| Governed CUDA | `INFRASTRUCTURE/PERMISSION GATE` | No approved restricted-data CUDA environment is evidenced | Institution/IT security | Approve a system against every acceptance item below and retain the signed qualification record |
| DDP/scaling | `REQUIRES CONFIRMATION` | One L4 only; upstream reference is four processes | Technical lead | Size the full run blindly; if more than one GPU/process is required, pass the mandatory public/dummy DDP preflight before any restricted execution |
| ASR/translation | `REQUIRES CONFIRMATION` | Interface and selection rules are frozen; exact local model weights are not yet selected | Language/technical lead | Run the bounded public-language selection substage below, then freeze revisions, hashes, licenses, and thresholds before ChildLens processing |
| German human validation | `FROZEN` | No German-speaking human annotator is available, and the agreement prohibits making the dataset accessible to third parties | Authorized applicant | Use no human rater and retain only explicitly model-derived claims; a future rater requires separate MPI authorization |
| \(C,H,r\), margins, seeds | `REQUIRES CONFIRMATION` | Blind rules and bounds are frozen below; numeric values require permitted aggregate inventory and real-only variance | Authorized applicant in locked statistician stage | Apply the registered algorithms without synthetic results and sign/hash the completed config amendment |
| Common benchmark | `FROZEN` | Exactly one asset for all arms; it may use public resources and authorized \(C\) only | Evaluation custodian | Build only after Phase 3 gates clear; hash and seal it before learner training |
| Score sealing/unblinding | `FROZEN` | Synthetic-arm scores remain inaccessible until the real-only gate passes; this is a disclosed single-operator protocol | Authorized applicant | Use separate procedural roles, coded outputs, append-only commitments, and the ordered unblinding script below |
| Cost comparison | `FROZEN` | Prospective like-for-like marginal and fully loaded ledgers; sunk ChildLens collection is not zero | Authorized applicant in locked cost stage | Insert pre-generation unit prices and distributions, then hash/sign the ledger |

Phase 3 cannot be marked PASS now because the current ChildLens sparsebundle is
unencrypted and a governed CUDA environment is not yet qualified. Dataset
access, non-commercial academic model calibration/evaluation, and aggregate
reporting are established; third-party access is prohibited. The remaining
status is a remediable infrastructure gate, not a permission, engineering, or
scientific failure. A confirmed consent/license incompatibility, inability to
form an independent evaluation split under the rules below, or absence of any
compliant restricted-data compute path after documented alternatives are
exhausted changes the status to `NO-GO`.

## Governance and permission matrix

“Governed boundary” means the institutionally approved restricted-data system,
not this repository, ordinary cloud storage, Hugging Face Jobs, external APIs,
or a personal unmanaged workstation. ChildLens permission is established by
the accepted signed agreement. The following rules operationalize its secure,
applicant-only, non-commercial handling conditions.

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
drive, workstation, CUDA host, or third-party rater.

| Material class | Permitted storage/execution | Network and egress | Access roles and logging | Retention and disposition | Git |
|---|---|---|---|---|---|
| ChildLens raw video/audio | Encrypted governed storage and qualified governed compute only | Default-deny outbound; never hosted services/APIs | Named data custodians and authorized preprocessing operators; read/access/export events logged | Source retention follows controlling agreement; custodian verifies deletion of task-local copies | Never |
| Transcripts, ASR, translations, frames, embeddings, ChildLens-derived prompts/statistics | Same boundary and controls as raw unless the steward explicitly reclassifies a named aggregate | No egress by default; no model telemetry | Least-privilege pipeline operators; artifact creation/read/export logged | Derived retention schedule approved before creation; pipeline owner deletes temporaries and custodian verifies | Never, except a specifically approved compact non-identifying aggregate |
| Participant/session identifiers and split ledgers | Separate encrypted governed namespace; direct identifiers never exposed to learner operators | No egress | Data custodian only for identity map; split operator sees opaque IDs; all mapping access logged | Identity map retained/deleted by custodian under agreement; analysis ledger archived as authorized | Never |
| Public/dummy media and public model weights | Repository-external public cache or hosted/public CUDA | Ordinary network allowed only for declared public sources | Technical staff; provenance and hashes logged | Reproducible caches may be deleted; immutable pins retained | Source/config/hash metadata only; not weights/media |
| Synthetic outputs, failed generations, QA labels, manifests | Public-only development outputs may use approved public compute; study outputs are governed because plans/statistics may derive from \(C\) | Study outputs inherit default-deny; public-only outputs may use declared public services | Generator/QA roles; every attempt, access, label, and disposition logged | Retain all attempts through locked analysis in governed run storage, then archive/delete per approved schedule | No media, attempts, prompts derived from \(C\), or row-level manifests; compact approved aggregates/config only |
| Compact permitted aggregates | Governed staging until disclosure review; then approved institutional/repository location | Export only after documented disclosure review | Data steward approves; exporter and exact fields logged | Curated decision records retained with provenance | Allowed only if non-identifying, authorized, compact, and no reconstructive content |
| Human-rater materials | Governed rater interface only; no local downloads/screenshots | No outbound transfer | Named authorized raters, rater manager, and auditor; item access and decisions logged | Rater caches deleted at session end; labels follow approved retention | Aggregate agreement/decision only if approved |

Deletion/archival responsibility is explicit: the data custodian owns raw and
identity records; pipeline owners delete their derived temporaries and caches;
the compute administrator verifies workspace/cache/checkpoint deletion; the
generator lead accounts for every synthetic attempt; the rater manager clears
rater workspaces; the repository maintainer admits only approved compact
records. “De-identified,” a keyed hash, or an embedding does not itself permit
egress.

## Governed CUDA acceptance contract

Before any restricted execution, a signed qualification must record:

1. institution and physical/legal owner, facility or approved tenancy, system
   identifier, administrator, and applicable agreement;
2. encryption at rest and in transit, key owner, encrypted temporary workspace,
   secure boot where required, and backup behavior;
3. named-user authentication, MFA, least privilege, prohibition on shared
   accounts, privileged-access review, and immediate revocation procedure;
4. immutable audit coverage for login, file access, job execution, privilege
   change, removable media, network connection, and export, with retention and
   reviewer;
5. outbound default-deny firewall/DNS/proxy policy. The qualification egress
   test must show a permitted internal endpoint succeeds and undeclared HTTPS,
   DNS, model telemetry, package-manager, object-store, and paste/upload
   destinations fail, with administrator-observed logs;
6. an ingress procedure in which a separate internet-connected machine obtains
   pinned public weights/packages, verifies license/revision/SHA-256, malware
   scans them, and transfers them one-way through an approved staging process;
   the restricted job never gains outbound access;
7. absolute locations and quotas for input, temporary frames/audio, model
   caches, compiler caches, checkpoints, logs, crash dumps, swap, and backups;
   none may resolve to Git, a home-directory sync service, or an unmanaged disk;
8. retention periods and accountable deleters for every location, followed by
   deletion verification that checks ordinary files, caches, recycle areas,
   snapshots/backups, scheduler scratch, and failed-job remnants;
9. the immutable upstream commit, patch/config hashes, public weight hashes,
   resolved environment lock, driver/CUDA/GPU inventory, container hash,
   deterministic settings, and the same fail-closed runtime assertions as the
   public preflight; and
10. capacity qualification from the resource-sizing exercise below, including
    whether the registered run is single-process or DDP.

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

After eligibility rules are frozen and before media review, determine counts
from the permitted number \(N\) of eligible children. The minimum allocation
is 3 training, 2 evaluation, 2 \(C\), and 1 validation child. Allocate children
above those eight by the largest-remainder method toward final proportions
50% training, 20% evaluation, 20% \(C\), and 10% validation (ties resolve
training, evaluation, \(C\), validation). Sort children by the HMAC of
`study_id || child_id`, then deal the required counts in repeating role order
evaluation, \(C\), validation, training, skipping a role once its count is
filled. This makes the assignment deterministic without content or outcomes.
Within a child, all sessions stay in that child's partition. If unequal
duration leaves the training pool too small, \(H\) decreases under the frozen
rule; children are not reassigned after duration is known. If policy forces one
child across operational roles, confirmatory independence fails; session-only
allocation is not an automatic substitute. The independent statistician may
authorize a preregistered leave-one-child-out design only before outcomes and
must narrow the claim accordingly; otherwise this is `NO-GO`.

The authorized aggregate inventory must return, per opaque child and without
content: eligible session count, recorded duration, preliminary technically
readable duration if permitted, date bucket sufficient for overlap detection,
and missing-metadata flags. It must not return direct IDs, filenames, text,
frames, hashes of media, or participant attributes. This is the smallest
missing input for split feasibility and \(H/r\).

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
substage, not with ChildLens. Candidate models must: run fully offline in the
qualified environment after one-way weight ingress; have immutable
revision/file SHA-256 hashes and licenses permitting the research and
restricted local processing; expose no telemetry; support German ASR with word
timestamps and German→English translation; preserve episode/utterance IDs,
word order, start/end timestamps, punctuation-normalized text, and abstention
flags; and fit the sized governed resource.

Use a fixed, redistributable public German speech set plus self-authored German
audio/text covering child-directed vocabulary, overlap, silence, noise, and
long utterances. No large weights are downloaded and no experiment runs in
Phase 3. The later substage passes only if a network-deny test succeeds,
timestamps are monotonic and within audio duration for 100% of non-abstained
items, ID/word/timestamp round-trip tests pass, license/hash manifests are
complete, no crashes or silent truncation occur, and blind selection minimizes
public-set word error rate then translation adequacy under a frozen resource
tie-break. Before launch, the language lead must set public-only numeric WER,
translation, timestamp-error, confidence, and maximum-abstention thresholds
without ChildLens. Confidence below threshold, missing/nonmonotonic timestamps,
empty output, language mismatch, or translation failure causes abstention and
zero credited time.

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

## Exact actions required to clear Phase 3

The package is protocol-complete but infrastructure-gated. Dataset access,
project-specific model calibration/evaluation, and aggregate reporting are
established. The remaining actions are:

1. **Encrypted storage remediation:** create an applicant-only encrypted APFS
   volume or encrypted sparsebundle, transfer and verify the restricted corpus
   without changing its contents, confirm both container/image and volume
   encryption, and securely retire the unencrypted source only after
   verification. Raw and row-level material remain non-exportable.
2. **Governed CUDA qualification:** a named system and recorded responses/evidence
   for all ten acceptance-contract items, including observed egress and
   deletion tests.
3. **Read-only aggregate inventory:** after item 1, the authorized applicant
   runs the frozen inventory locally. The user's authorization is already
   recorded; no third party receives media or participant records.
4. **Single-operator controls:** use the frozen staged-role, commitment, and
   scripted-unblinding procedure. No additional researcher is required.

After 1–4, the applicant applies the frozen blind rules to the permitted
inventory; the technical lead performs public-only resource sizing and, if
required, requests approval for the precisely scoped DDP job; and the language
lead performs the bounded public-only ASR/translation selection. Phase 3
becomes PASS only after those records are signed and the machine-readable
fields are filled without synthetic outcomes. Phase 4 remains unauthorized
until then.
