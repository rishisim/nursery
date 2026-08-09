# Synthetic corrective weak-alignment study v1

## Scope and question

This protocol tests one narrow synthetic claim: whether synchronized, non-semantic event cues available only during training can change an incorrect or genuinely ambiguous word-to-action mapping into a correct lexical representation that transfers to independently generated action instances and never-trained compositions. Cross-occurrence accumulation and joint noisy alignment are motivated by [Yu and Smith (2007)](https://doi.org/10.1111/j.1467-9280.2007.01915.x) and [Fazly, Alishahi, and Stevenson (2010)](https://doi.org/10.1111/j.1551-6709.2010.01104.x). It does not test infant learning or ecological validity. A null or adverse result is scientifically valid.

The sealed `CONFIRMATION_STOP` for the prior line is authoritative. That line had perfect top ordering in every action cell and an active sensor path that preserved the learner's ordering. Its 281xxx development identifiers and untouched 289xxx confirmation reserve are permanently quarantined. This protocol has new code, a new construct, and the disjoint 320--329 identifier family.

## Data-generating population

Each corpus independently draws an opaque lexicon and a separated, row-stochastic action-prototype geometry over four primitive and three manner meanings. Prototype rows have independently drawn off-diagonal mass and a random coordinate permutation; analytic rejection requires a self-versus-cross dot-product margin of at least 0.25. The same latent prototypes generate noisy training and evaluation instances, but independent RNG namespaces generate every instance, and prototype tables never enter learner-visible records. Four primitive×manner compositions are withheld while every constituent remains exposed in other compositions, following the constituent-preserving split principle in [Keysers et al. (2020)](https://openreview.net/forum?id=SygcCnNKwr). Training episodes are temporally extended bags of three to five candidate events plus an explicit latent null, drawing on multiple-instance bags ([Dietterich, Lathrop, and Lozano-Perez, 1997](https://doi.org/10.1016/S0004-3702(96)00034-3)) and explicit weak-temporal background modeling ([Lee et al., 2021](https://doi.org/10.1609/aaai.v35i3.16280)). Corpora independently vary ambiguity stratum, action geometry, candidate sets, visibility, speech/action lag, grounded-utterance rate, repetitions, side informativeness, and side noise. The correction stratum makes a stable foil more frequent than the true event; the ambiguity stratum makes language-only evidence competitive; the recoverable stratum permits language-only cross-occurrence learning. No seed-invariant factor lattice is used.

Training, lexical evaluation, composition evaluation, presence evaluation, and zero-exposure leakage controls use independent labeled RNG streams. Evaluation instance identifiers never overlap training identifiers, and held-out compositions are excluded from both utterance targets and every candidate event during training. Truly zero-exposure words are leakage controls only, because an unseen atomic mapping is not learnable.

## Side channel and causal controls

The raw one-dimensional synthetic sensor stream carries only event energy/boundaries. Its amplitude and position are independent of the lexical surface form and action identity conditional on event/null alignment. A detector integrates energy in every candidate interval and outside all intervals, producing event-plus-null logits. It never receives a word, concept index, answer key, or evaluation record.

All conditions use identical language/visual episodes, model seeds, update counts, and candidate pairing:

- `synchronized`: detector scores from the paired stream;
- `shuffled`: an exact candidate-count-blocked evidence derangement whose donor has a different composition and whose complete learner-visible detector-evidence marginal is byte-identical within each block;
- `shift_minus` and `shift_plus`: equal non-circular shifts with zero fill;
- `absent`: no side input;
- `uninformative`: constant-zero side input;
- `corrupted`: the target/null detector peak is moved to a prespecified wrong state;
- `oracle_alignment`: event/null positive control isolated to training;
- `exact_window`: event nearest speech, with no side evidence, as the old comparator.

The learner and each mechanism mutation are also run under a coupled corpus×model bundle. Synchronized models are mutated by zeroing semantic side weight, disconnecting all evidence, permuting evidence within bags, reinstating the old agreement gate, disconnecting semantic updates, zeroing only the null-side input while retaining ordinary null learning, freezing null updates at the configured initial prior of 0.30 while retaining null-side input, and separately ablating the learned null head to near zero after training.

## Corrective learner

For word occurrence \(i\), every candidate event and null enters a posterior. No event is filtered by the learner's current argmax and side evidence is not gated on learner agreement:

\[
q_i(z=j) \propto \exp\{\beta_l \log p_{\theta_w}(x_{ij})
+ \beta_t \log p(t_{ij}) + \beta_s a_{ij}\}.
\]

The semantic update uses the complete posterior-weighted event sufficient statistic. Thus, when the detector favors an event that the current lexical model ranks below a foil, that event still changes the lexical distribution. A separately updated null head uses event-versus-null posterior mass. Model replicates have different Dirichlet initializations and stochastic episode orders at every epoch; they are not metadata jitter, consistent with variance-aware model comparison ([Bouthillier et al., 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)).

Before population construction, 32 fixed hand-authored disagreement cases require the active path to change a wrong unique top into the true unique top in at least 75% of cases. Zero, disconnected, permuted, old-gated, and semantic-disconnected mutations may flip at most 25%. Null-head ablation must selectively harm null rejection without changing lexical semantics.

## Evaluation firewall and endpoints

The worker that fits and predicts receives only visible training episodes, condition-specific training evidence, and visible evaluation prompts. Protected training oracle rows and evaluation keys are stored in a different file and are read only by the serial parent after predictions exist. The serialized model contains lexical distributions and side-free null rates only. Evaluation rejects side, detector, oracle, target, concept, and answer fields recursively. This training-only boundary follows privileged-information deployment logic ([Karlsson et al., 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html)) and leakage-audit discipline ([Kapoor and Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

Co-primary endpoints are:

1. strict unique-top-1 lexical acquisition over independently generated held-out instances of exposed primitive and manner meanings;
2. strict unique-top-1 action transfer over independently generated instances of the four never-trained compositions.

Ties are wrong. Concepts/compositions receive equal weight before corpus averaging. Mean rank, MRR, multiclass log loss, multiclass Brier score, and tie rate are descriptive secondary metrics and cannot rescue a failed top-1 claim; log loss and Brier are retained because they are proper distributional scores ([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437)). No secondary p-values are produced. Presence/null balanced accuracy is a mechanism endpoint; present/null effect vectors within 1e-12 of equality fail separately for synchronized-minus-absent and synchronized-minus-shuffled, while high correlation alone remains diagnostic and crossed semantic/null mutations must pass.

## Headroom and qualification

Construction micro-fixtures and the 16-corpus construction cohort are preallocated and excluded from scientific inference. They are not selected or filtered by effect direction. For both co-primary endpoints, every absent or disrupted control must have accuracy at most 0.80 and at least 20% wrong/tied items. Oracle alignment must be at least 0.90. At least 12 action-geometry Gram-matrix digests and 12 episode-layout digests must be distinct, both primitive and manner mean pairwise prototype distances must span at least 0.02 across corpora, every prototype set must meet the 0.25 separation margin, and at least eight held-out split digests must be distinct. All seven scalar corpus-factor draws must vary. At least half the audited corpus×condition×endpoint cells must show model-replicate SD of at least 1e-6.

A failed frozen package is preserved and retired. It is never patched or rerun; a replacement must use a fresh package version and fresh namespace family.

## Frozen inference and decision rule

The corpus seed is the inferential unit. Three paired stochastic model replicates are averaged within corpus in frozen seed order using `math.fsum`. For endpoint \(e\) and corpus \(i\):

\[
B_{ie}=\max(A_{ie,absent}, A_{ie,shuffle}),\qquad
d_{ie}=A_{ie,synchronized}-B_{ie}.
\]

The global GO claim is an intersection-union test ([Berger and Hsu, 1996](https://doi.org/10.1214/ss/1032280304)): both endpoint estimands must pass every gate. Each mean effect must be at least +0.10, its one-sided 95% corpus-Student-t lower bound must be strictly greater than +0.10, synchronized performance must have a one-sided lower bound more than 0.20 above exact prompt-weighted chance, at least 60% of corpus effects must be positive, and no more than 20% may be at or below -0.10. An all-identical effect vector fails as degenerate. Because both alternatives are required, no alpha split is used. A 20,000-replicate paired corpus bootstrap is a sensitivity analysis only. Secondary metrics are descriptive. Corpus-level t summaries are motivated by heterogeneous-group inference ([Ibragimov and Müller, 2016](https://doi.org/10.1162/REST_a_00545)).

Development interpretability requires non-perfect disrupted controls, an oracle lower bound above 0.85 and lift above 0.15, equivalence of absent/uninformative/disconnected within ±0.03, a corrupted upper bound below +0.03, detector manipulation in both present and null strata, side-only leakage probes, and crossed semantic/null pathway checks. A positive development result can yield `GO` only if synchronized acquisition also exceeds both signed shifts and the exact-window comparator and the semantic-side, semantic-update, alignment-order, and old-gate mutations fail their positive controls. A cohort whose primary acquisition claim fails after structural controls pass yields scientific `STOP`; a positive primary result without causal attribution yields `REVISE`. Threshold rationales are frozen in `synthetic_corrective_alignment_v1_threshold_rationale.md`, following prospective simulation discipline ([Morris, White, and Crowther, 2019](https://doi.org/10.1002/sim.8086)).

## Parallel execution and one-shot transaction

The work plan is the exact corpus-seed×model-seed cross-product. One process bundle runs every matched condition and mutation for that pair. Spawned workers have isolated temporary and shard roots and all BLAS/OpenMP thread counts set to one. Each shard contains a chained exact-count operation ledger, exact file manifest, and completion seal. The parent requires the exact unique unit set, validates every shard, merges in frozen corpus/model/condition order, scores only after reading protected keys, and performs floating reductions serially.

Jobs=1 and jobs=4 must produce byte-identical adjudication files on non-scientific micro-fixtures. Jobs=4 is frozen only if measured speedup is at least 1.20 and CPU/RAM/disk telemetry supports it. Runtime telemetry is excluded from adjudication bytes.

The future development runner first verifies the actual parsed snapshot, authorization, output, staging and claim paths; exact argv; job count; CWD; interpreter and runner hashes; thread environment; exact package/snapshot manifests; pristine output; zero outcome registry; and sealed registries. It then atomically creates a non-replayable claim before any guarded development operation. Work occurs under a fixed hidden staging root. Only a complete, manifested, adjudicated output is atomically renamed to the exact final root. Failure or interruption consumes the authorization but creates no scientific output. There is no confirmation command.

## Package sequence

Before any official attempt-4 micro-fixture is created, an atomic prequalification design lock binds every non-config tracked source byte, the exact pre-freeze YAML, the exact anticipated YAML produced by the sole `protocol.status: pre_freeze` to `frozen` line edit, the Python/module environment, thread settings, zero-outcome registry, pristine scientific transaction paths, and a manifest of all earlier failed/superseded corrective-package evidence. The lock itself consumes attempt 4: a crash or failure retires package v1. Construction may begin exactly once only after the locked jobs=1/jobs=4 micro contracts, byte-equivalence report, benchmark, manifests, and zero-outcome state pass; an atomic construction-attempt receipt is created before construction work.

After construction passes, the only permitted tracked change is that exact status-line edit. A transition verification must reproduce the locked source, raw configuration, environment, fixture contracts, manifests, receipt, and pristine state before snapshotting. The pre-freeze evidence manifest binds the design lock, construction receipt, and transition verification. The frozen snapshot must reproduce the transition verification exactly and its environment record must equal the qualification environment.

All source, configuration, analysis, sample size, tests, and traceability are then frozen before exactly one excluded rehearsal. The rehearsal uses 321xxx only, suppresses scientific inference, and is independently recomputed from persisted learner inputs with byte identity required. Development (322xxx) and confirmation (329xxx) remain at zero outcomes; the confirmation reserve is never generated or inspected.
