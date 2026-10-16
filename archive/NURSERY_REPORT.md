# Nursery project report

**Prepared:** September 24, 2026
**Repository:** `rishisim/nursery`
**Active branch reviewed:** `research/embodied`
**Planning horizon:** September 24–December 2, 2026

## Executive decision

Nursery should move from a single-motion-generator pipeline to a **modular embodied-motion architecture**.

HOIDiNi should be retained for the narrow capability it currently provides unusually well: detailed articulated hand poses, hand–object contact targets, wrist/hand trajectories near manipulation, and object motion. It should not be responsible for locomotion, whole-body scene grounding, navigation, furniture avoidance, or the eventual infant motion prior.

A separate scene-aware body model should generate the root, pelvis, torso, legs, locomotion, floor contacts, and collision-aware approach to the target. That model should eventually be replaced or conditioned by an infant-aware prior. A fusion layer should reconcile the body and manipulation outputs into one reference before physics execution.

```text
task + generated room
          |
          v
   high-level planner
          |
     +----+--------------------+
     |                         |
     v                         v
scene/body model          HOIDiNi-like module
root and pelvis           hand/object contacts
locomotion                articulated fingers
floor grounding           wrist trajectory
obstacle avoidance        object trajectory
     |                         |
     +------------+------------+
                  v
          motion fusion layer
                  v
         unified reference motion
                  v
       physics controller/simulator
```

This reframes the immediate research question:

> Can Nursery preserve HOIDiNi's useful dexterous manipulation signal while replacing the adult, scene-weak whole-body motion prior that does not fit Nursery's long-term needs?

## Scope and assumptions

This report synthesizes the tracked repository, Git history, README evidence, saved Codex task history, and the two ChatGPT conversations reproduced in the appendices.

The schedule below is proposed because no external deadline was supplied. It assumes continued access to Juno, the existing upstream checkouts and checkpoints, and the current generated mug/desk canary. Each learning task is tied to a concrete implementation decision; the plan does not create a separate reading phase.

The two conversations contain model and literature claims that informed planning. Those claims must be verified against primary papers, official repositories, licenses, released weights, and runnable code before a model-selection decision.

## Project timeline

### July 3–15: natural-versus-synthetic developmental-data framing

Nursery began around evaluating whether controlled synthetic data could match selected properties of natural BabyView-style data while exposing side information unavailable in natural recordings. Early work focused on evaluation design, touch/grounding measurements, ChildLens calibration, and a defensible apples-to-apples comparison.

The scientific motivation was broader than making synthetic video: produce controllable developmental environments whose known state, contacts, geometry, and other modalities could support measurements or learning experiments that natural egocentric data cannot directly provide.

### July 16–25: qualification and structural no-go results

The repository accumulated strict qualification protocols for calibration, data access, leakage control, fixture construction, and reproducibility. Several approaches were stopped when public or ChildLens-derived evidence could not support the intended claims without missingness, weak grounding, or manufactured scoring.

This period established an important norm: failed gates are evidence, not invitations to weaken acceptance criteria.

### July 31–August 9: embodied-simulation prototype

The project built and evaluated a procedural embodied pipeline with rooms, human embodiment, bimanual interactions, rendering, contact evidence, and physics-oriented gates. The work demonstrated substantial engineering feasibility but also exposed limitations in visual fidelity, hand realism, and physical validity. This historical implementation is now archived under `archive/legacy_research/`.

### August 17–25: canonical EmbodiedGen pipeline

The active branch was reset around a smaller canonical workflow. EmbodiedGen became the native room-generation, planning, retrieval/generation, URDF export, and placement system. Nursery retained only a thin adapter and canonical configuration.

A reusable room bank and activity-conditioned static scene builder were completed. Generated rooms and external upstream repositories were kept outside tracked source.

### August 25–September 5: EmbodiedGen to HOIDiNi

The repository added a deterministic scene bridge that preserves the complete generated scene while selecting one manipulated object and its support for HOIDiNi. It then added scene-conditioned HOIDiNi inference from an explicit GRAB prefix.

The mug-lift canary produced correct prefix alignment and lifted the mug kinematically, but the generated body penetrated below the floor. The accepted saved result reached a lowest body vertex of approximately `-0.11228 m`, failing the `-0.05 m` gate.

### September 5–9: HOIDiNi to InterMimic conversion

Nursery converted the HOIDiNi SMPL-X motion, articulated fingers, object trajectory, and coarse wrist-level contact intent into InterMimic's reference format. The isolated loader and forward-kinematics validation passed with sub-micrometre numerical error, establishing format and skeleton compatibility.

This did not establish physical executability or preserved hand–object contact.

### September 9–16: physical execution canary

The repository added a bounded InterMimic launcher using the official general SMPL-X student, the generated room, and the dynamic mug. The revised Juno run initialized correctly and produced finite synchronized state/action traces.

It stopped after 32 of 149 control steps because the expected right-hand contacts were missed for 11 consecutive steps. The mug moved downward rather than upward. The decision record correctly classified the result as `interaction_failure`.

This is the current end of the implemented pipeline:

```text
text -> EmbodiedGen room -> HOIDiNi motion -> conversion -> InterMimic
                                                        -> failed physical interaction
```

### September 24: architectural reframing

The saved planning conversations compared GRAB and OMOMO, reconsidered HOIDiNi's role, and discussed newer scene-aware systems such as HOSIG and InfBaGel. The resulting decision is not to turn HOIDiNi into Nursery's universal scene-aware or infant-aware motion generator.

The next stage is to define and test a modular interface among:

1. a scene/body-motion layer;
2. a dexterous manipulation layer;
3. a fusion layer; and
4. a physics-execution layer.

## Current state

### What works

- Activity text can produce a reusable EmbodiedGen room and composed URDF scene.
- Nursery can deterministically bind a named object and support surface from the complete scene.
- HOIDiNi can run against the selected generated object and produce articulated SMPL-X motion, contact predictions, and object motion.
- The selected mug can be lifted kinematically in the saved canary.
- The HOIDiNi result can be converted into InterMimic's expected reference structure.
- The real InterMimic loader and skeleton reconstruction accept the converted reference.
- The physical executor loads the controller, humanoid, room, and dynamic object and records synchronized measurements.
- Focused local tests cover the active adapters and execution diagnostics.

### What has not been demonstrated

- A physically successful manipulation in the generated room.
- Preservation of hand–object contact through body retargeting and controller execution.
- A scene-aware body trajectory that respects floor and furniture geometry.
- Reliable fusion between independently generated body and hand/object trajectories.
- Navigation or long-horizon activity composition.
- Infant morphology, locomotion, posture transitions, balance, or developmental behavior.
- A successful activity rate across multiple objects, scenes, or seeds.
- A direct experimental connection from the embodied simulator back to the original natural-versus-synthetic developmental-learning question.

### Evidence-backed diagnosis

The current failure is not merely a serialization problem. Format and forward-kinematics compatibility passed, while physical hand contact did not.

The present pipeline asks HOIDiNi's adult GRAB-conditioned full-body trajectory to serve simultaneously as:

- the manipulation plan;
- the scene-grounded body motion;
- the controller reference; and
- an implicit solution to locomotion, balance, and obstacle constraints.

That bundles capabilities with different data requirements and failure modes. Floor penetration is one visible symptom. The more consequential failure is that useful kinematic hand/object behavior did not survive as a physical interaction.

## Open decisions

1. **Exact HOIDiNi output contract.** Decide whether the manipulation module exports contact points, contact timing, finger joint rotations, wrist SE(3) trajectories, object SE(3) trajectories, confidence, or some smaller reliable subset.

2. **Scene/body model.** Reproduce and compare HOSIG, InfBaGel, and at most one simpler baseline against Nursery's requirements. Do not select from paper claims alone.

3. **Fusion formulation.** Determine whether fusion is constrained inverse kinematics, trajectory optimization, diffusion guidance, learned residual correction, or a staged combination. The first implementation should be the smallest method that can expose incompatibility quantitatively.

4. **Ownership of object motion.** Decide whether HOIDiNi proposes the object trajectory, the body planner treats it as a constraint, or physics determines it after contact establishment.

5. **Physics target.** Decide whether InterMimic remains the first execution backend and whether its general student can track the fused reference closely enough, or whether another controller/embodiment is required.

6. **Infant transition point.** Define when adult-body architectural feasibility is sufficient and when infant morphology and motion priors become mandatory. Infant work should not be used to hide unresolved adult fusion failures.

7. **Immediate floor diagnostic.** A minimal HOIDiNi floor constraint is justified only if it cleanly distinguishes invalid source-body motion from lost hand/contact transfer. It is not a new long-term HOIDiNi development program.

8. **Scientific endpoint.** Choose the first Professor-Frank-facing result: an architecture plus validated canary, a comparative model study, or an experiment connecting simulator side information to a developmental-learning measurement.

## Recommended next stage

The next stage should be a **modular feasibility study**, not a broad implementation overhaul.

Freeze one generated room, one object/support pair, one prompt, and the existing failing mug run. Use them as a controlled canary. First measure what manipulation information HOIDiNi provides reliably. Then run a bounded scene/body-model bake-off. Finally, fuse the strongest body candidate with the preserved manipulation constraints and test the result in physics.

The stage succeeds if it produces either:

- one physically executed manipulation whose body respects the scene and whose intended hand contact is measured; or
- a precise, evidence-backed stop showing which interface or model capability prevents that outcome.

A visually plausible animation alone is not success.

## Milestones

### Milestone 1 — Freeze the modular contract

**Dates:** September 24–30, 2026

**Question:** What information should each module own and exchange?

**Concrete steps:**

1. Write one canonical architecture/contract document in the repository.
2. Inventory every relevant field currently emitted by HOIDiNi and consumed by the converter/executor.
3. Classify each field as manipulation-owned, body-owned, shared constraint, or derived validation.
4. Specify coordinate frames, units, rates, joint/body names, confidence semantics, and failure behavior.
5. Define one frozen mug canary and quantitative acceptance criteria.
6. Decide whether the one-run floor diagnostic is informative enough to retain.

**Deliverable:** A reviewed interface specification and frozen canary manifest; no new parallel pipeline.

**Learn for this milestone:**

- SMPL-X kinematic hierarchy and hand representation.
- SE(3), quaternions, local versus world transforms, and resampling.
- Contact representations: binary intent, surface correspondences, forces, and duration.
- How HOIDiNi phase-one contacts relate to its final hand and object trajectories.

### Milestone 2 — Extract and validate the manipulation signal

**Dates:** October 1–7, 2026

**Question:** Which HOIDiNi outputs remain useful when its lower-body motion is discarded?

**Concrete steps:**

1. Implement the smallest extractor at the existing HOIDiNi boundary.
2. Preserve wrist/hand trajectories, articulated finger pose, object trajectory, contact locations/timing, and provenance where available.
3. Compare extracted contact geometry against the saved generated mug mesh.
4. Measure continuity, hand–object distance, penetration, contact duration, and sensitivity to the GRAB prefix.
5. Reject fields that are inferred too coarsely to support physics.
6. Record a compact decision table identifying trustworthy and untrustworthy signals.

**Deliverable:** A validated manipulation bundle for the frozen canary plus a concise decision record.

**Learn for this milestone:**

- HOIDiNi's two-stage generation and optimization losses.
- GRAB annotations, interaction coverage, and articulated-hand data.
- Mesh surface sampling and point-to-surface distance.
- Contact precision/recall and temporal agreement metrics.

### Milestone 3 — Reproduce scene/body candidates

**Dates:** October 8–21, 2026

**Question:** Which available model can own body placement, grounding, navigation, and collision avoidance?

**Concrete steps:**

1. Verify HOSIG and InfBaGel claims against primary papers and official repositories.
2. Check code, weights, licenses, dependencies, input representations, and reproducibility.
3. Select at most two candidates plus one simple grounded-body baseline.
4. Run each on the same frozen scene or document the smallest incompatibility preventing a fair run.
5. Measure floor penetration, furniture collision, target approach, final wrist reachability, motion continuity, runtime, and automation reliability.
6. Choose one body-layer candidate or record a structural no-go.

**Deliverable:** A compact, source-backed bake-off with runnable commands and a body-layer decision.

**Learn for this milestone:**

- Scene occupancy/SDF representations and collision losses.
- Navigation/path planning versus local motion generation.
- Human–object–scene datasets and their body/hand representations.
- Reproducibility auditing for research repositories and checkpoints.

### Milestone 4 — Build the minimum fusion prototype

**Dates:** October 22–November 4, 2026

**Question:** Can a grounded body trajectory satisfy HOIDiNi's manipulation constraints without copying HOIDiNi's invalid lower body?

**Concrete steps:**

1. Use the chosen body trajectory as the root/torso/limb base.
2. Apply HOIDiNi wrist, finger, contact, and object constraints only over the manipulation window.
3. Start with constrained inverse kinematics or trajectory optimization before considering a learned fusion model.
4. Penalize discontinuity, joint-limit violations, scene collision, hand–object separation, and object/support inconsistency.
5. Compare multiple constraint weights without changing the frozen acceptance criteria.
6. Save only the compact chosen configuration and aggregate measurements.

**Deliverable:** One unified reference for the mug canary or a measured incompatibility report.

**Learn for this milestone:**

- Constrained inverse kinematics and trajectory optimization.
- Task-space versus joint-space objectives.
- Constraint weighting, feasibility, and Pareto tradeoffs.
- Motion blending, boundary continuity, and support/contact constraints.

### Milestone 5 — Execute the fused reference in physics

**Dates:** November 5–18, 2026

**Question:** Does the modular reference produce actual hand contact and task-relevant object motion?

**Concrete steps:**

1. Convert the fused reference through the canonical InterMimic boundary.
2. Verify frame-zero root, DOFs, object pose, rates, and coordinate conventions.
3. Run one deterministic canary before any multi-scene scaling.
4. Record pair-specific hand/object contacts if the backend permits; otherwise add the narrowest measurement needed.
5. Evaluate completion, fall/tracking termination, hand-contact agreement, object lift/displacement, scene collision, and synchronization.
6. Diagnose failure as reference infeasibility, retargeting loss, controller mismatch, asset mismatch, or simulation failure.

**Deliverable:** A passing physical mug canary or a defensible controller/interface stop decision.

**Learn for this milestone:**

- Position/torque control, imitation policies, and reference tracking.
- PhysX contact reporting, collision geometry, friction, and mass.
- Retargeting error propagation from wrists/fingers to contact.
- Why kinematic success does not imply dynamic feasibility.

### Milestone 6 — Add infant direction and prepare the research decision

**Dates:** November 19–December 2, 2026

**Question:** How should the validated modular architecture become developmentally meaningful?

**Concrete steps:**

1. Map adult-specific assumptions in the body and controller layers.
2. Review infant motion/morphology sources such as InfAGenC and the prior MIMo-related work.
3. Define an infant-aware interface without implementing every infant behavior.
4. Select one infant-relevant behavior that genuinely stresses the modular split.
5. Reconnect simulator outputs to the original scientific motivation: which extra modalities or controlled interventions can support a natural-versus-synthetic learning experiment?
6. Prepare a concise Professor Frank packet: question, evidence, failure history, architecture, canary result, open risks, and proposed experiment.

**Deliverable:** A go/no-go research memo and one concrete next experiment suitable for discussion with Professor Frank.

**Learn for this milestone:**

- Infant biomechanics, proportions, support strategies, and locomotor development.
- Domain shift from adult motion capture to infant behavior.
- Multimodal model inputs and architectural fusion.
- Experimental design for controlled synthetic-versus-natural comparisons.

## Decision gates

The schedule should stop or change direction at these gates:

- **September 30:** no implementation proceeds without a stable module contract and frozen canary.
- **October 7:** if HOIDiNi cannot expose sufficiently precise manipulation constraints, replace the manipulation layer rather than preserving it by assumption.
- **October 21:** if no scene/body candidate is runnable or wrist-reachable in the frozen scene, decide whether a simpler optimization baseline is scientifically adequate.
- **November 4:** if fusion cannot satisfy scene and contact constraints simultaneously, document the incompatibility before changing models or acceptance thresholds.
- **November 18:** if a feasible fused reference still fails under the controller, treat controller suitability as its own research problem.
- **December 2:** continue toward infant behavior only if the modular adult canary provides evidence that the interfaces are viable.

## What not to do next

- Do not spend a large engineering cycle making HOIDiNi a universal full-body scene model.
- Do not replace HOIDiNi solely because its lower body penetrates the floor; first test whether its manipulation signal is independently useful.
- Do not choose HOSIG, InfBaGel, or another model from reported metrics alone.
- Do not introduce an infant model before establishing whether the modular interface works.
- Do not scale to many scenes, objects, or prompts before one physical canary passes.
- Do not count animation quality, conversion success, or net object force as proof of successful hand–object interaction.
- Do not create versioned duplicate pipelines; update the canonical modules and use Git history plus ignored run metadata for provenance.

## Proposed near-term working rhythm

Until a hard external deadline is supplied, use three 30–60 minute blocks per week for learning and two longer blocks for implementation/experiments. Every learning block should end with a small artifact: an interface note, reproduced equation, field map, metric definition, or runnable probe. This keeps learning attached to the next decision rather than becoming an open-ended literature survey.

## Appendix provenance

The appendices below reproduce the conversation messages returned by the locally saved ChatGPT conversation records on September 24, 2026. Messages are ordered oldest to newest. Role labels and headings are added outside the message bodies for readability; the message bodies themselves are unabridged.


## Appendix A — Generated Room Challenges Motion

**Conversation ID:** `6aac4825-cd04-83ea-a8cd-f52d17e0c07f`
**Source:** https://chatgpt.com/c/6aac4825-cd04-83ea-a8cd-f52d17e0c07f

### Message 1 — Assistant

Yeah. The important part is that **the 15-frame prefix is not just a generic “person standing there” initialization**. It comes from a *specific real GRAB interaction*, and that interaction has geometry baked into it.

### Think about the original GRAB recording

Suppose the original GRAB sequence looked roughly like this:

```text
Original GRAB scene

          mug
           ☕
     ─────────────  table = 0.80 m high

       O
      /|\
      / \
       ↑
 person standing
 0.55 m from mug
```

The recorded human pose is appropriate for exactly that setup. Their shoulder height, arm position, torso lean, distance from the table, and eventually their grasp all make sense relative to **that mug at that height and distance**.

We then take the first 15 frames of that interaction. So HOIDiNi begins with a person whose pose originated from that specific physical arrangement.

But we don't keep that physical arrangement.

We effectively say:

```text
Keep this person's starting motion...

but replace:
original mug  → generated mug
original table → generated desk
original scene → generated nursery
```

That's where adaptation becomes necessary.

### A simple example: desk height

Imagine the original GRAB mug was sitting at **80 cm**, but our generated desk puts the new mug at **70 cm**.

The prefix was recorded for something like:

```text
person
   \_____ hand trajectory → ☕  80 cm
                         ───────────
```

But our actual target is now:

```text
person
   \_____ original tendency

                      ↓ target is lower
                      ☕  70 cm
                  ───────────
```

HOIDiNi can't simply replay the original motion. It has to generate a continuation where the person bends/reaches differently so that the hand gets to the new mug.

And this is where the losses matter.

HOIDiNi is strongly being told:

> **Get the hand to the mug.**

But it is *not* strongly being told:

> **While adapting the body to reach the mug, make sure the feet/body remain correctly attached to our room's floor.**

So one mathematically convenient way of satisfying the first requirement might involve lowering the root/body more than a physically grounded person should.

It can effectively find:

```text
"I need the hand lower."

       ↓ move/bend body downward
       ↓
      O
     /|────→ ☕       ✓ hand reaches mug
     /\
----------------     real floor
  feet ↓             ✗ body penetrates floor
```

That's not necessarily exactly *why* our particular trajectory descended, but it illustrates why scene mismatch gives the optimizer more opportunity to exploit the missing floor constraint.

### Mug geometry creates another adaptation

It's not only the mug's position.

Suppose the GRAB mug was:

```text
Original                         Generated

   ____                            __
  /    \                          /  \
 |      |)-- large handle        |    |)- small handle
 |______|                         |____|
```

The original motion's fingers, palm position, wrist orientation, and approach direction correspond to the original mug.

Our generated mug can have a different handle position, radius, height, orientation, etc.

HOIDiNi actually **is intended to handle object-geometry variation**, which is why replacing the mug isn't inherently wrong. It receives the new object's geometry/contact information and optimizes the motion accordingly.

But now it has multiple things to reconcile:

```text
15-frame human prefix from interaction A
              +
new mug geometry from scene B
              +
new mug position from scene B
              +
new desk geometry/height from scene B
              ↓
HOIDiNi must invent a continuation
that makes these compatible.
```

The farther scene B is from the conditions implied by interaction A, the more substantial that adaptation becomes.

### The subtle distinction: we aren't necessarily starting in an invalid pose

This is important.

Our alignment procedure means we're not simply dropping the GRAB person randomly into the nursery.

We align things so that initially:

```text
✓ Human is grounded at our Z=0 floor.
✓ Human/object horizontal relationship is aligned.
✓ Mug is on our generated desk.
✓ Coordinate systems are consistent.
```

So **frame 15 can be perfectly valid**.

The problem is what happens at frame 16 onward.

HOIDiNi has to answer:

> "Given this starting human pose, how do I turn it into a plausible interaction with *this new object in this new arrangement*?"

The generated continuation is where it has freedom to modify the body.

And that's exactly where we observed the penetration starting.

### Why this matters for interpreting the experiment

There are really **two different causes/pressures**, and I wouldn't combine them.

**Primary problem — missing physical rule**

```text
HOIDiNi does not know:
"the entire body must remain above THIS room's Z=0 floor."
```

This directly explains why a below-floor solution is permitted.

**Secondary problem — scene/prefix mismatch**

```text
The human prefix came from:
GRAB scene A

but the continuation must work in:
generated scene B
```

This means HOIDiNi has to modify the motion more substantially, increasing the chance that it finds a weird solution allowed by its losses.

An analogy would be asking someone to edit a recorded animation of a person grabbing a cup from a kitchen counter, but replacing the counter with a shorter desk and the cup with a differently shaped mug. The editor can adjust the person's movement to reach the new mug—but if you never tell the editor that the floor is immovable, **moving the entire person downward is technically one available way of making the reach work**.

So I wouldn't say:

> "The generated desk/mug caused the floor penetration."

We don't have evidence for that.

I'd say:

> **The missing floor constraint is what allows the penetration. The generated scene makes HOIDiNi perform a harder adaptation from the GRAB prefix, which may expose that missing constraint more strongly.**

And this gives us a useful experiment beyond adding the floor loss: **compare the exact original GRAB geometry against our generated geometry**—especially table/mug height, human-to-object distance, mug dimensions, and contact location. If there's a large discrepancy, we can quantify how far out of the original interaction setup we're asking HOIDiNi to adapt.

### Message 2 — User

long term, do you think we should continue using HOIDini for nursery or look for alternatives? Because how many GRAB interactions are there? if its limited, it may not be good for our work. Do you remember the other options we were looking at do you remember why we chose HOIDini? I thinkw e look through several alternatives but were between two or three at the end

### Message 3 — Assistant

Yes — I remember the broader comparison, and your concern about GRAB is important enough that I **would not assume HOIDiNi should be Nursery’s permanent motion generator**.

One correction first: the official GRAB dataset has **1,334 motion sequences from 10 people interacting with only 51 everyday objects**. :chatgpt-content-reference{index="0"} That doesn't mean HOIDiNi can only manipulate those 51 objects—its geometry conditioning is intended to generalize—but it tells us how limited the underlying interaction distribution is.

### Why we ended up with HOIDiNi

From our earlier discussions, the fundamental Nursery requirement wasn't just:

> text → human motion

It was closer to:

> **text + arbitrary object geometry → whole-body motion + meaningful hand/object contact + object trajectory → something we can eventually execute physically.**

That eliminated a lot of ordinary motion generators.

We discussed general motion systems such as **Motion-X / MotionDiffuse / MotionCraft**, but their strength is general human motion rather than precise object manipulation. The problem for Nursery is that generating a beautiful "pick up cup" animation is insufficient if the hand isn't geometrically attached to the cup.

We also discussed infant-specific work, particularly **InfAGenC / infant action generation**. That was attractive because Nursery ultimately has an **infant agent**, whereas HOIDiNi's human prior is adult. But the infant-specific approaches we found didn't give us the combination we needed: arbitrary object geometry, detailed hand contact, object trajectories, text conditioning, etc.

That's essentially why HOIDiNi survived the filtering.

Its architecture is unusually well matched to manipulation: it takes text/object information and separates generation into an object-centric stage that chooses hand-object contacts followed by a human-centric stage that produces a full-body motion satisfying them. :chatgpt-content-reference{index="1"}

And because we wanted things like:

> "pick up the mug"

with actual grasp information rather than merely an arm moving toward approximately the correct location, that was a major advantage.

We were also looking at the **simulation/controller side** separately: MIMo, infant→MIMo retargeting, WristMimic-style contact ideas, InterMimic, EgoSim, etc. That was a related but different decision: *how do we physically execute/retarget a reference motion?* HOIDiNi was solving *where does the reference HOI motion come from?*

### The GRAB limitation is bigger than just "51 objects"

This is where I think your concern is especially valid.

GRAB has 51 objects, but object count itself isn't the biggest issue. GRAB was explicitly designed around **whole-body grasping/manipulation of everyday objects**. :chatgpt-content-reference{index="3"}

HOIDiNi can potentially take an unseen mesh and reason from its geometry. That's actually one of the reasons these geometry-conditioned approaches are interesting.

The deeper issue is **behavioral coverage**.

Nursery eventually wants something more like:

```text
reach for toy
pick up toy
put toy down
push object
pull object
drag object
turn object
open/close things
touch surface
lean against furniture
sit down
stand up
crawl toward something
crawl while carrying something
transition crawling → sitting
roll over
kick object
push with both hands
bring object toward mouth
climb onto something
support body using furniture
...
```

GRAB's interaction distribution doesn't remotely represent that entire space.

So even if HOIDiNi generalizes perfectly from its 51 object **shapes**, we'd still have a problem generalizing from its relatively narrow set of **behaviors and human-object relationships**.

That's actually more concerning to me than "only 51 objects."

### And Nursery adds a second distribution shift: infants

This was something we had already identified earlier.

HOIDiNi learns an **adult human interaction prior**.

For simple manipulation:

```text
adult reaches for cup
infant reaches for toy
```

there may be enough common structure that we can adapt/retarget it.

But consider:

```text
crawl → stop → sit → reach upward
```

or:

```text
sit on floor → use furniture for support → stand
```

Those aren't simply adult motions scaled down to infant proportions.

The balance strategy, limb proportions, support contacts, locomotion, joint configurations, and transitions are fundamentally different.

So Nursery is asking HOIDiNi to generalize along **three axes simultaneously**:

```text
                 HOIDiNi training/evaluation
                         │
                         │
                 GRAB interactions
                         │
          ┌──────────────┼───────────────┐
          ↓              ↓               ↓
     new objects     new scenes     new behaviors
                                         │
                                         ↓
                                  infant behavior
```

And now we've discovered another issue:

```text
                 + full scene grounding
                   (floor/furniture/etc.)
```

HOIDiNi wasn't really built as a general scene-aware embodied motion planner.

### This doesn't mean I think we should throw HOIDiNi away

I'd change how we think about its role.

Instead of:

> **HOIDiNi = Nursery's motion-generation system**

I'd think:

> **HOIDiNi = Nursery's contact-aware manipulation-motion generator.**

That is a much narrower and more defensible role.

For example:

```text
Nursery action
      │
      ├── locomotion ─────────→ infant locomotion model/controller
      │
      ├── posture transition ─→ infant motion model
      │
      ├── manipulation ───────→ HOIDiNi-like HOI generator
      │                              ↓
      │                         contact targets
      │                              ↓
      └──────────────────────→ physics controller
```

HOIDiNi could remain extremely useful for:

**reach → grasp → lift → carry → place**

without us expecting it to learn everything an infant can do.

### There's another alternative worth revisiting now

A system like [Text2HOI](https://arxiv.org/abs/2404.00562?utm_source=chatgpt.com) is conceptually relevant because it explicitly decomposes the problem into contact generation and motion generation and conditions on object meshes; its authors specifically target generalization to unseen objects. :chatgpt-content-reference{index="5"}

But notice the tradeoff: Text2HOI is focused on **hand-object interaction**, whereas HOIDiNi gives us whole-body HOI. So replacing HOIDiNi isn't automatically an upgrade.

There's also BEHAVE, which gives more natural-environment whole-body HOI data, but it's a dataset/tracking method rather than a drop-in text-to-HOI generator, and it has only 20 objects. :chatgpt-content-reference{index="6"}

That illustrates the field's underlying problem: **there isn't an obvious existing model that simultaneously gives us arbitrary-object generalization + full-body interaction + detailed hands + diverse activities + infant motion + scene awareness + physical executability.**

If there were, we'd probably be using it already.

### What I think we should do before investing much more into HOIDiNi

I wouldn't switch immediately based on this floor bug. The floor problem is relatively straightforward to add as a constraint.

But **before doing significant custom engineering on HOIDiNi**, I'd redo the literature/model search now, specifically around the requirement we've learned from implementing it:

> **2025–2026 open/reproducible methods for scene-aware, object-geometry-conditioned, whole-body human-object motion generation with explicit contact, preferably SMPL-X and suitable for physics retargeting.**

And this time I'd compare candidates on Nursery-specific dimensions:

| Requirement | Why Nursery needs it |
|---|---|
| Unseen object meshes | generated nursery objects |
| Diverse interactions | far beyond GRAB |
| Full-body motion | not hand-only |
| Detailed fingers | manipulation |
| Explicit contact | physics transfer |
| Scene geometry | floor + furniture |
| SMPL-X / retargetable body | current pipeline |
| Motion prefix conditioning | connect actions |
| Physics compatibility | InterMimic/controller |
| Infant adaptability | eventual actual agent |
| Open code + weights | practical implementation |

HOIDiNi might still come out strongest for the **manipulation subset**. But now we know exactly where its assumptions break, so we'd be evaluating alternatives against the actual Nursery requirements rather than against a generic "text-to-motion" requirement.

And I think I can reconstruct the earlier shortlist more carefully too. I remember the broad candidates and why HOIDiNi won, but I **don't confidently remember the exact final 2–3 HOI generators we had narrowed it to**, and I don't want to invent them. If you want, I can search our prior chats/files plus the current literature and rebuild that comparison—including **the exact old finalists, why each lost to HOIDiNi, and whether newer 2025–26 work changes that decision today**.

### Message 4 — User

Sure

### Message 5 — Assistant

I reconstructed what I can from our earlier discussions and checked the newer 2025–2026 work. My conclusion has shifted somewhat:

> **I would keep HOIDiNi in Nursery right now, but I would not design Nursery around HOIDiNi as the permanent universal motion generator.**

The better long-term architecture is probably **modular**, with HOIDiNi—or something descended from it—handling precise manipulation/contact while another model handles scene-aware whole-body behavior.

## 1. What we actually compared before

I found the August 25 discussion you were remembering. The comparison was not originally three equivalent HOI generators.

The sequence was roughly:

| Candidate | Why we considered it | Why it wasn't enough |
|---|---|---|
| **HOIDiNi** | Object-conditioned, text-conditioned, full body, fingers, explicit contact, object trajectories | Adult/GRAB prior; limited behavior coverage |
| **InfAGenC / Infant Action Generative Modeling** | Actual infant motions: crawling, rolling, sitting, standing, etc. | Doesn't provide HOIDiNi-style arbitrary-object manipulation/contact |
| MotionCraft / Motion-X / MotionDiffuse | Much broader human motion generation | Adult-centric and not precise object-contact systems |

The practical decision we arrived at was essentially:

> **InfAGenC has the better infant motion prior. HOIDiNi has the better manipulation machinery.**

And at that point HOIDiNi won **for the manipulation component**, not because we thought it was an ideal complete infant controller.

MIMo was another important system we had discussed earlier, but for a different layer: infant embodiment/physics/sensing rather than generating these object-interaction reference motions. Likewise, InterMimic came later as the downstream physics-execution layer.

I **couldn't recover evidence that we literally had a final “HOIDiNi vs X vs Y” three-way shortlist of equivalent HOI generators**, so I don't want to manufacture that part. The clearest recovered comparison was HOIDiNi vs InfAGenC, after more generic motion models had already been ruled out.

---

# 2. Your GRAB concern is very real

And there's an important correction to how I described GRAB earlier.

GRAB has:

- **1,334 sequences**
- **10 adult subjects**
- **51 objects**
- but only **four top-level motion intents**:
  - use
  - pass
  - lift
  - off-hand pass

The “use” category can contain different object-specific actions—drinking from a mug, using binoculars, etc.—so it's not literally four identical movements. But the behavioral ontology is still quite narrow. :chatgpt-content-reference{index="1"}

And HOIDiNi itself acknowledges the dataset-size issue. Its paper says GRAB is relatively small compared with something like HumanML3D, and it uses GPT-4o to expand GRAB's discrete action annotations into richer text descriptions. :chatgpt-content-reference{index="2"}

That gives you:

```text
"lift"
        ↓
richer textual descriptions
```

but it **doesn't magically give the model new physical experience**.

If GRAB never contains:

```text
crawl to toy
push toy while crawling
support body on furniture
kneel against furniture
climb
roll over holding something
mouth an object
kick toy
transition sit → crawl → reach
pull self upward using furniture
```

rewriting `"lift"` into a nicer sentence doesn't teach those behaviors.

That's the fundamental limitation.

---

# 3. Why HOIDiNi was still such a good choice

The reason it survived our earlier filtering becomes clearer when you look at exactly what the model predicts.

HOIDiNi jointly models:

```text
human full-body motion
        +
finger motion
        +
object 6-DoF motion
        +
explicit hand ↔ object contact pairs
        +
text
        +
object geometry
```

Its human representation uses **52 relevant SMPL-X joints including hands**, and its first generation phase explicitly determines which hand/palm anchors should contact which locations on the object's surface. The second phase then generates the body and articulated fingers around those contacts. :chatgpt-content-reference{index="3"}

That's unusually useful for us.

Most alternatives lose at least one of:

```text
full body
precise fingers
arbitrary object geometry
contact location
object trajectory
text instruction
initial-motion conditioning
```

And for physics transfer, explicit contact is extremely valuable.

That's why our mug experiment actually got surprisingly far:

```text
generated mug
     ↓
HOIDiNi understands its geometry
     ↓
selects plausible hand contacts
     ↓
generates reaching/grasping
     ↓
moves mug successfully
```

Our current failure isn't “HOIDiNi has no idea how to manipulate an unseen mug.”

It's:

```text
HOIDiNi understands mug manipulation

BUT

it isn't sufficiently grounded to the entire generated room.
```

That's a significantly more fixable problem.

---

# 4. But the field has moved since we made that decision

There are now a couple of systems I'd take very seriously.

## HOSIG — probably the most interesting one for Nursery

**HOSIG: Full-Body Human-Object-Scene Interaction Generation with Hierarchical Scene Perception**, published at AAAI 2026, is almost suspiciously aligned with the problem we've just discovered.

Its authors explicitly start from:

> existing HOI generators often ignore scene context, causing penetration.

Their system splits the problem into:

```text
scene-aware grasp pose
        ↓
obstacle-aware navigation
        ↓
scene-guided full-body motion
```

and claims:

- whole-body motion
- object interaction
- static-scene reasoning
- obstacle avoidance
- precise hand-object contact
- **finger-level motion**
- scene collision handling
- long autoregressive interactions :chatgpt-content-reference{index="4"}


Notice how closely that corresponds to Nursery:

```text
HOIDiNi:

person ↔ object

HOSIG:

person ↔ object ↔ room
```

That's exactly the distinction causing our current problem.

### But HOSIG has a practical downside

Its implementation is much less mature than HOIDiNi's.

The code is now public, with scene-guided motion generation, navigation, grasp generation and evaluation components. However, the repository still lists unfinished demo/release work, and their grasp-generation instructions explicitly say that their grasp generator **doesn't produce a good result every time**: they generate ten candidates and manually select the best one. :chatgpt-content-reference{index="5"}

That's a pretty big issue for an automated Nursery dataset pipeline.

So:

**Conceptually:** extremely interesting.

**Drop-in production replacement today:** not obviously.

[HOSIG project page](https://yw0208.github.io/hosig/?utm_source=chatgpt.com)

---

# 5. InfBaGel may be even more interesting architecturally

The other 2026 model I'd investigate is **InfBaGel**, from ICLR 2026.

It explicitly tackles:

> **Human-Object-Scene Interaction**, rather than just Human-Object Interaction.

It does something clever about the exact data problem you're worried about.

Instead of saying:

> “We need one gigantic perfect motion-capture dataset containing every object interacting in every room.”

it combines:

```text
HOI data
   +
human-scene data
   +
synthetically generated scene/object combinations
```

during training. :chatgpt-content-reference{index="7"}

That's very attractive for Nursery because we're **already generating scenes procedurally**.

Its generation loop also repeatedly updates its understanding of scene occupancy as the person/object moves:

```text
generate rough interaction

        ↓

look at resulting scene occupancy

        ↓

detect likely collisions

        ↓

refine interaction

        ↓

update scene again

        ↓

refine again
```

And it adds “bump-aware” guidance specifically to reduce collisions and penetration. :chatgpt-content-reference{index="8"}

That is much closer to what we ultimately need than:

```text
generate motion
→ hope it fits room
```

---

# 6. But InfBaGel has a huge Nursery weakness

**Hands.**

The released InfBaGel representation is based primarily on SMPL-X:

- global orientation
- body pose
- translation
- body joints

plus object trajectory/contact labels and scene occupancy. Its released data format doesn't include the detailed articulated hand-pose representation we're exploiting with HOIDiNi. :chatgpt-content-reference{index="9"}

And its HOI source is OMOMO.

HOIDiNi's paper specifically notes that OMOMO has **27,952 sequences**, far more than GRAB's 1,334, but OMOMO **does not include finger motion**, which is why HOIDiNi says it can't enable precise interactions the same way. :chatgpt-content-reference{index="10"}

That's the tradeoff in one picture:

```text
                 HOIDiNi                 InfBaGel

Dataset          GRAB                    OMOMO + LINGO
Sequences        1,334                   much larger
Detailed hands   YES                     NO / limited
Precise contacts excellent               coarser
Scene awareness  limited                 strong
Navigation       limited                 yes
Collision world  limited                 much stronger
Behavior scale   limited                 broader
```

For Nursery, I don't want to simply trade:

> body goes through floor

for:

> body is beautifully grounded but fingers can't perform the grasp.

---

# 7. There are also newer HOI models

A few are worth monitoring but I wouldn't immediately migrate Nursery to them.

### DecHOI — CVPR 2026

DecHOI separates:

```text
trajectory generation
        ↓
detailed action generation
```

instead of forcing one network to simultaneously solve every objective.

It generates human/object paths without requiring hand-authored intermediate waypoints, then generates the interaction conditioned on those paths. It reports better contact/penetration behavior than earlier CHOIS-style methods and supports longer planning. :chatgpt-content-reference{index="11"}

That architecture makes sense for Nursery.

But it is still built around the **OMOMO/CHOIS ecosystem**, so detailed finger grasping remains a concern.

---

### ViHOI — CVPR 2026

ViHOI attacks another weakness we're worried about: **generalization beyond the training interaction categories**.

Instead of relying only on text like:

> “person uses weird new object”

it uses visual interaction priors from images and a VLM. At inference, it can use generated reference images to help reason about unseen objects and unseen interaction categories. :chatgpt-content-reference{index="12"}

That's fascinating for Nursery because an LLM/VLM could potentially understand:

```text
"This is a ring-stack toy."

→ what does using it look like?
→ where should hands go?
→ what movement does the object undergo?
```

even if that exact object isn't in GRAB.

But again, it inherits much of the CHOIS/OMOMO setup, so I wouldn't sacrifice HOIDiNi's detailed contact representation yet.

---

# 8. There's also a useful new data direction: HandX

This isn't a direct replacement model, but it may eventually solve one of our biggest training-data problems.

**HandX (CVPR 2026)** specifically consolidates and expands high-quality bimanual motion with:

- detailed finger dynamics
- contact events
- hand coordination
- richer semantic descriptions

and the authors report scaling trends where larger/better data improves dexterous generation. :chatgpt-content-reference{index="13"}

That could become relevant if we eventually train/fine-tune a Nursery-specific generator instead of relying completely on an off-the-shelf model.

---

# 9. So I would change the architecture, not just replace HOIDiNi

Right now we're conceptually doing:

```text
                 text task
                    ↓
               HOIDiNi
                    ↓
        entire reference motion
                    ↓
               InterMimic
                    ↓
                 physics
```

That forces HOIDiNi to solve too much.

Long term I'd rather Nursery look something like:

```text
                    Task
                     │
          "pick up the red cup"
                     │
                     ▼
             High-level planner
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
 scene/body planner         manipulation planner
 HOSIG / InfBaGel-like       HOIDiNi-like
        │                         │
 walking/crawling                 │
 positioning                contact points
 floor grounding            finger pose
 obstacle avoidance         object trajectory
        │                         │
        └────────────┬────────────┘
                     ▼
            unified reference
                     │
                     ▼
              physics tracker
                     │
                     ▼
              MIMo / InterMimic
                     │
                     ▼
             actual execution
```

And then separately:

```text
InfAGenC / infant data
          ↓
 infant motion prior
          ↓
replaces adult locomotion/posture prior
```

That would use each model for what it is actually good at.

---

# 10. And this solves the GRAB problem much better

We stop asking:

> “Can GRAB contain every behavior Nursery will ever need?”

Because the answer is clearly **no**.

Instead:

### GRAB teaches something narrow but valuable

```text
how humans establish precise contacts
how hands wrap around geometry
how objects and hands move together
```

### Infant data teaches

```text
crawl
sit
roll
stand
reach
balance
developmental body movement
```

### Scene-interaction data teaches

```text
don't walk through chair
stay on floor
navigate around furniture
position body near target
sit on surfaces
```

### Physics enforces

```text
gravity
collision
friction
balance
forces
actual successful manipulation
```

That is much more scalable.

---

# 11. What I would do with our current HOIDiNi implementation

I **would still make the floor fix we're discussing**.

It isn't wasted work.

We already have:

```text
EmbodiedGen scene
       ↓
generated mug/desk
       ↓
HOIDiNi
       ↓
SMPL-X + object motion
       ↓
validation
       ↓
InterMimic
```

working far enough to expose concrete failure modes.

That's valuable.

Adding:

```text
world-floor penetration
+
support-foot constraint
```

gives us a solid manipulation baseline.

Then I would **not spend months turning HOIDiNi itself into a complete scene/infant generator**.

That's the dividing line I'd use.

### Reasonable HOIDiNi improvements

- floor collision
- basic furniture collisions
- improved contact preservation
- conversion into physics
- testing unseen Nursery objects

### Things I would *not* try to force HOIDiNi to learn from scratch

- crawling
- infant locomotion
- climbing
- sitting/standing transitions
- room-scale navigation
- arbitrary furniture interaction
- developmental motor behavior
- every Nursery action

At that point we'd basically be writing a new paper/model on top of HOIDiNi anyway.

---

# 12. What I think our next model experiment should be

I'd run **three pipelines on exactly the same generated Nursery scene**:

```text
A. Current HOIDiNi + our floor fix

B. HOSIG

C. InfBaGel
```

And use something simple:

> “Approach the table, pick up the mug, move it, and put it down.”

Evaluate:

| Metric | HOIDiNi | HOSIG | InfBaGel |
|---|---:|---:|---:|
| Floor penetration | | | |
| Furniture penetration | | | |
| Human-object contact | | | |
| Finger grasp quality | | | |
| Object penetration | | | |
| Scene navigation | | | |
| Unseen mug | | | |
| Motion naturalness | | | |
| Physics transfer | | | |
| Automation/reliability | | | |

I suspect we'd discover that HOIDiNi wins one part of the table and HOSIG/InfBaGel wins another.

And **that result would actually support the modular design**, rather than forcing us to arbitrarily pick one model.

---

## My current view

If I were drawing the Nursery roadmap today, I'd put it like this:

**Now**

```text
HOIDiNi
   ↓
fix grounding
   ↓
finish physics-transfer investigation
```

**Next experiment**

```text
HOIDiNi vs HOSIG vs InfBaGel
on identical generated Nursery scenes
```

**Long term**

```text
scene-aware / infant-aware motion system
                 +
HOIDiNi-style dexterous contact generator
                 +
physics execution
```

So I don't think discovering the floor issue means **“HOIDiNi was the wrong choice.”** It was actually a useful choice because it gave us something few alternatives had: explicit object-surface contacts and articulated hands.

But your GRAB observation does reveal why **HOIDiNi probably shouldn't define the ceiling of Nursery**. The four-intent, 1,334-sequence adult manipulation distribution is simply too narrow for the developmental behavior space we're ultimately targeting. :chatgpt-content-reference{index="14"}

The two models I'd investigate most seriously **right now** are HOSIG and InfBaGel—not necessarily as direct replacements, but to see whether one should become Nursery's **scene/body-motion layer** while HOIDiNi becomes the **dexterous interaction layer**.

### Message 6 — User

I have a simple question. Why work on the floor penetration for HOIDini if we expect to only use its hand manipulation parts? Isn't that seem unnecessary?

### Message 7 — Assistant

Yes — **if we commit to using HOIDiNi only as a hand/contact planner, then fixing its full-body floor penetration is probably not worth much effort.**

The key distinction is what we want HOIDiNi to output.

If we still want HOIDiNi to generate the **entire reference motion**:

> whole body + hands + object trajectory

then floor penetration matters, because that bad body trajectory gets handed downstream to InterMimic.

But if we redefine HOIDiNi's role to something narrower:

> **generate grasp/contact targets, hand pose, wrist pose, and object motion**

while another model/controller generates the body, locomotion, balance, and scene grounding, then the fact that HOIDiNi's own legs sink into the floor becomes mostly irrelevant.

In that architecture we'd effectively do:

```text
HOIDiNi
  ↓
extract:
- which hand contacts which object points
- finger configuration
- wrist/hand trajectory near manipulation
- object trajectory

Scene/body model
  ↓
generate:
- pelvis/root
- torso
- legs
- locomotion
- floor contact
- obstacle avoidance

        ↓
combine them

        ↓
physics controller
```

Then we'd want to prevent HOIDiNi's bad lower-body motion from influencing the final result at all.

So your instinct is right: **we shouldn't spend significant engineering effort fixing HOIDiNi's floor behavior before deciding its long-term role.**

There's only one reason I'd still consider doing a *very small* floor fix now: as a diagnostic. Our current pipeline uses HOIDiNi's complete SMPL-X motion, so adding one simple floor penalty could tell us whether the present InterMimic failure is substantially caused by the invalid reference trajectory. That would help separate:

```text
bad HOIDiNi body trajectory
```

from

```text
bad HOIDiNi → InterMimic hand/contact transfer
```

But that's different from turning HOIDiNi into a robust scene-aware full-body generator.

So I'd revise the plan to:

1. **Do not heavily develop HOIDiNi's scene grounding.**
2. Either add a minimal floor constraint for one controlled diagnostic experiment, or skip even that if we already have enough evidence.
3. Spend the more important effort figuring out **what information we can reliably extract from HOIDiNi for manipulation**.
4. Evaluate HOSIG/InfBaGel/infant motion approaches for the body and scene layer.
5. Figure out how to fuse those outputs before physics.

The more important research question is no longer:

> “How do we make HOIDiNi's whole body perfect?”

It's:

> **“Can we preserve HOIDiNi's strong grasp/contact information while replacing the part of its motion prior that Nursery doesn't want?”**

That is probably a much better direction for the project.


## Appendix B — Compare GRAB And OMOMO

**Conversation ID:** `6aac4736-b850-83ea-8bf0-5ba77c03d86d`
**Source:** https://chatgpt.com/c/6aac4736-b850-83ea-8bf0-5ba77c03d86d

### Message 1 — Assistant

That specific distinction—**OMOMO vs. GRAB**—is important because HOIDiNi is effectively used in two somewhat different setups, and we chose GRAB for a good reason.

### GRAB vs. OMOMO in this context

Think of them as two different HOIDiNi configurations with different priorities:

| | **GRAB route — what we use** | **OMOMO route** |
|---|---|---|
| Human representation | **SMPL-X** | Less hand-detailed representation |
| Detailed fingers | **Yes** | Not the same detailed SMPL-X hand setup |
| Palm/contact anchors | **Yes — important for us** | Not the same setup |
| Good for detailed grasping | **Yes** | Less suited to our specific hand-contact needs |
| Explicit feet→floor term | **No** | **Yes** |
| Our mug pipeline | **This is why we chose it** | Not what we're currently using |

GRAB itself is a dataset specifically designed around **whole-body grasping**, with detailed articulated hands and body–object contact. :chatgpt-content-reference{index="0"} SMPL-X is also explicitly a whole-body model with fully articulated hands, which is why this representation is useful when the exact grasp matters. :chatgpt-content-reference{index="1"}

So our choice wasn't arbitrary:

**we wanted HOIDiNi's detailed hand/object interaction capability.**

---

### What does the OMOMO feet-to-floor penalty actually add?

Suppose HOIDiNi generates this:

```text
                  person
                    O
                   /|\
                   / \

============================  actual room floor
                  ↓
                feet
              8 cm below
```

The GRAB configuration we're using doesn't have a strong explicit rule saying:

> "Those feet need to be at the actual floor height."

Its foot-skating term is solving a different problem: **don't let a planted foot move around horizontally.**

Those are two separate properties.

For example:

```text
Foot position:

Frame 1: (x=1.0, y=2.0, z=-0.08)
Frame 2: (x=1.0, y=2.0, z=-0.08)
Frame 3: (x=1.0, y=2.0, z=-0.08)
```

The foot is **8 cm underground**, but it isn't sliding.

So the skating loss can essentially say:

> Great, the foot stayed put.

while our room says:

> No, the foot is inside my floor.

That's the gap.

---

### Why would HOIDiNi have a floor loss for OMOMO but not GRAB?

This makes more sense when you consider what the configurations were designed to optimize.

HOIDiNi isn't necessarily trying to construct one universal objective containing every possible physical constraint. Different datasets/configurations can use different losses appropriate to their representation and task.

For the GRAB side, the emphasis is particularly strong on **detailed human-object interaction and grasp/contact**. GRAB itself contains full-body motion, articulated hands, object pose, and detailed body-object contact. :chatgpt-content-reference{index="2"}

That's exactly the information we're interested in for:

```text
reach
  ↓
position fingers
  ↓
contact mug
  ↓
maintain grasp
  ↓
lift mug
```

The problem is that we're now asking that configuration to additionally understand:

```text
"This particular plane in my generated 3D room
is an immutable physical floor."
```

That's a requirement introduced by **our scene pipeline**.

---

### So why don't we just switch to the OMOMO configuration?

Because we'd potentially be solving one problem by weakening something more important.

Our ultimate task isn't simply:

> Generate a person who stands correctly.

It's:

> Generate a person who can interact with arbitrary generated objects, including detailed hand-object contact, and eventually transfer that motion into physics.

For something like a mug, the hand details matter a lot.

A tiny change in finger/palm geometry can be the difference between:

```text
       hand
        ↓
     [ MUG ]      ← actual contact
```

and

```text
       hand

        gap
         ↓
     [ MUG ]      ← no physical contact
```

The latter becomes especially disastrous once we get to InterMimic: the animation may visually resemble a grasp, but the physics engine sees **no collision/contact**, so gravity just takes over.

That's why sticking with the GRAB/SMPL-X pathway makes sense for our goal.

---

## The interesting solution: combine the advantages

We don't actually have to accept this as:

**GRAB OR floor grounding.**

A much better direction for our project is:

> **Keep GRAB's detailed grasp representation and add the missing scene-grounding constraints ourselves.**

Conceptually, our current HOIDiNi objective is something like:

\[
L =
L_{\text{hand-contact}}
+ L_{\text{object-penetration}}
+ L_{\text{desk}}
+ L_{\text{smoothness}}
+ L_{\text{foot-skating}}
+\dots
\]

We would extend it:

\[
L =
L_{\text{HOIDiNi-GRAB}}
+
\lambda_1 L_{\text{floor-penetration}}
+
\lambda_2 L_{\text{foot-contact}}
\]

So we're **not replacing HOIDiNi**.

We're telling its optimization:

> "Continue doing everything you're already doing for the grasp, but here's another rule about our scene."

### Rule 1 — don't go through the floor

If the floor is at \(z=0\), then for body vertices:

\[
L_{\text{floor}}
=
\frac{1}{N}\sum_i
\max(0,-z_i)^2
\]

Very intuitively:

```text
vertex at +20 cm → penalty 0
vertex at +1 cm  → penalty 0
vertex at 0 cm   → penalty 0
vertex at -1 cm  → small penalty
vertex at -10 cm → BIG penalty
```

That directly fixes the failure we're observing.

### Rule 2 — actually stay connected to the floor

But Rule 1 alone isn't sufficient.

HOIDiNi could theoretically discover:

```text
"Fine. I won't go BELOW the floor.
I'll put the entire person 5 cm ABOVE it."
```

No penetration—but now they're floating.

So we additionally want a foot-contact term:

```text
supporting foot ≈ actual floor
```

This is where the idea from HOIDiNi's OMOMO configuration becomes particularly useful.

And then we keep the existing foot-skating term:

```text
supporting foot ≈ floor       ← vertical contact
supporting foot doesn't slide ← horizontal stability
body ≥ floor                  ← no penetration
```

These three constraints complement each other rather than duplicate one another.

---

## And there's actually a bigger research idea hiding here

The more important distinction is:

**HOIDiNi understands the interaction geometry quite well, but it doesn't fully understand our generated scene as a physical world.**

Right now we're effectively giving it:

```text
Human
   ↓
Mug
   ↓
Desk
```

We're not really giving it:

```text
┌──────────────────────────────┐
│       PHYSICAL SCENE         │
│                              │
│ Human ← collision → desk     │
│   ↕                          │
│ mug                          │
│   ↕                          │
│ tabletop                     │
│                              │
├──────────────────────────────┤
│ PHYSICAL FLOOR               │
└──────────────────────────────┘
```

That explains why simply copying the OMOMO foot term may be a **good immediate fix**, but probably isn't the complete long-term solution for EmbodiedGen.

Eventually, we'd want HOIDiNi's optimization to know about more of the generated environment:

- floor,
- desk,
- chair,
- walls,
- nearby objects,
- support/contact surfaces.

Then the same principle generalizes beyond the mug experiment.

Instead of special-casing:

> "Don't put the feet below Z=0."

we could eventually formulate:

> **"The generated human must respect the geometry and contact surfaces of the generated 3D scene."**

That is much closer to what we ultimately need before passing the motion to a physics controller.

So I'd view the **OMOMO feet-floor loss as evidence that HOIDiNi's optimization framework can accommodate this kind of constraint**, not as evidence that we should abandon our GRAB route. The sensible progression is **GRAB + floor grounding first**, validate that it fixes the 7–14 cm sinking without damaging the mug grasp, and then consider broader scene-aware collision/contact losses.

### Message 2 — Assistant

The **“therefore”** there is making an important distinction:

> HOIDiNi producing a motion that *looks physically reasonable* does **not** mean a physics simulator can actually reproduce that motion.

Here’s why.

### Kinematic motion vs. physical motion

HOIDiNi essentially outputs a sequence of desired poses:

```text
Frame 1     Frame 2     Frame 3       ...      Frame

### Message 3 — Assistant

Yeah. The subtle part here is that **we aren't asking HOIDiNi to reproduce the original GRAB mug interaction unchanged**. We're taking the beginning of a real GRAB interaction and then effectively changing the world around it.

### What the original GRAB motion actually gives us

Imagine the original recorded GRAB sequence was:

```text
Person A
   │
   │  arm/hand positioned for this setup
   ▼
 [Mug A]       ← specific size + shape
 ───────
 Table A       ← specific height
══════════
 floor
```

The person's pose isn't independent of that setup. Their shoulder angle, elbow bend, torso lean, hand height, distance from the table, etc. all came from interacting with **that particular physical arrangement**.

Suppose, just as an example, the original GRAB interaction has:

```text
table height:       75 cm
mug height:         10 cm
mug center height:  80 cm

person → mug:       45 cm
```

The human motion was recorded with those relationships.

Our 15-frame prefix therefore contains more than just "a person standing."

It implicitly contains:

> "A person whose body is positioned appropriately for an object at roughly *this* location."

---

## Then we change the scene

We keep those first 15 human frames, but now the target might be:

```text
Person A
   │
   │
   │          [Generated Mug B]
   │                ↑
   ▼                │
              ──────────────
              Generated Desk B
══════════════════════════════
       generated-room floor
```

Maybe the generated desk is 68 cm high instead of 75 cm.

Now the mug might effectively be **7 cm lower than what the original human pose was prepared for**.

That's already interesting because HOIDiNi receives two pieces of information pulling it in slightly different directions:

```text
PREFIX:
"This is how the human has been moving so far."

SCENE:
"But the object you now need to interact with
is over here."
```

HOIDiNi has to generate a continuation that reconciles those.

### That's actually what HOIDiNi is supposed to help with

This doesn't mean we're using HOIDiNi completely incorrectly.

HOIDiNi isn't simply replaying GRAB. Its optimization can modify generated motion to satisfy object interaction constraints.

So if our new mug is lower, it can potentially generate something like:

```text
Original-style continuation:

       O
      /|────→ mug
      / \


Adapted continuation:

       O
      /|
     / |────→ lower mug
    / \
```

The torso might bend more. The shoulder might lower. The knees might bend.

That's desirable.

The problem is that there are **many mathematical ways** to get the hand to the new mug.

---

# And this connects directly to our floor problem

Suppose HOIDiNi needs the hand to move 8 cm downward.

A physically sensible solution might be:

```text
feet stay on floor
        ↓
knees bend
        ↓
torso leans
        ↓
shoulder lowers
        ↓
hand reaches mug
```

But another mathematically convenient solution is:

```text
whole body shifts downward 8 cm
        ↓
hand is now 8 cm lower
        ↓
hand reaches mug
```

Visually:

```text
GOOD SOLUTION

        O
       /|──[mug]
       /\
================ floor


BAD SOLUTION

        O
       /|──[mug]
-------/\------------- floor
       ↓
   feet underground
```

If HOIDiNi's objective strongly says:

> "Get the hand to the mug."

but does **not** strongly say:

> "The human's body must respect this exact world floor."

then the second solution may not be sufficiently punished.

That gives us a plausible connection between **scene adaptation** and the **missing floor constraint**.

Importantly, though, we haven't established that the changed desk height specifically caused the 7–14 cm sinking. The strongest established issue remains that the GRAB configuration lacks the world-floor constraint. Scene differences are an additional pressure that can expose that weakness.

---

## The mug itself can create the same problem

Desk height isn't the only difference.

Suppose the original GRAB mug was:

```text
   _____
  |     |)
  |     |)
  |_____|

wide body
large handle
```

and our generated mug is:

```text
   ___
  |   |)
  |   |)
  |___|

narrower
smaller handle
different orientation
```

The original prefix might have the hand approaching expecting something like:

```text
thumb  →      ← fingers
        [MUG]
```

But now those same hand coordinates don't correspond to the same surface locations.

HOIDiNi therefore needs to alter:

- wrist position,
- wrist rotation,
- finger configuration,
- palm position,
- possibly elbow position,
- possibly shoulder position,
- and therefore potentially the rest of the upper body.

And that's one reason the **60 palm-contact anchors** are useful. We're not merely saying "put the wrist near the mug." We're giving the optimization much richer information about where hand-object contact should occur.

---

# The important distinction: absolute coordinates vs. relationships

This is probably the key idea.

The GRAB prefix gives us an existing human pose in a coordinate system.

But what we actually care about is preserving meaningful **relationships**:

```text
feet ↔ floor
hand ↔ mug
mug ↔ desk
body ↔ desk
```

Our new scene changes some of those relationships.

For example:

```text
ORIGINAL GRAB

floor → foot       = 0 cm
floor → tabletop   = 75 cm
table → mug        = 0 cm
hand → mug         ≈ contact
```

After putting things into our room, maybe:

```text
GENERATED ROOM

floor → foot       = 0 cm       ✓
floor → tabletop   = 69 cm      changed
table → mug        = 0 cm       ✓
hand → mug         = 6 cm       mismatch
```

Now HOIDiNi has to resolve that last mismatch.

And its optimization might successfully reach:

```text
hand → mug = 0 cm ✓
```

while accidentally changing:

```text
floor → foot = -8 cm ✗
```

because the first relationship is explicitly optimized, whereas the second isn't adequately tied to the **actual room floor**.

That's a very useful way of thinking about what happened.

---

# Why our initial horizontal alignment doesn't completely solve this

We deliberately preserve the human-object relationship in the horizontal plane.

Basically, we're trying not to do something ridiculous like:

```text
person                         mug
  O                             []
 /|\

          4 meters
<------------------------------>
```

We align things so that the person is reasonably positioned relative to the generated object in **X/Y**.

But vertical placement is special.

The human must satisfy:

```text
feet → floor
```

while the mug must satisfy:

```text
mug → desk
```

And those two vertical chains meet at:

```text
floor
  │
  ├── human feet
  │
  └── desk
        │
        └── mug
```

Therefore the **desk height determines the mug's height relative to the person's feet**.

If the generated desk height differs from the GRAB desk/table height, simply aligning X/Y cannot preserve the original hand-object geometry.

That's why vertical scene grounding matters so much here.

---

## This also explains why simply moving the human afterward is dangerous

You might think:

> "Okay, if HOIDiNi puts the person 10 cm underground, why don't we just translate the generated human upward 10 cm afterward?"

That fixes:

```text
feet ↔ floor ✓
```

but potentially destroys:

```text
hand ↔ mug ✗
```

Imagine HOIDiNi generated:

```text
hand = mug height
feet = -10 cm
```

Move the entire human +10 cm:

```text
feet = floor        ✓
hand = mug + 10 cm  ✗
```

Now we've fixed the floor by breaking the grasp.

Moving both human **and mug** upward doesn't solve it either, because the mug is supposed to be supported by the desk.

So this really needs to be solved **during optimization**:

```text
                    ┌─ feet must respect floor
HOIDiNi must find ──┼─ hand must contact mug
one pose satisfying ├─ mug must respect desk
all of these        └─ body must avoid desk
```

rather than repairing one relationship afterward.

---

# This is why our experiment is harder than ordinary HOIDiNi evaluation

We're effectively asking for **scene-conditioned adaptation**.

The input isn't just:

> "Generate a plausible mug-lifting motion."

It's closer to:

> "Starting from these 15 human frames, generate a plausible mug-lifting motion for **this exact new mug, at this exact location, on this exact generated desk, while respecting this exact generated room**."

Those are substantially stronger requirements.

And eventually our goal is even broader:

```text
Recorded interaction prior
          ↓
       HOIDiNi
          ↓
 Adapt motion to:
 ├── arbitrary generated human
 ├── arbitrary generated object
 ├── arbitrary generated furniture
 ├── arbitrary object placement
 └── arbitrary generated room
          ↓
 physically valid reference motion
          ↓
      InterMimic
          ↓
 dynamically executable interaction
```

At that point, **floor grounding is only the first obvious missing scene constraint**.

Later we may encounter the exact same class of problem with:

```text
leg ↔ chair
hip ↔ chair
arm ↔ wall
body ↔ desk
foot ↔ rug/step
hand ↔ nearby objects
```

So the 7–14 cm floor penetration is useful diagnostically: it's showing us that our pipeline has moved from **human-object motion generation** toward **human-object-scene motion generation**, while HOIDiNi's released GRAB objective doesn't completely encode that larger world.

For the current experiment, though, I would keep the diagnosis disciplined: **the missing real-floor constraint is the demonstrated cause we should fix first; differences between our generated mug/desk and the source GRAB scene are plausible additional pressure, and we can measure how important they are afterward.**
