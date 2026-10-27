# Nursery side conversation

Research discussion about sensory transfer, language acquisition, additive and
joint learners, and the proposed acquisition and cross-situational studies.

**Scope:** begins with “So things didn't go according to plan ig?” and ends
with the plan-update confirmation for commit `20f1cc03`. The parent conversation
and the later repository-cleanup discussion are excluded.

**Source:** user-supplied chat copies, combined in conversation order without
rewriting the messages; trailing whitespace is removed. The design-document
exchange, small-study clarification and final confirmation were supplied in
separate pastes. Three short follow-up user messages about presentation are
retained from the conversation context and identified below. Images, collapsed
annotations and link destinations absent from the pasted text have not been
reconstructed. UI labels and timestamps are retained as supplied.

This is historical conversation, not current instructions or a queue of work.
See [PLAN.md](../../PLAN.md) for the current direction.

---

So things didn't go according to plan ig?

Can you delegate looking into why the sensory learning didn't carry over to word learning? Use the saved runs first, explain it in simple terms, and don't start new experiments yet.

9:47 AM
The experiment ran correctly, but the hoped-for benefit didn’t appear. The delegated reviews found a more specific explanation than “body information doesn’t help.”

1. Most of the improvement was about the hands, not necessarily the objects.

We taught the model to look at video and predict bodily signals. About 67% of the improvement in its combined sensory loss came from predicting whether the hands were visible. Predicting hand shape and contact improved too; predicting changes in contact improved only slightly.

That could teach “a hand is visible and closing” without teaching “this object is a cup.” This is our strongest clue, though it doesn’t prove the cause. Saved sensory results

2. We measured the whole sensory predictor, but only carried part of it into word learning.

The sensory predictor had a visual representation plus additional prediction layers. During word learning, we kept only the visual representation and discarded those extra layers.

Some of the improvement could have lived in the discarded layers. Better sensory prediction therefore didn’t establish that the retained representation had become better for learning words. Our experiment left that distinction unresolved. Transfer implementation (line 389)

3. The saved learning curves don’t show a clear advantage that later disappeared.

Across the first ten language epochs, aligned pretraining’s average advantage over video-only was just 0.07 percentage points on the development set. Differences stayed small and inconsistent, including against shuffled sensory training.

So “it learned something useful for words and then forgot it” remains possible, but isn’t supported by a visible early lead. We also didn’t measure before the first full language epoch, so very early loss remains unobserved. Example saved learning history

My interpretation is: we successfully taught a sensory prediction skill, but haven’t shown that it creates the object distinctions that weak speech needs. Recognizing hand state and figuring out which object someone means are different demands.

The next question to settle is what useful knowledge actually reached the part of the model that learns words. These saved runs narrow the possibilities, but cannot identify the exact cause. Neither review found a demonstrated wiring bug explaining the result.

No new experiments or model inference were run, and no files were changed.

9:50 AM
why did we teach the model to look at video and predict bodily signals? isn't the idea supposed to be that we train the model on language associatee with the video, and on language associated with the video + body signals.. so essentially the learner comprehends language?

10:11 AM
Worked for 17s
Yes—that is the core experiment you want. Both learners receive the same language and video; one also receives bodily signals. We then test whether those extra signals help it understand the language better.

For example, during an experience:

Learner	What it receives
Video-only	Sees an object being lifted and hears “That cup is heavy.”
Video + body	Receives the same video and speech, plus contact, movement, and load signals.
The question is whether the second learner can use that additional experience to learn what the words mean.

Step 3 took an indirect route. Its hypothesis was: perhaps bodily experience first helps organize visual knowledge, which then makes words easier to learn. We implemented that by having video predict sensory signals, then carrying the visual representation into language learning. The learner received no body signals during the language-learning phase.

That means Step 3 tested a narrower question:

Does practicing sensory prediction beforehand improve later language learning from video alone?

That’s a reasonable supporting diagnostic, but it doesn’t directly test your central comparison. I should have made that distinction much clearer before we committed to it. Its negative result tells us this particular preparation method didn’t help; it doesn’t establish that learning language with video and body together won’t help.

We did test the direct comparison in Step 1, where body didn’t improve ordinary noun learning. Step 2 showed that body inputs could help with explicitly aligned heavy/light language. Step 3 explored one possible explanation for the gap.

Your intended direction remains language learning with simultaneous video, speech, and bodily experience. We need to understand why the learner fails to use the bodily information in that setting. And “comprehension” should be measured through concrete language tasks—our current word-to-picture choices measure one narrow part of it.

10:12 AM
User attachment
Okay.. so yes it remains that but we have made no progress towards seeing that learning curve that we wanted to see?

10:16 AM
We haven’t demonstrated the separation in your drawing. If that is the measure of success, we haven’t achieved it yet.

We’ve made diagnostic progress: repaired the comparison, shown that bodily information can help with a narrow heavy/light task, and tested one preparation method that didn’t improve noun learning. Those findings haven’t yet produced the language-learning advantage we want.

We also haven’t properly measured your intended curves: language performance as we increase the amount of independent experience, comparing video with video + body. Repeating training over the same experiences doesn’t establish that curve or a higher ceiling.

Your drawing contains two questions:

Does body help the learner learn faster? It reaches a given level with less experience.
Does body enable more understanding eventually? Its curve levels off above video-only.
Our existing results establish neither for ordinary weak language learning. And the curves might not separate with the current signals, experiences, and learner—we need to allow that possibility.

I let the intermediate diagnostics take too much of the spotlight. They should help us build and interpret that matched language-learning comparison, with bodily signals available during language learning. Completing numbered steps is not the same as demonstrating the effect you’re aiming for.

10:16 AM
So yeah so what should we do? 'm tryna understand...

so whtat learner did we use? did we use the one advised by Michael Frank? unimail / uni-mail / universitymail / universityemail

and did we make mistakes witht he previous architectures we tried with but didn't go to plan?

like how do we approach solving this issue?

10:19 AM
Yes—we used the codebase Michael Frank recommended. But the way we added bodily information was our design, and several earlier experiments left important questions unresolved. I checked his emails, the current code, and the archived results.

In his August 20 email, Frank recommended vlm-headcam as a testbed and suggested exploring whether its CLIP-style architecture could combine three information sources. He didn’t specify our body encoder, fusion method, or sensory-pretraining procedure. His July 6 advice also emphasized matching aspects of natural experience and comparing conditions apples-to-apples.

Our learner is built on that recommended repository, pinned to a specific revision.

What learner did we actually use?

Think of it as a small word–experience matching model:

Vision: a pretrained, frozen DINOv2 model describes image regions. A trainable layer maps those descriptions into a 512-dimensional representation.
Language: it learns word vectors from our transcripts and averages them to represent each utterance.
Learning: matching utterances and visual experiences are encouraged to score higher than mismatched ones.
Body condition: another network processes contact, finger configuration, and arm movement. Its body–word score is added to the visual–word score.
So it learns associations between words and experiences. It isn’t a conversational language model, and it doesn’t learn vision from scratch. The body pathway genuinely participates in language training, but adding two scores is a limited way to combine the information. For example, the body may describe a held cup while speech refers to a nearby ball. Learner implementation (line 74)

Did we make mistakes? Yes—especially in experimental design and interpretation.

What happened	What we should conclude
Some speech and scoring had artificial inconsistencies. These were repaired in Step 1.	Earlier results need their historical caveats; the repaired comparison still found no gain.
The joint Transformer changed the processing architecture and reduced the representation from 512 to 128 dimensions. Its video-only control performed much worse.	That particular configuration failed. It did not establish that joint video–body processing is a bad idea.
We initially suspected the visual features couldn’t support better learning. Correct-label training later improved results substantially with the same features.	There was visual headroom. The original explanation was too strong.
Step 3 measured sensory prediction and later picture-only transfer.	It answered a supporting question, while your central simultaneous video–body–language comparison remained unresolved.
The older body models also performed much better on training experiences than unfamiliar clips. That supports a generalization problem, but doesn’t identify exactly what they memorized. Historical evidence and reassessment (line 29)

My recommendation is to bring the desired learning curve back into the next protocol, while making the comparison interpretable.

Keep the working visual learner fixed as the reference. Any proposed replacement must first preserve reasonable video-only performance. Otherwise we cannot tell whether its body mechanism failed or its visual learning deteriorated.

Name the specific help we expect from body information. For noun learning, is it helping identify which object is being discussed, or recognize that object across different interactions? For “heavy,” it supplies a property. Those mechanisms need different evidence. “More modalities should help” is too broad to guide a change.

Measure the direct comparison across amounts of independent experience. Use the same nested sets of clips, speech, evaluation, and matched seeds for video-only and video + body. Keep body available during the main language test; report picture-only transfer separately. We should not require a full-data advantage before checking whether body helps at smaller data amounts.

Let the pattern choose the next change. A training-only benefit points toward generalization. An advantage with clear naming that disappears under weak speech points toward learning the association. Failure even with informative inputs and clear naming points toward how the learner uses those inputs. We already have partial evidence for these cases; we should reuse it rather than repeat every diagnostic.

We don’t yet have evidence that a particular new architecture will solve this. We do have enough evidence to stop treating every null result as a reason to try another architecture. The next design should make your comparison the main outcome and make its failure informative.

No experiments or files were changed. I read the relevant university emails and closed Mail afterward.

10:21 AM
Can you tell me how the learners work in all the architectures we tried, and explain all the experiments we tried. In simple terms but in depth so that maybe I could find some insight?

10:33 AM
Yes. Looking across the history, we have tested several different ideas about how body information might help—and they do not all test the same thing. Understanding those differences is probably more useful than treating everything as a long list of failed runs.

I’ll cover the recorded language-learning experiments, including the earlier real-data branches. Simulator and rendering work matters as input preparation, but it isn’t itself evidence of language learning.

One caution throughout: compare the two conditions within an experiment. Different experiments used different data, splits, candidate pictures, and scoring rules. A score of 76% in one experiment is not necessarily better than 52% in another.

First, what does “learning language” mean in our current system?

Imagine an experience containing a cup, a spoon, and a hand, accompanied by:

“Can you give me the cup?”

The model receives visual features and the transcript. It learns to make this utterance and its accompanying experience match better than mismatched utterance–experience pairs.

It is not explicitly told which image region corresponds to “cup.” It must discover useful associations across examples.

At evaluation, a typical task is:

Given “cup,” which of these four pictures or experiences matches?

Chance is 25%. The heavy/light task instead has two choices, so chance is 50%.

This is a useful test of word grounding, but it is a narrow part of comprehension. Our current text representation averages word vectors, so it does not explicitly represent word order. “The cup touches the spoon” and “The spoon touches the cup” have the same bag of words. We should not describe success here as general sentence understanding.

The architectures: how each learner actually works

1. The main video–language learner: learn words against visual regions

This is the foundation inherited from the vlm-headcam codebase Frank recommended.

The visual side works like this:

Image → pretrained DINOv2 features → trainable visual projection → comparison with learned words.

DINOv2 supplies descriptions of 17 regions: a whole-image representation and a 4×4 grid. We keep DINOv2 frozen. The trainable projection and word vectors have 512 dimensions.

“Frozen” means we are not teaching the visual backbone to see from scratch. We are teaching a smaller system how its existing visual descriptions relate to our words.

The original region learner picks the strongest matching region. Our later soft-pooling version lets several promising regions contribute, while favoring strong matches.

What this can learn: recurring visual distinctions associated with words.

What it must struggle through: speech may refer to one of several objects, something outside the picture, or something from a nearby moment.

Current learner reference (line 74)

2. Contact-weighted learning: body chooses which examples matter more

This kept essentially the same visual–language learner.

Contact did not become a rich input representation. Instead, contact timing changed how much training weight an image–speech pair received:

“There was contact around this moment, so pay more attention to this example.”

That can help if contact identifies useful learning moments. It cannot directly teach what a particular contact pattern or load feels like.

This is a test of example selection or weighting, rather than the full video-plus-body learner you ultimately want.

3. Hand-guided learning: body suggests where to look

The next idea was:

“When the hand is interacting with something, image regions near the hand might contain the named object.”

We projected the hand into the picture. A small network learned how strongly to favor nearby regions.

There were two versions:

Hard selection: bias which region wins, then use that region’s visual–word similarity.
Soft selection: give nearby regions greater prior weight in a smoother pooled score.
The body’s main job was therefore to guide visual attention during training. It did not independently express “this interaction feels like a cup.”

An important limitation: the hand can be on a cup while the speaker names a ball. A useful hand cue must be used selectively.

4. The additive body learner: vision and body each contribute a word-matching score

This became our main direct multimodal architecture.

The body network receives:

Contact across 22 hand patches.
Finger configuration.
Arm positions and movement.
In later versions, modeled load and skin signals.
Small networks process the signal groups. A recurrent network summarizes the body sequence around the utterance, within the source clip.

Then:

Combined match = visual–word match + learned weight × body–word match.

The language-training loss updates the visual projection, word vectors, and body pathway together. So yes, this architecture genuinely learns language using video and body inputs simultaneously.

However, it combines the two pathways fairly late. The body does not directly tell each visual region how to change its representation. It contributes another overall score.

That distinction matters. “The hand is tightly closed” may be useful only after identifying what object is being held. A separate body vote has limited ability to express that interaction.

During training, we also removed body input on half the examples, encouraging the visual pathway to remain usable independently.

The body window looks both backward and forward around the utterance. This is an offline learning setup, not a strictly moment-by-moment simulation of what a baby knows at that instant.

5. The joint Transformer: let vision and body interact before matching words

Here we put visual-region representations and body states into a shared Transformer.

Visual regions + body sequence → shared processing → contextualized visual regions → word matching.

In principle, this can represent combinations such as:

“This particular visible object, together with this grasp and motion, corresponds to this word.”

This is more flexible than adding independent scores.

But our trial also reduced the visual and word representation from 512 to 128 dimensions, changed how visual regions interacted, and introduced a different contextual processing system. The Transformer’s internal dropout was zero, although other regularization remained.

Therefore, this was a bundled architecture change. Its failure cannot identify which change was responsible.

Joint architecture and retrospective review

6. Teacher–student learning: use an informed teacher to improve a visual learner

We gave a teacher privileged information: the true held-object identity.

Then we trained a video-only student partly to imitate the teacher’s scores.

The hoped-for mechanism was:

“A teacher that knows more can help the visual learner form better word associations.”

This is called distillation. The student does not receive the privileged information at test time.

It tests whether knowledge can be transferred from an informative teacher. It does not establish that actual touch or proprioception can provide that knowledge.

7. Sensory pretraining: prepare the visual representation before words

This was the recent Step 3.

First, video features learned temporal consistency and, in the aligned condition, predicted contact, hand aperture, and hand visibility.

Then we discarded the sensory prediction layers and transferred only the visual projection into a fresh video-only language learner.

So the body served as a training target before language learning, rather than an input during language learning.

That is why its failure has a narrower meaning than failure of the core simultaneous comparison.

8. Diagnostic classifiers: test whether information is present

Several experiments used small classifiers given correct object labels.

They asked questions such as:

“Can a classifier identify the object from these sensors?”
“Does adding sensors improve classification beyond video?”

These classifiers had easier supervision than the language learner. They did not have to discover what ambiguous caregiver speech referred to.

A positive diagnostic means information is usable under that supervised setup. It does not mean our weak-language learner knows how to extract it.

Now the experiments, in the order they changed our understanding

The early real-data branch: adult video, speech, and head movement

Before the current Frank-derived learner, we used a small model with three encoders:

A small convolutional network plus recurrent network for video.
A recurrent network for text.
A recurrent network for motion sensors.
Training encouraged video, text, and motion representations from the same experience to match.

The AEA development tests exposed serious task-construction problems. In a 48-window audit, about 71% of speech-derived action labels did not match a visible wearer action. Words could describe another person, a remote event, or phrases such as “read my mind.”

The visual learner also failed an easier training-fit check. An IMU classifier fitted training examples strongly but did not generalize. Some required shuffled-sensor comparisons were infeasible under the chosen group splits.

Two subsequent visible/coarse-action feasibility revisions stopped at annotation or support checks; their planned downstream learner comparisons were not run.

What this established: that exact-window speech-derived action task was not a sound foundation for the intended comparison. It did not establish that natural weak alignment is inherently unusable.

Archived AEA report

The earlier EgoBabyVLM real-versus-synthetic branch

A separate route used an EgoBabyVLM CLIP+ learner with three training objectives: video–text matching, masked-word prediction, and a DINOv2-based visual objective.

Here “triple” referred to objectives, not three input modalities.

The intended question was whether synthetic video could support learning comparable to real video. Initial real-data runs at one and three hours failed the learnability checks.

A later coverage correction gave more complete training exposure. Across three seeds, the mean lexical gain was about 1.77 percentage points, narrowly below the frozen two-point requirement. That was inconclusive, not proof of zero learning. The intended synthetic learner comparison was not completed in that recorded branch.

What this established: technical execution and positive-control performance needed attention before interpreting a synthetic-versus-real comparison. This was not a direct test of body helping language.

Recorded pilot outcomes

The contact-weighting program: 90 runs, then 100 confirmation runs

The first grid varied how often speech concerned held objects, with five seeds per condition.

At the primary setting, contact weighting improved accuracy by about 2.27 points, but the uncertainty interval was very wide and included no benefit.

The independent 100-run confirmation found:

Comparison	Contact weighting’s advantage
Versus no weighting	+0.94 points; inconclusive
Versus shuffled timing	+3.78 points
Versus shifted timing	+4.33 points
What this established: correct timing was better than deliberately corrupted timing. It did not establish an improvement over ordinary learning without the cue.

This is why beating shuffled body alone is insufficient: the unmodified baseline matters too.

Contact-study record

The two hand-location gates

Hand projection itself worked reasonably well: in the checked contact frames, the projected hand frequently landed near the held object.

But the learned trust in the hand cue collapsed toward zero.

There was also a real design mistake: the early protocol treated category-predictive hand shape as suspicious “leakage.” It reduced detailed hand pose to coarse grip measurements and added noise.

That restriction removed some of the legitimate information we wanted embodiment to provide. Hidden object IDs are privileged labels; a grasp revealing object shape is sensory evidence. Those should not have been treated as equivalent.

The hard-selection version also needed a special gradient approximation because choosing a single winning region is not smoothly differentiable. The soft version changed that mechanism, but did not produce the desired gain.

What this established: those particular hand-guidance mechanisms failed. It did not reject rich body inputs.

Experiment 4.1: the first full body-input learner

We trained the additive architecture with real contact, fingers, and arm inputs, plus a visual-to-sensory auxiliary objective.

Picture-only results were nearly unchanged: approximately 76.4% versus 75.4% on rendered pictures.

The body-only word score was around chance on unfamiliar-person tests, and combining it with vision did not help.

The body pathway was active; its learned contribution had not simply become zero.

One auxiliary “future contact prediction” result was misleading: the bidirectional body encoder already saw future context. It was not evidence of genuine forecasting.

What this established: an active, connected body pathway is not enough to produce useful language learning.

Experiment 4.2: give the body pathway the true object identity

This was the oracle test.

On trials naming the held object, the privileged body pathway scored about 94%, while the combined score was about 77%.

What this established: the architecture can use an extremely informative extra input. There was no universal “the body branch cannot learn words” failure.

But combining vision with that strong input could dilute its value. That exposed a limitation of the scoring combination.

It still produced no useful picture-only transfer.

Experiment 4.3: teach the visual learner using the oracle

We distilled the informed teacher into a video-only student.

The student’s final picture performance was effectively unchanged from its control.

An early explanation blamed the frozen visual features or simple projection. The next experiment showed that explanation was premature.

What this established: this teacher–student procedure did not improve visual word learning. It did not establish that the visual representation had reached its limit.

Experiment 4.4: replace weak speech with correct visible-object labels

Using the same visual features and learner, direct labels improved rendered performance from roughly 76% to 90%, and real-photo performance from about 69% to 84%.

This is a major result.

The visual system could learn substantially more when the association was made clear.

It tells us that the supervision problem matters: discovering what speech refers to is much harder than learning from correct labels.

The same stage also:

Re-scored existing body models with training-consistent soft pooling.
Trained a body model without the visual-to-body auxiliary loss.
Neither rescued the real-body language benefit.

What this established: neither scoring style nor that auxiliary loss alone explained the failure.

Experiment 4.5: inspect whether the sensors contain object information

We audited contact and trained supervised diagnostic classifiers.

Contact was present and changed with the mesh. “All objects share the same grasp” was not a sufficient explanation.

A body-feature classifier achieved around 18% balanced accuracy across 50 categories, where chance was 2%.

What this established: the sensors were not empty or wholly uninformative.

However, contact varied substantially between different clips of the same object. Finger motion was replayed from the source trajectory rather than adapting to each replacement mesh.

So the stream contained useful information mixed with considerable interaction-specific variation.

Body-signal audit

Experiment 4.6: explicitly force the body branch to match words

We added a body-only word-matching loss, using the same ambiguous speech.

The body’s held-out word performance improved only slightly, and the combined model became worse than video-only.

What this established: merely applying stronger language pressure to the body pathway did not solve the association problem.

It could still learn whatever correlations were easiest in the training set.

Experiment 4.7: check training experiences versus unfamiliar experiences

Using saved models, we found that the body pathway could perform well on familiar training experiences.

With the extra body-language loss, body-only accuracy was around 75% in-sample, but around 28% on the unfamiliar person.

What this established: this was not simply a complete inability to fit the task. The learned mapping failed to generalize.

But the comparison changed both person and clips, so it could not tell which mattered.

Experiment 4.8: test unfamiliar clips from familiar people

We held out whole source clips while retaining familiar people.

Body still did not help: about 45.6% versus 48.3% for video-only under that historical protocol.

Body-only performance fell near chance even on new clips from familiar people. There was no substantial early advantage that later vanished.

What this established: unfamiliar people alone were not the problem. Generalization across interactions was already failing.

“Memorizing episodes” is a reasonable description of the pattern, but the exact shortcuts were not identified.

Experiment 4.9: replace additive fusion with the joint Transformer

This was the architecture that allowed video and body to interact directly.

Results on unfamiliar clips were approximately:

Model	Accuracy
Existing linear video learner	48.3%
Transformer, video-only	36.9%
Transformer, video + body	36.0%
Training loss improved while development performance peaked early and declined.

What this established: this particular Transformer setup generalized poorly, and body did not reliably help it.

It did not establish that Transformers or joint processing are fundamentally unsuitable. We simultaneously changed representation size, contextual processing, and other training characteristics.

Calling the cause simply “memorization” overstated what had been isolated.

Experiment 4.10: does body add information beyond video under correct labels?

This was a supervised 50-category classification test.

Video scored about 56.6%; video plus ordinary body signals scored 57.6%, an uncertain one-point difference. Performance worsened on the unfamiliar person.

A benefit appeared for some tiny or hidden objects on familiar-person clips, but did not carry over to the new person.

What this established: the existing sensors had not shown a robust incremental category benefit under that probe.

An earlier interpretation treated passing this probe as necessary for any language benefit. That was too strong: even partly redundant signals could help a finite-data learner discover associations. The probe was informative, not a universal verdict.

Experiment 4.11: add modeled wrist load and temperature

We created sensory consequences of assigned mass and material:

Load from mass and recorded acceleration.
Simplified skin-temperature responses to contact.
These were computed over recorded motion; the sampled physical properties did not change the motion itself.

Supervised classification improved with load. Under per-clip property resampling, the gain was about 2.7 points on familiar-person new clips and 6.6 points on the held-out person.

Temperature in its original form added little.

What this established: the modeled load supplied useful information under clear supervision.

But some information was built into our authored category-dependent mass distributions. Also, assigning fresh properties per clip did not create genuinely new meshes, motions, or people.

Experiment 4.12: revise the thermal signal

Originally, temperature change was multiplied by contact area. We separated temperature and area into distinct channels.

Temperature-plus-area alone still failed the primary criterion, but combining it with load improved supervised classification more than load alone.

What this established: that bundle contained useful information.

It did not isolate a temperature effect, because contact area changed too. A material-ID oracle was also tested, but it supplied privileged information that a thermal sensor cannot necessarily recover.

Experiment 4.13: put those physical signals into ordinary word learning

We added load and skin channels to the full additive body learner.

The supervised diagnostic gain did not carry over:

Video-only: 48.3%.
Video plus full body and physical channels: 42.4%.
What this established: information useful with explicit category labels was still not being successfully learned from weak speech.

Possible explanations included difficult association learning, interaction-specific shortcuts, and unhelpful score combination. The experiment did not isolate their contributions.

It also did not isolate load and skin from the other body channels.

Experiment 4.14 / recent Step 1: repair connections and scoring, then repeat

We corrected artificial clip-boundary mismatches and ambiguous evaluation cases while preserving intentional weak speech.

The corrected matched comparison was:

Video-only: 52.28%.
Video plus full body and physical signals: 47.54%.
What this established: those defects were real, but fixing them was insufficient to reveal a benefit.

The old and corrected percentages are not a clean before/after improvement measurement because the evaluation rules changed.

Experiment 4.15 / recent Step 2: heavy versus light

This used the existing additive learner with load alone as its body input.

Each lifting experience received a clearly aligned “heavy” or “light” word. The test used categories absent from diagnostic training.

Results:

Video-only: 37.88%.
Video plus load: 92.42%.
Shuffled load: 46.21%.
What this established: the learner can use a relevant sensory signal to learn a word distinction that generalizes in this narrow task.

But several things changed together relative to ordinary noun learning:

Two words instead of a broad vocabulary.
Clear alignment instead of ambiguous speech.
A directly relevant physical signal.
Load alone instead of the full body bundle.
Therefore, it does not tell us which of those differences enabled success.

Picture-only testing after load training showed no useful transfer. Given that sampled mass did not affect RGB or motion, we should not automatically expect a new picture to reveal that randomly assigned mass.

Experiment 4.16 / recent Step 3: sensory pretraining before word learning

Three matched conditions:

Visual temporal pretraining.
Visual temporal pretraining plus aligned sensory prediction.
The same with shuffled sensory targets.
Only the visual projection transferred into later video-only language learning.

Word accuracy was 52.63%, 51.87%, and 52.87%, respectively.

Sensory prediction improved, but about two-thirds of its improvement over a mean reference came from hand visibility. This is a share of the chosen summed loss, not a measure of how much knowledge was learned.

The saved language curves showed no substantial, consistent early advantage.

What this established: this sensory-pretraining procedure did not improve subsequent weak noun learning.

It did not show whether useful object information was learned in the retained projection, whether much of the improvement resided in discarded prediction layers, or whether early language updates changed that representation.

Full record for experiments 4.1–4.16

The connections I think are most useful for finding an insight

First: we have repeatedly demonstrated different pieces, but not their connection.

We have evidence that:

Visual features support better word mapping with clear labels.
Bodily signals contain some category or property information.
The learner can use informative extra inputs.
Load can support learning heavy/light under clear alignment.
What remains missing is:

Can the learner discover when and how a bodily experience relates to an ambiguously spoken word, and reuse that relationship in unfamiliar experiences?

That is more specific than “the model needs a better architecture.”

Second: our successful body task and failed noun task differ in more than difficulty.

For heavy/light, the signal and meaning are directly related.

For cup/spoon, a grasp can vary across uses, people, and object instances. The speaker may also be naming something other than the held object.

The learner must discover a stable relationship inside that variability. We have not established that our current experience distribution makes this relationship easy enough to learn.

Third: our data have more frames than independent experiences.

The archived corpus contains over 100,000 frames, but about 1,063 unique source clips overall. Some categories have very few independent clips.

Repeatedly showing the same motion can improve fit without teaching invariance across genuinely different interactions.

That makes your requested horizontal axis—independent experience—particularly important.

Fourth: some early design choices narrowed the question too much.

We initially required picture-only transfer, restricted body information to guidance, and even weakened category-predictive sensory features. Later we restored richer body inputs, but many diagnostics continued focusing on transfer.

Those choices were not equivalent to your original question about comprehension with video and body available.

Fifth: the failed Transformer remains an unresolved architecture test, not a closed door.

It did test real joint processing. But it also sacrificed visual competence. Its result justifies caution about that configuration; it does not justify either “joint processing cannot work” or “a bigger Transformer will fix it.”

Finally: our main missing result is still the matched acquisition curve.

Most experiments tested one amount of data and asked why an endpoint did not improve. They did not measure whether body changes learning across independent amounts of experience.

The history gives us useful constraints for constructing that comparison. It does not yet tell us that the two curves will separate—or exactly what must change for them to do so.

I only inspected existing records and code for this explanation. No new experiments or file changes were made.

10:35 AM

1 annotation
Can you output your response again, but when explaining each architecture or wtv, could you go into detail about how it was trained and stuff? that will give me better insight

11:14 AM
Yes—and your clarification matters. We are following the precedent of using benchmark performance as an operational measure of language acquisition. Word grounding is a legitimate primary outcome for that project. My earlier comment about sentence understanding was meant to describe the learner, not to suggest that we should replace your goal with a broader language-modeling task. Annotation 1

There is still a distinction worth preserving: our rendered-picture and Konkle tests follow that picture–word evaluation tradition; our body-present clip tests and heavy/light task are additional project-specific evaluations. They are not all exact reproductions of the same benchmark.

Below is the expanded explanation. I’ll explain what the learner receives, what answer training rewards, which components change, and what the experiment demonstrated.

A few terms make the rest easier to read.

Representation: a list of numbers describing an image region, word, or bodily state.
Loss: a numerical measure of how badly the model performed its training task. Training adjusts parameters to reduce it.
Epoch: one pass through the available training examples. Sixty epochs do not mean sixty times as much independent experience.
Development set: separate examples used to select a checkpoint.
Test set: separate examples used to report the selected model’s performance.
Seed: a choice controlling random initialization and training order. Multiple seeds reveal how sensitive the result is to those choices.
The central distinction is between learning to match the training examples and learning associations that work on unfamiliar experiences.

1. The main video–language learner: how it learns a word without receiving an explicit object label

This is the foundation of the current experiments, built on Frank’s vlm-headcam code.

Suppose the transcript is:

“Can you give me the cup?”

The accompanying five-frame window contains a cup, spoon, table, and hands.

What enters the learner

The images first pass through the pretrained DINOv2 visual encoder. We cache its outputs rather than repeatedly processing the pixels during training.

Each frame contributes:

One whole-image representation.
Sixteen spatial-region representations from a 4×4 grid.
A normal five-frame window therefore supplies 85 visual representations.

Each starts as a 768-dimensional vector. A trainable projection maps it to 512 dimensions.

The transcript is tokenized. Every word has a trainable 512-dimensional vector. The model averages those vectors to represent the utterance.

What the model predicts during training

It does not receive a label saying:

“Region 12 is the cup, and the fourth word names it.”

Instead, a batch contains many utterance–window pairs. The model scores every window against every utterance in that batch.

For example:

“Give me the cup”	“Look at the ball”	“Where is the spoon?”
Window paired with cup utterance	Should score highest	Lower	Lower
Window paired with ball utterance	Lower	Should score highest	Lower
Window paired with spoon utterance	Lower	Lower	Should score highest
The training objective rewards the observed pairing in both directions:

Find the matching utterance for a window.
Find the matching window for an utterance.
This is contrastive learning.

The paired utterance is treated as the positive example even when its relationship to the visible objects is ambiguous. The training system does not possess the speaker’s intended referent.

Also, a different window in the batch can contain another cup. “Different example” does not necessarily mean “different meaning.” That is a property of this instance-matching objective, not a proven explanation of our failures.

How the visual regions contribute

The original implementation used the best-matching region. Later versions used soft pooling: several regions can contribute, but stronger matches count more.

The learner can therefore discover that some region repeatedly matches utterances containing “cup.” It does not need an object detector supplying cup boxes.

What changes during training

The visual projection.
The word vectors.
A learned score-scaling parameter.
The large DINOv2 encoder remains frozen.

For the main ordinary-language comparisons, we generally used batches of 256, AdamW updates with learning rate 0.0003, and runs of up to 60 epochs. The corrected split contains 18,613 training pairs.

How it is tested

A word is compared with candidate pictures or candidate visual windows. The highest-scoring candidate is selected.

This is the word-grounding proxy we are trying to improve.

Why this matters: the learner must discover stable word associations from ambiguous experience. A lower training loss could reflect better instance matching without a corresponding improvement in reusable word meanings.

Visual learner and training implementation (line 74)

2. Contact-weighted training: changing which experiences get emphasized

This was the first substantial cue-learning program.

What changed

The underlying visual–language learner stayed essentially the same. We changed the weight of each training example according to contact timing.

An example associated with the cue could receive weight 1.0; another could receive weight 0.2.

So, if two examples produced the same prediction error, the high-weight example caused a larger training update.

In ordinary language:

“Learn more strongly from moments associated with touching.”

What the model did not receive

It did not receive a rich representation of finger posture, pressure, or load as information to use in its prediction.

Contact primarily influenced the training process.

How it was trained

The initial program used:

Twenty epochs per run.
Batch size 256.
Five random seeds.
Several speech–holding relationships.
No cue, real contact, shuffled contact, shifted contact, oracle, and noisy-cue conditions.
That produced 90 runs.

A separate confirmation used 25 fresh seeds across four conditions, producing another 100 runs.

What happened

The confirmation found:

Contact versus no cue: +0.94 percentage points, with uncertainty including zero.
Contact versus shuffled cue: +3.78 points.
Contact versus shifted cue: +4.33 points.
What it means

Correct timing was better than corrupted timing. But weighting with contact was not shown to improve over ordinary unweighted learning.

This tested whether contact identifies useful teaching moments. It did not test the full information-bearing body input you now want.

Contact experiments

3. Hand-guided region selection: teaching the learner where to look

The next idea was to use the body to guide the visual association.

What entered the body component

The early version used heavily simplified contact, grip closure, and wrist-position features. Camera geometry located the active hand in the image.

A small network produced a trust value:

“How strongly should this example favor regions near the hand?”

The first version: hard selection

We biased the region-selection process toward the hand.

For example, if the table initially matched an utterance slightly better than the cup, a hand-location bias could make the cup region win.

The visual–language loss then trained using the selected region’s actual visual score.

There was a technical complication: a hard winning-region choice does not provide a normal smooth gradient to the gate. The implementation used a differentiable approximation for training the gate.

The second version: soft selection

Instead of forcing one region to win, hand proximity changed the weights used when combining region scores.

That allowed the trust value to be trained through a smoother objective.

What was trained

The same visual projection and word vectors.
Small networks determining how much to trust the hand cue.
The image encoder remained frozen. During picture-only testing, the hand guidance was removed.

What happened

The hand-location geometry itself was useful, but the learned trust in the cue collapsed toward zero. The learner largely stopped relying on it.

A genuine design mistake

The early protocol deliberately weakened hand features when they predicted object categories. Detailed finger posture was reduced, and noise was added.

The intention was to avoid hidden labels or shortcuts. But it went too far.

If a real grasp reveals object shape, that is legitimate bodily information. Removing it because it predicts category can remove the mechanism we are trying to study.

What the result does and does not say

It shows that these particular attention-guidance mechanisms did not help. It does not show that rich sensory input is useless.

Original design and implemented changes

4. The additive body learner: simultaneous video, body, and language training

This is the main architecture used for the direct comparison.

What enters the body pathway

The ordinary body stream contains:

22 contact values: how much each hand patch touches.
90 finger-pose values: local finger rotations.
27 arm features: positions, changes in position, and gravity direction.
Later experiments add load and skin channels.

Each signal group passes through a small neural network. Their outputs are combined and processed by a recurrent network.

The recurrent network reads an approximately 11-second body window, five seconds before and after the utterance. It produces a summary describing the interaction around that moment.

Its state resets at clip boundaries.

Crucially, this is short-window processing, not an ongoing memory of the entire simulated day. Experiences influence future learning through parameter updates, but there is no persistent object memory carried from one training example to the next.

How body and vision affect the answer

The model computes:

Visual–utterance similarity
plus
a learned weight × body–utterance similarity.

Both terms participate in the same contrastive matching objective.

For a positive training pair, the optimizer tries to make the combined score favor the correct utterance–experience pairing.

What receives learning updates

Visual projection.
Word vectors shared by both pathways.
Body feature networks.
Recurrent body encoder.
Body-to-word projection.
The weight controlling the body score.
Thus the body pathway really does participate in language learning.

However, the combination is late: the body contributes a separate score. It does not directly modify the visual representation of a particular object region.

Why that limitation might matter

Suppose the learner sees two objects and feels a narrow grasp.

A joint mechanism could potentially learn:

“The narrow grasp applies to this particular visible object.”

The additive mechanism instead produces something closer to:

“Vision supports these words; body independently supports these words; combine their scores.”

That can work, but it places a considerable burden on the separate body–word association.

Body dropout

On half the training examples, the body contribution is removed.

This encourages the visual pathway to remain useful by itself. It is a design choice—not a demonstrated optimal setting.

Auxiliary sensory prediction

In the initial version, visual regions also predicted contact, contact change, aperture, and hand visibility. This added another loss to the main language loss.

Later ordinary-language comparisons switched this auxiliary objective off, allowing us to test whether it was causing the problem.

Body encoder (line 401) · Combined training objective (line 215)

The experiments using this architecture

Experiment	Training change or diagnostic	What happened
4.1: first full-body run	Train with real body and visual-to-sensory auxiliary loss	No clear picture-transfer or body-present benefit.
4.4: remove sensory prediction	Keep body input and language learning; turn off the auxiliary prediction loss	Still no benefit. The auxiliary loss was not the sole explanation.
4.6: add body-only word loss	In addition to the combined objective, force body alone to match the same utterances	Better training fit, little held-out improvement, worse combined performance.
4.7: inspect familiar versus unfamiliar experiences	Reuse saved models; compare training examples with unfamiliar-person examples	Body fitted familiar experiences much better than unfamiliar ones.
4.8: hold out clips from familiar people	Separate source clips into train/dev/test while keeping people familiar	Body still failed to help, so unfamiliar people alone were not the explanation.
4.13: add load and skin	Add physical-signal groups to the existing body encoder	Clear-label probe gains did not carry into weak-language learning.
4.14 / recent Step 1	Repair speech connections and scoring, then repeat the matched comparison	Video 52.28%; full body plus physical signals 47.54%.
The body-only word loss in 4.6 is worth understanding. It rewarded:

“From the body sequence alone, identify which utterance belongs with it.”

But it used the same weak speech. If the utterance named an unheld object, stronger pressure did not make the relationship clearer. It simply pushed the model harder to fit the supplied pairing.

By 4.7, the body-only model could score roughly 75% on familiar training experiences but around 28% on the unfamiliar-person test. That indicates a generalization problem. It does not identify exactly which nuisance patterns it learned.

5. The oracle body learner: make the extra information almost perfectly informative

This used the additive architecture, but replaced real sensors with a code identifying the held object.

Training

The model still learned word associations from speech. But the extra input could distinguish held categories directly.

A code might consistently indicate “cup,” although the model had to learn which word vector should correspond to that code.

This was deliberately privileged information, used as a diagnostic.

What was being checked

“If we put strongly useful information through the extra-input pathway, can the learner use it at all?”

Result

On held-referent trials, the body-only score reached about 94%. The combined score was lower, around 77% under that historical scoring setup.

Meaning

The pathway was capable of learning useful word associations. But adding the visual score could dilute a highly informative body score.

That exposed a combination/calibration issue.

It did not establish that real contact and finger signals were nearly as informative as the oracle.

6. The teacher–student experiment: transfer privileged knowledge into vision

This followed the oracle result.

The teacher

A trained model with the privileged held-object input. Its parameters were frozen.

The student

A fresh video-only learner.

What the student was trained to do

It received two pressures:

Match video and speech through the ordinary contrastive objective.
Produce a distribution of scores similar to the teacher’s.
The teacher supplied more than a single winning answer. It supplied relative scores over possibilities.

This is distillation.

We used a fixed teacher-softening temperature of 2 and a distillation weight of 1 in the recorded experiment, with a matched student control that did not receive the extra distillation loss.

What happened

Picture-only performance was essentially unchanged.

The student showed some increased agreement with the teacher during training, but that did not become a useful held-out advantage.

What remains unresolved

Failure to imitate a useful teacher could reflect the teacher’s targets, the objective, the student’s representation, or the weak correspondence between the teacher’s privileged information and what the student sees.

It did not prove the frozen visual features had no remaining capacity. The direct-label experiment contradicted that stronger interpretation.

7. Direct-label training: make the word–object association explicit

This was one of our clearest diagnostic results.

What changed

We retained the same frozen visual features and visual–word learner.

But instead of the ordinary caregiver utterance, training supplied the correct word for an object visible in the window.

So the model no longer had to solve the full referential ambiguity problem.

What stayed the same

Visual features.
Trainable projection architecture.
Word-matching mechanism.
General picture evaluation.
The diagnostic still had to identify useful regions, but the word supplied to training was appropriate to a visible object.

Training and selection

The recorded run trained for up to 30 epochs and selected its checkpoint using development performance.

Result

Rendered-picture accuracy improved from approximately 76% to 90%, with a substantial improvement on Konkle pictures too.

The insight

The learner could achieve much better mapping using the same visual foundation when the teaching signal was clear.

That does not mean we should replace weak speech with labels in the main experiment. It means the difference between clear supervision and ambiguous speech is an important part of the unresolved problem.

8. The joint Transformer: allow body information to change visual representations

This architecture changes where fusion happens.

Inputs

Projected visual regions from the image window.
Body states from the recurrent body encoder.
Learned indicators identifying modality and time position.
These are placed together as tokens in a shared Transformer.

What the Transformer does

Each token can use information from other allowed tokens.

A visual region’s representation can therefore depend on nearby body states and other visual regions.

Afterward, we retain the contextualized visual tokens and compare them with the utterance representation.

Unlike the additive learner, there is no separate body–word score added at the end.

Training

The final word-matching loss sends gradients through:

The word vectors.
Shared Transformer.
Visual projection.
Body projection and body encoder.
DINOv2 remains frozen.

The recorded configuration used:

Two Transformer layers.
Four attention heads.
Width 128.
Feed-forward width 512.
Sixty epochs.
A matched video-only Transformer control.
Body dropout of 50%.
No auxiliary prediction or distillation objectives.
What changed beyond fusion

The visual and word representation shrank from 512 to 128 dimensions. Independent visual regions became context-dependent. Transformer internal dropout was zero.

These choices matter because the experiment changed several things at once.

Result

On the historical unfamiliar-clip evaluation:

Existing linear video learner: 48.3%.
Transformer video-only: 36.9%.
Transformer with body: 36.0%.
Training loss decreased strongly, while development performance peaked earlier and declined.

Interpretation

The architecture could fit its training objective, but generalized poorly in this setup.

We did test genuine joint processing. We did not isolate whether its failure arose from reduced representation size, contextual shortcuts, optimization, regularization, or another issue.

Joint-learner review

9. The supervised probes: finding out what the inputs contain

These were separate classifiers rather than ordinary weak-language learners.

How they were trained

Each example came with a correct category label.

The body audit used a small multilayer classifier on contact, pose, and arm features. Other probes encoded video and body features separately, combined them, and predicted one of 50 categories.

The objective directly penalized the wrong category.

There was no need to infer which noun in an utterance referred to which experience.

Experiment 4.5: body information alone

Body features supported roughly 18% balanced accuracy across 50 categories, versus 2% chance.

This showed that the inputs contained learnable category information.

It did not establish that this information added much beyond vision.

Experiment 4.10: video versus video plus body

The supervised comparison gave approximately:

Video: 56.6%.
Video plus ordinary body: 57.6%.
That small gain was uncertain, and performance worsened on the held-out person.

So the ordinary sensor bundle did not demonstrate a robust incremental advantage in this probe.

Experiments 4.11–4.12: load and thermal information

We added modeled physical signals and trained the classifiers again.

Load improved classification. Revised skin signals combined with load improved it further.

But the signals came from assigned mass/material properties and recorded trajectories. The sampled properties did not physically alter the motion.

The revised skin bundle also contained contact area, so its advantage could not be attributed specifically to temperature.

The important distinction

These probes established:

“A classifier with correct labels can exploit some of this information.”

They did not establish:

“Our language learner can discover the same relationships from ambiguous speech.”

The weak-language run in 4.13 demonstrated that this gap was real for that configuration.

Probe and physical-signal experiment records (line 295)

10. Heavy/light: the additive learner on a clearly grounded property-word task

This was recent Step 2.

Inputs

Each example contained a lifting experience, visual features, and one aligned word: “heavy” or “light.”

The multimodal learner received only load as its body signal. The control received the same video and word.

The label was determined from the sampled mass, but mass itself was not a learner input.

Why this is an easier association problem

The learner did not have to decide whether a long utterance named the held object, another visible object, or something absent.

It was told one word at the relevant experience.

Training

88 independent training clips.
18 development clips.
22 test clips.
Three matched seeds.
One hundred epochs.
Batch size 16.
The existing 512-dimensional additive learner.
No auxiliary losses.
Body dropout of 50%.
Training, development, and test used different category sets. Test categories were pyramid and stapler.

What was rewarded

The same general word–experience matching mechanism now had just two word meanings to separate.

At evaluation, the learner selected “heavy” or “light.”

Result

Video: 37.88%.
Video plus load: 92.42%.
Shuffled test load: 46.21%.
What it demonstrates

This learner can use a relevant sensory signal to learn a word distinction and apply it to the held-out categories in this diagnostic.

What it does not isolate

Compared with ordinary noun learning, we simultaneously simplified the vocabulary, clarified alignment, changed the word type, and isolated a relevant body channel.

Therefore, we cannot yet say which simplification was decisive.

Heavy/light protocol (line 622)

11. Sensory pretraining: the recent two-stage experiment

This was Step 3, and its training sequence is especially important.

Stage A: learn from video and sensory targets, without language

We selected one fixed five-frame window from each of 631 training clips.

All three conditions learned a temporal objective:

“Represent adjacent frames from this experience as matching better than frames from other experiences.”

This operated over visual regions. It did not explicitly track the same object across frames.

The aligned condition also predicted:

Contact across hand patches.
Changes in contact.
Hand aperture.
Whether each hand was visible.
The shuffled condition predicted a different clip’s sensory sequence.

The sensory predictor used projected visual regions plus additional neural-network layers. Both the visual projection and those prediction layers could improve.

Training budget

Each arm received:

Fifty passes over those same selected windows.
Batch size 64.
Five hundred optimizer updates.
Three seeds.
Repeating those windows fifty times did not create more independent sensory experience.

The sensory objective was added at weight 0.5 to the temporal objective.

Stage B: learn words

We copied only the visual projection into a fresh video-only language learner.

We discarded the sensory prediction layers and initialized fresh word vectors, score scale, and optimizer.

The visual projection remained trainable.

Each model then received the same 18,613 corrected weak-language pairs for thirty epochs.

There were nine complete runs: three pretraining conditions × three seeds.

Result

Pretraining	Subsequent word accuracy
Visual temporal only	52.63%
Plus aligned sensory prediction	51.87%
Plus shuffled sensory prediction	52.87%
Aligned sensory prediction itself improved. But that improvement did not yield better word learning.

Why the training details matter

We measured the sensory skill of the full predictor, but transferred only one component.

The improvement could partly reside in the discarded layers. Also, most of the reduction in the summed sensory loss came from predicting hand visibility, which does not require learning object identity.

The saved language curves do not show a substantial early advantage that clearly disappeared later.

So the supported conclusion is that this preparation procedure did not help, rather than that useful embodied word knowledge was definitely learned and then forgotten.

Step 3 protocol and result (line 692)

12. The two older learner branches, before the current approach

These belong in the history, but they tested different formulations.

The AEA video–text–motion learner

This was a small three-encoder system:

Convolutional image encoder plus recurrent temporal encoder.
Word embedding plus recurrent text encoder.
Recurrent motion-sensor encoder.
Unlike the current bag-of-words text component, its text encoder processed a sequence.

Training combined three matching losses:

Video–text matching

a weighted motion–text matching loss
a weighted motion–video matching loss.
The intended benefit was that synchronized motion would improve the shared grounding learned during training, with motion withheld at the primary test.

The later development test trained the video–text learner for thirty epochs across group-safe folds and seeds. An easier 64-example capacity test ran for 120 epochs.

The task had serious problems: speech-derived action words often did not describe a visible wearer action, the visual learner failed the easier fit criterion, and some shuffled-sensor controls were infeasible. Later visible/coarse-action revisions stopped before their downstream learner tests.

Therefore, these results did not provide a clean test of embodied language benefit.

The EgoBabyVLM CLIP+ real-versus-synthetic pilot

This was a separate training system with a schedule combining:

Contrastive video–text learning.
Masked-language learning.
A DINOv2-based visual objective.
The recorded “triple” schedule used four contrastive updates for each masked-language and visual-objective update.

Its purpose was to compare learning from real and synthetic video. It was not the current three-input body experiment.

Initial real-data positive controls failed the learnability criteria. A later correction increased effective training coverage; its mean lexical gain was about 1.77 points, narrowly below the frozen two-point criterion.

The intended synthetic comparison was not completed in that recorded branch.

These older branches should not be counted as additional independent failures of the current body-input architecture.

How the remaining experiments fit together

A few important pieces were changes to data or evaluation rather than new learners:

Work	What changed	Why it mattered
Vocabulary expansion	Added objects and speech coverage for the evaluation vocabulary	Reduced the problem of testing words the learner barely encountered or never encountered.
Speech-coupling revision	Changed rates of naming held, visible-unheld, and unseen objects	Made the simulated teaching distribution more deliberate; weak alignment remained intentional.
Body-signal audit	Inspected contact, grasp variation, and supervised predictability	Distinguished missing signals from difficult-to-use signals.
Source-clip split	Kept all replays of one source clip in one partition	Tested generalization to genuinely unfamiliar clips rather than another replay.
Connection and scoring repairs	Removed artificial cross-clip associations and ambiguous test candidates	Made the matched comparison more trustworthy.
Saved-run analyses	Compared training/test fit, modality scores, and learning histories	Investigated failures without necessarily training another learner.
The vocabulary and naming changes also substantially improved early visual smoke results. That matters: the teaching distribution can change performance even when the learner architecture is unchanged. Those smoke runs were not themselves evidence of a body advantage.

The full numerical sequence is retained in the experiment archive.

The training-level insights I would carry forward

First, the ordinary objective rewards matching an utterance to an experience—not explicitly discovering which object a noun refers to. That discovery must emerge indirectly across examples. The correct-label result shows how much easier the task becomes when that uncertainty is removed.

Second, a body stream can contain information without our language loss teaching the model to use it appropriately. The supervised probes and heavy/light result demonstrate pieces of capability. Neither automatically solves weak noun association.

Third, our main body learner has short temporal context but no explicit persistent object representation across experiences. It can learn statistical regularities in its parameters, but it is not explicitly maintaining “this is the same object I handled earlier and am now hearing named.”

Fourth, our architectures impose different jobs on body information. Weighting says when to learn; hand guidance says where to look; additive fusion supplies another word score; joint fusion changes visual representations; pretraining changes the starting representation. Their failures are not interchangeable.

Fifth, we should distinguish a failed mechanism from a failed implementation of that mechanism. The Transformer changed several things at once. The sensory-pretraining test did not isolate what its transferred component learned. Those are limits on what the experiments ruled out.

Finally, the desired acquisition curve remains a separate measurement. Most of these runs repeated one fixed dataset for more epochs. They tell us about fitting and generalization at that data amount. They do not yet tell us how video-only and video-plus-body compare as independent experience increases.

No new training, inference, or file changes were made for this explanation.

11:15 AM

Can you answer my annotations after doing thorough thinking and research? Have subagents do the research for you as well so that the same amount of compute power is put for each annotation equally so as not to dilute response quality.







12:13 PM






Worked for 11m 5s



Yes. I split the scientific annotations across seven research reviewers, then checked their conclusions against the implementation, saved results, and primary papers.

The main correction is that our body encoder already learned alongside the words. What remains unresolved is whether the learner combines bodily and visual evidence effectively enough to learn from ambiguous speech. Joint fusion is a reasonable direction, but the research points to the learning objective, informative experiences, and evaluation design as equally important.

I also completed and pushed the documentation/archive reset. Details are at the end.

1. Were body signals added afterward instead of being learned together with words? Annotation 1

My earlier explanation made this confusing. In the additive learner, these components trained together from the first language-training update:
- The visual projection, which adapts the frozen DINO features.
- The word embeddings, which start without learned meanings.
- The body encoder, which converts contact, finger configuration, and movement into a learned representation.
- The connection between that body representation and the words.
For each training example, the model scored the accompanying utterance against the experience. Incorrect matches produced an error that updated all those trainable components together.

“Additive” describes the combination rule:
Overall match = visual–word match + weighted body–word match.

The limitation is that vision and body produce relatively independent judgments. Body information cannot directly change which visual region represents the relevant object.

This distinction also appears in successful multimodal systems:
System	How modalities learn together	Relevant lesson
CLIP	Separate image/text encoders learn compatible representations through matching.	Shared meanings do not require shared processing.
ImageBind	New modality encoders learn against an established image representation; its main setup freezes image/text encoders.	Adding a modality later can work.
FLAVA	Separate encoders coexist with a shared multimodal Transformer and objectives for different input combinations.	Alignment, joint interaction, and handling missing inputs are separate design choices.


These systems use much larger datasets and stronger pretrained representations than Nursery. They provide architectural ideas, not an infant-learning recipe. CLIP, ImageBind, FLAVA.

The useful question for us is: should body and vision merely contribute separate evidence, or should each help interpret the other?

2. Have Transformers been used for this kind of developmental language learning? Annotation 2

Yes—and our current visual backbone is already a Transformer. DINO is a training method; DINOv2 uses a Vision Transformer. Three different architectural choices need separating:
Choice	Its job
Visual Transformer	Turn images into useful visual features.
Language Transformer	Interpret words in context and order.
Multimodal Transformer	Let visual and bodily observations interact.


There is direct developmental research:
- Vong et al., 2024: the main learner used a visual encoder and averaged word embeddings. They also tested a visual/text Transformer variant; it did not outperform the main model on their reported word-recognition tests. Paper and author code.
- Vong and Lake, revised January 2026: compared averaged embeddings with a small GPT-2-style language encoder, with and without an additional language-modeling objective. No language-encoder variant consistently won noun classification; some richer semantic measures benefited. Study.
- EgoBabyVLM: uses Transformer-based visual and language components, including DINOv2/BERT and DINOv2/GPT-2 configurations. Those models still struggled with weakly aligned naturalistic input. Paper.
So investing in a Transformer for a specific missing capability is justified. “Transformers are more powerful, therefore this will improve” is not supported.

For our current question, the most relevant capability is letting body information condition visual interpretation—not necessarily replacing the word encoder or the entire visual learner.

3. What are “video features,” and why predict bodily signals or learn temporal consistency? Annotation 3

A video feature is a numerical description extracted from an image.

For example:
Image pixels → DINO → numbers describing patterns in different image regions.

Those numbers can encode useful distinctions involving shape, texture, hands, or objects. They are not explicit labels such as “this is a cup.” Our trainable visual projection then adapts those numbers for the task.

Sensory prediction asked:
Can the visual representation help predict contact, hand configuration, or whether the hand is in view?

The intended benefit was indirect. To predict contact successfully, perhaps the visual representation would become more sensitive to hands interacting with objects. Those improved features might later help word learning.

Temporal consistency asked the representation to connect nearby views within a clip. Intuitively:
The object moved or rotated between these frames, but some aspects of the experience remain the same.

This can encourage useful stability. However, nearby frames also share backgrounds, people, and lighting. A model can learn those regularities without discovering reusable object categories. Our objective did not explicitly track “this particular object.”

Related work shows that carefully chosen temporal or cross-modal objectives can learn useful representations, but the choice of what counts as a matching pair matters greatly. Time-Contrastive Networks, Learning rich touch representations through cross-modal self-supervision.

4. Why discard the sensory predictors? Isn’t sensory pretraining followed by language training the obvious approach? Annotation 4

That sequence is essentially what we ran—but with a particular interpretation of “sensory pretraining”:
First:
video → visual representation → predict bodily measurements

Then:
video → transferred visual representation ↔ learn words
Body measurements were targets to predict, not inputs to a trained body encoder.

We retained the visual representation and removed the layers producing contact/aperture predictions. The reasoning was that the useful learning should remain in the representation underneath those output layers.

This is a standard transfer-learning strategy. SimCLR and BYOL, for example, discard training-specific components and retain an encoder. But it is an empirical choice: SimCLRv2 found benefits from retaining part of a deeper projection head. There is no rule that all useful knowledge must sit below whichever layer we choose to discard. SimCLR, BYOL, SimCLRv2.

Your proposed alternative could instead mean:
Learn visual and bodily representations first, retain both, then teach words while both inputs remain available.

That is a different experiment, and much closer to the central V versus V+B comparison. Step 3 did not test it.

5. Is the body window too large? Why include observations before and after speech? Annotation 5

It could be too broad for some events, but the available evidence does not establish that.

First, a correction: it contains 11 samples spanning ±5 seconds, rather than 11 elapsed seconds.

There is developmental motivation for including context around naming. Studies found that successful naming often accompanied sustained handling or visual dominance extending several seconds before and after the utterance. These findings are correlational; they do not establish an optimal model window. Yu and Smith, Schroer and Yu.

The intuitive reason is straightforward:
A child picks something up, inspects it, hears its name, and continues holding it.

The useful bodily evidence need not peak exactly when the word is spoken.

Our saved data also show that the nominal window often covers much of the episode: the median contiguous sampled clip run contains eight points, and only about 18.5% of speech pairs have all 11 same-clip samples available. The window is clipped at episode boundaries.

Two issues therefore matter:
- Which interval is available?
- Can the learner select the relevant moments within it?
The additive learner uses a recurrent summary, not a uniform average. Still, that summary may mix useful and irrelevant events.

My preference is to think in terms of the relevant handling event, with uncertainty about timing, rather than declaring a universal shorter window optimal. Using future observations is legitimate for offline learning, but it cannot represent an immediate decision made before those observations occur.

6. Does resetting mean the model has no ongoing memory? Annotation 6

It has one kind of memory and lacks others:
Kind of memory	Present?	Example
Knowledge accumulated in trained weights	Yes	Repeated experiences gradually teach what “cup” tends to accompany.
Working context outside the supplied window	No	A grasp 20 seconds earlier cannot directly inform the current example.
Retrieval of a particular earlier episode	No explicit mechanism	It cannot deliberately retrieve the cup interaction from earlier in the session.


The recurrent network is recurrent within an example. The joint Transformer also processes each example independently. Neither automatically remembers previous training examples as episodes.

This is not inherently a fatal limitation. Successful developmental word-learning models have learned useful mappings from independently processed frame–utterance pairs.

There is also a reason to retain some resets here: our simulator concatenates independently selected motion clips with hard cuts. Carrying immediate sensorimotor state across those cuts could connect unrelated experiences.

Persistent context could help if the relevant experience genuinely continues across time—for example, manipulation now and naming later. But that requires coherent sequences and appropriate chronological processing. Simply leaving the recurrent state running across shuffled minibatches would be wrong.

Architectures such as Transformer-XL demonstrate how context can be carried across segments; they do not establish that doing so fixes embodied noun learning. Transformer-XL.

7. Should we investigate joint processing? Annotation 7

Yes. Your reasoning identifies a plausible limitation of independent scores.

Imagine seeing two objects while feeling a narrow grasp. The grasp alone may not identify a noun. Its value could be:
“Of those visible things, this one is consistent with what I am handling.”

That requires relating the bodily evidence to particular visual candidates.

A joint mechanism allows the body to influence how visual regions are represented or selected before matching them to words. The body need not become a strong standalone noun recognizer to be useful.

Cross-attention models such as LXMERT explicitly learn interactions between streams, while ALBEF combines alignment with joint processing. These establish useful design principles, although their supervision and training scale differ substantially from ours. LXMERT, ALBEF.

The question is therefore well motivated: can bodily evidence improve interpretation of an ambiguous visual experience? Our previous joint pilot tested one implementation of that idea, not the entire possibility.

8. What was the auxiliary sensory-prediction loss doing? Annotation 8

It gave the learner an additional assignment alongside language learning.

The main assignment was:
Match this utterance to its accompanying experience.

The extra assignment was:
From the visual representation, predict contact, changes in contact, hand aperture, and hand visibility.

The intended effect was to make the visual representation attend to interaction-relevant information rather than learning only whatever happened to reduce the language error.

Mechanically, this auxiliary assignment updated the visual projection and sensory-prediction layers. It did not itself train the body encoder to understand words. The ordinary language objective trained that body pathway.

There were therefore two distinct uses of sensory prediction:
Use	When it happened
Earlier auxiliary objective	During language learning, alongside the language loss.
Step 3 pretraining	Before language learning, followed by transfer of the visual projection.


An auxiliary task helps only if the features it encourages are useful for the main task, and its influence is appropriately balanced. Predicting hand visibility well does not necessarily help distinguish “cup” from “spoon.”

We subsequently disabled this auxiliary objective in ordinary-language comparisons. That did not rescue the body-input result, so this objective alone is not an established explanation for the failure. [Historical record](/Users/rishisim/Documents/research/nursery/archive/language_learning/experiments.md)

9. Should the previous Transformer trial have been more carefully controlled? Can we guarantee the next one works? Annotation 9

It should have supported a cleaner interpretation.

The trial changed several things together:
- Visual and word representations shrank from 512 to 128 dimensions.
- Independent visual-region processing became shared contextual processing.
- Temporal and modality embeddings were introduced.
- Body information entered through attention rather than an independent word score.
- Transformer-layer dropout was zero, although other dropout and weight decay remained.
The historical video-only Transformer scored about 36.9%, compared with 48.3% for the earlier linear reference. Its body version scored about 36.0%. Those numbers use the historical evaluation protocol.

This tells us that the replacement lost substantial established performance. It does not identify dimensionality, regularization, attention, or any particular shortcut as the cause.

The matched Transformer video/body comparison remains informative about that implementation. It just cannot settle whether joint fusion is generally better or worse.

We can require:
A verified implementation, preserved baseline, justified training choices, fair comparisons, and a result we can interpret.

We cannot guarantee a positive body effect. More modalities can make optimization and generalization harder, even in established multimodal tasks. What Makes Training Multi-Modal Classification Networks Hard?.

Also, “maximize information” should mean retaining legitimate sensory evidence—not assuming every available channel must improve the result.

10. Isn’t heavy/light an important positive result? Annotation 10

Yes. It is an important result, and it should remain a central positive reference.
Condition	Held-out heavy/light accuracy
Video	37.88%
Video + load	92.42%
Same body-trained model with shuffled test load	46.21%


The learner connected informative bodily measurements to a word distinction and used that connection on two categories excluded from diagnostic training. Shuffling the load largely removed the advantage. [Saved comparison](/Users/rishisim/Documents/research/nursery/outputs/runs/step2_heavy_light/comparison.json).

But this success used the additive architecture. Body and words trained together from the beginning; no shared visual/body Transformer was required.

Therefore the strongest lesson is:
The existing learning machinery can acquire a meaningful sensory–word relationship when the signal and learning opportunities support it.

The challenge is extending that ability to weaker speech, more meanings, and more varied situations. Joint fusion might help with that extension, but the diagnostic does not prove it is necessary.

There is also a physical distinction: sampled mass changed the analytic load without changing the recorded video or movement. That created information available through body input that was not consistently recoverable from the corresponding picture. We should preserve this result as evidence of body-present understanding, rather than require picture-only transfer for it to count.

11. Why did we do Step 3, and did we lose what it learned? Annotation 11

The intended reasoning was:
Perhaps ambiguous language is too difficult to learn alongside sensory structure. First learn something useful about interactions without words, then use that preparation when words arrive.

That is a reasonable hypothesis. The chosen implementation asked a narrower question:
Does predicting bodily measurements improve a visual representation enough to help subsequent video-only word learning?

It was not necessary to complete that experiment before directly measuring V versus V+B acquisition curves.

The saved results establish:
- The complete aligned sensory predictor learned its prediction task better than controls.
- The transferred visual representation did not produce a reliable noun-learning improvement.
They do not establish that useful word-relevant knowledge was learned and subsequently forgotten.

Possible explanations include:
- Much of the improvement resided in the discarded prediction layers.
- The learned distinctions were useful for sensory prediction but not noun discrimination.
- Language training changed the transferred features.
We did not measure post-language sensory retention, so the last explanation remains untested. Most of the improvement in the summed sensory loss came from hand visibility, which also weakens the assumption that better sensory prediction necessarily meant better object representations.

Crucially, the sensory predictors were removed before language training, not merely during evaluation. [Exact experiment and results](/Users/rishisim/Documents/research/nursery/archive/language_learning/experiments.md#step-3-transfer-focused-sensory-pretraining).

Your concern about relevance to the main goal is justified. This was a bounded transfer diagnostic; it should not become a prerequisite that keeps delaying the direct comparison.

12. How have researchers handled uncertainty about what words refer to? Annotation 12

This is one of the fundamental problems in the field, and several approaches offer useful ideas.

Start with a simple example:
Encounter	Heard	Present
1	“dax”	An unfamiliar object, a spoon
2	“dax”	Another similar unfamiliar object, a ball
3	“dax”	A third similar unfamiliar object, a book


No individual encounter provides a certain answer. Across encounters, the recurring object category becomes a stronger candidate than the changing distractors.

That is cross-situational word learning.

Accumulating uncertain associations. Yu and Smith demonstrated this in adults; Smith and Yu demonstrated learning from individually ambiguous exposures in infants. The controlled studies establish that uncertainty within an encounter need not prevent learning across encounters. Their simplified displays and reliably present referents are easier than natural video. Adult study, Infant study.

Proposing a meaning and checking it later. Another approach is to retain a candidate interpretation and revise it when later evidence contradicts it. Trueswell and colleagues found adult behavior consistent with this strategy in their tasks. That does not establish that infants always retain exactly one hypothesis. For Nursery, an absent referent must not automatically invalidate a meaning: natural speech can concern something outside view. Propose-but-verify study.

Learning meanings and intentions together. Michael Frank, Noah Goodman, and Josh Tenenbaum modeled two linked questions:
What might these words mean?
Given those tentative meanings, what might the speaker be referring to here?

Tentative answers reinforce or revise one another. Their model also distinguishes referential from nonreferential language. However, it received human-coded object categories; it did not discover objects directly from pixels. Frank, Goodman and Tenenbaum.

Learning from a collection of possible visual matches. Multiple-instance learning treats a scene or time window as several candidates. The training signal can be explained by some region rather than the entire picture. Nursery already uses a version of this.

Its weakness is that the best-scoring candidate may be a recurring background, hand, or distractor. Giving the model candidates is not the same as teaching it to choose the actual referent.

A particularly relevant October 22, 2026 preprint from Frank and colleagues studies scaling on 2,441 hours of children’s video. Stronger visual representations substantially improved word mapping; informative naming moments carried much of the learning signal. Region selection and wider temporal windows provided conditional rather than universal gains. Their alignment analysis used an external model, so it does not demonstrate that a learner independently discovered those moments. Scaling cross-situational word learning.

For Nursery, my inference from this literature is:
Body information may be valuable because it helps the learner decide which experiences belong together and which candidates explain an utterance—not just because it independently predicts a noun.

We should preserve uncertainty: the utterance might concern the held object, another object, or no visible object. Body signals should contribute evidence, not impose “the held object is always the answer.”

13. Why need object continuity if the same word appears repeatedly? Annotation 13

Repeated words can support category learning. You are right about that.

But same word does not mean same physical object:
- Two different cups can both be called “cup.”
- One cup can be called “cup,” “that,” or “yours.”
- Many interactions happen without any name being spoken.
- At the beginning, the learner does not yet know what “cup” refers to.
Consider this sequence:
See an object clearly → pick it up and partly obscure it → hear a name → release it and see it clearly again.

Object continuity could connect the clear views, the bodily interaction, and the ambiguous naming event. Otherwise those observations may remain only loosely associated.

This separates two tasks:
Task	Question
Temporary object continuity	Are these observations about the same individual thing?
Word/category learning	What do different things called “cup” have in common?


Continuity could help assemble better evidence for category learning. It is not itself the category meaning.

It is also optional in the scientific sense: useful noun learning can occur without an explicit object-memory system. We should test its value rather than presume it is required. Simulator object IDs may help us audit or score continuity, but feeding those IDs to the learner would supply privileged information.

14. Which architecture is most defensible now? Annotation 14

My recommendation is a small mechanism that lets body information condition visual representations, trained directly through language learning, while preserving the working visual pathway.

Conceptually:
Video → existing visual representation ──────────────┐
                                                    ├→ body-conditioned
Body → learned sensory states → interaction module ─┘   visual evidence
                                                           ↕
                                                      learned words
The important choices would be:
- Preserve the established 512-dimensional visual/word representation.
- Let visual regions consult relevant bodily states.
- Retain temporal states so the interaction can distinguish different moments.
- Train the body encoder, interaction module, visual projection, and words together through the same weak-speech objective.
- Begin with an interaction module that initially leaves the working visual pathway unchanged.
That last idea has precedent in Flamingo’s gated cross-attention connections. It is an engineering precaution we can adapt, not evidence that a large-model recipe will work unchanged at our scale. Flamingo.

Pretraining is a separate decision. If used, the representations intended to help later should actually remain available during language learning. Direct vision–touch–language alignment has useful precedent, but existing work often uses explicit tactile descriptions and substantial pretrained semantics. Touch, Vision, and Language.

I would not commit us to “joint fusion plus pretraining is best.” Joint interaction is a defensible candidate; pretraining remains an additional hypothesis. Both should be judged by the acquisition curves, rather than by success on an intermediate prediction task.

15. Make acquisition curves the goal, and archive the previous work. Annotation 15

Agreed. The active question is now:
As independent experience increases, does a learner receiving video and body acquire language better than a matched learner receiving video alone?

The comparison needs:
- Increasing independent experience, rather than more repetitions of the same clips.
- Matched video, speech, splits, and evaluation situations at each data amount.
- A separately trained video-only reference.
- A declared training and model-selection budget.
- Pretraining, if used, restricted to each point’s allowed training experience.
- Body-present understanding as the primary comparison; picture-only transfer reported separately.
A low-data advantage would show faster acquisition. A sustained advantage near measured saturation would support a higher attainable level. Neither outcome should be assumed beforehand.

The documentation reset is complete:
- [PLAN.md](/Users/rishisim/Documents/research/nursery/PLAN.md) now centers that acquisition-curve objective.
- [Past experiments](/Users/rishisim/Documents/research/nursery/archive/language_learning/past_experiments.md) explains the architectures, training interventions, results, and unresolved interpretations.
- The older literature review is now [archived](/Users/rishisim/Documents/research/nursery/archive/language_learning/body_input_review.md).
- The README, archive index, historical cross-references, and learner status were updated.
I preserved the datasets and run artifacts at their recorded, ignored paths. Necessary source modules remain because some historical modules still supply live dependencies and reproduction support. Generated datasets and checkpoints were not uploaded to GitHub.

Documentation links and anchors, the diff, and preservation of the detailed historical protocols were checked. No training or implementation changes were made. Commit b4746250 is pushed, and the working tree is clean.


12 annotations





Worked for 1m 49s



My recommendation is to refine joint fusion while preserving the working visual learner, train it directly on language with all existing body streams, and measure acquisition curves. Leave sensory pretraining, persistent memory, and explicit intention modeling out of this first design.

The research gives us a reason to choose that direction. It does not establish that joint fusion will resolve our particular weak-speech problem. The design should make a positive or negative result interpretable.

What the direct comparisons say Annotation 1 Annotation 5

I found three particularly useful comparisons:
Study	What it compared and found	What we can take from it
Hendricks et al., “Decoupling the Role of Data, Attention, and Losses”	Compared multimodal attention against processing each modality separately. Cross-modal interaction outperformed deeper modality-specific alternatives on image retrieval. Dataset noise still strongly affected performance.	Interaction can contribute something that more independent processing cannot. It does not automatically solve noisy correspondence.
Nagrani et al., “Attention Bottlenecks for Multimodal Fusion”	Compared early, intermediate, and late fusion of audio/video. Intermediate interaction performed best in their ablations. On Epic-Kitchens, their model achieved 43.4% action accuracy versus 37.9% for late fusion.	Separate encoders followed by limited interaction are a strong alternative to either independent scores or mixing everything immediately.
Tsai et al., “Multimodal Transformer for Unaligned Multimodal Language Sequences”	Compared cross-attention against early- and late-fusion Transformers. On unaligned MOSEI sentiment classification, binary accuracy was 81.6% versus 78.9% and 78.6%, respectively.	Retaining temporal observations and learning which moments correspond can outperform compressing each stream separately.


Sources: Hendricks et al., Attention Bottlenecks, MulT.

These comparisons are relevant, but their fusion baselines are not exact copies of Nursery’s additive scorer. Also, temporal misalignment and referential ambiguity are different problems:
- Temporal misalignment: the useful gesture happened before the useful sound.
- Referential ambiguity: the speaker might mean another object—or something absent.
MulT addresses the first particularly directly. None of these studies establishes that joint fusion solves infant-style learning from ambiguous caregiver speech.

Nevertheless, they support a concrete choice: let separately encoded visual and bodily observations interact before the language decision. That is stronger evidence than simply choosing a Transformer because it is more expressive.

What our two architectures taught us Annotation 4 Annotation 12
Previous architecture	What worked	What we should change or preserve
Additive	Body and words learned together. Heavy/light demonstrated that the pathway can learn a useful sensory–word association. The visual pathway remained comparatively strong.	Preserve the visual representation and simple language objective. Improve the ability to relate body evidence to particular visual observations.
Joint Transformer	It genuinely allowed visual/body interaction. Its matched video/body comparison tested that implementation.	Avoid simultaneously replacing visual processing, shrinking visual/word representations from 512 to 128 dimensions, and changing several architectural choices.


The additive model’s limitation is not that its body representation was untrained. It is that its body score cannot depend on the particular visual candidate being considered.

For example:
A narrow grasp alone is ambiguous. A narrow grasp alongside these two visible objects could help distinguish which interaction is occurring.

The joint pilot could represent such relationships, but it also substantially weakened the video-only control: historical accuracy was about 48.3% for the linear reference, 36.9% for joint video, and 36.0% for joint video/body. These are historical-protocol numbers, not directly comparable with the repaired evaluation.

We therefore learned two things:
1. The information-bearing body pathway can work.
2. Replacing the established visual learner wholesale was an expensive way to investigate interaction.
Neither result identifies a universal winner. Together, they favor a smaller modification to the working learner. [Archived architecture results](/Users/rishisim/Documents/research/nursery/archive/language_learning/past_experiments.md).

The architecture I would choose

Keep the existing visual and word representations, and add one small cross-attention block through which visual regions can consult the body sequence:
Video → frozen DINO → existing 512-dimensional visual features
                                      │
                                      ▼
                       Small interaction with body states
                                      │
                                      ▼
                      Combined visual-and-body features
                                      ↕
                             Learned word vectors

All body streams → existing body encoder → states at each time
In plain terms, the interaction asks:
“For this visual observation, which bodily observations are useful, and what information should they contribute?”

The design commitments would be:
- Preserve the 512-dimensional visual/word interface. The attention module may use smaller internal calculations without shrinking that established representation.
- Preserve the original visual path. The body-conditioned contribution is an adjustment to it, initially having zero effect.
- Retain body states across time. Do not collapse the whole interval into one vector before the interaction.
- Train through the ordinary language-matching objective. Words, visual projection, body encoder, and interaction module learn together.
- Keep bodily information usable at the output. Load or skin information need not become visually predictable to contribute to word understanding.
- Use no separate sensory-pretraining stage initially. That would introduce another explanation for the result before we have established the direct acquisition comparison.
The zero-effect initialization is a precaution: before training, the new module should reproduce the original visual scores. It does not guarantee that training will preserve performance. Gated cross-attention has precedent in Flamingo, although its scale is very different from ours.

This is a refinement of joint processing, not a return to adding two independent word scores. The contribution of body information now depends on the visual input.

The timing decision Annotation 2

I would make the following choice for this comparison:
Keep the existing ±5-second body window, bounded by the source clip, but expose its individual time steps to the interaction module.

That resolves the immediate design question without simultaneously changing the available data and the fusion mechanism.

Specifically:
- Keep timestamps relative to the utterance.
- Mask missing observations and observations beyond the source clip.
- Let each visual region attend differently to the available body states.
- Keep the current visual window identical across the compared learners.
- Allow the body contribution to be small when it is unhelpful.
The existing body encoder already produces per-time states; the additive model uses a central summary, while the joint implementation exposes states to attention. We can reuse that capability rather than introduce a new event-segmentation system. [Current implementation](/Users/rishisim/Documents/research/nursery/nursery/learner/model.py).

This choice addresses how evidence is selected, while holding the candidate interval fixed. It does not establish that ±5 seconds is optimal. If timing remains a specific suspected limitation afterward, we can compare intervals using development data under a bounded protocol.

For now this remains offline learning: the model may use observations arriving after the utterance. We should describe it that way, rather than imply an instantaneous infant response.

Persistent memory and more elaborate learning mechanisms stay deferred Annotation 3 Annotation 11 Annotation 8 Annotation 9

I agree with your scope decisions:
- No persistent memory across separate clips.
- No explicit object-identity tracker.
- No propose-and-verify machinery.
- No model of the speaker’s intentions.
Repeated experiences can teach word/category associations through shared parameters. We can reasonably start with that mechanism.

What we should avoid claiming is that persistent memory would never help. It simply is not required for the next test, and adding it would make the result harder to interpret.

The learning objective can remain simple:
Make language compatible with its accompanying experience, while distinguishing it from mismatched experiences.

Include all existing bodily streams—but distinguish availability from forced use Annotation 6

Yes. The proposed full-body condition should include all five existing learner input groups:
Stream	Available information
Contact	Contact across hand patches
Finger configuration	Finger rotations and grasp configuration
Arm/proprioception	Positions, movement, and gravity-relative information
Load	Modeled forces and their magnitude/spread
Skin	Modeled temperature change and contact area


These are already supported by the [body data loader](/Users/rishisim/Documents/research/nursery/nursery/learner/body.py). They should be normalized using training data only, with explicit missing-data handling.

“All streams available” should not mean “every stream must contribute equally.” A model should be able to use load for a weight-related distinction and largely ignore it for a distinction where it is irrelevant.

We also need the evaluation vocabulary and situations to contain opportunities for those signals to matter. If the entire evaluation concerns visually obvious object categories, providing temperature or load does not ensure a benefit. A sensible language evaluation can retain noun recognition while also reporting property-word understanding where the experience and speech support it. Those scores should remain visible separately, rather than being mixed into a favorable aggregate.

That evaluation decision matters just as much as enabling the input channels.

Keep weak alignment intentional Annotation 10

Agreed. The design must not encode:
“The speaker is naming whatever is being held.”

Both learners should receive the same weak, delayed, sometimes nonheld speech. Contact can help interpret an experience without becoming an answer label.

The joint interaction should receive sensory observations—not generator-selected referent IDs or the hidden reason an utterance was emitted.

There is an important limitation to acknowledge: preserving natural weak speech does not itself make the current contrastive objective fully aware that some utterances have no visible referent. For this first comparison, I would keep that objective fixed rather than simultaneously add a new uncertainty model. That makes the architectural comparison interpretable.

Would a small cross-situational diagnostic help? Annotation 7

Yes, as one bounded capability check—not as another extended phase before the acquisition curves.

It would answer something heavy/light did not:
Can the learner accumulate a word association when each individual encounter contains several plausible referents?

A useful version would:
1. Use a small set of unfamiliar word forms and existing object experiences.
2. Vary the distractors across encounters, so no single encounter reveals the answer.
3. Give the learner ordinary observations and utterances, without referent labels.
4. Test on held-out source clips and varied views.
5. Compare video-only, additive body, and the proposed joint body learner under the same conditions.
To establish a body contribution, the situations must contain relevant sensory evidence, and a controlled disruption of its correspondence should reduce the advantage. Otherwise we would demonstrate cross-situational learning without explaining embodiment.

The interpretations would stay narrow:
Outcome	What it would tell us
Both V and V+B learn similarly	The association is learnable; body has not added demonstrated value there.
V+B learns better, and disrupting body correspondence removes the gain	Body helps resolve this controlled ambiguity.
Neither learns	This controlled setting already exposes a limitation worth locating.


Even a positive result would not prove that the remaining full-task problem is specifically “generalization.” It would show competence under those particular conditions.

I would include this only if its result will change the next design decision. The main deliverable remains your acquisition plot.

What we must verify before trusting that plot Annotation 12

The previous mistakes suggest a compact set of requirements:
- Preserve competence: the new module reproduces the established visual scores at initialization.
- Verify actual learning paths: language training reaches every enabled body group and the interaction module; a neutral gate must be able to open.
- Verify timing and masking: padding, hard cuts, and missing sensors cannot supply accidental evidence.
- Keep the score consistent: training and evaluation use the same intended matching rule.
- Keep the comparison matched: independent experience, speech, splits, selection rules, and training opportunity are specified in advance.
- Retain the additive reference: it distinguishes “body helps” from “joint fusion helped more than the previous mechanism.”
- Inspect held-out learning across data amounts: training fit alone cannot justify proceeding as though generalization improved.
The primary curves should be video-only, additive V+B, and refined joint V+B. That directly answers both questions we now care about: whether body helps, and whether interaction improves how it helps.

Your desired curve is the hypothesis. The purpose of this design is to give that hypothesis a fair test while preserving enough controls to understand a different outcome.

No files or experiments were changed during this review.


3:45 PM







3 annotations





Worked for 1m 18s



Your distinction is correct: the learning target is the relationship between language and the whole experience—not the usefulness of body information to vision. Rechecking the code and literature makes me revise one part of my recommendation: we should preserve direct access from both modalities to language, then add interaction between them.

1. Body should contribute to word understanding directly Annotation 1

Consider three cases:
Word distinction	Potentially useful evidence
Red versus blue	Mainly vision
Heavy versus light	Mainly bodily load
Which visible object is being handled?	The relationship between vision and body


We need a learner capable of all three. Body information should not have to be visible, predictable from video, or useful for visual recognition before it can contribute.

My previous proposed module was trained through a language loss, so “vision asks body” did not mean its objective was to improve vision. Such a module could carry weight information into its output even if the image contained no weight cue.

However, making visual tokens the only route to the language decision is an unnecessary restriction that we have not justified. The existing joint implementation does exactly that: it processes visual and body tokens together, then retains only the visual-token outputs for word scoring. That does not prove information was lost, but it deserves scrutiny. [Current implementation (line 324)](/Users/rishisim/Documents/research/nursery/nursery/learner/model.py:324).

The corrected design I recommend is:
Visual representation ───────────────────┐
                                        │
Body representation ─────────────────────┼→ Match the experience to language
                                        │
Learned visual–body interaction ─────────┘
In practical terms: retain the established visual–word and body–word pathways, and add a small joint interaction pathway. The language objective trains the combined decision.

This also gives us a cleaner experiment:
Does adding interaction improve on an otherwise matched additive learner?

Adding an interaction contribution does not make this equivalent to the old additive model. The difference is that the new contribution depends on the combination of vision and body, rather than calculating their evidence independently.

For example, “this grasp” could mean different things when combined with different visible objects. Meanwhile, load can still support “heavy” directly.

The research supports preserving both shared and modality-specific information. FactorCL explicitly studies failures that arise when learning emphasizes what modalities share while neglecting useful information unique to one modality. It supports this design principle, although it does not prove that our previous model lost such information. FactorCL.

More directly, the Touch, Vision, and Language study found that including tactile–language supervision improved tactile–language alignment compared with omitting that connection. Its descriptions and pretrained models make it a different learning setting, but it reinforces the value of a direct sensory–language relationship. TVL.

We do not need to import those papers’ entire training recipes or add another auxiliary loss. Our ordinary language loss already reaches the body pathway.

After rechecking, these earlier recommendations still stand:
- Preserve the working 512-dimensional visual/word representation.
- Include all existing body streams.
- Preserve temporal body states and clip-boundary masks.
- Train directly through language matching.
- Keep persistent memory, intention modeling, and sensory pretraining outside this first comparison.
The revision is specifically to stop making body information pass exclusively through a visual representation.

2. Does the learner automatically discover which streams matter? :cod-explanation-placeholder{}

It learns that from training; we do not manually assign a stream to each word. But successful selection is not automatic merely because the architecture permits it. Annotation 2

A simplified learning sequence is:
1. The learner encounters an utterance alongside video and body observations.
2. It scores that utterance against the experience and against mismatched experiences.
3. The language error updates the representations and their connections.
4. Across varied encounters, useful relationships can become stronger.
Consequently, the same load representation can contribute strongly to “heavy” and weakly to “red.” That can emerge through learned word-specific similarities; it does not require a hand-written rule saying “weight words use load.”

But the learner can also:
- Rely on the easier visual pathway and neglect body.
- Learn an accidental correlation.
- Combine signals poorly.
- Fail to extract a relationship that exists in the data.
There are therefore three separate claims:
Claim	What establishes it
Every stream is available	Input and masking checks
Every stream can receive learning signals	Gradient and implementation checks
The learner uses a stream meaningfully	Held-out behavior and appropriate disruption controls


Attention weights alone would not establish the last claim.

And initially, the model does not “know” which bodily observation matters. The language objective provides the pressure to discover useful relationships. Keeping the direct body–language route makes that discovery possible without requiring agreement with vision.

3. Yes—the controlled diagnostic could be a miniature acquisition-curve experiment Annotation 3

That is a good way to frame it. It would add a challenge that heavy/light largely avoided:
Each individual encounter is ambiguous, but evidence across encounters can reveal the word’s meaning.

It is not necessarily harder in every respect than heavy/light. It would be more demanding about reference, while remaining simpler than the full task in vocabulary, distractors, and language.

For example:
Encounter	Heard	Possible referents
1	“dax”	Object A and a spoon
2	“dax”	Another A-like object and a ball
3	“dax”	Another A-like object and a book


The learner receives observations, not the answer “A.” Bodily experience could provide additional evidence about the objects or interactions.

Rather than report only one final accuracy, we could measure performance after a few increasing amounts of independent experience. That would give a small, controlled version of the acquisition plot.

I would bound it as follows:
- A small vocabulary and a small number of plausible referents.
- Existing cached video and body data where suitable.
- Changing distractors across encounters.
- Held-out source clips for evaluation.
- The same learner and language objective used by the main comparison.
- A fixed budget, with no expanding sweep if the result is disappointing.
The body advantage must come from informative sensory observations. We should not create it by supplying hidden referent IDs or making “held object = named object” a guaranteed rule.

This could proceed alongside the broader comparison once the shared implementation is verified. It should reuse that implementation, so we are not maintaining a separate diagnostic architecture. Its role would be to help interpret the main curves:
- Can the learner acquire associations under controlled ambiguity?
- Does body improve the rate or extent of that learning?
- Does the advantage survive unfamiliar clips?
- Does disrupting body correspondence remove it?
A small dataset does not by itself guarantee a quick run, so the runtime budget still needs to be set before launch. I have not started training or changed files here.

---

Great. So now we have the main design and a smaller experiment to do. We will do them both parallely. Can you in simple terms but in detail just show the design of both experiments. Basically exactly how you did in these responses in the way you formatted them so you talk about the conceptual idea of the experiment, the detaisl in simple terms and like an expected output and stuff. And then write them both to two different .md files and we'll use that as reference for starting them parallely.

*The following three user follow-ups are retained from the conversation context; their timestamps were not supplied:*

note: not what different results would mean. just what we want the output to be like.

like how i showed you the plot for the langauge acquisition curve, and how the results from the heavy/light etc.

yep. it doesn't have to be super long or anything. pretty reasonable sized and normal way of talking about it thats succint and readable, basically how you have been explaining and responding in this entier convos

4:23 PM

Worked for 6m 15s

Saved both designs in readable documents, with conceptual explanations, training details, and the outputs we want.

**1. [Main language-acquisition experiment](/Users/rishisim/Documents/research/nursery/docs/acquisition_curves.md)**

Give learners progressively more independent experience, comparing:

- Video only.
- Video + body with additive scoring.
- Video + body with direct language pathways **and** learned interaction.

All receive the same video and weak speech. Body-enabled learners receive all five sensory streams.

**Intended output:** your acquisition plot—training experience along the horizontal axis, held-out word understanding vertically, with three measured curves. The document includes a clearly labeled sketch of the desired body advantage.

**2. [Small cross-situational experiment](/Users/rishisim/Documents/research/nursery/docs/cross_situational_learning.md)**

Use four unfamiliar words in situations containing multiple possible referents. Distractors change across encounters, allowing the learner to accumulate evidence about each word.

Compare the same three learners across smaller amounts of experience.

**Intended output:** a miniature acquisition curve, a few example encounters, and a heavy/light-style table comparing video, additive body, joint body, and shuffled-body controls. Future measurements are left blank.

Both studies share one verified implementation but use separate trained models and run directories, allowing parallel execution without one changing the other.

I linked both from [PLAN.md](/Users/rishisim/Documents/research/nursery/PLAN.md), checked links and the diff, and pushed commit `2c14cfa1`. The working tree is clean. No training started.

&#x20; Edited 3 files+230-9View changes

UndoView changes

PLAN.mdPLAN.md+14-9

docs/acquisition_curves.mddocs/acquisition_curves.md+115-0

docs/cross_situational_learning.mddocs/cross_situational_learning.md+101-0

4:29 PM

okay both look great. one note though, i feel like the small cross istuational experiment was just slightly overcomplicated. i think for the heavy/light we just did the video only and the additiive. lets keep it that way and the output shouldnt even be the curves we'll just do how we did with the heavy/light where its just a table at the end. between video and additive. (any reason we need shuffled? we didn't have it in heavy/light).

just need that clarification

5:07 PM

Agreed—the small experiment can be much simpler:

- **Two learners:** video-only and additive video + body.
- **One fixed training-data amount**, rather than acquisition curves.
- **One final accuracy table**, like heavy/light.
- Same ambiguous encounters for both, followed by testing on unfamiliar clips.

The main experiment keeps the architectural comparison and acquisition curves.

One correction: **heavy/light did include shuffled load**:

| Condition                              | Accuracy |
| -------------------------------------- | -------- |
| Video-only                             | 37.88%   |
| Video + load                           | 92.42%   |
| Same trained model, shuffled test load | 46.21%   |

Shuffling wasn’t another training run. We took the already-trained body model and gave situations the wrong load histories during evaluation.

Its purpose was to check: **did the improvement depend on the correctly associated bodily information?** The drop supported that explanation.

My recommendation: keep **only the two trained learners**, as you suggest, and retain shuffled-body evaluation as a small supporting check. The headline table can contain just video versus additive; the shuffled result can be one sentence underneath. No extra architecture or curves needed.

5:08 PM

okay update the plan for that accordingly

5:08 PM

Worked for 42s

Updated [the small-study design](/Users/rishisim/Documents/research/nursery/docs/cross_situational_learning.md) and [PLAN.md](/Users/rishisim/Documents/research/nursery/PLAN.md):

- Video-only versus additive video + body.
- One fixed training-data amount.
- Two-row results table; no curves or joint learner.
- Shuffled-body evaluation reported underneath as a supporting check, without additional training.

The main study is unchanged. Links and diff checked; pushed as `20f1cc03`. Working tree clean.
