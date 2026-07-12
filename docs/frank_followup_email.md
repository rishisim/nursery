# Follow-up email to Professor Frank

**Subject:** Follow-up: preparing a BabyView-matched embodied-language experiment

Hi Professor Frank,

Thank you again for your suggestion. It helped me clarify the broader question I want to study:

**Do action, proprioception, and touch help models learn grounded language when vision and language are only weakly aligned, as they are in natural infant experience?**

While my Databrary authorization is pending, I have prepared the experimental infrastructure needed to test that question:

- A configurable simulator that produces video, timed language, embodied cues, and exact causal ground truth.
- Controlled synchronized, shuffled, shifted, and absent cue conditions, with held-out tests and leakage checks.
- A small learner that can use an embodied cue during training but must perform the language-grounding test without that cue.
- The public Machine-DevBench evaluation path, validated by reproducing its published CLIP-L overall result within 0.6 percentage points.

Once I receive BabyView access, I plan to:

1. Measure its vocabulary, utterance timing, visual-language alignment, referential ambiguity, visibility, and clutter.
2. Use those measurements to calibrate and validate a distribution-matched simulation.
3. Run the same learner and evaluations on the natural and matched simulated data.
4. Measure whether synchronized action cues improve language grounding over shuffled, shifted, and absent controls, then extend the same design to proprioception and touch.

Does this capture the apples-to-apples comparison you had in mind, and which BabyView properties would you prioritize first? Everything completed so far is preparation and evaluation infrastructure built while I wait for Databrary access; the BabyView-dependent analysis has not begun.

Best,  
Rishi Simhadri  
UT Dallas
