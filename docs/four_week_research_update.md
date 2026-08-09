---
title: "Four Week Research Update"
author: "Rishi Simhadri"
date: "August 9, 2026"
geometry: margin=0.7in
fontsize: 11pt
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - |
    ```{=latex}
    \usepackage{graphicx}
    \usepackage{float}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.55em}
    ```
---

I wanted to send you a consolidated update on what I have done over the past four weeks and ask for your advice about the next phase. Below are the referenced figures, followed by a detailed explanation of what I did.

## FIGURES

```{=latex}
\begingroup
\setlength{\parskip}{0pt}
\noindent\includegraphics[width=\linewidth]{docs/assets/four_week_research_update/figure-1-two-routes.png}\par\nointerlineskip
\noindent\includegraphics[width=\linewidth]{docs/assets/four_week_research_update/figure-2-frame-comparison.png}\par\nointerlineskip
\noindent\includegraphics[width=\linewidth]{docs/assets/four_week_research_update/figure-3-evaluation-metrics.png}\par\nointerlineskip
\noindent\includegraphics[width=\linewidth]{docs/assets/four_week_research_update/figure-4-cost-projection.png}\par\nointerlineskip
\noindent\includegraphics[width=\linewidth]{docs/assets/four_week_research_update/figure-5-model-selection.png}\par
\endgroup
```

## EXPLANATION

In your previous email, you suggested matching aspects of the natural data distribution measured in BabyView, generating the additional side-information streams, and then applying the same models to the natural and simulated data apples-to-apples. I used that as the central target for this work.

## What I have done so far

I started from the [EgoBabyVLM leaderboard](https://facebookresearch.github.io/egobabyvlm/leaderboard.html), where the CLIP+ baseline trained on BabyView has an overall aggregate score of **53.6** on Machine-DevBench. Because infants receive information beyond vision and language—including action, proprioception, touch, contact, and vestibular cues—I wanted to test whether these additional cues could help fill part of that gap.

I pursued two routes in parallel:

1. **Embodied, physics-based simulation.** I built a simulator pipeline that generates egocentric video (sample frames attached) together with synchronized action, proprioception, contact, touch, IMU/vestibular signals, depth, segmentation, object state, and language.

   As I showed in the previous email, at the 2D environment level, providing aligned physical cues during training helped the model connect words with actions more effectively than when those cues were scrambled.

   I then tried to move toward the apples-to-apples comparison you suggested by matching measurable properties of natural egocentric data, including speech timing, motion, visual persistence, and scene changes. The physics and side-information streams worked, but the visual matching did not: the 24-episode batch remained too static and simulation-like, while attempts to add more natural movement created other visual mismatches.

   Because of this, I was not able to establish a clean apples-to-apples embodied comparison. Any measured difference could still be caused by the visual domain gap rather than by the additional embodied cues. Reaching the point where that comparison would be interpretable appears to require substantially more simulator and rendering work.

2. **Text-to-video synthetic data.** I therefore tested the same general comparison using current text-to-video systems, including Hailuo through an API and LTX-2.3 through both an API and local H100 inference (sample frames attached).

   For this comparison, I used one public natural egocentric source clip from the RekaAI/RekaDaily-10k-raw dataset, fixed the description used for generation, selected activity-matched frames for the visual comparison, and applied the same frozen CLIP-based probes to the real and synthetic videos. This route was much more successful visually: the generated sequence preserved the egocentric activity, hands, shirt, and other principal referents, and the real and synthetic clips produced similar noun, adjective, and action results on the initial probe.

   The attachment now includes a direct comparison between the real source frame and an activity-matched frame from each of the three generated videos. It also includes the CLIP-based results and cost curves. Local LTX-2.3 is the cheapest synthetic option shown in the attachment, while offering controllability, targeted fidelity, promising realism, and potential scalability.

   The Hailuo and API-hosted LTX outputs were generated from the public RekaDaily clip and are included only as demonstrations. Because BabyView is controlled-access, I would not send its data to third-party APIs; for BabyView, I would use only LTX-2.3 run locally on Juno's H100.

The central result is therefore that I tried to implement the apples-to-apples embodied comparison, but visual realism remained a significant blocker. Text-to-video does not yet provide measured embodied streams, so it cannot directly answer the original cue-lift question, but it provides a much stronger path toward visually comparable and controllable synthetic training data.

## Proposed next steps

I see two separate experiments for the text-to-video route:

1. **Test the dataset-scale hypothesis.** Using local LTX-2.3, generate increasing amounts of BabyView-like synthetic video and compare a BabyView-only baseline against the same training pipeline augmented with increasing quantities of synthetic data. This would test whether limited training hours are an important bottleneck.
2. **Test targeted gap-filling.** Evaluate BabyView for gaps in activities or vocabulary, generate synthetic video locally with LTX-2.3 specifically for those gaps, and test whether the targeted additions improve performance.

Embodied cues could then be revisited later. If better and more abundant synthetic visual-language data improves performance but still leaves a meaningful gap, that would provide a more concrete reason to investigate whether additional embodied cues can explain the remaining difference.

Considering the embodied video path is blocked, I'm thinking of taking the text-to-video route and implementing the proposed next steps above. Let me know your thoughts on whether this direction would be appropriate.
