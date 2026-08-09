# ChildLens v1.3.1 pivot decision memo

## Recommendation

For a strong Michael Frank-facing naturalistic linguistic-acquisition prototype, choose **C: a separately initialized BabyView study**, subject to its own access, governance, and language-qualified annotation plan. Do not pool corpora or transfer ChildLens vocabulary, tokenizers, checkpoints, weights, empirical aggregates, pseudo-labels, or calibration values. This recommendation prioritizes a defensible lexical claim over sunk-cost reuse.

While that study is being initialized, **D: a fully synthetic causal prototype** is a defensible engineering demonstration if it uses synthetic language and simulator-oracle evaluation and makes no ChildLens-language claim. It is not a naturalistic corpus result and must not silently incorporate ChildLens empirical language information if the one-corpus boundary remains active.

## Options

| Option | What it can support | Required action | Scientific limitation |
|---|---|---|---|
| A. Qualified German annotator | Restores the bounded ChildLens lexical-feasibility audit | Obtain written authorization for the person, recruit a source-German-qualified annotator, initialize a clean corrected 15-minute blind pass | Still a small feasibility audit; no inter-human reliability under the one-person remedy |
| B. Descriptive ChildLens study | Visual/activity/speech-presence calibration only | Version a separate nonlexical protocol | Cannot validate transcript, words, semantics, child/adult role, or lexical grounding |
| C. Separate BabyView study | Best route to the original naturalistic linguistic-acquisition prototype | Initialize a new governance/protocol task from zero | Must remain completely separate from ChildLens; no pooling or artifact transfer |
| D. Fully synthetic causal prototype | Fastest clean test of the causal mechanism | Freeze simulator language, oracle labels, matched arms, and leakage controls | Makes no ChildLens-language or naturalistic acquisition claim |

## Exact next task

After the user's substantive choice, the recommended next task is: **initialize a BabyView-only governance and lexical-measurement feasibility protocol as a new corpus study with zero ChildLens empirical ancestry**. Do not begin acquisition or learner training in this ChildLens task. If the user instead chooses A, the next task is limited to authorization receipt plus a clean v1.3.1 German-qualified 15-minute audit initialization.
