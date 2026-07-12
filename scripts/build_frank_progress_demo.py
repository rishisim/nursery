from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "grounding_provisional" / "examples.jsonl"
OUT = ROOT / "output" / "demo" / "frank_progress_demo.html"


def _load_example() -> tuple[dict, dict[str, list[dict]]]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    base_id = rows[0]["base_episode_id"]
    selected = {
        row["motor_condition"]: row
        for row in rows
        if row["base_episode_id"] == base_id
        and row["alignment_condition"] == "weak"
    }
    return selected["synchronized"], {
        name: row["model_inputs"]["motor"] for name, row in selected.items()
    }


def build() -> Path:
    episode, motors = _load_example()
    frame_paths = [
        "../../data/grounding_provisional/" + path
        for path in episode["model_inputs"]["frame_paths"]
    ]
    utterance = next(
        item["text"]
        for item in episode["model_inputs"]["utterances"]
        if item["onset_frame"] >= 0
    )
    target = episode["evaluation_targets"]
    motor_json = {
        name: [[sample["x"], sample["y"], sample["available"]] for sample in values]
        for name, values in motors.items()
    }
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From synthetic nursery to a BabyView-matched experiment</title>
<style>
:root {{
  --ink:#12212c; --muted:#647681; --paper:#f7f8f5; --panel:#ffffff;
  --teal:#0f766e; --teal-soft:#dff3ef; --coral:#e8674f; --line:#d8e0de;
  --yellow:#f3c85b; --shadow:0 22px 60px rgba(18,33,44,.12);
}}
* {{ box-sizing:border-box; }}
html, body {{ width:100%; height:100%; margin:0; overflow:hidden; background:var(--paper); color:var(--ink); }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
.deck {{ height:100vh; position:relative; }}
.slide {{ position:absolute; inset:0; padding:54px 74px 86px; opacity:0; transform:translateX(35px); pointer-events:none; transition:.42s ease; }}
.slide.active {{ opacity:1; transform:none; pointer-events:auto; }}
.slide.exit {{ opacity:0; transform:translateX(-35px); }}
.topline {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:38px; font-size:15px; font-weight:700; letter-spacing:.04em; color:var(--muted); }}
.brand {{ color:var(--teal); }}
h1 {{ font-family:Georgia,"Times New Roman",serif; font-size:60px; line-height:1.04; letter-spacing:-.035em; max-width:1050px; margin:0 0 24px; font-weight:500; }}
h2 {{ font-family:Georgia,"Times New Roman",serif; font-size:46px; line-height:1.08; letter-spacing:-.025em; margin:0 0 24px; font-weight:500; }}
.lead {{ font-size:25px; line-height:1.5; color:#3c515c; max-width:1000px; }}
.question {{ margin-top:72px; max-width:1040px; }}
.question strong {{ color:var(--teal); font-weight:500; }}
.flowline {{ display:flex; gap:10px; align-items:center; margin-top:54px; font-weight:700; font-size:17px; }}
.flowline span {{ padding:14px 18px; background:var(--panel); border:1px solid var(--line); border-radius:10px; }}
.arrow {{ color:var(--coral); font-size:25px; }}
.episode-grid {{ display:grid; grid-template-columns:minmax(420px,1.05fr) minmax(440px,.95fr); gap:40px; align-items:center; }}
.scene {{ background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:18px; box-shadow:var(--shadow); }}
.scene img {{ display:block; width:100%; aspect-ratio:16/10; object-fit:contain; border-radius:13px; background:#f5efe4; }}
.utterance {{ margin-top:14px; font-size:18px; padding:14px 16px; border-left:4px solid var(--coral); background:#fff8f5; border-radius:7px; }}
.channels {{ display:grid; gap:12px; }}
.channel {{ display:flex; align-items:center; justify-content:space-between; padding:18px 20px; background:var(--panel); border-bottom:1px solid var(--line); font-size:19px; }}
.channel:first-child {{ border-radius:14px 14px 0 0; }} .channel:last-child {{ border-radius:0 0 14px 14px; border-bottom:0; }}
.channel b {{ font-size:20px; }} .channel small {{ color:var(--muted); font-size:15px; }}
.dot {{ width:11px; height:11px; border-radius:50%; background:var(--teal); margin-right:12px; display:inline-block; }}
.control-grid {{ display:grid; grid-template-columns:380px 1fr; gap:36px; align-items:start; }}
.mini-scene img {{ width:100%; aspect-ratio:16/10; object-fit:contain; background:#f5efe4; border-radius:16px; border:1px solid var(--line); }}
.fixed-note {{ font-size:17px; line-height:1.45; color:var(--muted); margin-top:12px; }}
.tabs {{ display:flex; gap:9px; margin-bottom:18px; }}
.cue {{ appearance:none; border:1px solid var(--line); background:white; color:var(--ink); border-radius:9px; padding:12px 14px; font:700 15px inherit; cursor:pointer; transition:.18s ease; }}
.cue:hover {{ transform:translateY(-2px); border-color:var(--teal); }}
.cue.active {{ background:var(--teal); color:white; border-color:var(--teal); }}
.plot {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:var(--shadow); }}
.plot-head {{ display:flex; justify-content:space-between; align-items:baseline; }}
.plot-head b {{ font-size:22px; }} .plot-head span {{ color:var(--muted); }}
svg {{ display:block; width:100%; height:260px; margin-top:8px; }}
.axis {{ stroke:#cbd6d3; stroke-width:1; }} .trajectory {{ fill:none; stroke:var(--coral); stroke-width:5; stroke-linecap:round; stroke-linejoin:round; }}
.cursor-dot {{ fill:var(--yellow); stroke:var(--ink); stroke-width:2; }}
.control-caption {{ margin-top:16px; font-size:22px; font-weight:700; color:var(--teal); }}
.train-test {{ display:grid; grid-template-columns:1fr 110px 1fr; align-items:center; gap:18px; margin-top:70px; }}
.stage {{ min-height:290px; background:white; border:1px solid var(--line); border-radius:20px; padding:34px; box-shadow:var(--shadow); }}
.stage h3 {{ font-size:16px; text-transform:uppercase; letter-spacing:.12em; color:var(--teal); margin:0 0 25px; }}
.stage .big {{ font-family:Georgia,serif; font-size:35px; line-height:1.25; }}
.stage .rule {{ margin-top:28px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:18px; }}
.big-arrow {{ text-align:center; color:var(--coral); font-size:58px; }}
.evidence {{ display:grid; grid-template-columns:repeat(3,1fr); gap:22px; margin-top:70px; }}
.metric {{ background:white; border-top:5px solid var(--teal); padding:28px; min-height:220px; box-shadow:var(--shadow); }}
.metric .number {{ font-family:Georgia,serif; font-size:54px; margin:18px 0 12px; }}
.metric p {{ color:var(--muted); font-size:18px; line-height:1.45; }}
.next-flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:34px; margin-top:70px; position:relative; }}
.next-flow:before {{ content:""; position:absolute; left:11%; right:11%; top:48px; height:3px; background:var(--line); z-index:0; }}
.next-step {{ position:relative; z-index:1; text-align:center; }}
.step-num {{ width:94px; height:94px; margin:0 auto 20px; border-radius:50%; display:grid; place-items:center; background:var(--teal); color:white; font-family:Georgia,serif; font-size:38px; box-shadow:0 0 0 9px var(--paper); }}
.next-step b {{ font-size:21px; }} .next-step p {{ color:var(--muted); line-height:1.45; font-size:17px; }}
.pending {{ margin-top:64px; display:inline-flex; gap:12px; align-items:center; background:#fff7df; border:1px solid #ead28d; border-radius:12px; padding:14px 18px; font-weight:700; }}
.pending .dot {{ background:#d89d00; margin:0; }}
.controls {{ position:absolute; left:74px; right:74px; bottom:24px; display:flex; align-items:center; justify-content:space-between; z-index:10; }}
.nav {{ display:flex; gap:10px; }}
.nav button {{ appearance:none; border:1px solid var(--line); background:white; border-radius:9px; padding:11px 18px; font:700 15px inherit; cursor:pointer; }}
.nav button.primary {{ background:var(--ink); color:white; border-color:var(--ink); }}
.nav button:disabled {{ opacity:.35; cursor:default; }}
.progress {{ display:flex; align-items:center; gap:11px; color:var(--muted); font-size:14px; }}
.bar {{ width:220px; height:6px; background:#dfe5e3; border-radius:99px; overflow:hidden; }} .fill {{ height:100%; background:var(--coral); transition:.35s ease; }}
</style>
</head>
<body>
<main class="deck">
  <section class="slide active">
    <div class="topline"><span class="brand">Nursery progress demo</span><span>1 / 6</span></div>
    <div class="question">
      <h1>Do embodied cues improve <strong>grounded language learning</strong> from weak visual–linguistic alignment?</h1>
      <p class="lead">The long-term question covers action, proprioception, and touch. The first controlled study isolates action.</p>
      <div class="flowline"><span>BabyView measurements</span><b class="arrow">→</b><span>Matched simulation</span><b class="arrow">→</b><span>Cue controls</span><b class="arrow">→</b><span>Language grounding</span></div>
    </div>
  </section>

  <section class="slide">
    <div class="topline"><span class="brand">One instrumented experience</span><span>2 / 6</span></div>
    <h2>Naturalistic inputs, plus signals a head camera cannot record</h2>
    <div class="episode-grid">
      <div class="scene"><img id="episodeFrame" src="{frame_paths[0]}" alt="Simulated nursery episode"><div class="utterance">“{html.escape(utterance)}”</div></div>
      <div class="channels">
        <div class="channel"><span><i class="dot"></i><b>Video + timed language</b></span><small>learner input</small></div>
        <div class="channel"><span><i class="dot"></i><b>Action trajectory</b></span><small>current study</small></div>
        <div class="channel"><span><i class="dot"></i><b>Proprioception + touch</b></span><small>next cue families</small></div>
        <div class="channel"><span><i class="dot"></i><b>Exact causal labels</b></span><small>audit only</small></div>
      </div>
    </div>
  </section>

  <section class="slide">
    <div class="topline"><span class="brand">The core causal comparison</span><span>3 / 6</span></div>
    <h2>Keep video and language fixed. Change only the cue.</h2>
    <div class="control-grid">
      <div class="mini-scene"><img id="controlFrame" src="{frame_paths[5]}" alt="Fixed visual episode"><p class="fixed-note"><b>Fixed across all four conditions</b><br>Same scene · same utterance · same split · same learner</p></div>
      <div>
        <div class="tabs">
          <button class="cue active" data-cue="synchronized">Synchronized</button>
          <button class="cue" data-cue="shuffled">Shuffled</button>
          <button class="cue" data-cue="time_shifted">Time-shifted</button>
          <button class="cue" data-cue="null">Absent</button>
        </div>
        <div class="plot"><div class="plot-head"><b>Hand trajectory over the event</b><span id="cueStatus">aligned with this episode</span></div><svg viewBox="0 0 650 260" aria-label="Cue trajectory"><line class="axis" x1="30" y1="220" x2="625" y2="220"></line><line class="axis" x1="30" y1="25" x2="30" y2="220"></line><polyline id="trajectory" class="trajectory" points=""></polyline><circle id="cursorDot" class="cursor-dot" cx="30" cy="220" r="8"></circle></svg><div id="cueCaption" class="control-caption">Useful temporal relationship preserved</div></div>
      </div>
    </div>
  </section>

  <section class="slide">
    <div class="topline"><span class="brand">The learning test</span><span>4 / 6</span></div>
    <h2>The embodied cue can teach—but it cannot answer the test.</h2>
    <div class="train-test">
      <div class="stage"><h3>Training</h3><div class="big">Video + language<br><b>+ action cue</b></div><div class="rule">The model can use synchronized experience to shape its representations.</div></div>
      <div class="big-arrow">→</div>
      <div class="stage"><h3>Evaluation</h3><div class="big">Video + language<br><b>cue removed</b></div><div class="rule">Held-out object–action combinations test grounded language learning.</div></div>
    </div>
  </section>

  <section class="slide">
    <div class="topline"><span class="brand">Harness validation</span><span>5 / 6</span></div>
    <h2>The experimental and external-evaluation paths run end to end.</h2>
    <div class="evidence">
      <div class="metric"><span>Repository checks</span><div class="number">24</div><p>Tests passed, including cue controls, split integrity, leakage, and motor-free evaluation.</p></div>
      <div class="metric"><span>Machine-DevBench</span><div class="number">3,721</div><p>Every public trial runs through the same feature-extractor boundary.</p></div>
      <div class="metric"><span>CLIP-L validation</span><div class="number">78.28</div><p>Published overall: 78.8. Local reproduction is within 0.6 points.</p></div>
    </div>
  </section>

  <section class="slide">
    <div class="topline"><span class="brand">What BabyView access unlocks</span><span>6 / 6</span></div>
    <h2>Preparation is complete. Calibration comes next.</h2>
    <div class="next-flow">
      <div class="next-step"><div class="step-num">1</div><b>Measure BabyView</b><p>Vocabulary, timing, alignment, ambiguity, visibility, and clutter.</p></div>
      <div class="next-step"><div class="step-num">2</div><b>Calibrate simulation</b><p>Replace provisional settings and validate the distributional match.</p></div>
      <div class="next-step"><div class="step-num">3</div><b>Run cue controls</b><p>Compare synchronized, shuffled, shifted, and absent experience.</p></div>
      <div class="next-step"><div class="step-num">4</div><b>Measure language lift</b><p>Action first; then proprioception and touch using the same design.</p></div>
    </div>
    <div class="pending"><i class="dot"></i>Databrary authorization pending · BabyView-dependent analysis has not begun</div>
  </section>

  <div class="controls">
    <div class="progress"><span id="counter">1 of 6</span><div class="bar"><div id="fill" class="fill" style="width:16.67%"></div></div></div>
    <div class="nav"><button id="prev" disabled>Back</button><button id="next" class="primary">Next →</button></div>
  </div>
</main>
<script>
const frames={json.dumps(frame_paths)};
const motors={json.dumps(motor_json)};
let frameIndex=0, slideIndex=0;
setInterval(()=>{{ frameIndex=(frameIndex+1)%frames.length; document.getElementById('episodeFrame').src=frames[frameIndex]; document.getElementById('controlFrame').src=frames[frameIndex]; }},420);
const slides=[...document.querySelectorAll('.slide')];
function showSlide(next){{
  slides[slideIndex].classList.remove('active'); slides[slideIndex].classList.add('exit');
  setTimeout(()=>slides[slideIndex]?.classList.remove('exit'),450);
  slideIndex=Math.max(0,Math.min(slides.length-1,next)); slides[slideIndex].classList.add('active');
  document.getElementById('counter').textContent=`${{slideIndex+1}} of ${{slides.length}}`;
  document.getElementById('fill').style.width=`${{(slideIndex+1)/slides.length*100}}%`;
  document.getElementById('prev').disabled=slideIndex===0; document.getElementById('next').disabled=slideIndex===slides.length-1;
}}
document.getElementById('next').onclick=()=>showSlide(slideIndex+1);
document.getElementById('prev').onclick=()=>showSlide(slideIndex-1);
document.addEventListener('keydown',e=>{{ if(e.key==='ArrowRight'||e.key===' ') showSlide(slideIndex+1); if(e.key==='ArrowLeft') showSlide(slideIndex-1); }});
const cueText={{ synchronized:['aligned with this episode','Useful temporal relationship preserved'], shuffled:['borrowed from another episode','Same amount of cue data, wrong relationship'], time_shifted:['correct trajectory, wrong time','Temporal alignment disrupted'], null:['all cue values set to zero','No embodied cue available'] }};
function renderCue(name){{
  document.querySelectorAll('.cue').forEach(b=>b.classList.toggle('active',b.dataset.cue===name));
  const values=motors[name]; const valid=values.filter(v=>v[2]);
  const pts=values.map((v,i)=>{{ const x=35+i/(values.length-1)*580; const y=v[2] ? 225-(v[1]-70)/150*190 : 220; return [x,Math.max(30,Math.min(220,y))]; }});
  document.getElementById('trajectory').setAttribute('points',pts.map(p=>p.join(',')).join(' '));
  const end=pts[Math.max(0,valid.length-1)]; document.getElementById('cursorDot').setAttribute('cx',end[0]); document.getElementById('cursorDot').setAttribute('cy',end[1]);
  document.getElementById('cueStatus').textContent=cueText[name][0]; document.getElementById('cueCaption').textContent=cueText[name][1];
}}
document.querySelectorAll('.cue').forEach(button=>button.onclick=()=>renderCue(button.dataset.cue)); renderCue('synchronized');
</script>
</body>
</html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    return OUT


if __name__ == "__main__":
    print(build())
