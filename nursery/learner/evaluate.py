"""Body-side tests: body-present 4AFC on the held-out subject and the body probe."""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .losses import soft_pool


def body_present_context(manifest, frames, lut, window, body_half, subject=None):
    """Renderer truth shared by all controls; full-frame visibility does not
    establish recognizability after the image encoder crop."""
    if not {"video_id", "frame_idx", "referent"} <= set(manifest.columns):
        raise ValueError("Body-present manifest requires video_id, frame_idx and referent")
    if window < 0 or body_half < 0:
        raise ValueError("Body-present windows must be nonnegative")
    required = {"video_id", "frame_idx", "objects", "clip", "held_category", "subject"}
    if not required <= set(frames.columns):
        raise ValueError(f"Body-present scoring requires renderer metadata: {sorted(required - set(frames.columns))}")
    if frames.duplicated(["video_id", "frame_idx"]).any():
        raise ValueError("Duplicate renderer frame metadata")
    table = {(r.video_id, int(r.frame_idx)): r for r in frames.itertuples()}
    visible, accessible, held, subjects = [], [], [], []
    for r in manifest.itertuples():
        word = r.referent
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Body-present target referent must be a nonempty category")
        key = (r.video_id, int(r.frame_idx))
        if key not in table or f"{key[0]}|{key[1]}" not in lut:
            raise ValueError(f"Missing aligned renderer/cache frame: {key}")
        centre = table[key]
        if pd.isna(centre.clip) or pd.isna(centre.subject):
            raise ValueError(f"Missing aligned renderer clip/subject: {key}")
        def seen(row):
            return {o["category"] for o in json.loads(row.objects) if o["visible_px"] > 0}
        if word not in seen(centre):
            raise ValueError(f"Body-present positive target absent at aligned frame: {key}, {word}")
        words = set()
        for f in range(key[1] - window, key[1] + window + 1):
            if f"{key[0]}|{f}" in lut:
                if (key[0], f) not in table:
                    raise ValueError(f"Missing renderer metadata for supplied visual frame: {(key[0], f)}")
                words.update(seen(table[key[0], f]))
        body_words = {centre.held_category} if pd.notna(centre.held_category) else set()
        for direction in (-1, 1):
            for delta in range(1, body_half + 1):
                row = table.get((key[0], key[1] + direction * delta))
                if row is None or row.clip != centre.clip or pd.isna(row.clip):
                    break
                if pd.notna(row.held_category):
                    body_words.add(row.held_category)
        visible.append(words)
        accessible.append(body_words)
        held.append(centre.held_category == word)
        subjects.append(centre.subject)
    if subject is not None and not np.array_equal(np.asarray(subject), np.asarray(subjects)):
        raise ValueError("Body-present subjects disagree with renderer metadata")
    ref, videos = manifest.referent.to_numpy(), manifest.video_id.to_numpy()
    pools_by_query = {}
    pools = []
    for i, word in enumerate(ref):
        key = (subjects[i], videos[i], word)
        if key not in pools_by_query:
            base = np.flatnonzero((np.asarray(subjects) == subjects[i]) & (videos != videos[i]) & (ref != word))
            pools_by_query[key] = np.array([j for j in base if word not in visible[j] and word not in accessible[j]], dtype=int)
        pools.append(pools_by_query[key])
    return {"pools": pools, "held": held, "subjects": subjects, "identity": list(zip(videos.tolist(), manifest.frame_idx.astype(int).tolist(), ref.tolist())),
            "definition": {"version": 2, "visual_half_window": int(window), "body_half_window": int(body_half),
                           "foil_rule": "different_session_same_subject_target_absent_from_cached_visual_and_clip_local_body_windows",
                           "visibility": "renderer_full_frame_visible_px_gt_0", "correct_position": "seeded_uniform_permutation", "partition": "provided_manifest_only", "n_rows": len(manifest),
                           "n_eligible_trials": sum(len(p) >= 3 for p in pools),
                           "n_insufficient_foils": sum(len(p) < 3 for p in pools)}}


def _body_trials(manifest, context, seed, subject):
    if context is None:
        raise ValueError("Body-present scoring requires validated renderer context")
    identity = list(zip(manifest.video_id.tolist(), manifest.frame_idx.astype(int).tolist(), manifest.referent.tolist()))
    if identity != context["identity"]:
        raise ValueError("Body-present context does not match manifest rows")
    if subject is not None and not np.array_equal(np.asarray(subject), np.asarray(context["subjects"])):
        raise ValueError("Body-present context subjects disagree")
    rng = np.random.default_rng(seed)
    sampled = []
    for i, pool in enumerate(context["pools"]):
        if len(pool) >= 3:
            candidates = rng.permutation(np.concatenate([[i], rng.choice(pool, 3, replace=False)]))
            sampled.append((i, candidates, int(np.flatnonzero(candidates == i)[0])))
    return sampled


@torch.no_grad()
def body_present_4afc(model, upstream, ds, data, manifest, device, seed=0, batch=256, subject=None, trials=False, context=None):
    """4AFC with the word given and four candidate situations (image window + body), held-out subject only.

    `manifest` holds the held-out subject's named utterances with a visible referent; `ds` is the upstream
    FramePairs over those rows and `data` a BodyData over the same rows (None without a body). Each trial: the true situation plus three foils from other sessions with other
    referents. Scores per candidate: picture (hard max over the window's regions), body (fused-projection cosine to
    the word) and their sum (late fusion), plus `picture_soft` / `fused_soft`: the same with the training pooling
    (soft-MIL, uniform region prior, tau_p), which is the score the model was trained on and the one to read.
    Foils use their own body, so the body must match the situation. `subject` [rows] restricts foils to the target's
    subject; `trials` adds `trial_rows` (manifest row of each scored trial) and per-trial 0/1 `trials` for the soft scores.
    """
    sampled = _body_trials(manifest, context, seed, subject)
    if len(ds) != len(manifest):
        raise ValueError("Body-present dataset and manifest lengths disagree")
    model.eval()
    tokens, counts, regions, latents = [], [], [], []
    generator = torch.Generator().manual_seed(seed)
    for start in range(0, len(ds), batch):
        rows = list(range(start, min(start + batch, len(ds))))
        _, v, t, n = upstream.collate([ds[i] for i in rows])
        tokens.append(model.enc_text(t.to(device), n.to(device)))
        regions.append(model.enc_regions(v.to(device)).half())
        if model.encoder is not None:
            latents.append(model.latent_of(data.batch(np.array(rows), generator, training=False)))
    T, R = torch.cat(tokens), torch.cat(regions)
    Z = torch.cat(latents) if latents else None
    held_all = np.asarray(context["held"], dtype=bool)
    held, rows_used = [], []
    correct = {"picture": [], "body": [], "fused": [], "picture_soft": [], "fused_soft": []}
    tau = getattr(model, "tau_p", None)
    for i, cand, answer in sampled:
        held.append(held_all[i])
        rows_used.append(i)
        pic = (R[cand].float() @ T[i]).max(1).values
        scores = {"picture": pic}
        if tau is not None:
            soft = soft_pool((R[cand].float() @ T[i]).unsqueeze(-1), torch.full((4, R.shape[1]), -np.log(R.shape[1]), device=R.device), tau).squeeze(1)
            scores["picture_soft"] = soft
        if Z is not None:
            body = model.body_similarity(Z[cand], T[i:i + 1]).squeeze(1)
            scores["body"], scores["fused"] = body, pic + model.beta * body
            if tau is not None:
                scores["fused_soft"] = soft + model.beta * body
        for k, s in scores.items():
            correct[k].append(float(s.argmax() == answer))
    held = np.array(held, bool)
    out = {"n_trials": len(next(iter(correct.values()))), "chance": 0.25}
    for k, c in correct.items():
        if c:
            c = np.array(c)
            out[k] = {"acc": round(100 * float(c.mean()), 3), "acc_held": round(100 * float(c[held].mean()), 3) if held.any() else None,
                      "acc_not_held": round(100 * float(c[~held].mean()), 3) if (~held).any() else None}
    if Z is not None:
        out["beta"] = round(float(model.beta), 4)
    out["trial_definition"] = context["definition"]
    out["trial_seed"] = int(seed)
    out["n_insufficient_foils"] = context["definition"]["n_insufficient_foils"]
    if trials:
        out["candidate_rows"] = [cand.tolist() for _, cand, _ in sampled]
        out["correct_slots"] = [answer for _, _, answer in sampled]
        out["candidate_identities"] = [[context["identity"][j] for j in cand] for _, cand, _ in sampled]
        out["trial_rows"] = rows_used
        out["trials"] = {k: [int(x) for x in correct[k]] for k in ("picture_soft", "fused_soft") if correct[k]}
    return out


@torch.no_grad()
def joint_reps(model, upstream, ds, data, layout, device, batch=256):
    """Joint model over every pair of `ds`: word vectors T [N,D], visual-token outputs without a body R and (with `data`,
    a BodyData over the same rows) with the pair's own body Rb, and token validity [N,F*17]. `layout` = (offset, valid)
    [N,F] of the image windows."""
    model.eval()
    offset, valid = layout
    generator = torch.Generator().manual_seed(0)
    T, R, Rb, ok = [], [], [], []
    for start in range(0, len(ds), batch):
        rows = list(range(start, min(start + batch, len(ds))))
        _, v, t, n = upstream.collate([ds[i] for i in rows])
        v, ix = v.to(device), torch.as_tensor(rows, device=device)
        T.append(model.enc_text(t.to(device), n.to(device)))
        r, o = model.represent(v, offset[ix], valid[ix])
        R.append(r), ok.append(o)
        if data is not None:
            Rb.append(model.represent(v, offset[ix], valid[ix], data.batch(np.array(rows), generator, training=False))[0])
    return torch.cat(T), torch.cat(R), (torch.cat(Rb) if Rb else None), torch.cat(ok)


@torch.no_grad()
def joint_body_present_4afc(model, upstream, ds, data, manifest, layout, device, seed=0, batch=256, subject=None, trials=False, context=None):
    """`body_present_4afc` for the joint model, same trial sampling. `picture_soft`: each candidate's window scored with
    no body tokens; `fused_soft` (body model): with the candidate's own body tokens. Both use `model.score`, the
    training score. No body-only score exists for this model."""
    sampled = _body_trials(manifest, context, seed, subject)
    if len(ds) != len(manifest):
        raise ValueError("Body-present dataset and manifest lengths disagree")
    T, R, Rb, ok = joint_reps(model, upstream, ds, data, layout, device, batch)
    held_all = np.asarray(context["held"], dtype=bool)
    held, rows_used = [], []
    correct = {"picture_soft": [], "fused_soft": []}
    for i, cand, answer in sampled:
        held.append(held_all[i])
        rows_used.append(i)
        scores = {"picture_soft": model.score(R[cand], ok[cand], T[i:i + 1]).squeeze(1)}
        if Rb is not None:
            scores["fused_soft"] = model.score(Rb[cand], ok[cand], T[i:i + 1]).squeeze(1)
        for k, s in scores.items():
            correct[k].append(float(s.argmax() == answer))
    held = np.array(held, bool)
    out = {"n_trials": len(correct["picture_soft"]), "chance": 0.25}
    for k, c in correct.items():
        if c:
            c = np.array(c)
            out[k] = {"acc": round(100 * float(c.mean()), 3), "acc_held": round(100 * float(c[held].mean()), 3) if held.any() else None,
                      "acc_not_held": round(100 * float(c[~held].mean()), 3) if (~held).any() else None}
    out["trial_definition"] = context["definition"]
    out["trial_seed"] = int(seed)
    out["n_insufficient_foils"] = context["definition"]["n_insufficient_foils"]
    if trials:
        out["candidate_rows"] = [cand.tolist() for _, cand, _ in sampled]
        out["correct_slots"] = [answer for _, _, answer in sampled]
        out["candidate_identities"] = [[context["identity"][j] for j in cand] for _, cand, _ in sampled]
        out["trial_rows"] = rows_used
        out["trials"] = {k: [int(x) for x in c] for k, c in correct.items() if c}
    return out


@torch.no_grad()
def probe_4afc_soft(model, region, emb, lut, ev, vocab, dev, n_trials=100, seed=0, max_len=16):
    """Upstream `eval_4afc_region` (same rng, candidate draws, OOV at chance, `per_cat`) scoring each candidate frame
    with the model's soft `score` (single frame, no body) instead of the hard max over regions."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    pools, cat_ids, oov = {}, {}, []
    for cat, g in ev.groupby("category"):
        ids = region.encode(cat, vocab, max_len)
        rows = [lut[region.frame_key(v, f)] for v, f in zip(g.video_id, g.frame_idx) if region.frame_key(v, f) in lut]
        if len(rows) < 4:
            continue
        if ids:
            pools[cat], cat_ids[cat] = rows, ids
        else:
            oov.append(cat)
    cats = sorted(pools)
    if len(cats) < 4:
        model.train(was_training)
        return float("nan"), {}
    all_rows = sorted({r for rs in pools.values() for r in rs})
    V = torch.from_numpy(np.asarray(emb[all_rows], dtype=np.float32)).to(dev)
    Rp = model.enc_regions(V)
    r2i = {r: i for i, r in enumerate(all_rows)}
    valid = torch.ones(4, Rp.shape[1], dtype=torch.bool, device=dev)
    accs = []
    for cat in cats:
        toks = cat_ids[cat]
        t = torch.zeros(1, max_len, dtype=torch.long, device=dev)
        t[0, :len(toks)] = torch.tensor(toks, device=dev)
        tv = model.enc_text(t, torch.tensor([len(toks)], device=dev))
        others = [c for c in cats if c != cat]
        correct = 0
        for _ in range(n_trials):
            cand = [rng.choice(pools[cat])] + [rng.choice(pools[c]) for c in rng.choice(others, 3, replace=False)]
            correct += model.score(Rp[[r2i[r] for r in cand]], valid, tv).squeeze(1).argmax().item() == 0
        accs.append(correct / n_trials)
    model.train(was_training)
    per_cat = dict(zip(cats, accs))
    per_cat.update({c: 0.25 for c in oov})
    return float(np.mean(list(per_cat.values()))), {"per_cat": per_cat, "n_scored": len(cats), "n_oov": len(oov),
                                                   "n_total": len(cats) + len(oov)}


@torch.no_grad()
def distill_agreement(student, teacher, upstream, ds, tdata, device, temp, seed=0, batch=128):
    """Student (picture only) vs teacher (picture + oracle body) on the training score matrix, over fixed random
    batches of `ds` pairs (candidate words are the batch's, as in training). Image -> words direction:
    KL(teacher || student) at `temp` (unscaled), top-1 agreement, and each model's accuracy at picking the true pair."""
    student.eval(), teacher.eval()
    order = np.random.default_rng(seed).permutation(len(ds))
    kl, agree, acc_s, acc_t, n = 0.0, 0, 0, 0, 0
    for start in range(0, len(order) - batch + 1, batch):
        rows = order[start:start + batch]
        _, v, t, c = upstream.collate([ds[i] for i in rows])
        v, t, c = v.to(device), t.to(device), c.to(device)
        ls = student.scale() * student.pair_scores(v, t, c)[2]
        lt = teacher.teacher_logits(v, t, c, tdata.batch(rows, None, training=False))
        kl += float(F.kl_div(F.log_softmax(ls / temp, -1), F.softmax(lt / temp, -1), reduction="sum"))
        agree += int((ls.argmax(1) == lt.argmax(1)).sum())
        diag = torch.arange(len(rows), device=device)
        acc_s += int((ls.argmax(1) == diag).sum())
        acc_t += int((lt.argmax(1) == diag).sum())
        n += len(rows)
    return {"n_pairs": n, "kl": round(kl / n, 5), "top1_agree": round(100 * agree / n, 3),
            "student_acc": round(100 * acc_s / n, 3), "teacher_acc": round(100 * acc_t / n, 3)}


def _fit_probe(x_train, y_train, classes, multilabel=False, steps=300):
    """Linear probe on standardised inputs (Adam, full batch, small weight decay)."""
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x = (x_train - mean) / std
    layer = torch.nn.Linear(x.shape[1], classes)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        out = layer(x)
        loss = F.binary_cross_entropy_with_logits(out, y_train) if multilabel else F.cross_entropy(out, y_train)
        loss.backward()
        opt.step()
    return lambda z: layer((z - mean) / std).detach()


@torch.no_grad()
def _latents(model, data, store, device, batch=2048):
    """Body latent (at each present second) and the raw input vector, plus the (session, second) of each row."""
    s, t = np.nonzero(store["present"])
    data.ps, data.pt = torch.as_tensor(s, device=device), torch.as_tensor(t, device=device)
    data.win_frames = torch.full((len(s), 1), -1, device=device)
    generator = torch.Generator().manual_seed(0)
    z = []
    for start in range(0, len(s), batch):
        z.append(model.latent_of(data.batch(np.arange(start, min(start + batch, len(s))), generator, training=False)).cpu())
    raw = torch.cat([torch.as_tensor(g[s, t]) for g in store["groups"].values()], 1)
    return s, t, torch.cat(z), raw


def body_probe(model, data, store, heldout_subject, device):
    """Probe held word, grasp family and next-second contact from the body latent.

    Splits: by subject (probe fitted on the other subjects, tested on the held-out one) and by source grasp family
    (fitted on three quarters of the source objects, tested on the rest; same subjects). The source grasp family
    is the GRAB object of the clip (clip file name), which differs from the held word because grasps are borrowed.
    Baselines: majority class / train mean for every task, and persistence (contact now = contact next second).
    Also probes the raw input vector, to separate what the encoder adds.
    """
    model.eval()
    s, t, z, raw = _latents(model, data, store, device)
    subject = store["subject"][s]
    names = store["clip_names"]
    family_names = [[n.split("/")[-1].split("_")[0] for n in row] for row in names]
    family_of = lambda i, c: family_names[i][c]
    clip_id = store["clip"][s, t]
    family = np.array([family_of(i, c) for i, c in zip(s, clip_id)])
    held = store["held"][s, t]
    contact = torch.as_tensor(store["targets"]["contact"][s, t]) if "contact" in store["targets"] else None
    nxt = np.minimum(t + 1, store["steps"] - 1)
    next_ok = store["next_valid"][s, t]
    next_contact = torch.as_tensor(store["targets"]["contact"][s, nxt]) if contact is not None else None
    train_subject = subject != heldout_subject
    families = sorted(set(family[train_subject]))
    test_families = set(families[::4])
    in_test_family = np.array([f in test_families for f in family])
    splits = {"subject": (train_subject, ~train_subject), "family": (train_subject & ~in_test_family, train_subject & in_test_family)}
    features = {"latent": z, **({"raw": raw} if contact is not None else {})}
    out = {"heldout_subject": heldout_subject, "test_families": sorted(test_families), "splits": {}}
    for split, (train, test) in splits.items():
        result = {}
        for task in ("held_word", "grasp_family", "next_contact"):
            if task == "grasp_family" and split == "family":
                continue  # a held-out family has no training label
            if task == "next_contact":
                if contact is None:
                    continue
                tr, te = train & next_ok, test & next_ok
                y = (next_contact > 0).float()
                persistence = ((contact > 0).float() == y).float()
                anyc = lambda a: (a > 0).any(1)
                entry = {"n_test": int(te.sum()),
                         "persistence": {"any_acc": round(100 * float((anyc(contact[te]) == anyc(y[te])).float().mean()), 3),
                                         "patch_acc": round(100 * float(persistence[te].mean()), 3)}}
                prior = (y[tr].mean(0) > 0.5).float()
                majority_any = anyc(y[tr]).float().mean() > 0.5
                entry["mean"] = {"any_acc": round(100 * float((anyc(y[te]) == majority_any).float().mean()), 3),
                                 "patch_acc": round(100 * float((y[te] == prior).float().mean()), 3)}
                for name, x in features.items():
                    predict = _fit_probe(x[tr], y[tr], y.shape[1], multilabel=True)
                    guess = (predict(x[te]) > 0).float()
                    entry[name] = {"any_acc": round(100 * float((anyc(guess) == anyc(y[te])).float().mean()), 3),
                                   "patch_acc": round(100 * float((guess == y[te]).float().mean()), 3)}
                result[task] = entry
                continue
            if task == "held_word":
                label = held
                valid = held >= 0
            else:
                label = np.array([families.index(f) if f in families else -1 for f in family])
                valid = held >= 0
            tr, te = train & valid, test & valid
            classes = np.unique(label[tr])
            remap = {c: i for i, c in enumerate(classes)}
            te = te & np.isin(label, classes)
            if tr.sum() == 0 or te.sum() == 0:
                continue
            y_tr = torch.as_tensor([remap[c] for c in label[tr]])
            y_te = torch.as_tensor([remap[c] for c in label[te]])
            majority = int(torch.bincount(y_tr).argmax())
            entry = {"n_test": int(te.sum()), "n_classes": len(classes), "majority": round(100 * float((y_te == majority).float().mean()), 3)}
            for name, x in features.items():
                predict = _fit_probe(x[tr], y_tr, len(classes))
                entry[name] = round(100 * float((predict(x[te]).argmax(1) == y_te).float().mean()), 3)
            result[task] = entry
        out["splits"][split] = result
    return out


def label_manifest(manifest, episode_root, seed):
    """LABEL DIAGNOSTIC: one pair per speech pair, with the speech replaced by the word of one object that is visible
    (visible_px > 0 in `frames.parquet`) in that pair's centre frame, drawn uniformly once (seeded). Only the rows of
    `manifest` (training sessions) are used, so a held-out subject never enters. Pairs with no visible object are dropped.
    Adds `visible`: all visible categories of the frame."""
    frames = pd.read_parquet(episode_root / "frames.parquet", columns=["video_id", "frame_idx", "objects"])
    table = manifest.merge(frames, on=["video_id", "frame_idx"], how="left", validate="many_to_one")
    if table.objects.isna().any():
        raise ValueError("Training pairs without a frame record")
    table["visible"] = table.objects.map(lambda o: sorted({x["category"] for x in json.loads(o) if x["visible_px"] > 0}))
    table = table.drop(columns="objects")
    rng = np.random.default_rng(seed)
    some = table.visible.map(len) > 0
    kept = table[some].reset_index(drop=True)
    kept["text"] = [rng.choice(v) for v in kept.visible]
    info = {"n_speech_pairs": int(len(table)), "n_pairs": int(len(kept)), "n_dropped_none_visible": int((~some).sum()),
            "mean_visible_objects": round(float(kept.visible.map(len).mean()), 3), "n_words": int(kept.text.nunique())}
    return kept, info


@torch.no_grad()
def label_fit(model, upstream, ds, manifest, vocab, device, seed=0, n=4096, batch=256):
    """Top-1 of the label words on training pairs: each pair's image window against every object word (soft pooling,
    uniform region prior). `top1` counts the drawn label; `top1_visible` counts any object visible in the frame.
    A fixed random subset of `n` pairs. `ds` is the FramePairs over `manifest` (same order)."""
    model.eval()
    words = sorted({w for v in manifest.visible for w in v})
    t = torch.zeros(len(words), 16, dtype=torch.long)
    for i, w in enumerate(words):
        ids = upstream.encode(w, vocab)
        t[i, :len(ids)] = torch.tensor(ids)
    T = model.enc_text(t.to(device), torch.ones(len(words), dtype=torch.long, device=device))
    rows = np.sort(np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False))
    top, any_visible = 0, 0
    for start in range(0, len(rows), batch):
        chosen = rows[start:start + batch]
        _, v, _, _ = upstream.collate([ds[i] for i in chosen])
        R = model.enc_regions(v.to(device))
        s = torch.einsum("brd,md->brm", R, T)
        uniform = torch.full(s.shape[:2], -np.log(s.shape[1]), device=device, dtype=s.dtype)
        pred = soft_pool(s, uniform, model.tau_p).argmax(1).cpu().numpy()
        for p, i in zip(pred, chosen):
            top += words[p] == manifest.text[i]
            any_visible += words[p] in manifest.visible[i]
    return {"n_pairs": int(len(rows)), "n_words": len(words), "top1": round(100 * top / len(rows), 3),
            "top1_visible": round(100 * any_visible / len(rows), 3)}
