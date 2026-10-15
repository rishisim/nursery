"""Body-side tests: body-present 4AFC on the held-out subject and the body probe."""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import soft_pool


@torch.no_grad()
def body_present_4afc(model, upstream, ds, data, manifest, device, seed=0, batch=256):
    """4AFC with the word given and four candidate situations (image window + body), held-out subject only.

    `manifest` holds the held-out subject's named utterances with a visible referent; `ds` is the upstream
    FramePairs over those rows and `data` a BodyData over the same rows (None without a body). Each trial: the true situation plus three foils from other sessions with other
    referents. Scores per candidate: picture (hard max over the window's regions), body (fused-projection cosine to
    the word) and their sum (late fusion), plus `picture_soft` / `fused_soft`: the same with the training pooling
    (soft-MIL, uniform region prior, tau_p), which is the score the model was trained on and the one to read.
    Foils use their own body, so the body must match the situation.
    """
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
    rng = np.random.default_rng(seed)
    referent = manifest.referent.to_numpy()
    video = manifest.video_id.to_numpy()
    held_all = manifest.held_referent.to_numpy().astype(bool)
    held = []
    correct = {"picture": [], "body": [], "fused": [], "picture_soft": [], "fused_soft": []}
    tau = getattr(model, "tau_p", None)
    for i in range(len(manifest)):
        pool = np.flatnonzero((referent != referent[i]) & (video != video[i]))
        if len(pool) < 3:
            pool = np.flatnonzero(referent != referent[i])
        if len(pool) < 3:
            continue
        held.append(held_all[i])
        cand = np.concatenate([[i], rng.choice(pool, 3, replace=False)])
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
            correct[k].append(float(s.argmax() == 0))
    held = np.array(held, bool)
    out = {"n_trials": len(next(iter(correct.values()))), "chance": 0.25}
    for k, c in correct.items():
        if c:
            c = np.array(c)
            out[k] = {"acc": round(100 * float(c.mean()), 3), "acc_held": round(100 * float(c[held].mean()), 3) if held.any() else None,
                      "acc_not_held": round(100 * float(c[~held].mean()), 3) if (~held).any() else None}
    if Z is not None:
        out["beta"] = round(float(model.beta), 4)
    return out


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
