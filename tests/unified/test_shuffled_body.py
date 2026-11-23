import csv
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from nursery.unified import train as training
from nursery.unified.evaluate import UnifiedEvaluation, log_routing


class Dataset(SimpleNamespace):
    def __len__(self):
        return len(self.ids)


def test_routing_weights_source_clips_equally(tmp_path):
    panels = {
        "test": pd.DataFrame(dict(word=["lift"] * 3, clip=["a", "a", "b"],
                                  p_vis=[1., 0., 0.], p_temp=[0., 1., 1.], p_body=[0.] * 3)),
        "s8": pd.DataFrame(dict(word=["lift"], clip=["c"], p_vis=[1.], p_temp=[0.], p_body=[0.])),
    }

    def evaluate(model, panel, *, body_mode, routing):
        assert body_mode == "shuffled" and routing
        return panels[panel]

    study = SimpleNamespace(word_types={"action": {"held_out": {"test": "test", "s8": "s8"}}},
                            evaluate=evaluate, body_mode="shuffled")
    log_routing(None, study, tmp_path)
    assert json.loads((tmp_path / "routing.json").read_text()) == [
        dict(word="lift", kind="action", n_clips=3, p_vis=0.5, p_temp=0.5, p_body=0.)
    ]


def test_shuffled_body_training_and_evaluation(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    clip = np.array([[0, 0, 1, 1, 2, 2]])
    signals = np.arange(6, dtype=np.float32).reshape(1, 6, 1)
    store = dict(video_ids=["v"], row={"v": 0}, steps=6, clip=clip, present=clip >= 0,
                 hand=np.zeros_like(clip), next_valid=clip >= 0, targets={},
                 groups={name: signals for name in ("contact", "fingers", "arm", "load", "skin")})
    words = ["ball", "push", "cup", "lift"]
    rows = pd.DataFrame(dict(video_id=["v"] * 4, frame_idx=range(4), referent=words))
    frame_keys = {f"v|{i}": (0, i) for i in range(4)}
    dataset = Dataset(caches=[rng.standard_normal((4, 257, 768), dtype=np.float32)],
                      locs=[[loc] for loc in frame_keys.values()], Fmax=1, cls_only=False,
                      max_len=1, ids=[[i] for i in range(1, 5)])
    trials = dict(trial_rows=list(range(4)), candidate_rows=[list(range(4))] * 4,
                  correct_slots=list(range(4)), donor_rows=[2, 3, 0, 1])
    temporal = SimpleNamespace(batch=lambda *args, **kwargs: None)
    gradients = []
    make_model = training.UnifiedEmbodiedLearner

    def model_with_gradient_hook(**kwargs):
        model = make_model(**kwargs)
        model.body.project.weight.register_hook(lambda grad: gradients.append(grad.detach()))
        return model

    def study_factory(n_clips, condition, device, *, shuffle_body):
        study = UnifiedEvaluation.__new__(UnifiedEvaluation)
        study.condition, study.device = condition, device
        study.body_mode = "real"
        study.include = clip < 2
        study.shuffle_info = None
        study.store, study.lut = store, frame_keys
        study.source_clips = pd.Series(clip[0, :4], index=pd.MultiIndex.from_frame(
            rows[["video_id", "frame_idx"]]))
        study.vocab = {word: i for i, word in enumerate(["<pad>", *words])}
        study.upstream = SimpleNamespace(encode=lambda word, vocab: [vocab[word]])
        study.protocol = dict(lr=0.0003, weight_decay=0.1, batch=4, body_window=5, body_dropout=0.)
        study.word_types = dict(noun=dict(dev="panel", held_out={"test": "panel"}),
                                action=dict(dev="panel", held_out={"test": "panel", "s8": "panel"},
                                            cluster="video_id"))
        study.gaze_panels, study.temporal_panels = {"panel": None}, {"panel": temporal}
        real_body = study._training_body(rows, 0, 7)
        study.body_mode = "shuffled" if shuffle_body else "real"
        batch = real_body.batch

        def panel_body_batch(ids, generator, *, training):
            expected = trials["donor_rows"] if shuffle_body else trials["trial_rows"]
            assert ids.tolist() == expected
            return batch(ids, generator, training=training)

        real_body.batch = panel_body_batch
        study._panel = lambda panel: (rows, trials, dataset, real_body, None)

        def training_data(*, seed):
            body = study._training_body(rows, 0, seed)
            centers = body.batch(np.arange(4), None, training=False)["groups"]["contact"][:, 5, 0]
            assert centers.tolist() == ([2., 3., 0., 1.] if shuffle_body else [0., 1., 2., 3.])
            return dataset, None, temporal, body

        study.training_data = training_data
        return study

    def make_test_dir(path, *, exist_ok):
        path.mkdir(exist_ok=exist_ok)
        return path

    monkeypatch.setattr(training, "_output_directory", make_test_dir)
    monkeypatch.setattr(training, "UnifiedEvaluation", study_factory)
    monkeypatch.setattr(training, "UnifiedEmbodiedLearner", model_with_gradient_hook)

    normal, shuffled = (tmp_path / name for name in ("V+G+B", "V+G+B_shuffled_body"))
    for path, shuffle_body in ((normal, False), (shuffled, True)):
        training.train(SimpleNamespace(epochs=1, aux_weight=0.25,
                                       device="cpu", out_dir=path, n_clips=2,
                                       condition="V+G+B", seed=7, lambda_gen=0.2, tau_m=0.35,
                                       shuffle_body=shuffle_body))

    assert len(gradients) == 2
    assert all(torch.isfinite(grad).all() and grad.count_nonzero() for grad in gradients)
    for path, mode in ((normal, "real"), (shuffled, "shuffled")):
        metrics = json.loads((path / "metrics.json").read_text())
        checkpoint = torch.load(path / "best.pt", weights_only=True)
        run = metrics["run"]
        assert run == checkpoint["run"]
        assert run["tau_m"] == checkpoint["model_args"]["tau_m"] == 0.35
        assert run["aux_weight"] == checkpoint["model_args"]["aux_weight"] == 0.25
        assert run["name"] == path.name and run["condition"] == "V+G+B"
        assert run["shuffle_body"] == (mode == "shuffled") and run["seed"] == 7
        result = metrics["held_out"]["as_trained"]["noun"]["test"]
        assert result["body_mode"] == mode
        if mode == "shuffled":
            assert run["training_body_shuffle"] == {"units": 2, "moved": 2}
            assert result["donor_rows"] == trials["donor_rows"]
        else:
            assert run["training_body_shuffle"] is None
            assert "donor_rows" not in result
        assert (path / "history.json").is_file()
        routing = json.loads((path / "routing.json").read_text())
        with (path / "routing.csv").open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [
                {key: str(value) for key, value in row.items()} for row in routing
            ]
        assert {(row["kind"], row["word"]) for row in routing} == {
            (kind, word) for kind in ("noun", "action") for word in words
        }
        assert all(row["n_clips"] == 1 for row in routing)
        assert all(row["p_temp"] == 0 and row["p_body"] > 0 for row in routing)
        np.testing.assert_allclose([row["p_vis"] + row["p_body"] for row in routing], 1)
