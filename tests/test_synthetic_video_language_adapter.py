from pathlib import Path
import importlib.util


SPEC = importlib.util.spec_from_file_location("adapter", Path("scripts/synthetic_video_language_adapter.py"))
assert SPEC and SPEC.loader
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def prediction(**updates):
    value = {"text": "Der Ball", "language": "de", "words": [{"word": " Der", "start": 0.0, "end": 0.4, "probability": 0.9}, {"word": " Ball", "start": 0.4, "end": 0.8, "probability": 0.8}]}
    value.update(updates)
    return value


def test_adapter_accepts_and_normalizes_valid_german():
    result = ADAPTER.validate_asr_prediction(prediction(), 1.0)
    assert result["status"] == "ACCEPT"
    assert result["text_de"] == "Der Ball"
    assert result["mean_word_confidence"] == 0.85


def test_adapter_abstains_on_language_empty_and_confidence():
    assert ADAPTER.validate_asr_prediction(prediction(language="en"), 1.0)["reason"] == "LANGUAGE_MISMATCH"
    assert ADAPTER.validate_asr_prediction(prediction(text="", words=[]), 1.0)["reason"] == "EMPTY_ASR"
    low = prediction(words=[{"word": "x", "start": 0, "end": 0.2, "probability": 0.34}])
    assert ADAPTER.validate_asr_prediction(low, 1.0)["reason"] == "LOW_CONFIDENCE"


def test_adapter_abstains_on_nonmonotonic_or_out_of_bounds_timestamps():
    nonmonotonic = prediction(words=[{"word": "a", "start": 0.4, "end": 0.6, "probability": 1}, {"word": "b", "start": 0.3, "end": 0.7, "probability": 1}])
    assert ADAPTER.validate_asr_prediction(nonmonotonic, 1.0)["reason"] == "INVALID_TIMESTAMP"
    out = prediction(words=[{"word": "a", "start": 0, "end": 1.1, "probability": 1}])
    assert ADAPTER.validate_asr_prediction(out, 1.0)["reason"] == "INVALID_TIMESTAMP"


class FakeTensor:
    shape = (1, 5)
    def to(self, _device): return self


class FakeTokenizer:
    model_max_length = 4
    def __call__(self, *_args, **_kwargs): return {"input_ids": FakeTensor()}
    def batch_decode(self, *_args, **_kwargs): return [""]


class FakeTranslator:
    device = "cpu"
    def generate(self, **_kwargs): return [[1]]


def test_adapter_abstains_before_silent_translation_truncation():
    accepted = ADAPTER.validate_asr_prediction(prediction(), 1.0)
    result = ADAPTER.translate_accepted(accepted, FakeTokenizer(), FakeTranslator())
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "SILENT_TRUNCATION"


def test_adapter_abstains_on_empty_translation():
    tokenizer = FakeTokenizer()
    tokenizer.model_max_length = 10
    accepted = ADAPTER.validate_asr_prediction(prediction(), 1.0)
    result = ADAPTER.translate_accepted(accepted, tokenizer, FakeTranslator())
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == "EMPTY_TRANSLATION"


def test_same_adapter_is_used_by_public_and_governed_entrypoints():
    public_source = Path("scripts/run_synthetic_video_language_gate.py").read_text()
    governed_source = Path("scripts/build_synthetic_video_phase4_assets.py").read_text()
    assert "from synthetic_video_language_adapter import" in public_source
    assert "from synthetic_video_language_adapter import" in governed_source
    assert 'segment.get("status") != "ACCEPT"' in governed_source
