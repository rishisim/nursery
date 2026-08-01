#!/usr/bin/env python3
"""Frozen offline German ASR→English adapter shared by qualification and governed runs."""

from __future__ import annotations

import math
import re
import unicodedata


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def serialize_words(words, decimals: int = 6):
    return [
        {
            "word": normalize_text(str(item.get("word", ""))),
            "start": round(float(item["start"]), decimals),
            "end": round(float(item["end"]), decimals),
            "probability": round(float(item.get("probability", 0.0)), decimals),
        }
        for item in words
    ]


def validate_asr_prediction(prediction, audio_duration: float, confidence_min: float = 0.35):
    text = normalize_text(str(prediction.get("text", "")))
    language = str(prediction.get("language", ""))
    try:
        words = serialize_words(prediction.get("words", []))
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": "ABSTAIN", "reason": "INVALID_TIMESTAMP", "text_de": text, "words": []}
    reason = None
    if language != "de":
        reason = "LANGUAGE_MISMATCH"
    elif not text or not words:
        reason = "EMPTY_ASR"
    else:
        previous_end = 0.0
        for word in words:
            values = (word["start"], word["end"], word["probability"])
            if not all(math.isfinite(x) for x in values) or word["start"] < previous_end or word["end"] < word["start"] or word["start"] < 0 or word["end"] > audio_duration:
                reason = "INVALID_TIMESTAMP"
                break
            previous_end = word["end"]
    confidence = sum(w["probability"] for w in words) / len(words) if words else 0.0
    if reason is None and confidence < confidence_min:
        reason = "LOW_CONFIDENCE"
    return {
        "status": "ABSTAIN" if reason else "ACCEPT",
        "reason": reason,
        "text_de": text,
        "language": language,
        "words": words,
        "mean_word_confidence": round(confidence, 6),
    }


def translate_accepted(adjudication, tokenizer, translator, max_new_tokens: int = 128):
    if adjudication["status"] != "ACCEPT":
        return {**adjudication, "text_en": ""}
    encoded = tokenizer(adjudication["text_de"], return_tensors="pt", truncation=False)
    length = int(encoded["input_ids"].shape[1])
    if length > int(tokenizer.model_max_length):
        return {**adjudication, "status": "ABSTAIN", "reason": "SILENT_TRUNCATION", "text_en": ""}
    encoded = {key: value.to(translator.device) for key, value in encoded.items()}
    generated = translator.generate(**encoded, max_new_tokens=max_new_tokens)
    english = normalize_text(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
    if not english:
        return {**adjudication, "status": "ABSTAIN", "reason": "EMPTY_TRANSLATION", "text_en": ""}
    return {**adjudication, "text_en": english}


def whisper_prediction(model, audio_path, decoding):
    result = model.transcribe(
        str(audio_path), language="de", task="transcribe",
        temperature=decoding["temperature"], beam_size=decoding["beam_size"],
        word_timestamps=True, condition_on_previous_text=False, fp16=decoding.get("fp16", False),
        verbose=False,
    )
    words = [word for segment in result.get("segments", []) for word in segment.get("words", [])]
    return {"text": result.get("text", ""), "language": result.get("language", ""), "words": words}
