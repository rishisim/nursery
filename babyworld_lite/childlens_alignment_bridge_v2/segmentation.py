"""Pinned Silero VAD inference and deterministic postprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def silero_speech_segments(
    waveform: np.ndarray,
    model_path: str,
    *,
    sample_rate: int = 16_000,
    threshold: float = 0.5,
    negative_threshold: float = 0.35,
    minimum_speech_ms: int = 300,
    minimum_silence_ms: int = 500,
    speech_padding_ms: int = 150,
    maximum_segment_seconds: float = 12.0,
) -> list[dict[str, float]]:
    """Run the frozen Silero 6.2.1 ONNX graph and return bounded segments."""

    import onnxruntime as ort

    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if sample_rate != 16_000:
        raise ValueError("E_SAMPLE_RATE")
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    chunk_size = 512
    context_size = 64
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    probabilities: list[float] = []
    for offset in range(0, len(signal), chunk_size):
        chunk = signal[offset : offset + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        value = np.concatenate((context, chunk.reshape(1, -1)), axis=1)
        output, state = session.run(
            None,
            {
                "input": value,
                "state": state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        probabilities.append(float(output[0][0]))
        context = value[:, -context_size:]

    minimum_speech = round(minimum_speech_ms * sample_rate / 1000)
    minimum_silence = round(minimum_silence_ms * sample_rate / 1000)
    speech_pad = round(speech_padding_ms * sample_rate / 1000)
    triggered = False
    current_start = 0
    temporary_end = 0
    raw: list[list[int]] = []
    for index, probability in enumerate(probabilities):
        current = index * chunk_size
        if probability >= threshold and not triggered:
            triggered = True
            current_start = current
            temporary_end = 0
            continue
        if probability >= threshold and temporary_end:
            temporary_end = 0
        if probability < negative_threshold and triggered:
            if not temporary_end:
                temporary_end = current
            if current - temporary_end >= minimum_silence:
                if temporary_end - current_start >= minimum_speech:
                    raw.append([current_start, temporary_end])
                triggered = False
                temporary_end = 0
    if triggered and len(signal) - current_start >= minimum_speech:
        raw.append([current_start, len(signal)])

    padded: list[list[int]] = []
    for index, (start, end) in enumerate(raw):
        start = max(0, start - speech_pad)
        end = min(len(signal), end + speech_pad)
        if padded and start < padded[-1][1]:
            midpoint = (start + padded[-1][1]) // 2
            padded[-1][1] = midpoint
            start = midpoint
        padded.append([start, end])

    maximum_samples = round(maximum_segment_seconds * sample_rate)
    result: list[dict[str, float]] = []
    for start, end in padded:
        cursor = start
        while end - cursor > maximum_samples:
            result.append(
                {
                    "start_seconds": cursor / sample_rate,
                    "end_seconds": (cursor + maximum_samples) / sample_rate,
                }
            )
            cursor += maximum_samples
        if end - cursor >= minimum_speech:
            result.append(
                {
                    "start_seconds": cursor / sample_rate,
                    "end_seconds": end / sample_rate,
                }
            )
    return result
