# Gemma 4 public-canary rendered-placeholder order correction v1.3

Status: `FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY`

The CAF revision passed the public fixture, runtime, and model-load stages. Generation was never called: the bound processor rendered the unchanged declarative audio-then-five-images-then-text message as system text, five image placeholders, user text, then the audio placeholder. The existing fail-closed order assertion therefore raised `E_CONTENT_ORDER` before any model output existed.

This is a public, content-independent processor-template compatibility defect. It is not semantic evidence, a parser failure, restricted calibration evidence, or a scientific result. The correction permits one operation after exact chat-template rendering: move the single literal `<|audio|>` immediately ahead of the five contiguous literal `<|image|>` placeholders. A deletion check requires every non-audio-placeholder byte to remain identical.

The declarative messages remain audio, five ordered images, then the exact user text. Prompt text, message content, CAF and final-WAV contracts, frame generation and offsets, model identity and hashes, schema, parser allowance, sample, export rules, gates, calibration bounds, causal protocol, and simulator-oracle boundary are unchanged. The correction spends no parser allowance and permits no fallback.

The prior CAF failure receipt remains authoritative for that attempt. Any future public rerun must use the new output namespace and emit fresh receipts. This amendment does not execute a canary, authorize restricted inference, or open a scientific endpoint.
