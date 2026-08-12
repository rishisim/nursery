#!/usr/bin/env python3
"""Narrow VTC 2.2 adapter: invoke pinned inference with CSV disabled."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "gpu"), default="cuda")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("pinned_vtc_infer", args.upstream_script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pinned_segma_inference = module.run_inference_on_audios

    def compatible_segma_inference(*, save_probs=False, stride_pct=0.25, **kwargs):
        """Bridge the frozen VTC legacy keywords to its lock-resolved Segma API."""
        if stride_pct != 0.25:
            raise ValueError("unsupported nondefault frozen VTC stride")
        return pinned_segma_inference(save_logits=save_probs, **kwargs)

    module.run_inference_on_audios = compatible_segma_inference
    module.main(output=str(args.output), wavs=str(args.wavs), config=str(args.config),
                checkpoint=str(args.checkpoint), thresholds=args.thresholds,
                device=args.device, write_csv=False)


if __name__ == "__main__":
    main()
