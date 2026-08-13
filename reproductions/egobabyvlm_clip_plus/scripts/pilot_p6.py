#!/usr/bin/env python3
"""Governed Pilot P6 tokenizer gate and BERT rehearsal entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


RUN_ID = "p6-6d31a4e7"
P3_RUN_ID = "p3-3d96f71c"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


class GateError(RuntimeError):
    """A frozen P6 gate failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require_owner_only(path: Path, directory: bool = True) -> None:
    if not path.is_absolute() or not path.exists():
        raise GateError("governed root is absent or not absolute")
    if directory and not path.is_dir():
        raise GateError("governed root is not a directory")
    mode = stat.S_IMODE(path.stat().st_mode)
    expected = 0o700 if directory else 0o600
    if mode != expected:
        raise GateError(f"governed path mode must be {expected:o}")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["engineering_run_id"] != RUN_ID or value["stage"] != "P6":
        raise GateError("wrong frozen P6 config")
    forbidden = json.dumps(value).lower()
    for needle in ("model_name_or_path", "from_pretrained", "torch.hub"):
        if needle in forbidden:
            raise GateError("forbidden pretrained initialization reference")
    return value


def find_manifest(p3_root: Path, slot: str) -> Path:
    matches = list((p3_root / slot).glob("*/manifests/text.json"))
    if len(matches) != 1:
        raise GateError("expected exactly one governed retained-text manifest per slot")
    return matches[0]


def load_text(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise GateError("retained-text manifest is not a list")
    texts: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise GateError("retained-text row is malformed")
        text = row["text"].strip()
        if text:
            texts.append(text)
    return texts


def write_corpus(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text("".join(f"{text}\n" for text in texts), encoding="utf-8")
    os.chmod(path, 0o600)


def train_tokenizer(texts: list[str], output: Path, cfg: dict[str, Any]) -> tuple[int, str]:
    from tokenizers import BertWordPieceTokenizer

    tok_cfg = cfg["tokenizer"]
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=False,
    )
    tokenizer.train_from_iterator(
        texts,
        vocab_size=tok_cfg["vocab_size"],
        min_frequency=tok_cfg["min_frequency"],
        limit_alphabet=tok_cfg["limit_alphabet"],
        special_tokens=list(SPECIAL_TOKENS),
        wordpieces_prefix=tok_cfg["continuing_subword_prefix"],
        show_progress=False,
    )
    vocab = tokenizer.get_vocab()
    if len(vocab) != len(set(vocab)):
        raise GateError("tokenizer vocabulary is not unique")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if vocab.get(token) != expected_id:
            raise GateError("special-token ID contract failed")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    tokenizer.save_model(str(output))
    tokenizer.save(str(output / "tokenizer.json"))
    for path in output.iterdir():
        os.chmod(path, 0o600)
    return len(vocab), sha256(output / "tokenizer.json")


def prepare(args: argparse.Namespace) -> int:
    scratch_root = args.scratch_root.resolve()
    durable_root = args.durable_root.resolve()
    require_owner_only(scratch_root)
    require_owner_only(durable_root)
    config = load_config(args.config.resolve())
    config_hash = sha256(args.config.resolve())
    run_root = scratch_root / "runs" / "pilot_p6" / RUN_ID
    durable = durable_root / "run_records" / "pilot_p0" / "p0-4cc3af23" / "pilot_p6"
    completion = durable / "completion.json"
    if completion.exists():
        raise GateError("P6 already has a completion marker")
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(run_root, 0o700)
    durable.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(durable, 0o700)
    frozen_copy = run_root / "resolved_config.json"
    if frozen_copy.exists() and sha256(frozen_copy) != config_hash:
        raise GateError("existing frozen config hash mismatch")
    if not frozen_copy.exists():
        shutil.copy2(args.config, frozen_copy)
        os.chmod(frozen_copy, 0o600)

    p3_root = scratch_root / "runs" / "pilot_p3" / P3_RUN_ID
    split = config["input"]["recording_level_split"]
    train_manifest = find_manifest(p3_root, split["train_slot"])
    validation_manifest = find_manifest(p3_root, split["validation_slot"])
    if train_manifest.parent.parent.parent == validation_manifest.parent.parent.parent:
        raise GateError("recording-level split is not disjoint")
    if sha256(train_manifest) != split["train"]["manifest_sha256"]:
        raise GateError("training retained-text checksum mismatch")
    if sha256(validation_manifest) != split["validation"]["manifest_sha256"]:
        raise GateError("validation retained-text checksum mismatch")
    train_text = load_text(train_manifest)
    validation_text = load_text(validation_manifest)
    counts = {
        "train_utterances": len(train_text),
        "train_words": sum(len(text.split()) for text in train_text),
        "validation_utterances": len(validation_text),
        "validation_words": sum(len(text.split()) for text in validation_text),
    }
    expected_counts = {
        "train_utterances": split["train"]["utterances"],
        "train_words": split["train"]["words"],
        "validation_utterances": split["validation"]["utterances"],
        "validation_words": split["validation"]["words"],
    }
    if counts != expected_counts:
        raise GateError("privacy-safe split counts mismatch")
    corpus_dir = run_root / "corpus"
    write_corpus(corpus_dir / "train.txt", train_text)
    write_corpus(corpus_dir / "validation.txt", validation_text)
    tokenizer_dir = run_root / "tokenizer"
    if tokenizer_dir.exists():
        raise GateError("tokenizer gate is single-execution and output already exists")
    vocab_size, tokenizer_hash = train_tokenizer(train_text, tokenizer_dir, config)
    result = {
        "schema_version": 1,
        "record_type": "pilot_p6_tokenizer_gate_detail",
        "engineering_run_id": RUN_ID,
        "classification": config["classification"],
        "status": "tokenizer_gate_passed" if vocab_size == config["tokenizer"]["vocab_size"] else "stopped_natural_vocabulary_below_target",
        "split": {"recording_level_disjoint": True, "fallback_used": False, **counts},
        "tokenizer": {
            "algorithm": "WordPiece",
            "training_split_only": True,
            "external_tokens": False,
            "fabricated_tokens": False,
            "observed_unique_vocab_size": vocab_size,
            "target_vocab_size": config["tokenizer"]["vocab_size"],
            "required_special_token_count": len(SPECIAL_TOKENS),
            "special_token_ids_match": True,
            "tokenizer_json_sha256": tokenizer_hash,
        },
        "config_sha256": config_hash,
        "bert_training_started": False,
        "p5_or_dino_loaded": False,
        "p4_loaded": False,
        "p7_started": False,
        "scientific_status_effect": "none",
        "inventory_status": "incomplete_inventory",
    }
    atomic_json(run_root / "tokenizer_gate.json", result)
    atomic_json(durable / "tokenizer_gate.json", result)
    if vocab_size != config["tokenizer"]["vocab_size"]:
        print(json.dumps({"status": result["status"], "observed_unique_vocab_size": vocab_size, "target_vocab_size": config["tokenizer"]["vocab_size"]}))
        return 3
    print(json.dumps({"status": result["status"], "vocab_size": vocab_size}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-tokenizer",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare-tokenizer":
            return prepare(args)
        raise GateError("unsupported command")
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"P6 gate error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
