#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGGREGATE = ROOT / "pilots" / "juno_sample" / "p1_aggregate.json"


def require(value, message):
    if not value:
        raise AssertionError(message)


def load_audit():
    spec = importlib.util.spec_from_file_location("pilot_p1_audit", ROOT / "pilot_p1_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_probe(path):
    path.write_text("""#!/usr/bin/env python3
import json,sys
if '-version' in sys.argv:
 print('ffprobe version synthetic-test')
else:
 print(json.dumps({'streams':[{'codec_type':'video','codec_name':'h264','width':640,'height':480,'avg_frame_rate':'30/1'},{'codec_type':'audio','codec_name':'aac','sample_rate':'16000','channels':1}],'format':{'duration':'2.5','format_name':'mov,mp4'}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def synthetic_tests():
    audit = load_audit()
    with tempfile.TemporaryDirectory() as name:
        root = Path(name); media = root / "media"; media.mkdir()
        queue = []
        for index in range(2):
            key = "synthetic_key_%d" % index; content = ("fixture-%d" % index).encode()
            target = media / (key + ".mp4"); target.write_bytes(content)
            queue.append({"record_key": key, "status": "downloaded", "size_bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()})
        ledger = root / "ledger.json"; ledger.write_text(json.dumps({"queue": queue}))
        executable = root / "ffprobe"; fake_probe(executable)
        provenance = {"asset_sha256": "a" * 64, "ffprobe_sha256": "b" * 64,
                      "release_tag": "synthetic", "platform": "linux-x86_64"}
        first = audit.audit(ledger, media, root / "detail.json", "p0-4cc3af23",
                            str(executable), provenance)
        second = audit.audit(ledger, media, root / "detail.json", "p0-4cc3af23",
                             str(executable), provenance)
        require(first == second, "aggregation is not deterministic")
        require(first["input_count"] == 2 and first["p1_gate"]["passed"], "two-input gate failed")
        require(stat.S_IMODE((root / "detail.json").stat().st_mode) == 0o600,
                "governed record mode is not owner-only")
        rendered = json.dumps(first)
        require("synthetic_key" not in rendered and "ledger.json" not in rendered,
                "aggregate leaks governed details")
        bad = json.loads(ledger.read_text()); bad["queue"].append(dict(queue[0]))
        ledger.write_text(json.dumps(bad))
        try:
            audit.select_records(bad)
        except audit.AuditError:
            pass
        else:
            raise AssertionError("exact two-input contract not enforced")
        require(audit.probe_media(Path("sensitive-name"), str(executable))["error_status"] is None,
                "synthetic probe unexpectedly failed")
        failing = root / "failing-ffprobe"
        failing.write_text("#!/bin/sh\necho 'private-filename-and-path' >&2\nexit 1\n", encoding="utf-8")
        failing.chmod(failing.stat().st_mode | stat.S_IXUSR)
        failure = audit.probe_media(Path("private-media-name"), str(failing))
        require(failure["error_status"] == "media_probe_failed", "probe error not normalized")
        require("private" not in json.dumps(failure), "probe error leaked sensitive text")


def main():
    synthetic_tests()
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    require(aggregate["status"] == "p1_complete", "P1 is not complete")
    require(aggregate["engineering_run_id"] == "p0-4cc3af23", "P0 namespace changed")
    require(aggregate["inventory_status"] == "incomplete_inventory", "inventory overclaimed")
    require(aggregate["input_count"] == 2, "tracked input count changed")
    require(aggregate["scientific_status_effect"] == "none", "scientific status changed")
    require(aggregate["p0_status_preserved"] is True, "P0 status not preserved")
    require(aggregate["later_stage_started"] is False, "later stage started")
    require(aggregate["full_corpus_discrepancy"] == "unresolved_and_out_of_scope",
            "full-corpus blocker changed")
    require(aggregate["p1_gate"] == {"passed": True, "stop_reason": None}, "P1 gate failed")
    text = AGGREGATE.read_text(encoding="utf-8")
    forbidden = ["record_key", "source_link", "media_path", "filename", "expected_sha256",
                 "observed_sha256", "/work/", "/scratch/", "participant", "session", "asset_id"]
    require(not any(token in text for token in forbidden), "tracked aggregate violates privacy boundary")
    print("Pilot P1 verification passed")


if __name__ == "__main__":
    main()
