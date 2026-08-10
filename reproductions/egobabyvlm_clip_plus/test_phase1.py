import json, os, tempfile, unittest
from pathlib import Path
from unittest import mock
import audit_phase1 as audit

class AuditTests(unittest.TestCase):
    def test_deterministic_discovery_keys_hash_and_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "b.mp4").write_bytes(b"b"); (root / "a.mov").write_bytes(b"a")
            self.assertEqual([p.name for p in audit.discover(root)], ["a.mov", "b.mp4"])
            self.assertEqual(audit.stream_sha256(root / "a.mov"), __import__("hashlib").sha256(b"a").hexdigest())
            self.assertEqual(audit.private_key(b"x" * 32, "v"), audit.private_key(b"x" * 32, "v"))
            rows = [{"duration_seconds": 2.0, "decision": "include", "exclusion_reason": "none"}, {"duration_seconds": 3.0, "decision": "exclude", "exclusion_reason": "audio_missing"}]
            result = audit.aggregate(rows)
            self.assertEqual((result["duration_seconds"], result["included_seconds"], result["excluded_seconds"]), (5, 2, 3))
            self.assertEqual(list(result["by_exclusion_reason"]), ["audio_missing", "none"])

    def test_atomic_write_replaces_complete_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "ledger.jsonl"; audit.atomic_write(target, [{"z": 1, "a": 2}])
            self.assertEqual(target.read_text(), '{"a":2,"z":1}\n')

    def test_existing_ledger_is_resumable(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "ledger.jsonl"; row = {"record_key": "a" * 64, "decision": "undetermined"}
            audit.atomic_write(target, [row])
            self.assertEqual(audit.load_existing(target), {"a" * 64: row})

    def test_probe_does_not_put_identifier_in_error(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("sensitive-name")):
            self.assertEqual(audit.probe(Path("sensitive-name.mp4"), "ffprobe")[3], "FileNotFoundError")

    def test_output_outside_governed_roots_is_rejected(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("storage.storage_root", side_effect=[Path(td) / "d", Path(td) / "s"]):
            with self.assertRaisesRegex(ValueError, "configured governed storage"):
                audit.validate_governed_output(Path(td) / "elsewhere" / "ledger.jsonl")

if __name__ == "__main__": unittest.main()
