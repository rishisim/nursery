from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "childlens_human_validation_v1_2.py"
SPEC = importlib.util.spec_from_file_location("childlens_human_validation_v1_2", MODULE_PATH)
assert SPEC and SPEC.loader
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
sys.modules["childlens_human_validation_v1_2"] = workflow
LAUNCH_PATH = ROOT / "scripts" / "launch_childlens_human_validation_v1_2.py"
LAUNCH_SPEC = importlib.util.spec_from_file_location("launch_childlens_human_validation_v1_2", LAUNCH_PATH)
assert LAUNCH_SPEC and LAUNCH_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCH_SPEC)
LAUNCH_SPEC.loader.exec_module(launcher)


class WorkflowFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="childlens-v12-test-")
        self.root = Path(self.temp.name) / "restricted"
        self.root.mkdir(mode=0o700)
        os.chmod(self.root, 0o700)
        (self.root / workflow.INDEX_SENTINEL).touch(mode=0o600)
        (self.root / "media").mkdir(mode=0o700)
        self.previous_env = os.environ.get(workflow.ROOT_ENV)
        os.environ[workflow.ROOT_ENV] = str(self.root)
        workflow.bootstrap_policy(self.root, attest=True)
        self.skeleton = self.root / "frozen_v1_1_skeleton.json"
        skeleton = {
            "schema_version": "childlens-human-validation-packet-v1.1.0",
            "packet_status": "SKELETON_AWAITING_MEDIA_AND_HUMANS",
            "pilot_selection_sha256": "a" * 64,
            "selected_count": 15,
            "human_judgment_required": True,
            "codex_may_act_as_human": False,
            "timing_and_speaker_items_double_coded": True,
            "referential_items_minimum_double_code_fraction": 0.2,
            "learner_training_permitted": False,
            "items": [
                {
                    "blinded_item_key": f"opaqueitemkey{index:04d}",
                    "acquisition_status": "PENDING_EXACT_REMOTE_BYTES",
                    "language_judgment": None,
                    "audio_integrity": None,
                    "utterance_timing_text_role_review": None,
                    "referential_status_review": None,
                    "adjudication_status": "NOT_STARTED",
                }
                for index in range(15)
            ],
        }
        self.skeleton.write_text(json.dumps(skeleton), encoding="utf-8")
        os.chmod(self.skeleton, 0o600)
        digest = hashlib.sha256(workflow._canonical(skeleton)).hexdigest()
        self.digest_patcher = mock.patch.object(workflow, "FROZEN_V1_1_HUMAN_PACKET_SHA256", digest)
        self.digest_patcher.start()
        objects = []
        annotations = []
        media_rows = []
        native_items = []
        for index in range(15):
            media_key = f"opaqueitemkey{index:04d}"
            object_key = hashlib.sha256(f"source-object-{index}".encode()).hexdigest()
            annotation_object_key = hashlib.sha256(f"annotation-object-{index}".encode()).hexdigest()
            annotation_sha = hashlib.sha256(f"annotation-bytes-{index}".encode()).hexdigest()
            media_payload = f"synthetic-test-media-{index}".encode()
            media_sha = hashlib.sha256(media_payload).hexdigest()
            relative = f"raw_v1_2/{media_sha}.bin"
            media_path = self.root / relative
            media_path.parent.mkdir(mode=0o700, exist_ok=True)
            media_path.write_bytes(media_payload)
            os.chmod(media_path, 0o600)
            objects.extend(
                [
                    {"object_key": object_key, "local_sha256": None},
                    {"object_key": annotation_object_key, "local_sha256": annotation_sha},
                ]
            )
            annotations.append({"linked_media_key": media_key, "object_key": annotation_object_key})
            media_rows.append(
                {
                    "media_key": media_key,
                    "object_key": object_key,
                    "duration_milliseconds": 2_000_000,
                    "coarse_activity_label": f"ACTIVITY_{index % 3}",
                    "speech_presence_bin": "PRESENT",
                    "location_label": f"LOCATION_{index % 2}",
                }
            )
            native_items.append(
                {
                    "selection_rank": index + 1,
                    "object_key": object_key,
                    "status": "COMPLETE",
                    "local_sha256": media_sha,
                    "stored_relative_path": relative,
                    "transferred_bytes": len(media_payload),
                }
            )
        restricted_input = {"schema_version": "synthetic", "objects": objects, "annotations": annotations, "media": media_rows}
        self.restricted_input = self.root / "frozen_restricted_input.json"
        self.restricted_input.write_text(json.dumps(restricted_input), encoding="utf-8")
        os.chmod(self.restricted_input, 0o600)
        input_digest = hashlib.sha256(workflow._canonical(restricted_input)).hexdigest()
        self.input_digest_patcher = mock.patch.object(
            workflow, "FROZEN_V1_1_RESTRICTED_INPUT_SHA256", input_digest
        )
        self.input_digest_patcher.start()
        self.frozen_bindings = workflow._validate_restricted_input(
            self.restricted_input, self.root, {row["blinded_item_key"] for row in skeleton["items"]}
        )
        native_receipt = {
            "schema_version": "childlens-native-transfer-restricted-receipt-v1.2.0",
            "status": "COMPLETE",
            "pilot_selection_sha256": "a" * 64,
            "items": native_items,
            "restricted_receipt_sha256": None,
        }
        native_receipt["restricted_receipt_sha256"] = hashlib.sha256(
            workflow._canonical(native_receipt)
        ).hexdigest()
        self.native_receipt = self.root / "native_transfer_receipt.json"
        self.native_receipt.write_text(json.dumps(native_receipt), encoding="utf-8")
        os.chmod(self.native_receipt, 0o600)
        self.native_by_object = {row["object_key"]: row for row in native_items}

    def tearDown(self) -> None:
        self.input_digest_patcher.stop()
        self.digest_patcher.stop()
        if self.previous_env is None:
            os.environ.pop(workflow.ROOT_ENV, None)
        else:
            os.environ[workflow.ROOT_ENV] = self.previous_env
        self.temp.cleanup()

    def manifest(self, *, duration_ms: int = 2_000_000) -> Path:
        rows = []
        for index in range(15):
            internal_key = f"opaqueitemkey{index:04d}"
            frozen = self.frozen_bindings[internal_key]
            native = self.native_by_object[frozen["source_object_key"]]
            rows.append(
                {
                    "internal_key": internal_key,
                    "display_key": f"HV-{index+1:03d}",
                    "media_relpath": native["stored_relative_path"],
                    "stratum_key": frozen["stratum_key"],
                    "duration_ms": duration_ms,
                    "batch_number": index // 5 + 1,
                    "source_object_key": frozen["source_object_key"],
                    "media_sha256": native["local_sha256"],
                    "annotation_linkage_sha256": frozen["annotation_linkage_sha256"],
                }
            )
        payload = {
            "schema_version": workflow.VERSION,
            "opaque_key_attestation": True,
            "frozen_selection_digest": "a" * 64,
            "items": rows,
        }
        path = self.root / "packet.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def initialize(self) -> None:
        result = workflow.initialize(
            self.root, self.manifest(), self.skeleton, self.restricted_input, self.native_receipt
        )
        self.assertEqual(result["status"], "initialized")

    def assign_all(self) -> dict[str, str]:
        assignment_path = self.root / workflow.WORKFLOW_DIR / workflow.CODER_ASSIGNMENTS_FILE
        tokens = json.loads(assignment_path.read_text(encoding="utf-8"))["tokens"]
        for slot, token in tokens.items():
            workflow.assign_coder(self.root, slot, token)
        return tokens

    def complete_languages(self, tokens: dict[str, str]) -> None:
        for index in range(15):
            item_key = f"HV-{index+1:03d}"
            for slot in workflow.LANGUAGE_SLOTS:
                workflow.save_language_label(
                    self.root,
                    slot=slot,
                    coder_token=tokens[slot],
                    display_key=item_key,
                    language_code="de",
                    competence="FLUENT",
                    audio_integrity="USABLE",
                    speech_present=True,
                    overlap_present=False,
                    lock=True,
                )
            workflow.adjudicate_language(
                self.root,
                coder_token=tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key=item_key,
                language_code="de",
                audio_integrity="USABLE",
                speech_present=True,
                overlap_present=False,
                reason_code="SYNTHETIC_AGREEMENT",
            )


class QuarantineAndManifestTests(WorkflowFixture):
    def test_finite_sample_nominal_alpha_known_answers(self) -> None:
        self.assertEqual(workflow._nominal_alpha([("A", "A"), ("B", "B")]), 1.0)
        self.assertAlmostEqual(workflow._nominal_alpha([("A", "A"), ("A", "B")]), 0.0)
        self.assertAlmostEqual(workflow._nominal_alpha([("A", "B"), ("A", "B")]), -0.5)

    def test_operational_codebook_and_launcher_token_controls_exist(self) -> None:
        codebook = (ROOT / "docs/childlens_feasibility_v1_2/human_validation_codebook_v1_2.md").read_text(
            encoding="utf-8"
        )
        app_source = (ROOT / "scripts/childlens_human_validation_app_v1_2.py").read_text(encoding="utf-8")
        for required in (
            "Speaker-role decision tree",
            "VISIBLE_MULTIPLE",
            "Pre-adjudication reliability",
            "NONSPEECH",
            "−5 to +5 seconds",
        ):
            self.assertIn(required, codebook)
        self.assertIn("CHILDLENS_V12_UI_NONCE", app_source)
        self.assertIn("E_UI_LAUNCH_TOKEN", app_source)
        self.assertNotIn("st.dataframe", app_source)

    def test_requires_double_opt_in_and_owner_only_policy(self) -> None:
        os.environ.pop(workflow.ROOT_ENV)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_QUARANTINE_ENV_MISSING"):
            workflow.validate_quarantine_root(self.root)
        os.environ[workflow.ROOT_ENV] = str(self.root)
        os.chmod(self.root, 0o755)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_QUARANTINE_NOT_OWNER_ONLY"):
            workflow.validate_quarantine_root(self.root)

    def test_packet_must_stay_inside_quarantine(self) -> None:
        outside = Path(self.temp.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_PACKET_OUTSIDE_QUARANTINE"):
            workflow.initialize(self.root, outside, self.skeleton, self.restricted_input, self.native_receipt)

    def test_rejects_restricted_manifest_fields_and_nonopaque_media_name(self) -> None:
        path = self.manifest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["items"][0]["participant_id"] = "forbidden"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_PACKET_RESTRICTED_FIELD"):
            workflow.initialize(self.root, path, self.skeleton, self.restricted_input, self.native_receipt)

        payload["items"][0].pop("participant_id")
        payload["items"][0]["media_relpath"] = "media/source-name.mp4"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_MEDIA_NAME_NOT_OPAQUE"):
            workflow.initialize(self.root, path, self.skeleton, self.restricted_input, self.native_receipt)

    def test_initialization_is_immutable_and_database_is_mode_600(self) -> None:
        self.initialize()
        second = workflow.initialize(
            self.root, self.root / "packet.json", self.skeleton, self.restricted_input, self.native_receipt
        )
        self.assertEqual(second["status"], "already_initialized")
        database = self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
        self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
        payload = json.loads((self.root / "packet.json").read_text(encoding="utf-8"))
        payload["items"][0]["duration_ms"] += 1
        (self.root / "packet.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_KEY_MEDIA_LINKAGE_MISMATCH"):
            workflow.initialize(
                self.root,
                self.root / "packet.json",
                self.skeleton,
                self.restricted_input,
                self.native_receipt,
            )

    def test_exact_frozen_digest_and_key_set_are_required(self) -> None:
        manifest_path = self.manifest()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["items"][0]["internal_key"] = "opaqueitemkey9999"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_FROZEN_ITEM_SET_MISMATCH"):
            workflow.initialize(
                self.root, manifest_path, self.skeleton, self.restricted_input, self.native_receipt
            )
        skeleton = json.loads(self.skeleton.read_text(encoding="utf-8"))
        skeleton["packet_status"] = "MUTATED"
        self.skeleton.write_text(json.dumps(skeleton), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_FROZEN_SKELETON_DIGEST_MISMATCH"):
            workflow.initialize(
                self.root, manifest_path, self.skeleton, self.restricted_input, self.native_receipt
            )

    def test_key_to_media_and_stratum_permutation_is_rejected(self) -> None:
        manifest_path = self.manifest()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["items"][0]["source_object_key"], payload["items"][1]["source_object_key"] = (
            payload["items"][1]["source_object_key"],
            payload["items"][0]["source_object_key"],
        )
        payload["items"][0]["stratum_key"], payload["items"][1]["stratum_key"] = (
            payload["items"][1]["stratum_key"],
            payload["items"][0]["stratum_key"],
        )
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_KEY_MEDIA_LINKAGE_MISMATCH"):
            workflow.initialize(
                self.root, manifest_path, self.skeleton, self.restricted_input, self.native_receipt
            )

    def test_distinct_source_objects_may_share_byte_deduplicated_media(self) -> None:
        receipt = json.loads(self.native_receipt.read_text(encoding="utf-8"))
        first = receipt["items"][0]
        second = receipt["items"][1]
        second["local_sha256"] = first["local_sha256"]
        second["stored_relative_path"] = first["stored_relative_path"]
        second["transferred_bytes"] = first["transferred_bytes"]
        receipt["restricted_receipt_sha256"] = None
        receipt["restricted_receipt_sha256"] = hashlib.sha256(
            workflow._canonical(receipt)
        ).hexdigest()
        self.native_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        self.native_by_object = {row["object_key"]: row for row in receipt["items"]}

        result = workflow.initialize(
            self.root,
            self.manifest(),
            self.skeleton,
            self.restricted_input,
            self.native_receipt,
        )
        self.assertEqual(result["status"], "initialized")
        database = self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            item_count, distinct_paths = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT media_relpath) FROM items"
            ).fetchone()
        self.assertEqual(item_count, 15)
        self.assertEqual(distinct_paths, 14)

    def test_duplicate_source_object_key_is_rejected_even_with_shared_bytes(self) -> None:
        manifest_path = self.manifest()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["items"][1]["source_object_key"] = payload["items"][0]["source_object_key"]
        payload["items"][1]["media_relpath"] = payload["items"][0]["media_relpath"]
        payload["items"][1]["media_sha256"] = payload["items"][0]["media_sha256"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_SOURCE_OBJECT_DUPLICATE"):
            workflow.initialize(
                self.root,
                manifest_path,
                self.skeleton,
                self.restricted_input,
                self.native_receipt,
            )

    def test_native_transfer_receipt_tamper_is_rejected(self) -> None:
        manifest_path = self.manifest()
        receipt = json.loads(self.native_receipt.read_text(encoding="utf-8"))
        receipt["items"][0]["transferred_bytes"] += 1
        self.native_receipt.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_NATIVE_RECEIPT_INVALID"):
            workflow.initialize(
                self.root, manifest_path, self.skeleton, self.restricted_input, self.native_receipt
            )

    def test_coder_tokens_are_preassigned_unique_and_high_entropy(self) -> None:
        self.initialize()
        assignment_path = self.root / workflow.WORKFLOW_DIR / workflow.CODER_ASSIGNMENTS_FILE
        payload = json.loads(assignment_path.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
        self.assertEqual(set(tokens), workflow.SLOTS)
        self.assertEqual(len(set(tokens.values())), len(workflow.SLOTS))
        self.assertTrue(all(len(token) >= 40 for token in tokens.values()))
        self.assertEqual(stat.S_IMODE(assignment_path.stat().st_mode), 0o600)

    def test_browser_profile_and_cache_are_confined_to_quarantine(self) -> None:
        self.initialize()
        command = launcher.confined_browser_command(
            self.root, Path("/usr/bin/true"), "http://127.0.0.1:8501/?launch_token=synthetic"
        )
        browser_cwd = launcher.browser_working_directory(self.root)
        profile_arg = next(value for value in command if value.startswith("--user-data-dir="))
        cache_arg = next(value for value in command if value.startswith("--disk-cache-dir="))
        download_arg = next(value for value in command if value.startswith("--download-default-directory="))
        for argument in (profile_arg, cache_arg, download_arg):
            self.assertNotIn(str(self.root), argument)
            path = (browser_cwd / argument.split("=", 1)[1]).resolve(strict=True)
            self.assertTrue(workflow._is_relative_to(path, self.root.resolve(strict=True)))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertIn("--disable-sync", command)
        self.assertIn("--disable-background-networking", command)
        preferences = self.root.resolve() / workflow.WORKFLOW_DIR / "browser_profile/Default/Preferences"
        preference_value = json.loads(preferences.read_text(encoding="utf-8"))
        self.assertTrue(
            workflow._is_relative_to(
                Path(preference_value["download"]["default_directory"]).resolve(), self.root.resolve()
            )
        )


class IndependentCodingTests(WorkflowFixture):
    def setUp(self) -> None:
        super().setUp()
        self.initialize()
        self.tokens = self.assign_all()

    def test_independent_passes_cannot_share_a_coder_token(self) -> None:
        self.assertNotEqual(self.tokens["LANGUAGE_A"], self.tokens["LANGUAGE_B"])
        with self.assertRaisesRegex(workflow.WorkflowError, "E_CODER_NOT_AUTHORIZED"):
            workflow.assign_coder(self.root, "LANGUAGE_B", self.tokens["LANGUAGE_A"])

    def test_structural_pass_is_not_human_completion(self) -> None:
        structural = workflow.validate_workflow(self.root)
        readiness = workflow.validate_readiness(self.root)
        self.assertEqual(structural["status"], "STRUCTURAL_PASS")
        self.assertEqual(readiness["status"], "LANGUAGE_PASS_READY")
        self.assertTrue(readiness["runtime_handoff_ready"])
        self.assertFalse(readiness["genuine_human_evidence_complete"])

    def test_timing_phase_cannot_start_before_all_language_routes(self) -> None:
        self.assertEqual(
            workflow.get_task_batch(
                self.root, slot="TIMING_A", coder_token=self.tokens["TIMING_A"]
            ),
            [],
        )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_LANGUAGE_PHASE_INCOMPLETE"):
            workflow.add_timing_segment(
                self.root,
                slot="TIMING_A",
                coder_token=self.tokens["TIMING_A"],
                display_item_key="HV-001",
                display_segment_key="U-0001",
                onset_ms=0,
                offset_ms=1000,
                source_text="synthetic",
                language_code="de",
                speaker_role="NON_CHILD",
            )

    def test_language_autosave_lock_and_adjudication(self) -> None:
        for slot in workflow.LANGUAGE_SLOTS:
            workflow.save_language_label(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_key="HV-001",
                language_code="de",
                competence="FLUENT",
                audio_integrity="USABLE",
                speech_present=True,
                overlap_present=False,
                lock=True,
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_RECORD_LOCKED"):
            workflow.save_language_label(
                self.root,
                slot="LANGUAGE_A",
                coder_token=self.tokens["LANGUAGE_A"],
                display_key="HV-001",
                language_code="en",
                competence="FLUENT",
                audio_integrity="USABLE",
                speech_present=True,
                overlap_present=False,
            )
        tasks = workflow.language_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )
        self.assertEqual(tasks[0]["display_key"], "HV-001")
        workflow.adjudicate_language(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_item_key="HV-001",
            language_code="de",
            audio_integrity="USABLE",
            speech_present=True,
            overlap_present=False,
            reason_code="AGREE",
        )
        self.assertEqual(
            workflow.language_adjudication_tasks(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            ),
            [],
        )

    def test_role_ontology_and_peer_blinding(self) -> None:
        self.complete_languages(self.tokens)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_ROLE_INVALID"):
            workflow.add_timing_segment(
                self.root,
                slot="TIMING_A",
                coder_token=self.tokens["TIMING_A"],
                display_item_key="HV-001",
                display_segment_key="U-0001",
                onset_ms=0,
                offset_ms=1000,
                source_text="restricted test placeholder",
                language_code="de",
                speaker_role="ADULT",
            )
        workflow.add_timing_segment(
            self.root,
            slot="TIMING_A",
            coder_token=self.tokens["TIMING_A"],
            display_item_key="HV-001",
            display_segment_key="U-0001",
            onset_ms=0,
            offset_ms=1000,
            source_text="restricted test placeholder",
            language_code="de",
            speaker_role="NON_CHILD",
        )
        own = workflow.own_timing_segments(
            self.root,
            slot="TIMING_A",
            coder_token=self.tokens["TIMING_A"],
            display_item_key="HV-001",
        )
        self.assertEqual(len(own), 1)
        batch = workflow.get_task_batch(
            self.root, slot="TIMING_B", coder_token=self.tokens["TIMING_B"]
        )
        self.assertNotIn("source_text", batch[0])
        self.assertNotIn("media_relpath", batch[0])

    def test_direct_adjudication_requires_both_item_passes_closed(self) -> None:
        self.complete_languages(self.tokens)
        for slot in workflow.TIMING_SLOTS:
            workflow.add_timing_segment(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_item_key="HV-001",
                display_segment_key="U-0001",
                onset_ms=0,
                offset_ms=1000,
                source_text="synthetic",
                language_code="de",
                speaker_role="NON_CHILD",
                lock=True,
            )
        database = self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT slot, segment_id FROM timing_segments ORDER BY slot"
            ).fetchall()
        source_ids = {slot: segment_id for slot, segment_id in rows}
        with self.assertRaisesRegex(workflow.WorkflowError, "E_TIMING_SOURCES_NOT_CLOSED"):
            workflow.adjudicate_utterance(
                self.root,
                coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key="HV-001",
                display_utterance_key="R-0001",
                source_segment_a=source_ids["TIMING_A"],
                source_segment_b=source_ids["TIMING_B"],
                onset_ms=0,
                offset_ms=1000,
                source_text="synthetic",
                language_code="de",
                speaker_role="NON_CHILD",
                reason_code="TEST",
            )

    def test_item_completion_rejects_undispositioned_locked_sources(self) -> None:
        self.complete_languages(self.tokens)
        for slot in workflow.TIMING_SLOTS:
            for index in range(2):
                workflow.add_timing_segment(
                    self.root,
                    slot=slot,
                    coder_token=self.tokens[slot],
                    display_item_key="HV-001",
                    display_segment_key=f"U-{index+1:04d}",
                    onset_ms=index * 2000,
                    offset_ms=index * 2000 + 1000,
                    source_text="synthetic",
                    language_code="de",
                    speaker_role="NON_CHILD",
                    lock=True,
                )
            workflow.lock_timing_item(
                self.root, slot=slot, coder_token=self.tokens[slot], display_item_key="HV-001"
            )
        task = workflow.timing_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )[0]
        a = next(row for row in task["locked_sources"] if row["slot"] == "TIMING_A")
        b = next(row for row in task["locked_sources"] if row["slot"] == "TIMING_B")
        workflow.adjudicate_utterance(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_item_key="HV-001",
            display_utterance_key="R-0001",
            source_segment_a=a["segment_id"],
            source_segment_b=b["segment_id"],
            onset_ms=0,
            offset_ms=1000,
            source_text="synthetic",
            language_code="de",
            speaker_role="NON_CHILD",
            reason_code="MATCHED",
        )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_TIMING_SOURCES_UNDISPOSITIONED"):
            workflow.complete_timing_adjudication_item(
                self.root,
                coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key="HV-001",
            )

    def test_unmatched_segments_reduce_reliability_coverage(self) -> None:
        self.complete_languages(self.tokens)
        for slot, count in (("TIMING_A", 2), ("TIMING_B", 1)):
            for index in range(count):
                workflow.add_timing_segment(
                    self.root,
                    slot=slot,
                    coder_token=self.tokens[slot],
                    display_item_key="HV-001",
                    display_segment_key=f"U-{index+1:04d}",
                    onset_ms=index * 2000,
                    offset_ms=index * 2000 + 1000,
                    source_text="synthetic",
                    language_code="de",
                    speaker_role="NON_CHILD",
                    lock=True,
                )
            workflow.lock_timing_item(
                self.root, slot=slot, coder_token=self.tokens[slot], display_item_key="HV-001"
            )
        task = workflow.timing_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )[0]
        a_rows = [row for row in task["locked_sources"] if row["slot"] == "TIMING_A"]
        b_row = next(row for row in task["locked_sources"] if row["slot"] == "TIMING_B")
        for index, (a_row, b_id) in enumerate(((a_rows[0], b_row["segment_id"]), (a_rows[1], None)), 1):
            workflow.adjudicate_utterance(
                self.root,
                coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key="HV-001",
                display_utterance_key=f"R-{index:04d}",
                source_segment_a=a_row["segment_id"],
                source_segment_b=b_id,
                onset_ms=(index - 1) * 2000,
                offset_ms=(index - 1) * 2000 + 1000,
                source_text="synthetic",
                language_code="de",
                speaker_role="NON_CHILD",
                reason_code="MATCH_OR_UNMATCHED",
            )
        workflow.complete_timing_adjudication_item(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_item_key="HV-001",
        )
        reliability = workflow.reliability_aggregates(self.root)
        self.assertEqual(reliability["timing_validation_record_count"], 2)
        self.assertEqual(reliability["timing_matched_double_coded_count"], 1)
        self.assertEqual(reliability["timing_unmatched_b_missing_count"], 1)
        self.assertEqual(reliability["timing_matched_coverage_fraction"], 0.5)
        self.assertEqual(reliability["timing_both_edges_within_500ms_fraction"], 0.5)

    def test_referential_inventory_cannot_freeze_before_frozen_minimum(self) -> None:
        with self.assertRaisesRegex(workflow.WorkflowError, "E_FROZEN_MINIMUM_NOT_REACHED"):
            workflow.freeze_referential_sample(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            )

    def test_nonspeech_does_not_satisfy_frozen_stopping_rule(self) -> None:
        self.complete_languages(self.tokens)
        for slot in workflow.TIMING_SLOTS:
            workflow.add_timing_segment(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_item_key="HV-001",
                display_segment_key="U-0001",
                onset_ms=0,
                offset_ms=1_800_000,
                source_text="",
                language_code="de",
                speaker_role="NONSPEECH",
                lock=True,
            )
            workflow.lock_timing_item(
                self.root, slot=slot, coder_token=self.tokens[slot], display_item_key="HV-001"
            )
        task = workflow.timing_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )[0]
        a = next(row for row in task["locked_sources"] if row["slot"] == "TIMING_A")
        b = next(row for row in task["locked_sources"] if row["slot"] == "TIMING_B")
        workflow.adjudicate_utterance(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_item_key="HV-001",
            display_utterance_key="R-0001",
            source_segment_a=a["segment_id"],
            source_segment_b=b["segment_id"],
            onset_ms=0,
            offset_ms=1_800_000,
            source_text="",
            language_code="de",
            speaker_role="NONSPEECH",
            reason_code="NONSPEECH_MATCH",
        )
        status = workflow.progress(self.root)
        self.assertEqual(status["adjudicated_utterances"], 0)
        self.assertFalse(status["frozen_minimum_reached"])


class ReferentialWorkflowTests(WorkflowFixture):
    def setUp(self) -> None:
        super().setUp()
        self.initialize()
        self.tokens = self.assign_all()
        self.complete_languages(self.tokens)
        for slot in workflow.TIMING_SLOTS:
            for index in range(10):
                onset_ms = index * 180_000
                offset_ms = (index + 1) * 180_000
                workflow.add_timing_segment(
                    self.root,
                    slot=slot,
                    coder_token=self.tokens[slot],
                    display_item_key="HV-001",
                    display_segment_key=f"U-{index+1:04d}",
                    onset_ms=onset_ms,
                    offset_ms=offset_ms,
                    source_text="synthetic unit-test text",
                    language_code="de",
                    speaker_role="NON_CHILD",
                    lock=True,
                )
            workflow.lock_timing_item(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_item_key="HV-001",
            )
        tasks = workflow.timing_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )
        source_a = [r for r in tasks[0]["locked_sources"] if r["slot"] == "TIMING_A"]
        source_b = [r for r in tasks[0]["locked_sources"] if r["slot"] == "TIMING_B"]
        for index in range(10):
            onset_ms = 0 if index == 1 else index * 180_000
            offset_ms = 180_000 if index == 1 else (index + 1) * 180_000
            workflow.adjudicate_utterance(
                self.root,
                coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key="HV-001",
                display_utterance_key=f"R-{index+1:04d}",
                source_segment_a=source_a[index]["segment_id"],
                source_segment_b=source_b[index]["segment_id"],
                onset_ms=onset_ms,
                offset_ms=offset_ms,
                source_text="synthetic accepted text",
                language_code="de",
                speaker_role="NON_CHILD",
                reason_code="MATCHED",
            )
        workflow.complete_timing_adjudication_item(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_item_key="HV-001",
        )

    def test_freeze_is_deterministic_and_at_least_twenty_percent(self) -> None:
        self.assertEqual(
            workflow.timing_adjudication_tasks(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            ),
            [],
        )
        receipt = workflow.freeze_referential_sample(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )
        self.assertEqual(receipt["eligible_count"], 10)
        self.assertGreaterEqual(receipt["double_code_fraction"], 0.2)
        second = workflow.freeze_referential_sample(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )
        self.assertEqual(second["double_coded_count"], receipt["double_coded_count"])
        validation = workflow.validate_workflow(self.root)
        self.assertEqual(validation["status"], "STRUCTURAL_PASS")
        self.assertGreaterEqual(validation["referential_double_code_fraction"], 0.2)
        database = self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                """SELECT r.assignment_hash, i.internal_key, u.onset_ms, u.offset_ms,
                          a.display_key source_a, b.display_key source_b
                   FROM referential_assignments r
                   JOIN adjudicated_utterances u ON u.utterance_id=r.utterance_id
                   JOIN items i ON i.item_id=u.item_id
                   LEFT JOIN timing_segments a ON a.segment_id=u.source_segment_a
                   LEFT JOIN timing_segments b ON b.segment_id=u.source_segment_b
                   ORDER BY u.display_key LIMIT 1"""
            ).fetchone()
            collision_rows = connection.execute(
                """SELECT u.onset_ms, u.offset_ms, r.assignment_hash
                   FROM referential_assignments r JOIN adjudicated_utterances u
                     ON u.utterance_id=r.utterance_id
                   WHERE u.display_key IN ('R-0001','R-0002') ORDER BY u.display_key"""
            ).fetchall()
        self.assertEqual(collision_rows[0][:2], collision_rows[1][:2])
        self.assertNotEqual(collision_rows[0][2], collision_rows[1][2])
        expected = hashlib.sha256(
            b"childlens-v1.2-referential-double-code\0"
            + b"a" * 64
            + b"\0"
            + row[1].encode()
            + b"\0"
            + str(row[2]).encode()
            + b"\0"
            + str(row[3]).encode()
            + b"\0"
            + row[4].encode()
            + b"\0"
            + row[5].encode()
        ).hexdigest()
        self.assertEqual(row[0], expected)

    def test_freeze_requires_every_contributing_item_completion(self) -> None:
        database = self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM timing_adjudication_item_locks")
            connection.commit()
        with self.assertRaisesRegex(workflow.WorkflowError, "E_TIMING_ADJUDICATION_NOT_COMPLETED"):
            workflow.freeze_referential_sample(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            )

    def test_invalid_status_and_nonassigned_second_pass_fail_closed(self) -> None:
        workflow.freeze_referential_sample(self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT])
        with self.assertRaisesRegex(workflow.WorkflowError, "E_REFERENTIAL_STATUS_INVALID"):
            workflow.save_referential_label(
                self.root,
                slot="REFERENTIAL_A",
                coder_token=self.tokens["REFERENTIAL_A"],
                display_utterance_key="R-0001",
                status="VISIBLE",
                mention_family="NOUN_OBJECT",
                candidate_band="ONE",
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_REFERENTIAL_COMBINATION_INVALID"):
            workflow.save_referential_label(
                self.root,
                slot="REFERENTIAL_A",
                coder_token=self.tokens["REFERENTIAL_A"],
                display_utterance_key="R-0001",
                status="VISIBLE_SINGLE",
                mention_family="NOUN_OBJECT",
                candidate_band="THREE_PLUS",
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_CENSORING_WITHOUT_BOUNDARY"):
            workflow.save_referential_label(
                self.root,
                slot="REFERENTIAL_A",
                coder_token=self.tokens["REFERENTIAL_A"],
                display_utterance_key="R-0001",
                status="UNDECIDABLE",
                mention_family="NEITHER",
                candidate_band="UNKNOWN",
                boundary_censored=True,
            )
        b_tasks = workflow.get_task_batch(
            self.root, slot="REFERENTIAL_B", coder_token=self.tokens["REFERENTIAL_B"], batch_size=50
        )
        double_keys = {row["display_key"] for row in b_tasks}
        non_double = next(f"R-{index+1:04d}" for index in range(10) if f"R-{index+1:04d}" not in double_keys)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_NOT_DOUBLE_CODED"):
            workflow.save_referential_label(
                self.root,
                slot="REFERENTIAL_B",
                coder_token=self.tokens["REFERENTIAL_B"],
                display_utterance_key=non_double,
                status="NULL_NOT_VISIBLE",
                mention_family="NOUN_OBJECT",
                candidate_band="ZERO",
            )

    def test_fixed_window_lock_and_adjudication_preserve_independent_labels(self) -> None:
        workflow.freeze_referential_sample(self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT])
        double_key = workflow.get_task_batch(
            self.root, slot="REFERENTIAL_B", coder_token=self.tokens["REFERENTIAL_B"], batch_size=50
        )[0]["display_key"]
        context = workflow.referential_task_context(
            self.root,
            slot="REFERENTIAL_A",
            coder_token=self.tokens["REFERENTIAL_A"],
            display_utterance_key=double_key,
        )
        for slot, status in (("REFERENTIAL_A", "VISIBLE_SINGLE"), ("REFERENTIAL_B", "VISIBLE_MULTIPLE")):
            workflow.save_referential_label(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_utterance_key=double_key,
                status=status,
                mention_family="NOUN_OBJECT",
                candidate_band="ONE" if status == "VISIBLE_SINGLE" else "TWO",
                boundary_onset_ms=max(0, context["onset_ms"] - 100),
                boundary_offset_ms=context["onset_ms"] + 100,
                lock=True,
            )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_RECORD_LOCKED"):
            workflow.save_referential_label(
                self.root,
                slot="REFERENTIAL_A",
                coder_token=self.tokens["REFERENTIAL_A"],
                display_utterance_key=double_key,
                status="NULL_NOT_VISIBLE",
                mention_family="NOUN_OBJECT",
                candidate_band="ZERO",
            )
        tasks = workflow.referential_adjudication_tasks(
            self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
        )
        self.assertEqual(tasks[0]["display_key"], double_key)
        workflow.adjudicate_referential(
            self.root,
            coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
            display_utterance_key=double_key,
            status="VISIBLE_MULTIPLE",
            mention_family="NOUN_OBJECT",
            candidate_band="TWO",
            reason_code="AMBIGUITY_RETAINED",
            boundary_onset_ms=max(0, context["onset_ms"] - 100),
            boundary_offset_ms=context["onset_ms"] + 100,
        )
        self.assertFalse(
            workflow.referential_adjudication_tasks(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            )
        )

    def test_frozen_minimum_stops_further_adjudication(self) -> None:
        for slot in workflow.TIMING_SLOTS:
            workflow.add_timing_segment(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_item_key="HV-002",
                display_segment_key="U-9000",
                onset_ms=0,
                offset_ms=1_800_000,
                source_text="synthetic long unit-test segment",
                language_code="de",
                speaker_role="NON_CHILD",
                lock=True,
            )
            workflow.lock_timing_item(
                self.root,
                slot=slot,
                coder_token=self.tokens[slot],
                display_item_key="HV-002",
            )
        item = next(
            row
            for row in workflow.timing_adjudication_tasks(
                self.root, coder_token=self.tokens[workflow.ADJUDICATOR_SLOT]
            )
            if row["display_key"] == "HV-002"
        )
        source_a = next(r for r in item["locked_sources"] if r["slot"] == "TIMING_A")
        source_b = next(r for r in item["locked_sources"] if r["slot"] == "TIMING_B")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_FROZEN_MINIMUM_REACHED_FREEZE_REQUIRED"):
            workflow.adjudicate_utterance(
                self.root,
                coder_token=self.tokens[workflow.ADJUDICATOR_SLOT],
                display_item_key="HV-002",
                display_utterance_key="R-9999",
                source_segment_a=source_a["segment_id"],
                source_segment_b=source_b["segment_id"],
                onset_ms=0,
                offset_ms=1_800_000,
                source_text="synthetic",
                language_code="de",
                speaker_role="NON_CHILD",
                reason_code="TEST",
            )
        self.assertTrue(workflow.progress(self.root)["frozen_minimum_reached"])


if __name__ == "__main__":
    unittest.main()
