from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "childlens_author_audit_v1_3.py"
SPEC = importlib.util.spec_from_file_location("childlens_author_audit_v1_3", MODULE_PATH)
assert SPEC and SPEC.loader
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class AuthorAuditFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "restricted"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)
        self.previous = os.environ.get(workflow.ROOT_ENV)
        os.environ[workflow.ROOT_ENV] = str(self.root)
        self._private(self.root / workflow.INDEX_SENTINEL, b"never-index")
        policy = {
            "schema_version": workflow.VERSION,
            "retention_deadline": workflow.RETENTION_DEADLINE,
            "owner_only_verified": True,
            "git_exclusion_verified": True,
            "indexing_exclusion_verified": True,
            "backup_exclusion_verified_or_encrypted_local_only": True,
            "local_only": True,
            "signed_agreement_controls_inherited": True,
        }
        self._private(self.root / workflow.POLICY_FILE, canonical(policy))
        durations = [30_000, 90_000] + [60_000] * 13
        items = []
        for index, duration in enumerate(durations):
            payload = f"synthetic-media-{index}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            relative = f"raw_v1_2/{digest}.bin"
            path = self.root / relative
            path.parent.mkdir(mode=0o700, exist_ok=True)
            path.parent.chmod(0o700)
            self._private(path, payload)
            items.append(
                {
                    "audit_item_key": f"AA-{index:032x}",
                    "prediction_join_key": f"PJ-{index:032x}",
                    "media_relpath": relative,
                    "expected_media_sha256": digest,
                    "duration_ms": duration,
                    "segments": [{"start_ms": 1_000, "end_ms": 1_000 + duration}],
                }
            )
        self.packet_value = {
            "schema_version": workflow.PACKET_VERSION,
            "sample_kind": "PRIMARY_AUTHOR_AUDIT",
            "route": workflow.ROUTE,
            "activation": "READY_BEFORE_MODEL_REVEAL",
            "frozen_selection_digest": "1" * 64,
            "sampler_policy_sha256": "2" * 64,
            "official_windows_manifest_sha256": "3" * 64,
            "item_count": 15,
            "opaque_key_attestation": True,
            "selection_independent_of_model_outputs": True,
            "sample_frozen_before_predictions": True,
            "model_predictions_present": False,
            "model_predictions_used_for_selection": False,
            "exact_intervals_restricted": True,
            "total_duration_ms": 900_000,
            "items": items,
        }
        self.packet = self.root / "primary_author_audit_sample.json"
        self._private(self.packet, canonical(self.packet_value))

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop(workflow.ROOT_ENV, None)
        else:
            os.environ[workflow.ROOT_ENV] = self.previous
        workflow._DISCOVERED_RUNTIME_ROOT = None
        self.temporary.cleanup()

    @staticmethod
    def _private(path: Path, payload: bytes) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o600)

    def initialize(self) -> str:
        result = workflow.initialize(self.root, self.packet)
        self.assertEqual(result["status"], "initialized")
        assignment = json.loads(
            (self.root / workflow.WORKFLOW_DIR / workflow.AUTHOR_ASSIGNMENT_FILE).read_text()
        )
        return assignment["token"]

    def prediction_binding(self, *, valid_json: bool = True) -> Path:
        store = self.root / "pseudo_annotations_v1_3/predictions.json"
        payload = (
            canonical({"schema_version": "synthetic", "predictions": [{"join": "synthetic"}]})
            if valid_json
            else b"not-json"
        )
        self._private(store, payload)
        sample_digest = hashlib.sha256(canonical(self.packet_value)).hexdigest()
        binding = {
            "schema_version": workflow.PREDICTION_BINDING_VERSION,
            "sample_digest": sample_digest,
            "prediction_store_relpath": str(store.relative_to(self.root)),
            "prediction_store_sha256": hashlib.sha256(payload).hexdigest(),
            "local_offline_inference_attested": True,
            "network_disabled_during_inference": True,
            "pseudo_labels_are_ground_truth": False,
            "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        }
        path = self.root / "prediction_binding.json"
        self._private(path, canonical(binding))
        return path

    def complete_all_as_nonspeech(self, token: str) -> None:
        for index in range(1, 16):
            key = f"AUDIT-{index:03d}"
            workflow.save_item_label(
                self.root,
                author_token=token,
                display_key=key,
                disposition="NO_LINGUISTIC_SPEECH",
                language_code="UNDECIDABLE",
                language_competence="UNDECIDABLE",
                audio_usability="UNCERTAIN",
            )
            workflow.lock_item(self.root, author_token=token, display_key=key)


class InitializationAndBoundaryTests(AuthorAuditFixture):
    def test_initialization_consumes_only_model_independent_primary_sample(self) -> None:
        token = self.initialize()
        self.assertGreaterEqual(len(token), 40)
        workflow_dir = self.root / workflow.WORKFLOW_DIR
        self.assertFalse((workflow_dir / workflow.PREDICTION_BINDING_SNAPSHOT_FILE).exists())
        database = workflow_dir / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            state, binding, path, digest = connection.execute(
                "SELECT workflow_state,prediction_binding_digest,prediction_store_relpath,prediction_store_sha256 FROM workflow_meta"
            ).fetchone()
        self.assertEqual(state, "AUTHOR_BLIND_OPEN")
        self.assertIsNone(binding)
        self.assertIsNone(path)
        self.assertIsNone(digest)
        self.assertEqual(stat.S_IMODE(database.stat().st_mode), 0o600)
        receipt = workflow.aggregate_receipt(self.root)
        self.assertEqual(receipt["status"], "READY")
        self.assertEqual(receipt["audit_item_count"], 15)
        self.assertEqual(receipt["audit_speech_minutes"], 15.0)
        self.assertEqual(receipt["audit_speech_seconds"], 900)
        self.assertTrue(receipt["predictions_hidden"])
        for field in (
            "autosave_to_quarantine_only",
            "predictions_hidden_before_author_lock",
            "author_record_lock_immutable",
            "sample_independent_of_model_outputs",
            "uncertain_route_available",
            "unusable_route_available",
            "qualification_instruction_present",
        ):
            self.assertIs(receipt[field], True)
        for field in (
            "external_hosting",
            "network_exposure",
            "model_predictions_revealed_before_lock",
            "human_evidence_fabricated",
        ):
            self.assertIs(receipt[field], False)
        self.assertFalse(receipt["inter_human_reliability_available"])

    def test_variable_per_item_duration_is_allowed_but_total_is_frozen(self) -> None:
        self.initialize()
        with sqlite3.connect(self.root / workflow.WORKFLOW_DIR / workflow.DATABASE_FILE) as db:
            values = [row[0] for row in db.execute("SELECT duration_ms FROM audit_items ORDER BY item_id")]
        self.assertEqual(values[:2], [30_000, 90_000])
        self.assertEqual(sum(values), 900_000)

    def test_packet_rejects_model_derived_selection_or_payload(self) -> None:
        for mutation in (
            lambda value: value.__setitem__("model_predictions_used_for_selection", True),
            lambda value: value["items"][0].__setitem__("transcript", "synthetic"),
            lambda value: value.__setitem__("selection_independent_of_model_outputs", False),
        ):
            value = json.loads(json.dumps(self.packet_value))
            mutation(value)
            self._private(self.packet, canonical(value))
            with self.assertRaises(workflow.WorkflowError):
                workflow.initialize(self.root, self.packet)

    def test_packet_and_media_must_be_private_and_confined(self) -> None:
        self.packet.chmod(0o644)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_PACKET_FILE"):
            workflow.initialize(self.root, self.packet)
        self.packet.chmod(0o600)
        self.packet_value["items"][0]["media_relpath"] = "../escape.bin"
        self._private(self.packet, canonical(self.packet_value))
        with self.assertRaisesRegex(workflow.WorkflowError, "E_MEDIA_PATH_NOT_CONTENT_ADDRESSED"):
            workflow.initialize(self.root, self.packet)

    def test_initialization_is_immutable(self) -> None:
        self.initialize()
        self.assertEqual(workflow.initialize(self.root, self.packet)["status"], "already_initialized")
        value = json.loads(json.dumps(self.packet_value))
        value["items"][0]["duration_ms"] += 1
        value["items"][0]["segments"][0]["end_ms"] += 1
        value["items"][1]["duration_ms"] -= 1
        value["items"][1]["segments"][0]["end_ms"] -= 1
        self._private(self.packet, canonical(value))
        with self.assertRaisesRegex(workflow.WorkflowError, "E_INITIALIZATION_IMMUTABLE"):
            workflow.initialize(self.root, self.packet)

    def test_incompatible_existing_schema_fails_closed(self) -> None:
        directory = self.root / workflow.WORKFLOW_DIR
        directory.mkdir(mode=0o700)
        database = directory / workflow.DATABASE_FILE
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE workflow_meta(singleton INTEGER PRIMARY KEY)")
        database.chmod(0o600)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_SCHEMA_INCOMPATIBLE"):
            workflow.initialize(self.root, self.packet)

    def test_launcher_token_requires_process_local_discovery_capability(self) -> None:
        token = self.initialize()
        with self.assertRaisesRegex(workflow.WorkflowError, "E_LAUNCHER_CAPABILITY_REQUIRED"):
            workflow.launcher_author_token(self.root)
        workflow._DISCOVERED_RUNTIME_ROOT = self.root.resolve()
        self.assertEqual(workflow.launcher_author_token(self.root), token)


class AuthorCodingTests(AuthorAuditFixture):
    def setUp(self) -> None:
        super().setUp()
        self.token = self.initialize()

    def test_single_route_and_autosave_resume(self) -> None:
        self.assertEqual(workflow.assign_author(self.root, self.token)["route"], "AUTHOR_AUDIT_A")
        tasks = workflow.get_task_batch(self.root, author_token=self.token, batch_size=3)
        self.assertEqual([row["display_key"] for row in tasks], ["AUDIT-001", "AUDIT-002", "AUDIT-003"])
        workflow.save_item_label(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            disposition="ANNOTATED",
            language_code="de",
            language_competence="FLUENT",
            audio_usability="USABLE",
        )
        workflow.save_utterance(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            display_utterance_key="U-0001",
            segment_index=0,
            onset_ms=100,
            offset_ms=900,
            source_text="synthetic source text",
            speaker_role="NON_CHILD",
            referential_status="VISIBLE_CANDIDATE",
            noun_object_decision="PRESENT",
            verb_action_decision="ABSENT",
        )
        resumed = workflow.own_item_record(
            self.root, author_token=self.token, display_key="AUDIT-001"
        )
        self.assertEqual(resumed["item_label"]["language_code"], "de")
        self.assertEqual(resumed["utterances"][0]["source_text"], "synthetic source text")
        self.assertEqual(workflow.progress(self.root)["draft_item_count"], 1)

    def test_uncertain_and_unusable_routes_do_not_force_labels(self) -> None:
        for index, disposition in ((1, "UNCERTAIN"), (2, "UNUSABLE")):
            key = f"AUDIT-{index:03d}"
            workflow.save_item_label(
                self.root,
                author_token=self.token,
                display_key=key,
                disposition=disposition,
                language_code="UNDECIDABLE",
                language_competence="INSUFFICIENT",
                audio_usability="UNUSABLE" if disposition == "UNUSABLE" else "UNCERTAIN",
            )
            self.assertEqual(workflow.lock_item(self.root, author_token=self.token, display_key=key)["status"], "locked")

    def test_referential_combinations_fail_closed(self) -> None:
        workflow.save_item_label(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            disposition="ANNOTATED",
            language_code="de",
            language_competence="FLUENT",
            audio_usability="USABLE",
        )
        with self.assertRaisesRegex(workflow.WorkflowError, "E_REFERENTIAL_COMBINATION"):
            workflow.save_utterance(
                self.root,
                author_token=self.token,
                display_key="AUDIT-001",
                display_utterance_key="U-0001",
                segment_index=0,
                onset_ms=0,
                offset_ms=500,
                source_text="synthetic",
                speaker_role="NON_CHILD",
                referential_status="IRRELEVANT",
                noun_object_decision="PRESENT",
                verb_action_decision="NOT_APPLICABLE",
            )

    def test_item_lock_is_irreversible(self) -> None:
        workflow.save_item_label(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            disposition="NO_LINGUISTIC_SPEECH",
            language_code="UNDECIDABLE",
            language_competence="UNDECIDABLE",
            audio_usability="UNCERTAIN",
        )
        workflow.lock_item(self.root, author_token=self.token, display_key="AUDIT-001")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_ITEM_LOCKED"):
            workflow.save_item_label(
                self.root,
                author_token=self.token,
                display_key="AUDIT-001",
                disposition="UNUSABLE",
                language_code="UNDECIDABLE",
                language_competence="UNDECIDABLE",
                audio_usability="UNUSABLE",
            )

    def test_mention_span_candidate_band_and_visible_time_are_autosaved(self) -> None:
        workflow.save_item_label(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            disposition="ANNOTATED",
            language_code="de",
            language_competence="FLUENT",
            wer_applicability="APPLICABLE",
            audio_usability="USABLE",
        )
        workflow.save_utterance(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            display_utterance_key="U-0001",
            segment_index=0,
            onset_ms=100,
            offset_ms=900,
            source_text="synthetic object",
            speaker_role="NON_CHILD",
            referential_status="VISIBLE_CANDIDATE",
            noun_object_decision="PRESENT",
            verb_action_decision="ABSENT",
        )
        workflow.save_mention(
            self.root,
            author_token=self.token,
            display_key="AUDIT-001",
            display_utterance_key="U-0001",
            display_mention_key="M-0001",
            mention_family="NOUN_OBJECT",
            mention_start_char=10,
            mention_end_char=16,
            referential_status="VISIBLE_CANDIDATE",
            candidate_count_band="ONE",
            visible_segment_index=0,
            visible_onset_ms=200,
            visible_offset_ms=700,
        )
        own = workflow.own_item_record(
            self.root, author_token=self.token, display_key="AUDIT-001"
        )
        self.assertEqual(own["mentions"][0]["mention_family"], "NOUN_OBJECT")
        self.assertEqual(workflow.lock_item(self.root, author_token=self.token, display_key="AUDIT-001")["status"], "locked")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_AUTHOR_PASS_LOCKED|E_ITEM_LOCKED|E_MENTION_LOCKED"):
            workflow.save_mention(
                self.root,
                author_token=self.token,
                display_key="AUDIT-001",
                display_utterance_key="U-0001",
                display_mention_key="M-0001",
                mention_family="NOUN_OBJECT",
                mention_start_char=10,
                mention_end_char=16,
                referential_status="NULL_NOT_VISIBLE",
                candidate_count_band="ZERO",
            )


class BlindingAndLockTests(AuthorAuditFixture):
    def setUp(self) -> None:
        super().setUp()
        self.token = self.initialize()
        self.binding = self.prediction_binding()

    def test_prediction_binding_and_access_are_impossible_before_author_lock(self) -> None:
        with self.assertRaisesRegex(workflow.WorkflowError, "E_PREDICTION_BIND_REQUIRES_AUTHOR_LOCK"):
            workflow.attach_prediction_store_after_lock(self.root, self.binding)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_PREDICTIONS_BLINDED_UNTIL_AUTHOR_LOCK"):
            workflow.open_predictions_after_lock(self.root, author_token=self.token)
        app_source = (REPO_ROOT / "scripts/childlens_author_audit_app_v1_3.py").read_text()
        self.assertNotIn("open_predictions_after_lock", app_source)
        self.assertNotIn("attach_prediction_store_after_lock", app_source)

    def test_global_lock_requires_all_items_and_blinding_confirmation(self) -> None:
        with self.assertRaisesRegex(workflow.WorkflowError, "E_BLINDING_CONFIRMATION_REQUIRED"):
            workflow.lock_author_pass(self.root, author_token=self.token, confirm_blinded=False)
        with self.assertRaisesRegex(workflow.WorkflowError, "E_ALL_ITEMS_NOT_LOCKED"):
            workflow.lock_author_pass(self.root, author_token=self.token, confirm_blinded=True)

    def test_state_machine_and_post_lock_prediction_join(self) -> None:
        self.complete_all_as_nonspeech(self.token)
        locked = workflow.lock_author_pass(self.root, author_token=self.token, confirm_blinded=True)
        self.assertTrue(locked["human_audit_complete"])
        self.assertEqual(workflow.progress(self.root)["workflow_state"], "AUTHOR_LOCKED")
        with self.assertRaisesRegex(workflow.WorkflowError, "E_AUTHOR_PASS_LOCKED"):
            workflow.save_item_label(
                self.root,
                author_token=self.token,
                display_key="AUDIT-001",
                disposition="UNUSABLE",
                language_code="UNDECIDABLE",
                language_competence="UNDECIDABLE",
                audio_usability="UNUSABLE",
            )
        attached = workflow.attach_prediction_store_after_lock(self.root, self.binding)
        self.assertEqual(attached["workflow_state"], "PREDICTION_JOIN_ENABLED")
        predictions = workflow.open_predictions_after_lock(self.root, author_token=self.token)
        self.assertEqual(predictions["schema_version"], "synthetic")
        receipt = workflow.aggregate_receipt(self.root)
        self.assertFalse(receipt["predictions_hidden"])
        self.assertTrue(receipt["prediction_join_enabled"])
        self.assertTrue(receipt["human_audit_complete"])
        self.assertEqual(workflow.validate_workflow(self.root)["status"], "STRUCTURAL_PASS")

    def test_attached_prediction_binding_is_immutable(self) -> None:
        self.complete_all_as_nonspeech(self.token)
        workflow.lock_author_pass(self.root, author_token=self.token, confirm_blinded=True)
        workflow.attach_prediction_store_after_lock(self.root, self.binding)
        value = json.loads(self.binding.read_text())
        value["network_disabled_during_inference"] = False
        self._private(self.binding, canonical(value))
        with self.assertRaises(workflow.WorkflowError):
            workflow.attach_prediction_store_after_lock(self.root, self.binding)

    def test_unaudited_pseudo_labels_are_not_evaluation_truth(self) -> None:
        receipt = workflow.aggregate_receipt(self.root)
        self.assertEqual(receipt["primary_evaluation_truth"], "SIMULATOR_ORACLE_ONLY")
        self.assertFalse(receipt["unaudited_pseudo_label_primary_evaluation_truth"])
        self.assertEqual(
            receipt["unaudited_pseudo_label_permitted_uses"],
            ["AGGREGATE_CALIBRATION", "CANDIDATE_GENERATION"],
        )


if __name__ == "__main__":
    unittest.main()
