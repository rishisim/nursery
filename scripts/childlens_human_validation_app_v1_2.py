#!/usr/bin/env python3
"""Localhost-only Streamlit UI for the restricted ChildLens v1.2 workflow.

Start this application only through ``launch_childlens_human_validation_v1_2.py``.
The UI deliberately shows opaque display keys, never filesystem paths or source
identifiers, and catches errors without rendering exception details.
"""

from __future__ import annotations

import os
import hmac

import streamlit as st

import childlens_human_validation_v1_2 as workflow


st.set_page_config(page_title="ChildLens blinded validation", page_icon="🔒", layout="wide")


class _CallFailed:
    def __bool__(self) -> bool:
        return False


CALL_FAILED = _CallFailed()


def safe_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except workflow.WorkflowError as exc:
        st.error(f"The workflow rejected this action ({exc.code}).")
    except Exception:
        st.error("The action failed closed (E_UI_INTERNAL). No record was changed.")
    return CALL_FAILED


def root_from_launcher():
    expected_nonce = os.environ.get("CHILDLENS_V12_UI_NONCE", "")
    supplied_nonce = st.query_params.get("launch_token", "")
    if not expected_nonce or not supplied_nonce or not hmac.compare_digest(expected_nonce, supplied_nonce):
        st.error("This browser session was not opened by the restricted launcher (E_UI_LAUNCH_TOKEN).")
        st.stop()
    checked = safe_call(workflow.discover_runtime_root)
    if checked is CALL_FAILED:
        st.stop()
    return checked


def show_media(root: str, slot: str, coder_token: str, display_key: str, *, window=None) -> None:
    media_path = safe_call(
        workflow.media_for_display,
        root,
        slot=slot,
        coder_token=coder_token,
        display_key=display_key,
    )
    if media_path is CALL_FAILED:
        return
    try:
        kwargs = {}
        if window is not None:
            kwargs["start_time"] = max(0, window[0] / 1000)
            kwargs["end_time"] = max(0, window[1] / 1000)
        st.video(str(media_path), **kwargs)
    except Exception:
        st.error("Local media could not be rendered (E_MEDIA_RENDER). The path was not disclosed.")


def slot_login(root: str) -> tuple[str, str]:
    st.sidebar.header("Coder session")
    slot = st.sidebar.selectbox("Assigned pass", sorted(workflow.SLOTS))
    token = st.sidebar.text_input("Private coder token", type="password")
    st.sidebar.caption("Use the coordinator-preassigned token for this pass. A/B tokens belong to distinct humans.")
    if st.sidebar.button("Open assigned pass", type="primary"):
        if safe_call(workflow.assign_coder, root, slot, token) is not CALL_FAILED:
            st.session_state["slot"] = slot
            st.session_state["coder_token"] = token
    active_slot = st.session_state.get("slot")
    active_token = st.session_state.get("coder_token")
    if not active_slot or not active_token:
        st.info("Enter the pass and private token assigned to you. The first valid token permanently claims that pass.")
        st.stop()
    if active_slot != slot:
        st.sidebar.warning(f"Active pass: {active_slot}")
    if st.sidebar.button("End local session"):
        st.session_state.clear()
        st.rerun()
    return active_slot, active_token


def render_progress(root: str) -> None:
    status = safe_call(workflow.progress, root)
    if not status:
        return
    cols = st.columns(4)
    cols[0].metric("Adjudicated utterances", status["adjudicated_utterances"])
    cols[1].metric("Adjudicated speech minutes", status["adjudicated_speech_minutes"])
    cols[2].metric("Referential pass A locked", status["referential_a_locked"])
    cols[3].metric("Referential pass B locked", status["referential_b_locked"])
    if status["frozen_minimum_reached"]:
        st.success("The frozen timing block reached 300 utterances or 30 speech minutes. Reliability and adjudication still apply.")
    else:
        st.caption("Timing block target: stop at the first of 300 adjudicated utterances or 30 adjudicated speech minutes.")


def language_instructions() -> None:
    with st.expander("Decision rules", expanded=False):
        st.write(
            "Use a lowercase ISO 639 code only from audible evidence and your competence. "
            "Use MIXED_OR_CODE_SWITCHED for material language alternation and UNDECIDABLE "
            "for sparse, masked, or competence-limited evidence. Do not infer from metadata or geography."
        )
        st.write("USABLE: required speech is codable. PARTIAL: a meaningful subset survives. UNUSABLE: required judgments cannot be made.")


def timing_instructions() -> None:
    with st.expander("Segmentation and speaker-role rules", expanded=False):
        st.write(
            "Mark one continuous contribution. Split at a clear turn, sustained new contribution, or role change. "
            "Transcribe source-language speech verbatim; never translate or guess."
        )
        st.write(
            "Role order: no linguistic speech → NONSPEECH; simultaneous speakers → OVERLAP; "
            "enrolled child → CHILD; any other person → NON_CHILD; unresolved role → UNCERTAIN."
        )


def referential_instructions() -> None:
    with st.expander("Six-status decision rules", expanded=False):
        st.write(
            "Apply in order: UNUSABLE for technical failure; UNDECIDABLE for unresolved usable evidence; "
            "IRRELEVANT for no eligible concrete object/dynamic action mention; NULL_NOT_VISIBLE when an eligible "
            "mention has no visible candidate; VISIBLE_SINGLE for one candidate; VISIBLE_MULTIPLE for two or more."
        )
        st.write("Use only this ±5-second window. Do not force a referent. Nonvisible statuses cannot receive a boundary.")


def language_page(root: str, slot: str, token: str) -> None:
    st.header("Language and audio integrity — independent pass")
    st.caption("Machine proposals and the peer pass are hidden. Locking is irreversible.")
    language_instructions()
    tasks = safe_call(workflow.get_task_batch, root, slot=slot, coder_token=token, batch_size=20) or []
    if not tasks:
        st.success("No pending items in this pass.")
        return
    key = st.selectbox("Opaque item", [row["display_key"] for row in tasks])
    show_media(root, slot, token, key)
    with st.form("language-form"):
        language = st.text_input("ISO 639 language code, MIXED_OR_CODE_SWITCHED, or UNDECIDABLE")
        competence = st.selectbox("Your language competence", ["NATIVE", "FLUENT", "PROFICIENT", "INSUFFICIENT"])
        integrity = st.selectbox("Audio integrity", sorted(workflow.AUDIO_INTEGRITY_VALUES))
        speech = st.checkbox("Speech present")
        overlap = st.checkbox("Overlapping speech present")
        lock = st.checkbox("Lock this independent judgment")
        submitted = st.form_submit_button("Autosave judgment", type="primary")
    if submitted:
        result = safe_call(
            workflow.save_language_label,
            root,
            slot=slot,
            coder_token=token,
            display_key=key,
            language_code=language,
            competence=competence,
            audio_integrity=integrity,
            speech_present=speech,
            overlap_present=overlap,
            lock=lock,
        )
        if result is not CALL_FAILED:
            st.success("Saved locally." if not lock else "Saved and locked locally.")


def timing_page(root: str, slot: str, token: str) -> None:
    st.header("Utterance timing, source text, and speaker role — independent pass")
    st.caption("Use source-language verbatim text. Do not translate or guess. The peer pass is hidden.")
    timing_instructions()
    tasks = safe_call(workflow.get_task_batch, root, slot=slot, coder_token=token, batch_size=20) or []
    if not tasks:
        st.success("No pending items in this pass.")
        return
    item_key = st.selectbox("Opaque item", [row["display_key"] for row in tasks])
    show_media(root, slot, token, item_key)
    own = safe_call(
        workflow.own_timing_segments,
        root,
        slot=slot,
        coder_token=token,
        display_item_key=item_key,
    ) or []
    if own:
        for row in own:
            with st.container(border=True):
                st.write(
                    f"{row['display_key']} · {row['onset_ms']/1000:.3f}s–{row['offset_ms']/1000:.3f}s · "
                    f"{row['speaker_role']} · {row['language_code']} · {'LOCKED' if row['locked'] else 'DRAFT'}"
                )
                st.text_area(
                    "Own source text",
                    value=row["source_text"],
                    disabled=True,
                    key=f"own-text-{slot}-{item_key}-{row['display_key']}",
                )
    with st.form("timing-form", clear_on_submit=False):
        suggested = f"U-{len(own)+1:04d}"
        segment_key = st.text_input("Opaque utterance key", value=suggested)
        c1, c2 = st.columns(2)
        onset_seconds = c1.number_input("Onset (seconds)", min_value=0.0, step=0.1, format="%.3f")
        offset_seconds = c2.number_input("Offset (seconds)", min_value=0.001, step=0.1, format="%.3f")
        source_text = st.text_area("Verbatim source text or explicit unintelligible marker")
        language = st.text_input("Language code")
        role = st.selectbox("Speaker role", sorted(workflow.ROLE_VALUES))
        lock_segment = st.checkbox("Lock this segment")
        submitted = st.form_submit_button("Autosave segment", type="primary")
    if submitted:
        saved = safe_call(
            workflow.add_timing_segment,
            root,
            slot=slot,
            coder_token=token,
            display_item_key=item_key,
            display_segment_key=segment_key,
            onset_ms=round(onset_seconds * 1000),
            offset_ms=round(offset_seconds * 1000),
            source_text=source_text,
            language_code=language,
            speaker_role=role,
            lock=lock_segment,
        )
        if saved is not CALL_FAILED:
            st.success("Segment saved locally.")
    st.warning("Close the item only after every segment is present. Closing locks all records for this pass.")
    if st.button("Close and lock this item"):
        if safe_call(
            workflow.lock_timing_item,
            root,
            slot=slot,
            coder_token=token,
            display_item_key=item_key,
        ) is not CALL_FAILED:
            st.success("Item locked. The adjudicator can see both passes only after both close.")


def referential_page(root: str, slot: str, token: str) -> None:
    st.header("Referential judgment — independent pass")
    st.caption("Fixed ±5-second window. No machine hypothesis or peer label is displayed. Null and ambiguity are valid outcomes.")
    referential_instructions()
    tasks = safe_call(workflow.get_task_batch, root, slot=slot, coder_token=token, batch_size=20) or []
    if not tasks:
        st.success("No pending items in this assigned pass.")
        return
    utterance_key = st.selectbox("Opaque utterance", [row["display_key"] for row in tasks])
    context = safe_call(
        workflow.referential_task_context,
        root,
        slot=slot,
        coder_token=token,
        display_utterance_key=utterance_key,
    )
    if not context:
        return
    show_media(
        root,
        slot,
        token,
        utterance_key,
        window=(max(0, context["onset_ms"] - 5000), context["onset_ms"] + 5000),
    )
    st.text_area("Accepted source-language utterance (read only)", value=context["source_text"], disabled=True)
    with st.form("referential-form"):
        status = st.selectbox("Referential status", sorted(workflow.REFERENTIAL_VALUES), index=None, placeholder="Choose one")
        family = st.selectbox("Mention family", sorted(workflow.MENTION_FAMILIES), index=None, placeholder="Choose one")
        band = st.selectbox("Candidate count band", sorted(workflow.CANDIDATE_BANDS), index=None, placeholder="Choose one")
        has_boundary = st.checkbox("Visible candidate boundary is annotatable")
        c1, c2 = st.columns(2)
        relative_onset = c1.number_input("Boundary onset relative to utterance onset (seconds)", min_value=-5.0, max_value=5.0, step=0.1)
        relative_offset = c2.number_input("Boundary offset relative to utterance onset (seconds)", min_value=-5.0, max_value=5.0, step=0.1)
        censored = st.checkbox("Boundary is censored by the window")
        lock = st.checkbox("Lock this independent judgment")
        submitted = st.form_submit_button("Autosave judgment", type="primary")
    if submitted:
        boundary_start = context["onset_ms"] + round(relative_onset * 1000) if has_boundary else None
        boundary_end = context["onset_ms"] + round(relative_offset * 1000) if has_boundary else None
        if safe_call(
            workflow.save_referential_label,
            root,
            slot=slot,
            coder_token=token,
            display_utterance_key=utterance_key,
            status=status,
            mention_family=family,
            candidate_band=band,
            boundary_onset_ms=boundary_start,
            boundary_offset_ms=boundary_end,
            boundary_censored=censored,
            lock=lock,
        ) is not CALL_FAILED:
            st.success("Judgment saved locally." if not lock else "Judgment saved and locked locally.")


def language_adjudication(root: str, token: str) -> None:
    st.subheader("Language adjudication")
    tasks = safe_call(workflow.language_adjudication_tasks, root, coder_token=token, batch_size=20) or []
    if not tasks:
        st.info("No language items await adjudication.")
        return
    labels = {row["display_key"]: row for row in tasks}
    key = st.selectbox("Opaque item", list(labels), key="lang-adj-item")
    st.write(labels[key])
    with st.form("lang-adj-form"):
        language = st.text_input("Adjudicated language code")
        integrity = st.selectbox("Adjudicated audio integrity", sorted(workflow.AUDIO_INTEGRITY_VALUES))
        speech = st.checkbox("Adjudicated speech present")
        overlap = st.checkbox("Adjudicated overlap present")
        reason = st.text_input("Adjudication reason code")
        submit = st.form_submit_button("Lock language adjudication")
    if submit and safe_call(
        workflow.adjudicate_language,
        root,
        coder_token=token,
        display_item_key=key,
        language_code=language,
        audio_integrity=integrity,
        speech_present=speech,
        overlap_present=overlap,
        reason_code=reason,
    ) is not CALL_FAILED:
        st.success("Language adjudication locked; both independent records were preserved.")


def timing_adjudication(root: str, token: str) -> None:
    st.subheader("Timing/text/role adjudication")
    tasks = safe_call(workflow.timing_adjudication_tasks, root, coder_token=token, batch_size=10) or []
    if not tasks:
        st.info("No timing items have two closed independent passes.")
        return
    by_key = {row["display_key"]: row for row in tasks}
    item_key = st.selectbox("Opaque item", list(by_key), key="timing-adj-item")
    show_media(root, workflow.ADJUDICATOR_SLOT, token, item_key)
    sources = by_key[item_key]["locked_sources"]
    for row in sources:
        with st.container(border=True):
            st.write(
                f"{row['slot']} · {row['display_key']} · {row['onset_ms']/1000:.3f}s–"
                f"{row['offset_ms']/1000:.3f}s · {row['speaker_role']} · {row['language_code']}"
            )
            st.text_area(
                "Locked source text",
                value=row["source_text"],
                disabled=True,
                key=f"adj-source-{item_key}-{row['slot']}-{row['display_key']}",
            )
    a_rows = [row for row in sources if row["slot"] == "TIMING_A"]
    b_rows = [row for row in sources if row["slot"] == "TIMING_B"]
    a_labels = {"NO_MATCH": None, **{f"{r['display_key']} @ {r['onset_ms']/1000:.3f}s": r["segment_id"] for r in a_rows}}
    b_labels = {"NO_MATCH": None, **{f"{r['display_key']} @ {r['onset_ms']/1000:.3f}s": r["segment_id"] for r in b_rows}}
    next_key = safe_call(workflow.next_utterance_display_key, root, coder_token=token)
    if next_key is CALL_FAILED:
        return
    with st.form("timing-adj-form"):
        accepted_key = st.text_input("Accepted opaque utterance key", value=next_key)
        source_a = st.selectbox("Pass A source", list(a_labels))
        source_b = st.selectbox("Pass B source", list(b_labels))
        c1, c2 = st.columns(2)
        onset = c1.number_input("Accepted onset (seconds)", min_value=0.0, step=0.1, format="%.3f")
        offset = c2.number_input("Accepted offset (seconds)", min_value=0.001, step=0.1, format="%.3f")
        text = st.text_area("Accepted source-language text")
        language = st.text_input("Accepted language code", key="timing-adj-language")
        role = st.selectbox("Accepted speaker role", sorted(workflow.ROLE_VALUES))
        reason = st.text_input("Adjudication reason code", key="timing-adj-reason")
        submit = st.form_submit_button("Lock accepted utterance")
    if submit and safe_call(
        workflow.adjudicate_utterance,
        root,
        coder_token=token,
        display_item_key=item_key,
        display_utterance_key=accepted_key,
        source_segment_a=a_labels[source_a],
        source_segment_b=b_labels[source_b],
        onset_ms=round(onset * 1000),
        offset_ms=round(offset * 1000),
        source_text=text,
        language_code=language,
        speaker_role=role,
        reason_code=reason,
    ) is not CALL_FAILED:
        st.success("Accepted utterance locked; both independent sources were preserved.")
    st.warning("Complete this item only after every accepted or unmatched source has been resolved.")
    if st.button("Complete timing adjudication for this item"):
        if safe_call(
            workflow.complete_timing_adjudication_item,
            root,
            coder_token=token,
            display_item_key=item_key,
        ) is not CALL_FAILED:
            st.success("Item completed and removed from the adjudication queue.")


def referential_adjudication(root: str, token: str) -> None:
    st.subheader("Referential adjudication")
    if st.button("Freeze referential inventory and ≥20% double-code sample"):
        result = safe_call(workflow.freeze_referential_sample, root, coder_token=token)
        if result:
            st.success(
                f"Frozen: {result['eligible_count']} eligible; {result['double_coded_count']} assigned to pass B."
            )
    tasks = safe_call(workflow.referential_adjudication_tasks, root, coder_token=token, batch_size=20) or []
    if not tasks:
        st.info("No independently double-coded referential item awaits adjudication.")
        return
    by_key = {row["display_key"]: row for row in tasks}
    key = st.selectbox("Opaque utterance", list(by_key), key="ref-adj-item")
    st.write(
        {
            field: value
            for field, value in by_key[key].items()
            if field not in {"utterance_onset_ms", "source_text"}
        }
    )
    current = by_key[key]
    show_media(
        root,
        workflow.ADJUDICATOR_SLOT,
        token,
        key,
        window=(max(0, current["utterance_onset_ms"] - 5000), current["utterance_onset_ms"] + 5000),
    )
    st.text_area("Accepted source-language utterance (read only)", value=current["source_text"], disabled=True)
    with st.form("ref-adj-form"):
        status = st.selectbox("Adjudicated status", sorted(workflow.REFERENTIAL_VALUES), index=None, placeholder="Choose one")
        family = st.selectbox("Adjudicated mention family", sorted(workflow.MENTION_FAMILIES), index=None, placeholder="Choose one")
        band = st.selectbox("Adjudicated candidate band", sorted(workflow.CANDIDATE_BANDS), index=None, placeholder="Choose one")
        has_boundary = st.checkbox("Adjudicated boundary present")
        c1, c2 = st.columns(2)
        onset_ms = c1.number_input("Boundary onset (restricted milliseconds)", min_value=0, step=100)
        offset_ms = c2.number_input("Boundary offset (restricted milliseconds)", min_value=0, step=100)
        censored = st.checkbox("Boundary censored")
        reason = st.text_input("Adjudication reason code", key="ref-adj-reason")
        submit = st.form_submit_button("Lock referential adjudication")
    if submit and safe_call(
        workflow.adjudicate_referential,
        root,
        coder_token=token,
        display_utterance_key=key,
        status=status,
        mention_family=family,
        candidate_band=band,
        reason_code=reason,
        boundary_onset_ms=onset_ms if has_boundary else None,
        boundary_offset_ms=offset_ms if has_boundary else None,
        boundary_censored=censored,
    ) is not CALL_FAILED:
        st.success("Referential adjudication locked; both independent labels were preserved.")


def adjudicator_page(root: str, token: str) -> None:
    st.header("Adjudication — peer records unlock only after independent locks")
    tabs = st.tabs(["Language", "Timing/text/role", "Referential"])
    with tabs[0]:
        language_adjudication(root, token)
    with tabs[1]:
        timing_adjudication(root, token)
    with tabs[2]:
        referential_adjudication(root, token)


def main() -> None:
    st.title("🔒 ChildLens blinded human validation v1.2")
    st.caption("Local-only restricted workflow · no learner or causal outcome · human-only reference condition")
    root = root_from_launcher()
    slot, token = slot_login(root)
    render_progress(root)
    if slot in workflow.LANGUAGE_SLOTS:
        language_page(root, slot, token)
    elif slot in workflow.TIMING_SLOTS:
        timing_page(root, slot, token)
    elif slot in workflow.REFERENTIAL_SLOTS:
        referential_page(root, slot, token)
    else:
        adjudicator_page(root, token)
    receipt = safe_call(workflow.validate_readiness, root)
    if receipt:
        st.sidebar.write(f"Workflow state: {receipt['status']}")


if __name__ == "__main__":
    main()
