#!/usr/bin/env python3
"""Loopback-only Streamlit UI for the blinded ChildLens v1.3 author audit.

Start only through ``launch_childlens_author_audit_v1_3.py``.  Model
predictions have no UI route.  Media, source text, timings, and labels remain in
the owner-private quarantine.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

import childlens_author_audit_v1_3 as workflow


st.set_page_config(page_title="ChildLens author audit", page_icon="🔒", layout="wide")


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
    expected = os.environ.get("CHILDLENS_V13_UI_NONCE", "")
    supplied = st.query_params.get("launch_token", "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        st.error("This session was not opened by the restricted launcher (E_UI_LAUNCH_TOKEN).")
        st.stop()
    root = safe_call(workflow.discover_runtime_root)
    if root is CALL_FAILED:
        st.stop()
    return root


def author_login(root):
    st.sidebar.header("Author pass")
    st.sidebar.code(workflow.ROUTE)
    if "author_token" not in st.session_state:
        token = safe_call(workflow.launcher_author_token, root)
        if token is CALL_FAILED:
            st.stop()
        st.session_state["author_token"] = token
    active = st.session_state["author_token"]
    st.sidebar.caption("The dedicated local launcher authenticated AUTHOR_AUDIT_A. No pass token is displayed or transmitted.")
    st.sidebar.caption("Close this dedicated window to end the local session; relaunching resumes autosaved work.")
    return active


def progress_panel(root):
    status = safe_call(workflow.progress, root)
    if status is CALL_FAILED:
        return None
    st.sidebar.divider()
    st.sidebar.metric("Minutes in frozen audit", f"{status['audit_speech_minutes']:.0f}")
    st.sidebar.metric("Items locked", f"{status['locked_item_count']} / {status['audit_item_count']}")
    st.sidebar.metric("Estimated time remaining", f"{status['estimated_remaining_minutes']} min")
    st.sidebar.progress(status["locked_item_count"] / status["audit_item_count"])
    st.sidebar.caption("Autosaves are immediate. Closing and relaunching resumes this pass.")
    return status


def instructions():
    with st.expander("Short decision guide", expanded=True):
        st.markdown(
            """
1. Listen and watch the raw clips. Record the audible language only if you understand it well enough to correct the source-language transcript and assign roles.
2. Add each utterance with clip-relative boundaries, verbatim source text, and one role: NON_CHILD, CHILD, OVERLAP, UNCERTAIN, or NONSPEECH.
3. For the same utterance, choose visible candidate, null/not visible, irrelevant, undecidable, or unusable. Then disposition noun/object and verb/action candidates.
4. Use UNCERTAIN, UNDECIDABLE, or UNUSABLE instead of guessing. Save drafts freely. Lock only after checking the whole item.
"""
        )
        st.warning(
            "Model predictions are inaccessible throughout author coding. Item locks and the final pass lock are irreversible. "
            "Inter-human reliability is unavailable in this single-author prototype; the locked record supports model–human agreement only."
        )


def show_clips(root, token, display_key):
    result = safe_call(
        workflow.media_segments_for_display,
        root,
        author_token=token,
        display_key=display_key,
    )
    if result is CALL_FAILED:
        return []
    media_path, segments = result
    for index, (start_ms, end_ms) in enumerate(segments):
        st.caption(f"Clip {index + 1} · annotate boundaries relative to this clip")
        try:
            st.video(str(media_path), start_time=start_ms / 1000, end_time=end_ms / 1000)
        except Exception:
            st.error("Local media could not be rendered (E_MEDIA_RENDER). No path was displayed.")
            return []
    return segments


def label_form(root, token, display_key, current):
    existing = current.get("item_label") or {}
    dispositions = sorted(workflow.ITEM_DISPOSITIONS)
    competencies = ["NATIVE", "FLUENT", "PROFICIENT", "INSUFFICIENT", "UNDECIDABLE"]
    wer_values = ["APPLICABLE", "NOT_APPLICABLE", "UNDECIDABLE"]
    usability = sorted(workflow.AUDIO_USABILITY_VALUES)
    with st.form("item-label-form"):
        st.subheader("Item-level language and usability")
        disposition = st.selectbox(
            "Item disposition",
            dispositions,
            index=dispositions.index(existing.get("disposition", "ANNOTATED")),
        )
        language = st.text_input(
            "ISO 639 language code, MIXED_OR_CODE_SWITCHED, or UNDECIDABLE",
            value=existing.get("language_code", ""),
        )
        competence = st.selectbox(
            "Your competence for transcript correction",
            competencies,
            index=competencies.index(existing.get("language_competence", "UNDECIDABLE")),
        )
        wer = st.selectbox(
            "Is whitespace word tokenization linguistically valid for WER?",
            wer_values,
            index=wer_values.index(existing.get("wer_applicability", "UNDECIDABLE")),
            help="Choose before predictions are available. CER is still evaluated for usable non-child text.",
        )
        audio = st.selectbox(
            "Audio usability",
            usability,
            index=usability.index(existing.get("audio_usability", "UNCERTAIN")),
        )
        submitted = st.form_submit_button("Autosave item label", type="primary")
    if submitted:
        if safe_call(
            workflow.save_item_label,
            root,
            author_token=token,
            display_key=display_key,
            disposition=disposition,
            language_code=language,
            language_competence=competence,
            audio_usability=audio,
            wer_applicability=wer,
        ) is not CALL_FAILED:
            st.success("Item label autosaved locally.")
            st.rerun()


def utterance_list(root, token, display_key, current):
    rows = current.get("utterances", [])
    if not rows:
        st.caption("No utterances saved yet.")
        return
    st.subheader("Saved utterances")
    for row in rows:
        with st.container(border=True):
            st.write(
                f"{row['display_utterance_key']} · clip {row['segment_index'] + 1} · "
                f"{row['onset_ms']/1000:.2f}–{row['offset_ms']/1000:.2f}s · {row['speaker_role']}"
            )
            st.text_area(
                "Your source-language text",
                value=row["source_text"],
                disabled=True,
                key=f"saved-text-{display_key}-{row['display_utterance_key']}",
            )
            st.caption(
                f"Reference: {row['referential_status']} · noun/object: {row['noun_object_decision']} · "
                f"verb/action: {row['verb_action_decision']}"
            )
            if row.get("uncertain_unusable_reason"):
                st.caption(f"Uncertain/unusable reason: {row['uncertain_unusable_reason']}")
            if not row["locked"] and st.button(
                "Delete this draft",
                key=f"delete-{display_key}-{row['display_utterance_key']}",
            ):
                if safe_call(
                    workflow.delete_draft_utterance,
                    root,
                    author_token=token,
                    display_key=display_key,
                    display_utterance_key=row["display_utterance_key"],
                ) is not CALL_FAILED:
                    st.rerun()


def utterance_form(root, token, display_key, segments, current):
    if not segments:
        return
    rows = current.get("utterances", [])
    roles = sorted(workflow.ROLE_VALUES)
    statuses = sorted(workflow.REFERENTIAL_VALUES)
    candidate_values = sorted(workflow.CANDIDATE_VALUES)
    with st.form("utterance-form", clear_on_submit=False):
        st.subheader("Add or update an utterance")
        utterance_key = st.text_input("Utterance key", value=f"U-{len(rows)+1:04d}")
        clip_number = st.selectbox("Clip", list(range(1, len(segments) + 1)))
        clip_duration = (segments[clip_number - 1][1] - segments[clip_number - 1][0]) / 1000
        c1, c2 = st.columns(2)
        onset = c1.number_input("Onset within clip (seconds)", min_value=0.0, max_value=clip_duration, step=0.1, format="%.3f")
        offset = c2.number_input("Offset within clip (seconds)", min_value=0.001, max_value=clip_duration, value=min(1.0, clip_duration), step=0.1, format="%.3f")
        source_text = st.text_area("Verbatim source-language text or explicit unintelligible marker")
        role = st.selectbox("Source role", roles)
        status = st.selectbox("Coarse referential status", statuses)
        c3, c4 = st.columns(2)
        noun = c3.selectbox("Noun/object candidate", candidate_values)
        verb = c4.selectbox("Verb/action candidate", candidate_values)
        reason = st.text_input("Reason if undecidable or unusable")
        submitted = st.form_submit_button("Autosave utterance", type="primary")
    if submitted:
        if safe_call(
            workflow.save_utterance,
            root,
            author_token=token,
            display_key=display_key,
            display_utterance_key=utterance_key,
            segment_index=clip_number - 1,
            onset_ms=round(onset * 1000),
            offset_ms=round(offset * 1000),
            source_text=source_text,
            speaker_role=role,
            referential_status=status,
            noun_object_decision=noun,
            verb_action_decision=verb,
            uncertain_unusable_reason=reason,
        ) is not CALL_FAILED:
            st.success("Utterance autosaved locally.")
            st.rerun()


def mention_section(root, token, display_key, segments, current):
    utterances = current.get("utterances", [])
    if not utterances:
        return
    st.subheader("Mention-level candidate records")
    st.caption(
        "Add a record for each eligible noun/object or verb/action mention. Character spans refer to your locked source text; visible time bands are clip-relative."
    )
    mentions = current.get("mentions", [])
    for row in mentions:
        with st.container(border=True):
            st.write(
                f"{row['display_mention_key']} · {row['display_utterance_key']} · "
                f"{row['mention_family']} · {row['referential_status']} · {row['candidate_count_band']}"
            )
            st.caption(
                f"Transcript characters {row['mention_start_char']}–{row['mention_end_char']}"
            )
            if not row["locked"] and st.button(
                "Delete this mention draft",
                key=f"delete-mention-{display_key}-{row['display_utterance_key']}-{row['display_mention_key']}",
            ):
                if safe_call(
                    workflow.delete_draft_mention,
                    root,
                    author_token=token,
                    display_key=display_key,
                    display_utterance_key=row["display_utterance_key"],
                    display_mention_key=row["display_mention_key"],
                ) is not CALL_FAILED:
                    st.rerun()
    with st.form("mention-form"):
        utterance_key = st.selectbox(
            "Utterance for this mention",
            [row["display_utterance_key"] for row in utterances],
        )
        mention_key = st.text_input("Mention key", value=f"M-{len(mentions)+1:04d}")
        family = st.selectbox("Mention family", sorted(workflow.MENTION_FAMILIES))
        c1, c2 = st.columns(2)
        start_char = c1.number_input("Mention start character", min_value=0, step=1)
        end_char = c2.number_input("Mention end character (exclusive)", min_value=1, step=1)
        status = st.selectbox(
            "Mention referential status",
            ["VISIBLE_CANDIDATE", "NULL_NOT_VISIBLE", "UNDECIDABLE", "UNUSABLE"],
        )
        count_band = st.selectbox("Candidate count band", sorted(workflow.CANDIDATE_COUNT_BANDS))
        clip_number = st.selectbox("Visible candidate clip (used only when visible)", list(range(1, len(segments) + 1)))
        clip_duration = (segments[clip_number - 1][1] - segments[clip_number - 1][0]) / 1000
        c3, c4 = st.columns(2)
        visible_onset = c3.number_input("Visible band onset in clip (seconds)", min_value=0.0, max_value=clip_duration, step=0.1, format="%.3f")
        visible_offset = c4.number_input("Visible band offset in clip (seconds)", min_value=0.001, max_value=clip_duration, value=min(1.0, clip_duration), step=0.1, format="%.3f")
        reason = st.text_input("Reason if undecidable or unusable")
        submitted = st.form_submit_button("Autosave mention", type="primary")
    if submitted:
        visible = status == "VISIBLE_CANDIDATE"
        if safe_call(
            workflow.save_mention,
            root,
            author_token=token,
            display_key=display_key,
            display_utterance_key=utterance_key,
            display_mention_key=mention_key,
            mention_family=family,
            mention_start_char=int(start_char),
            mention_end_char=int(end_char),
            referential_status=status,
            candidate_count_band=count_band,
            visible_segment_index=clip_number - 1 if visible else None,
            visible_onset_ms=round(visible_onset * 1000) if visible else None,
            visible_offset_ms=round(visible_offset * 1000) if visible else None,
            uncertain_unusable_reason=reason,
        ) is not CALL_FAILED:
            st.success("Mention autosaved locally.")
            st.rerun()


def item_lock(root, token, display_key):
    st.divider()
    confirm = st.checkbox("I reviewed the entire item and understand this lock cannot be undone.")
    if st.button("Irreversibly lock this item", disabled=not confirm):
        if safe_call(workflow.lock_item, root, author_token=token, display_key=display_key) is not CALL_FAILED:
            st.success("Item locked. Model predictions remain hidden.")
            st.rerun()


def final_lock(root, token, status):
    st.header("Finalize blinded author pass")
    st.write("All 15 audit items are locked. This final lock makes the complete author record immutable before any model comparison can be opened.")
    blinded = st.checkbox("I completed the pass from raw audio/video without seeing model predictions.")
    typed = st.text_input("Type LOCK AUTHOR PASS")
    if st.button("Irreversibly lock AUTHOR_AUDIT_A", type="primary", disabled=not (blinded and typed == "LOCK AUTHOR PASS")):
        if safe_call(workflow.lock_author_pass, root, author_token=token, confirm_blinded=True) is not CALL_FAILED:
            st.success("AUTHOR_AUDIT_A is locked. You may close this dedicated window.")
            st.rerun()


def main():
    root = root_from_launcher()
    st.title("ChildLens blinded author audit")
    st.caption("A frozen 15-minute feasibility audit · local only · model predictions hidden")
    token = author_login(root)
    status = progress_panel(root)
    if status is None:
        st.stop()
    instructions()
    if status["human_audit_complete"]:
        st.success("AUTHOR_AUDIT_A is irreversibly locked. No further author edits are possible.")
        return
    if status["remaining_item_count"] == 0:
        final_lock(root, token, status)
        return
    tasks = safe_call(workflow.get_task_batch, root, author_token=token, batch_size=5)
    if tasks is CALL_FAILED or not tasks:
        st.warning("No editable item is available.")
        return
    display_key = st.selectbox("Audit item", [row["display_key"] for row in tasks])
    segments = show_clips(root, token, display_key)
    current = safe_call(workflow.own_item_record, root, author_token=token, display_key=display_key)
    if current is CALL_FAILED:
        return
    label_form(root, token, display_key, current)
    utterance_list(root, token, display_key, current)
    utterance_form(root, token, display_key, segments, current)
    mention_section(root, token, display_key, segments, current)
    item_lock(root, token, display_key)


main()
