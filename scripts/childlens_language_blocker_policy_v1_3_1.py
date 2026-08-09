#!/usr/bin/env python3
"""Pure policy logic for the v1.3.1 language-resource disposition."""

from __future__ import annotations

from typing import Iterable


TERMINAL_STATE = "CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES"
QUALIFIED_COMPETENCIES = frozenset({"NATIVE", "FLUENT", "PROFICIENT"})


def author_is_language_qualified(
    dominant_language: str,
    understood_languages: Iterable[str],
    competence: str,
) -> bool:
    normalized = {value.strip().casefold() for value in understood_languages}
    return dominant_language.strip().casefold() in normalized and competence in QUALIFIED_COMPETENCIES


def validate_evidence_uses(*, translation_as_gold: bool, model_model_as_human: bool) -> None:
    if translation_as_gold:
        raise ValueError("E_TRANSLATION_AS_GOLD_PROHIBITED")
    if model_model_as_human:
        raise ValueError("E_MODEL_MODEL_AS_HUMAN_PROHIBITED")


def disposition(
    *,
    german_dominance_diagnostic: bool,
    qualified_author_available: bool,
    translation_as_gold: bool = False,
    model_model_as_human: bool = False,
) -> str:
    validate_evidence_uses(
        translation_as_gold=translation_as_gold,
        model_model_as_human=model_model_as_human,
    )
    if german_dominance_diagnostic and not qualified_author_available:
        return TERMINAL_STATE
    raise ValueError("E_DISPOSITION_NOT_ESTABLISHED")
