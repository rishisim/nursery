"""Post-access-only corpus adapter interface with explicit one-corpus binding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .policy import CHILD_CORPORA, PolicyViolation, reject_forbidden_source_markers


@runtime_checkable
class SelectedChildCorpusAdapter(Protocol):
    """Interface to be implemented separately after restricted access exists."""

    @property
    def selected_corpus(self) -> str: ...

    @property
    def corpus_instance_id(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    def iter_timed_utterance_text(self) -> Iterable[Mapping[str, Any]]: ...

    def iter_video_episode_descriptors(self) -> Iterable[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class BoundCorpusAdapter:
    selected_corpus: str
    corpus_instance_id: str
    adapter_version: str
    adapter: SelectedChildCorpusAdapter


def bind_selected_corpus_adapter(
    adapter: SelectedChildCorpusAdapter,
    *,
    selected_corpus: str | None,
    corpus_instance_id: str | None,
    restricted_access_available: bool,
) -> BoundCorpusAdapter:
    """Bind one adapter without iterating it or touching restricted records."""

    if selected_corpus not in CHILD_CORPORA:
        raise PolicyViolation("exactly one child corpus must be selected before an adapter can bind")
    if not restricted_access_available or not corpus_instance_id:
        raise PolicyViolation("child corpus adapter is disabled until restricted access is available")
    descriptor = {
        "selected_corpus": adapter.selected_corpus,
        "corpus_instance_id": adapter.corpus_instance_id,
        "adapter_version": adapter.adapter_version,
    }
    reject_forbidden_source_markers(descriptor)
    if descriptor["selected_corpus"] != selected_corpus:
        raise PolicyViolation("adapter belongs to the other child corpus")
    if descriptor["corpus_instance_id"] != corpus_instance_id:
        raise PolicyViolation("adapter belongs to a different corpus instance")
    if not descriptor["adapter_version"]:
        raise PolicyViolation("adapter version is required")
    return BoundCorpusAdapter(
        selected_corpus=selected_corpus,
        corpus_instance_id=corpus_instance_id,
        adapter_version=descriptor["adapter_version"],
        adapter=adapter,
    )
