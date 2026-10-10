"""Roll section hits up to parent hits (spec 088, US3; query-semantics.md s.6).

Pure: no database access.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from iris_vector_rag.note_search.polarity import section_polarity


@dataclass(frozen=True)
class SectionHit:
    """One matched section of a parent document."""

    id: str
    parent_id: str
    label: str
    text: str
    score: float
    polarity: str
    ordinal: int


@dataclass(frozen=True)
class ParentHit:
    """One parent document, scored by its best matched section."""

    id: str
    text: str
    score: float
    polarity: str
    matched_sections: tuple[SectionHit, ...]


def _section_rank(section: SectionHit) -> tuple[float, int]:
    """Sort key: score descending, then ordinal ascending."""
    return (-section.score, section.ordinal)


def roll_up(hits: Iterable[SectionHit], k: int) -> list[ParentHit]:
    """Group section hits by parent and return the top ``k`` parents.

    Each parent takes the score and text of its best section (highest score,
    ties by lower ordinal) and lists all its sections best first. Parents are
    ordered by score descending, ties by parent id ascending.
    """
    groups: dict[str, list[SectionHit]] = {}
    for hit in hits:
        groups.setdefault(hit.parent_id, []).append(hit)

    parents: list[ParentHit] = []
    for parent_id, sections in groups.items():
        ordered = tuple(sorted(sections, key=_section_rank))
        best = ordered[0]
        parents.append(
            ParentHit(
                id=parent_id,
                text=best.text,
                score=best.score,
                polarity=section_polarity(section.polarity for section in ordered),
                matched_sections=ordered,
            )
        )

    parents.sort(key=lambda p: (-p.score, p.id))
    return parents[:k]
