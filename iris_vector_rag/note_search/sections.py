"""Split a note into labelled sections (spec 088, contracts/query-semantics.md s.2)."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

_HEADING = re.compile(r"^[ \t]*([A-Z](?:[A-Z ]*[A-Z])?):", re.MULTILINE)


@dataclass(frozen=True)
class Heading:
    """A heading in a note; [start, end) covers its characters, colon included."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Section:
    """A labelled section; text is note[start:end], stripped of surrounding space."""

    ordinal: int
    label: str
    text: str
    start: int
    end: int


HeadingDetector = Callable[[str], list[Heading]]


def normalise_label(raw: str) -> str:
    """Lower-case a label, drop a trailing colon and collapse whitespace."""
    text = raw.strip()
    if text.endswith(":"):
        text = text[:-1]
    return " ".join(text.lower().split())


def default_heading_detector(text: str) -> list[Heading]:
    """Report each line that opens with an upper-case phrase ending in a colon."""
    return [
        Heading(start=m.start(1), end=m.end(), label=m.group(1))
        for m in _HEADING.finditer(text)
    ]


def _strip_span(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Return the offsets of text[lo:hi] with surrounding whitespace removed."""
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def split_sections(
    text: str,
    detector: HeadingDetector,
    label_map: Mapping[str, str] | None = None,
) -> list[Section]:
    """Split a note at the headings the detector reports, in order of appearance.

    A section runs from its heading to the next heading (or the end of the note),
    stripped of whitespace. Text before the first heading becomes a 'rest' section.
    Sections with no characters are dropped; ordinals count the rest.
    """
    headings = sorted(detector(text), key=lambda h: h.start)
    bounds: list[tuple[str, int, int]] = [
        ("rest", 0, headings[0].start if headings else len(text))
    ]
    for i, heading in enumerate(headings):
        stop = headings[i + 1].start if i + 1 < len(headings) else len(text)
        label = normalise_label(heading.label)
        if label_map and label in label_map:
            label = label_map[label]
        bounds.append((label, heading.end, stop))

    sections: list[Section] = []
    for label, lo, hi in bounds:
        start, end = _strip_span(text, lo, hi)
        if start >= end:
            continue
        sections.append(
            Section(
                ordinal=len(sections),
                label=label,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
    return sections
