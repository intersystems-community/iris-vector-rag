"""Split a note into labelled sections (spec 088, contracts/query-semantics.md s.2)."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
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


_CASINGS = ("upper", "title", "any_case")
_COLON_MODES = ("optional", "required", "forbidden")


def _label_pattern(label: str, colon: str, standalone: bool) -> re.Pattern[str]:
    """Compile a pattern for one label, applied to a single line (no newline)."""
    words = label.split()
    if not words:
        raise ValueError("labels must not be blank")
    body = r"[ \t]+".join(re.escape(w) for w in words)
    eol = r"[ \t]*\Z"
    if colon == "required":
        tail = ":"
    elif colon == "forbidden":
        tail = r"(?!:)" + (f"(?={eol})" if standalone else "")
    elif standalone:
        tail = f"(?::|(?={eol}))"
    else:
        tail = ":?"
    # The name must end at a space, tab, colon or end of line (no 'PLANNING' for 'Plan').
    return re.compile(rf"[ \t]*(?P<name>{body})(?![^ \t:]){tail}", re.IGNORECASE)


class AllowlistHeadingDetector:
    """Report headings that match a known list of labels, in the casings accepted.

    A heading begins its line (leading spaces and tabs allowed); words of a label may
    be split by spaces or tabs. ``accept`` picks casings: "upper" (the label's upper
    case form), "title" (each word title-cased) or "any_case". A colon after the label
    is part of the heading; without one, a standalone heading must be the whole rest of
    its line. The longest label wins at a position. Results are in text order.
    """

    def __init__(
        self,
        labels: Iterable[str],
        *,
        accept: Iterable[str] = ("upper", "title"),
        colon: str = "optional",
        standalone: bool = True,
    ) -> None:
        names = list(labels)
        if not names:
            raise ValueError("AllowlistHeadingDetector needs at least one label")
        casings = tuple(accept)
        if not casings or any(c not in _CASINGS for c in casings):
            raise ValueError(f"accept must be a non-empty subset of {_CASINGS}")
        if colon not in _COLON_MODES:
            raise ValueError(f"colon must be one of {_COLON_MODES}")
        self._accept = casings
        ordered = sorted(names, key=lambda n: len(" ".join(n.split())), reverse=True)
        self._patterns = [
            (name, _label_pattern(name, colon, standalone)) for name in ordered
        ]

    def __call__(self, text: str) -> list[Heading]:
        headings: list[Heading] = []
        offset = 0
        for raw in text.split("\n"):
            heading = self._heading_in(raw.removesuffix("\r"), offset)
            if heading is not None:
                headings.append(heading)
            offset += len(raw) + 1
        return headings

    def _heading_in(self, line: str, offset: int) -> Heading | None:
        for label, pattern in self._patterns:
            m = pattern.match(line)
            if m and self._casing_ok(label, m.group("name")):
                return Heading(offset + m.start("name"), offset + m.end(), label)
        return None

    def _casing_ok(self, label: str, found: str) -> bool:
        words = found.split()
        for casing in self._accept:
            if casing == "any_case":
                return True
            if casing == "upper" and words == label.upper().split():
                return True
            if casing == "title" and words == [w.title() for w in label.split()]:
                return True
        return False


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
