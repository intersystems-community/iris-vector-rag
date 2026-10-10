"""Roll-up (spec 088, US3; contracts/query-semantics.md section 6). Pure: no database."""

from iris_vector_rag.note_search.rollup import SectionHit, roll_up


def hit(id, parent, label, score, polarity="affirmed", ordinal=0, text="t"):
    return SectionHit(
        id=id,
        parent_id=parent,
        label=label,
        text=text,
        score=score,
        polarity=polarity,
        ordinal=ordinal,
    )


def test_one_hit_per_parent():
    out = roll_up(
        [
            hit("A#0", "A", "hpi", 0.5),
            hit("A#1", "A", "plan", 0.3),
            hit("B#0", "B", "hpi", 0.4),
        ],
        k=10,
    )
    assert [p.id for p in out] == ["A", "B"]


def test_score_and_text_are_the_best_sections():
    (p,) = roll_up(
        [
            hit("A#0", "A", "hpi", 0.2, text="low"),
            hit("A#1", "A", "plan", 0.9, text="best"),
        ],
        k=5,
    )
    assert (p.score, p.text) == (0.9, "best")


def test_matched_sections_are_listed_best_first_ties_by_ordinal():
    (p,) = roll_up(
        [
            hit("A#2", "A", "c", 0.5, ordinal=2),
            hit("A#0", "A", "a", 0.5, ordinal=0),
            hit("A#1", "A", "b", 0.9, ordinal=1),
        ],
        k=5,
    )
    assert [s.id for s in p.matched_sections] == ["A#1", "A#0", "A#2"]


def test_a_parent_with_more_sections_does_not_outrank_a_better_best_section():
    out = roll_up(
        [
            hit("A#0", "A", "x", 0.3),
            hit("A#1", "A", "y", 0.3),
            hit("A#2", "A", "z", 0.3),
            hit("B#0", "B", "x", 0.8),
        ],
        k=1,
    )
    assert [p.id for p in out] == ["B"]


def test_ties_between_parents_break_by_parent_id():
    out = roll_up([hit("B#0", "B", "x", 0.5), hit("A#0", "A", "x", 0.5)], k=10)
    assert [p.id for p in out] == ["A", "B"]


def test_at_most_k_parents():
    hits = [hit(f"P{i}#0", f"P{i}", "x", 1 - i / 10) for i in range(5)]
    assert len(roll_up(hits, k=3)) == 3


def test_polarity_is_affirmed_if_any_matched_section_is():
    (p,) = roll_up(
        [
            hit("A#0", "A", "x", 0.9, polarity="negated"),
            hit("A#1", "A", "y", 0.1, polarity="affirmed"),
        ],
        k=5,
    )
    assert p.polarity == "affirmed"
    assert [s.polarity for s in p.matched_sections] == ["negated", "affirmed"]


def test_polarity_is_negated_when_nothing_is_affirmed():
    (p,) = roll_up(
        [
            hit("A#0", "A", "x", 0.9, polarity="unknown"),
            hit("A#1", "A", "y", 0.1, polarity="negated"),
        ],
        k=5,
    )
    assert p.polarity == "negated"


def test_no_hits_no_parents():
    assert roll_up([], k=5) == []
