# Release Notes: iris-vector-rag v0.16.0

Released 2026-10-10 as `v0.16.0`.

0.16.0 adds search over long documents such as clinical notes: cut a document into
labelled sections, find an exact phrase and say whether each mention is affirmed or
negated, and answer with the whole document. Nothing in it changes how the existing
pipelines behave, and nothing is validated on clinical text yet: the rules shipped here
are references, and the accepted evidence is a synthetic fixture with known answers.

Requires `iris-vector-graph>=4.2.0,<5`. Python 3.11 and 3.12.

## What is new

- **Sections.** `split_sections` cuts a document with a heading detector you supply. The
  default detector (`UPPER CASE:` at line start) is a demo; `AllowlistHeadingDetector`
  finds only headings you list, in the casings you accept, which is what real notes need.
  `SectionStore` writes one chunk-shaped row per section and replaces a document's
  sections in one transaction.
- **`retrieval="phrase"`.** Exact phrase (adjacent words in order, no stemming, stop words
  kept), optionally limited to one section with `QueryOptions(section=...)`, applied in SQL
  before ranking. Each candidate is re-scanned and dropped if the phrase is not there:
  IRIS 2026.1 returns reverse-order false positives for two-word phrases at the start of a
  text. Hits roll up to one result per parent document, scored by its best section.
- **Polarity, experiencer, temporality.** Every mention is judged `affirmed`, `negated` or
  `unknown`; who it is about (`patient` or `other`); and when (`current`, `historical`,
  `hypothetical`). All three are pluggable detectors with small reference rules. Negated
  and family-history mentions are reported, never dropped.
- **Concept postings.** `ConceptIndex` stores which concepts each section mentions, with
  polarity, from a dictionary you supply (no clinical vocabulary ships), and
  `retrieval="concept"` searches by concept id. No model or embedding is used.
- **Opt-in `section_label` column.** `ensure_section_column` adds a bitmap-indexed column
  for deployments that filter on selective labels; results are identical with or without.
- **For embedded Python callers.** `note_search.api` (`judge_section`, `judge_sections`,
  `classify`, ...) and the pure modules import only the standard library and each other, so
  they can be copied out as a package. A test proves it.

## Upgrading

Nothing to do. No existing table, setting or default changes. The phrase and concept modes
and the extra column are used only when you ask for them.

## Known limitations

- The reference detectors are small rule lists. They miss negation across long lists, do
  not resolve who "she" refers to, and have no accuracy figure on any clinical corpus.
- Searching by concept reads every posting of the requested concepts and groups them in
  Python; it will not scale to millions of postings without a rewrite.
- Phrase search ranks every hit of a common phrase before applying the page limit, so a
  common phrase over hundreds of thousands of sections takes seconds. A benchmark is in
  `specs/088-note-search-first-slice/research-section-label-benchmark.md`.
- Nothing here claims throughput or latency at Baystate scale.

Design, semantics and shared test vectors for other implementations:
`specs/088-note-search-first-slice/`, `specs/089-section-label-column/`,
`specs/091-concept-postings/`.
