# Section label in metadata versus a bitmap-indexed column

**Measured** 2026-10-10 on `iris-vector-rag-iris` (IRIS 2026.1, a shared dev container, one
run per size, median of five timed runs after one warm-up). Script and raw results are
outside the repo (`/tmp/ns_bench/`); the table below is the result.

Setup: synthetic section rows of about 700 characters, eight common labels and one rare
label (0.5 percent of rows), an `%iFind.Index.Basic` index on the text. Table A keeps the
label in the `metadata` JSON and filters with the library's `LIKE` predicate. Table B has a
`section_label VARCHAR(100)` column with a bitmap index and filters with `=`. Query: the
quoted phrase `"chest pain"`, top 20 by the index rank.

| Rows    | common label A / B (ms) | rare label A / B (ms) | no filter A / B (ms) | rare phrase + common label A / B (ms) |
| ------- | ----------------------- | --------------------- | -------------------- | ------------------------------------- |
| 20,000  | 382 / 313               | 409 / 143             | 384 / 390            | 77 / 81                               |
| 100,000 | 1,656 / 1,115           | 1,679 / 539           | 1,648 / 1,653        | 273 / 275                             |
| 300,000 | 4,800 / 3,224           | 4,903 / 1,574         | 4,822 / 4,863        | 692 / 693                             |

Phrase matches for "chest pain": 12.6k, 63k and 189k at the three sizes. iFind build took
57 to 197 s per table (the 120 s at 20,000 rows is contention, not a trend); the bitmap
index built in under a second.

## What it says

- The `LIKE` filter on metadata costs about 1.5 times the column for a common label and
  about 3 times for a rare label at 300,000 rows, and a rare-label query is no faster than
  no filter at all: the metadata design cannot use the label to skip work.
- Most of the time, in both designs, is ranking the phrase hits (189,000 of them at
  300,000 rows). A selective phrase is cheap (692 ms either way); a common phrase is not.
  A bitmap column helps only when the label is selective.
- At 28 million section rows (the Baystate ED scale) a common phrase would be far
  slower than these numbers in both designs. Nothing here supports a latency claim; the
  numbers only compare the two designs at 20,000 to 300,000 rows.

## Consequence for 0.16.0

Keep the label in `metadata` by default (no schema change), and add an opt-in, separate
`section_label` column with a bitmap index for deployments that filter on selective labels.
The writer fills both; the reader uses the column when the table has it. Test-first, on
real IRIS. Reducing the cost of ranking common phrases (a cap on rank candidates, or a
filter-first plan) is a separate question to measure.
