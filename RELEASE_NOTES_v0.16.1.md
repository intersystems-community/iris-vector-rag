# Release Notes: iris-vector-rag v0.16.1

Released 2026-10-10 as `v0.16.1`.

A speed release for the note-search judging added in 0.16.0. Nothing changes in what it
returns: the polarity, experiencer and temporality answers are identical, checked on 120
synthetic sections of 800 to 7,000 characters and by tests that compare every new routine
with the one it replaces on fuzzed text.

Judging a section takes about 0.5 ms on average (1.4 ms for 7,000 characters), down from
2.1 ms in 0.16.0 and 46 ms in 0.15.0. The cost was tokenising the same text several times
and stripping punctuation one word at a time. See the CHANGELOG for the details.

Upgrading: nothing to do. Callers that vendor the pure note-search modules should
re-vendor `tokens.py`, `phrase.py`, `polarity.py` and `context.py`.
