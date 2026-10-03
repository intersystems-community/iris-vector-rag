# Verify the plausible findings

- type: research (AFK)
- status: closed
- blocked-by: none
- claimed-by: research subagent (2026-10-02, done)
- output: [research/R1-verify-plausible-findings.md](../research/R1-verify-plausible-findings.md)

## Question

For each finding in [findings.md](../findings.md) with status `P` (plausible, not traced),
is it real? Findings: PL-05, ST-13, ST-14, ST-15, ST-16, ST-17, AP-14, AP-15, CM-11,
CM-12, CR-14, VM-13, OE-11, OE-14, OS-05, OS-10, OS-12, OS-13.

For each, answer `CONFIRMED`, `REFUTED` or `UNDECIDABLE WITHOUT IRIS`, with the code
path that proves it (file, line, what runs) and, where cheap, a pure-Python reproduction.
Correct the severity if the evidence says it is wrong.

## Resolution

17 of the 18 plausible findings are real (CM-11 and OE-14 each keep one sub-claim that
cannot be decided without IRIS or `dspy`). ST-14 is refuted: its only caller regenerates
embeddings on any count mismatch, so the branch is unreachable. Severity changes: OS-05 A
to C, AP-14 C to B, CM-12 C to B, ST-15 B to C. OS-12 should be fixed by deleting or
resyncing the vendored copy, not patching it. Three new findings were added to the ledger
(ST-23, OS-22, OS-23) and OS-13 gained the fact that 8 of 14 vendored `sp/*.cls` files
differ from ivg 4.1.1's. Evidence: [research/R1-verify-plausible-findings.md](../research/R1-verify-plausible-findings.md).
