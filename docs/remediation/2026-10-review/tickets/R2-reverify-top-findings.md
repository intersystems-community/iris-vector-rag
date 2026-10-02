# Re-verify the top-severity findings

- type: research (AFK)
- status: closed
- blocked-by: none
- claimed-by: research subagent (2026-10-02, done)
- output: [research/R2-reverify-top-findings.md](../research/R2-reverify-top-findings.md)

## Question

The reviewers' top findings were traced by one reader each. Independently re-verify every
finding in [findings.md](../findings.md) with severity `A` and status `C`: ST-01, ST-03,
ST-04, AP-01, AP-04, AP-05, AP-06, AP-08, CM-01, CM-02, CM-03, CM-08, CM-09, CR-01, CR-06,
CR-09, VM-01, VM-02, VM-10, OE-01, OE-02, OE-03, OS-02, OS-14.

For each: `CONFIRMED`, `REFUTED` or `PARTLY TRUE`, the code path (file, line, what
runs), whether a real caller can reach it today (not only in principle), and a corrected
one-line finding and severity if the original was off. Note anything a finding leaves out
that changes how it should be fixed.

## Resolution

All 24 severity-A findings re-verified: 15 confirmed, 9 partly true, none refuted. Thirteen
drop out of class A (to B: ST-01, ST-03, AP-04, AP-08, CM-03, CM-09, CR-06, VM-01, VM-02,
OE-03; to C: CM-08, VM-10, OS-14), mostly because nothing calls the code today. Still A:
ST-04, AP-01, AP-05, AP-06, CM-01, CM-02, CR-01, CR-09, OE-01, OE-02, OS-02. The whole
`api/` group (AP-01, AP-04, AP-05, AP-06) is dormant behind AP-01, and `IRISStorage`
(ST-01, ST-03) and `MCPBridge` (VM-01) have no production caller, which feeds the fate
tickets. Corrections and additions are in the ledger (ST-04 chunks migrator also drops;
second injection site in ST-03; CR-01 fix must hydrate content by id; new AP-23, AP-24,
CM-16). Evidence: [research/R2-reverify-top-findings.md](../research/R2-reverify-top-findings.md).
