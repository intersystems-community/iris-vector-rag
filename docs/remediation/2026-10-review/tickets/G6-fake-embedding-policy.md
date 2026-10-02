# Decide the fake-embedding fallback policy

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

## Question

When no real embedding model is available, the code substitutes fake vectors and carries
on: random seeded vectors from a stub that is cached permanently (and bypasses the
configured fallback backends), hash-derived vectors when every backend fails, and a zero
vector when query embedding fails. The result is stored or searched as if real.

Decide: should any of this ever happen silently? Options: raise always; allow a fake
embedder only under an explicit test or dev setting that is impossible to leave on in
production; or degrade with a loud, machine-readable marker on the result. Also decide
what the configured dimension's authority is (OpenAI primary currently reports 384), since
the dimension and the fake vectors are tied.

Findings: OE-01, OE-02, OE-03, CR-09, and the related swallowing in CR-11.

## Resolution

(pending)
