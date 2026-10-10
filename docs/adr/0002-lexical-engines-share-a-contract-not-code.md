# Lexical engines share a contract, not code

ivr and fhir-017 each need lexical search with phrase, negation and section filtering. We
keep two implementations, one in Python over ivr tables and one in ObjectScript over
fhir-017's chunk class, and tie them together with a written query-semantics contract and
shared test vectors.

We chose this over fhir-017 calling into ivr, or ivr becoming the only engine. fhir-017
runs inside the FHIR server and ivr runs outside it, so a code dependency in either
direction would couple their release cycles. The cost is two engines to keep in
agreement; the shared test vectors are what catch drift.
