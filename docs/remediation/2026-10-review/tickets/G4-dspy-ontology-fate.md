# Decide the fate of `dspy_modules/`, `ontology/` and entity extraction

- type: grilling (HITL)
- status: open
- blocked-by: R3 (Gather support-status facts)
- claimed-by: none

## Question

`dspy` is not installed in the dev environment and no `dspy` extra exists. The ontology
reasoner returns wrong similarity for parent and child, the entity extractor's `Person`
regex matches any two words and never calls an LLM, and `services/entity_extraction.py` and
`batch_entity_processor.py` are largely stubs that report success. Which of these are
supported product? Fix, mark experimental, or delete each?

Findings that move with the decision: OE-05 to OE-09, OE-11 to OE-12, OE-14 to OE-17,
CR-06, CR-10, CR-11, CR-13.

## Resolution

(pending)
