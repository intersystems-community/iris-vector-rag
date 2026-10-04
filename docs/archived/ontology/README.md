# Archived design: the ontology package (`iris_vector_rag.ontology`)

Status: archived. Deprecated in 0.15, removed in 0.16.

`iris_vector_rag/ontology/` was a loader, a reasoner and a general-purpose plugin for
OWL, RDF and SKOS ontologies. Entity extraction stopped consulting it in 0.15, the
`ontology:` configuration section had been a documented no-op since then, and the
reasoner returned wrong similarity for parent and child concepts (findings OE-08, OE-09,
OE-11, OE-17). We deleted it rather than fix it, because no one in this repository used
it.

## Where the code is

- Git tag **`archive/ontology-v1`**: the last commit that contains `iris_vector_rag/ontology/`
  and its tests (`tests/test_ontology_support.py`, `tests/test_ontology_loader.py`).
  Recover it with `git show archive/ontology-v1:iris_vector_rag/ontology/reasoner.py`, or
  `git worktree add ../ontology-v1 archive/ontology-v1`.
- The tag is local until a maintainer pushes it: `git push origin archive/ontology-v1`.

## Contents

| File                                                                                   | What it holds                                               |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [ontology_integration_guide.md](ontology_integration_guide.md)                         | The user guide as it stood, with configuration and examples |
| [graphrag_ontology_integration_analysis.md](graphrag_ontology_integration_analysis.md) | The architecture analysis of ontology use in GraphRAG       |

Nothing in these files is executable or current.
