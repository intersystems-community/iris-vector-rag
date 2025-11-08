# Test Contract: Knowledge Graph Traversal Query Path

**Contract ID**: KG-001
**Requirements**: FR-013, FR-014, FR-015
**Test File**: `tests/contract/test_kg_traversal_contract.py`

## Contract Overview

This contract validates that HybridGraphRAG's knowledge graph traversal path (`method="kg"`) executes multi-hop graph queries correctly, inheriting functionality from GraphRAGPipeline.

## Test Cases

### TC-013: Knowledge Graph Traversal Success (FR-013)
**Given**: HybridGraphRAG pipeline initialized with knowledge graph data
**And**: Graph contains entities and relationships
**When**: Query executed with `method="kg"`
**Then**:
- Pure knowledge graph traversal executes
- Returns list of relevant documents via entity paths
- Metadata indicates `retrieval_method="knowledge_graph"` or similar
- Graph traversal respects configured parameters

**Test Method**: `test_kg_traversal_executes_successfully`

### TC-014: Seed Entity Finding (FR-014)
**Given**: HybridGraphRAG pipeline with knowledge graph
**When**: Query contains entity mentions (e.g., "diabetes", "insulin")
**Then**:
- System identifies seed entities from query
- Seed entities used as starting points for graph traversal
- Retrieved documents linked to seed entities
- Metadata includes seed entity information

**Test Method**: `test_kg_seed_entity_finding`

### TC-015: Multi-Hop Traversal Depth Limits (FR-015)
**Given**: HybridGraphRAG pipeline with knowledge graph
**And**: Configured traversal depth limit (e.g., max_hops=2)
**When**: Query executed with `method="kg"`
**Then**:
- Graph traversal respects depth limit
- Does not traverse beyond max_hops
- Returns documents within specified hop distance
- Execution completes in reasonable time

**Test Method**: `test_kg_multi_hop_depth_limits`

## Assertions

All test cases MUST assert:
1. Result is RAGResponse object
2. Metadata contains `retrieval_method` key
3. Documents retrieved via graph traversal
4. For seed entity tests: metadata includes entity information
5. For depth limit tests: traversal stays within bounds
6. No infinite loops or excessive traversal

## Fixtures Required

- `graphrag_pipeline`: HybridGraphRAG pipeline with KG data
- `config_manager`: For setting traversal depth limits
- Entity data: Test assumes existing entity/relationship data from data loading

## Success Criteria

- All 3 test cases pass
- Tests use @pytest.mark.requires_database
- Execution time <45 seconds (graph queries may be slower)
- Tests validate inherited GraphRAGPipeline functionality
