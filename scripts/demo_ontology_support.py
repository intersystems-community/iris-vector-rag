#!/usr/bin/env python3
"""
General-Purpose Ontology Support Demonstration for GraphRAG

This script demonstrates the enhanced general-purpose ontology capabilities including:
- Universal ontology plugin that works with ANY domain
- Auto-detection of domain from ontology files
- Ontology-aware entity extraction and enrichment
- Query expansion using ontology concepts
- Reasoning-based relationship inference
- Domain-agnostic knowledge representation

Usage:
    python scripts/demo_ontology_support.py [--ontology PATH] [--enable-reasoning]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.ontology.plugins import (
    GeneralOntologyPlugin,
    create_plugin_from_config,
    get_ontology_plugin,
)
from iris_vector_rag.ontology.reasoner import OntologyReasoner, QueryExpander
from iris_vector_rag.pipelines.graphrag_merged import GraphRAGPipeline
from iris_vector_rag.services.entity_extraction import OntologyAwareEntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[Document]:
    """Create sample documents for different domains."""
    return [
        # Medical domain document
        Document(
            id="med_001",
            page_content="""
            Patient John Smith, age 45, presents with type 2 diabetes mellitus and hypertension.
            He has been experiencing frequent urination, excessive thirst, and fatigue.
            Current medications include metformin 1000mg twice daily and lisinopril 10mg daily.
            Blood pressure is elevated at 150/95 mmHg. HbA1c level is 8.2%.
            Recommended treatment includes lifestyle modifications and possible addition of an ACE inhibitor.
            The pancreas is not producing sufficient insulin to regulate blood sugar levels effectively.
            Patient should monitor blood glucose regularly and follow up in 3 months.
            """,
            metadata={"domain": "medical", "document_type": "clinical_note"},
        ),
        # IT Systems domain document
        Document(
            id="it_001",
            page_content="""
            Server IRIS-PROD-01 is experiencing high CPU utilization and memory consumption.
            The TrakCare application database connection pool is reaching maximum capacity.
            Network monitoring shows increased latency between the application server and IRIS database.
            System administrators need to investigate potential performance bottlenecks.
            The Apache web server is generating error logs indicating timeout issues.
            Load balancer configuration may need adjustment to distribute traffic more effectively.
            Database query optimization and index tuning should be considered.
            """,
            metadata={"domain": "it_systems", "document_type": "incident_report"},
        ),
        # Software Development domain document
        Document(
            id="dev_001",
            page_content="""
            The user authentication feature has a critical bug in the login validation logic.
            The API endpoint /auth/login is returning 500 internal server error responses.
            Code review identified an issue with the password hashing algorithm implementation.
            Unit tests are failing for the UserService class methods.
            The development team needs to deploy a hotfix to the production environment.
            Git repository shows conflicting changes in the authentication module.
            Continuous integration pipeline is blocked due to test failures.
            """,
            metadata={"domain": "software_development", "document_type": "bug_report"},
        ),
        # Legal domain document
        Document(
            id="legal_001",
            page_content="""
            Contract review for vendor agreement VND-2024-001 requires legal approval.
            The terms include data processing clauses under GDPR compliance requirements.
            Intellectual property rights need clarification regarding software licensing.
            Liability limitations and indemnification provisions require attorney review.
            The contract includes arbitration clauses for dispute resolution.
            Confidentiality agreements protect proprietary business information.
            Termination conditions specify 30-day notice period requirements.
            """,
            metadata={"domain": "legal", "document_type": "contract_review"},
        ),
        # Financial domain document
        Document(
            id="finance_001",
            page_content="""
            Quarterly financial report shows revenue growth of 15% year-over-year.
            Operating expenses increased due to expanded marketing campaigns.
            Cash flow analysis indicates positive trends in accounts receivable.
            Investment portfolio performance shows diversified asset allocation.
            Budget variance analysis reveals cost overruns in technology infrastructure.
            Tax compliance requires quarterly estimated payment calculations.
            Audit findings recommend improved internal controls for expense reporting.
            """,
            metadata={"domain": "financial", "document_type": "financial_report"},
        ),
    ]


def create_sample_ontology_config(ontology_path: str = None) -> Dict[str, Any]:
    """Create sample ontology configuration for demonstration."""
    config = {
        "enabled": True,
        "type": "general",
        "auto_detect_domain": True,
        "sources": [],
    }

    if ontology_path and Path(ontology_path).exists():
        config["sources"].append({"type": "owl", "path": ontology_path})
    else:
        # Add example configurations for different domains
        config["sources"] = [
            {
                "type": "example",
                "domain": "medical",
                "concepts": [
                    {
                        "id": "diabetes",
                        "label": "Diabetes Mellitus",
                        "synonyms": ["diabetes", "DM"],
                    },
                    {
                        "id": "hypertension",
                        "label": "Hypertension",
                        "synonyms": ["high blood pressure", "HTN"],
                    },
                    {
                        "id": "medication",
                        "label": "Medication",
                        "synonyms": ["drug", "medicine"],
                    },
                    {"id": "metformin", "label": "Metformin", "parent": "medication"},
                ],
            },
            {
                "type": "example",
                "domain": "technology",
                "concepts": [
                    {
                        "id": "server",
                        "label": "Server",
                        "synonyms": ["host", "machine"],
                    },
                    {
                        "id": "database",
                        "label": "Database",
                        "synonyms": ["DB", "data store"],
                    },
                    {
                        "id": "cpu",
                        "label": "CPU",
                        "synonyms": ["processor", "central processing unit"],
                    },
                    {"id": "memory", "label": "Memory", "synonyms": ["RAM", "storage"]},
                ],
            },
        ]

    return config


def demonstrate_general_ontology_plugin(ontology_config: Dict[str, Any]):
    """Demonstrate general-purpose ontology plugin capabilities."""
    print("\n" + "=" * 80)
    print("GENERAL-PURPOSE ONTOLOGY PLUGIN DEMONSTRATION")
    print("=" * 80)

    try:
        # Create general ontology plugin
        plugin = GeneralOntologyPlugin()

        print(f"Plugin Class: {plugin.__class__.__name__}")
        print(f"Plugin Type: General-purpose, domain-agnostic")

        # Load ontology from configuration
        if ontology_config.get("sources"):
            print(
                f"\nLoading ontology from {len(ontology_config['sources'])} sources..."
            )

            for source in ontology_config["sources"]:
                if source["type"] == "example":
                    # Load example concepts for demonstration
                    plugin._load_example_concepts(source)
                elif source["type"] == "owl" and "path" in source:
                    plugin.load_ontology_from_file(source["path"])

            print(f"✓ Ontology loaded successfully")
            print(f"✓ Concepts loaded: {len(plugin.hierarchy.concepts)}")

            # Auto-detect domain
            if ontology_config.get("auto_detect_domain", True):
                detected_domain = plugin.auto_detect_domain(
                    {"concepts": plugin.hierarchy.concepts, "metadata": {}}
                )
                print(f"✓ Auto-detected domain: {detected_domain}")
                plugin.domain = detected_domain

            # Show sample concepts
            print(f"\nSample concepts:")
            for i, (concept_id, concept) in enumerate(
                plugin.hierarchy.concepts.items()
            ):
                if i >= 10:
                    break
                synonyms = list(concept.get_all_synonyms())[:3]
                print(f"  - {concept.label} ({concept_id})")
                if synonyms:
                    print(f"    Synonyms: {', '.join(synonyms)}")
                if hasattr(concept, "parent") and concept.parent:
                    parent = plugin.hierarchy.concepts.get(concept.parent)
                    if parent:
                        print(f"    Parent: {parent.label}")

            # Show auto-generated entity mappings
            if plugin.entity_mappings:
                print(f"\nAuto-generated entity mappings:")
                for entity_type, concepts in list(plugin.entity_mappings.items())[:5]:
                    print(f"  - {entity_type}: {concepts[:3]}")

        else:
            print("No ontology sources configured - using minimal example")

    except Exception as e:
        logger.error(f"Error demonstrating general ontology plugin: {e}")


def demonstrate_entity_extraction(
    documents: List[Document], ontology_config: Dict[str, Any]
):
    """Demonstrate ontology-aware entity extraction with general plugin."""
    print("\n" + "=" * 80)
    print("GENERAL ONTOLOGY-AWARE ENTITY EXTRACTION")
    print("=" * 80)

    try:
        # Initialize configuration with ontology enabled
        config_manager = ConfigurationManager()
        config_manager._config["ontology"] = ontology_config

        # Initialize entity extractor with general ontology
        extractor = OntologyAwareEntityExtractor(config_manager=config_manager)

        for doc in documents:
            print(
                f"\n--- DOCUMENT: {doc.id} ({doc.metadata.get('domain', 'unknown')}) ---"
            )
            print(f"Content preview: {doc.page_content[:100]}...")

            # Extract entities using general ontology
            entities = extractor.extract_with_ontology(doc.page_content, doc)

            print(f"Extracted {len(entities)} entities:")
            for entity in entities[:10]:  # Show first 10
                print(
                    f"  - '{entity.text}' ({entity.entity_type}) [confidence: {entity.confidence:.2f}]"
                )

                # Show ontology metadata if available
                if "domain" in entity.metadata:
                    print(f"    Auto-detected domain: {entity.metadata['domain']}")
                if "concept_id" in entity.metadata:
                    print(f"    Matched concept: {entity.metadata['concept_id']}")
                if "inferred_relations" in entity.metadata:
                    relations = entity.metadata["inferred_relations"][:3]
                    print(f"    Related concepts: {[r['label'] for r in relations]}")

            if len(entities) > 10:
                print(f"  ... and {len(entities) - 10} more entities")

    except Exception as e:
        logger.error(f"Error in general entity extraction demo: {e}")


def demonstrate_reasoning_capabilities(ontology_config: Dict[str, Any]):
    """Demonstrate general ontology reasoning features."""
    print("\n" + "=" * 80)
    print("GENERAL ONTOLOGY REASONING DEMONSTRATION")
    print("=" * 80)

    try:
        # Create general ontology plugin
        plugin = GeneralOntologyPlugin()

        # Load ontology
        if ontology_config.get("sources"):
            for source in ontology_config["sources"]:
                if source["type"] == "example":
                    plugin._load_example_concepts(source)

        if plugin.hierarchy.concepts:
            reasoner = OntologyReasoner(plugin.hierarchy)

            print("--- GENERAL ONTOLOGY REASONING ---")

            # Find a concept to reason about
            test_concept = None
            for concept in plugin.hierarchy.concepts.values():
                if concept.label.lower() in ["diabetes", "server", "medication"]:
                    test_concept = concept
                    break

            if not test_concept:
                test_concept = list(plugin.hierarchy.concepts.values())[0]

            if test_concept:
                print(f"Reasoning about: {test_concept.label}")

                # Get hierarchical relationships
                ancestors = plugin.hierarchy.get_ancestors(test_concept.id, max_depth=3)
                descendants = plugin.hierarchy.get_descendants(
                    test_concept.id, max_depth=2
                )

                print(f"Ancestor concepts: {len(ancestors)}")
                for ancestor_id in ancestors[:5]:
                    ancestor = plugin.hierarchy.concepts.get(ancestor_id)
                    if ancestor:
                        print(f"  - {ancestor.label}")

                print(f"Descendant concepts: {len(descendants)}")
                for desc_id in descendants[:5]:
                    descendant = plugin.hierarchy.concepts.get(desc_id)
                    if descendant:
                        print(f"  - {descendant.label}")

            # Test query expansion
            print("\n--- QUERY EXPANSION ---")
            query_expander = QueryExpander(plugin.hierarchy)

            test_queries = [
                "diabetes treatment options",
                "server performance issues",
                "authentication problems",
                "contract review process",
            ]

            for query in test_queries:
                try:
                    expanded = query_expander.expand_query(query, strategy="synonyms")
                    print(f"Original: {query}")
                    print(f"Expanded: {expanded.expanded_query}")
                    print(f"Added terms: {expanded.expansion_terms}")
                    print(f"Confidence: {expanded.confidence:.2f}\n")
                except Exception as e:
                    print(f"Query expansion failed for '{query}': {e}\n")
        else:
            print("No concepts loaded for reasoning demonstration")

    except Exception as e:
        logger.error(f"Error in general reasoning demo: {e}")


def demonstrate_graphrag_integration(
    documents: List[Document], ontology_config: Dict[str, Any]
):
    """Demonstrate GraphRAG pipeline with general ontology support."""
    print("\n" + "=" * 80)
    print("GRAPHRAG PIPELINE WITH GENERAL ONTOLOGY INTEGRATION")
    print("=" * 80)

    try:
        # Initialize configuration with general ontology enabled
        config_manager = ConfigurationManager()
        config_manager._config["ontology"] = ontology_config
        config_manager._config["pipelines"] = {
            "graphrag": {
                "ontology_integration": {
                    "query_expansion": True,
                    "entity_enrichment": True,
                    "relationship_inference": True,
                }
            }
        }

        connection_manager = ConnectionManager()

        # Initialize GraphRAG pipeline
        pipeline = GraphRAGPipeline(
            connection_manager=connection_manager, config_manager=config_manager
        )

        print("Loading documents with general ontology-enhanced extraction...")

        # Load documents (this will trigger ontology-aware entity extraction)
        try:
            pipeline.load_documents("", documents=documents, generate_embeddings=False)
            print("✓ Documents loaded successfully")
        except Exception as e:
            print(f"⚠ Document loading had issues: {e}")
            # Continue with demo using existing data

        # Test ontology-enhanced queries for different domains
        test_queries = [
            "What medications are used for diabetes?",  # Medical
            "How to resolve server performance issues?",  # IT
            "What are the steps to fix authentication bugs?",  # Software
            "How to review legal contracts?",  # Legal
            "What are the financial reporting requirements?",  # Financial
        ]

        print(f"\nTesting {len(test_queries)} general ontology-enhanced queries...")

        for i, query in enumerate(test_queries, 1):
            print(f"\n--- QUERY {i}: {query} ---")

            try:
                # Test regular query first
                basic_result = pipeline.query(query, top_k=3, generate_answer=False)
                print(
                    f"Basic query found {basic_result['metadata']['num_retrieved']} documents"
                )

                # Show ontology insights if available
                if hasattr(pipeline, "_get_ontology_insights"):
                    insights = pipeline._get_ontology_insights(query, basic_result)

                    if insights.get("detected_domain"):
                        print(f"Auto-detected domain: {insights['detected_domain']}")

                    if insights.get("inferred_concepts"):
                        concepts = [
                            c["label"] for c in insights["inferred_concepts"][:3]
                        ]
                        print(f"Inferred concepts: {concepts}")

                    if insights.get("semantic_relationships"):
                        relations = insights["semantic_relationships"][:2]
                        for rel in relations:
                            print(
                                f"Relationship: {rel['source']} -> {rel['target']} ({rel['relationship']})"
                            )

                else:
                    print("Ontology insights not available")

            except Exception as e:
                logger.error(f"Query failed: {e}")

    except Exception as e:
        logger.error(f"Error in GraphRAG integration demo: {e}")


def run_performance_tests(ontology_config: Dict[str, Any]):
    """Run basic performance tests for general ontology operations."""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTING - GENERAL ONTOLOGY")
    print("=" * 80)

    import time

    try:
        # Test plugin loading performance
        start_time = time.time()

        plugin = GeneralOntologyPlugin()

        # Load ontology sources
        if ontology_config.get("sources"):
            for source in ontology_config["sources"]:
                if source["type"] == "example":
                    plugin._load_example_concepts(source)

        load_time = time.time() - start_time

        total_concepts = len(plugin.hierarchy.concepts)

        print(f"General plugin loading performance:")
        print(f"  - Loaded general ontology plugin in {load_time:.2f}s")
        print(f"  - Total concepts loaded: {total_concepts}")
        print(
            f"  - Loading rate: {total_concepts / max(load_time, 0.001):.0f} concepts/second"
        )

        # Test auto-detection performance
        if total_concepts > 0:
            start_time = time.time()
            domain = plugin.auto_detect_domain(
                {"concepts": plugin.hierarchy.concepts, "metadata": {}}
            )
            detection_time = time.time() - start_time

            print(f"\nDomain auto-detection performance:")
            print(f"  - Auto-detected domain '{domain}' in {detection_time*1000:.1f}ms")

        # Test reasoning performance
        if total_concepts > 0:
            start_time = time.time()
            reasoner = OntologyReasoner(plugin.hierarchy)

            # Test subsumption queries
            test_count = min(10, total_concepts)
            for i, concept in enumerate(plugin.hierarchy.concepts.values()):
                if i >= test_count:
                    break
                ancestors = plugin.hierarchy.get_ancestors(concept.id, max_depth=2)

            reasoning_time = time.time() - start_time

            print(f"\nReasoning performance:")
            print(f"  - {test_count} subsumption queries in {reasoning_time:.3f}s")
            print(f"  - Average query time: {reasoning_time/test_count*1000:.1f}ms")

    except Exception as e:
        logger.error(f"Error in performance testing: {e}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate general-purpose ontology support capabilities"
    )
    parser.add_argument(
        "--ontology", type=str, help="Path to ontology file (OWL/RDF/SKOS/TTL/N3)"
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable reasoning demonstrations",
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--skip-graphrag", action="store_true", help="Skip GraphRAG integration demo"
    )
    parser.add_argument(
        "--domain", type=str, help="Filter documents by domain for focused testing"
    )

    args = parser.parse_args()

    print("IRIS RAG Framework - General-Purpose Ontology Support Demonstration")
    print("=" * 80)

    # Create sample documents
    documents = create_sample_documents()

    # Filter documents by domain if specified
    if args.domain:
        documents = [
            doc for doc in documents if doc.metadata.get("domain") == args.domain
        ]
        print(f"Focusing on {args.domain} domain ({len(documents)} documents)")

    # Create ontology configuration
    ontology_config = create_sample_ontology_config(args.ontology)

    if args.ontology:
        print(f"Using ontology file: {args.ontology}")
    else:
        print("Using example ontology data for demonstration")

    try:
        # Run demonstrations
        demonstrate_general_ontology_plugin(ontology_config)

        demonstrate_entity_extraction(documents, ontology_config)

        if args.enable_reasoning:
            demonstrate_reasoning_capabilities(ontology_config)

        if not args.skip_graphrag:
            demonstrate_graphrag_integration(documents, ontology_config)

        if args.performance:
            run_performance_tests(ontology_config)

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\nKey Features Demonstrated:")
        print("✓ General-purpose ontology plugin (works with ANY domain)")
        print("✓ Auto-detection of domain from ontology content")
        print("✓ Ontology-aware entity extraction and enrichment")
        print("✓ Dynamic entity mapping generation")
        print("✓ Domain-agnostic concept hierarchy navigation")
        if args.enable_reasoning:
            print("✓ Query expansion using ontology concepts")
            print("✓ General subsumption and reasoning capabilities")
        if not args.skip_graphrag:
            print("✓ GraphRAG pipeline integration with general ontology support")
        if args.performance:
            print("✓ Performance benchmarking and optimization insights")

        print("\nSupported Ontology Formats:")
        print("- OWL (Web Ontology Language)")
        print("- RDF (Resource Description Framework)")
        print("- SKOS (Simple Knowledge Organization System)")
        print("- TTL (Turtle)")
        print("- N3 (Notation3)")
        print("- XML (with ontology structure)")

        print("\nNext steps:")
        print("- Load your domain-specific ontology files")
        print("- Configure custom domain definitions if needed")
        print("- Enable ontology support in production configuration")
        print("- The system will automatically adapt to your domain vocabulary")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
