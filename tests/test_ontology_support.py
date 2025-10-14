"""
Comprehensive Test Suite for General-Purpose Ontology Support

Tests all ontology-related functionality including:
- Ontology data models and structures
- Multi-format ontology loading (OWL/RDF/SKOS/TTL/N3/XML)
- General-purpose ontology plugin
- Auto-detection of domains from ontology content
- Reasoning engine capabilities
- Entity extraction with ontology awareness
- GraphRAG pipeline integration
- Performance and scalability
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document, Entity
from iris_rag.ontology.loader import OntologyLoader

# Import the ontology components
from iris_rag.ontology.models import (
    Concept,
    ConceptHierarchy,
    InferenceRule,
    OntologyRelationship,
    SemanticMapping,
)
from iris_rag.ontology.plugins import (
    GeneralOntologyPlugin,
    create_plugin_from_config,
    get_ontology_plugin,
)
from iris_rag.ontology.reasoner import OntologyReasoner, QueryExpander
from iris_rag.services.entity_extraction import OntologyAwareEntityExtractor


class TestOntologyModels:
    """Test ontology data models and basic operations."""

    def test_concept_creation(self):
        """Test Concept model creation and methods."""
        concept = Concept(
            id="test_001",
            label="Test Disease",
            description="A test disease concept",
            synonyms=["test illness", "test condition"],
            external_ids={"UMLS": "C1234567"},
            metadata={"domain": "medical", "confidence": 0.9},
        )

        assert concept.id == "test_001"
        assert concept.label == "Test Disease"
        assert "test illness" in concept.synonyms
        assert "UMLS" in concept.external_ids

        # Test synonym methods
        all_synonyms = concept.get_all_synonyms()
        assert "test illness" in all_synonyms
        assert "test condition" in all_synonyms

        # Test metadata access
        assert concept.get_metadata("domain") == "medical"
        assert concept.get_metadata("unknown_key") is None

    def test_ontology_relationship(self):
        """Test OntologyRelationship model."""
        relationship = OntologyRelationship(
            id="rel_001",
            source_concept_id="concept_1",
            target_concept_id="concept_2",
            relationship_type="is_a",
            confidence=0.95,
            metadata={"inferred": False},
        )

        assert relationship.source_concept_id == "concept_1"
        assert relationship.relationship_type == "is_a"
        assert relationship.confidence == 0.95

    def test_concept_hierarchy(self):
        """Test ConceptHierarchy operations."""
        hierarchy = ConceptHierarchy()

        # Add concepts
        parent = Concept("parent", "Parent Concept", "A parent concept")
        child = Concept("child", "Child Concept", "A child concept")
        grandchild = Concept("grandchild", "Grandchild Concept", "A grandchild concept")

        hierarchy.add_concept(parent)
        hierarchy.add_concept(child)
        hierarchy.add_concept(grandchild)

        # Add relationships
        hierarchy.add_relationship("child", "parent", "is_a")
        hierarchy.add_relationship("grandchild", "child", "is_a")

        # Test hierarchy navigation
        ancestors = hierarchy.get_ancestors("grandchild", max_depth=5)
        assert "child" in ancestors
        assert "parent" in ancestors

        descendants = hierarchy.get_descendants("parent", max_depth=5)
        assert "child" in descendants
        assert "grandchild" in descendants

        # Test direct relationships
        parents = hierarchy.get_parents("child")
        assert "parent" in parents

        children = hierarchy.get_children("parent")
        assert "child" in children


class TestOntologyLoader:
    """Test ontology loading from different formats."""

    def test_loader_initialization(self):
        """Test OntologyLoader initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = OntologyLoader(temp_dir)
            assert loader.ontology_path == temp_dir
            assert isinstance(loader.concepts, dict)
            assert isinstance(loader.relationships, dict)

    def test_owl_format_support(self):
        """Test OWL format loading capabilities."""
        loader = OntologyLoader("/tmp")

        # Create mock OWL content
        mock_owl_data = {
            "concepts": [
                {"id": "owl_001", "label": "OWL Concept", "type": "Class"},
                {"id": "owl_002", "label": "Another Concept", "type": "Class"},
            ],
            "relationships": [
                {"source": "owl_002", "target": "owl_001", "type": "subClassOf"}
            ],
        }

        # Test OWL loading (mocked)
        result = loader._process_owl_data(mock_owl_data)
        assert "concepts" in result
        assert "relationships" in result

    def test_rdf_format_support(self):
        """Test RDF format loading capabilities."""
        loader = OntologyLoader("/tmp")

        # Test RDF processing capabilities
        mock_rdf_data = {
            "triples": [
                {
                    "subject": "rdf_001",
                    "predicate": "rdfs:label",
                    "object": "RDF Concept",
                },
                {
                    "subject": "rdf_002",
                    "predicate": "rdfs:subClassOf",
                    "object": "rdf_001",
                },
            ]
        }

        result = loader._process_rdf_data(mock_rdf_data)
        assert isinstance(result, dict)

    def test_skos_format_support(self):
        """Test SKOS vocabulary loading."""
        loader = OntologyLoader("/tmp")

        mock_skos_data = {
            "concepts": [
                {
                    "id": "skos_001",
                    "prefLabel": "SKOS Concept",
                    "altLabel": ["Alternative Label"],
                },
                {
                    "id": "skos_002",
                    "prefLabel": "Related Concept",
                    "related": ["skos_001"],
                },
            ]
        }

        result = loader._process_skos_data(mock_skos_data)
        assert isinstance(result, dict)


class TestGeneralOntologyPlugin:
    """Test general-purpose ontology plugin."""

    def test_plugin_initialization(self):
        """Test GeneralOntologyPlugin initialization."""
        plugin = GeneralOntologyPlugin()

        assert plugin.__class__.__name__ == "GeneralOntologyPlugin"
        assert isinstance(plugin.hierarchy, ConceptHierarchy)
        assert isinstance(plugin.entity_mappings, dict)
        assert plugin.domain is None  # Not set until ontology loaded

    def test_auto_detect_domain(self):
        """Test automatic domain detection."""
        plugin = GeneralOntologyPlugin()

        # Test medical domain detection
        medical_data = {
            "concepts": {
                "diabetes": Concept("diabetes", "Diabetes", "Diabetes mellitus"),
                "medication": Concept("medication", "Medication", "Medical drug"),
                "patient": Concept("patient", "Patient", "Medical patient"),
            },
            "metadata": {},
        }

        detected_domain = plugin.auto_detect_domain(medical_data)
        assert detected_domain == "medical"

        # Test technology domain detection
        tech_data = {
            "concepts": {
                "server": Concept("server", "Server", "Computer server"),
                "database": Concept("database", "Database", "Data storage system"),
                "network": Concept("network", "Network", "Computer network"),
            },
            "metadata": {},
        }

        detected_domain = plugin.auto_detect_domain(tech_data)
        assert detected_domain == "technology"

        # Test unknown domain
        unknown_data = {
            "concepts": {
                "concept1": Concept("concept1", "Unknown Concept", "Unknown concept")
            },
            "metadata": {},
        }

        detected_domain = plugin.auto_detect_domain(unknown_data)
        assert detected_domain == "general"

    def test_auto_generate_mappings(self):
        """Test automatic entity mapping generation."""
        plugin = GeneralOntologyPlugin()

        # Create test concepts
        concepts = [
            Concept("disease", "Disease", "Medical condition"),
            Concept("diabetes", "Diabetes", "Diabetes mellitus"),
            Concept("server", "Server", "Computer server"),
            Concept("database", "Database", "Data storage"),
        ]

        mappings = plugin.auto_generate_mappings(concepts)

        assert isinstance(mappings, dict)
        assert len(mappings) > 0

        # Should create mappings based on concept labels
        found_disease = any("disease" in key.lower() for key in mappings.keys())
        found_server = any("server" in key.lower() for key in mappings.keys())

        assert found_disease or found_server  # At least one mapping created

    def test_load_example_concepts(self):
        """Test loading example concepts for demonstration."""
        plugin = GeneralOntologyPlugin()

        # Medical example
        medical_source = {
            "type": "example",
            "domain": "medical",
            "concepts": [
                {"id": "diabetes", "label": "Diabetes", "synonyms": ["DM"]},
                {
                    "id": "medication",
                    "label": "Medication",
                    "synonyms": ["drug", "medicine"],
                },
                {"id": "metformin", "label": "Metformin", "parent": "medication"},
            ],
        }

        plugin._load_example_concepts(medical_source)

        assert len(plugin.hierarchy.concepts) == 3
        assert "diabetes" in plugin.hierarchy.concepts
        assert "medication" in plugin.hierarchy.concepts
        assert "metformin" in plugin.hierarchy.concepts

        # Test parent-child relationship
        parents = plugin.hierarchy.get_parents("metformin")
        assert "medication" in parents

    def test_load_custom_domain(self):
        """Test loading custom domain definition."""
        plugin = GeneralOntologyPlugin()

        custom_domain = {
            "name": "legal",
            "description": "Legal domain ontology",
            "entity_types": ["Contract", "Clause", "Party"],
            "concepts": [
                {"id": "contract", "label": "Contract", "type": "Document"},
                {"id": "clause", "label": "Clause", "type": "TextSection"},
            ],
            "relationships": [
                {"source": "clause", "target": "contract", "type": "part_of"}
            ],
        }

        plugin.load_custom_domain(custom_domain)

        assert plugin.domain == "legal"
        assert len(plugin.hierarchy.concepts) == 2
        assert "contract" in plugin.hierarchy.concepts
        assert "clause" in plugin.hierarchy.concepts

    def test_plugin_factory_function(self):
        """Test plugin creation from configuration."""
        config = {
            "type": "general",
            "auto_detect_domain": True,
            "sources": [
                {
                    "type": "example",
                    "domain": "financial",
                    "concepts": [
                        {"id": "account", "label": "Account", "synonyms": ["acct"]},
                        {
                            "id": "transaction",
                            "label": "Transaction",
                            "synonyms": ["txn"],
                        },
                    ],
                }
            ],
        }

        plugin = create_plugin_from_config(config)

        assert isinstance(plugin, GeneralOntologyPlugin)
        assert len(plugin.hierarchy.concepts) == 2


class TestReasoningEngine:
    """Test ontology reasoning capabilities with general plugin."""

    def setup_method(self):
        """Set up test hierarchy for reasoning tests."""
        self.hierarchy = ConceptHierarchy()

        # Create test concepts for any domain
        concepts = [
            Concept("entity", "Entity", "General entity concept"),
            Concept("physical_entity", "Physical Entity", "Physical object"),
            Concept("abstract_entity", "Abstract Entity", "Abstract concept"),
            Concept("document", "Document", "Document or file"),
            Concept("contract", "Contract", "Legal contract"),
            Concept("medication", "Medication", "Medical medication"),
        ]

        for concept in concepts:
            self.hierarchy.add_concept(concept)

        # Add hierarchical relationships
        self.hierarchy.add_relationship("physical_entity", "entity", "is_a")
        self.hierarchy.add_relationship("abstract_entity", "entity", "is_a")
        self.hierarchy.add_relationship("document", "abstract_entity", "is_a")
        self.hierarchy.add_relationship("contract", "document", "is_a")
        self.hierarchy.add_relationship("medication", "physical_entity", "is_a")

        self.reasoner = OntologyReasoner(self.hierarchy)

    def test_subsumption_reasoning(self):
        """Test hierarchical subsumption reasoning."""
        # Test direct subsumption
        assert self.reasoner.is_subsumed_by("contract", "document")
        assert self.reasoner.is_subsumed_by("contract", "entity")

        # Test non-subsumption
        assert not self.reasoner.is_subsumed_by("medication", "document")

    def test_relationship_inference(self):
        """Test relationship inference capabilities."""
        entities = [
            Entity(text="legal contract", entity_type="Document", confidence=0.9),
            Entity(text="aspirin", entity_type="Drug", confidence=0.8),
        ]

        # Mock concept mapping
        entities[0].metadata = {"concept_id": "contract"}
        entities[1].metadata = {"concept_id": "medication"}

        inferred_relationships = self.reasoner.infer_relationships(entities)
        assert isinstance(inferred_relationships, list)

    def test_query_expansion(self):
        """Test query expansion using general ontology."""
        expander = QueryExpander(self.hierarchy)

        # Test synonym expansion
        result = expander.expand_query("contract review", strategy="synonyms")
        assert result.original_query == "contract review"
        assert result.expanded_query != result.original_query
        assert isinstance(result.expansion_terms, list)
        assert isinstance(result.confidence, float)

        # Test hierarchical expansion
        result = expander.expand_query("document analysis", strategy="hierarchical")
        assert result.expanded_query != result.original_query


class TestEntityExtraction:
    """Test ontology-aware entity extraction with general plugin."""

    def setup_method(self):
        """Set up entity extraction tests."""
        # Create mock configuration
        self.config_manager = ConfigurationManager()
        self.config_manager._config = {
            "entity_extraction": {
                "method": "ontology_hybrid",
                "confidence_threshold": 0.6,
                "entity_types": ["CONCEPT", "ENTITY", "TERM"],
                "max_entities": 50,
            },
            "ontology": {
                "enabled": True,
                "type": "general",
                "auto_detect_domain": True,
                "reasoning": {"enable_inference": True},
                "sources": [
                    {
                        "type": "example",
                        "domain": "general",
                        "concepts": [
                            {
                                "id": "contract",
                                "label": "Contract",
                                "synonyms": ["agreement"],
                            },
                            {
                                "id": "medication",
                                "label": "Medication",
                                "synonyms": ["drug", "medicine"],
                            },
                            {
                                "id": "server",
                                "label": "Server",
                                "synonyms": ["host", "machine"],
                            },
                        ],
                    }
                ],
            },
        }

        self.extractor = OntologyAwareEntityExtractor(
            config_manager=self.config_manager
        )

    def test_general_domain_detection(self):
        """Test automatic domain detection with general plugin."""
        medical_text = "Patient has diabetes and takes medication daily"
        legal_text = "Review the contract terms and agreement clauses"
        tech_text = "Server database connection failed with timeout"

        # Test that extraction works with any domain text
        medical_entities = self.extractor.extract_with_ontology(medical_text)
        legal_entities = self.extractor.extract_with_ontology(legal_text)
        tech_entities = self.extractor.extract_with_ontology(tech_text)

        # Should extract entities from any domain
        assert isinstance(medical_entities, list)
        assert isinstance(legal_entities, list)
        assert isinstance(tech_entities, list)

    def test_entity_extraction_with_general_ontology(self):
        """Test ontology-enhanced entity extraction with general plugin."""
        test_documents = [
            Document(
                id="medical_doc",
                page_content="Patient takes medication for diabetes treatment.",
                metadata={"domain": "medical"},
            ),
            Document(
                id="legal_doc",
                page_content="The contract agreement requires legal review.",
                metadata={"domain": "legal"},
            ),
            Document(
                id="tech_doc",
                page_content="Server performance monitoring shows high CPU usage.",
                metadata={"domain": "technology"},
            ),
        ]

        for doc in test_documents:
            entities = self.extractor.extract_with_ontology(doc.page_content, doc)

            assert isinstance(entities, list)

            # Check for general ontology metadata
            for entity in entities:
                assert "domain" in entity.metadata
                assert "method" in entity.metadata

                # Should have auto-detected or assigned domain
                if "auto_detected_domain" in entity.metadata:
                    assert isinstance(entity.metadata["auto_detected_domain"], str)

    def test_entity_enrichment_general(self):
        """Test entity enrichment with general ontology metadata."""
        test_text = "Review contract medication server requirements"
        entities = self.extractor.extract_with_ontology(test_text)

        # Check for ontology metadata in extracted entities
        for entity in entities:
            assert "domain" in entity.metadata
            assert "method" in entity.metadata

            # Check for ontology-specific metadata
            if "concept_id" in entity.metadata:
                assert isinstance(entity.metadata["concept_id"], str)

            # Should have confidence scores
            assert isinstance(entity.confidence, float)
            assert 0.0 <= entity.confidence <= 1.0


class TestMultiDomainSupport:
    """Test support for multiple domains in a single ontology."""

    def test_multi_domain_ontology_loading(self):
        """Test loading ontology with multiple domains."""
        plugin = GeneralOntologyPlugin()

        # Load concepts from multiple domains
        medical_source = {
            "type": "example",
            "domain": "medical",
            "concepts": [
                {"id": "diabetes", "label": "Diabetes", "domain": "medical"},
                {"id": "medication", "label": "Medication", "domain": "medical"},
            ],
        }

        legal_source = {
            "type": "example",
            "domain": "legal",
            "concepts": [
                {"id": "contract", "label": "Contract", "domain": "legal"},
                {"id": "clause", "label": "Clause", "domain": "legal"},
            ],
        }

        plugin._load_example_concepts(medical_source)
        plugin._load_example_concepts(legal_source)

        assert len(plugin.hierarchy.concepts) == 4

        # Should detect mixed domain
        detected_domain = plugin.auto_detect_domain(
            {"concepts": plugin.hierarchy.concepts, "metadata": {}}
        )

        assert detected_domain in ["medical", "legal", "general", "mixed"]

    def test_cross_domain_reasoning(self):
        """Test reasoning across multiple domains."""
        plugin = GeneralOntologyPlugin()

        # Create multi-domain hierarchy
        concepts = [
            Concept("entity", "Entity", "Universal entity"),
            Concept("legal_entity", "Legal Entity", "Legal concept"),
            Concept("medical_entity", "Medical Entity", "Medical concept"),
            Concept("contract", "Contract", "Legal contract"),
            Concept("medication", "Medication", "Medical drug"),
        ]

        for concept in plugin.hierarchy.concepts:
            plugin.hierarchy.add_concept(concept)

        # Add cross-domain relationships
        plugin.hierarchy.add_relationship("legal_entity", "entity", "is_a")
        plugin.hierarchy.add_relationship("medical_entity", "entity", "is_a")
        plugin.hierarchy.add_relationship("contract", "legal_entity", "is_a")
        plugin.hierarchy.add_relationship("medication", "medical_entity", "is_a")

        reasoner = OntologyReasoner(plugin.hierarchy)

        # Should be able to reason about relationships across domains
        assert reasoner.is_subsumed_by("contract", "entity")
        assert reasoner.is_subsumed_by("medication", "entity")
        assert not reasoner.is_subsumed_by("contract", "medical_entity")


class TestPerformance:
    """Test general ontology system performance and scalability."""

    def test_general_plugin_loading_performance(self):
        """Test performance of loading general plugin."""
        import time

        start_time = time.time()

        # Create plugin with sample ontology
        plugin = GeneralOntologyPlugin()

        # Load sample multi-domain ontology
        sample_source = {
            "type": "example",
            "domain": "general",
            "concepts": [
                {
                    "id": f"concept_{i}",
                    "label": f"Concept {i}",
                    "synonyms": [f"synonym_{i}"],
                }
                for i in range(100)  # Load 100 concepts
            ],
        }

        plugin._load_example_concepts(sample_source)

        load_time = time.time() - start_time

        # Should load reasonably quickly
        assert load_time < 2.0  # 2 seconds max for 100 concepts
        assert len(plugin.hierarchy.concepts) == 100

    def test_auto_detection_performance(self):
        """Test performance of domain auto-detection."""
        import time

        plugin = GeneralOntologyPlugin()

        # Load large ontology
        large_source = {
            "type": "example",
            "domain": "mixed",
            "concepts": [
                {
                    "id": f"medical_{i}",
                    "label": f"Medical Concept {i}",
                    "domain": "medical",
                }
                for i in range(50)
            ]
            + [
                {"id": f"legal_{i}", "label": f"Legal Concept {i}", "domain": "legal"}
                for i in range(50)
            ],
        }

        plugin._load_example_concepts(large_source)

        start_time = time.time()

        detected_domain = plugin.auto_detect_domain(
            {"concepts": plugin.hierarchy.concepts, "metadata": {}}
        )

        detection_time = time.time() - start_time

        # Should detect quickly even with large ontology
        assert detection_time < 0.5  # 500ms max
        assert detected_domain in ["medical", "legal", "mixed", "general"]

    def test_reasoning_performance_general(self):
        """Test reasoning performance with general ontology."""
        plugin = GeneralOntologyPlugin()

        # Create hierarchical ontology
        for i in range(50):
            concept = Concept(f"concept_{i}", f"Concept {i}", f"Description {i}")
            plugin.hierarchy.add_concept(concept)

            # Create hierarchy (each concept is child of previous)
            if i > 0:
                plugin.hierarchy.add_relationship(
                    f"concept_{i}", f"concept_{i-1}", "is_a"
                )

        reasoner = OntologyReasoner(plugin.hierarchy)

        import time

        start_time = time.time()

        # Test subsumption queries
        for i in range(10, 40):
            ancestors = plugin.hierarchy.get_ancestors(f"concept_{i}", max_depth=5)

        reasoning_time = time.time() - start_time

        # Should complete reasoning quickly
        assert reasoning_time < 1.0  # 1 second max for 30 queries


# Integration test
class TestIntegration:
    """Test integration between general ontology components."""

    def test_end_to_end_general_workflow(self):
        """Test complete general ontology workflow."""
        # 1. Create general plugin
        plugin = GeneralOntologyPlugin()

        # 2. Load sample ontology
        sample_source = {
            "type": "example",
            "domain": "general",
            "concepts": [
                {"id": "entity", "label": "Entity", "synonyms": ["thing", "object"]},
                {"id": "concept", "label": "Concept", "synonyms": ["idea", "notion"]},
                {"id": "document", "label": "Document", "synonyms": ["file", "record"]},
            ],
        }

        plugin._load_example_concepts(sample_source)
        assert len(plugin.hierarchy.concepts) == 3

        # 3. Test auto-detection
        detected_domain = plugin.auto_detect_domain(
            {"concepts": plugin.hierarchy.concepts, "metadata": {}}
        )
        assert detected_domain == "general"

        # 4. Create reasoner
        reasoner = OntologyReasoner(plugin.hierarchy)

        # 5. Test entity extraction
        config_manager = ConfigurationManager()
        config_manager._config = {
            "entity_extraction": {
                "method": "ontology_hybrid",
                "confidence_threshold": 0.6,
            },
            "ontology": {
                "enabled": True,
                "type": "general",
                "auto_detect_domain": True,
            },
        }

        extractor = OntologyAwareEntityExtractor(config_manager=config_manager)

        # 6. Process test document
        test_text = "The document contains important concepts and entities for analysis"
        entities = extractor.extract_with_ontology(test_text)

        # 7. Verify results
        assert isinstance(entities, list)

        # Should have extracted some entities
        if entities:
            for entity in entities:
                assert hasattr(entity, "text")
                assert hasattr(entity, "entity_type")
                assert hasattr(entity, "confidence")
                assert "domain" in entity.metadata


class TestOntologyFormats:
    """Test support for different ontology file formats."""

    def test_format_detection(self):
        """Test auto-detection of ontology file formats."""
        plugin = GeneralOntologyPlugin()

        # Test format detection based on file extension
        assert plugin._detect_format("ontology.owl") == "owl"
        assert plugin._detect_format("ontology.rdf") == "rdf"
        assert plugin._detect_format("ontology.ttl") == "ttl"
        assert plugin._detect_format("ontology.n3") == "n3"
        assert plugin._detect_format("ontology.skos") == "skos"
        assert plugin._detect_format("ontology.xml") == "xml"
        assert plugin._detect_format("ontology.unknown") == "unknown"

    def test_file_format_support(self):
        """Test that plugin supports various ontology formats."""
        plugin = GeneralOntologyPlugin()

        supported_formats = plugin.get_supported_formats()

        expected_formats = ["owl", "rdf", "ttl", "n3", "skos", "xml"]
        for fmt in expected_formats:
            assert fmt in supported_formats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
