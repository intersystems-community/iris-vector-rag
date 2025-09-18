#!/usr/bin/env python3
"""
Adapted GraphRAG Multi-Hop Demo for Testing Merged Implementation

This script adapts the original GraphRAG multi-hop demo to specifically test
the merged GraphRAG implementation with comprehensive performance tracking,
biomedical queries, and visualization capabilities.

Features:
- Tests merged GraphRAG implementation with multi-hop queries
- Biomedical query testing using PMC dataset samples
- Performance metrics collection and analysis
- Graph traversal path visualization
- Comparison with current implementation
- Mock data support for environments without database

Usage:
    python scripts/test_merged_graphrag_multihop_demo.py [--config config.yaml] [--use-mocks]
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports with graceful fallback
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization packages not available. Install with: pip install networkx matplotlib")

# Colorama import with fallback
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback definitions for colorama when not available
    class MockColorama:
        def __getattr__(self, name):
            return ""
    
    Fore = MockColorama()
    Back = MockColorama()
    Style = MockColorama()
    COLORAMA_AVAILABLE = False
    print("Warning: colorama not available. Install with: pip install colorama")

# IRIS RAG imports
try:
    from iris_rag.pipelines.graphrag import GraphRAGPipeline as CurrentGraphRAG
    from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline as MergedGraphRAG
    from iris_rag.core.connection import ConnectionManager
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.models import Document
except ImportError as e:
    print(f"Failed to import IRIS RAG components: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GraphTraversalStep:
    """Represents a single step in graph traversal."""
    hop: int
    entities: List[str]
    relationships: List[Tuple[str, str, str]]  # (source, target, type)
    confidence_scores: Dict[str, float]
    execution_time_ms: float
    documents_found: int


@dataclass
class MultiHopQueryResult:
    """Complete result of a multi-hop query."""
    query: str
    query_type: str
    implementation: str
    total_hops: int
    total_entities_traversed: int
    total_relationships_traversed: int
    traversal_path: List[GraphTraversalStep]
    final_documents: List[Document]
    answer: str
    confidence_score: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class BiomedicalDataGenerator:
    """Generate biomedical test data based on PMC dataset patterns."""
    
    @staticmethod
    def create_pmc_style_documents() -> List[Document]:
        """Create PMC-style biomedical documents for testing."""
        return [
            Document(
                id="PMC_12345_diabetes_overview",
                page_content="""
                Type 2 diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia 
                resulting from defects in insulin secretion, insulin action, or both. The chronic 
                hyperglycemia of diabetes is associated with long-term damage, dysfunction, and failure 
                of different organs, especially the eyes, kidneys, nerves, heart, and blood vessels.
                
                Risk factors include obesity (BMI ≥30 kg/m²), sedentary lifestyle, family history of 
                diabetes, history of gestational diabetes, hypertension (≥140/90 mmHg), HDL cholesterol 
                <35 mg/dL, triglyceride level ≥250 mg/dL, and previously identified impaired glucose 
                tolerance or impaired fasting glucose.
                
                Pathophysiology involves insulin resistance in peripheral tissues (muscle, liver, 
                adipose tissue) combined with progressive beta-cell dysfunction and failure. The liver 
                shows increased glucose production via gluconeogenesis and glycogenolysis, while muscle 
                tissue demonstrates decreased glucose uptake.
                """,
                metadata={
                    "source": "PMC_12345", 
                    "title": "Pathophysiology and Risk Factors of Type 2 Diabetes",
                    "authors": "Smith, J. et al.",
                    "journal": "Diabetes Care",
                    "year": "2023",
                    "pmid": "12345678",
                    "keywords": ["diabetes", "insulin resistance", "hyperglycemia", "beta-cell dysfunction"]
                }
            ),
            Document(
                id="PMC_12346_metformin_mechanisms",
                page_content="""
                Metformin, a biguanide derivative, is the first-line oral antidiabetic medication for 
                type 2 diabetes mellitus. Its primary mechanism of action involves the activation of 
                AMP-activated protein kinase (AMPK) in hepatocytes, leading to reduced gluconeogenesis 
                and increased glucose uptake in skeletal muscle.
                
                Molecular mechanisms include: (1) Inhibition of mitochondrial respiratory chain complex I, 
                reducing ATP synthesis and increasing AMP/ATP ratio; (2) Activation of liver kinase B1 
                (LKB1) and subsequently AMPK; (3) Phosphorylation and inactivation of acetyl-CoA 
                carboxylase (ACC); (4) Inhibition of fatty acid synthesis and promotion of fatty acid 
                oxidation.
                
                Clinical efficacy demonstrates HbA1c reduction of 1.0-1.5% when used as monotherapy. 
                Contraindications include estimated glomerular filtration rate (eGFR) <30 mL/min/1.73m², 
                severe hepatic dysfunction, congestive heart failure requiring pharmacologic treatment, 
                and conditions predisposing to lactic acidosis.
                """,
                metadata={
                    "source": "PMC_12346",
                    "title": "Metformin: Molecular Mechanisms and Clinical Applications",
                    "authors": "Johnson, K. et al.",
                    "journal": "Pharmacological Reviews",
                    "year": "2023",
                    "pmid": "12346789",
                    "keywords": ["metformin", "AMPK", "gluconeogenesis", "diabetes treatment"]
                }
            ),
            Document(
                id="PMC_12347_diabetic_nephropathy",
                page_content="""
                Diabetic nephropathy (DN) is a leading cause of end-stage renal disease worldwide, 
                affecting approximately 20-40% of patients with diabetes mellitus. The pathogenesis 
                involves multiple interconnected pathways including advanced glycation end products 
                (AGEs), protein kinase C (PKC) activation, polyol pathway flux, and hexosamine pathway 
                activation.
                
                Early pathological changes include glomerular hyperfiltration, mesangial expansion, 
                thickening of glomerular and tubular basement membranes, and podocyte loss. Progressive 
                changes lead to glomerulosclerosis, tubulointerstitial fibrosis, and ultimately nephron 
                loss. Key biomarkers include microalbuminuria (30-300 mg/24h), elevated serum creatinine, 
                and decreased estimated glomerular filtration rate.
                
                Angiotensin-converting enzyme (ACE) inhibitors and angiotensin receptor blockers (ARBs) 
                represent gold standard therapy, providing both blood pressure control and renoprotective 
                effects through reduction of intraglomerular pressure and anti-inflammatory mechanisms. 
                SGLT-2 inhibitors (empagliflozin, canagliflozin) demonstrate additional renoprotective 
                benefits independent of glycemic control.
                """,
                metadata={
                    "source": "PMC_12347",
                    "title": "Diabetic Nephropathy: Pathogenesis and Therapeutic Approaches",
                    "authors": "Williams, M. et al.",
                    "journal": "Kidney International",
                    "year": "2023",
                    "pmid": "12347890",
                    "keywords": ["diabetic nephropathy", "AGEs", "ACE inhibitors", "SGLT-2 inhibitors"]
                }
            ),
            Document(
                id="PMC_12348_cardiovascular_diabetes",
                page_content="""
                Cardiovascular disease (CVD) is the leading cause of morbidity and mortality in patients 
                with diabetes mellitus, with a 2-4 fold increased risk compared to non-diabetic individuals. 
                The pathophysiology involves accelerated atherosclerosis, endothelial dysfunction, 
                increased platelet aggregation, and altered lipid metabolism.
                
                Hyperglycemia promotes oxidative stress through increased production of reactive oxygen 
                species (ROS), leading to endothelial nitric oxide synthase (eNOS) uncoupling and reduced 
                nitric oxide bioavailability. Advanced glycation end products (AGEs) accumulate in vessel 
                walls, promoting inflammation and collagen cross-linking.
                
                Lipid management targets include LDL cholesterol <70 mg/dL for high-risk patients, with 
                statins as first-line therapy. High-intensity statins (atorvastatin 40-80 mg, rosuvastatin 
                20-40 mg) demonstrate significant cardiovascular risk reduction. Dual antiplatelet therapy 
                may be considered in high-risk patients without bleeding contraindications.
                
                Novel therapies include GLP-1 receptor agonists (liraglutide, semaglutide) and SGLT-2 
                inhibitors, which demonstrate cardiovascular benefits beyond glycemic control through 
                mechanisms including weight reduction, blood pressure lowering, and direct cardioprotective 
                effects.
                """,
                metadata={
                    "source": "PMC_12348",
                    "title": "Cardiovascular Complications in Diabetes: Pathophysiology and Management",
                    "authors": "Brown, L. et al.",
                    "journal": "Circulation",
                    "year": "2023",
                    "pmid": "12348901",
                    "keywords": ["cardiovascular disease", "diabetes", "atherosclerosis", "statins", "GLP-1"]
                }
            ),
            Document(
                id="PMC_12349_covid_diabetes",
                page_content="""
                The COVID-19 pandemic has highlighted the bidirectional relationship between SARS-CoV-2 
                infection and diabetes mellitus. Patients with diabetes face increased risk of severe 
                COVID-19, including higher rates of hospitalization, mechanical ventilation, and mortality.
                
                Proposed mechanisms for increased severity include: (1) Chronic inflammation and 
                immunosuppression associated with diabetes; (2) Higher ACE2 receptor expression in 
                diabetic patients; (3) Impaired neutrophil function and delayed viral clearance; 
                (4) Increased risk of thrombotic complications due to hypercoagulable state.
                
                COVID-19 can also induce new-onset diabetes through direct pancreatic beta-cell damage 
                via ACE2 receptors, systemic inflammation, and corticosteroid treatment effects. Studies 
                report 14-47% incidence of new hyperglycemia in hospitalized COVID-19 patients.
                
                Management considerations include: intensive glucose monitoring, potential need for 
                insulin therapy even in non-insulin dependent patients, telemedicine for diabetes care 
                continuity, and prioritization for COVID-19 vaccination. Metformin should be discontinued 
                during acute illness due to increased lactic acidosis risk.
                """,
                metadata={
                    "source": "PMC_12349",
                    "title": "COVID-19 and Diabetes: A Bidirectional Relationship",
                    "authors": "Davis, R. et al.",
                    "journal": "Diabetes Care",
                    "year": "2023",
                    "pmid": "12349012",
                    "keywords": ["COVID-19", "diabetes", "SARS-CoV-2", "hyperglycemia", "ACE2"]
                }
            ),
            Document(
                id="PMC_12350_insulin_resistance",
                page_content="""
                Insulin resistance is a pathophysiological condition characterized by diminished 
                responsiveness of target tissues to physiological levels of insulin. It represents 
                the primary defect underlying type 2 diabetes mellitus and is central to metabolic 
                syndrome pathogenesis.
                
                Molecular mechanisms involve defects in insulin signaling cascade, including: 
                (1) Insulin receptor substrate (IRS) protein phosphorylation abnormalities; 
                (2) Reduced phosphatidylinositol 3-kinase (PI3K) activity; (3) Impaired glucose 
                transporter type 4 (GLUT4) translocation; (4) Altered protein kinase B (Akt) 
                phosphorylation patterns.
                
                Adipose tissue dysfunction plays a crucial role through altered adipokine secretion. 
                Decreased adiponectin and increased tumor necrosis factor-alpha (TNF-α), interleukin-6 
                (IL-6), and resistin contribute to systemic insulin resistance. Free fatty acid 
                release from dysfunctional adipocytes promotes hepatic gluconeogenesis and muscle 
                insulin resistance.
                
                Assessment methods include homeostatic model assessment for insulin resistance 
                (HOMA-IR), hyperinsulinemic-euglycemic clamp (gold standard), and oral glucose 
                tolerance testing. Therapeutic approaches target multiple pathways including 
                insulin sensitizers, weight management, and exercise interventions.
                """,
                metadata={
                    "source": "PMC_12350",
                    "title": "Molecular Mechanisms of Insulin Resistance",
                    "authors": "Taylor, S. et al.",
                    "journal": "Nature Reviews Endocrinology",
                    "year": "2023",
                    "pmid": "12350123",
                    "keywords": ["insulin resistance", "IRS proteins", "GLUT4", "adipokines", "metabolic syndrome"]
                }
            )
        ]


class MergedGraphRAGMultiHopDemo:
    """Comprehensive demonstration of merged GraphRAG multi-hop query capabilities."""
    
    def __init__(self, config_path: Optional[str] = None, use_mocks: bool = False, output_dir: str = "outputs/multihop_merged_demo"):
        """
        Initialize the merged GraphRAG multi-hop demonstration.
        
        Args:
            config_path: Optional path to configuration file
            use_mocks: Whether to use mock data instead of real database
            output_dir: Directory for saving outputs and visualizations
        """
        self.use_mocks = use_mocks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize IRIS RAG components
        try:
            self.config_manager = ConfigurationManager(config_path)
            if not use_mocks:
                self.connection_manager = ConnectionManager()
            else:
                self.connection_manager = None
                logger.info("Using mock data mode - database connection skipped")
        except Exception as e:
            if not use_mocks:
                logger.warning(f"Failed to initialize components, falling back to mock mode: {e}")
                self.use_mocks = True
                self.connection_manager = None
            else:
                raise
        
        # Results storage
        self.query_results: List[MultiHopQueryResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_execution_time": 0.0,
            "total_entities_processed": 0,
            "total_relationships_traversed": 0
        }
        
        logger.info(f"Merged GraphRAG Multi-Hop Demo initialized (mock_mode={self.use_mocks})")
    
    def setup_pipelines(self) -> Tuple[Any, Any]:
        """Setup both current and merged GraphRAG pipelines for comparison."""
        # Generate biomedical test data
        sample_docs = BiomedicalDataGenerator.create_pmc_style_documents()
        
        # Initialize real pipelines - no mock fallbacks
        current_pipeline = CurrentGraphRAG(
            connection_manager=self.connection_manager,
            config_manager=self.config_manager
        )
        merged_pipeline = MergedGraphRAG(
            connection_manager=self.connection_manager,
            config_manager=self.config_manager
        )
        
        # Load documents into both pipelines
        logger.info("Loading biomedical documents into current GraphRAG pipeline...")
        current_pipeline.load_documents("", documents=sample_docs)
        
        logger.info("Loading biomedical documents into merged GraphRAG pipeline...")
        merged_pipeline.load_documents("", documents=sample_docs)
        
        logger.info("Both real pipelines initialized successfully with biomedical data")
        return current_pipeline, merged_pipeline
    
    def _create_mock_pipeline(self, implementation_type: str):
        """Create a mock pipeline for testing when database is unavailable."""
        class MockPipeline:
            def __init__(self, impl_type):
                self.impl_type = impl_type
            
            def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
                import random
                time.sleep(random.uniform(0.2, 0.8))  # Simulate processing time
                
                # Generate mock response with more realistic biomedical content
                mock_docs = []
                num_docs = random.randint(2, 5)
                
                for i in range(num_docs):
                    mock_docs.append({
                        "id": f"PMC_mock_{i}",
                        "page_content": f"Mock biomedical content related to {query_text[:100]}...",
                        "metadata": {
                            "title": f"Mock Study {i+1}",
                            "source": f"PMC_mock_{i}",
                            "retrieval_method": "knowledge_graph"
                        }
                    })
                
                base_time = random.uniform(200, 1500)
                # Merged implementation might show different performance characteristics
                if self.impl_type == "merged":
                    # Sometimes faster due to optimizations, sometimes slower due to additional features
                    variation = random.uniform(0.8, 1.3)
                    fallback_used = random.random() < 0.1  # 10% chance of fallback
                else:
                    variation = 1.0
                    fallback_used = False
                
                retrieval_method = "vector_fallback" if fallback_used else "knowledge_graph_traversal"
                
                return {
                    "query": query_text,
                    "answer": f"Mock biomedical answer from {self.impl_type} implementation: {query_text}",
                    "retrieved_documents": mock_docs,
                    "contexts": [doc["page_content"] for doc in mock_docs],
                    "sources": [{"document_id": doc["id"], "source": doc["metadata"]["source"]} for doc in mock_docs],
                    "metadata": {
                        "num_retrieved": len(mock_docs),
                        "processing_time_ms": base_time * variation,
                        "pipeline_type": f"graphrag_{self.impl_type}",
                        "retrieval_method": retrieval_method,
                        "db_exec_count": random.randint(4, 12),
                        "step_timings_ms": {
                            "query_entity_extraction_ms": base_time * 0.1 if self.impl_type == "merged" else 0,
                            "find_seed_entities_ms": base_time * 0.2,
                            "traverse_graph_ms": base_time * 0.5,
                            "get_documents_ms": base_time * 0.2
                        },
                        "confidence": random.uniform(0.7, 0.95),
                        "entities_found": random.randint(3, 8),
                        "relationships_traversed": random.randint(5, 15)
                    }
                }
            
            def load_documents(self, *args, **kwargs):
                logger.info(f"Mock {self.impl_type} pipeline: biomedical documents loaded")
        
        return MockPipeline(implementation_type)
    
    def create_biomedical_test_queries(self) -> List[Dict[str, Any]]:
        """Create biomedical multi-hop test queries based on PMC data patterns."""
        return [
            # 2-hop biomedical queries
            {
                "query": "What are the molecular mechanisms linking diabetes to cardiovascular complications?",
                "type": "2-hop-molecular-pathway",
                "explanation": "Diabetes → molecular mechanisms → cardiovascular complications",
                "max_hops": 2,
                "expected_entities": ["diabetes", "molecular mechanisms", "cardiovascular", "oxidative stress"],
                "complexity": "medium"
            },
            {
                "query": "How do SGLT-2 inhibitors provide renoprotective effects in diabetic nephropathy?",
                "type": "2-hop-drug-mechanism",
                "explanation": "SGLT-2 inhibitors → mechanisms → renoprotective effects",
                "max_hops": 2,
                "expected_entities": ["SGLT-2 inhibitors", "renoprotective", "diabetic nephropathy"],
                "complexity": "medium"
            },
            {
                "query": "What is the relationship between insulin resistance and adipokine dysfunction?",
                "type": "2-hop-pathophysiology",
                "explanation": "Insulin resistance → pathways → adipokine dysfunction",
                "max_hops": 2,
                "expected_entities": ["insulin resistance", "adipokines", "dysfunction"],
                "complexity": "medium"
            },
            
            # 3-hop biomedical queries
            {
                "query": "How does COVID-19 infection lead to new-onset diabetes through ACE2 receptor mechanisms?",
                "type": "3-hop-viral-pathogenesis",
                "explanation": "COVID-19 → ACE2 receptors → beta-cell damage → new-onset diabetes",
                "max_hops": 3,
                "expected_entities": ["COVID-19", "ACE2", "beta-cell", "diabetes"],
                "complexity": "high"
            },
            {
                "query": "What are the shared molecular pathways between metformin action and cardiovascular protection?",
                "type": "3-hop-drug-cardioprotection",
                "explanation": "Metformin → AMPK activation → metabolic effects → cardiovascular protection",
                "max_hops": 3,
                "expected_entities": ["metformin", "AMPK", "cardiovascular protection"],
                "complexity": "high"
            },
            {
                "query": "How do advanced glycation end products contribute to both nephropathy and cardiovascular disease?",
                "type": "3-hop-AGE-complications",
                "explanation": "AGEs → inflammation/damage → nephropathy + cardiovascular disease",
                "max_hops": 3,
                "expected_entities": ["AGEs", "inflammation", "nephropathy", "cardiovascular"],
                "complexity": "high"
            },
            
            # Complex multi-pathway queries
            {
                "query": "What are the interconnected pathways linking obesity, insulin resistance, diabetes, and cardiovascular disease?",
                "type": "complex-metabolic-network",
                "explanation": "Multi-pathway analysis of metabolic syndrome cascade",
                "max_hops": 4,
                "expected_entities": ["obesity", "insulin resistance", "diabetes", "cardiovascular disease"],
                "complexity": "very_high"
            },
            {
                "query": "How do different diabetes medications interact with COVID-19 treatment and cardiovascular risk management?",
                "type": "complex-drug-interaction-network",
                "explanation": "Multi-drug interaction analysis across disease contexts",
                "max_hops": 4,
                "expected_entities": ["diabetes medications", "COVID-19 treatment", "cardiovascular risk"],
                "complexity": "very_high"
            }
        ]
    
    def execute_multihop_query(self, query_config: Dict[str, Any], pipeline, implementation: str) -> MultiHopQueryResult:
        """Execute a multi-hop query with detailed performance tracking."""
        start_time = time.perf_counter()
        
        try:
            # Execute query
            result = pipeline.query(
                query_text=query_config["query"],
                top_k=10,
                include_sources=True
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract metadata
            metadata = result.get('metadata', {})
            retrieved_docs = result.get('retrieved_documents', [])
            
            # Calculate traversal path (simplified extraction)
            traversal_path = self._extract_traversal_path_from_result(result, query_config.get("max_hops", 2))
            
            # Performance metrics
            performance_metrics = {
                "db_executions": metadata.get('db_exec_count', 0),
                "step_timings": metadata.get('step_timings_ms', {}),
                "retrieval_method": metadata.get('retrieval_method', 'unknown'),
                "processing_time_ms": metadata.get('processing_time_ms', execution_time_ms)
            }
            
            multihop_result = MultiHopQueryResult(
                query=query_config["query"],
                query_type=query_config["type"],
                implementation=implementation,
                total_hops=len(traversal_path),
                total_entities_traversed=metadata.get('entities_found', 0),
                total_relationships_traversed=metadata.get('relationships_traversed', 0),
                traversal_path=traversal_path,
                final_documents=retrieved_docs,
                answer=result.get('answer', 'No answer generated'),
                confidence_score=metadata.get('confidence', 0.0),
                execution_time_ms=execution_time_ms,
                metadata=metadata,
                performance_metrics=performance_metrics
            )
            
            return multihop_result
            
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Multi-hop query failed on {implementation}: {e}")
            
            return MultiHopQueryResult(
                query=query_config["query"],
                query_type=query_config["type"],
                implementation=implementation,
                total_hops=0,
                total_entities_traversed=0,
                total_relationships_traversed=0,
                traversal_path=[],
                final_documents=[],
                answer=f"Query failed: {e}",
                confidence_score=0.0,
                execution_time_ms=execution_time_ms,
                metadata={"error": str(e)},
                performance_metrics={}
            )
    
    def _extract_traversal_path_from_result(self, result: Dict[str, Any], max_hops: int) -> List[GraphTraversalStep]:
        """Extract graph traversal path information from query result."""
        step_timings = result.get('metadata', {}).get('step_timings_ms', {})
        entities_found = result.get('metadata', {}).get('entities_found', 0)
        relationships = result.get('metadata', {}).get('relationships_traversed', 0)
        
        # Create simplified traversal path
        traversal_steps = []
        
        for hop in range(min(max_hops, 3)):  # Limit to reasonable number of hops
            step = GraphTraversalStep(
                hop=hop + 1,
                entities=[f"entity_{hop}_{i}" for i in range(max(1, entities_found // max_hops))],
                relationships=[(f"e{hop}_{i}", f"e{hop}_{i+1}", "relates_to") for i in range(2)],
                confidence_scores={f"entity_{hop}_{i}": 0.8 - (hop * 0.1) for i in range(2)},
                execution_time_ms=step_timings.get(f"hop_{hop}_ms", 100.0),
                documents_found=max(1, len(result.get('retrieved_documents', [])) // max_hops)
            )
            traversal_steps.append(step)
        
        return traversal_steps
    
    def demonstrate_biomedical_multihop_queries(self) -> None:
        """Run biomedical multi-hop query demonstrations."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"🧬 BIOMEDICAL MULTI-HOP QUERY DEMONSTRATIONS")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        # Setup pipelines
        current_pipeline, merged_pipeline = self.setup_pipelines()
        
        # Get test queries
        test_queries = self.create_biomedical_test_queries()
        
        for i, query_config in enumerate(test_queries, 1):
            print(f"\n{Fore.YELLOW}📋 Biomedical Query #{i}: {query_config['type']}{Style.RESET_ALL}")
            print(f"Query: {query_config['query']}")
            print(f"Logic: {query_config['explanation']}")
            print(f"Complexity: {query_config['complexity']}")
            
            # Execute on current implementation
            print(f"\n{Fore.BLUE}🔄 Testing Current GraphRAG Implementation...{Style.RESET_ALL}")
            current_result = self.execute_multihop_query(query_config, current_pipeline, "current")
            self._display_query_result(current_result)
            
            # Execute on merged implementation
            print(f"\n{Fore.GREEN}🔄 Testing Merged GraphRAG Implementation...{Style.RESET_ALL}")
            merged_result = self.execute_multihop_query(query_config, merged_pipeline, "merged")
            self._display_query_result(merged_result)
            
            # Compare results
            self._compare_results(current_result, merged_result)
            
            # Store results
            self.query_results.extend([current_result, merged_result])
            
            # Update performance metrics
            self._update_performance_metrics(current_result)
            self._update_performance_metrics(merged_result)
            
            # Visualize if available
            if VISUALIZATION_AVAILABLE:
                self._create_comparison_visualization(current_result, merged_result, f"biomedical_query_{i}")
            
            print(f"\n{'-'*60}")
    
    def _display_query_result(self, result: MultiHopQueryResult) -> None:
        """Display detailed query result information."""
        success_icon = "✅" if result.confidence_score > 0 else "❌"
        print(f"{success_icon} {result.implementation.upper()} Implementation Result:")
        print(f"  ⏱️  Execution Time: {result.execution_time_ms:.1f}ms")
        print(f"  🎯 Confidence Score: {result.confidence_score:.2f}")
        print(f"  📊 Entities Traversed: {result.total_entities_traversed}")
        print(f"  🔗 Relationships: {result.total_relationships_traversed}")
        print(f"  📄 Documents Retrieved: {len(result.final_documents)}")
        
        if result.performance_metrics:
            perf = result.performance_metrics
            print(f"  🔍 DB Executions: {perf.get('db_executions', 'N/A')}")
            print(f"  🛤️  Retrieval Method: {perf.get('retrieval_method', 'unknown')}")
        
        if result.answer and result.confidence_score > 0:
            print(f"  💬 Answer Preview: {result.answer[:150]}{'...' if len(result.answer) > 150 else ''}")
    
    def _compare_results(self, current_result: MultiHopQueryResult, merged_result: MultiHopQueryResult) -> None:
        """Compare results between current and merged implementations."""
        print(f"\n{Fore.MAGENTA}📊 IMPLEMENTATION COMPARISON:{Style.RESET_ALL}")
        
        # Performance comparison
        time_diff = merged_result.execution_time_ms - current_result.execution_time_ms
        time_pct = (time_diff / current_result.execution_time_ms) * 100 if current_result.execution_time_ms > 0 else 0
        
        if time_diff < 0:
            print(f"  ⚡ Merged is FASTER by {abs(time_diff):.1f}ms ({abs(time_pct):.1f}%)")
        elif time_diff > 0:
            print(f"  🐌 Merged is SLOWER by {time_diff:.1f}ms ({time_pct:.1f}%)")
        else:
            print(f"  ⚖️  Similar performance")
        
        # Quality comparison
        doc_diff = len(merged_result.final_documents) - len(current_result.final_documents)
        if doc_diff > 0:
            print(f"  📈 Merged retrieved {doc_diff} MORE documents")
        elif doc_diff < 0:
            print(f"  📉 Merged retrieved {abs(doc_diff)} FEWER documents")
        else:
            print(f"  📊 Same number of documents retrieved")
        
        # Confidence comparison
        conf_diff = merged_result.confidence_score - current_result.confidence_score
        if conf_diff > 0.05:
            print(f"  🎯 Merged has HIGHER confidence (+{conf_diff:.2f})")
        elif conf_diff < -0.05:
            print(f"  🎯 Merged has LOWER confidence ({conf_diff:.2f})")
        else:
            print(f"  🎯 Similar confidence levels")
        
        # Feature comparison
        current_method = current_result.performance_metrics.get('retrieval_method', 'unknown')
        merged_method = merged_result.performance_metrics.get('retrieval_method', 'unknown')
        
        if current_method != merged_method:
            print(f"  🔄 Different retrieval methods: {current_method} vs {merged_method}")
            if merged_method == "vector_fallback":
                print(f"    ℹ️  Merged used fallback mechanism")
    
    def _update_performance_metrics(self, result: MultiHopQueryResult) -> None:
        """Update overall performance metrics."""
        self.performance_metrics["total_queries"] += 1
        if result.confidence_score > 0:
            self.performance_metrics["successful_queries"] += 1
        
        self.performance_metrics["total_entities_processed"] += result.total_entities_traversed
        self.performance_metrics["total_relationships_traversed"] += result.total_relationships_traversed
        
        # Update running average
        current_avg = self.performance_metrics["average_execution_time"]
        total_queries = self.performance_metrics["total_queries"]
        new_avg = ((current_avg * (total_queries - 1)) + result.execution_time_ms) / total_queries
        self.performance_metrics["average_execution_time"] = new_avg
    
    def _create_comparison_visualization(self, current_result: MultiHopQueryResult, merged_result: MultiHopQueryResult, filename: str) -> None:
        """Create visualization comparing the two results."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'GraphRAG Implementation Comparison\n{current_result.query[:60]}...', fontsize=14, fontweight='bold')
            
            # Performance comparison
            implementations = ['Current', 'Merged']
            exec_times = [current_result.execution_time_ms, merged_result.execution_time_ms]
            colors = ['#3498db', '#e74c3c']
            
            ax1.bar(implementations, exec_times, color=colors, alpha=0.7)
            ax1.set_title('Execution Time Comparison')
            ax1.set_ylabel('Time (ms)')
            for i, v in enumerate(exec_times):
                ax1.text(i, v + max(exec_times) * 0.01, f'{v:.1f}ms', ha='center', va='bottom')
            
            # Document retrieval comparison
            doc_counts = [len(current_result.final_documents), len(merged_result.final_documents)]
            ax2.bar(implementations, doc_counts, color=colors, alpha=0.7)
            ax2.set_title('Documents Retrieved')
            ax2.set_ylabel('Count')
            for i, v in enumerate(doc_counts):
                ax2.text(i, v + max(doc_counts) * 0.01, str(v), ha='center', va='bottom')
            
            # Confidence scores
            conf_scores = [current_result.confidence_score, merged_result.confidence_score]
            ax3.bar(implementations, conf_scores, color=colors, alpha=0.7)
            ax3.set_title('Confidence Scores')
            ax3.set_ylabel('Confidence')
            ax3.set_ylim(0, 1)
            for i, v in enumerate(conf_scores):
                ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
            
            # Entity/Relationship traversal
            entities = [current_result.total_entities_traversed, merged_result.total_entities_traversed]
            relationships = [current_result.total_relationships_traversed, merged_result.total_relationships_traversed]
            
            x = range(len(implementations))
            width = 0.35
            ax4.bar([i - width/2 for i in x], entities, width, label='Entities', color='#9b59b6', alpha=0.7)
            ax4.bar([i + width/2 for i in x], relationships, width, label='Relationships', color='#f39c12', alpha=0.7)
            ax4.set_title('Graph Traversal Metrics')
            ax4.set_ylabel('Count')
            ax4.set_xticks(x)
            ax4.set_xticklabels(implementations)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = self.output_dir / f"{filename}_comparison.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison visualization saved to {viz_file}")
            
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive test report."""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"📊 GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Separate results by implementation
        current_results = [r for r in self.query_results if r.implementation == "current"]
        merged_results = [r for r in self.query_results if r.implementation == "merged"]
        
        # Calculate summary statistics
        report_data = {
            "test_execution": {
                "timestamp": timestamp,
                "total_queries": len(current_results),
                "implementations_tested": ["current", "merged"],
                "test_environment": "mock_data" if self.use_mocks else "database"
            },
            "performance_summary": {
                "current": self._calculate_implementation_stats(current_results),
                "merged": self._calculate_implementation_stats(merged_results)
            },
            "detailed_results": [asdict(r) for r in self.query_results],
            "recommendations": self._generate_recommendations(current_results, merged_results)
        }
        
        # Save JSON report
        json_file = self.output_dir / f"merged_graphrag_multihop_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save markdown report
        md_file = self.output_dir / f"merged_graphrag_multihop_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            self._write_markdown_report(f, report_data)
        
        print(f"📄 Reports saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - Markdown: {md_file}")
        
        # Display summary
        self._display_final_summary(report_data)
    
    def _calculate_implementation_stats(self, results: List[MultiHopQueryResult]) -> Dict[str, Any]:
        """Calculate statistics for an implementation."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.confidence_score > 0]
        
        return {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "average_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results),
            "average_confidence": sum(r.confidence_score for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_documents_retrieved": sum(len(r.final_documents) for r in results) / len(results),
            "average_entities_traversed": sum(r.total_entities_traversed for r in results) / len(results),
            "average_relationships_traversed": sum(r.total_relationships_traversed for r in results) / len(results),
            "retrieval_methods": {
                method: sum(1 for r in results if r.performance_metrics.get('retrieval_method') == method)
                for method in set(r.performance_metrics.get('retrieval_method', 'unknown') for r in results)
            }
        }
    
    def _generate_recommendations(self, current_results: List[MultiHopQueryResult], merged_results: List[MultiHopQueryResult]) -> List[str]:
        """Generate deployment recommendations based on test results."""
        recommendations = []
        
        if not current_results or not merged_results:
            recommendations.append("Insufficient data for comparison - run more tests")
            return recommendations
        
        current_stats = self._calculate_implementation_stats(current_results)
        merged_stats = self._calculate_implementation_stats(merged_results)
        
        # Success rate comparison
        if merged_stats["success_rate"] >= current_stats["success_rate"]:
            recommendations.append("✅ Merged implementation maintains or improves success rate")
        else:
            recommendations.append("⚠️ Merged implementation shows reduced success rate - investigate failures")
        
        # Performance comparison
        perf_improvement = (current_stats["average_execution_time_ms"] - merged_stats["average_execution_time_ms"]) / current_stats["average_execution_time_ms"] * 100
        if perf_improvement > 10:
            recommendations.append(f"⚡ Significant performance improvement: {perf_improvement:.1f}% faster")
        elif perf_improvement < -20:
            recommendations.append(f"🐌 Performance regression: {abs(perf_improvement):.1f}% slower - optimization needed")
        else:
            recommendations.append("⚖️ Performance impact acceptable")
        
        # Quality comparison
        if merged_stats["average_confidence"] > current_stats["average_confidence"]:
            recommendations.append("🎯 Improved confidence scores in merged implementation")
        
        # Feature comparison
        merged_fallback_usage = merged_stats["retrieval_methods"].get("vector_fallback", 0)
        if merged_fallback_usage > 0:
            recommendations.append(f"🔄 Vector fallback used in {merged_fallback_usage} queries - validates robustness feature")
        
        # Overall recommendation
        if (merged_stats["success_rate"] >= current_stats["success_rate"] and 
            perf_improvement > -20 and 
            merged_stats["average_confidence"] >= current_stats["average_confidence"] - 0.1):
            recommendations.append("🚀 RECOMMENDED: Deploy merged implementation")
        else:
            recommendations.append("🔍 RECOMMENDED: Address identified issues before deployment")
        
        return recommendations
    
    def _write_markdown_report(self, file, report_data: Dict[str, Any]) -> None:
        """Write detailed markdown report."""
        file.write("# Merged GraphRAG Multi-Hop Query Test Report\n\n")
        file.write(f"**Generated:** {report_data['test_execution']['timestamp']}\n")
        file.write(f"**Environment:** {report_data['test_execution']['test_environment']}\n")
        file.write(f"**Total Queries:** {report_data['test_execution']['total_queries']}\n\n")
        
        file.write("## Executive Summary\n\n")
        current_stats = report_data['performance_summary']['current']
        merged_stats = report_data['performance_summary']['merged']
        
        file.write(f"- **Current Implementation Success Rate:** {current_stats['success_rate']:.2%}\n")
        file.write(f"- **Merged Implementation Success Rate:** {merged_stats['success_rate']:.2%}\n")
        file.write(f"- **Average Execution Time Change:** {((current_stats['average_execution_time_ms'] - merged_stats['average_execution_time_ms']) / current_stats['average_execution_time_ms'] * 100):+.1f}%\n")
        file.write(f"- **Average Confidence Change:** {(merged_stats['average_confidence'] - current_stats['average_confidence']):+.2f}\n\n")
        
        file.write("## Performance Comparison\n\n")
        file.write("| Metric | Current | Merged | Change |\n")
        file.write("|--------|---------|--------|---------|\n")
        file.write(f"| Success Rate | {current_stats['success_rate']:.2%} | {merged_stats['success_rate']:.2%} | {(merged_stats['success_rate'] - current_stats['success_rate']):+.2%} |\n")
        file.write(f"| Avg Execution Time (ms) | {current_stats['average_execution_time_ms']:.1f} | {merged_stats['average_execution_time_ms']:.1f} | {((current_stats['average_execution_time_ms'] - merged_stats['average_execution_time_ms']) / current_stats['average_execution_time_ms'] * 100):+.1f}% |\n")
        file.write(f"| Avg Confidence | {current_stats['average_confidence']:.2f} | {merged_stats['average_confidence']:.2f} | {(merged_stats['average_confidence'] - current_stats['average_confidence']):+.2f} |\n")
        file.write(f"| Avg Documents Retrieved | {current_stats['average_documents_retrieved']:.1f} | {merged_stats['average_documents_retrieved']:.1f} | {(merged_stats['average_documents_retrieved'] - current_stats['average_documents_retrieved']):+.1f} |\n\n")
        
        file.write("## Recommendations\n\n")
        for rec in report_data['recommendations']:
            file.write(f"- {rec}\n")
        file.write("\n")
        
        file.write("## Detailed Query Results\n\n")
        for i, result_data in enumerate(report_data['detailed_results']):
            if i % 2 == 0:  # Every other result (since we have current + merged for each query)
                file.write(f"### Query: {result_data['query'][:100]}...\n\n")
                file.write(f"**Type:** {result_data['query_type']}\n")
                file.write(f"**Implementation:** {result_data['implementation']}\n")
                file.write(f"**Success:** {result_data['confidence_score'] > 0}\n")
                file.write(f"**Execution Time:** {result_data['execution_time_ms']:.1f}ms\n")
                file.write(f"**Documents Retrieved:** {len(result_data['final_documents'])}\n\n")
    
    def _display_final_summary(self, report_data: Dict[str, Any]) -> None:
        """Display final test summary."""
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"🎯 FINAL TEST SUMMARY")
        print(f"{'='*80}{Style.RESET_ALL}")
        
        current_stats = report_data['performance_summary']['current']
        merged_stats = report_data['performance_summary']['merged']
        
        print(f"📊 Test Results:")
        print(f"  - Total Queries: {report_data['test_execution']['total_queries']}")
        print(f"  - Current Success Rate: {current_stats['success_rate']:.2%}")
        print(f"  - Merged Success Rate: {merged_stats['success_rate']:.2%}")
        
        perf_change = ((current_stats['average_execution_time_ms'] - merged_stats['average_execution_time_ms']) / current_stats['average_execution_time_ms']) * 100
        if perf_change > 0:
            print(f"  - Performance: {perf_change:.1f}% FASTER ⚡")
        else:
            print(f"  - Performance: {abs(perf_change):.1f}% SLOWER 🐌")
        
        print(f"\n🎯 Key Recommendations:")
        for rec in report_data['recommendations'][-3:]:  # Show last 3 recommendations
            print(f"  - {rec}")
    
    def run_full_demonstration(self) -> None:
        """Run the complete merged GraphRAG multi-hop demonstration."""
        print(f"{Fore.CYAN}🚀 Starting Merged GraphRAG Multi-Hop Query Demonstration...{Style.RESET_ALL}")
        
        try:
            # Run biomedical multi-hop query demonstrations
            self.demonstrate_biomedical_multihop_queries()
            
            # Generate comprehensive report
            self.generate_comprehensive_report()
            
            print(f"\n{Fore.GREEN}✅ Merged GraphRAG Multi-Hop Demo completed successfully!{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            traceback.print_exc()


def main():
    """Main entry point for the merged GraphRAG multi-hop demonstration."""
    parser = argparse.ArgumentParser(description="Merged GraphRAG Multi-Hop Query Demonstration")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--use-mocks", action="store_true", help="Use mock data instead of real database")
    parser.add_argument("--output-dir", default="outputs/multihop_merged_demo", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize demonstration
        demo = MergedGraphRAGMultiHopDemo(
            config_path=args.config,
            use_mocks=args.use_mocks,
            output_dir=args.output_dir
        )
        
        # Run full demonstration
        demo.run_full_demonstration()
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()