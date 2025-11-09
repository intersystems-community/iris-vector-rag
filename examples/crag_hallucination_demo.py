#!/usr/bin/env python3
"""
CRAG Hallucination Detection and Correction Demo

Real-time demonstration showing how CRAG (Corrective RAG) identifies and fixes
hallucinations compared to basic RAG.

This script provides an interactive side-by-side comparison demonstrating:
- Factual error detection and correction
- Temporal inconsistency fixes
- Entity confusion resolution
- Numerical accuracy improvements
- Causal misattribution corrections
- Overgeneralization prevention

Usage:
    python examples/crag_hallucination_demo.py --interactive
    python examples/crag_hallucination_demo.py --test-cases examples/hallucination_test_cases.json
    python examples/crag_hallucination_demo.py --medical-domain --export-html
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from colorama import Back, Fore, Style, init

    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    # Fallback color placeholders
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

    class Style:
        BRIGHT = DIM = RESET_ALL = ""


import os

# Import RAG pipelines
import sys

import openai
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import iris_vector_rag
from iris_vector_rag.core.models import Document

# Load environment variables
load_dotenv()


class HallucinationType(Enum):
    """Types of hallucinations that CRAG can detect and correct."""

    FACTUAL_ERROR = "factual_error"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    ENTITY_CONFUSION = "entity_confusion"
    NUMERICAL_INACCURACY = "numerical_inaccuracy"
    CAUSAL_MISATTRIBUTION = "causal_misattribution"
    OVERGENERALIZATION = "overgeneralization"
    UNSUPPORTED_CLAIM = "unsupported_claim"


@dataclass
class HallucinationDetection:
    """Represents a detected hallucination with correction information."""

    type: HallucinationType
    original_text: str
    corrected_text: str
    confidence_score: float
    evidence_source: str
    explanation: str
    start_pos: int
    end_pos: int


@dataclass
class CorrectionMetrics:
    """Performance metrics for hallucination detection and correction."""

    total_queries: int = 0
    hallucinations_detected: int = 0
    corrections_made: int = 0
    false_positives: int = 0
    processing_time_ms: float = 0.0
    confidence_avg: float = 0.0

    @property
    def detection_rate(self) -> float:
        return (
            (self.hallucinations_detected / self.total_queries)
            if self.total_queries > 0
            else 0.0
        )

    @property
    def correction_accuracy(self) -> float:
        return (
            (self.corrections_made / self.hallucinations_detected)
            if self.hallucinations_detected > 0
            else 0.0
        )


class HallucinationDetector:
    """Detects various types of hallucinations in RAG responses."""

    def __init__(self):
        self.patterns = self._load_detection_patterns()

    def _load_detection_patterns(self) -> Dict[HallucinationType, List[str]]:
        """Load regex patterns for detecting different hallucination types."""
        return {
            HallucinationType.FACTUAL_ERROR: [
                r"(?i)\b(always|never|all|every|none|nobody)\b.*\b(cure|treat|prevent)\b",
                r"(?i)\b(100%|completely|totally|absolutely) (effective|safe|harmless)\b",
                r"(?i)\b(no side effects|perfectly safe|zero risk)\b",
            ],
            HallucinationType.TEMPORAL_INCONSISTENCY: [
                r"(?i)\b(discovered|invented|created) in (\d{4})\b",
                r"(?i)\b(since|from|until) (\d{4})\b",
                r"(?i)\b(recently|new|latest|modern)\b.*\b(ancient|old|traditional)\b",
            ],
            HallucinationType.NUMERICAL_INACCURACY: [
                r"\b(\d+(?:\.\d+)?)\s*%\b",
                r"\b(\d+(?:,\d{3})*|\d+) (deaths|cases|patients|people)\b",
                r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)\b",
            ],
            HallucinationType.OVERGENERALIZATION: [
                r"(?i)\b(all|every|any) (cancer|disease|condition|treatment)\b",
                r"(?i)\b(works for|treats|cures) (everything|all conditions|any disease)\b",
            ],
            HallucinationType.ENTITY_CONFUSION: [
                r"(?i)\b(insulin|aspirin|metformin)\b.*\b(for|treats|cures)\b.*\b(diabetes|cancer|heart disease)\b"
            ],
        }

    def detect_hallucinations(
        self, text: str, retrieved_docs: List[Any]
    ) -> List[HallucinationDetection]:
        """
        Detect hallucinations in the given text based on retrieved documents.

        Args:
            text: The generated text to analyze
            retrieved_docs: Documents used for retrieval (can be Document objects or dicts)

        Returns:
            List of detected hallucinations with correction information
        """
        detections = []
        # Handle both Document objects and dictionaries
        doc_contents = []
        for doc in retrieved_docs:
            if hasattr(doc, "page_content"):
                doc_contents.append(doc.page_content)
            elif isinstance(doc, dict) and "content" in doc:
                doc_contents.append(doc["content"])
            elif isinstance(doc, dict) and "page_content" in doc:
                doc_contents.append(doc["page_content"])
            else:
                doc_contents.append(str(doc))

        for halluc_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    detection = self._analyze_match(
                        match, halluc_type, text, doc_contents
                    )
                    if detection:
                        detections.append(detection)

        # Additional content-based validation
        detections.extend(self._detect_unsupported_claims(text, doc_contents))

        return detections

    def _analyze_match(
        self,
        match: re.Match,
        halluc_type: HallucinationType,
        text: str,
        doc_contents: List[str],
    ) -> Optional[HallucinationDetection]:
        """Analyze a pattern match to determine if it's a hallucination."""
        original_text = match.group(0)
        start_pos = match.start()
        end_pos = match.end()

        # Check if claim is supported by retrieved documents
        is_supported = self._is_claim_supported(original_text, doc_contents)

        if not is_supported:
            corrected_text, explanation = self._generate_correction(
                original_text, halluc_type, doc_contents
            )

            confidence = self._calculate_confidence(original_text, doc_contents)

            return HallucinationDetection(
                type=halluc_type,
                original_text=original_text,
                corrected_text=corrected_text,
                confidence_score=confidence,
                evidence_source="Pattern matching + document validation",
                explanation=explanation,
                start_pos=start_pos,
                end_pos=end_pos,
            )

        return None

    def _is_claim_supported(self, claim: str, doc_contents: List[str]) -> bool:
        """Check if a claim is supported by the retrieved documents."""
        claim_lower = claim.lower()

        # Simple keyword overlap check (can be enhanced with semantic similarity)
        for doc in doc_contents:
            doc_lower = doc.lower()
            # Check for keyword overlap
            claim_words = set(re.findall(r"\b\w+\b", claim_lower))
            doc_words = set(re.findall(r"\b\w+\b", doc_lower))

            overlap = len(claim_words.intersection(doc_words))
            if overlap >= max(2, len(claim_words) * 0.4):  # At least 40% overlap
                return True

        return False

    def _generate_correction(
        self, original: str, halluc_type: HallucinationType, doc_contents: List[str]
    ) -> Tuple[str, str]:
        """Generate a correction for the detected hallucination."""
        corrections = {
            HallucinationType.FACTUAL_ERROR: (
                f"Based on available evidence, {original.lower()}",
                "Removed absolute claims without sufficient evidence",
            ),
            HallucinationType.OVERGENERALIZATION: (
                f"Some {original.lower().replace('all ', '').replace('every ', '')}",
                "Qualified overly broad generalization",
            ),
            HallucinationType.NUMERICAL_INACCURACY: (
                "approximately " + original,
                "Added qualifier to indicate uncertainty in numerical claims",
            ),
            HallucinationType.TEMPORAL_INCONSISTENCY: (
                f"According to available sources, {original}",
                "Added source qualifier for temporal claims",
            ),
            HallucinationType.ENTITY_CONFUSION: (
                f"[Correction needed: verify specific medication/condition relationship for '{original}']",
                "Flagged potential entity confusion for verification",
            ),
        }

        return corrections.get(
            halluc_type, (original, "No specific correction available")
        )

    def _detect_unsupported_claims(
        self, text: str, doc_contents: List[str]
    ) -> List[HallucinationDetection]:
        """Detect claims that are not supported by retrieved documents."""
        detections = []

        # Split text into sentences
        sentences = re.split(r"[.!?]+", text)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            if not self._is_claim_supported(sentence, doc_contents):
                # This is a potential unsupported claim
                detection = HallucinationDetection(
                    type=HallucinationType.UNSUPPORTED_CLAIM,
                    original_text=sentence,
                    corrected_text=f"[Citation needed: {sentence}]",
                    confidence_score=0.7,
                    evidence_source="Document content analysis",
                    explanation="Statement not sufficiently supported by retrieved documents",
                    start_pos=text.find(sentence),
                    end_pos=text.find(sentence) + len(sentence),
                )
                detections.append(detection)

        return detections

    def _calculate_confidence(self, text: str, doc_contents: List[str]) -> float:
        """Calculate confidence score for hallucination detection."""
        # Simple confidence calculation based on document support
        text_words = set(re.findall(r"\b\w+\b", text.lower()))

        max_overlap = 0
        for doc in doc_contents:
            doc_words = set(re.findall(r"\b\w+\b", doc.lower()))
            overlap = len(text_words.intersection(doc_words))
            max_overlap = max(max_overlap, overlap)

        # Higher overlap = lower confidence in hallucination
        if len(text_words) == 0:
            return 0.5

        overlap_ratio = max_overlap / len(text_words)
        confidence = max(0.1, 1.0 - overlap_ratio)

        return round(confidence, 2)


class CorrectionVisualizer:
    """Visualizes corrections made by CRAG with color-coded output."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and COLORS_AVAILABLE

    def show_side_by_side_comparison(
        self,
        query: str,
        basic_result: Dict[str, Any],
        crag_result: Dict[str, Any],
        detections: List[HallucinationDetection],
    ) -> str:
        """Display side-by-side comparison of BasicRAG vs CRAG results."""
        output = []

        # Header
        output.append("=" * 80)
        output.append(f"HALLUCINATION DETECTION DEMONSTRATION")
        output.append("=" * 80)
        output.append(f"Query: {self._colorize(query, Fore.CYAN, Style.BRIGHT)}")
        output.append("")

        # Basic RAG Result
        output.append(
            f"{self._colorize('BASIC RAG RESPONSE:', Fore.RED, Style.BRIGHT)}"
        )
        output.append("-" * 40)
        basic_answer = basic_result.get("answer", "No answer generated")
        output.append(self._wrap_text(basic_answer, 75))
        output.append("")

        # CRAG Result with corrections highlighted
        output.append(
            f"{self._colorize('CRAG CORRECTED RESPONSE:', Fore.GREEN, Style.BRIGHT)}"
        )
        output.append("-" * 40)
        crag_answer = crag_result.get("answer", "No answer generated")
        highlighted_answer = self._highlight_corrections(crag_answer, detections)
        output.append(self._wrap_text(highlighted_answer, 75))
        output.append("")

        # Detailed corrections
        if detections:
            output.append(
                f"{self._colorize('CORRECTIONS MADE:', Fore.YELLOW, Style.BRIGHT)}"
            )
            output.append("-" * 40)
            for i, detection in enumerate(detections, 1):
                output.append(
                    f"{i}. {self._colorize(detection.type.value.upper(), Fore.MAGENTA, Style.BRIGHT)}"
                )
                output.append(
                    f"   Original: {self._colorize(detection.original_text, Fore.RED)}"
                )
                output.append(
                    f"   Corrected: {self._colorize(detection.corrected_text, Fore.GREEN)}"
                )
                output.append(f"   Confidence: {detection.confidence_score:.2f}")
                output.append(f"   Explanation: {detection.explanation}")
                output.append("")
        else:
            output.append(
                f"{self._colorize('No hallucinations detected.', Fore.GREEN)}"
            )
            output.append("")

        # Performance metrics
        basic_time = basic_result.get("execution_time", 0)
        crag_time = crag_result.get("execution_time", 0)

        output.append(
            f"{self._colorize('PERFORMANCE METRICS:', Fore.BLUE, Style.BRIGHT)}"
        )
        output.append("-" * 40)
        output.append(f"Basic RAG execution time: {basic_time:.3f}s")
        output.append(f"CRAG execution time: {crag_time:.3f}s")
        output.append(
            f"Processing overhead: {(crag_time - basic_time):.3f}s ({((crag_time/basic_time - 1) * 100):.1f}% increase)"
        )
        output.append(f"Hallucinations detected: {len(detections)}")
        output.append("")

        return "\n".join(output)

    def _highlight_corrections(
        self, text: str, detections: List[HallucinationDetection]
    ) -> str:
        """Highlight corrected portions in the text."""
        # Handle None text (when CRAG pipeline fails)
        if text is None:
            return "No response available (pipeline failed)"

        if not detections:
            return text

        # Sort detections by position (reverse order for replacement)
        sorted_detections = sorted(detections, key=lambda d: d.start_pos, reverse=True)

        highlighted_text = text
        for detection in sorted_detections:
            # Replace original with highlighted corrected version
            start = detection.start_pos
            end = detection.end_pos

            if 0 <= start < len(highlighted_text) and start < end <= len(
                highlighted_text
            ):
                correction = self._colorize(
                    detection.corrected_text, Fore.GREEN, Back.BLACK
                )
                highlighted_text = (
                    highlighted_text[:start] + correction + highlighted_text[end:]
                )

        return highlighted_text

    def _colorize(
        self, text: str, fore_color: str = "", back_color: str = "", style: str = ""
    ) -> str:
        """Apply color formatting if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{style}{fore_color}{back_color}{text}{Style.RESET_ALL}"

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def generate_html_report(
        self, results: List[Dict[str, Any]], output_path: str
    ) -> None:
        """Generate an HTML report with all demonstration results."""
        html_content = self._create_html_template()

        results_html = ""
        for i, result in enumerate(results, 1):
            results_html += self._create_result_html(result, i)

        html_content = html_content.replace("{{RESULTS}}", results_html)
        html_content = html_content.replace("{{TIMESTAMP}}", datetime.now().isoformat())
        html_content = html_content.replace("{{TOTAL_QUERIES}}", str(len(results)))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML report generated: {output_path}")

    def _create_html_template(self) -> str:
        """Create HTML template for the report."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>CRAG Hallucination Detection Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f8ff; padding: 20px; border-radius: 5px; }
        .comparison { display: flex; gap: 20px; margin: 20px 0; }
        .basic-rag, .crag { flex: 1; padding: 15px; border-radius: 5px; }
        .basic-rag { background: #ffe6e6; border-left: 5px solid #ff4444; }
        .crag { background: #e6ffe6; border-left: 5px solid #44ff44; }
        .corrections { background: #fff9e6; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .correction-item { margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }
        .metrics { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .original { color: #cc0000; text-decoration: line-through; }
        .corrected { color: #008800; font-weight: bold; }
        .query { font-size: 18px; font-weight: bold; color: #0066cc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>CRAG Hallucination Detection and Correction Report</h1>
        <p>Generated: {{TIMESTAMP}}</p>
        <p>Total Queries Analyzed: {{TOTAL_QUERIES}}</p>
    </div>
    
    {{RESULTS}}
</body>
</html>
        """

    def _create_result_html(self, result: Dict[str, Any], index: int) -> str:
        """Create HTML for a single result."""
        query = result.get("query", "Unknown query")
        basic_answer = result.get("basic_answer", "No answer")
        crag_answer = result.get("crag_answer", "No answer")
        detections = result.get("detections", [])

        corrections_html = ""
        if detections:
            corrections_html = "<div class='corrections'><h4>Corrections Made:</h4>"
            for detection in detections:
                corrections_html += f"""
                <div class='correction-item'>
                    <strong>{detection['type']}</strong><br>
                    Original: <span class='original'>{detection['original_text']}</span><br>
                    Corrected: <span class='corrected'>{detection['corrected_text']}</span><br>
                    Confidence: {detection['confidence_score']:.2f}<br>
                    <em>{detection['explanation']}</em>
                </div>
                """
            corrections_html += "</div>"
        else:
            corrections_html = (
                "<div class='corrections'>No hallucinations detected.</div>"
            )

        return f"""
        <div style="border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px;">
            <h3>Query {index}</h3>
            <div class="query">{query}</div>
            
            <div class="comparison">
                <div class="basic-rag">
                    <h4>Basic RAG Response</h4>
                    <p>{basic_answer}</p>
                </div>
                <div class="crag">
                    <h4>CRAG Corrected Response</h4>
                    <p>{crag_answer}</p>
                </div>
            </div>
            
            {corrections_html}
            
            <div class="metrics">
                <h4>Performance Metrics</h4>
                <p>Basic RAG Time: {result.get('basic_time', 0):.3f}s</p>
                <p>CRAG Time: {result.get('crag_time', 0):.3f}s</p>
                <p>Hallucinations Detected: {len(detections)}</p>
            </div>
        </div>
        """


class CRAGHallucinationDemo:
    """Main demonstration class for CRAG hallucination detection and correction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detector = HallucinationDetector()
        self.visualizer = CorrectionVisualizer(
            use_colors=self.config.get("use_colors", True)
        )
        self.metrics = CorrectionMetrics()
        self.results_history: List[Dict[str, Any]] = []

        # Setup logging first
        self._setup_logging()

        # Initialize RAG pipelines
        self.basic_rag = None
        self.crag = None
        self._setup_pipelines()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _setup_pipelines(self):
        """Initialize real iris_rag pipelines with database connections."""
        try:
            # Setup LLM function for pipelines
            llm_func = self._get_llm_function()

            # Create BasicRAG pipeline
            self.logger.info("Creating BasicRAG pipeline...")
            self.basic_rag = iris_rag.create_pipeline(
                pipeline_type="basic",
                llm_func=llm_func,
                auto_setup=True,
                validate_requirements=True,
            )
            self.logger.info("✓ BasicRAG pipeline created successfully")

            # Create CRAG pipeline
            self.logger.info("Creating CRAG pipeline...")
            self.crag = iris_rag.create_pipeline(
                pipeline_type="crag",
                llm_func=llm_func,
                auto_setup=True,
                validate_requirements=True,
            )
            self.logger.info("✓ CRAG pipeline created successfully")

            self.logger.info(
                "Real iris_rag pipelines initialized with database connections"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize iris_rag pipelines: {e}")
            raise RuntimeError(
                f"Could not initialize RAG pipelines for demonstration: {e}"
            )

    def _get_llm_function(self):
        """Get LLM function for pipeline initialization."""
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key:
            try:
                client = openai.OpenAI()

                def openai_llm(prompt: str) -> str:
                    """Real LLM function using OpenAI."""
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful medical assistant. Provide accurate, evidence-based information and acknowledge uncertainty when appropriate.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,  # Low temperature for more consistent responses
                    )
                    return response.choices[0].message.content.strip()

                # Test the LLM function
                test_response = openai_llm("Test prompt")
                self.logger.info("Using OpenAI LLM for demonstration")
                return openai_llm

            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI LLM: {e}")

        # Fallback to dummy LLM for demonstration
        self.logger.warning(
            "Using dummy LLM - set OPENAI_API_KEY for real LLM responses"
        )

        def dummy_llm(prompt: str) -> str:
            """Dummy LLM that creates realistic medical responses for demonstration."""
            # Extract query from prompt
            query_match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else "medical query"

            # Create basic vs corrected responses based on query
            if "mortality rate" in query.lower() and "covid" in query.lower():
                if "basic" in prompt.lower():
                    return "COVID-19 has a 100% mortality rate in all patients and always causes death within 24 hours."
                else:
                    return "COVID-19 mortality rates vary significantly based on factors like age, underlying conditions, and healthcare access. Current estimates suggest an overall mortality rate of approximately 1-3% globally, with higher rates in older adults and those with comorbidities."

            elif "aspirin" in query.lower() and "diabetes" in query.lower():
                if "basic" in prompt.lower():
                    return "Aspirin completely cures diabetes and eliminates all symptoms within one week of treatment."
                else:
                    return "Aspirin does not cure diabetes. While aspirin may be prescribed for diabetic patients to reduce cardiovascular risk, diabetes management requires appropriate medication like insulin or metformin, along with lifestyle modifications."

            else:
                # Generic response that acknowledges uncertainty
                return f"Based on the available evidence, treatment approaches for conditions related to '{query}' vary in effectiveness and should be evaluated based on individual patient factors and current medical evidence. Further consultation with healthcare providers is recommended."

        return dummy_llm

    def demonstrate_correction(self, query: str) -> Dict[str, Any]:
        """
        Demonstrate hallucination detection and correction for a single query.

        Args:
            query: The query to demonstrate with

        Returns:
            Dictionary containing demonstration results
        """
        self.logger.info(f"Demonstrating correction for query: {query}")

        start_time = time.time()

        # Get responses from both pipelines
        basic_result = self.basic_rag.query(query)
        crag_result = self.crag.query(query)

        # Detect hallucinations in basic RAG response
        basic_docs = basic_result.get("retrieved_documents", [])
        detections = self.detector.detect_hallucinations(
            basic_result.get("answer", ""), basic_docs
        )

        # Update metrics
        self.metrics.total_queries += 1
        self.metrics.hallucinations_detected += len(detections)
        self.metrics.corrections_made += len(
            [d for d in detections if d.corrected_text != d.original_text]
        )
        self.metrics.processing_time_ms += (time.time() - start_time) * 1000
        if detections:
            self.metrics.confidence_avg = sum(
                d.confidence_score for d in detections
            ) / len(detections)

        # Prepare result
        result = {
            "query": query,
            "basic_answer": basic_result.get("answer", ""),
            "crag_answer": crag_result.get("answer", ""),
            "detections": [asdict(d) for d in detections],
            "basic_time": basic_result.get("execution_time", 0),
            "crag_time": crag_result.get("execution_time", 0),
            "retrieved_documents": len(basic_docs),
            "timestamp": datetime.now().isoformat(),
        }

        self.results_history.append(result)

        return result

    def run_interactive_demo(self):
        """Run an interactive demonstration session."""
        print(
            self.visualizer._colorize(
                "🔬 CRAG Hallucination Detection Interactive Demo",
                Fore.CYAN,
                Style.BRIGHT,
            )
        )
        print("=" * 60)
        print("Enter queries to see real-time hallucination detection and correction.")
        print(
            "Type 'medical' for medical examples, 'metrics' for performance stats, or 'quit' to exit."
        )
        print("")

        while True:
            try:
                query = input(f"{Fore.YELLOW}Enter query: {Style.RESET_ALL}").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    break
                elif query.lower() == "medical":
                    self._run_medical_examples()
                    continue
                elif query.lower() == "metrics":
                    self._show_performance_metrics()
                    continue
                elif not query:
                    continue

                # Demonstrate correction
                result = self.demonstrate_correction(query)

                # Show visualization
                detections = [HallucinationDetection(**d) for d in result["detections"]]
                comparison = self.visualizer.show_side_by_side_comparison(
                    query,
                    {
                        "answer": result["basic_answer"],
                        "execution_time": result["basic_time"],
                    },
                    {
                        "answer": result["crag_answer"],
                        "execution_time": result["crag_time"],
                    },
                    detections,
                )

                print("\n" + comparison)
                print("\n" + "=" * 60 + "\n")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Demo interrupted by user{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

        self._show_final_summary()

    def _run_medical_examples(self):
        """Run predefined medical domain examples."""
        medical_queries = [
            "What is the mortality rate of COVID-19?",
            "Can aspirin cure diabetes?",
            "What are the side effects of metformin?",
            "How effective is chemotherapy for all cancers?",
            "When was insulin discovered?",
        ]

        print(f"\n{Fore.CYAN}Running Medical Domain Examples:{Style.RESET_ALL}")
        print("-" * 40)

        for i, query in enumerate(medical_queries, 1):
            print(f"\n{Fore.YELLOW}Example {i}: {query}{Style.RESET_ALL}")

            result = self.demonstrate_correction(query)
            detections = [HallucinationDetection(**d) for d in result["detections"]]

            comparison = self.visualizer.show_side_by_side_comparison(
                query,
                {
                    "answer": result["basic_answer"],
                    "execution_time": result["basic_time"],
                },
                {
                    "answer": result["crag_answer"],
                    "execution_time": result["crag_time"],
                },
                detections,
            )

            print(comparison)
            print("\n" + "-" * 60)

            # Brief pause between examples
            time.sleep(1)

    def _show_performance_metrics(self):
        """Display current performance metrics."""
        print(f"\n{Fore.BLUE}PERFORMANCE METRICS:{Style.RESET_ALL}")
        print("-" * 30)
        print(f"Total Queries: {self.metrics.total_queries}")
        print(f"Hallucinations Detected: {self.metrics.hallucinations_detected}")
        print(f"Corrections Made: {self.metrics.corrections_made}")
        print(f"Detection Rate: {self.metrics.detection_rate:.2%}")
        print(f"Correction Accuracy: {self.metrics.correction_accuracy:.2%}")
        print(f"Average Processing Time: {self.metrics.processing_time_ms:.1f}ms")
        print(f"Average Confidence: {self.metrics.confidence_avg:.2f}")
        print("")

    def _show_final_summary(self):
        """Show final summary of the demonstration session."""
        print(f"\n{Fore.GREEN}DEMONSTRATION SUMMARY:{Style.RESET_ALL}")
        print("=" * 40)
        self._show_performance_metrics()

        if self.results_history:
            print(
                f"Session completed with {len(self.results_history)} queries analyzed."
            )

            # Offer to export results
            export = (
                input(f"{Fore.YELLOW}Export results to HTML? (y/N): {Style.RESET_ALL}")
                .strip()
                .lower()
            )
            if export == "y":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"crag_demo_results_{timestamp}.html"
                self.visualizer.generate_html_report(self.results_history, output_path)

    def run_test_cases(self, test_cases_file: str):
        """Run demonstration with predefined test cases from JSON file."""
        try:
            with open(test_cases_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)

            test_cases = test_data.get("test_cases", [])

            print(f"Running {len(test_cases)} test cases from {test_cases_file}")
            print("=" * 60)

            for i, test_case in enumerate(test_cases, 1):
                query = test_case.get("query", "")
                expected_hallucinations = test_case.get("expected_hallucinations", [])

                print(f"\nTest Case {i}: {query}")
                print(
                    f"Expected hallucination types: {', '.join(expected_hallucinations)}"
                )
                print("-" * 40)

                result = self.demonstrate_correction(query)
                detections = [HallucinationDetection(**d) for d in result["detections"]]

                # Show results
                comparison = self.visualizer.show_side_by_side_comparison(
                    query,
                    {
                        "answer": result["basic_answer"],
                        "execution_time": result["basic_time"],
                    },
                    {
                        "answer": result["crag_answer"],
                        "execution_time": result["crag_time"],
                    },
                    detections,
                )

                print(comparison)

                # Validate detection accuracy
                detected_types = [d.type.value for d in detections]
                print(
                    f"Detected: {', '.join(detected_types) if detected_types else 'None'}"
                )

                print("\n" + "=" * 60)

        except FileNotFoundError:
            print(f"Test cases file not found: {test_cases_file}")
        except json.JSONDecodeError as e:
            print(f"Error parsing test cases file: {e}")
        except Exception as e:
            print(f"Error running test cases: {e}")


def main():
    """Main entry point for the demonstration script."""
    parser = argparse.ArgumentParser(
        description="CRAG Hallucination Detection and Correction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python crag_hallucination_demo.py --interactive
    python crag_hallucination_demo.py --test-cases hallucination_test_cases.json
    python crag_hallucination_demo.py --medical-domain --export-html
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive demonstration mode",
    )

    parser.add_argument(
        "--test-cases", "-t", type=str, help="Path to JSON file with test cases"
    )

    parser.add_argument(
        "--medical-domain",
        "-m",
        action="store_true",
        help="Run medical domain examples",
    )

    parser.add_argument(
        "--export-html", "-e", action="store_true", help="Export results to HTML report"
    )

    parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output"
    )

    parser.add_argument(
        "--config", "-c", type=str, help="Path to configuration JSON file"
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")

    config["use_colors"] = not args.no_colors

    # Initialize demonstration
    demo = CRAGHallucinationDemo(config)

    try:
        if args.interactive:
            demo.run_interactive_demo()
        elif args.test_cases:
            demo.run_test_cases(args.test_cases)
        elif args.medical_domain:
            demo._run_medical_examples()
            if args.export_html:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"crag_medical_demo_{timestamp}.html"
                demo.visualizer.generate_html_report(demo.results_history, output_path)
        else:
            # Default: run a quick demonstration
            demo._run_medical_examples()
            demo._show_final_summary()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Demo interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error running demonstration: {e}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    main()
