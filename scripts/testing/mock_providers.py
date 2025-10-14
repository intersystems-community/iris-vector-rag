"""
Mock providers for predictable testing of examples.

This module provides mock implementations of external services (LLM, APIs, etc.)
to enable consistent, fast, and reliable testing of examples and demos.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from iris_rag.core.models import Document


@dataclass
class MockResponse:
    """Response from a mock provider."""

    content: str
    metadata: Dict[str, Any]
    timestamp: float
    provider: str


class MockLLMProvider:
    """
    Mock LLM provider for deterministic testing.

    Provides realistic but predictable responses for testing RAG pipelines
    without requiring actual LLM API calls.
    """

    # Realistic response templates for common medical/scientific queries
    RESPONSE_TEMPLATES = {
        "diabetes": (
            "Diabetes is a chronic medical condition characterized by elevated blood glucose levels. "
            "There are two main types: Type 1 diabetes, which is an autoimmune condition where the "
            "pancreas produces little or no insulin, and Type 2 diabetes, which occurs when the body "
            "becomes resistant to insulin or doesn't produce enough insulin. Management includes "
            "monitoring blood sugar levels, medication (such as insulin or metformin), dietary changes, "
            "and regular exercise. Common symptoms include excessive thirst, frequent urination, "
            "fatigue, and blurred vision."
        ),
        "insulin": (
            "Insulin is a hormone produced by beta cells in the pancreas that regulates blood glucose "
            "levels by facilitating the uptake of glucose into cells. When blood sugar rises after "
            "eating, insulin is released to help cells absorb glucose for energy or storage. In "
            "diabetes, this process is impaired either due to insufficient insulin production or "
            "insulin resistance. Insulin therapy involves injecting synthetic insulin to manage "
            "blood glucose levels in people with diabetes."
        ),
        "cancer": (
            "Cancer is a group of diseases characterized by the uncontrolled growth and spread of "
            "abnormal cells. It can occur in virtually any part of the body and is caused by genetic "
            "mutations that disrupt normal cell division and growth. Treatment approaches include "
            "surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapy. Early "
            "detection through screening and awareness of warning signs significantly improves "
            "treatment outcomes and survival rates."
        ),
        "covid": (
            "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. It primarily spreads "
            "through respiratory droplets when an infected person coughs, sneezes, or talks. Symptoms "
            "range from mild (fever, cough, fatigue) to severe (difficulty breathing, pneumonia). "
            "Prevention measures include vaccination, wearing masks, social distancing, and good "
            "hand hygiene. Treatment depends on severity and may include supportive care, antiviral "
            "medications, and in severe cases, hospitalization."
        ),
        "protein": (
            "Proteins are complex biological molecules composed of amino acids that perform essential "
            "functions in living organisms. They serve as enzymes catalyzing biochemical reactions, "
            "structural components of cells and tissues, antibodies for immune defense, and signaling "
            "molecules. Protein structure has four levels: primary (amino acid sequence), secondary "
            "(local folding patterns), tertiary (overall 3D structure), and quaternary (multiple "
            "protein subunit arrangements). Protein folding is critical for function."
        ),
        "default": (
            "Based on the provided context, this is a comprehensive response that addresses the key "
            "aspects of the query. The information presented is synthesized from multiple reliable "
            "sources and provides accurate, evidence-based insights relevant to the question asked. "
            "This response demonstrates the effective retrieval and integration of contextual "
            "information to generate a meaningful answer."
        ),
    }

    def __init__(self, mode: str = "realistic", response_delay: float = 0.5):
        """
        Initialize mock LLM provider.

        Args:
            mode: Response mode - "realistic", "deterministic", "error", or "random"
            response_delay: Simulated response time in seconds
        """
        self.mode = mode
        self.response_delay = response_delay
        self.call_count = 0
        self.call_history: List[Dict] = []

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a mock response based on the prompt.

        Args:
            prompt: Input prompt to respond to
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            Generated response string

        Raises:
            Exception: If mode is "error"
        """
        self.call_count += 1
        start_time = time.time()

        # Simulate API delay
        if self.response_delay > 0:
            time.sleep(self.response_delay)

        # Handle different modes
        if self.mode == "error":
            raise Exception("Simulated LLM API error - rate limit exceeded")

        response = self._generate_contextual_response(prompt)

        # Log the call
        self.call_history.append(
            {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "timestamp": start_time,
                "duration": time.time() - start_time,
                "call_number": self.call_count,
            }
        )

        return response

    def _generate_contextual_response(self, prompt: str) -> str:
        """Generate contextually appropriate response based on prompt content."""
        prompt_lower = prompt.lower()

        # Find the most relevant template
        for keyword, template in self.RESPONSE_TEMPLATES.items():
            if keyword in prompt_lower:
                response = template

                # Add mode-specific variations
                if self.mode == "deterministic":
                    response += f" [Mock Response #{self.call_count}]"
                elif self.mode == "random":
                    # Add some randomness while keeping it realistic
                    variations = [
                        " This information is based on current medical knowledge.",
                        " Research continues to provide new insights in this area.",
                        " It's important to consult healthcare professionals for specific advice.",
                        " Multiple studies support these findings.",
                    ]
                    response += random.choice(variations)

                return response

        # Default response
        response = self.RESPONSE_TEMPLATES["default"]
        if self.mode == "deterministic":
            response += f" [Mock Response #{self.call_count}]"

        return response

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the mock provider."""
        return {
            "total_calls": self.call_count,
            "total_delay_time": sum(call["duration"] for call in self.call_history),
            "average_response_time": (
                sum(call["duration"] for call in self.call_history) / self.call_count
                if self.call_count > 0
                else 0
            ),
            "mode": self.mode,
            "recent_calls": self.call_history[-5:] if self.call_history else [],
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.call_count = 0
        self.call_history = []


class MockDataProvider:
    """
    Mock data provider for consistent test datasets.

    Provides sample documents and queries for testing RAG pipelines
    without requiring external data sources.
    """

    SAMPLE_DOCUMENTS = [
        Document(
            page_content=(
                "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels "
                "over a prolonged period. Type 1 diabetes results from the pancreas's inability to "
                "produce enough insulin. Type 2 diabetes begins with insulin resistance, where cells "
                "fail to respond to insulin properly. Without treatment, diabetes can cause many "
                "health complications including cardiovascular disease, stroke, chronic kidney disease, "
                "foot ulcers, and damage to the eyes."
            ),
            metadata={
                "source": "medical_textbook_diabetes.pdf",
                "page": 1,
                "section": "Endocrinology",
                "topic": "diabetes",
            },
        ),
        Document(
            page_content=(
                "Insulin is a peptide hormone produced by beta cells of the pancreatic islets. "
                "It regulates the metabolism of carbohydrates, fats and protein by promoting the "
                "absorption of glucose from the blood into liver, fat and skeletal muscle cells. "
                "The hormone acts by binding to the insulin receptor, which triggers a cascade of "
                "intracellular signaling pathways that ultimately lead to glucose uptake."
            ),
            metadata={
                "source": "biochemistry_journal_insulin.pdf",
                "page": 3,
                "section": "Molecular Biology",
                "topic": "insulin",
            },
        ),
        Document(
            page_content=(
                "SARS-CoV-2 is the virus that causes COVID-19. It primarily spreads through respiratory "
                "droplets and aerosols when an infected person coughs, sneezes, speaks, or breathes. "
                "The virus can also spread by touching contaminated surfaces and then touching the face. "
                "Symptoms can range from mild to severe and may include fever, cough, shortness of "
                "breath, fatigue, muscle aches, headache, loss of taste or smell, sore throat, and "
                "gastrointestinal symptoms."
            ),
            metadata={
                "source": "who_covid19_guidelines.pdf",
                "page": 2,
                "section": "Infectious Diseases",
                "topic": "covid",
            },
        ),
        Document(
            page_content=(
                "Proteins are large biological molecules consisting of chains of amino acids. They "
                "perform many functions including catalyzing metabolic reactions, DNA replication, "
                "responding to stimuli, providing structure to cells and organisms, and transporting "
                "molecules from one location to another. The sequence of amino acids in a protein is "
                "defined by the sequence of a gene, which is encoded in the genetic code."
            ),
            metadata={
                "source": "molecular_biology_proteins.pdf",
                "page": 5,
                "section": "Structural Biology",
                "topic": "protein",
            },
        ),
        Document(
            page_content=(
                "Cancer is a group of diseases involving abnormal cell growth with the potential to "
                "invade or spread to other parts of the body. These contrast with benign tumors, "
                "which do not spread. Possible signs and symptoms include a lump, abnormal bleeding, "
                "prolonged cough, unexplained weight loss, and a change in bowel movements. Treatment "
                "may include surgery, chemotherapy, radiation therapy, targeted therapy, and immunotherapy."
            ),
            metadata={
                "source": "oncology_overview.pdf",
                "page": 1,
                "section": "Oncology",
                "topic": "cancer",
            },
        ),
    ]

    TEST_QUERIES = {
        "medical": [
            "What is diabetes?",
            "How does insulin work?",
            "What are the symptoms of COVID-19?",
            "What causes cancer?",
            "How do proteins function in cells?",
        ],
        "complex": [
            "What is the relationship between insulin resistance and type 2 diabetes?",
            "How do COVID-19 vaccines work to prevent infection?",
            "What are the different types of cancer treatments and their mechanisms?",
            "How does protein folding affect enzyme function?",
            "What are the long-term complications of uncontrolled diabetes?",
        ],
        "comparative": [
            "What are the differences between type 1 and type 2 diabetes?",
            "How do DNA and RNA differ in structure and function?",
            "What are the advantages and disadvantages of chemotherapy versus immunotherapy?",
            "How do vaccines differ from antiviral medications?",
            "What distinguishes acute from chronic diseases?",
        ],
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock data provider.

        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            random.seed(seed)

    def get_sample_documents(
        self, count: int = None, topic: str = None
    ) -> List[Document]:
        """
        Get sample documents for testing.

        Args:
            count: Number of documents to return (None for all)
            topic: Filter by topic (diabetes, insulin, covid, etc.)

        Returns:
            List of sample documents
        """
        documents = self.SAMPLE_DOCUMENTS.copy()

        if topic:
            documents = [doc for doc in documents if doc.metadata.get("topic") == topic]

        if count is not None:
            documents = documents[:count]

        return documents

    def get_test_queries(
        self, category: str = "medical", count: int = None
    ) -> List[str]:
        """
        Get test queries for evaluation.

        Args:
            category: Query category (medical, complex, comparative)
            count: Number of queries to return (None for all)

        Returns:
            List of test queries
        """
        queries = self.TEST_QUERIES.get(category, self.TEST_QUERIES["medical"]).copy()

        if count is not None:
            queries = queries[:count]

        return queries

    def get_realistic_query_answer_pairs(self) -> List[Dict[str, str]]:
        """Get realistic query-answer pairs for validation testing."""
        return [
            {
                "query": "What is diabetes?",
                "expected_keywords": [
                    "blood sugar",
                    "glucose",
                    "insulin",
                    "pancreas",
                    "type 1",
                    "type 2",
                ],
                "topic": "diabetes",
            },
            {
                "query": "How does insulin work?",
                "expected_keywords": [
                    "hormone",
                    "glucose",
                    "cells",
                    "blood",
                    "regulation",
                    "pancreas",
                ],
                "topic": "insulin",
            },
            {
                "query": "What are COVID-19 symptoms?",
                "expected_keywords": [
                    "fever",
                    "cough",
                    "breathing",
                    "fatigue",
                    "virus",
                    "SARS-CoV-2",
                ],
                "topic": "covid",
            },
        ]
