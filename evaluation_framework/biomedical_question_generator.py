"""
Biomedical Question Generation System for RAGAS Evaluation

This module generates diverse, high-quality biomedical questions with ground truth answers
from PMC document collections for comprehensive RAG pipeline evaluation.

Design Principles:
1. Domain-specific biomedical expertise
2. Question diversity across complexity levels
3. Ground truth validation from source documents
4. Scalable generation for 10K+ documents
5. Statistical sampling for balanced evaluation
"""

import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


@dataclass
class BiomedicalQuestion:
    """Structured biomedical question with metadata and ground truth."""

    question: str
    ground_truth_answer: str
    question_type: str  # factual, analytical, procedural, comparative, causal
    complexity_level: str  # basic, intermediate, advanced
    medical_domain: str  # cardiology, oncology, neurology, etc.
    source_doc_ids: List[str]
    source_passages: List[str]
    keywords: List[str]
    confidence_score: float
    question_id: str


@dataclass
class QuestionGenerationConfig:
    """Configuration for question generation process."""

    total_questions: int = 1000
    questions_per_document: int = 3
    min_answer_length: int = 50
    max_answer_length: int = 500
    min_confidence_score: float = 0.7
    question_types_distribution: Dict[str, float] = None
    complexity_distribution: Dict[str, float] = None

    def __post_init__(self):
        if self.question_types_distribution is None:
            self.question_types_distribution = {
                "factual": 0.3,
                "analytical": 0.25,
                "procedural": 0.2,
                "comparative": 0.15,
                "causal": 0.1,
            }
        if self.complexity_distribution is None:
            self.complexity_distribution = {
                "basic": 0.4,
                "intermediate": 0.4,
                "advanced": 0.2,
            }


class BiomedicalQuestionGenerator:
    """
    Advanced biomedical question generator using state-of-the-art NLP models
    and domain-specific heuristics for high-quality question-answer pair generation.
    """

    def __init__(self, config: QuestionGenerationConfig):
        self.config = config
        self.nlp = None
        self.question_generator = None
        self.answer_extractor = None
        self.medical_entities = set()
        self.domain_keywords = defaultdict(list)

        self._initialize_models()
        self._load_medical_knowledge()

    def _initialize_models(self):
        """Initialize NLP models for question generation and answer extraction."""
        logger.info("Initializing biomedical NLP models...")

        # Load spaCy model with biomedical entities
        try:
            self.nlp = spacy.load("en_core_sci_sm")  # ScispaCy for biomedical text
            logger.info("Loaded ScispaCy biomedical model")
        except OSError:
            logger.warning("ScispaCy not available, falling back to standard model")
            self.nlp = spacy.load("en_core_web_sm")

        # Load question generation model
        self.question_generator = pipeline(
            "text2text-generation",
            model="microsoft/DialoGPT-large",  # Can be replaced with biomedical-specific model
            tokenizer="microsoft/DialoGPT-large",
        )

        # Load answer extraction model for validation
        self.answer_extractor = pipeline(
            "question-answering",
            model="dmis-lab/biobert-base-cased-v1.1-squad",  # BioBERT for biomedical QA
        )

        logger.info("NLP models initialized successfully")

    def _load_medical_knowledge(self):
        """Load medical knowledge bases and domain-specific keywords."""
        # Medical entity lists (expandable with external knowledge bases)
        self.medical_entities = {
            # Diseases
            "diabetes",
            "hypertension",
            "cancer",
            "alzheimer",
            "parkinson",
            "stroke",
            "heart disease",
            "pneumonia",
            "tuberculosis",
            "malaria",
            "covid-19",
            "influenza",
            "asthma",
            "copd",
            "arthritis",
            # Treatments/Procedures
            "chemotherapy",
            "radiotherapy",
            "surgery",
            "immunotherapy",
            "dialysis",
            "transplantation",
            "vaccination",
            "antibiotic",
            "insulin",
            "stent",
            "bypass",
            "angioplasty",
            # Body systems/anatomy
            "cardiovascular",
            "respiratory",
            "nervous",
            "digestive",
            "immune",
            "endocrine",
            "musculoskeletal",
            "reproductive",
            "heart",
            "lung",
            "brain",
            "liver",
            "kidney",
            "pancreas",
            # Biomarkers/tests
            "biomarker",
            "blood test",
            "mri",
            "ct scan",
            "ecg",
            "eeg",
            "biopsy",
            "genetic test",
            "protein",
            "enzyme",
            "hormone",
        }

        # Domain-specific question templates
        self.domain_keywords = {
            "cardiology": [
                "heart",
                "cardiac",
                "coronary",
                "artery",
                "blood pressure",
                "cholesterol",
            ],
            "oncology": [
                "cancer",
                "tumor",
                "metastasis",
                "chemotherapy",
                "radiation",
                "biopsy",
            ],
            "neurology": [
                "brain",
                "neural",
                "cognitive",
                "memory",
                "seizure",
                "dementia",
            ],
            "endocrinology": [
                "diabetes",
                "insulin",
                "hormone",
                "thyroid",
                "metabolism",
            ],
            "infectious_disease": [
                "bacteria",
                "virus",
                "infection",
                "antibiotic",
                "vaccine",
            ],
            "pulmonology": ["lung", "respiratory", "asthma", "copd", "pneumonia"],
            "gastroenterology": ["digestive", "stomach", "liver", "intestine", "gut"],
        }

    def generate_questions_from_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[BiomedicalQuestion]:
        """
        Generate biomedical questions from PMC document collection.

        Args:
            documents: List of document dictionaries with 'content', 'doc_id', 'title' keys

        Returns:
            List of generated biomedical questions with ground truth answers
        """
        logger.info(f"Generating questions from {len(documents)} documents...")

        # Filter and prioritize biomedical documents
        biomedical_docs = self._filter_biomedical_documents(documents)
        logger.info(f"Filtered to {len(biomedical_docs)} biomedical documents")

        # Extract high-quality passages for question generation
        passages = self._extract_informative_passages(biomedical_docs)
        logger.info(f"Extracted {len(passages)} informative passages")

        # Generate diverse question types
        all_questions = []
        target_per_type = self.config.total_questions // len(
            self.config.question_types_distribution
        )

        for (
            question_type,
            distribution,
        ) in self.config.question_types_distribution.items():
            target_count = int(self.config.total_questions * distribution)
            questions = self._generate_questions_by_type(
                passages, question_type, target_count
            )
            all_questions.extend(questions)
            logger.info(f"Generated {len(questions)} {question_type} questions")

        # Quality filtering and validation
        validated_questions = self._validate_and_filter_questions(all_questions)
        logger.info(f"Validated {len(validated_questions)} high-quality questions")

        # Balance complexity and domains
        balanced_questions = self._balance_question_distribution(validated_questions)
        logger.info(f"Final balanced dataset: {len(balanced_questions)} questions")

        return balanced_questions

    def _filter_biomedical_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter documents for biomedical content using domain-specific indicators."""
        biomedical_docs = []

        for doc in documents:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()

            # Score biomedical relevance
            biomedical_score = 0

            # Check for medical entities
            medical_entity_count = sum(
                1 for entity in self.medical_entities if entity in content
            )
            biomedical_score += medical_entity_count * 2

            # Check title for biomedical keywords
            title_medical_count = sum(
                1 for entity in self.medical_entities if entity in title
            )
            biomedical_score += title_medical_count * 5

            # Check for journal/source indicators
            biomedical_indicators = [
                "pubmed",
                "pmc",
                "medline",
                "clinical",
                "medical",
                "journal",
            ]
            source_score = sum(
                1
                for indicator in biomedical_indicators
                if indicator in doc.get("source", "").lower()
            )
            biomedical_score += source_score * 3

            # Threshold for inclusion
            if biomedical_score >= 5:  # Adjustable threshold
                doc["biomedical_score"] = biomedical_score
                biomedical_docs.append(doc)

        # Sort by biomedical relevance score
        return sorted(
            biomedical_docs, key=lambda x: x["biomedical_score"], reverse=True
        )

    def _extract_informative_passages(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract high-information content passages for question generation."""
        passages = []

        for doc in documents:
            content = doc["content"]
            doc_id = doc["doc_id"]

            # Split into sentences
            doc_nlp = self.nlp(content)
            sentences = [
                sent.text.strip()
                for sent in doc_nlp.sents
                if len(sent.text.strip()) > 20
            ]

            # Create overlapping passages
            passage_size = 5  # sentences per passage
            overlap = 2  # sentence overlap

            for i in range(
                0, len(sentences) - passage_size + 1, passage_size - overlap
            ):
                passage_sentences = sentences[i : i + passage_size]
                passage_text = " ".join(passage_sentences)

                # Score passage informativeness
                info_score = self._score_passage_informativeness(passage_text)

                if info_score > 0.6:  # Quality threshold
                    passages.append(
                        {
                            "text": passage_text,
                            "doc_id": doc_id,
                            "doc_title": doc.get("title", ""),
                            "info_score": info_score,
                            "medical_domain": self._identify_medical_domain(
                                passage_text
                            ),
                        }
                    )

        # Sort by informativeness score
        return sorted(passages, key=lambda x: x["info_score"], reverse=True)

    def _score_passage_informativeness(self, passage: str) -> float:
        """Score passage for information density and biomedical relevance."""
        score = 0.0
        passage_lower = passage.lower()

        # Medical entity density
        medical_entity_count = sum(
            1 for entity in self.medical_entities if entity in passage_lower
        )
        score += min(medical_entity_count / 10, 0.4)  # Cap at 0.4

        # Numerical data presence (studies, measurements, etc.)
        numerical_pattern = r"\d+\.?\d*\s*(%|mg|ml|years?|patients?|subjects?)"
        numerical_matches = len(re.findall(numerical_pattern, passage_lower))
        score += min(numerical_matches / 5, 0.2)  # Cap at 0.2

        # Scientific language indicators
        scientific_terms = [
            "study",
            "research",
            "analysis",
            "treatment",
            "diagnosis",
            "therapy",
            "clinical",
            "patient",
            "mechanism",
            "pathway",
        ]
        scientific_count = sum(1 for term in scientific_terms if term in passage_lower)
        score += min(scientific_count / 8, 0.3)  # Cap at 0.3

        # Length normalization (prefer substantial content)
        word_count = len(passage.split())
        if 50 <= word_count <= 200:
            score += 0.1

        return min(score, 1.0)

    def _identify_medical_domain(self, passage: str) -> str:
        """Identify primary medical domain of a passage."""
        passage_lower = passage.lower()
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in passage_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general_medicine"

    def _generate_questions_by_type(
        self, passages: List[Dict[str, Any]], question_type: str, target_count: int
    ) -> List[BiomedicalQuestion]:
        """Generate questions of specific type from passages."""
        questions = []
        type_templates = self._get_question_templates(question_type)

        # Sample passages for this question type
        sampled_passages = random.sample(passages, min(len(passages), target_count * 2))

        for passage in sampled_passages:
            if len(questions) >= target_count:
                break

            # Generate questions using templates and NLP
            for template in type_templates:
                try:
                    question_text, answer_text = self._apply_question_template(
                        passage, template, question_type
                    )

                    if question_text and answer_text:
                        # Validate question quality
                        confidence = self._validate_question_answer_pair(
                            question_text, answer_text, passage["text"]
                        )

                        if confidence >= self.config.min_confidence_score:
                            question = BiomedicalQuestion(
                                question=question_text,
                                ground_truth_answer=answer_text,
                                question_type=question_type,
                                complexity_level=self._assess_complexity(
                                    question_text, answer_text
                                ),
                                medical_domain=passage["medical_domain"],
                                source_doc_ids=[passage["doc_id"]],
                                source_passages=[passage["text"]],
                                keywords=self._extract_keywords(passage["text"]),
                                confidence_score=confidence,
                                question_id=f"{question_type}_{len(questions)}_{passage['doc_id']}",
                            )
                            questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to generate question: {e}")
                    continue

        return questions

    def _get_question_templates(self, question_type: str) -> List[Dict[str, str]]:
        """Get question templates for specific question types."""
        templates = {
            "factual": [
                {"pattern": "medical_entity", "template": "What is {entity}?"},
                {"pattern": "definition", "template": "How is {concept} defined?"},
                {
                    "pattern": "function",
                    "template": "What is the function of {entity}?",
                },
                {
                    "pattern": "characteristic",
                    "template": "What are the characteristics of {entity}?",
                },
            ],
            "analytical": [
                {"pattern": "mechanism", "template": "How does {process} work?"},
                {
                    "pattern": "relationship",
                    "template": "What is the relationship between {entity1} and {entity2}?",
                },
                {
                    "pattern": "analysis",
                    "template": "What factors contribute to {condition}?",
                },
            ],
            "procedural": [
                {"pattern": "treatment", "template": "How is {condition} treated?"},
                {"pattern": "diagnosis", "template": "How is {condition} diagnosed?"},
                {
                    "pattern": "procedure",
                    "template": "What are the steps in {procedure}?",
                },
            ],
            "comparative": [
                {
                    "pattern": "comparison",
                    "template": "How does {entity1} compare to {entity2}?",
                },
                {
                    "pattern": "effectiveness",
                    "template": "Which is more effective: {option1} or {option2}?",
                },
            ],
            "causal": [
                {"pattern": "cause", "template": "What causes {condition}?"},
                {
                    "pattern": "effect",
                    "template": "What are the effects of {intervention}?",
                },
                {
                    "pattern": "risk",
                    "template": "What are the risk factors for {condition}?",
                },
            ],
        }
        return templates.get(question_type, [])

    def _apply_question_template(
        self, passage: Dict[str, Any], template: Dict[str, str], question_type: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Apply question template to generate question-answer pair."""
        passage_text = passage["text"]
        doc_nlp = self.nlp(passage_text)

        # Extract entities and concepts
        entities = []
        for ent in doc_nlp.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DISEASE", "CHEMICAL"]:
                entities.append(ent.text)

        # Medical entity extraction
        medical_entities_found = [
            entity
            for entity in self.medical_entities
            if entity.lower() in passage_text.lower()
        ]
        entities.extend(medical_entities_found)

        if not entities:
            return None, None

        # Select primary entity
        primary_entity = random.choice(entities)

        # Generate question based on template
        question_template = template["template"]

        if "{entity}" in question_template:
            question = question_template.replace("{entity}", primary_entity)
        elif "{concept}" in question_template:
            question = question_template.replace("{concept}", primary_entity)
        else:
            # More complex template handling
            question = question_template

        # Extract answer from passage using BioBERT
        try:
            qa_result = self.answer_extractor(question=question, context=passage_text)
            answer = qa_result["answer"]

            if len(answer) >= self.config.min_answer_length:
                return question, answer
        except Exception as e:
            logger.debug(f"Answer extraction failed: {e}")

        return None, None

    def _validate_question_answer_pair(
        self, question: str, answer: str, context: str
    ) -> float:
        """Validate question-answer pair quality and return confidence score."""
        score = 0.0

        # Length validation
        if (
            self.config.min_answer_length
            <= len(answer)
            <= self.config.max_answer_length
        ):
            score += 0.3

        # Answer relevance to context
        if answer.lower() in context.lower():
            score += 0.3

        # Question-answer coherence using BioBERT
        try:
            qa_result = self.answer_extractor(question=question, context=context)
            if qa_result["score"] > 0.5:  # BioBERT confidence
                score += 0.4
        except Exception:
            pass

        return min(score, 1.0)

    def _assess_complexity(self, question: str, answer: str) -> str:
        """Assess question complexity level."""
        # Simple heuristics for complexity assessment
        complexity_indicators = {
            "basic": ["what is", "define", "name", "identify"],
            "intermediate": ["how does", "why does", "explain", "describe"],
            "advanced": ["analyze", "evaluate", "compare", "synthesize", "integrate"],
        }

        question_lower = question.lower()
        answer_words = len(answer.split())

        # Check for complexity indicators
        for level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                # Adjust based on answer length
                if level == "basic" and answer_words > 100:
                    return "intermediate"
                elif level == "intermediate" and answer_words > 200:
                    return "advanced"
                return level

        # Default based on answer length
        if answer_words < 50:
            return "basic"
        elif answer_words < 150:
            return "intermediate"
        else:
            return "advanced"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key medical terms from text."""
        doc = self.nlp(text)
        keywords = []

        # Named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "DISEASE", "CHEMICAL"]:
                keywords.append(ent.text.lower())

        # Medical entities
        text_lower = text.lower()
        for entity in self.medical_entities:
            if entity in text_lower:
                keywords.append(entity)

        # Important nouns and adjectives
        for token in doc:
            if (
                token.pos_ in ["NOUN", "ADJ"]
                and len(token.text) > 3
                and not token.is_stop
                and token.is_alpha
            ):
                keywords.append(token.lemma_.lower())

        return list(set(keywords))

    def _validate_and_filter_questions(
        self, questions: List[BiomedicalQuestion]
    ) -> List[BiomedicalQuestion]:
        """Apply comprehensive quality filtering to generated questions."""
        validated = []

        for question in questions:
            # Confidence threshold
            if question.confidence_score < self.config.min_confidence_score:
                continue

            # Duplicate detection (simple text similarity)
            is_duplicate = False
            for existing in validated:
                if (
                    self._calculate_text_similarity(
                        question.question, existing.question
                    )
                    > 0.8
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                validated.append(question)

        return validated

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]

    def _balance_question_distribution(
        self, questions: List[BiomedicalQuestion]
    ) -> List[BiomedicalQuestion]:
        """Balance question distribution across complexity levels and domains."""
        # Group by complexity and domain
        complexity_groups = defaultdict(list)
        domain_groups = defaultdict(list)

        for question in questions:
            complexity_groups[question.complexity_level].append(question)
            domain_groups[question.medical_domain].append(question)

        # Balance complexity distribution
        balanced_questions = []
        for complexity, target_ratio in self.config.complexity_distribution.items():
            target_count = int(self.config.total_questions * target_ratio)
            available = complexity_groups.get(complexity, [])
            selected = random.sample(available, min(len(available), target_count))
            balanced_questions.extend(selected)

        # Ensure we don't exceed total
        if len(balanced_questions) > self.config.total_questions:
            balanced_questions = random.sample(
                balanced_questions, self.config.total_questions
            )

        return balanced_questions

    def save_questions_dataset(
        self, questions: List[BiomedicalQuestion], output_path: str
    ):
        """Save generated questions to structured dataset files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        questions_data = []
        for q in questions:
            questions_data.append(
                {
                    "question_id": q.question_id,
                    "question": q.question,
                    "ground_truth_answer": q.ground_truth_answer,
                    "question_type": q.question_type,
                    "complexity_level": q.complexity_level,
                    "medical_domain": q.medical_domain,
                    "source_doc_ids": q.source_doc_ids,
                    "source_passages": q.source_passages,
                    "keywords": q.keywords,
                    "confidence_score": q.confidence_score,
                }
            )

        # Save as JSON
        with open(output_dir / "biomedical_questions_dataset.json", "w") as f:
            json.dump(questions_data, f, indent=2)

        # Save as CSV for analysis
        df = pd.DataFrame(questions_data)
        df.to_csv(output_dir / "biomedical_questions_dataset.csv", index=False)

        # Generate summary statistics
        self._generate_dataset_summary(questions, output_dir)

        logger.info(f"Saved {len(questions)} questions to {output_path}")

    def _generate_dataset_summary(
        self, questions: List[BiomedicalQuestion], output_dir: Path
    ):
        """Generate comprehensive summary statistics for the question dataset."""
        summary = {
            "total_questions": len(questions),
            "question_types": Counter(q.question_type for q in questions),
            "complexity_levels": Counter(q.complexity_level for q in questions),
            "medical_domains": Counter(q.medical_domain for q in questions),
            "confidence_scores": {
                "mean": np.mean([q.confidence_score for q in questions]),
                "std": np.std([q.confidence_score for q in questions]),
                "min": min(q.confidence_score for q in questions),
                "max": max(q.confidence_score for q in questions),
            },
            "answer_lengths": {
                "mean": np.mean([len(q.ground_truth_answer) for q in questions]),
                "std": np.std([len(q.ground_truth_answer) for q in questions]),
                "min": min(len(q.ground_truth_answer) for q in questions),
                "max": max(len(q.ground_truth_answer) for q in questions),
            },
        }

        with open(output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Generated dataset summary statistics")


def create_biomedical_question_generator(
    total_questions: int = 1000,
) -> BiomedicalQuestionGenerator:
    """Factory function to create a configured biomedical question generator."""
    config = QuestionGenerationConfig(total_questions=total_questions)
    return BiomedicalQuestionGenerator(config)


if __name__ == "__main__":
    # Example usage
    generator = create_biomedical_question_generator(1000)

    # Mock documents for testing
    sample_documents = [
        {
            "doc_id": "pmc123",
            "title": "Cardiovascular Disease and Diabetes Management",
            "content": "Diabetes mellitus significantly increases cardiovascular disease risk. Patients with diabetes have a 2-4 fold higher risk of developing coronary artery disease compared to non-diabetic individuals. The pathophysiology involves advanced glycation end products, oxidative stress, and endothelial dysfunction. Treatment approaches include glycemic control with metformin, statins for cholesterol management, and ACE inhibitors for blood pressure control.",
            "source": "PMC_database",
        }
    ]

    questions = generator.generate_questions_from_documents(sample_documents)
    generator.save_questions_dataset(questions, "output/biomedical_questions")
