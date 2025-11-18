"""Test fixture data for entity types configuration tests."""

from langchain_core.documents import Document

# Sample document with known entities for testing
SHIRLEY_TEMPLE_DOC = Document(
    page_content=(
        "Shirley Temple was an American actress, singer, dancer, and diplomat. "
        "As an adult, she was named United States ambassador to Ghana and "
        "to Czechoslovakia and also served as Chief of Protocol of the United States."
    ),
    metadata={
        "source": "hotpotqa_q2_test",
        "question": "What government position was held by the woman who portrayed Corliss Archer?"
    }
)

# Expected entities for SHIRLEY_TEMPLE_DOC
EXPECTED_ENTITIES_PERSON_TITLE_LOCATION = {
    "PERSON": ["Shirley Temple"],
    "TITLE": ["Chief of Protocol", "ambassador"],
    "LOCATION": ["United States", "Ghana", "Czechoslovakia"]
}

# Simple test document
SIMPLE_TEST_DOC = Document(
    page_content="Shirley Temple served as Chief of Protocol.",
    metadata={"source": "simple_test"}
)

# Multiple entity type document
MULTI_ENTITY_DOC = Document(
    page_content=(
        "Microsoft CEO Satya Nadella announced the new product in Seattle. "
        "The company will expand operations to London and Paris."
    ),
    metadata={"source": "multi_entity_test"}
)

EXPECTED_ENTITIES_MULTI = {
    "PERSON": ["Satya Nadella"],
    "ORGANIZATION": ["Microsoft"],
    "TITLE": ["CEO"],
    "LOCATION": ["Seattle", "London", "Paris"],
    "PRODUCT": []  # May vary based on extraction
}

# Healthcare domain document (TrakCare-specific)
HEALTHCARE_DOC = Document(
    page_content=(
        "User JohnDoe reported error in TrakCare Lab Module version 2024.1. "
        "The issue occurs when accessing InterSystems IRIS database."
    ),
    metadata={"source": "trakcare_test"}
)

EXPECTED_ENTITIES_HEALTHCARE = {
    "USER": ["JohnDoe"],
    "MODULE": ["Lab Module"],
    "VERSION": ["2024.1"],
    "PRODUCT": ["TrakCare", "InterSystems IRIS"],
    "ORGANIZATION": ["InterSystems"]
}

# Empty document for edge case testing
EMPTY_DOC = Document(
    page_content="",
    metadata={"source": "empty_test"}
)

# Document list for batch testing
BATCH_TEST_DOCS = [
    SHIRLEY_TEMPLE_DOC,
    SIMPLE_TEST_DOC,
    MULTI_ENTITY_DOC
]
