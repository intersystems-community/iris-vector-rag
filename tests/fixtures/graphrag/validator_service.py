"""Test data validation service for GraphRAG fixtures."""

from typing import Dict, List, Any


class ValidationError:
    """Represents a validation error."""
    
    def __init__(self, field: str, message: str, severity: str = 'error'):
        self.field = field
        self.message = message
        self.severity = severity
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.field}: {self.message}"
    
    def __repr__(self) -> str:
        return f"ValidationError(field='{self.field}', message='{self.message}', severity='{self.severity}')"


class ValidatorService:
    """Service for validating test fixtures and test data."""
    
    # Validation thresholds
    MIN_CONTENT_LENGTH = 100
    MIN_ENTITY_COUNT = 2
    
    def validate_document(self, document: Dict[str, Any]) -> List[ValidationError]:
        """Validate a document fixture.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ['doc_id', 'title', 'content']
        for field in required_fields:
            if field not in document:
                errors.append(ValidationError(field, f"Required field '{field}' is missing"))
        
        # Validate content length
        content = document.get('content', '')
        if len(content) < self.MIN_CONTENT_LENGTH:
            errors.append(
                ValidationError(
                    'content', 
                    f"Content too short ({len(content)} chars, minimum {self.MIN_CONTENT_LENGTH} chars)"
                )
            )
        
        # Validate entity count
        expected_entities = document.get('expected_entities', [])
        if len(expected_entities) < self.MIN_ENTITY_COUNT:
            errors.append(
                ValidationError(
                    'expected_entities',
                    f"Too few entities ({len(expected_entities)}, minimum {self.MIN_ENTITY_COUNT} entities)"
                )
            )
        
        # Validate entities are in content
        entity_errors = self.validate_entities_in_content(document)
        errors.extend(entity_errors)
        
        # Validate relationships reference existing entities
        relationship_errors = self.validate_relationships(document)
        errors.extend(relationship_errors)
        
        return errors
    
    def validate_entities_in_content(self, document: Dict[str, Any]) -> List[ValidationError]:
        """Validate that expected entities appear in document content.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        content = document.get('content', '').lower()
        expected_entities = document.get('expected_entities', [])
        
        for entity in expected_entities:
            entity_name = entity.get('name', '')
            if entity_name.lower() not in content:
                errors.append(
                    ValidationError(
                        'expected_entities',
                        f"Entity '{entity_name}' not found in content",
                        severity='warning'
                    )
                )
        
        return errors
    
    def validate_relationships(self, document: Dict[str, Any]) -> List[ValidationError]:
        """Validate that relationships reference existing entities.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        expected_entities = document.get('expected_entities', [])
        expected_relationships = document.get('expected_relationships', [])
        
        entity_ids = {e.get('entity_id') for e in expected_entities}
        
        for rel in expected_relationships:
            source = rel.get('source')
            target = rel.get('target')
            
            if source not in entity_ids:
                errors.append(
                    ValidationError(
                        'expected_relationships',
                        f"Relationship source '{source}' not found in entities"
                    )
                )
            
            if target not in entity_ids:
                errors.append(
                    ValidationError(
                        'expected_relationships',
                        f"Relationship target '{target}' not found in entities"
                    )
                )
        
        return errors
    
    def validate_test_run(self, test_run: Dict[str, Any]) -> List[ValidationError]:
        """Validate test run statistics.
        
        Args:
            test_run: Test run dictionary to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check total matches sum
        total_tests = test_run.get('total_tests', 0)
        passed = test_run.get('passed_tests', 0)
        failed = test_run.get('failed_tests', 0)
        skipped = test_run.get('skipped_tests', 0)
        
        calculated_total = passed + failed + skipped
        
        if calculated_total != total_tests:
            errors.append(
                ValidationError(
                    'test_run',
                    f"Test totals don't match: total={total_tests}, sum={calculated_total} (passed={passed}, failed={failed}, skipped={skipped})"
                )
            )
        
        # Check coverage
        coverage = test_run.get('coverage_percentage')
        if coverage is not None and coverage < 90.0:
            errors.append(
                ValidationError(
                    'coverage_percentage',
                    f"Coverage below requirement: {coverage}% (minimum 90%)",
                    severity='warning'
                )
            )
        
        return errors
    
    def validate_fixture_batch(self, fixtures: List[Dict[str, Any]]) -> Dict[str, List[ValidationError]]:
        """Validate a batch of fixtures.
        
        Args:
            fixtures: List of fixture documents to validate
            
        Returns:
            Dictionary mapping doc_id to list of errors
        """
        results = {}
        
        for fixture in fixtures:
            doc_id = fixture.get('doc_id', 'unknown')
            errors = self.validate_document(fixture)
            if errors:
                results[doc_id] = errors
        
        return results
    
    def is_valid(self, document: Dict[str, Any], allow_warnings: bool = True) -> bool:
        """Check if a document is valid.
        
        Args:
            document: Document to validate
            allow_warnings: Whether to consider warnings as valid
            
        Returns:
            True if document is valid, False otherwise
        """
        errors = self.validate_document(document)
        
        if allow_warnings:
            # Only fail on actual errors, not warnings
            return not any(e.severity == 'error' for e in errors)
        else:
            # Fail on any validation issue
            return len(errors) == 0
