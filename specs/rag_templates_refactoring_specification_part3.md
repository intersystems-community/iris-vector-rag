# RAG Templates Refactoring Specification - Part 3

## 6.2 Validation and Testing Strategy (Continued)

```python
# PSEUDOCODE: Migration Validation (Continued)

    def _validate_interface_preservation(self) -> ValidationCheck:
        """
        Validate that public interfaces are preserved
        
        TDD Anchor: test_validate_interface_preservation()
        """
        try:
            sys.path.insert(0, self.target_path)
            
            from rag_templates import create_pipeline
            
            # Test basic interface
            config = {
                'database': {
                    'host': 'localhost',
                    'port': 1972,
                    'namespace': 'USER',
                    'schema': 'RAG'
                },
                'embedding': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'dimension': 384
                },
                'storage': {
                    'table_name': 'TestDocuments'
                }
            }
            
            # This should not raise an exception
            pipeline = create_pipeline('basic', config)
            
            # Check required methods exist
            required_methods = [
                'retrieve_documents',
                'store_document', 
                'get_document_count',
                'health_check'
            ]
            
            for method in required_methods:
                if not hasattr(pipeline, method):
                    return ValidationCheck(
                        name="interface_preservation",
                        passed=False,
                        message=f"Missing required method: {method}"
                    )
            
            return ValidationCheck(
                name="interface_preservation",
                passed=True,
                message="All required interfaces preserved"
            )
            
        except Exception as e:
            return ValidationCheck(
                name="interface_preservation",
                passed=False,
                message=f"Interface validation failed: {e}"
            )
        finally:
            if self.target_path in sys.path:
                sys.path.remove(self.target_path)

@dataclass
class ValidationCheck:
    name: str
    passed: bool
    message: str

@dataclass
class ValidationReport:
    overall_success: bool
    checks: List[ValidationCheck]
```

## 7. TDD Test Anchors and Quality Gates

### 7.1 Core Component Tests

**TDD Anchor**: [`test_core_component_functionality()`](rag_templates_refactoring_specification_part3.md:7.1)

```python
# PSEUDOCODE: Core Component Tests

class TestRAGPipelineInterface:
    """
    Test suite for RAG pipeline interface standardization
    
    TDD Anchor: test_rag_pipeline_interface_compliance()
    """
    
    def test_pipeline_creation(self):
        """
        Test pipeline can be created with valid configuration
        
        TDD Anchor: test_pipeline_creation()
        """
        config = self._get_valid_config()
        
        pipeline = create_pipeline('basic', config)
        
        assert isinstance(pipeline, RAGPipeline)
        assert pipeline.config == config
        assert pipeline.connection_manager is not None
    
    def test_pipeline_creation_invalid_type(self):
        """
        Test pipeline creation fails with invalid type
        
        TDD Anchor: test_pipeline_creation_invalid_type()
        """
        config = self._get_valid_config()
        
        with pytest.raises(ValueError, match="Unknown pipeline type"):
            create_pipeline('invalid_type', config)
    
    def test_pipeline_creation_invalid_config(self):
        """
        Test pipeline creation fails with invalid configuration
        
        TDD Anchor: test_pipeline_creation_invalid_config()
        """
        invalid_config = {'invalid': 'config'}
        
        with pytest.raises(ValueError, match="Missing required config"):
            create_pipeline('basic', invalid_config)
    
    def test_document_retrieval_interface(self):
        """
        Test document retrieval interface compliance
        
        TDD Anchor: test_document_retrieval_interface()
        """
        pipeline = self._create_test_pipeline()
        
        # Mock the database response
        with patch.object(pipeline.connection_manager, 'get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('doc1', 'Title 1', 'Content 1', 0.95, '{}', 'file1.txt'),
                ('doc2', 'Title 2', 'Content 2', 0.85, '{}', 'file2.txt')
            ]
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            documents = pipeline.retrieve_documents("test query", top_k=2, threshold=0.8)
            
            assert len(documents) == 2
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(doc.score >= 0.8 for doc in documents)
            assert documents[0].score >= documents[1].score  # Sorted by score
    
    def test_document_storage_interface(self):
        """
        Test document storage interface compliance
        
        TDD Anchor: test_document_storage_interface()
        """
        pipeline = self._create_test_pipeline()
        
        document = Document(
            id="test_doc_1",
            content="Test document content",
            metadata={"title": "Test Document", "source": "test"}
        )
        
        with patch.object(pipeline.connection_manager, 'get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            doc_id = pipeline.store_document(document)
            
            assert doc_id == "test_doc_1"
            mock_cursor.execute.assert_called_once()
            mock_conn.return_value.commit.assert_called_once()
    
    def test_health_check_interface(self):
        """
        Test health check interface compliance
        
        TDD Anchor: test_health_check_interface()
        """
        pipeline = self._create_test_pipeline()
        
        with patch.object(pipeline.connection_manager, 'get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (1,)
            mock_conn.return_value.cursor.return_value = mock_cursor
            
            is_healthy = pipeline.health_check()
            
            assert is_healthy is True
            mock_cursor.execute.assert_called_with("SELECT 1")
    
    def _get_valid_config(self) -> Dict[str, Any]:
        """Get valid test configuration"""
        return {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG'
            },
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384
            },
            'storage': {
                'table_name': 'TestDocuments'
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            }
        }
    
    def _create_test_pipeline(self) -> RAGPipeline:
        """Create pipeline for testing"""
        config = self._get_valid_config()
        
        # Mock embedding function
        def mock_embedding_func(texts):
            return [[0.1] * 384 for _ in texts]
        
        return create_pipeline('basic', config, embedding_func=mock_embedding_func)

class TestConnectionManager:
    """
    Test suite for connection manager functionality
    
    TDD Anchor: test_connection_manager_functionality()
    """
    
    def test_connection_manager_creation(self):
        """
        Test connection manager can be created
        
        TDD Anchor: test_connection_manager_creation()
        """
        manager = ConnectionManager('odbc')
        
        assert manager.connection_type == 'odbc'
        assert manager._connection is None
    
    def test_connection_manager_invalid_type(self):
        """
        Test connection manager rejects invalid types
        
        TDD Anchor: test_connection_manager_invalid_type()
        """
        with pytest.raises(ValueError, match="Invalid connection type"):
            ConnectionManager('invalid')
    
    def test_connection_manager_fallback(self):
        """
        Test connection manager falls back gracefully
        
        TDD Anchor: test_connection_manager_fallback()
        """
        # Mock JDBC failure, should fallback to ODBC
        with patch('rag_templates.core.connection_manager.get_iris_jdbc_connection', 
                  side_effect=ImportError("JDBC not available")):
            
            manager = ConnectionManager('jdbc')
            
            # Should fallback to ODBC
            with patch('rag_templates.core.connection_manager.get_iris_connection') as mock_odbc:
                mock_odbc.return_value = MagicMock()
                
                connection = manager.connect()
                
                assert manager.connection_type == 'odbc'
                assert connection is not None
    
    def test_connection_manager_execute(self):
        """
        Test connection manager execute method
        
        TDD Anchor: test_connection_manager_execute()
        """
        manager = ConnectionManager('odbc')
        
        with patch.object(manager, 'connect') as mock_connect:
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('result',)]
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection
            manager._connection = mock_connection
            
            result = manager.execute("SELECT 1", [])
            
            assert result == [('result',)]
            mock_cursor.execute.assert_called_with("SELECT 1", [])

class TestConfigurationSystem:
    """
    Test suite for configuration system
    
    TDD Anchor: test_configuration_system_functionality()
    """
    
    def test_config_loading_from_dict(self):
        """
        Test configuration loading from dictionary
        
        TDD Anchor: test_config_loading_from_dict()
        """
        config_dict = {
            'database': {
                'host': 'test_host',
                'port': 1972
            }
        }
        
        manager = ConfigurationManager()
        config = manager.load_config(config_dict)
        
        assert config['database']['host'] == 'test_host'
        assert config['database']['port'] == 1972
    
    def test_config_validation_success(self):
        """
        Test configuration validation with valid config
        
        TDD Anchor: test_config_validation_success()
        """
        valid_config = {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG'
            },
            'embedding': {
                'model_name': 'test-model',
                'dimension': 384
            }
        }
        
        validator = ConfigValidator()
        result = validator.validate(valid_config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_config_validation_failure(self):
        """
        Test configuration validation with invalid config
        
        TDD Anchor: test_config_validation_failure()
        """
        invalid_config = {
            'database': {
                'host': '',  # Invalid: empty string
                'port': -1,  # Invalid: negative port
                # Missing required fields
            }
        }
        
        validator = ConfigValidator()
        result = validator.validate(invalid_config)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any('namespace' in error for error in result.errors)
        assert any('schema' in error for error in result.errors)
    
    def test_environment_override(self):
        """
        Test environment variable overrides
        
        TDD Anchor: test_environment_override()
        """
        base_config = {
            'database': {
                'host': 'localhost',
                'port': 1972
            }
        }
        
        with patch.dict(os.environ, {
            'RAG_TEMPLATES_DATABASE_HOST': 'override_host',
            'RAG_TEMPLATES_DATABASE_PORT': '9999'
        }):
            manager = ConfigurationManager()
            config = manager.load_config(base_config)
            
            assert config['database']['host'] == 'override_host'
            assert config['database']['port'] == 9999
```

### 7.2 Integration Tests

**TDD Anchor**: [`test_integration_functionality()`](rag_templates_refactoring_specification_part3.md:7.2)

```python
# PSEUDOCODE: Integration Tests

class TestPersonalAssistantIntegration:
    """
    Test suite for personal assistant integration
    
    TDD Anchor: test_personal_assistant_integration_functionality()
    """
    
    def test_survival_config_conversion(self):
        """
        Test conversion from survival mode config to rag-templates config
        
        TDD Anchor: test_survival_config_conversion()
        """
        survival_config = {
            'database': {
                'db_host': 'survival_host',
                'db_port': 1972,
                'db_namespace': 'SURVIVAL',
                'schema': 'SURVIVAL_RAG'
            },
            'embedding_model': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384
            },
            'survival_mode': {
                'ingestion': {
                    'table_name': 'SurvivalModeDocuments',
                    'batch_size': 10
                },
                'knowledge_agent': {
                    'retrieval': {
                        'top_k': 3,
                        'similarity_threshold': 0.65
                    }
                }
            }
        }
        
        adapter = PersonalAssistantAdapter(survival_config)
        rag_config = adapter.rag_config
        
        assert rag_config['database']['host'] == 'survival_host'
        assert rag_config['database']['schema'] == 'SURVIVAL_RAG'
        assert rag_config['storage']['table_name'] == 'SurvivalModeDocuments'
        assert rag_config['retrieval']['top_k'] == 3
        assert rag_config['retrieval']['similarity_threshold'] == 0.65
    
    def test_adapter_pipeline_initialization(self):
        """
        Test adapter can initialize pipeline correctly
        
        TDD Anchor: test_adapter_pipeline_initialization()
        """
        survival_config = self._get_test_survival_config()
        
        adapter = PersonalAssistantAdapter(survival_config)
        
        with patch('rag_templates.create_pipeline') as mock_create:
            mock_pipeline = MagicMock()
            mock_create.return_value = mock_pipeline
            
            pipeline = adapter.initialize_pipeline()
            
            assert pipeline == mock_pipeline
            mock_create.assert_called_once_with('basic', adapter.rag_config)
    
    def test_adapter_document_operations(self):
        """
        Test adapter document operations work correctly
        
        TDD Anchor: test_adapter_document_operations()
        """
        survival_config = self._get_test_survival_config()
        adapter = PersonalAssistantAdapter(survival_config)
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve_documents.return_value = [
            Document(id="doc1", content="content1", score=0.9)
        ]
        mock_pipeline.store_document.return_value = "stored_doc_id"
        adapter.pipeline = mock_pipeline
        
        # Test retrieval
        documents = adapter.retrieve_documents("test query")
        assert len(documents) == 1
        assert documents[0].id == "doc1"
        
        # Test storage
        doc = Document(id="new_doc", content="new content")
        doc_id = adapter.store_document(doc)
        assert doc_id == "stored_doc_id"
    
    def test_survival_mode_rag_service(self):
        """
        Test survival mode RAG service functionality
        
        TDD Anchor: test_survival_mode_rag_service()
        """
        config_path = "test_survival_config.yaml"
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(self._get_test_survival_config()))):
            service = SurvivalModeRAGService(config_path)
            
            # Mock adapter
            mock_adapter = MagicMock()
            mock_adapter.retrieve_documents.return_value = [
                Document(id="doc1", content="content1", score=0.9)
            ]
            service.adapter = mock_adapter
            
            # Test retrieval with error handling
            documents = service.retrieve_documents("test query")
            assert len(documents) == 1
            
            # Test retrieval with exception (should return empty list)
            mock_adapter.retrieve_documents.side_effect = Exception("Database error")
            documents = service.retrieve_documents("test query")
            assert len(documents) == 0
    
    def _get_test_survival_config(self) -> Dict[str, Any]:
        """Get test survival mode configuration"""
        return {
            'database': {
                'db_host': 'localhost',
                'db_port': 1972,
                'db_namespace': 'USER',
                'schema': 'RAG'
            },
            'embedding_model': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384
            },
            'survival_mode': {
                'ingestion': {
                    'table_name': 'SurvivalModeDocuments',
                    'batch_size': 10
                },
                'knowledge_agent': {
                    'retrieval': {
                        'top_k': 5,
                        'similarity_threshold': 0.7
                    }
                },
                'logging': {
                    'log_level': 'INFO'
                }
            }
        }

class TestEndToEndWorkflow:
    """
    End-to-end workflow tests
    
    TDD Anchor: test_end_to_end_workflow()
    """
    
    def test_complete_rag_workflow(self):
        """
        Test complete RAG workflow from config to retrieval
        
        TDD Anchor: test_complete_rag_workflow()
        """
        # Setup configuration
        config = {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG'
            },
            'embedding': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384
            },
            'storage': {
                'table_name': 'TestDocuments'
            }
        }
        
        # Mock embedding function
        def mock_embedding_func(texts):
            return [[0.1] * 384 for _ in texts]
        
        # Create pipeline
        with patch('rag_templates.core.connection_manager.get_iris_connection') as mock_conn:
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_conn.return_value = mock_connection
            
            pipeline = create_pipeline('basic', config, embedding_func=mock_embedding_func)
            
            # Test document storage
            document = Document(
                id="test_doc",
                content="This is a test document",
                metadata={"title": "Test Document"}
            )
            
            doc_id = pipeline.store_document(document)
            assert doc_id == "test_doc"
            
            # Test document retrieval
            mock_cursor.fetchall.return_value = [
                ('test_doc', 'Test Document', 'This is a test document', 0.95, '{"title": "Test Document"}', None)
            ]
            
            documents = pipeline.retrieve_documents("test query")
            assert len(documents) == 1
            assert documents[0].id == "test_doc"
            assert documents[0].score == 0.95
    
    def test_migration_end_to_end(self):
        """
        Test complete migration process
        
        TDD Anchor: test_migration_end_to_end()
        """
        source_path = "/tmp/test_rag_templates"
        target_path = "/tmp/test_rag_templates_migrated"
        
        # Setup mock source structure
        with patch('os.path.isdir') as mock_isdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('os.makedirs') as mock_makedirs, \
             patch('builtins.open', mock_open()) as mock_file:
            
            # Mock source validation
            mock_isdir.return_value = True
            mock_isfile.return_value = True
            
            orchestrator = MigrationOrchestrator(source_path, target_path)
            
            # Mock all migration steps to succeed
            for step in orchestrator.migration_steps:
                step.function = MagicMock(return_value=StepResult(success=True))
            
            result = orchestrator.execute_migration()
            
            assert result.success is True
            assert len(result.completed_steps) == len(orchestrator.migration_steps)
```

### 7.3 Quality Gates

**TDD Anchor**: [`test_quality_gates()`](rag_templates_refactoring_specification_part3.md:7.3)

```python
# PSEUDOCODE: Quality Gates

class QualityGateValidator:
    """
    Validates quality gates for the refactored package
    
    TDD Anchor: test_quality_gate_validation()
    """
    
    def __init__(self, package_path: str):
        self.package_path = package_path
        self.quality_gates = self._define_quality_gates()
    
    def _define_quality_gates(self) -> List[QualityGate]:
        """
        Define quality gates for package validation
        
        TDD Anchor: test_define_quality_gates()
        """
        return [
            QualityGate(
                name="test_coverage",
                description="Minimum 80% test coverage",
                validator=self._validate_test_coverage,
                threshold=80.0
            ),
            QualityGate(
                name="code_quality",
                description="Code quality metrics (pylint score > 8.0)",
                validator=self._validate_code_quality,
                threshold=8.0
            ),
            QualityGate(
                name="type_coverage",
                description="Type annotation coverage > 90%",
                validator=self._validate_type_coverage,
                threshold=90.0
            ),
            QualityGate(
                name="documentation_coverage",
                description="Docstring coverage > 85%",
                validator=self._validate_documentation_coverage,
                threshold=85.0
            ),
            QualityGate(
                name="security_scan",
                description="No high-severity security issues",
                validator=self._validate_security,
                threshold=0
            ),
            QualityGate(
                name="performance_benchmarks",
                description="Performance within acceptable limits",
                validator=self._validate_performance,
                threshold=None
            )
        ]
    
    def validate_all_gates(self) -> QualityReport:
        """
        Validate all quality gates
        
        TDD Anchor: test_validate_all_gates()
        """
        results = []
        overall_passed = True
        
        for gate in self.quality_gates:
            try:
                result = gate.validator()
                
                if gate.threshold is not None:
                    passed = result.value >= gate.threshold
                else:
                    passed = result.passed
                
                results.append(QualityGateResult(
                    gate=gate,
                    passed=passed,
                    value=result.value,
                    message=result.message
                ))
                
                if not passed:
                    overall_passed = False
                    
            except Exception as e:
                results.append(QualityGateResult(
                    gate=gate,
                    passed=False,
                    value=None,
                    message=f"Validation failed: {e}"
                ))
                overall_passed = False
        
        return QualityReport(
            overall_passed=overall_passed,
            gate_results=results
        )
    
    def _validate_test_coverage(self) -> QualityMetric:
        """
        Validate test coverage meets minimum threshold
        
        TDD Anchor: test_validate_test_coverage()
        """
        try:
            import coverage
            
            cov = coverage.Coverage()
            cov.load()
            
            # Get coverage percentage
            total_coverage = cov.report(show_missing=False)
            
            return QualityMetric(
                value=total_coverage,
                passed=total_coverage >= 80.0,
                message=f"Test coverage: {total_coverage:.1f}%"
            )
            
        except ImportError:
            return QualityMetric(
                value=0.0,
                passed=False,
                message="Coverage package not available"
            )
    
    def _validate_code_quality(self) -> QualityMetric:
        """
        Validate code quality using pylint
        
        TDD Anchor: test_validate_code_quality()
        """
        try:
            import subprocess
            
            result = subprocess.run(
                ['pylint', self.package_path, '--output-format=json'],
                capture_output=True,
                text=True
            )
            
            # Parse pylint output to get score
            # This is simplified - real implementation would parse JSON
            score = 8.5  # Mock score for example
            
            return QualityMetric(
                value=score,
                passed=score >= 8.0,
                message=f"Pylint score: {score:.1f}/10"
            )
            
        except Exception as e:
            return QualityMetric(
                value=0.0,
                passed=False,
                message=f"Code quality check failed: {e}"
            )
    
    def _validate_type_coverage(self) -> QualityMetric:
        """
        Validate type annotation coverage
        
        TDD Anchor: test_validate_type_coverage()
        """
        try:
            import subprocess
            
            result = subprocess.run(
                ['mypy', '--strict', self.package_path],
                capture_output=True,
                text=True
            )
            
            # Calculate type coverage based on mypy output
            # This is simplified - real implementation would parse mypy output
            type_coverage = 92.0  # Mock coverage for example
            
            return QualityMetric(
                value=type_coverage,
                passed=type_coverage >= 90.0,
                message=f"Type coverage: {type_coverage:.1f}%"
            )
            
        except Exception as e:
            return QualityMetric(
                value=0.0,
                passed=False,
                message=f"Type coverage check failed: {e}"
            )
    
    def _validate_documentation_coverage(self) -> QualityMetric:
        """
        Validate docstring coverage
        
        TDD Anchor: test_validate_documentation_coverage()
        """
        try:
            import subprocess
            
            result = subprocess.run(
                ['pydocstyle', self.package_path],
                capture_output=True,
                text=True
            )
            
            # Calculate documentation coverage
            # This is simplified - real implementation would analyze docstrings
            doc_coverage = 88.0  # Mock coverage for example
            
            return QualityMetric(
                value=doc_coverage,
                passed=doc_coverage >= 85.0,
                message=f"Documentation coverage: {doc_coverage:.1f}%"
            )
            
        except Exception as e:
            return QualityMetric(
                value=0.0,
                passed=False,
                message=f"Documentation coverage check failed: {e}"
            )
    
    def _validate_security(self) -> QualityMetric:
        """
        Validate security using bandit
        
        TDD Anchor: test_validate_security()
        """
        try:
            import subprocess
            
            result = subprocess.run(
                ['bandit', '-r', self.package_path, '-f', 'json'],
                capture_output=True,
                text=True
            )
            
            # Parse bandit output for high-severity issues
            # This is simplified - real implementation would parse JSON
            high_severity_issues = 0  # Mock count for example
            
            return QualityMetric(
                value=high_severity_issues,
                passed=high_severity_issues == 0,
                message=f"High-severity security issues: {high_severity_issues}"
            )
            
        except Exception as e:
            return QualityMetric(
                value=999,  # High number to indicate failure
                passed=False,
                message=f"Security scan failed: {e}"
            )
    
    def _validate_performance(self) -> QualityMetric:
        """
        Validate performance benchmarks
        
        TDD Anchor: test_validate_performance()
        """
        try:
            # Run performance benchmarks
            benchmark_results = self._run_performance_benchmarks()
            
            # Check if all benchmarks pass
            all_passed = all(result.passed for result in benchmark_results)
            
            return QualityMetric(
                value=None,
                passed=all_passed,
                message=f"Performance benchmarks: {len([r for r in benchmark_results if r.passed])}/{len(benchmark_results)} passed"
            )
            
        except Exception as e:
            return QualityMetric(
                value=None,
                passed=False,
                message=f"Performance validation failed: {e}"
            )
    
    def _run_performance_benchmarks(self) -> List[BenchmarkResult]:
        """Run performance benchmarks"""
        # Mock benchmark results
        return [
            BenchmarkResult(
                name="document_retrieval_latency",
                value=0.15,  # seconds
                threshold=0.5,
                passed=True
            ),
            BenchmarkResult(
                name="document_storage_latency", 
                value=0.08,  # seconds
                threshold=0.2,
                passed=True
            ),
            BenchmarkResult(
                name="embedding_generation_latency",
                value=0.05,  # seconds
                threshold=0.1,
                passed=True
            )
        ]

@dataclass
class QualityGate:
    name: str
    description: str
    validator: Callable[[], 'QualityMetric']
    threshold: Optional[float]

@dataclass
class QualityMetric:
    value: Optional[float]
    passed: bool
    message: str

@dataclass
class QualityGateResult:
    gate: QualityGate
    passed: bool
    value: Optional[float]
    message: str

@dataclass
class QualityReport:
    overall_passed: bool
    gate_results: List[QualityGateResult]

@dataclass
class BenchmarkResult:
    name: str
    value: float
    threshold: float
    passed: bool
```

## 8. Implementation Roadmap

### 8.1 Phase-Based Implementation

**TDD Anchor**: [`test_implementation_