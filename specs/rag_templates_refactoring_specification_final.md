# RAG Templates Refactoring Specification - Final

## 8. Implementation Roadmap

### 8.1 Phase-Based Implementation

**TDD Anchor**: [`test_implementation_roadmap()`](rag_templates_refactoring_specification_final.md:8.1)

```python
# PSEUDOCODE: Implementation Roadmap

class ImplementationRoadmap:
    """
    Manages phased implementation of rag-templates refactoring
    
    TDD Anchor: test_implementation_roadmap()
    """
    
    def __init__(self):
        self.phases = self._define_implementation_phases()
        self.current_phase = None
        self.completed_phases = []
    
    def _define_implementation_phases(self) -> List[ImplementationPhase]:
        """
        Define implementation phases with dependencies and deliverables
        
        TDD Anchor: test_define_implementation_phases()
        """
        return [
            ImplementationPhase(
                name="Phase 1: Foundation",
                duration_weeks=2,
                description="Core infrastructure and base classes",
                deliverables=[
                    "Package structure creation",
                    "Core base classes (RAGPipeline, ConnectionManager)",
                    "Configuration system",
                    "Basic test framework"
                ],
                dependencies=[],
                success_criteria=[
                    "Package can be imported",
                    "Basic configuration loading works",
                    "Connection manager can connect to IRIS",
                    "Test coverage > 70%"
                ]
            ),
            ImplementationPhase(
                name="Phase 2: Basic Pipeline",
                duration_weeks=2,
                description="Basic RAG pipeline implementation",
                deliverables=[
                    "BasicRAGPipeline implementation",
                    "Document storage and retrieval",
                    "Embedding integration",
                    "Pipeline factory"
                ],
                dependencies=["Phase 1: Foundation"],
                success_criteria=[
                    "Basic pipeline can store and retrieve documents",
                    "Embedding generation works",
                    "Vector search functionality",
                    "Test coverage > 80%"
                ]
            ),
            ImplementationPhase(
                name="Phase 3: Personal Assistant Integration",
                duration_weeks=1,
                description="Integration with personal assistant survival mode",
                deliverables=[
                    "PersonalAssistantAdapter",
                    "SurvivalModeRAGService",
                    "Configuration conversion",
                    "Integration tests"
                ],
                dependencies=["Phase 2: Basic Pipeline"],
                success_criteria=[
                    "Personal assistant can use rag-templates",
                    "Survival mode configuration works",
                    "All integration tests pass",
                    "Performance meets requirements"
                ]
            ),
            ImplementationPhase(
                name="Phase 4: Advanced Pipelines",
                duration_weeks=3,
                description="ColBERT and CRAG pipeline implementations",
                deliverables=[
                    "ColBERTRAGPipeline implementation",
                    "CRAGPipeline implementation",
                    "Advanced embedding support",
                    "Performance optimizations"
                ],
                dependencies=["Phase 3: Personal Assistant Integration"],
                success_criteria=[
                    "All pipeline types functional",
                    "Performance benchmarks met",
                    "Advanced features working",
                    "Test coverage > 85%"
                ]
            ),
            ImplementationPhase(
                name="Phase 5: Production Readiness",
                duration_weeks=2,
                description="Production deployment and monitoring",
                deliverables=[
                    "Packaging and distribution",
                    "Documentation completion",
                    "Performance monitoring",
                    "Security hardening"
                ],
                dependencies=["Phase 4: Advanced Pipelines"],
                success_criteria=[
                    "Package can be installed via pip",
                    "Complete documentation",
                    "All quality gates pass",
                    "Security scan clean"
                ]
            )
        ]
    
    def execute_phase(self, phase_name: str) -> PhaseResult:
        """
        Execute specific implementation phase
        
        TDD Anchor: test_execute_phase()
        """
        phase = self._get_phase_by_name(phase_name)
        
        if not phase:
            raise ValueError(f"Phase not found: {phase_name}")
        
        # Check dependencies
        missing_deps = self._check_dependencies(phase)
        if missing_deps:
            raise DependencyError(f"Missing dependencies: {missing_deps}")
        
        self.current_phase = phase
        
        try:
            # Execute phase tasks
            task_results = []
            for deliverable in phase.deliverables:
                task_result = self._execute_deliverable(deliverable, phase)
                task_results.append(task_result)
            
            # Validate success criteria
            criteria_results = self._validate_success_criteria(phase)
            
            # Mark phase as completed
            self.completed_phases.append(phase)
            self.current_phase = None
            
            return PhaseResult(
                phase=phase,
                success=True,
                task_results=task_results,
                criteria_results=criteria_results
            )
            
        except Exception as e:
            return PhaseResult(
                phase=phase,
                success=False,
                error=str(e)
            )
    
    def _check_dependencies(self, phase: ImplementationPhase) -> List[str]:
        """Check if phase dependencies are satisfied"""
        missing = []
        completed_names = [p.name for p in self.completed_phases]
        
        for dep in phase.dependencies:
            if dep not in completed_names:
                missing.append(dep)
        
        return missing
    
    def _execute_deliverable(self, deliverable: str, phase: ImplementationPhase) -> TaskResult:
        """Execute individual deliverable"""
        # This would contain specific implementation logic for each deliverable
        # For now, return mock result
        return TaskResult(
            deliverable=deliverable,
            success=True,
            duration_hours=8.0
        )
    
    def _validate_success_criteria(self, phase: ImplementationPhase) -> List[CriteriaResult]:
        """Validate phase success criteria"""
        results = []
        
        for criteria in phase.success_criteria:
            # This would contain specific validation logic
            # For now, return mock result
            results.append(CriteriaResult(
                criteria=criteria,
                passed=True,
                value=85.0  # Mock metric value
            ))
        
        return results

@dataclass
class ImplementationPhase:
    name: str
    duration_weeks: int
    description: str
    deliverables: List[str]
    dependencies: List[str]
    success_criteria: List[str]

@dataclass
class PhaseResult:
    phase: ImplementationPhase
    success: bool
    task_results: Optional[List['TaskResult']] = None
    criteria_results: Optional[List['CriteriaResult']] = None
    error: Optional[str] = None

@dataclass
class TaskResult:
    deliverable: str
    success: bool
    duration_hours: float
    error: Optional[str] = None

@dataclass
class CriteriaResult:
    criteria: str
    passed: bool
    value: Optional[float] = None
    message: Optional[str] = None

class DependencyError(Exception):
    """Phase dependency errors"""
    pass
```

### 8.2 Success Criteria and Validation

**TDD Anchor**: [`test_success_criteria_validation()`](rag_templates_refactoring_specification_final.md:8.2)

```python
# PSEUDOCODE: Success Criteria Validation

class SuccessCriteriaValidator:
    """
    Validates success criteria for each implementation phase
    
    TDD Anchor: test_success_criteria_validator()
    """
    
    def __init__(self, package_path: str):
        self.package_path = package_path
        self.validators = self._setup_validators()
    
    def _setup_validators(self) -> Dict[str, Callable]:
        """
        Setup validation functions for different criteria types
        
        TDD Anchor: test_setup_validators()
        """
        return {
            'package_importable': self._validate_package_import,
            'configuration_loading': self._validate_config_loading,
            'iris_connection': self._validate_iris_connection,
            'test_coverage': self._validate_test_coverage,
            'document_operations': self._validate_document_operations,
            'embedding_generation': self._validate_embedding_generation,
            'vector_search': self._validate_vector_search,
            'performance_requirements': self._validate_performance,
            'integration_tests': self._validate_integration_tests,
            'quality_gates': self._validate_quality_gates
        }
    
    def validate_criteria(self, criteria: str) -> CriteriaResult:
        """
        Validate specific success criteria
        
        TDD Anchor: test_validate_criteria()
        """
        # Parse criteria to extract type and threshold
        criteria_type, threshold = self._parse_criteria(criteria)
        
        validator = self.validators.get(criteria_type)
        if not validator:
            return CriteriaResult(
                criteria=criteria,
                passed=False,
                message=f"No validator for criteria type: {criteria_type}"
            )
        
        try:
            result = validator(threshold)
            return CriteriaResult(
                criteria=criteria,
                passed=result.passed,
                value=result.value,
                message=result.message
            )
        except Exception as e:
            return CriteriaResult(
                criteria=criteria,
                passed=False,
                message=f"Validation failed: {e}"
            )
    
    def _parse_criteria(self, criteria: str) -> Tuple[str, Optional[float]]:
        """Parse criteria string to extract type and threshold"""
        # Examples:
        # "Test coverage > 70%" -> ("test_coverage", 70.0)
        # "Package can be imported" -> ("package_importable", None)
        
        if "test coverage" in criteria.lower():
            if ">" in criteria:
                threshold = float(criteria.split(">")[1].strip().rstrip("%"))
                return ("test_coverage", threshold)
            return ("test_coverage", None)
        
        elif "package can be imported" in criteria.lower():
            return ("package_importable", None)
        
        elif "configuration loading" in criteria.lower():
            return ("configuration_loading", None)
        
        elif "iris connection" in criteria.lower():
            return ("iris_connection", None)
        
        elif "document operations" in criteria.lower():
            return ("document_operations", None)
        
        elif "performance" in criteria.lower():
            return ("performance_requirements", None)
        
        else:
            return ("unknown", None)
    
    def _validate_package_import(self, threshold: Optional[float]) -> ValidationResult:
        """Validate package can be imported"""
        try:
            sys.path.insert(0, self.package_path)
            
            import rag_templates
            from rag_templates import create_pipeline, ConnectionManager
            
            return ValidationResult(
                passed=True,
                value=None,
                message="Package imports successfully"
            )
        except ImportError as e:
            return ValidationResult(
                passed=False,
                value=None,
                message=f"Import failed: {e}"
            )
        finally:
            if self.package_path in sys.path:
                sys.path.remove(self.package_path)
    
    def _validate_config_loading(self, threshold: Optional[float]) -> ValidationResult:
        """Validate configuration loading works"""
        try:
            sys.path.insert(0, self.package_path)
            
            from rag_templates.config import load_config
            
            test_config = {
                'database': {'host': 'localhost', 'port': 1972},
                'embedding': {'model_name': 'test', 'dimension': 384}
            }
            
            config = load_config(config_dict=test_config)
            
            return ValidationResult(
                passed=config is not None,
                value=None,
                message="Configuration loading works"
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                value=None,
                message=f"Config loading failed: {e}"
            )
        finally:
            if self.package_path in sys.path:
                sys.path.remove(self.package_path)
    
    def _validate_iris_connection(self, threshold: Optional[float]) -> ValidationResult:
        """Validate IRIS connection works"""
        try:
            sys.path.insert(0, self.package_path)
            
            from rag_templates.core import ConnectionManager
            
            # Mock connection test
            manager = ConnectionManager('odbc')
            
            with patch.object(manager, 'connect') as mock_connect:
                mock_connection = MagicMock()
                mock_connect.return_value = mock_connection
                
                connection = manager.connect()
                
                return ValidationResult(
                    passed=connection is not None,
                    value=None,
                    message="IRIS connection successful"
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                value=None,
                message=f"IRIS connection failed: {e}"
            )
        finally:
            if self.package_path in sys.path:
                sys.path.remove(self.package_path)
    
    def _validate_test_coverage(self, threshold: Optional[float]) -> ValidationResult:
        """Validate test coverage meets threshold"""
        try:
            import subprocess
            
            result = subprocess.run(
                ['coverage', 'report', '--show-missing'],
                cwd=self.package_path,
                capture_output=True,
                text=True
            )
            
            # Parse coverage percentage from output
            # This is simplified - real implementation would parse coverage output
            coverage_percent = 82.0  # Mock value
            
            passed = threshold is None or coverage_percent >= threshold
            
            return ValidationResult(
                passed=passed,
                value=coverage_percent,
                message=f"Test coverage: {coverage_percent:.1f}%"
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                value=0.0,
                message=f"Coverage check failed: {e}"
            )

@dataclass
class ValidationResult:
    passed: bool
    value: Optional[float]
    message: str
```

## 9. Packaging and Distribution

### 9.1 Package Configuration

**TDD Anchor**: [`test_package_configuration()`](rag_templates_refactoring_specification_final.md:9.1)

```python
# PSEUDOCODE: Package Configuration

class PackageConfigurator:
    """
    Configures package for distribution
    
    TDD Anchor: test_package_configurator()
    """
    
    def __init__(self, package_path: str):
        self.package_path = package_path
        self.package_name = "rag-templates"
        self.version = "1.0.0"
    
    def create_setup_py(self) -> str:
        """
        Create setup.py for package distribution
        
        TDD Anchor: test_create_setup_py()
        """
        setup_content = f'''
"""Setup configuration for {self.package_name}"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{self.package_name}",
    version="{self.version}",
    author="Personal Assistant Team",
    author_email="team@personalassistant.dev",
    description="Modular RAG pipeline implementations for IRIS databases",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/personalassistant/rag-templates",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={{
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "pylint>=2.15.0",
        ],
        "jdbc": ["jaydebeapi>=1.2.3"],
        "odbc": ["pyodbc>=4.0.34"],
        "dbapi": ["iris>=0.1.0"],
        "sentence-transformers": ["sentence-transformers>=2.2.0"],
        "transformers": ["transformers>=4.20.0", "torch>=1.12.0"],
    }},
    entry_points={{
        "console_scripts": [
            "rag-templates=rag_templates.cli:main",
        ],
    }},
    include_package_data=True,
    package_data={{
        "rag_templates": ["config/*.yaml", "schemas/*.json"],
    }},
)
'''
        return setup_content
    
    def create_pyproject_toml(self) -> str:
        """
        Create pyproject.toml for modern Python packaging
        
        TDD Anchor: test_create_pyproject_toml()
        """
        pyproject_content = f'''
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{self.package_name}"
version = "{self.version}"
description = "Modular RAG pipeline implementations for IRIS databases"
readme = "README.md"
license = {{text = "MIT"}}
authors = [
    {{name = "Personal Assistant Team", email = "team@personalassistant.dev"}},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "pyyaml>=6.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
    "pylint>=2.15.0",
]
jdbc = ["jaydebeapi>=1.2.3"]
odbc = ["pyodbc>=4.0.34"]
dbapi = ["iris>=0.1.0"]
sentence-transformers = ["sentence-transformers>=2.2.0"]
transformers = ["transformers>=4.20.0", "torch>=1.12.0"]
all = [
    "jaydebeapi>=1.2.3",
    "pyodbc>=4.0.34",
    "sentence-transformers>=2.2.0",
    "transformers>=4.20.0",
    "torch>=1.12.0",
]

[project.scripts]
rag-templates = "rag_templates.cli:main"

[project.urls]
Homepage = "https://github.com/personalassistant/rag-templates"
Documentation = "https://rag-templates.readthedocs.io/"
Repository = "https://github.com/personalassistant/rag-templates.git"
"Bug Tracker" = "https://github.com/personalassistant/rag-templates/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["rag_templates*"]

[tool.setuptools.package-data]
rag_templates = ["config/*.yaml", "schemas/*.json"]

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pylint.messages_control]
disable = ["C0330", "C0326"]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers --cov=rag_templates --cov-report=term-missing"
testpaths = ["tests"]
'''
        return pyproject_content
    
    def create_manifest_in(self) -> str:
        """
        Create MANIFEST.in for package data inclusion
        
        TDD Anchor: test_create_manifest_in()
        """
        manifest_content = '''
include README.md
include LICENSE
include requirements.txt
include pyproject.toml
recursive-include rag_templates *.py
recursive-include rag_templates/config *.yaml
recursive-include rag_templates/schemas *.json
recursive-include tests *.py
recursive-include docs *.md *.rst
global-exclude *.pyc
global-exclude __pycache__
global-exclude .git*
global-exclude .pytest_cache
global-exclude *.egg-info
'''
        return manifest_content
    
    def create_requirements_txt(self) -> str:
        """
        Create requirements.txt with core dependencies
        
        TDD Anchor: test_create_requirements_txt()
        """
        requirements_content = '''
# Core dependencies
numpy>=1.21.0
pyyaml>=6.0
typing-extensions>=4.0.0

# Optional database drivers (install as needed)
# pyodbc>=4.0.34  # For ODBC connections
# jaydebeapi>=1.2.3  # For JDBC connections

# Optional embedding backends (install as needed)
# sentence-transformers>=2.2.0  # For sentence transformer embeddings
# transformers>=4.20.0  # For HuggingFace transformers
# torch>=1.12.0  # Required for transformers
'''
        return requirements_content

    def generate_all_files(self) -> PackageFiles:
        """
        Generate all package configuration files
        
        TDD Anchor: test_generate_all_files()
        """
        return PackageFiles(
            setup_py=self.create_setup_py(),
            pyproject_toml=self.create_pyproject_toml(),
            manifest_in=self.create_manifest_in(),
            requirements_txt=self.create_requirements_txt()
        )

@dataclass
class PackageFiles:
    setup_py: str
    pyproject_toml: str
    manifest_in: str
    requirements_txt: str
```

## 10. Summary and Next Steps

### 10.1 Specification Summary

This comprehensive specification defines the complete refactoring of the rag-templates repository into a clean, installable Python package with the following key achievements:

**Package Structure**: Clean hierarchical organization with [`rag_templates/core/`](rag_templates_refactoring_specification.md:1.2), [`rag_templates/pipelines/`](rag_templates_refactoring_specification.md:1.2), [`rag_templates/storage/`](rag_templates_refactoring_specification.md:1.2), and [`rag_templates/config/`](rag_templates_refactoring_specification.md:1.2) modules.

**Unified API**: Standardized [`RAGPipeline`](rag_templates_refactoring_specification.md:2.1) interface with [`create_pipeline()`](rag_templates_refactoring_specification.md:2.2) factory function for consistent usage across all pipeline types.

**Dependency Management**: Graceful handling of optional dependencies with [`DependencyManager`](rag_templates_refactoring_specification.md:3.1) and [`EmbeddingManager`](rag_templates_refactoring_specification.md:3.2) for flexible deployment scenarios.

**Configuration System**: Environment-based configuration with [`ConfigurationManager`](rag_templates_refactoring_specification_part2.md:4.1) and [`ConfigValidator`](rag_templates_refactoring_specification_part2.md:4.2) for robust configuration handling.

**Personal Assistant Integration**: Seamless integration via [`PersonalAssistantAdapter`](rag_templates_refactoring_specification_part2.md:5.1) and [`SurvivalModeRAGService`](rag_templates_refactoring_specification_part2.md:5.2) maintaining clean service boundaries.

### 10.2 Implementation Priority

**Phase 1 (Weeks 1-2)**: Foundation - Core infrastructure and base classes
**Phase 2 (Weeks 3-4)**: Basic Pipeline - BasicRAGPipeline implementation  
**Phase 3 (Week 5)**: Personal Assistant Integration - Adapter and service layer
**Phase 4 (Weeks 6-8)**: Advanced Pipelines - ColBERT and CRAG implementations
**Phase 5 (Weeks 9-10)**: Production Readiness - Packaging and quality gates

### 10.3 Success Metrics

- **Test Coverage**: >85% across all modules
- **Performance**: Document retrieval <500ms, storage <200ms
- **Quality Gates**: All quality gates pass including security, documentation, and type coverage
- **Integration**: Seamless drop-in replacement for existing [`initialize_iris_rag_pipeline()`](src/common/utils.py:206) function

The refactored package will provide a clean, maintainable, and extensible foundation for RAG operations while maintaining full compatibility with the personal assistant's survival mode architecture.