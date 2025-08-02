#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master Test Orchestration Script for RAG Templates Project.

This script provides a centralized way to define, manage, and execute
various test targets within the project, including Pytest runs, custom
Python test scripts, and Makefile-like targets. It supports dependency
management, parallel execution for safe targets, and comprehensive
reporting in JSON and Markdown formats.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
import concurrent.futures
from enum import Enum

# --- Constants ---
CONDA_ENV_NAME = "iris_vector"
CONDA_RUN_PREFIX_CMD = f"conda run -n {CONDA_ENV_NAME} --no-capture-output"
PYTHON_CMD = "python" # Assumes python on PATH is the correct one within conda env if activated, or use full path if necessary
DEFAULT_TIMEOUT = 3600  # 1 hour
DEFAULT_REPORTS_DIR = Path("outputs/test_orchestrator_reports")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Enums ---
class TestStatus(Enum):
    """Status of a test execution."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR" # For errors in the test runner itself

# --- Data Classes ---
@dataclass
class TestTarget:
    """Represents a single test target to be executed."""
    id: str
    command: List[str]
    description: str
    category: str
    cwd: Path = PROJECT_ROOT
    env_vars: Optional[Dict[str, str]] = None
    dependencies: List[str] = field(default_factory=list)
    timeout: int = DEFAULT_TIMEOUT  # seconds
    parallel_safe: bool = False
    runnable: bool = True # If the target can be run (e.g., script exists)
    allow_failure: bool = False # If failure of this target should not stop the whole run
    setup_target: bool = False # If this is a setup/teardown target

@dataclass
class TestResult:
    """Stores the result of a single test target execution."""
    target_id: str
    status: TestStatus
    start_time: float
    end_time: float
    duration: float
    stdout: str
    stderr: str
    return_code: Optional[int] = None
    error_message: Optional[str] = None # For runner errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "status": self.status.value,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "duration": self.duration,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_message": self.error_message,
        }

@dataclass
class ComprehensiveTestResults:
    """Stores the results of the entire test suite execution."""
    run_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    environment_info: Dict[str, Any]
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration": self.total_duration,
            "environment_info": self.environment_info,
            "summary": self.summary,
            "results": [res.to_dict() for res in self.results],
        }

# --- Main Orchestrator Class ---
class MasterTestOrchestrator:
    """Orchestrates the execution of defined test targets."""

    def __init__(self, reports_dir: Path, parallel_workers: int = 4, default_timeout: int = DEFAULT_TIMEOUT, conda_env_name: str = CONDA_ENV_NAME):
        self.reports_dir = reports_dir
        self.parallel_workers = parallel_workers
        self.default_timeout = default_timeout
        self.conda_env_name = conda_env_name
        self.conda_run_prefix = f"conda run -n {self.conda_env_name} --no-capture-output"
        
        self.all_targets: Dict[str, TestTarget] = {}
        self.results: ComprehensiveTestResults = ComprehensiveTestResults(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            end_time=datetime.now(), # Placeholder
            total_duration=0.0, # Placeholder
            environment_info=self.collect_environment_info()
        )
        self.logger = self.setup_logging()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> logging.Logger:
        """Sets up logging for the orchestrator."""
        logger = logging.getLogger("MasterTestOrchestrator")
        logger.setLevel(logging.INFO)
        
        log_file_path = self.reports_dir / f"{self.results.run_id}.log"
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
        
        # File Handler
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        
        logger.info(f"Logging initialized. Log file: {log_file_path}")
        return logger

    def define_test_targets(self) -> None:
        """
        Defines all test targets based on project structure and analysis.
        This method populates `self.all_targets`.
        Information derived from testing_system_analysis.md.
        """
        targets = []
        
        # Pipeline names for parameterized targets
        pipeline_names = ["basic", "hyde", "crag", "colbert", "noderag", "graphrag", "hybrid_ifind"]

        # --- 1.1. Core Pytest Execution ---
        targets.extend([
            TestTarget(id="test-unit", command=[self.conda_run_prefix, "pytest", "tests/test_core/", "tests/test_pipelines/", "-v"], description="Run unit tests for core and pipelines.", category="core_pytest", parallel_safe=True),
            TestTarget(id="test-integration", command=[self.conda_run_prefix, "pytest", "tests/test_integration/", "-v"], description="Run integration tests.", category="core_pytest", parallel_safe=True),
            TestTarget(id="test-e2e-pytest", command=[self.conda_run_prefix, "pytest", "tests/test_e2e_*", "-v"], description="Run end-to-end tests (pytest based).", category="core_pytest", parallel_safe=True, dependencies=[]), # Assuming e2e tests might have setup dependencies handled elsewhere or are self-contained
            TestTarget(id="test", command=[], description="Aggregate for core unit and integration tests.", category="core_pytest", dependencies=["test-unit", "test-integration"]), # Command handled by dependency resolution
        ])

        # --- 1.2. Comprehensive & E2E Tests (Specific Scripts) ---
        targets.extend([
            TestTarget(id="test-1000", command=[self.conda_run_prefix, PYTHON_CMD, "test_comprehensive_e2e_iris_rag_1000_docs.py"], cwd=PROJECT_ROOT / "tests", description="Run comprehensive E2E test with 1000 PMC documents.", category="comprehensive_e2e", timeout=DEFAULT_TIMEOUT * 3), # Longer timeout
            TestTarget(id="benchmark", command=[self.conda_run_prefix, "pytest", "test_comprehensive_e2e_iris_rag_1000_docs.py::test_comprehensive_e2e_all_rag_techniques_1000_docs", "-v"], cwd=PROJECT_ROOT / "tests", description="Run performance benchmarks using 1000-doc E2E suite.", category="comprehensive_e2e", timeout=DEFAULT_TIMEOUT * 3, dependencies=["test-1000"]), # Depends on data from test-1000 or similar setup
        ])

        # --- 1.3. RAGAS Evaluations (Original/Comprehensive Scripts) ---
        # Corrected script path: scripts/utilities/run_complete_7_technique_ragas_evaluation.py
        ragas_eval_script = str(PROJECT_ROOT / "scripts/utilities/run_complete_7_technique_ragas_evaluation.py")
        
        targets.extend([
            TestTarget(
                id="test-ragas-1000-enhanced", 
                command=[self.conda_run_prefix, PYTHON_CMD, ragas_eval_script, "--verbose", "--pipelines"] + pipeline_names + ["--iterations", "3"],
                description="Run RAGAs evaluation on all 7 pipelines with 1000 docs (3 iterations).", 
                category="ragas_evaluation", 
                timeout=DEFAULT_TIMEOUT * 4 # RAGAS can be very long
            ),
            TestTarget(
                id="eval-all-ragas-1000", 
                command=[self.conda_run_prefix, PYTHON_CMD, ragas_eval_script, "--verbose", "--pipelines"] + pipeline_names + ["--iterations", "5"], # Output redirection handled by script or wrapper if needed
                description="Comprehensive RAGAs evaluation with full metrics (5 iterations).", 
                category="ragas_evaluation", 
                timeout=DEFAULT_TIMEOUT * 6
            ),
        ])
        for p_name in pipeline_names:
            targets.append(TestTarget(
                id=f"debug-ragas-{p_name}",
                command=[self.conda_run_prefix, PYTHON_CMD, ragas_eval_script, "--verbose", "--pipelines", p_name, "--iterations", "1", "--no-ragas"],
                description=f"Debug RAG pipeline '{p_name}' without RAGAs metric calculation.",
                category="ragas_evaluation"
            ))

        # --- 1.4. Lightweight RAGAS Testing (Missing Script `run_ragas.py`) ---
        # These targets point to a missing script. Mark as not runnable.
        missing_ragas_script = "eval/run_ragas.py" # Placeholder for the missing script path
        lightweight_ragas_targets_defs = [
            ("ragas-debug", [missing_ragas_script, "--pipelines", "basic", "--metrics-level", "core", "--max-queries", "3", "--verbose"], "Quick debug run of RAGAs."),
            ("ragas-test", [missing_ragas_script, "--pipelines", "basic", "hyde", "--metrics-level", "extended", "--verbose"], "Standard RAGAs test run."),
            ("ragas-full", [missing_ragas_script, "--pipelines"] + pipeline_names + ["--metrics-level", "full", "--verbose"], "Full RAGAs evaluation with all pipelines."),
            ("ragas-cache-check", [missing_ragas_script, "--cache-check"], "Check RAGAs cache status."),
            ("ragas-clean", [missing_ragas_script, "--clear-cache", "--pipelines", "basic", "--metrics-level", "core", "--max-queries", "3", "--verbose"], "Clear RAGAs cache and run debug."),
            ("ragas-no-cache", [missing_ragas_script, "--no-cache", "--pipelines", "basic", "--metrics-level", "core", "--max-queries", "5", "--verbose"], "Run RAGAs without cache."),
        ]
        for tid, tcmd_args, tdesc in lightweight_ragas_targets_defs:
            targets.append(TestTarget(
                id=tid,
                command=[self.conda_run_prefix, PYTHON_CMD] + tcmd_args,
                description=f"{tdesc} (NOTE: Script '{missing_ragas_script}' reported as missing)",
                category="ragas_lightweight",
                runnable=False # Mark as not runnable
            ))
        # Parameterized ragas target (also missing script)
        targets.append(TestTarget(
            id="ragas-parameterized",
            command=[self.conda_run_prefix, PYTHON_CMD, missing_ragas_script, "--pipelines", "$(PIPELINES)", "--metrics-level", "$(METRICS)", "$(QUERIES)"], # Placeholder for actual parameter substitution logic if implemented
            description=f"Parameterized RAGAs run. (NOTE: Script '{missing_ragas_script}' reported as missing, parameter substitution not implemented in this orchestrator version)",
            category="ragas_lightweight",
            runnable=False
        ))

        # --- 1.5. TDD with RAGAS Testing ---
        tdd_ragas_test_script = "tests/test_tdd_performance_with_ragas.py"
        targets.extend([
            TestTarget(id="test-performance-ragas-tdd", command=[self.conda_run_prefix, "pytest", tdd_ragas_test_script, "-m", "performance_ragas", "-v"], description="Run TDD performance benchmark tests with RAGAS quality metrics.", category="tdd_ragas", parallel_safe=True),
            TestTarget(id="test-scalability-ragas-tdd", command=[self.conda_run_prefix, "pytest", tdd_ragas_test_script, "-m", "scalability_ragas", "-v"], description="Run TDD scalability tests with RAGAS.", category="tdd_ragas", parallel_safe=True),
            TestTarget(id="test-tdd-comprehensive-ragas", command=[self.conda_run_prefix, "pytest", tdd_ragas_test_script, "-m", "ragas_integration", "-v"], description="Run all TDD RAGAS integration tests.", category="tdd_ragas", parallel_safe=True),
            TestTarget(id="test-1000-enhanced-tdd", command=[self.conda_run_prefix, "pytest", tdd_ragas_test_script, "-m", "ragas_integration", "-v"], env_vars={"TEST_DOCUMENT_COUNT": "1000"}, description="TDD RAGAS tests with 1000+ documents.", category="tdd_ragas", parallel_safe=True),
            TestTarget(id="test-tdd-ragas-quick", command=[self.conda_run_prefix, "pytest", tdd_ragas_test_script, "-m", "performance_ragas", "-v"], env_vars={"TDD_RAGAS_QUICK_MODE": "true"}, description="Quick version of TDD RAGAS performance tests.", category="tdd_ragas", parallel_safe=True),
            TestTarget(id="ragas-with-tdd-report-generation", command=[self.conda_run_prefix, PYTHON_CMD, "scripts/generate_tdd_ragas_performance_report.py"], description="Generate detailed report for TDD RAGAS tests.", category="tdd_ragas", dependencies=["test-tdd-comprehensive-ragas"]),
            TestTarget(id="ragas-with-tdd", command=[], description="Run comprehensive TDD RAGAS tests and generate report.", category="tdd_ragas", dependencies=["test-tdd-comprehensive-ragas", "ragas-with-tdd-report-generation"]),
        ])
        
        # --- 1.6. Validation Tests ---
        validate_pipeline_script = str(PROJECT_ROOT / "scripts/utilities/validate_pipeline.py")
        
        targets.append(self.build_iris_rag_validation_command()) # validate-iris-rag
        
        for p_name in pipeline_names:
            targets.append(TestTarget(
                id=f"validate-pipeline-{p_name}",
                command=[self.conda_run_prefix, PYTHON_CMD, validate_pipeline_script, "validate", p_name],
                description=f"Validate pipeline '{p_name}' with pre-condition checks.",
                category="validation"
            ))
        targets.append(TestTarget(
            id="validate-all-pipelines", 
            command=[], # Handled by dependencies
            description="Validate all 7 pipeline types.", 
            category="validation",
            dependencies=[f"validate-pipeline-{p_name}" for p_name in pipeline_names]
        ))
        targets.extend([
            TestTarget(id="test-framework-integration", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/validate_testing_framework_integration.py"), "--verbose"], description="Validate testing framework integration.", category="validation"),
            TestTarget(id="test-install", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/run_post_installation_tests.py")], description="Run post-installation validation tests.", category="validation", dependencies=["install"]), # Assuming 'install' is a setup target
            TestTarget(id="test-e2e-validation-script", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/run_e2e_validation.py"), "--verbose"], description="Run comprehensive E2E validation script with Docker management.", category="validation", timeout=DEFAULT_TIMEOUT * 2),
            TestTarget(id="test-mode-validator-pytest", command=[self.conda_run_prefix, "pytest", "tests/test_mode_validator.py", "-v"], description="Validate mock control system for test modes using pytest.", category="validation", parallel_safe=True),
            TestTarget(id="validate-all", command=[], description="Comprehensive system validation.", category="validation", dependencies=["validate-iris-rag", "test-dbapi", "check-data", "validate-all-pipelines"]), # check-data needs to be defined
            TestTarget(id="prod-check", command=[], description="Production readiness checks with auto-setup.", category="validation", dependencies=["validate-iris-rag", "test-dbapi", "auto-setup-all"]), # auto-setup-all needs to be defined
        ])

        # --- 1.7. Test Mode Framework Specific Targets ---
        targets.extend([
            TestTarget(id="test-unit-mode", command=[self.conda_run_prefix, "pytest", "tests/", "-m", "unit or not e2e", "-v"], env_vars={"RAG_TEST_MODE": "unit"}, description="Run tests in UNIT mode (mocks enabled).", category="test_mode_framework", parallel_safe=True),
            TestTarget(id="test-e2e-mode", command=[self.conda_run_prefix, "pytest", "tests/", "-m", "e2e or not unit", "-v"], env_vars={"RAG_TEST_MODE": "e2e", "RAG_MOCKS_DISABLED": "true"}, description="Run tests in E2E mode (mocks disabled).", category="test_mode_framework", parallel_safe=True),
        ])

        # --- 1.8. Other Test-Related Targets ---
        targets.extend([
            TestTarget(id="test-dbapi", command=[self.conda_run_prefix, PYTHON_CMD, "-c", "from common.iris_connection_manager import get_dbapi_connection; conn = get_dbapi_connection(); print(f'DBAPI Connection: {conn}'); conn.close()"], description="Test DBAPI connection.", category="other"),
            TestTarget(id="test-jdbc", command=[self.conda_run_prefix, PYTHON_CMD, "-c", "from common.iris_connection_manager import IRISConnectionManager; icm = IRISConnectionManager(); conn = icm.get_connection(); print(f'JDBC Connection: {conn}'); conn.close()"], description="Test JDBC connection.", category="other"),
            TestTarget(id="proof-of-concept", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/proof_of_concept_demo.py")], description="Run proof of concept demo script.", category="other"),
        ])
        # Parameterized test-pipeline: This would require more complex command generation or a wrapper.
        # For now, let's add one example.
        targets.append(TestTarget(
            id="test-pipeline-basic-example",
            command=[self.conda_run_prefix, PYTHON_CMD, "-c", f"from iris_rag.pipelines import BasicRAGPipeline; p = BasicRAGPipeline(); print(p.invoke('test query'))"], # Simplified example
            description="Quick test for 'basic' pipeline (example, needs auto-setup dependency).",
            category="other",
            dependencies=["auto-setup-pipeline-basic"] # auto-setup-pipeline-basic needs to be defined
        ))

        # --- Self-Healing Data Validation Targets ---
        # These are more complex and might involve scripts like data_population_manager.py
        # Adding placeholders, actual commands might need refinement.
        data_pop_mgr_script = str(PROJECT_ROOT / "scripts/data_population_manager.py") # Assuming this script exists and has relevant commands
        targets.extend([
            TestTarget(id="validate-healing", command=[self.conda_run_prefix, PYTHON_CMD, data_pop_mgr_script, "validate-healing-status"], description="Validate data healing status.", category="data_healing"),
            TestTarget(id="heal-data", command=[self.conda_run_prefix, PYTHON_CMD, data_pop_mgr_script, "heal"], description="Run data healing process.", category="data_healing", setup_target=True), # This is more of a setup
            TestTarget(id="heal-and-test-1000", command=[], description="Heal data and run test-1000.", category="data_healing", dependencies=["heal-data", "test-1000"]),
            TestTarget(id="heal-and-validate-all", command=[], description="Heal data and run validate-all.", category="data_healing", dependencies=["heal-data", "validate-all"]),
        ])
        
        # Placeholder for 'check-data' if it's a script
        targets.append(TestTarget(id="check-data", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/utilities/check_data_integrity.py")], description="Check data integrity.", category="validation", runnable=Path(PROJECT_ROOT / "scripts/utilities/check_data_integrity.py").exists()))


        self.all_targets = {t.id: t for t in targets}
        self.logger.info(f"Defined {len(self.all_targets)} test targets.")

    def define_setup_targets(self) -> None:
        """Defines setup and teardown targets."""
        setup_targets_list = []
        pipeline_names = ["basic", "hyde", "crag", "colbert", "noderag", "graphrag", "hybrid_ifind"]
        validate_pipeline_script = str(PROJECT_ROOT / "scripts/utilities/validate_pipeline.py") # Used for auto-setup

        # Install might be a make target or a script. Assuming a general concept.
        setup_targets_list.append(TestTarget(id="install", command=["make", "install"], description="Run project installation.", category="setup", setup_target=True, allow_failure=False)) # Critical
        setup_targets_list.append(TestTarget(id="clean", command=["make", "clean"], description="Clean project build artifacts and caches.", category="setup", setup_target=True))
        
        # Docker related targets (example, actual commands might vary)
        setup_targets_list.append(TestTarget(id="docker-up", command=["docker-compose", "up", "-d"], description="Start Docker services.", category="setup", setup_target=True, allow_failure=False))
        setup_targets_list.append(TestTarget(id="docker-down", command=["docker-compose", "down"], description="Stop Docker services.", category="setup", setup_target=True))

        # Data loading
        # Assuming a script or make target for this, e.g., from Makefile: data-load-1000: scripts/load_1000_docs.sh
        # For simplicity, let's assume a python script exists or a make target
        setup_targets_list.append(TestTarget(id="data-load-1000", command=[self.conda_run_prefix, PYTHON_CMD, str(PROJECT_ROOT / "scripts/utilities/load_pmc_docs.py"), "--count", "1000"], description="Load 1000 PMC documents into IRIS.", category="setup", setup_target=True, allow_failure=False, runnable=Path(PROJECT_ROOT / "scripts/utilities/load_pmc_docs.py").exists()))
        
        # Auto-setup targets (from testing_system_analysis.md, `validate_pipeline.py` seems to handle `auto-setup` action)
        for p_name in pipeline_names:
            setup_targets_list.append(TestTarget(
                id=f"auto-setup-pipeline-{p_name}",
                command=[self.conda_run_prefix, PYTHON_CMD, validate_pipeline_script, "auto-setup", p_name],
                description=f"Auto-setup for pipeline '{p_name}'.",
                category="setup",
                setup_target=True,
                allow_failure=False
            ))
        setup_targets_list.append(TestTarget(
            id="auto-setup-all", 
            command=[], # Handled by dependencies
            description="Auto-setup for all 7 pipeline types.", 
            category="setup",
            setup_target=True,
            allow_failure=False,
            dependencies=[f"auto-setup-pipeline-{p_name}" for p_name in pipeline_names]
        ))

        for t in setup_targets_list:
            if t.id not in self.all_targets:
                self.all_targets[t.id] = t
            else: # If it was already defined as a test target, update its setup_target flag
                self.all_targets[t.id].setup_target = True
                self.all_targets[t.id].category = "setup" # Prioritize setup category
                self.all_targets[t.id].allow_failure = t.allow_failure # Ensure critical setup steps are not allowed to fail silently

        self.logger.info(f"Defined/updated {len(setup_targets_list)} setup targets.")


    def build_iris_rag_validation_command(self) -> TestTarget:
        """Builds the TestTarget for 'validate-iris-rag'."""
        # This command is a series of inline Python imports.
        # For simplicity, we can try to run them as separate -c commands or combine them.
        # Combining them is safer for sequential imports.
        py_commands = [
            "from iris_rag import IRISRAG",
            "from iris_rag.embeddings import BaseRAGEmbeddings, OpenAIEmbeddings",
            "from iris_rag.llms import BaseRAGLLM, OpenAI",
            "from iris_rag.vector_stores import BaseRAGVectorStore, IRISVectorStore",
            "from iris_rag.retrievers import BaseRAGRetriever, IRISRetriever",
            "from iris_rag.loaders import BaseRAGLoader, IRISLoader",
            "from iris_rag.text_splitters import BaseRAGTextSplitter, IRISTokenSplitter",
            "print('Successfully imported core IRIS RAG components.')"
        ]
        full_py_script = "; ".join(py_commands)
        cmd = [self.conda_run_prefix, PYTHON_CMD, "-c", full_py_script]
        return TestTarget(
            id="validate-iris-rag",
            command=cmd,
            description="Validates iris_rag package imports and basic model functionality.",
            category="validation",
            parallel_safe=True # Simple import check
        )

    def collect_environment_info(self) -> Dict[str, Any]:
        """Collects information about the execution environment."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "conda_env": os.getenv("CONDA_DEFAULT_ENV", "N/A"),
            "project_root": str(PROJECT_ROOT),
            "cwd": str(Path.cwd()),
            "user": os.getenv("USER", os.getenv("USERNAME", "N/A")),
            "orchestrator_version": "1.0.0" # Example version
        }

    def filter_targets_by_categories(self, targets_to_run: List[str], categories: Optional[List[str]]) -> List[str]:
        """Filters the list of targets to run based on specified categories."""
        if not categories:
            return targets_to_run
        
        filtered_targets = []
        for target_id in targets_to_run:
            target = self.all_targets.get(target_id)
            if target and target.category in categories:
                filtered_targets.append(target_id)
        
        self.logger.info(f"Filtered targets by categories: {categories}. Original: {len(targets_to_run)}, Filtered: {len(filtered_targets)}")
        return filtered_targets

    def resolve_dependencies(self, target_ids: List[str]) -> List[str]:
        """
        Resolves dependencies and returns a topologically sorted list of target IDs.
        Includes all dependencies, even if not in the initial `target_ids` list.
        """
        resolved_order: List[str] = []
        visited: Set[str] = set()  # For detecting cycles and marking completion
        recursion_stack: Set[str] = set() # For detecting cycles during current recursion

        all_targets_to_consider: Set[str] = set()
        
        # Build the full set of targets to consider, including all dependencies
        queue = list(target_ids)
        processed_for_deps: Set[str] = set()
        while queue:
            current_id = queue.pop(0)
            if current_id in processed_for_deps:
                continue
            processed_for_deps.add(current_id)
            all_targets_to_consider.add(current_id)
            
            target = self.all_targets.get(current_id)
            if not target:
                self.logger.warning(f"Dependency resolution: Target '{current_id}' not defined. Skipping.")
                continue
            for dep_id in target.dependencies:
                if dep_id not in processed_for_deps:
                    queue.append(dep_id)

        self.logger.info(f"Full set of targets and their dependencies to resolve: {sorted(list(all_targets_to_consider))}")

        def visit(target_id: str):
            if target_id not in self.all_targets:
                self.logger.error(f"Dependency '{target_id}' not found in defined targets. Cycle or missing definition.")
                # Optionally raise an error or mark as unrunnable
                # For now, we'll skip it, but this indicates a definition issue.
                if target_id in recursion_stack: recursion_stack.remove(target_id) # Clean up stack
                return

            if target_id in recursion_stack:
                raise Exception(f"Circular dependency detected: ... -> {target_id} -> ...")
            
            if target_id not in visited:
                recursion_stack.add(target_id)
                target = self.all_targets[target_id]
                for dep_id in target.dependencies:
                    visit(dep_id)
                
                recursion_stack.remove(target_id)
                visited.add(target_id)
                resolved_order.append(target_id)

        for target_id in sorted(list(all_targets_to_consider)): # Sort for deterministic behavior if possible
            if target_id not in visited:
                visit(target_id)
        
        # Filter resolved_order to only include initially requested targets and their *actual* dependencies
        # The current `resolved_order` contains all items from `all_targets_to_consider` in order.
        # If we only want to run the explicitly requested `target_ids` and their necessary precursors,
        # this list is correct. If `target_ids` was meant as a filter *after* full graph resolution,
        # then a further filter step would be needed. The current implementation assumes `target_ids`
        # are the "entry points" and all their dependencies must run.
        
        self.logger.info(f"Resolved execution order: {resolved_order}")
        return resolved_order

    def execute_target(self, target_id: str) -> TestResult:
        """Executes a single test target and returns its result."""
        target = self.all_targets.get(target_id)
        if not target:
            self.logger.error(f"Target '{target_id}' not defined. Skipping execution.")
            return TestResult(target_id=target_id, status=TestStatus.ERROR, start_time=time.time(), end_time=time.time(), duration=0, stdout="", stderr="", error_message="Target not defined")

        if not target.runnable:
            self.logger.warning(f"Target '{target.id}' is marked as not runnable (e.g., script missing). Skipping.")
            start_time = time.time()
            return TestResult(target_id=target.id, status=TestStatus.SKIPPED, start_time=start_time, end_time=start_time, duration=0, stdout="", stderr="Marked as not runnable")

        self.logger.info(f"Executing target: {target.id} ({target.description})")
        self.logger.debug(f"Command: {' '.join(target.command)}")
        if target.cwd != PROJECT_ROOT:
             self.logger.debug(f"CWD: {target.cwd}")
        if target.env_vars:
            self.logger.debug(f"Env Vars: {target.env_vars}")

        start_time = time.time()
        status = TestStatus.PENDING
        stdout_str, stderr_str = "", ""
        return_code = None
        
        current_env = os.environ.copy()
        if target.env_vars:
            current_env.update(target.env_vars)

        try:
            status = TestStatus.RUNNING
            process = subprocess.run(
                target.command,
                cwd=target.cwd,
                env=current_env,
                capture_output=True,
                text=True,
                timeout=target.timeout,
                check=False # We check returncode manually
            )
            stdout_str = process.stdout
            stderr_str = process.stderr
            return_code = process.returncode
            
            if process.returncode == 0:
                status = TestStatus.SUCCESS
                self.logger.info(f"Target '{target.id}' completed successfully.")
            else:
                status = TestStatus.FAILURE
                self.logger.error(f"Target '{target.id}' failed with return code {process.returncode}.")
                self.logger.error(f"Stderr for {target.id}:\n{stderr_str}")

        except subprocess.TimeoutExpired:
            status = TestStatus.TIMEOUT
            stderr_str = f"Target '{target.id}' timed out after {target.timeout} seconds."
            self.logger.error(stderr_str)
        except Exception as e:
            status = TestStatus.ERROR # Error in execution framework
            stderr_str = f"Error executing target '{target.id}': {e}\n{traceback.format_exc()}"
            self.logger.error(stderr_str)
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.debug(f"STDOUT for {target.id}:\n{stdout_str}")
        if status != TestStatus.SUCCESS and stderr_str:
             self.logger.debug(f"STDERR for {target.id}:\n{stderr_str}")


        return TestResult(
            target_id=target.id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            stdout=stdout_str,
            stderr=stderr_str,
            return_code=return_code
        )

    def execute_targets_sequentially(self, target_ids_ordered: List[str]) -> List[TestResult]:
        """Executes a list of targets sequentially."""
        results = []
        for target_id in target_ids_ordered:
            result = self.execute_target(target_id)
            results.append(result)
            self.results.results.append(result) # Add to comprehensive results
            if result.status in [TestStatus.FAILURE, TestStatus.TIMEOUT, TestStatus.ERROR] and not self.all_targets[target_id].allow_failure:
                self.logger.error(f"Critical target '{target_id}' failed. Aborting sequential execution.")
                # Mark remaining targets in this sequence as skipped (if any were planned beyond this)
                # This logic depends on how `target_ids_ordered` is used. If it's the full plan, then subsequent ones are skipped.
                # For now, this function just executes what's passed to it. The caller (`run_comprehensive_tests`) handles overall flow.
                break 
        return results

    def execute_targets_parallel(self, target_ids: List[str]) -> List[TestResult]:
        """Executes a list of targets in parallel if they are parallel_safe."""
        results = []
        
        safe_targets_to_run = [tid for tid in target_ids if self.all_targets.get(tid) and self.all_targets[tid].parallel_safe and self.all_targets[tid].runnable]
        unsafe_targets_to_run_sequentially = [tid for tid in target_ids if tid not in safe_targets_to_run] # Includes unrunnable or non-parallel-safe

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_target = {executor.submit(self.execute_target, target_id): target_id for target_id in safe_targets_to_run}
            for future in concurrent.futures.as_completed(future_to_target):
                target_id = future_to_target[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self.logger.error(f"Target '{target_id}' generated an exception during parallel execution: {exc}")
                    start_time = time.time() # Approximate
                    result = TestResult(target_id=target_id, status=TestStatus.ERROR, start_time=start_time, end_time=time.time(), duration=0, stdout="", stderr=str(exc), error_message=str(exc))
                results.append(result)
                self.results.results.append(result) # Add to comprehensive results

        # Execute non-parallel-safe targets sequentially
        if unsafe_targets_to_run_sequentially:
            self.logger.info(f"Executing {len(unsafe_targets_to_run_sequentially)} non-parallel-safe or unrunnable (will be skipped) targets sequentially...")
            sequential_results = self.execute_targets_sequentially(unsafe_targets_to_run_sequentially)
            results.extend(sequential_results)
            # self.results.results is already updated by execute_targets_sequentially

        return results
        
    def format_duration(self, seconds: float) -> str:
        """Formats duration in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}m"
        else:
            return f"{seconds/3600:.2f}h"

    def generate_summary_statistics(self) -> Dict[str, int]:
        """Generates summary statistics from the test results."""
        summary = {status.value: 0 for status in TestStatus}
        for result in self.results.results:
            summary[result.status.value] += 1
        self.results.summary = summary
        return summary

    def save_json_report(self) -> None:
        """Saves the comprehensive test results as a JSON file."""
        self.results.end_time = datetime.now()
        self.results.total_duration = (self.results.end_time - self.results.start_time).total_seconds()
        self.generate_summary_statistics() # Ensure summary is up-to-date

        report_path = self.reports_dir / f"{self.results.run_id}_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(self.results.to_dict(), f, indent=4)
            self.logger.info(f"JSON report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON report: {e}")

    def save_markdown_report(self) -> None:
        """Saves a summary of test results as a Markdown file."""
        self.results.end_time = datetime.now() # Ensure end_time is current
        self.results.total_duration = (self.results.end_time - self.results.start_time).total_seconds()
        summary = self.generate_summary_statistics()

        report_path = self.reports_dir / f"{self.results.run_id}_summary.md"
        
        try:
            with open(report_path, 'w') as f:
                f.write(f"# Test Run Summary: {self.results.run_id}\n\n")
                f.write(f"- **Start Time**: {self.results.start_time.isoformat()}\n")
                f.write(f"- **End Time**: {self.results.end_time.isoformat()}\n")
                f.write(f"- **Total Duration**: {self.format_duration(self.results.total_duration)}\n\n")
                
                f.write("## Environment Information\n")
                for key, value in self.results.environment_info.items():
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")

                f.write("## Overall Summary\n")
                for status_val, count in summary.items():
                    f.write(f"- **{status_val}**: {count}\n")
                f.write("\n")

                f.write("## Detailed Results\n")
                f.write("| Target ID | Status | Duration | Return Code |\n")
                f.write("|-----------|--------|----------|-------------|\n")
                for result in sorted(self.results.results, key=lambda r: r.start_time):
                    target = self.all_targets.get(result.target_id)
                    desc_short = f" ({target.description[:30]}...)" if target else ""
                    f.write(f"| {result.target_id}{desc_short} | {result.status.value} | {self.format_duration(result.duration)} | {result.return_code if result.return_code is not None else 'N/A'} |\n")
                
                f.write("\n## Failures and Errors\n")
                failures_found = False
                for result in self.results.results:
                    if result.status in [TestStatus.FAILURE, TestStatus.TIMEOUT, TestStatus.ERROR]:
                        failures_found = True
                        f.write(f"### Target: {result.target_id} - Status: {result.status.value}\n")
                        f.write(f"**Command:** `{' '.join(self.all_targets[result.target_id].command)}`\n")
                        if result.stderr:
                            f.write("\n**Stderr:**\n```\n")
                            f.write(result.stderr[:2000] + ("..." if len(result.stderr) > 2000 else "")) # Limit length
                            f.write("\n```\n")
                        if result.stdout:
                            f.write("\n**Stdout (last 10 lines):**\n```\n")
                            stdout_lines = result.stdout.strip().split('\n')
                            f.write("\n".join(stdout_lines[-10:]))
                            f.write("\n```\n")
                        f.write("\n---\n")
                if not failures_found:
                     f.write("No failures, timeouts, or errors reported.\n")

            self.logger.info(f"Markdown report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save Markdown report: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def run_comprehensive_tests(self, targets_to_run_ids: Optional[List[str]] = None, 
                                categories: Optional[List[str]] = None, 
                                run_parallel: bool = False, 
                                skip_setup: bool = False) -> None:
        """
        Main orchestration logic.
        
        Args:
            targets_to_run_ids: Specific list of target IDs to run. If None, runs all non-setup targets.
            categories: Filter targets by these categories.
            run_parallel: If True, attempts to run parallel_safe targets concurrently.
            skip_setup: If True, skips execution of targets marked as setup_target.
        """
        self.results.start_time = datetime.now() # Reset start time for this specific run
        self.logger.info(f"Starting comprehensive test run: {self.results.run_id}")

        self.define_test_targets() # Define test targets
        self.define_setup_targets() # Define setup targets

        if targets_to_run_ids is None:
            # Default: run all non-setup targets
            runnable_target_ids = [tid for tid, t in self.all_targets.items() if not t.setup_target and t.runnable]
        else:
            runnable_target_ids = [tid for tid in targets_to_run_ids if self.all_targets.get(tid) and self.all_targets[tid].runnable]
            missing_ids = [tid for tid in targets_to_run_ids if not self.all_targets.get(tid)]
            if missing_ids:
                self.logger.warning(f"Specified target IDs not defined and will be skipped: {missing_ids}")
        
        if categories:
            runnable_target_ids = self.filter_targets_by_categories(runnable_target_ids, categories)

        if not runnable_target_ids:
            self.logger.warning("No runnable targets selected after filtering. Exiting.")
            self.save_reports()
            return

        try:
            execution_plan = self.resolve_dependencies(runnable_target_ids)
        except Exception as e:
            self.logger.error(f"Failed to resolve dependencies: {e}. Aborting run.")
            # Add an error result to the main results
            err_res = TestResult("dependency_resolution", TestStatus.ERROR, time.time(), time.time(), 0, "", str(e), error_message=str(e))
            self.results.results.append(err_res)
            self.save_reports()
            return

        # Separate setup targets from the main execution plan if skip_setup is not True
        final_setup_targets = []
        final_test_targets = []

        for target_id in execution_plan:
            target = self.all_targets.get(target_id)
            if not target: continue # Should have been caught by resolve_dependencies

            if target.setup_target:
                if not skip_setup:
                    final_setup_targets.append(target_id)
                else:
                    self.logger.info(f"Skipping setup target due to --skip-setup: {target_id}")
            else:
                # Only add if it was part of the initial runnable_target_ids or a dependency of one.
                # resolve_dependencies gives *all* precursors. We need to ensure we only run what was asked for or its deps.
                # The current `execution_plan` should be correct as it's built from `runnable_target_ids` and their deps.
                final_test_targets.append(target_id)
        
        if final_setup_targets:
            self.logger.info(f"Executing {len(final_setup_targets)} setup targets sequentially...")
            setup_results = self.execute_targets_sequentially(final_setup_targets)
            # Check for critical setup failures
            if any(r.status != TestStatus.SUCCESS and not self.all_targets[r.target_id].allow_failure for r in setup_results):
                self.logger.error("Critical setup target failed. Aborting further test execution.")
                self.save_reports()
                return
        
        self.logger.info(f"Executing {len(final_test_targets)} test targets...")
        if run_parallel:
            self.logger.info("Attempting parallel execution where possible.")
            self.execute_targets_parallel(final_test_targets)
        else:
            self.logger.info("Executing targets sequentially.")
            self.execute_targets_sequentially(final_test_targets)

        self.save_reports()
        self.logger.info(f"Comprehensive test run {self.results.run_id} finished.")
        self.log_summary()

    def save_reports(self):
        """Saves JSON and Markdown reports."""
        self.save_json_report()
        self.save_markdown_report()
    
    def log_summary(self):
        """Logs a brief summary to the console."""
        summary = self.results.summary
        self.logger.info("--- Test Run Summary ---")
        for status, count in summary.items():
            if count > 0:
                self.logger.info(f"{status}: {count}")
        self.logger.info(f"Total duration: {self.format_duration(self.results.total_duration)}")
        self.logger.info(f"Reports saved in: {self.reports_dir.resolve()}")
        self.logger.info(f"JSON report: {self.reports_dir.resolve() / (self.results.run_id + '_report.json')}")
        self.logger.info(f"Markdown summary: {self.reports_dir.resolve() / (self.results.run_id + '_summary.md')}")
        self.logger.info(f"Log file: {self.reports_dir.resolve() / (self.results.run_id + '.log')}")


def main():
    parser = argparse.ArgumentParser(description="Master Test Orchestration Script for RAG Templates Project.")
    parser.add_argument(
        "--targets", 
        nargs="*", 
        help="Specific list of target IDs to run. If not provided, runs all relevant non-setup tests."
    )
    parser.add_argument(
        "--categories", 
        nargs="*", 
        help="Filter targets to run by these categories (e.g., core_pytest validation)."
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Enable parallel execution for 'parallel_safe' targets."
    )
    parser.add_argument(
        "--parallel-workers", 
        type=int, 
        default=4, 
        help="Number of workers for parallel execution."
    )
    parser.add_argument(
        "--reports-dir", 
        type=str, 
        default=str(DEFAULT_REPORTS_DIR), 
        help="Directory to save reports and logs."
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=DEFAULT_TIMEOUT, 
        help="Default timeout in seconds for each test target."
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List all defined test targets and their categories, then exit."
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all unique target categories and then exit."
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip execution of targets marked as setup_target (e.g. install, data-load)."
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=CONDA_ENV_NAME,
        help=f"Name of the conda environment to use (default: {CONDA_ENV_NAME})."
    )

    args = parser.parse_args()

    orchestrator = MasterTestOrchestrator(
        reports_dir=Path(args.reports_dir),
        parallel_workers=args.parallel_workers,
        default_timeout=args.timeout,
        conda_env_name=args.conda_env
    )
    
    # Populate targets to allow listing
    orchestrator.define_test_targets()
    orchestrator.define_setup_targets()


    if args.list_targets:
        print("Defined Test Targets:")
        print("---------------------")
        for target_id, target in sorted(orchestrator.all_targets.items()):
            runnable_status = "" if target.runnable else " (NOT RUNNABLE)"
            setup_status = " [SETUP]" if target.setup_target else ""
            print(f"- ID: {target.id}{setup_status}{runnable_status}")
            print(f"  Description: {target.description}")
            print(f"  Category: {target.category}")
            print(f"  Command: {' '.join(target.command)}")
            if target.dependencies:
                print(f"  Dependencies: {', '.join(target.dependencies)}")
            print(f"  Parallel Safe: {target.parallel_safe}")
            print("---")
        return

    if args.list_categories:
        print("Available Target Categories:")
        print("---------------------------")
        categories = sorted(list(set(t.category for t in orchestrator.all_targets.values())))
        for cat in categories:
            print(f"- {cat}")
        return

    orchestrator.run_comprehensive_tests(
        targets_to_run_ids=args.targets,
        categories=args.categories,
        run_parallel=args.parallel,
        skip_setup=args.skip_setup
    )

if __name__ == "__main__":
    import traceback # For main exception block
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)