# API Contracts: Configurable Test Backend Modes

**Date**: 2025-10-08
**Feature**: 035-make-2-modes
**Phase**: Phase 1 - Contracts

## Overview

This directory contains API contracts defining the interfaces for backend mode configuration, IRIS edition detection, and iris-devtools integration. These contracts serve as the specification for contract tests (TDD) and implementation.

## Contract Files

### 1. backend_config_contract.yaml
- **Purpose**: Backend mode configuration API
- **Operations**: Load config, validate config, get current mode
- **Test File**: `tests/contract/test_backend_mode_config.py`
- **Requirements**: FR-001, FR-002, FR-009, FR-012

### 2. edition_detection_contract.yaml
- **Purpose**: IRIS edition detection API
- **Operations**: Detect edition from connection, validate match
- **Test File**: `tests/contract/test_edition_detection.py`
- **Requirements**: FR-008

### 3. iris_devtools_contract.yaml
- **Purpose**: iris-devtools bridge API
- **Operations**: Container lifecycle, schema reset, connection validation, health checks
- **Test File**: `tests/contract/test_iris_devtools_integration.py`
- **Requirements**: FR-006, FR-007, FR-013

### 4. connection_pool_contract.yaml
- **Purpose**: Connection pool management API
- **Operations**: Acquire connection, release connection, get active count
- **Test File**: `tests/contract/test_connection_pooling.py`
- **Requirements**: FR-003, FR-011

### 5. execution_strategy_contract.yaml
- **Purpose**: Test execution strategy API
- **Operations**: Determine strategy, enforce limits
- **Test File**: `tests/contract/test_execution_strategies.py`
- **Requirements**: FR-004, FR-005

## Contract Format

All contracts follow this structure:

```yaml
# Contract metadata
name: <Contract Name>
version: 1.0.0
description: <Purpose>

# Operations
operations:
  - name: <operation_name>
    description: <What it does>
    inputs:
      - name: <param_name>
        type: <type>
        required: <true|false>
        description: <param purpose>
    outputs:
      type: <return_type>
      description: <what is returned>
    errors:
      - code: <ERROR_CODE>
        message: <error message template>
        resolution: <how to fix>

# Test scenarios (for contract tests)
test_scenarios:
  - name: <scenario_name>
    description: <test purpose>
    given: <initial state>
    when: <action>
    then: <expected outcome>
    requirement: <FR-XXX reference>
```

## Usage in Tests

Contract tests are generated from these contracts and MUST fail initially (no implementation). They validate:

1. **Interface Compliance**: Operations match contract signatures
2. **Error Handling**: All documented errors are raised correctly
3. **Validation**: Input validation per contract specs
4. **Return Types**: Outputs match contract types

Example contract test pattern:

```python
import pytest
from iris_rag.testing.backend_manager import BackendConfiguration

def test_load_config_returns_valid_configuration():
    """Contract: backend_config_contract.yaml :: load_config"""
    # This test will FAIL until implementation exists
    config = BackendConfiguration.load()

    assert isinstance(config, BackendConfiguration)
    assert config.mode in [BackendMode.COMMUNITY, BackendMode.ENTERPRISE]
    assert config.max_connections > 0
```

## Constitutional Alignment

- **II. Pipeline Validation**: Contracts define validation requirements
- **III. TDD**: Tests written from contracts before implementation
- **VI. Explicit Error Handling**: All error scenarios documented
- **VII. Standardized Interfaces**: Contracts enforce consistent patterns
