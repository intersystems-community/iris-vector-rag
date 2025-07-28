# Container Lifecycle Management Architecture

## Overview

This document outlines the architecture for managing IRIS container lifecycle with dynamic compose file detection and container restart capabilities.

## Current Architecture

```
pytest session
    ↓
iris_container fixture (session scope)
    ↓
ContainerManager
    ↓
Docker Compose operations
```

## Enhanced Architecture

```
pytest session
    ↓
iris_container fixture (session scope)
    ↓
ContainerLifecycleManager
    ├── ComposeFileTracker (detects changes)
    ├── ContainerManager (manages containers)
    └── ContainerStateManager (tracks state)
```

## Components

### 1. ComposeFileTracker
- **Responsibility**: Detect changes to COMPOSE_FILE environment variable
- **Interface**: `has_compose_file_changed() -> bool`
- **State Management**: Tracks last known compose file path

### 2. ContainerManager (Enhanced)
- **Responsibility**: Docker container operations
- **New Methods**: `stop_iris()`, `restart_iris()`
- **Interface**: Maintains existing API for backward compatibility

### 3. ContainerStateManager
- **Responsibility**: Track container state across test sessions
- **Interface**: `get_last_compose_file()`, `set_last_compose_file()`
- **Storage**: Uses temporary file for persistence

### 4. ContainerLifecycleManager
- **Responsibility**: Orchestrate container lifecycle decisions
- **Interface**: `ensure_correct_container_running()`
- **Logic**: Coordinates between tracker, manager, and state components

## Design Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Open/Closed**: Easy to extend without modifying existing code
3. **Dependency Inversion**: Components depend on abstractions
4. **Separation of Concerns**: Clear boundaries between detection, management, and state

## Implementation Strategy

1. Create modular components with clear interfaces
2. Maintain backward compatibility with existing fixtures
3. Use composition over inheritance
4. Implement proper error handling and logging
5. Keep files under 500 lines per project rules

## Benefits

- **Reliability**: Ensures correct container is always running
- **Flexibility**: Easy to extend for other environment variables
- **Maintainability**: Clear separation of concerns
- **Testability**: Each component can be tested independently