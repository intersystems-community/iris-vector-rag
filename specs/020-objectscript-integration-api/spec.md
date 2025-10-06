# ObjectScript Integration API Specification

## Overview
The RAG Templates ObjectScript Integration API provides native ObjectScript access to the RAG framework capabilities, enabling IRIS applications to consume RAG services directly through ObjectScript classes and methods. This integration maintains API parity with Python and JavaScript implementations while leveraging IRIS's native ObjectScript capabilities for seamless enterprise integration.

## Primary User Story
IRIS application developers and ObjectScript programmers need native ObjectScript classes to integrate RAG capabilities into their applications without requiring Python dependencies or external API calls. The ObjectScript integration should provide the same functionality as Python and JavaScript APIs while leveraging ObjectScript's strengths in data access, class hierarchies, and IRIS integration patterns.

## Acceptance Scenarios

### AC-001: Native ObjectScript RAG Pipeline Access
**GIVEN** an ObjectScript application needs RAG functionality
**WHEN** the developer imports RAG.Pipeline class and creates a pipeline instance
**THEN** the system provides native ObjectScript access to BasicRAG, CRAG, GraphRAG, and HybridGraphRAG pipelines
**AND** supports query execution with ObjectScript data types and return values
**AND** maintains consistent API patterns with Python and JavaScript implementations

### AC-002: ObjectScript-to-Python Bridge Integration
**GIVEN** ObjectScript code calling RAG framework methods
**WHEN** the ObjectScript bridge translates calls to Python RAG pipelines
**THEN** the system provides seamless interoperability without exposing Python implementation details
**AND** handles ObjectScript data types (Lists, Arrays, Objects) conversion to Python equivalents
**AND** provides proper error handling and exception translation between languages

### AC-003: IRIS Database Integration Patterns
**GIVEN** ObjectScript applications with existing IRIS data structures
**WHEN** integrating with RAG pipelines for document processing
**THEN** the system leverages native IRIS SQL, Globals, and Objects for optimal performance
**AND** supports zero-copy data access patterns for existing business data
**AND** integrates with IRIS security and namespace management

### AC-004: Enterprise ObjectScript Development Patterns
**GIVEN** enterprise ObjectScript development requirements
**WHEN** implementing RAG functionality in production applications
**THEN** the system follows ObjectScript best practices for class design, error handling, and logging
**AND** supports ObjectScript debugging, profiling, and monitoring tools
**AND** integrates with IRIS production environments and deployment patterns

### AC-005: Multi-Language API Consistency
**GIVEN** applications using Python, JavaScript, and ObjectScript RAG APIs
**WHEN** comparing functionality and behavior across language implementations
**THEN** all three APIs provide equivalent functionality with consistent behavior
**AND** support the same pipeline types, configuration options, and response formats
**AND** maintain version compatibility across language bindings

## Functional Requirements

### ObjectScript Class Hierarchy
- **FR-001**: System MUST provide RAG.Pipeline abstract class with CreatePipeline() class method for pipeline instantiation
- **FR-002**: System MUST implement specific pipeline classes (RAG.BasicPipeline, RAG.CRAGPipeline, RAG.GraphRAGPipeline, RAG.HybridGraphRAGPipeline)
- **FR-003**: System MUST support ObjectScript method signatures with native %Status return values and ByRef parameters
- **FR-004**: System MUST provide RAG.Response class for standardized query response handling

### Bridge Integration Architecture
- **FR-005**: System MUST implement RAG.Bridge class that manages Python RAG framework integration
- **FR-006**: System MUST provide automatic Python environment detection and initialization with a 10-second timeout before falling back to error handling
- **FR-007**: System MUST handle ObjectScript-to-Python data type conversion for all supported data structures, throwing ObjectScript exceptions immediately when conversion failures occur
- **FR-008**: System MUST translate Python exceptions to ObjectScript %Status error handling patterns

### Configuration and Setup
- **FR-009**: System MUST support ObjectScript parameter classes for pipeline configuration management
- **FR-010**: System MUST integrate with IRIS configuration management (^%SYS, parameters, settings)
- **FR-011**: System MUST provide validation methods for required Python dependencies and environment setup
- **FR-012**: System MUST support namespace-specific configuration and multi-tenant deployments

### Query Processing and Response Handling
- **FR-013**: System MUST support Query() method with ObjectScript %String input and RAG.Response output
- **FR-014**: System MUST provide async query processing through ObjectScript job scheduling and callbacks
- **FR-015**: System MUST handle large document collections through ObjectScript stream processing
- **FR-016**: System MUST support ObjectScript %Collection types for batch query processing

### IRIS Integration Features
- **FR-017**: System MUST leverage IRIS SQL integration for direct database query processing
- **FR-018**: System MUST support IRIS Globals for high-performance data caching and retrieval
- **FR-019**: System MUST integrate with IRIS Object classes for business data object processing
- **FR-020**: System MUST provide ObjectScript methods for vector database operations and management

## Non-Functional Requirements

### Performance
- **NFR-001**: ObjectScript-to-Python bridge overhead MUST add less than 50ms to query processing
- **NFR-002**: System MUST support at least 50 concurrent ObjectScript RAG queries with linear scaling
- **NFR-003**: IRIS-native data access MUST be optimized to avoid unnecessary data copying with a maximum memory usage limit of 500 MB per bridge operation to prevent resource exhaustion
- **NFR-004**: ObjectScript class instantiation MUST complete within 100ms for all pipeline types

### Reliability
- **NFR-005**: ObjectScript integration MUST maintain 99.9% availability through proper error handling
- **NFR-006**: Python environment failures MUST not crash ObjectScript processes or IRIS instances
- **NFR-007**: System MUST provide graceful degradation when Python components are unavailable
- **NFR-008**: ObjectScript error handling MUST follow IRIS %Status patterns with actionable error messages

### Compatibility
- **NFR-009**: System MUST support IRIS 2023.1+ versions with ObjectScript compatibility
- **NFR-010**: ObjectScript API MUST maintain version compatibility with Python RAG framework updates
- **NFR-011**: System MUST work across different IRIS deployment patterns (Community, Standard, Advanced)
- **NFR-012**: Integration MUST support both Windows and Linux IRIS installations

### Enterprise Integration
- **NFR-013**: System MUST integrate with IRIS security model (users, roles, resources) by inheriting IRIS user privileges directly in Python execution environments, following established Embedded Python patterns where Python processes inherit the complete IRIS context including namespace, user, and security privileges
- **NFR-014**: ObjectScript classes MUST support IRIS licensing and auditing requirements
- **NFR-015**: System MUST work within IRIS namespace security and privilege restrictions
- **NFR-016**: Integration MUST support IRIS clustering and mirror database configurations

## Key Entities

### ObjectScript Class Structure
- **RAG.Pipeline**: Abstract base class for all RAG pipeline implementations
- **RAG.BasicPipeline**: ObjectScript wrapper for BasicRAG functionality
- **RAG.CRAGPipeline**: ObjectScript wrapper for CRAG (Corrective RAG) implementation
- **RAG.GraphRAGPipeline**: ObjectScript wrapper for GraphRAG functionality
- **RAG.HybridGraphRAGPipeline**: ObjectScript wrapper for hybrid search capabilities

### Bridge and Integration Components
- **RAG.Bridge**: Core bridge class managing Python integration and lifecycle
- **RAG.PythonManager**: Handles Python environment setup, validation, and process management
- **RAG.DataConverter**: Manages ObjectScript-to-Python data type conversion
- **RAG.ConfigurationManager**: Handles configuration synchronization between ObjectScript and Python

### Response and Data Handling
- **RAG.Response**: Standardized response class with Answer, Sources, and Metadata properties
- **RAG.DocumentCollection**: ObjectScript collection class for document management
- **RAG.QueryRequest**: Request wrapper class for complex query parameters
- **RAG.StreamProcessor**: Handles large data processing through ObjectScript streams

### IRIS Integration Classes
- **RAG.IRISDataAccess**: Optimized data access using IRIS SQL and Globals
- **RAG.VectorManager**: ObjectScript interface to IRIS vector database operations
- **RAG.SecurityManager**: Integration with IRIS security and user management
- **RAG.NamespaceManager**: Multi-tenant support and namespace isolation

## Implementation Guidelines

### ObjectScript Class Design Pattern
```objectscript
Class RAG.Pipeline Extends %RegisteredObject [ Abstract ]
{
    Property Configuration As RAG.Configuration;
    Property Bridge As RAG.Bridge [ Private ];

    ClassMethod CreatePipeline(pipelineType As %String, config As RAG.Configuration = "") As RAG.Pipeline
    {
        // Factory method for pipeline creation with validation
    }

    Method Query(queryText As %String, Output response As RAG.Response) As %Status [ Abstract ]
    {
        // Abstract query method implemented by concrete pipeline classes
    }

    Method LoadDocuments(documents As %Collection.ListOfObj) As %Status
    {
        // Document loading with ObjectScript collection support
    }
}
```

### Bridge Integration Pattern
- Implement lazy Python environment initialization leveraging IRIS Embedded Python infrastructure
- Provide connection pooling for Python process management following %SYS.Python patterns
- Support graceful fallback when Python components are unavailable
- Enable configuration synchronization between ObjectScript and Python
- Inherit IRIS security context and user privileges automatically, following Embedded Python's context inheritance model

### Error Handling and Status Management
- Follow IRIS %Status patterns for all method returns
- Provide detailed error context in ObjectScript error logs
- Support IRIS debugging and profiling tools
- Integrate with IRIS monitoring and alerting systems

### Configuration Structure
```objectscript
Class RAG.Configuration Extends %RegisteredObject
{
    Property PipelineType As %String [ Required ];
    Property DatabaseConfig As RAG.DatabaseConfig;
    Property ModelConfig As RAG.ModelConfig;
    Property PythonEnvironment As %String;

    Method Validate() As %Status
    {
        // Configuration validation with specific error reporting
    }
}
```

## Dependencies

### ObjectScript Dependencies
- IRIS 2023.1+ with ObjectScript support
- %Collection classes for data structure handling
- %Status error handling framework
- %RegisteredObject class hierarchy

### Python Integration Dependencies
- Python RAG framework (iris_rag package)
- Python-ObjectScript bridge utilities
- Python environment management tools
- Data serialization libraries (JSON, pickle)

### IRIS Platform Dependencies
- IRIS Vector Search capabilities
- IRIS SQL and Global storage
- IRIS security and namespace management
- IRIS process and job management

### External Integration Points
- Python process lifecycle management
- Configuration synchronization mechanisms
- Error and logging integration
- Performance monitoring and profiling

## Clarifications

### Session 2025-01-28
- Q: What should happen when the Python RAG framework environment is corrupted or missing? → A: Fail all ObjectScript RAG operations with clear error
- Q: What should be the maximum memory limit per ObjectScript-to-Python bridge operation to prevent resource exhaustion? → A: 500 MB per operation
- Q: How should the system handle ObjectScript data type conversion failures when translating complex collections to Python? → A: Throw ObjectScript exception immediately
- Q: What should be the timeout limit for ObjectScript-to-Python bridge initialization before falling back to error handling? → A: 10 seconds
- Q: How should ObjectScript security contexts be mapped to Python execution environments for secure bridge operations? → A: Inherit IRIS user privileges directly (following Embedded Python patterns)

## Success Metrics

### API Parity Achievement
- Achieve 100% functional parity with Python and JavaScript APIs
- Support all pipeline types available in other language bindings
- Maintain consistent response formats and error handling patterns
- Enable seamless migration between language implementations

### Performance Characteristics
- ObjectScript bridge overhead under 50ms for standard queries
- Support enterprise-scale concurrent usage (50+ simultaneous queries)
- Optimize IRIS-native data access for zero-copy operations where possible
- Achieve sub-second response times for typical RAG workflows

### Enterprise Integration Success
- Seamless integration with existing IRIS applications and workflows
- Support for IRIS security, namespace, and deployment patterns
- Enable production deployment in enterprise IRIS environments
- Provide comprehensive documentation and developer resources

### Developer Experience Excellence
- Intuitive ObjectScript class hierarchy following IRIS conventions
- Comprehensive error messages and debugging support
- Easy configuration and setup for ObjectScript developers
- Rich documentation with ObjectScript-specific examples and patterns

## Testing Strategy

### ObjectScript Unit Testing
- Test all ObjectScript class methods with various input scenarios
- Validate ObjectScript-to-Python data type conversion accuracy
- Test error handling and %Status return value consistency
- Verify configuration management and validation logic

### Integration Testing with Python Framework
- Test ObjectScript-Python bridge functionality with real RAG pipelines
- Validate query processing workflows across language boundaries
- Test concurrent access patterns and resource management
- Verify configuration synchronization between ObjectScript and Python

### IRIS Platform Integration Testing
- Test integration with IRIS security and namespace management
- Validate performance with IRIS SQL and Global data access
- Test deployment scenarios across different IRIS configurations
- Verify compatibility with IRIS clustering and mirroring

### Enterprise Workflow Testing
- Test integration with existing IRIS application patterns
- Validate performance under enterprise-scale load conditions
- Test multi-tenant deployment scenarios and namespace isolation
- Verify production deployment and maintenance procedures