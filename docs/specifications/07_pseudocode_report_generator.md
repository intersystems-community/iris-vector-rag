# mem0-Supabase Validation Workflow: Report Generator Pseudocode

## 1. Module Overview

### 1.1 Purpose
Generates comprehensive, human-readable validation reports in multiple formats (console, JSON, HTML, CSV) for easy interpretation of validation results.

### 1.2 Dependencies
- ConfigurationManager
- TemplateEngine
- FileSystemManager
- ErrorHandler
- DataFormatter

### 1.3 Key Responsibilities
- Multi-format report generation
- Template-based report rendering
- Data visualization and summarization
- File output management
- Error reporting aggregation

## 2. Main Report Generator Class

```pseudocode
CLASS ReportGenerator:
    
    // TEST: Initialize report generator with valid configuration
    CONSTRUCTOR(config_manager: ConfigurationManager):
        PRE: config_manager is not null
        
        self.config_manager = config_manager
        self.template_engine = TemplateEngine(config_manager)
        self.file_system = FileSystemManager(config_manager)
        self.error_handler = ErrorHandler(config_manager)
        self.data_formatter = DataFormatter(config_manager)
        self.report_config = config_manager.get_report_config()
        
        POST: All components initialized successfully
    
    // TEST: Generate reports in all configured formats
    // TEST: Handle template rendering failures gracefully
    // TEST: Ensure all report paths are accessible and valid
    // TEST: Create reports with comprehensive validation data
    METHOD generate_reports(validation_result: ValidationResult, report_formats: Array<ReportFormat>, execution_id: String) -> ReportGenerationResult:
        PRE: validation_result is not null
        PRE: report_formats is not null
        PRE: report_formats.length > 0
        PRE: execution_id is not empty
        
        generation_result = ReportGenerationResult()
        generation_result.execution_id = execution_id
        generation_result.started_at = current_timestamp()
        generation_result.generated_report_paths = []
        generation_result.errors = []
        generation_result.success = true
        
        // Prepare report data structure
        report_data = self._prepare_report_data(validation_result)
        
        // Generate each requested format
        FOR format IN report_formats:
            TRY:
                format_result = self._generate_format_report(report_data, format, execution_id)
                
                IF format_result.success:
                    generation_result.generated_report_paths.extend(format_result.file_paths)
                    self._log_report_format_success(format, format_result.file_paths)
                ELSE:
                    generation_result.errors.extend(format_result.errors)
                    self._log_report_format_failure(format, format_result.errors)
                
            CATCH Exception as e:
                error = self._create_format_generation_error(format, e)
                generation_result.errors.append(error)
                generation_result.success = false
                self._log_report_format_exception(format, e)
        
        // Generate summary report if multiple formats
        IF len(report_formats) > 1:
            summary_result = self._generate_summary_report(report_data, execution_id)
            IF summary_result.success:
                generation_result.generated_report_paths.extend(summary_result.file_paths)
        
        generation_result.completed_at = current_timestamp()
        generation_result.generation_time_ms = generation_result.completed_at - generation_result.started_at
        
        // Validate all generated files exist
        self._validate_generated_files(generation_result.generated_report_paths)
        
        self._log_report_generation_complete(generation_result)
        
        RETURN generation_result
        
        POST: generation_result is not null
        POST: generation_result.execution_id == execution_id
    
    // TEST: Prepare comprehensive report data structure
    // TEST: Include all validation metrics and error details
    // TEST: Calculate summary statistics correctly
    METHOD _prepare_report_data(validation_result: ValidationResult) -> ReportData:
        PRE: validation_result is not null
        
        report_data = ReportData()
        report_data.execution_summary = self._create_execution_summary(validation_result)
        report_data.validation_metrics = self._create_validation_metrics(validation_result)
        report_data.error_analysis = self._create_error_analysis(validation_result)
        report_data.consistency_analysis = self._create_consistency_analysis(validation_result)
        report_data.performance_metrics = self._create_performance_metrics(validation_result)
        report_data.recommendations = self._generate_recommendations(validation_result)
        report_data.generated_at = current_timestamp()
        
        RETURN report_data
        
        POST: report_data is not null
        POST: report_data.generated_at is not null
    
    // TEST: Generate report in specific format
    // TEST: Handle format-specific rendering requirements
    // TEST: Ensure proper file naming and directory structure
    METHOD _generate_format_report(report_data: ReportData, format: ReportFormat, execution_id: String) -> FormatGenerationResult:
        PRE: report_data is not null
        PRE: format in [CONSOLE, JSON, HTML, CSV]
        PRE: execution_id is not empty
        
        format_result = FormatGenerationResult()
        format_result.format = format
        format_result.success = false
        format_result.file_paths = []
        format_result.errors = []
        
        SWITCH format:
            CASE CONSOLE:
                format_result = self._generate_console_report(report_data, execution_id)
            CASE JSON:
                format_result = self._generate_json_report(report_data, execution_id)
            CASE HTML:
                format_result = self._generate_html_report(report_data, execution_id)
            CASE CSV:
                format_result = self._generate_csv_report(report_data, execution_id)
            DEFAULT:
                error = ValidationError()
                error.error_type = UNSUPPORTED_REPORT_FORMAT
                error.severity = CRITICAL
                error.message = "Unsupported report format: " + format
                format_result.errors.append(error)
        
        RETURN format_result
        
        POST: format_result is not null
        POST: format_result.format == format
    
    // TEST: Generate console report with clear formatting
    // TEST: Include color coding for success/failure status
    // TEST: Provide readable summary and details
    METHOD _generate_console_report(report_data: ReportData, execution_id: String) -> FormatGenerationResult:
        PRE: report_data is not null
        PRE: execution_id is not empty
        
        result = FormatGenerationResult()
        result.format = CONSOLE
        result.success = true
        result.file_paths = []
        result.errors = []
        
        TRY:
            console_output = self._render_console_template(report_data)
            
            // Output to console
            console.print(console_output)
            
            // Save to file if configured
            IF self.report_config.save_console_output:
                console_file_path = self._get_report_file_path(execution_id, "console", "txt")
                self.file_system.write_text_file(console_file_path, console_output)
                result.file_paths.append(console_file_path)
            
        CATCH Exception as e:
            error = ValidationError()
            error.error_type = CONSOLE_REPORT_ERROR
            error.severity = CRITICAL
            error.message = "Failed to generate console report: " + e.message
            result.errors.append(error)
            result.success = false
        
        RETURN result
        
        POST: result is not null
        POST: result.format == CONSOLE
    
    // TEST: Generate structured JSON report
    // TEST: Ensure valid JSON format and schema compliance
    // TEST: Include all validation data with proper nesting
    METHOD _generate_json_report(report_data: ReportData, execution_id: String) -> FormatGenerationResult:
        PRE: report_data is not null
        PRE: execution_id is not empty
        
        result = FormatGenerationResult()
        result.format = JSON
        result.success = true
        result.file_paths = []
        result.errors = []
        
        TRY:
            json_data = self.data_formatter.convert_to_json_structure(report_data)
            json_content = self.data_formatter.serialize_json(json_data, pretty_print=true)
            
            json_file_path = self._get_report_file_path(execution_id, "validation_report", "json")
            self.file_system.write_text_file(json_file_path, json_content)
            result.file_paths.append(json_file_path)
            
            // Generate schema validation
            schema_validation = self.data_formatter.validate_json_schema(json_data)
            IF NOT schema_validation.is_valid:
                error = ValidationError()
                error.error_type = JSON_SCHEMA_VALIDATION_ERROR
                error.severity = WARNING
                error.message = "Generated JSON does not match expected schema"
                error.context = {"schema_errors": schema_validation.errors}
                result.errors.append(error)
            
        CATCH Exception as e:
            error = ValidationError()
            error.error_type = JSON_REPORT_ERROR
            error.severity = CRITICAL
            error.message = "Failed to generate JSON report: " + e.message
            result.errors.append(error)
            result.success = false
        
        RETURN result
        
        POST: result is not null
        POST: result.format == JSON
    
    // TEST: Generate comprehensive HTML report with visualizations
    // TEST: Include interactive elements and charts
    // TEST: Ensure proper CSS styling and responsive design
    METHOD _generate_html_report(report_data: ReportData, execution_id: String) -> FormatGenerationResult:
        PRE: report_data is not null
        PRE: execution_id is not empty
        
        result = FormatGenerationResult()
        result.format = HTML
        result.success = true
        result.file_paths = []
        result.errors = []
        
        TRY:
            // Render main HTML report
            html_content = self.template_engine.render_template(
                template_name="validation_report.html",
                data=report_data
            )
            
            html_file_path = self._get_report_file_path(execution_id, "validation_report", "html")
            self.file_system.write_text_file(html_file_path, html_content)
            result.file_paths.append(html_file_path)
            
            // Generate supporting assets
            assets_result = self._generate_html_assets(report_data, execution_id)
            result.file_paths.extend(assets_result.file_paths)
            result.errors.extend(assets_result.errors)
            
            // Generate charts and visualizations
            IF self.report_config.include_charts:
                charts_result = self._generate_charts(report_data, execution_id)
                result.file_paths.extend(charts_result.file_paths)
                result.errors.extend(charts_result.errors)
            
        CATCH TemplateRenderingException as e:
            error = ValidationError()
            error.error_type = HTML_TEMPLATE_ERROR
            error.severity = CRITICAL
            error.message = "Failed to render HTML template: " + e.message
            result.errors.append(error)
            result.success = false
            
        CATCH Exception as e:
            error = ValidationError()
            error.error_type = HTML_REPORT_ERROR
            error.severity = CRITICAL
            error.message = "Failed to generate HTML report: " + e.message
            result.errors.append(error)
            result.success = false
        
        RETURN result
        
        POST: result is not null
        POST: result.format == HTML
    
    // TEST: Generate CSV reports for tabular data analysis
    // TEST: Create separate CSV files for different data categories
    // TEST: Ensure proper CSV formatting and encoding
    METHOD _generate_csv_report(report_data: ReportData, execution_id: String) -> FormatGenerationResult:
        PRE: report_data is not null
        PRE: execution_id is not empty
        
        result = FormatGenerationResult()
        result.format = CSV
        result.success = true
        result.file_paths = []
        result.errors = []
        
        TRY:
            // Generate validation summary CSV
            summary_csv = self._create_validation_summary_csv(report_data)
            summary_path = self._get_report_file_path(execution_id, "validation_summary", "csv")
            self.file_system.write_text_file(summary_path, summary_csv)
            result.file_paths.append(summary_path)
            
            // Generate memory details CSV
            memory_csv = self._create_memory_details_csv(report_data)
            memory_path = self._get_report_file_path(execution_id, "memory_details", "csv")
            self.file_system.write_text_file(memory_path, memory_csv)
            result.file_paths.append(memory_path)
            
            // Generate error analysis CSV
            IF len(report_data.error_analysis.errors) > 0:
                error_csv = self._create_error_analysis_csv(report_data)
                error_path = self._get_report_file_path(execution_id, "error_analysis", "csv")
                self.file_system.write_text_file(error_path, error_csv)
                result.file_paths.append(error_path)
            
            // Generate consistency analysis CSV
            IF len(report_data.consistency_analysis.consistency_records) > 0:
                consistency_csv = self._create_consistency_analysis_csv(report_data)
                consistency_path = self._get_report_file_path(execution_id, "consistency_analysis", "csv")
                self.file_system.write_text_file(consistency_path, consistency_csv)
                result.file_paths.append(consistency_path)
            
        CATCH Exception as e:
            error = ValidationError()
            error.error_type = CSV_REPORT_ERROR
            error.severity = CRITICAL
            error.message = "Failed to generate CSV report: " + e.message
            result.errors.append(error)
            result.success = false
        
        RETURN result
        
        POST: result is not null
        POST: result.format == CSV
    
    // TEST: Create execution summary with key metrics
    METHOD _create_execution_summary(validation_result: ValidationResult) -> ExecutionSummary:
        PRE: validation_result is not null
        
        summary = ExecutionSummary()
        summary.execution_id = validation_result.execution_id
        summary.overall_status = validation_result.status
        summary.started_at = validation_result.started_at
        summary.completed_at = validation_result.completed_at
        summary.total_execution_time_ms = validation_result.completed_at - validation_result.started_at
        summary.task_name = validation_result.task_id  // Resolve task name from ID
        summary.validation_phases = self._extract_validation_phases(validation_result)
        
        RETURN summary
        
        POST: summary is not null
        POST: summary.execution_id == validation_result.execution_id
    
    // TEST: Create comprehensive validation metrics
    METHOD _create_validation_metrics(validation_result: ValidationResult) -> ValidationMetrics:
        PRE: validation_result is not null
        
        metrics = ValidationMetrics()
        metrics.memories_created = validation_result.metrics.memories_created
        metrics.memories_validated_mem0 = validation_result.metrics.memories_validated_mem0
        metrics.memories_validated_supabase = validation_result.metrics.memories_validated_supabase
        metrics.consistency_rate = self._calculate_consistency_rate(validation_result)
        metrics.success_rate = self._calculate_success_rate(validation_result)
        metrics.performance_metrics = validation_result.metrics
        
        RETURN metrics
        
        POST: metrics is not null
        POST: metrics.consistency_rate >= 0.0 AND metrics.consistency_rate <= 1.0
    
    // TEST: Create detailed error analysis
    METHOD _create_error_analysis(validation_result: ValidationResult) -> ErrorAnalysis:
        PRE: validation_result is not null
        
        analysis = ErrorAnalysis()
        analysis.total_error_count = len(validation_result.errors)
        analysis.errors_by_severity = self._group_errors_by_severity(validation_result.errors)
        analysis.errors_by_type = self._group_errors_by_type(validation_result.errors)
        analysis.errors_by_component = self._group_errors_by_component(validation_result.errors)
        analysis.error_timeline = self._create_error_timeline(validation_result.errors)
        analysis.top_error_patterns = self._identify_error_patterns(validation_result.errors)
        
        RETURN analysis
        
        POST: analysis is not null
        POST: analysis.total_error_count >= 0
    
    // TEST: Render console output with proper formatting
    METHOD _render_console_template(report_data: ReportData) -> String:
        PRE: report_data is not null
        
        output = StringBuilder()
        
        // Header
        output.append("=" * 80)
        output.append("\n  mem0-Supabase Validation Report")
        output.append("\n  Execution ID: " + report_data.execution_summary.execution_id)
        output.append("\n  Generated: " + format_timestamp(report_data.generated_at))
        output.append("\n" + "=" * 80)
        
        // Overall Status
        status_color = self._get_status_color(report_data.execution_summary.overall_status)
        output.append("\n\nOVERALL STATUS: " + colorize(report_data.execution_summary.overall_status, status_color))
        
        // Summary Metrics
        output.append("\n\nVALIDATION SUMMARY:")
        output.append("\n  Memories Created: " + report_data.validation_metrics.memories_created)
        output.append("\n  mem0 Validated: " + report_data.validation_metrics.memories_validated_mem0)
        output.append("\n  Supabase Validated: " + report_data.validation_metrics.memories_validated_supabase)
        output.append("\n  Consistency Rate: " + format_percentage(report_data.validation_metrics.consistency_rate))
        output.append("\n  Success Rate: " + format_percentage(report_data.validation_metrics.success_rate))
        
        // Performance Metrics
        output.append("\n\nPERFORMANCE METRICS:")
        output.append("\n  Total Execution Time: " + format_duration(report_data.execution_summary.total_execution_time_ms))
        output.append("\n  Task Execution: " + format_duration(report_data.performance_metrics.task_execution_time_ms))
        output.append("\n  mem0 Validation: " + format_duration(report_data.performance_metrics.mem0_validation_time_ms))
        output.append("\n  Supabase Validation: " + format_duration(report_data.performance_metrics.supabase_validation_time_ms))
        
        // Error Summary
        IF report_data.error_analysis.total_error_count > 0:
            output.append("\n\nERROR SUMMARY:")
            output.append("\n  Total Errors: " + report_data.error_analysis.total_error_count)
            FOR severity, count IN report_data.error_analysis.errors_by_severity:
                severity_color = self._get_severity_color(severity)
                output.append("\n  " + colorize(severity, severity_color) + ": " + count)
        
        // Recommendations
        IF len(report_data.recommendations) > 0:
            output.append("\n\nRECOMMENDATIONS:")
            FOR recommendation IN report_data.recommendations:
                output.append("\n  • " + recommendation)
        
        output.append("\n\n" + "=" * 80)
        
        RETURN output.toString()
        
        POST: result is not empty
    
    // TEST: Generate file path with proper naming convention
    METHOD _get_report_file_path(execution_id: String, report_name: String, extension: String) -> String:
        PRE: execution_id is not empty
        PRE: report_name is not empty
        PRE: extension is not empty
        
        timestamp = format_timestamp_for_filename(current_timestamp())
        filename = report_name + "_" + execution_id + "_" + timestamp + "." + extension
        
        report_directory = self.report_config.output_directory
        file_path = join_path(report_directory, filename)
        
        RETURN file_path
        
        POST: file_path is not empty
        POST: file_path contains execution_id
    
    // TEST: Validate all generated files exist and are accessible
    METHOD _validate_generated_files(file_paths: Array<String>) -> void:
        FOR file_path IN file_paths:
            IF NOT self.file_system.file_exists(file_path):
                self._log_missing_report_file(file_path)
            ELSE:
                file_size = self.file_system.get_file_size(file_path)
                IF file_size == 0:
                    self._log_empty_report_file(file_path)
    
    // Utility methods for data processing
    METHOD _calculate_consistency_rate(validation_result: ValidationResult) -> Float:
        total_validated = validation_result.metrics.memories_validated_supabase
        IF total_validated == 0:
            RETURN 0.0
        
        // Extract consistency count from supabase validation results
        consistent_count = 0  // This would be extracted from detailed results
        RETURN Float(consistent_count) / Float(total_validated)
    
    METHOD _calculate_success_rate(validation_result: ValidationResult) -> Float:
        total_memories = validation_result.metrics.memories_created
        IF total_memories == 0:
            RETURN 0.0
        
        successful_validations = min(
            validation_result.metrics.memories_validated_mem0,
            validation_result.metrics.memories_validated_supabase
        )
        RETURN Float(successful_validations) / Float(total_memories)
    
    METHOD _get_status_color(status: ValidationStatus) -> Color:
        SWITCH status:
            CASE SUCCESS: RETURN GREEN
            CASE PARTIAL_SUCCESS: RETURN YELLOW
            CASE FAILURE: RETURN RED
            DEFAULT: RETURN WHITE
    
    METHOD _get_severity_color(severity: ErrorSeverity) -> Color:
        SWITCH severity:
            CASE CRITICAL: RETURN RED
            CASE WARNING: RETURN YELLOW
            CASE INFO: RETURN BLUE
            DEFAULT: RETURN WHITE
    
    // Error handling methods
    METHOD _create_format_generation_error(format: ReportFormat, exception: Exception) -> ValidationError:
        error = ValidationError()
        error.error_type = REPORT_FORMAT_ERROR
        error.severity = CRITICAL
        error.message = "Failed to generate " + format + " report: " + exception.message
        error.context = {"format": format, "error_details": exception.message}
        error.occurred_at = current_timestamp()
        RETURN error
    
    // Logging methods for audit trail
    METHOD _log_report_format_success(format: ReportFormat, file_paths: Array<String>) -> void:
        log_entry = create_log_entry("REPORT_FORMAT_SUCCESS")
        log_entry.context = {"format": format, "file_count": len(file_paths)}
        self.error_handler.log_audit_event(log_entry)
    
    METHOD _log_report_generation_complete(generation_result: ReportGenerationResult) -> void:
        log_entry = create_log_entry("REPORT_GENERATION_COMPLETE")
        log_entry.context = {
            "execution_id": generation_result.execution_id,
            "success": generation_result.success,
            "report_count": len(generation_result.generated_report_paths),
            "generation_time_ms": generation_result.generation_time_ms
        }
        self.error_handler.log_audit_event(log_entry)

END CLASS
```

## 3. Supporting Data Structures

```pseudocode
STRUCTURE ReportGenerationResult:
    execution_id: String
    success: Boolean
    generated_report_paths: Array<String>
    errors: Array<ValidationError>
    started_at: Timestamp
    completed_at: Timestamp
    generation_time_ms: Integer

STRUCTURE FormatGenerationResult:
    format: ReportFormat
    success: Boolean
    file_paths: Array<String>
    errors: Array<ValidationError>

STRUCTURE ReportData:
    execution_summary: ExecutionSummary
    validation_metrics: ValidationMetrics
    error_analysis: ErrorAnalysis
    consistency_analysis: ConsistencyAnalysis
    performance_metrics: PerformanceMetrics
    recommendations: Array<String>
    generated_at: Timestamp

STRUCTURE ExecutionSummary:
    execution_id: String
    overall_status: ValidationStatus
    started_at: Timestamp
    completed_at: Timestamp
    total_execution_time_ms: Integer
    task_name: String
    validation_phases: Array<ValidationPhase>

STRUCTURE ErrorAnalysis:
    total_error_count: Integer
    errors_by_severity: Map<ErrorSeverity, Integer>
    errors_by_type: Map<ErrorType, Integer>
    errors_by_component: Map<String, Integer>
    error_timeline: Array<ErrorTimelineEntry>
    top_error_patterns: Array<ErrorPattern>

ENUM ReportFormat:
    CONSOLE, JSON, HTML, CSV

ENUM Color:
    RED, GREEN, YELLOW, BLUE, WHITE
```

## 4. Template System Integration

### 4.1 HTML Template Structure
- Main report template with navigation
- Section templates for modular content
- Chart and visualization templates
- Error detail templates
- Responsive CSS framework integration

### 4.2 Report Sections
- Executive summary with key metrics
- Detailed validation results
- Error analysis with categorization
- Performance benchmarks
- Consistency analysis
- Recommendations and action items

### 4.3 Interactive Features
- Expandable error details
- Sortable data tables
- Interactive charts and graphs
- Search and filter capabilities

## 5. Error Handling Strategy

### 5.1 Template Rendering Failures
- Graceful fallback to basic templates
- Error context preservation
- Alternative format generation
- Template validation before rendering

### 5.2 File System Operations
- Directory creation for report output
- File permission handling
- Disk space validation
- Atomic file operations where possible

### 5.3 Data Formatting Issues
- Input data validation before processing
- Format-specific error handling
- Encoding and character set management
- Large data set optimization

## 6. Performance Considerations

### 6.1 Report Generation Optimization
- Lazy loading for large datasets
- Template caching for repeated generations
- Parallel processing for multiple formats
- Memory-efficient data streaming

### 6.2 File Operations
- Efficient file I/O operations
- Compression for large reports
- Progress tracking for long operations
- Resource cleanup after generation

### 6.3 Template Processing
- Template compilation and caching
- Efficient data binding
- Minimal memory footprint
- Optimized rendering pipelines