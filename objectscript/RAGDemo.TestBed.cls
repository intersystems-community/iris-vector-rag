/// RAGDemo.TestBed - ObjectScript class for comprehensive RAG pipeline testing
/// 
/// This class provides ObjectScript methods for running comprehensive tests
/// and benchmarks of all RAG pipelines through Embedded Python integration.
Class RAGDemo.TestBed Extends %RegisteredObject
{

/// Check if the TestBed class exists and is properly configured
ClassMethod TestBedExists() As %Integer [ SqlProc ]
{
    Quit 1
}

/// Run all RAG pipeline tests
/// @return JSON string with test results
ClassMethod RunAllRAGTests() As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to run comprehensive tests
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        
        // Get available pipelines
        Set pipelinesJson = bridge."get_available_pipelines"()
        Set pipelines = ##class(%DynamicObject).%FromJSON(pipelinesJson)
        
        // Initialize test results
        Set testResults = {
            "test_summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "start_time": ($ZDateTime($Horolog, 3)),
                "end_time": "",
                "duration_seconds": 0
            },
            "individual_tests": []
        }
        
        Set startTime = $Piece($Horolog, ",", 2)
        
        // Test each pipeline
        Set pipelineIterator = pipelines.%GetIterator()
        While pipelineIterator.%GetNext(.key, .pipelineInfo) {
            Set testResult = ..TestSinglePipeline(key, pipelineInfo)
            Do testResults."individual_tests".%Push(testResult)
            Set testResults."test_summary"."total_tests" = testResults."test_summary"."total_tests" + 1
            If testResult."success" {
                Set testResults."test_summary"."passed_tests" = testResults."test_summary"."passed_tests" + 1
            } Else {
                Set testResults."test_summary"."failed_tests" = testResults."test_summary"."failed_tests" + 1
            }
        }
        
        Set endTime = $Piece($Horolog, ",", 2)
        Set testResults."test_summary"."end_time" = $ZDateTime($Horolog, 3)
        Set testResults."test_summary"."duration_seconds" = endTime - startTime
        
        Quit testResults.%ToJSON()
        
    } Catch ex {
        // Return error in standardized format
        Set error = {
            "success": (false),
            "result": null,
            "error": (ex.DisplayString()),
            "timestamp": ($ZDateTime($Horolog, 3))
        }
        Quit error.%ToJSON()
    }
}

/// Test a single RAG pipeline
/// @param pipelineName Name of the pipeline to test
/// @param pipelineInfo Pipeline information object
/// @return Test result object
ClassMethod TestSinglePipeline(pipelineName As %String, pipelineInfo As %DynamicObject) As %DynamicObject [ Private ]
{
    Set testResult = {
        "pipeline_name": (pipelineName),
        "pipeline_class": (pipelineInfo."class"),
        "success": (false),
        "error": "",
        "test_query": "What are the effects of COVID-19?",
        "response_time_ms": 0,
        "result_validation": {}
    }
    
    Try {
        Set startTime = $ZH
        
        // Create test configuration
        Set config = {
            "embedding_func": null,
            "llm_func": null
        }
        
        // Call the appropriate pipeline method
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set methodName = "invoke_" _ $Replace(pipelineName, "_", "")
        If pipelineName = "basic_rag" Set methodName = "invoke_basic_rag"
        
        Set resultJson = $Method(bridge, methodName, testResult."test_query", config.%ToJSON())
        Set endTime = $ZH
        
        Set testResult."response_time_ms" = (endTime - startTime) * 1000
        
        // Parse and validate result
        Set result = ##class(%DynamicObject).%FromJSON(resultJson)
        If result."success" {
            Set testResult."success" = true
            
            // Validate the result structure
            Set validationJson = bridge."validate_results"(result."result".%ToJSON())
            Set validation = ##class(%DynamicObject).%FromJSON(validationJson)
            Set testResult."result_validation" = validation."result"
        } Else {
            Set testResult."error" = result."error"
        }
        
    } Catch ex {
        Set testResult."error" = ex.DisplayString()
        Set testResult."response_time_ms" = ($ZH - startTime) * 1000
    }
    
    Quit testResult
}

/// Run benchmarks for all RAG pipelines
/// @return JSON string with benchmark results
ClassMethod BenchmarkAllPipelines() As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to run benchmarks
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        
        // Get available pipelines
        Set pipelinesJson = bridge."get_available_pipelines"()
        Set pipelines = ##class(%DynamicObject).%FromJSON(pipelinesJson)
        
        // Extract pipeline names
        Set pipelineNames = []
        Set pipelineIterator = pipelines.%GetIterator()
        While pipelineIterator.%GetNext(.key, .pipelineInfo) {
            Do pipelineNames.%Push(key)
        }
        
        // Run benchmarks
        Set result = bridge."run_benchmarks"(pipelineNames.%ToJSON())
        Quit result
        
    } Catch ex {
        // Return error in standardized format
        Set error = {
            "success": (false),
            "result": null,
            "error": (ex.DisplayString()),
            "timestamp": ($ZDateTime($Horolog, 3))
        }
        Quit error.%ToJSON()
    }
}

/// Validate pipeline results
/// @param results JSON string with pipeline results to validate
/// @return JSON string with validation results
ClassMethod ValidatePipelineResults(results As %String) As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to validate results
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."validate_results"(results)
        Quit result
        
    } Catch ex {
        // Return error in standardized format
        Set error = {
            "success": (false),
            "result": null,
            "error": (ex.DisplayString()),
            "timestamp": ($ZDateTime($Horolog, 3))
        }
        Quit error.%ToJSON()
    }
}

/// Run performance tests for a specific pipeline
/// @param pipelineName Name of the pipeline to test
/// @param iterations Number of test iterations
/// @return JSON string with performance results
ClassMethod RunPerformanceTest(pipelineName As %String, iterations As %Integer = 10) As %String [ SqlProc ]
{
    Try {
        Set performanceResults = {
            "pipeline_name": (pipelineName),
            "iterations": (iterations),
            "test_query": "What are the effects of COVID-19?",
            "results": {
                "total_time_ms": 0,
                "average_time_ms": 0,
                "min_time_ms": 999999,
                "max_time_ms": 0,
                "successful_runs": 0,
                "failed_runs": 0
            },
            "individual_runs": []
        }
        
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set config = {}.%ToJSON()
        
        // Determine method name
        Set methodName = "invoke_" _ $Replace(pipelineName, "_", "")
        If pipelineName = "basic_rag" Set methodName = "invoke_basic_rag"
        
        For i=1:1:iterations {
            Set runResult = {
                "run_number": (i),
                "success": (false),
                "time_ms": 0,
                "error": ""
            }
            
            Try {
                Set startTime = $ZH
                Set resultJson = $Method(bridge, methodName, performanceResults."test_query", config)
                Set endTime = $ZH
                
                Set runTime = (endTime - startTime) * 1000
                Set runResult."time_ms" = runTime
                
                Set result = ##class(%DynamicObject).%FromJSON(resultJson)
                If result."success" {
                    Set runResult."success" = true
                    Set performanceResults."results"."successful_runs" = performanceResults."results"."successful_runs" + 1
                    
                    // Update timing statistics
                    Set performanceResults."results"."total_time_ms" = performanceResults."results"."total_time_ms" + runTime
                    If runTime < performanceResults."results"."min_time_ms" {
                        Set performanceResults."results"."min_time_ms" = runTime
                    }
                    If runTime > performanceResults."results"."max_time_ms" {
                        Set performanceResults."results"."max_time_ms" = runTime
                    }
                } Else {
                    Set runResult."error" = result."error"
                    Set performanceResults."results"."failed_runs" = performanceResults."results"."failed_runs" + 1
                }
                
            } Catch ex {
                Set runResult."error" = ex.DisplayString()
                Set runResult."time_ms" = ($ZH - startTime) * 1000
                Set performanceResults."results"."failed_runs" = performanceResults."results"."failed_runs" + 1
            }
            
            Do performanceResults."individual_runs".%Push(runResult)
        }
        
        // Calculate average time
        If performanceResults."results"."successful_runs" > 0 {
            Set performanceResults."results"."average_time_ms" = performanceResults."results"."total_time_ms" / performanceResults."results"."successful_runs"
        }
        
        Quit performanceResults.%ToJSON()
        
    } Catch ex {
        // Return error in standardized format
        Set error = {
            "success": (false),
            "result": null,
            "error": (ex.DisplayString()),
            "timestamp": ($ZDateTime($Horolog, 3))
        }
        Quit error.%ToJSON()
    }
}

/// Get system health status
/// @return JSON string with health status
ClassMethod GetSystemHealth() As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to check health
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."health_check"()
        Quit result
        
    } Catch ex {
        // Return error in standardized format
        Set error = {
            "success": (false),
            "result": null,
            "error": (ex.DisplayString()),
            "timestamp": ($ZDateTime($Horolog, 3))
        }
        Quit error.%ToJSON()
    }
}

}