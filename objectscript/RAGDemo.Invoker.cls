/// RAGDemo.Invoker - ObjectScript wrapper class for invoking RAG pipelines
/// 
/// This class provides ObjectScript methods that can be called from SQL
/// to invoke Python RAG pipelines through Embedded Python integration.
Class RAGDemo.Invoker Extends %RegisteredObject
{

/// Check if the Invoker class exists and is properly configured
ClassMethod InvokerExists() As %Integer [ SqlProc ]
{
    Quit 1
}

/// Invoke Basic RAG pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeBasicRAG(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_basic_rag"(query, config)
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

/// Invoke ColBERT pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeColBERT(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_colbert"(query, config)
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

/// Invoke GraphRAG pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeGraphRAG(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_graphrag"(query, config)
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

/// Invoke HyDE pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeHyDE(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_hyde"(query, config)
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

/// Invoke CRAG pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeCRAG(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_crag"(query, config)
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

/// Invoke NodeRAG pipeline
/// @param query The search query
/// @param config JSON configuration string
/// @return JSON string with results
ClassMethod InvokeNodeRAG(query As %String, config As %String = "{}") As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."invoke_noderag"(query, config)
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

/// Get list of available RAG pipelines
/// @return JSON string with available pipeline information
ClassMethod GetAvailablePipelines() As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
        Set bridge = ##class(%SYS.Python).Import("objectscript.python_bridge")
        Set result = bridge."get_available_pipelines"()
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

/// Validate RAG pipeline results
/// @param results JSON string with RAG results to validate
/// @return JSON string with validation results
ClassMethod ValidateResults(results As %String) As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
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

/// Perform health check of the Python bridge
/// @return JSON string with health status
ClassMethod HealthCheck() As %String [ SqlProc ]
{
    Try {
        // Use Embedded Python to call the Python bridge
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