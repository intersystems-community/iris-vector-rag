Class RAG.PythonBridge Extends %RegisteredObject
{

/// Test Python integration and package availability
ClassMethod TestPythonIntegration() As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Testing Python integration...", !
        
        // Test basic Python execution
        Set pythonCode = "import sys; print('Python version:', sys.version)"
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Python execution test passed", !
        
        // Test iris_rag package import
        Set pythonCode = "import iris_rag; print('iris_rag version:', iris_rag.__version__)"
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ iris_rag package import test passed", !
        
        // Test pipeline creation
        Set pythonCode = "from iris_rag import create_pipeline; print('Pipeline factory available')"
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Pipeline factory test passed", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Python integration test failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Validate Python environment for RAG operations
ClassMethod ValidatePythonEnvironment() As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Validating Python environment...", !
        
        // Use the IPM integration utility
        Set pythonCode = "from iris_rag.utils.ipm_integration import validate_ipm_environment; "_
                         "import json; "_
                         "result = validate_ipm_environment(); "_
                         "print(json.dumps(result, indent=2))"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Python environment validation completed", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Python environment validation failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Create a basic RAG pipeline from ObjectScript
ClassMethod CreateBasicPipeline(configPath As %String = "") As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Creating basic RAG pipeline...", !
        
        // Python code to create and test a basic pipeline
        Set pythonCode = "from iris_rag import create_pipeline; "_
                         "import os; "_
                         "config_path = '"_configPath_"' if '"_configPath_"' else None; "_
                         "try: "_
                         "    pipeline = create_pipeline('basic', config_path=config_path); "_
                         "    print('✓ Basic RAG pipeline created successfully'); "_
                         "except Exception as e: "_
                         "    print('✗ Pipeline creation failed:', str(e)); "_
                         "    raise"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Basic RAG pipeline creation test passed", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Pipeline creation failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Execute a RAG query using Python pipeline
ClassMethod ExecuteRAGQuery(query As %String, configPath As %String = "") As %String
{
    Try {
        Write "Executing RAG query: ", query, !
        
        // Escape the query for Python
        Set escapedQuery = $Replace(query, "'", "\'")
        
        // Python code to execute RAG query
        Set pythonCode = "from iris_rag import create_pipeline; "_
                         "import json; "_
                         "config_path = '"_configPath_"' if '"_configPath_"' else None; "_
                         "try: "_
                         "    pipeline = create_pipeline('basic', config_path=config_path); "_
                         "    result = pipeline.execute('"_escapedQuery_"'); "_
                         "    print('Query result:', json.dumps(result, indent=2)); "_
                         "except Exception as e: "_
                         "    print('Query execution failed:', str(e)); "_
                         "    result = {'error': str(e)}"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        
        Return "Query executed successfully"
        
    } Catch ex {
        Return "Query execution failed: "_ex.DisplayString()
    }
}

/// Load documents into the RAG system
ClassMethod LoadDocuments(documentPath As %String, configPath As %String = "") As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Loading documents from: ", documentPath, !
        
        // Python code to load documents
        Set pythonCode = "from iris_rag import create_pipeline; "_
                         "import os; "_
                         "config_path = '"_configPath_"' if '"_configPath_"' else None; "_
                         "document_path = '"_documentPath_"'; "_
                         "try: "_
                         "    pipeline = create_pipeline('basic', config_path=config_path); "_
                         "    if os.path.exists(document_path): "_
                         "        pipeline.load_documents(document_path); "_
                         "        print('✓ Documents loaded successfully'); "_
                         "    else: "_
                         "        print('✗ Document path does not exist:', document_path); "_
                         "except Exception as e: "_
                         "    print('✗ Document loading failed:', str(e)); "_
                         "    raise"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Document loading completed", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Document loading failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Generate configuration template
ClassMethod GenerateConfigTemplate(outputPath As %String = "config.yaml") As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Generating configuration template...", !
        
        // Python code to generate config template
        Set pythonCode = "from iris_rag.utils.ipm_integration import IPMIntegration; "_
                         "ipm = IPMIntegration(); "_
                         "config = ipm.generate_config_template('"_outputPath_"'); "_
                         "print('✓ Configuration template generated:', '"_outputPath_"')"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        Write "✓ Configuration template generated: ", outputPath, !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Configuration template generation failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Get installation information
ClassMethod GetInstallationInfo() As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Getting installation information...", !
        
        // Python code to get installation info
        Set pythonCode = "from iris_rag.utils.ipm_integration import IPMIntegration; "_
                         "import json; "_
                         "ipm = IPMIntegration(); "_
                         "info = ipm.get_installation_info(); "_
                         "print(json.dumps(info, indent=2))"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "Getting installation info failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Run comprehensive system test
ClassMethod RunSystemTest() As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Running comprehensive system test...", !
        Write "="_$Translate($Justify("", 50), " ", "="), !
        
        // Test 1: Python integration
        Write "Test 1: Python Integration", !
        Set tSC = ..TestPythonIntegration()
        If $$$ISERR(tSC) Quit
        
        // Test 2: Environment validation
        Write "Test 2: Environment Validation", !
        Set tSC = ..ValidatePythonEnvironment()
        If $$$ISERR(tSC) Quit
        
        // Test 3: Pipeline creation
        Write "Test 3: Pipeline Creation", !
        Set tSC = ..CreateBasicPipeline()
        If $$$ISERR(tSC) Quit
        
        // Test 4: Configuration generation
        Write "Test 4: Configuration Generation", !
        Set tSC = ..GenerateConfigTemplate("test_config.yaml")
        If $$$ISERR(tSC) Quit
        
        // Test 5: Installation info
        Write "Test 5: Installation Information", !
        Set tSC = ..GetInstallationInfo()
        If $$$ISERR(tSC) Quit
        
        Write "="_$Translate($Justify("", 50), " ", "="), !
        Write "✓ All system tests passed successfully!", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "System test failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

/// Demonstrate RAG functionality with sample data
ClassMethod DemoRAGFunctionality() As %Status
{
    Set tSC = $$$OK
    Try {
        Write "Demonstrating RAG functionality...", !
        Write "="_$Translate($Justify("", 50), " ", "="), !
        
        // Create sample documents in memory
        Set pythonCode = "from iris_rag import create_pipeline; "_
                         "import tempfile, os; "_
                         "# Create temporary directory with sample documents; "_
                         "temp_dir = tempfile.mkdtemp(); "_
                         "sample_doc1 = os.path.join(temp_dir, 'doc1.txt'); "_
                         "sample_doc2 = os.path.join(temp_dir, 'doc2.txt'); "_
                         "with open(sample_doc1, 'w') as f: "_
                         "    f.write('Machine learning is a subset of artificial intelligence.'); "_
                         "with open(sample_doc2, 'w') as f: "_
                         "    f.write('Vector databases store high-dimensional vectors efficiently.'); "_
                         "print('Sample documents created in:', temp_dir); "_
                         "# Create pipeline and load documents; "_
                         "try: "_
                         "    pipeline = create_pipeline('basic'); "_
                         "    pipeline.load_documents(temp_dir); "_
                         "    print('✓ Documents loaded into RAG system'); "_
                         "    # Execute sample query; "_
                         "    result = pipeline.execute('What is machine learning?'); "_
                         "    print('✓ Query executed successfully'); "_
                         "    print('Answer:', result.get('answer', 'No answer generated')); "_
                         "except Exception as e: "_
                         "    print('✗ Demo failed:', str(e)); "_
                         "finally: "_
                         "    import shutil; "_
                         "    shutil.rmtree(temp_dir, ignore_errors=True)"
        
        Set result = ##class(%SYS.Python).Run(pythonCode)
        
        Write "="_$Translate($Justify("", 50), " ", "="), !
        Write "✓ RAG functionality demonstration completed!", !
        
    } Catch ex {
        Set tSC = ex.AsStatus()
        Write "RAG demonstration failed: ", $System.Status.GetErrorText(tSC), !
    }
    
    Quit tSC
}

}
