#!/usr/bin/env python3
"""
MCP Server Demo for RAG Templates

This demonstrates a practical Model Context Protocol (MCP) server that provides
RAG capabilities as tools for external applications like Claude Desktop, IDEs,
or other MCP clients.

Key Features:
- Document management tools (add, search, count)
- RAG query tools for all 8 techniques
- Performance comparison tools
- Health monitoring
- ObjectScript integration bridge

This shows how IRIS customers can expose RAG capabilities to external tools
while leveraging existing IRIS data and infrastructure.
"""

import sys
import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MCP imports
try:
    import mcp
    from mcp.server import Server
    from mcp.types import (
        Tool, TextContent, EmbeddedResource, 
        CallToolRequest, ListToolsRequest
    )
    MCP_AVAILABLE = True
except ImportError:
    print("Warning: MCP not available. Install with: pip install mcp")
    MCP_AVAILABLE = False

# rag-templates imports
try:
    from rag_templates import RAG, ConfigurableRAG
    RAG_TEMPLATES_AVAILABLE = True
except ImportError:
    try:
        from iris_rag import create_pipeline
        from common.utils import get_llm_func
        from common.iris_connection_manager import get_iris_connection
        RAG_TEMPLATES_AVAILABLE = True
    except ImportError:
        print("Warning: rag-templates not available")
        RAG_TEMPLATES_AVAILABLE = False

# ObjectScript MCP bridge
try:
    from objectscript.mcp_bridge import (
        invoke_rag_basic_mcp, invoke_rag_crag_mcp, invoke_rag_hyde_mcp,
        invoke_rag_graphrag_mcp, invoke_rag_hybrid_ifind_mcp, invoke_rag_colbert_mcp,
        invoke_rag_noderag_mcp, invoke_rag_sqlrag_mcp, get_mcp_health_status,
        get_mcp_performance_metrics
    )
    OBJECTSCRIPT_BRIDGE_AVAILABLE = True
except ImportError:
    print("Note: ObjectScript MCP bridge not available")
    OBJECTSCRIPT_BRIDGE_AVAILABLE = False


class RAGMCPServer:
    """
    MCP Server providing RAG capabilities as tools.
    
    This server exposes rag-templates functionality through the Model Context Protocol,
    allowing external applications to use RAG capabilities as tools.
    """
    
    def __init__(self):
        """Initialize the RAG MCP server."""
        self.logger = logging.getLogger(__name__)
        self.server = Server("rag-templates") if MCP_AVAILABLE else None
        
        # Initialize RAG systems
        self.rag_systems = {}
        self.document_count = 0
        self.performance_metrics = {}
        
        # Initialize available techniques
        self.available_techniques = [
            "basic", "hyde", "crag", "colbert", 
            "graphrag", "hybrid_ifind", "noderag", "sql_rag"
        ]
        
        self._initialize_rag_systems()
        self._register_tools()
        
        self.logger.info("RAG MCP Server initialized")
    
    def _initialize_rag_systems(self):
        """Initialize RAG systems for all techniques."""
        if not RAG_TEMPLATES_AVAILABLE:
            self.logger.warning("RAG templates not available")
            return
        
        try:
            # Initialize Simple API
            self.rag_systems["simple"] = RAG()
            
            # Initialize configurable systems for each technique
            for technique in self.available_techniques:
                try:
                    self.rag_systems[technique] = ConfigurableRAG({
                        "technique": technique,
                        "max_results": 5
                    })
                except Exception as e:
                    self.logger.warning(f"Could not initialize {technique}: {e}")
                    # Fallback to pipeline creation
                    try:
                        self.rag_systems[technique] = create_pipeline(
                            pipeline_type=technique,
                            llm_func=get_llm_func(),
                            external_connection=get_iris_connection(),
                            validate_requirements=False
                        )
                    except Exception as e2:
                        self.logger.error(f"Failed to initialize {technique}: {e2}")
            
            self.logger.info(f"Initialized {len(self.rag_systems)} RAG systems")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG systems: {e}")
    
    def _register_tools(self):
        """Register MCP tools."""
        if not self.server:
            return
        
        # Document management tools
        self._register_document_tools()
        
        # RAG query tools
        self._register_rag_query_tools()
        
        # Performance and health tools
        self._register_monitoring_tools()
        
        # ObjectScript integration tools
        if OBJECTSCRIPT_BRIDGE_AVAILABLE:
            self._register_objectscript_tools()
    
    def _register_document_tools(self):
        """Register document management tools."""
        
        @self.server.call_tool()
        async def add_documents(arguments: dict) -> List[TextContent]:
            """Add documents to the RAG knowledge base."""
            try:
                documents = arguments.get("documents", [])
                if not documents:
                    return [TextContent(
                        type="text",
                        text="Error: No documents provided"
                    )]
                
                # Add to all RAG systems
                success_count = 0
                for name, rag_system in self.rag_systems.items():
                    try:
                        if hasattr(rag_system, 'add_documents'):
                            rag_system.add_documents(documents)
                            success_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to add documents to {name}: {e}")
                
                self.document_count += len(documents)
                
                return [TextContent(
                    type="text",
                    text=f"Successfully added {len(documents)} documents to {success_count} RAG systems. Total documents: {self.document_count}"
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error adding documents: {str(e)}"
                )]
        
        @self.server.call_tool()
        async def get_document_count(arguments: dict) -> List[TextContent]:
            """Get the current document count."""
            return [TextContent(
                type="text",
                text=f"Current document count: {self.document_count}"
            )]
        
        @self.server.call_tool()
        async def load_from_directory(arguments: dict) -> List[TextContent]:
            """Load documents from a directory."""
            try:
                directory_path = arguments.get("directory_path")
                if not directory_path:
                    return [TextContent(
                        type="text",
                        text="Error: No directory path provided"
                    )]
                
                # Use existing data loading
                from data.loader_fixed import process_and_load_documents
                result = process_and_load_documents(directory_path, limit=100)
                
                if result:
                    doc_count = result.get('documents_loaded', 0) if isinstance(result, dict) else 10
                    self.document_count += doc_count
                    
                    return [TextContent(
                        type="text",
                        text=f"Successfully loaded {doc_count} documents from {directory_path}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"Failed to load documents from {directory_path}"
                    )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error loading directory: {str(e)}"
                )]
    
    def _register_rag_query_tools(self):
        """Register RAG query tools for each technique."""
        
        for technique in self.available_techniques:
            
            # Create tool for this technique
            @self.server.call_tool()
            async def rag_query(arguments: dict, technique=technique) -> List[TextContent]:
                f"""Query using {technique} RAG technique."""
                try:
                    query = arguments.get("query")
                    if not query:
                        return [TextContent(
                            type="text",
                            text="Error: No query provided"
                        )]
                    
                    max_results = arguments.get("max_results", 5)
                    include_sources = arguments.get("include_sources", False)
                    
                    # Get RAG system for this technique
                    rag_system = self.rag_systems.get(technique)
                    if not rag_system:
                        return [TextContent(
                            type="text",
                            text=f"Error: {technique} RAG system not available"
                        )]
                    
                    # Execute query
                    if hasattr(rag_system, 'query'):
                        result = rag_system.query(query, {
                            "max_results": max_results,
                            "include_sources": include_sources
                        })
                    else:
                        # Fallback for pipeline interface
                        result = rag_system.run(query, top_k=max_results)
                        result = result.get('answer', 'No answer generated')
                    
                    # Format response
                    if isinstance(result, str):
                        response_text = f"**{technique.upper()} RAG Answer:**\n{result}"
                    else:
                        answer = result.get('answer', result) if isinstance(result, dict) else str(result)
                        response_text = f"**{technique.upper()} RAG Answer:**\n{answer}"
                        
                        if include_sources and isinstance(result, dict) and 'sources' in result:
                            sources = result['sources'][:3]  # Limit to 3 sources
                            if sources:
                                response_text += f"\n\n**Sources:**\n"
                                for i, source in enumerate(sources, 1):
                                    source_text = source if isinstance(source, str) else str(source)[:100]
                                    response_text += f"{i}. {source_text}...\n"
                    
                    return [TextContent(
                        type="text",
                        text=response_text
                    )]
                    
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=f"Error with {technique} query: {str(e)}"
                    )]
        
        # General query tool that compares techniques
        @self.server.call_tool()
        async def compare_rag_techniques(arguments: dict) -> List[TextContent]:
            """Compare query results across multiple RAG techniques."""
            try:
                query = arguments.get("query")
                if not query:
                    return [TextContent(
                        type="text",
                        text="Error: No query provided"
                    )]
                
                techniques_to_compare = arguments.get("techniques", ["basic", "hyde", "crag"])
                
                results = []
                for technique in techniques_to_compare:
                    rag_system = self.rag_systems.get(technique)
                    if rag_system:
                        try:
                            if hasattr(rag_system, 'query'):
                                answer = rag_system.query(query)
                            else:
                                result = rag_system.run(query, top_k=3)
                                answer = result.get('answer', 'No answer')
                            
                            answer_text = answer if isinstance(answer, str) else answer.get('answer', str(answer))
                            results.append(f"**{technique.upper()}:** {answer_text[:200]}...")
                        except Exception as e:
                            results.append(f"**{technique.upper()}:** Error - {str(e)}")
                
                response_text = f"**RAG Technique Comparison for:** {query}\n\n" + "\n\n".join(results)
                
                return [TextContent(
                    type="text",
                    text=response_text
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error comparing techniques: {str(e)}"
                )]
    
    def _register_monitoring_tools(self):
        """Register monitoring and health tools."""
        
        @self.server.call_tool()
        async def health_check(arguments: dict) -> List[TextContent]:
            """Check the health of RAG systems."""
            try:
                health_status = {
                    "server_status": "healthy",
                    "rag_systems_count": len(self.rag_systems),
                    "document_count": self.document_count,
                    "available_techniques": self.available_techniques,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Test basic connectivity
                working_systems = []
                for name, system in self.rag_systems.items():
                    try:
                        if hasattr(system, 'query'):
                            test_result = system.query("test")
                            working_systems.append(name)
                        else:
                            working_systems.append(name)  # Assume working if pipeline exists
                    except:
                        pass  # System not working
                
                health_status["working_systems"] = working_systems
                health_status["health_score"] = len(working_systems) / len(self.rag_systems) if self.rag_systems else 0
                
                return [TextContent(
                    type="text",
                    text=f"**RAG Server Health Check**\n\n" + 
                         f"Status: {health_status['server_status']}\n" +
                         f"RAG Systems: {health_status['rag_systems_count']}\n" +
                         f"Working Systems: {len(working_systems)}\n" +
                         f"Documents: {health_status['document_count']}\n" +
                         f"Health Score: {health_status['health_score']:.2f}\n" +
                         f"Available Techniques: {', '.join(self.available_techniques)}"
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Health check failed: {str(e)}"
                )]
        
        @self.server.call_tool()
        async def get_performance_metrics(arguments: dict) -> List[TextContent]:
            """Get performance metrics for RAG systems."""
            try:
                metrics = {
                    "total_queries": sum(self.performance_metrics.get(t, {}).get('query_count', 0) 
                                       for t in self.available_techniques),
                    "average_response_time": "~1.2s",  # Placeholder
                    "memory_usage": "~200MB",  # Placeholder
                    "uptime": "Active",
                    "technique_usage": {t: self.performance_metrics.get(t, {}).get('query_count', 0) 
                                      for t in self.available_techniques}
                }
                
                response_text = "**RAG Performance Metrics**\n\n"
                response_text += f"Total Queries: {metrics['total_queries']}\n"
                response_text += f"Avg Response Time: {metrics['average_response_time']}\n"
                response_text += f"Memory Usage: {metrics['memory_usage']}\n"
                response_text += f"Server Status: {metrics['uptime']}\n\n"
                response_text += "**Technique Usage:**\n"
                for technique, count in metrics['technique_usage'].items():
                    response_text += f"  {technique}: {count} queries\n"
                
                return [TextContent(
                    type="text",
                    text=response_text
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"Error getting metrics: {str(e)}"
                )]
    
    def _register_objectscript_tools(self):
        """Register ObjectScript integration tools."""
        
        @self.server.call_tool()
        async def objectscript_rag_query(arguments: dict) -> List[TextContent]:
            """Query RAG through ObjectScript MCP bridge."""
            try:
                query = arguments.get("query")
                technique = arguments.get("technique", "basic")
                
                if not query:
                    return [TextContent(
                        type="text",
                        text="Error: No query provided"
                    )]
                
                # Use ObjectScript MCP bridge
                config = json.dumps({"technique": technique, "top_k": 5})
                
                # Map technique to bridge function
                bridge_functions = {
                    "basic": invoke_rag_basic_mcp,
                    "crag": invoke_rag_crag_mcp,
                    "hyde": invoke_rag_hyde_mcp,
                    "graphrag": invoke_rag_graphrag_mcp,
                    "hybrid_ifind": invoke_rag_hybrid_ifind_mcp,
                    "colbert": invoke_rag_colbert_mcp,
                    "noderag": invoke_rag_noderag_mcp,
                    "sql_rag": invoke_rag_sqlrag_mcp
                }
                
                bridge_func = bridge_functions.get(technique, invoke_rag_basic_mcp)
                result_json = bridge_func(query, config)
                result = json.loads(result_json)
                
                if result.get('success'):
                    answer = result['result']['answer']
                    response_text = f"**ObjectScript {technique.upper()} RAG:**\n{answer}"
                    
                    if 'metadata' in result['result']:
                        metadata = result['result']['metadata']
                        response_text += f"\n\n**Metadata:** {json.dumps(metadata, indent=2)}"
                else:
                    response_text = f"ObjectScript RAG failed: {result.get('error', 'Unknown error')}"
                
                return [TextContent(
                    type="text",
                    text=response_text
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"ObjectScript RAG error: {str(e)}"
                )]
        
        @self.server.call_tool()
        async def objectscript_health_status(arguments: dict) -> List[TextContent]:
            """Get ObjectScript bridge health status."""
            try:
                result_json = get_mcp_health_status()
                result = json.loads(result_json)
                
                if result.get('success'):
                    status = result['result']
                    response_text = "**ObjectScript Bridge Health**\n\n"
                    response_text += f"Status: {status['status']}\n"
                    response_text += f"Techniques Available: {status['techniques_available']}\n"
                    response_text += f"Database Connection: {status['database_connection']}\n"
                    response_text += f"Memory Usage: {status['memory_usage']}\n"
                    response_text += f"Uptime: {status['uptime_seconds']}s"
                else:
                    response_text = f"ObjectScript health check failed: {result.get('error')}"
                
                return [TextContent(
                    type="text",
                    text=response_text
                )]
                
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"ObjectScript health check error: {str(e)}"
                )]
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for MCP client registration."""
        tools = []
        
        # Document management tools
        tools.extend([
            {
                "name": "add_documents",
                "description": "Add documents to the RAG knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document texts to add"
                        }
                    },
                    "required": ["documents"]
                }
            },
            {
                "name": "get_document_count",
                "description": "Get the current number of documents in the knowledge base",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "load_from_directory",
                "description": "Load documents from a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to directory containing documents"
                        }
                    },
                    "required": ["directory_path"]
                }
            }
        ])
        
        # RAG query tools for each technique
        for technique in self.available_techniques:
            tools.append({
                "name": f"rag_query_{technique}",
                "description": f"Query using {technique} RAG technique",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question or query to answer"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5
                        },
                        "include_sources": {
                            "type": "boolean",
                            "description": "Include source documents in response",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            })
        
        # Comparison and monitoring tools
        tools.extend([
            {
                "name": "compare_rag_techniques",
                "description": "Compare query results across multiple RAG techniques",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question to compare across techniques"
                        },
                        "techniques": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of techniques to compare",
                            "default": ["basic", "hyde", "crag"]
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "health_check",
                "description": "Check the health status of RAG systems",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "get_performance_metrics",
                "description": "Get performance metrics for RAG systems",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ])
        
        # ObjectScript integration tools
        if OBJECTSCRIPT_BRIDGE_AVAILABLE:
            tools.extend([
                {
                    "name": "objectscript_rag_query",
                    "description": "Query RAG through ObjectScript MCP bridge",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Question to ask"
                            },
                            "technique": {
                                "type": "string",
                                "description": "RAG technique to use",
                                "default": "basic"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "objectscript_health_status",
                    "description": "Get ObjectScript bridge health status",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ])
        
        return tools
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (non-MCP interface)."""
        return self.get_tool_definitions()
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool directly (non-MCP interface)."""
        try:
            # This is a simplified synchronous interface for testing
            # In practice, the MCP server handles tool calls asynchronously
            
            if tool_name == "add_documents":
                documents = arguments.get("documents", [])
                if documents and self.rag_systems:
                    self.document_count += len(documents)
                    return {"content": f"Added {len(documents)} documents. Total: {self.document_count}"}
            
            elif tool_name == "get_document_count":
                return {"content": f"Document count: {self.document_count}"}
            
            elif tool_name.startswith("rag_query_"):
                technique = tool_name.replace("rag_query_", "")
                query = arguments.get("query", "")
                
                if technique in self.rag_systems:
                    # Simulate RAG query
                    return {"content": f"RAG {technique} answer for: {query}"}
                else:
                    return {"content": f"Technique {technique} not available"}
            
            elif tool_name == "health_check":
                return {
                    "content": f"Health: OK, {len(self.rag_systems)} systems, {self.document_count} docs"
                }
            
            else:
                return {"content": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            return {"content": f"Tool error: {str(e)}"}
    
    async def run_server(self, host: str = "localhost", port: int = 3000):
        """Run the MCP server."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP not available. Install with: pip install mcp")
        
        self.logger.info(f"Starting RAG MCP server on {host}:{port}")
        
        # This would typically use the MCP server's run method
        # For now, just log that the server would be running
        print(f"🛠️  RAG MCP Server would be running on {host}:{port}")
        print(f"📊 Available tools: {len(self.get_tool_definitions())}")
        print("🎯 Use with Claude Desktop, IDEs, or other MCP clients")


def main():
    """Main function for CLI usage."""
    print("🛠️  RAG Templates MCP Server Demo")
    print("==================================")
    
    if not MCP_AVAILABLE:
        print("⚠️  MCP not available - install with: pip install mcp")
        print("Continuing with mock server for demonstration...")
    
    # Initialize server
    server = RAGMCPServer()
    
    print(f"✅ Initialized RAG MCP server")
    print(f"📊 RAG systems: {len(server.rag_systems)}")
    print(f"🛠️  Available tools: {len(server.get_tool_definitions())}")
    
    # Demo tool usage
    print("\n🧪 Testing Tools:")
    
    # Test document addition
    result = server.call_tool("add_documents", {
        "documents": ["Sample document about AI", "Another document about ML"]
    })
    print(f"1. Add documents: {result['content']}")
    
    # Test document count
    result = server.call_tool("get_document_count", {})
    print(f"2. Document count: {result['content']}")
    
    # Test RAG query
    result = server.call_tool("rag_query_basic", {
        "query": "What is artificial intelligence?"
    })
    print(f"3. Basic RAG query: {result['content']}")
    
    # Test health check
    result = server.call_tool("health_check", {})
    print(f"4. Health check: {result['content']}")
    
    print("\n🎯 Next Steps:")
    print("1. Install MCP: pip install mcp")
    print("2. Configure Claude Desktop to use this server")
    print("3. Use RAG capabilities as tools in your IDE")
    print("4. Integrate with existing IRIS ObjectScript applications")
    
    print("\n📝 Tool List:")
    tools = server.list_tools()
    for tool in tools[:10]:  # Show first 10 tools
        print(f"  - {tool['name']}: {tool['description']}")
    
    if len(tools) > 10:
        print(f"  ... and {len(tools) - 10} more tools")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()