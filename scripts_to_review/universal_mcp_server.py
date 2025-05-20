#!/usr/bin/env python3
"""
Universal MCP Server

A completely standalone MCP server designed to work in any environment.
This server has no external dependencies beyond Python standard library
and avoids all file operations and timing issues.
"""

import sys
import json

def respond_immediately(response):
    """Send a JSON response to stdout and flush immediately"""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()

def main():
    """Main entry point - ultra minimal implementation"""
    # Read a single line from stdin
    line = sys.stdin.readline().strip()
    if not line:
        return 1
    
    try:
        # Parse the request
        request = json.loads(line)
        
        # Handle initialization immediately
        if request.get("method") == "initialize":
            respond_immediately({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "simple_test": {
                                "description": "Test function",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        }
                    },
                    "serverInfo": {
                        "name": "Universal MCP Server",
                        "version": "1.0.0"
                    }
                }
            })
            
            # Continue reading input for further requests
            while True:
                line = sys.stdin.readline().strip()
                if not line:
                    break
                
                try:
                    request = json.loads(line)
                    
                    if request.get("method") in ["tools/execute", "tools/call"]:
                        tool_name = request.get("params", {}).get("name")
                        
                        if tool_name == "simple_test":
                            respond_immediately({
                                "jsonrpc": "2.0",
                                "id": request.get("id"),
                                "result": {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Test successful from universal_mcp_server.py!"
                                        }
                                    ]
                                }
                            })
                        else:
                            respond_immediately({
                                "jsonrpc": "2.0",
                                "id": request.get("id"),
                                "error": {
                                    "code": -32601,
                                    "message": f"Tool not found: {tool_name}"
                                }
                            })
                    
                    elif request.get("method") in ["shutdown", "exit"]:
                        respond_immediately({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": None
                        })
                        break
                    
                    else:
                        respond_immediately({
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {request.get('method')}"
                            }
                        })
                
                except json.JSONDecodeError:
                    respond_immediately({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Invalid JSON"
                        }
                    })
                except Exception as e:
                    respond_immediately({
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    })
        
        # Handle unexpected initial method
        else:
            respond_immediately({
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Expected 'initialize' as first method, got '{request.get('method')}'"
                }
            })
    
    except json.JSONDecodeError:
        respond_immediately({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32700,
                "message": "Invalid JSON"
            }
        })
    except Exception as e:
        respond_immediately({
            "jsonrpc": "2.0",
            "id": None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        })
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
