"""
MCP Server CLI Management Tool.

Provides command-line interface for managing the MCP server:
- start: Start MCP server (standalone or integrated mode)
- stop: Stop running MCP server
- status: Show MCP server status
- health: Check MCP server health
- list-tools: List available MCP tools

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import asyncio
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import requests
import json

from iris_rag.mcp.config import load_config, MCPConfig


class MCPServerCLI:
    """CLI interface for MCP server management."""

    def __init__(self):
        self.config: Optional[MCPConfig] = None
        self.nodejs_dir = Path(__file__).parent.parent.parent / 'nodejs'

    def load_configuration(self) -> MCPConfig:
        """Load MCP configuration."""
        if self.config is None:
            self.config = load_config()
        return self.config

    def start(self, mode: Optional[str] = None, transport: Optional[str] = None) -> int:
        """Start MCP server."""
        config = self.load_configuration()

        # Override config with CLI arguments
        deployment_mode = mode or config.deployment_mode
        transport_mode = transport or config.transport

        print(f"Starting MCP Server...")
        print(f"  Mode: {deployment_mode}")
        print(f"  Transport: {transport_mode}")

        if deployment_mode == 'standalone':
            return self._start_standalone(transport_mode)
        elif deployment_mode == 'integrated':
            return self._start_integrated()
        else:
            print(f"ERROR: Unknown deployment mode '{deployment_mode}'")
            print("Valid modes: standalone, integrated")
            return 1

    def _start_standalone(self, transport: str) -> int:
        """Start MCP server in standalone mode."""
        print("\nStarting Standalone Mode:")
        print("  1. Python FastAPI bridge on port 8001")
        print(f"  2. Node.js MCP server ({transport} transport)")
        print()

        # Check if Node.js CLI exists
        cli_path = self.nodejs_dir / 'dist' / 'mcp' / 'cli.js'
        if not cli_path.exists():
            print(f"ERROR: Node.js CLI not found at {cli_path}")
            print("Please run: cd nodejs && npm run build")
            return 1

        # Import and run __main__.py (starts both Python and Node.js)
        try:
            from iris_rag.mcp import __main__
            __main__.main()
            return 0
        except KeyboardInterrupt:
            print("\nShutdown requested")
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    def _start_integrated(self) -> int:
        """Start MCP server in integrated mode (with REST API)."""
        print("\nStarting Integrated Mode:")
        print("  REST API with embedded MCP bridge on port 8000")
        print()

        try:
            # Start REST API with MCP integration
            subprocess.run([
                'uvicorn',
                'iris_rag.api.main:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--log-level', 'info'
            ], check=True)
            return 0
        except KeyboardInterrupt:
            print("\nShutdown requested")
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    def stop(self) -> int:
        """Stop MCP server."""
        config = self.load_configuration()

        print("Stopping MCP Server...")

        # Try to stop via HTTP endpoints
        try:
            if config.deployment_mode == 'standalone':
                url = f"{config.python_bridge_url}/shutdown"
            else:
                url = "http://localhost:8000/api/v1/shutdown"

            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print("  ✓ MCP server stopped successfully")
                return 0
        except requests.exceptions.RequestException:
            pass

        # Fallback: kill processes by port
        print("  Attempting to kill processes...")
        try:
            if config.deployment_mode == 'standalone':
                subprocess.run(['pkill', '-f', 'iris_rag.mcp'], check=False)
                subprocess.run(['pkill', '-f', 'nodejs/dist/mcp/cli.js'], check=False)
            else:
                subprocess.run(['pkill', '-f', 'uvicorn.*iris_rag.api.main'], check=False)

            print("  ✓ Processes terminated")
            return 0
        except Exception as e:
            print(f"  ERROR: {e}")
            return 1

    def status(self) -> int:
        """Show MCP server status."""
        config = self.load_configuration()

        print("MCP Server Status:")
        print(f"  Deployment Mode: {config.deployment_mode}")
        print(f"  Transport: {config.transport}")
        print(f"  Python Bridge URL: {config.python_bridge_url}")
        print()

        # Check if services are running
        running = False

        if config.deployment_mode == 'standalone':
            # Check Python bridge
            try:
                response = requests.get(
                    f"{config.python_bridge_url}/mcp/health_check",
                    timeout=2
                )
                if response.status_code == 200:
                    print("  ✓ Python Bridge: RUNNING")
                    running = True
                else:
                    print("  ✗ Python Bridge: ERROR")
            except requests.exceptions.RequestException:
                print("  ✗ Python Bridge: NOT RUNNING")

            # Check Node.js server (if HTTP transport)
            if config.transport in ('http', 'both'):
                try:
                    response = requests.get(
                        f"http://localhost:{config.http_port}/health",
                        timeout=2
                    )
                    if response.status_code == 200:
                        print("  ✓ MCP Server (HTTP): RUNNING")
                        running = True
                    else:
                        print("  ✗ MCP Server (HTTP): ERROR")
                except requests.exceptions.RequestException:
                    print("  ✗ MCP Server (HTTP): NOT RUNNING")

        else:  # integrated mode
            try:
                response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
                if response.status_code == 200:
                    print("  ✓ REST API + MCP: RUNNING")
                    running = True
                else:
                    print("  ✗ REST API + MCP: ERROR")
            except requests.exceptions.RequestException:
                print("  ✗ REST API + MCP: NOT RUNNING")

        return 0 if running else 1

    def health(self) -> int:
        """Check MCP server health."""
        config = self.load_configuration()

        print("MCP Server Health Check:")
        print()

        try:
            if config.deployment_mode == 'standalone':
                url = f"{config.python_bridge_url}/mcp/health_check"
            else:
                url = "http://localhost:8000/api/v1/mcp/health"

            response = requests.get(url, timeout=5, params={
                'include_details': True,
                'include_performance_metrics': True
            })

            if response.status_code == 200:
                health_data = response.json()
                print(json.dumps(health_data, indent=2))
                return 0
            else:
                print(f"  ERROR: HTTP {response.status_code}")
                print(f"  {response.text}")
                return 1

        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Cannot connect to MCP server")
            print(f"  {e}")
            return 1

    def list_tools(self) -> int:
        """List available MCP tools."""
        config = self.load_configuration()

        print("Available MCP Tools:")
        print()

        try:
            if config.deployment_mode == 'standalone':
                url = f"{config.python_bridge_url}/mcp/list_techniques"
            else:
                url = "http://localhost:8000/api/v1/mcp/tools"

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                tools = response.json()
                if isinstance(tools, dict) and 'tools' in tools:
                    tools = tools['tools']

                for i, tool in enumerate(tools, 1):
                    if isinstance(tool, dict):
                        print(f"{i}. {tool.get('name', 'unknown')}")
                        print(f"   {tool.get('description', 'No description')}")
                    else:
                        print(f"{i}. {tool}")
                    print()

                return 0
            else:
                print(f"  ERROR: HTTP {response.status_code}")
                return 1

        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Cannot connect to MCP server")
            print(f"  {e}")
            return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='MCP Server Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start in standalone mode (default)
  python -m iris_rag.mcp.cli start

  # Start in integrated mode
  python -m iris_rag.mcp.cli start --mode integrated

  # Start with HTTP transport
  python -m iris_rag.mcp.cli start --transport http

  # Check server status
  python -m iris_rag.mcp.cli status

  # Check health
  python -m iris_rag.mcp.cli health

  # List available tools
  python -m iris_rag.mcp.cli list-tools

  # Stop server
  python -m iris_rag.mcp.cli stop
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start MCP server')
    start_parser.add_argument(
        '--mode',
        choices=['standalone', 'integrated'],
        help='Deployment mode (default: from config)'
    )
    start_parser.add_argument(
        '--transport',
        choices=['stdio', 'http', 'both'],
        help='Transport mode (default: from config)'
    )

    # Stop command
    subparsers.add_parser('stop', help='Stop MCP server')

    # Status command
    subparsers.add_parser('status', help='Show MCP server status')

    # Health command
    subparsers.add_parser('health', help='Check MCP server health')

    # List-tools command
    subparsers.add_parser('list-tools', help='List available MCP tools')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cli = MCPServerCLI()

    if args.command == 'start':
        return cli.start(mode=args.mode, transport=getattr(args, 'transport', None))
    elif args.command == 'stop':
        return cli.stop()
    elif args.command == 'status':
        return cli.status()
    elif args.command == 'health':
        return cli.health()
    elif args.command == 'list-tools':
        return cli.list_tools()
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
