"""
Entry point for the iris_rag CLI module.

This allows the CLI to be executed as:
    python -m iris_rag.cli reconcile --help
    python -m iris_rag.cli reconcile run --pipeline colbert
"""

from iris_rag.cli.reconcile_cli import reconcile

if __name__ == '__main__':
    reconcile()