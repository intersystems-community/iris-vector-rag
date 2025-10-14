"""
Makefile Target Consistency Audit Tool

This module provides static analysis of Makefile targets to detect:
- Hardcoded configuration values
- Manual schema initialization (framework violations)
- Missing prerequisite verification
- Inconsistent patterns across targets
- Pipeline test setup issues
- Documentation gaps
- Environment variable precedence conflicts
"""

__version__ = "1.0.0"
