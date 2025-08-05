"""
Quick Start Setup Pipeline Package.

This package provides the one-command setup system that builds on the CLI wizard
to provide streamlined setup with single commands for different profiles.
"""

from .pipeline import OneCommandSetupPipeline
from .steps import SetupStep, SetupStepResult
from .validators import SetupValidator
from .rollback import RollbackManager
from .makefile_integration import MakefileTargetHandler

__all__ = [
    'OneCommandSetupPipeline',
    'SetupStep',
    'SetupStepResult',
    'SetupValidator',
    'RollbackManager',
    'MakefileTargetHandler'
]