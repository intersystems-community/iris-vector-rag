"""
Module Loader Service.

This module provides the ModuleLoader class for dynamically loading
pipeline classes from module paths.
"""

import importlib
import logging
from typing import Any, Dict, Type

from ..core.base import RAGPipeline
from ..core.exceptions import ModuleLoadingError


class ModuleLoader:
    """
    Service for dynamically loading pipeline classes from modules.

    This service handles:
    - Dynamic import of Python modules
    - Loading specific classes from modules
    - Validation that loaded classes are RAGPipeline subclasses
    - Caching of loaded modules for performance
    """

    def __init__(self):
        """Initialize the module loader."""
        self.logger = logging.getLogger(__name__)
        self._module_cache: Dict[str, Any] = {}

    def load_pipeline_class(
        self, module_path: str, class_name: str, package_type: str = "core"
    ) -> Type:
        """
        Load a pipeline class from the specified module (supports both core and plugin packages).

        Args:
            module_path: Python module path (e.g., 'iris_rag.pipelines.basic' or 'hybridgraphrag.pipeline')
            class_name: Name of the class to load (e.g., 'BasicRAGPipeline')
            package_type: Type of package - "core" for iris_rag modules, "plugin" for external packages

        Returns:
            The loaded pipeline class

        Raises:
            ModuleLoadingError: If module or class cannot be loaded or is invalid
        """
        if package_type == "plugin":
            return self._load_external_pipeline_class(module_path, class_name)
        else:
            return self._load_core_pipeline_class(module_path, class_name)

    def _load_core_pipeline_class(self, module_path: str, class_name: str) -> Type:
        """
        Load a pipeline class from core iris_rag modules.

        Args:
            module_path: Python module path (e.g., 'iris_rag.pipelines.basic')
            class_name: Name of the class to load (e.g., 'BasicRAGPipeline')

        Returns:
            The loaded pipeline class

        Raises:
            ModuleLoadingError: If module or class cannot be loaded or is invalid
        """
        return self._load_and_validate_class(module_path, class_name)

    def _load_external_pipeline_class(self, module_path: str, class_name: str) -> Type:
        """
        Load a pipeline class from external plugin package.

        Args:
            module_path: Full import path to plugin module (e.g., 'hybridgraphrag.pipeline')
            class_name: Name of the class to load (e.g., 'HybridGraphRAGPipeline')

        Returns:
            The loaded pipeline class

        Raises:
            ModuleLoadingError: If module or class cannot be loaded or is invalid
        """
        # For plugin modules, the module_path is already the full import path
        # e.g., "hybridgraphrag.pipeline" instead of "iris_rag.pipelines.basic"
        self.logger.debug(
            f"Loading external plugin class: {class_name} from {module_path}"
        )
        return self._load_and_validate_class(module_path, class_name)

    def _load_and_validate_class(self, module_path: str, class_name: str) -> Type:
        """
        Load and validate a pipeline class from any module.

        Args:
            module_path: Python module path
            class_name: Name of the class to load

        Returns:
            The loaded and validated pipeline class

        Raises:
            ModuleLoadingError: If module or class cannot be loaded or is invalid
        """
        # Validate inputs
        if module_path is None:
            raise ModuleLoadingError("Module path cannot be None")
        if class_name is None:
            raise ModuleLoadingError("Class name cannot be None")
        if not module_path.strip():
            raise ModuleLoadingError("Module path cannot be empty")
        if not class_name.strip():
            raise ModuleLoadingError("Class name cannot be empty")

        try:
            # Check cache first
            if module_path in self._module_cache:
                module = self._module_cache[module_path]
                self.logger.debug(f"Using cached module: {module_path}")
            else:
                # Import the module
                self.logger.debug(f"Importing module: {module_path}")
                module = importlib.import_module(module_path)
                self._module_cache[module_path] = module

            # Get the class from the module
            if not hasattr(module, class_name):
                raise ModuleLoadingError(
                    f"Class '{class_name}' not found in module '{module_path}'"
                )

            pipeline_class = getattr(module, class_name)

            # Validate that it's a class and a subclass of RAGPipeline
            if not isinstance(pipeline_class, type):
                raise ModuleLoadingError(
                    f"'{class_name}' in module '{module_path}' is not a class"
                )

            if not issubclass(pipeline_class, RAGPipeline):
                raise ModuleLoadingError(
                    f"Class '{class_name}' in module '{module_path}' is not a subclass of RAGPipeline"
                )

            self.logger.info(f"Successfully loaded pipeline class: {class_name}")
            return pipeline_class

        except ImportError as e:
            error_msg = f"Failed to import module '{module_path}': {str(e)}"
            self.logger.error(error_msg)
            raise ModuleLoadingError(error_msg)

        except AttributeError as e:
            error_msg = (
                f"Class '{class_name}' not found in module '{module_path}': {str(e)}"
            )
            self.logger.error(error_msg)
            raise ModuleLoadingError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error loading class '{class_name}' from module '{module_path}': {str(e)}"
            self.logger.error(error_msg)
            raise ModuleLoadingError(error_msg)
