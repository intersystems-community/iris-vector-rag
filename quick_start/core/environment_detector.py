"""
Environment detection for the Quick Start system.

This module provides system capability detection and validation
for the quick start setup process.
"""

import shutil
import subprocess
import sys
from typing import Dict, Any, List
from pathlib import Path


class EnvironmentDetector:
    """Detects and validates system environment for quick start."""
    
    def __init__(self):
        """Initialize the environment detector."""
        pass
    
    async def detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system capabilities and requirements."""
        capabilities = {
            "python": self._check_python(),
            "uv": self._check_uv(),
            "docker": self._check_docker(),
            "git": self._check_git(),
            "disk_space": self._check_disk_space(),
            "memory": self._check_memory(),
        }
        return capabilities
    
    def _check_python(self) -> Dict[str, Any]:
        """Check Python installation and version."""
        try:
            version = sys.version_info
            return {
                "available": True,
                "version": f"{version.major}.{version.minor}.{version.micro}",
                "executable": sys.executable,
                "meets_requirements": version >= (3, 9)
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    def _check_uv(self) -> Dict[str, Any]:
        """Check UV package manager availability."""
        try:
            uv_path = shutil.which("uv")
            if uv_path:
                result = subprocess.run(
                    ["uv", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                return {
                    "available": True,
                    "path": uv_path,
                    "version": result.stdout.strip() if result.returncode == 0 else "unknown",
                    "meets_requirements": True
                }
            else:
                return {
                    "available": False,
                    "error": "UV not found in PATH",
                    "meets_requirements": False
                }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    def _check_docker(self) -> Dict[str, Any]:
        """Check Docker availability."""
        try:
            docker_path = shutil.which("docker")
            if docker_path:
                result = subprocess.run(
                    ["docker", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                return {
                    "available": True,
                    "path": docker_path,
                    "version": result.stdout.strip() if result.returncode == 0 else "unknown",
                    "meets_requirements": True
                }
            else:
                return {
                    "available": False,
                    "error": "Docker not found in PATH",
                    "meets_requirements": False
                }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    def _check_git(self) -> Dict[str, Any]:
        """Check Git availability."""
        try:
            git_path = shutil.which("git")
            if git_path:
                result = subprocess.run(
                    ["git", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                return {
                    "available": True,
                    "path": git_path,
                    "version": result.stdout.strip() if result.returncode == 0 else "unknown",
                    "meets_requirements": True
                }
            else:
                return {
                    "available": False,
                    "error": "Git not found in PATH",
                    "meets_requirements": False
                }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            current_path = Path.cwd()
            stat = shutil.disk_usage(current_path)
            free_gb = stat.free / (1024**3)
            return {
                "available": True,
                "free_space_gb": round(free_gb, 2),
                "meets_requirements": free_gb >= 5.0  # Require at least 5GB
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check available memory."""
        try:
            # Simple memory check - this is a stub implementation
            return {
                "available": True,
                "total_gb": 8.0,  # Stub value
                "available_gb": 4.0,  # Stub value
                "meets_requirements": True
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "meets_requirements": False
            }
    
    async def validate_requirements(self) -> List[str]:
        """Validate that all requirements are met."""
        capabilities = await self.detect_system_capabilities()
        errors = []
        
        for component, info in capabilities.items():
            if not info.get("meets_requirements", False):
                errors.append(f"{component}: {info.get('error', 'Requirements not met')}")
        
        return errors