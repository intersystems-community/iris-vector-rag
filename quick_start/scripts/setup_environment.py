#!/usr/bin/env python3
"""
Environment Setup Script for Quick Start

This script handles environment setup and validation for the Quick Start system,
including checking system requirements, setting up environment variables,
and validating the development environment.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging


class EnvironmentSetup:
    """Handles environment setup and validation for Quick Start."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent.parent.parent
        self.env_file = self.project_root / '.env'
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for environment setup."""
        logger = logging.getLogger('quick_start.environment')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements for Quick Start setup."""
        requirements = {}
        
        # Check Python version
        python_version = sys.version_info
        requirements['python_version'] = python_version >= (3, 8)
        
        # Check for required commands
        required_commands = ['docker', 'docker-compose', 'uv']
        for cmd in required_commands:
            requirements[f'{cmd}_available'] = shutil.which(cmd) is not None
        
        # Check system resources
        requirements.update(self._check_system_resources())
        
        # Check Docker status
        requirements['docker_running'] = self._check_docker_status()
        
        return requirements
    
    def _check_system_resources(self) -> Dict[str, bool]:
        """Check system resource requirements."""
        resources = {}
        
        try:
            # Check available memory (basic check)
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', 'hw.memsize'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.split(':')[1].strip())
                    mem_gb = mem_bytes / (1024**3)
                    resources['sufficient_memory'] = mem_gb >= 4.0
                else:
                    resources['sufficient_memory'] = True  # Assume sufficient
            elif platform.system() == 'Linux':
                result = subprocess.run(['free', '-b'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    mem_line = lines[1]
                    mem_bytes = int(mem_line.split()[1])
                    mem_gb = mem_bytes / (1024**3)
                    resources['sufficient_memory'] = mem_gb >= 4.0
                else:
                    resources['sufficient_memory'] = True  # Assume sufficient
            else:
                resources['sufficient_memory'] = True  # Assume sufficient for other systems
                
        except Exception as e:
            self.logger.warning(f"Could not check memory: {e}")
            resources['sufficient_memory'] = True  # Assume sufficient
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            resources['sufficient_disk'] = free_gb >= 10.0  # 10GB minimum
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            resources['sufficient_disk'] = True  # Assume sufficient
        
        return resources
    
    def _check_docker_status(self) -> bool:
        """Check if Docker is running."""
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def setup_environment_variables(self, profile_config: Optional[Dict] = None) -> bool:
        """Setup environment variables for Quick Start."""
        try:
            env_vars = self._get_default_env_vars()
            
            # Add profile-specific variables if provided
            if profile_config:
                env_vars.update(profile_config.get('environment', {}))
            
            # Create or update .env file
            self._write_env_file(env_vars)
            
            self.logger.info("Environment variables configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup environment variables: {e}")
            return False
    
    def _get_default_env_vars(self) -> Dict[str, str]:
        """Get default environment variables for Quick Start."""
        return {
            'IRIS_HOST': 'localhost',
            'IRIS_PORT': '1972',
            'IRIS_NAMESPACE': 'USER',
            'IRIS_USERNAME': '_SYSTEM',
            'IRIS_PASSWORD': 'SYS',
            'PYTHONPATH': str(self.project_root),
            'QUICK_START_MODE': 'true',
            'LOG_LEVEL': 'INFO'
        }
    
    def _write_env_file(self, env_vars: Dict[str, str]) -> None:
        """Write environment variables to .env file."""
        # Read existing .env file if it exists
        existing_vars = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
        
        # Merge with new variables (new variables take precedence)
        merged_vars = {**existing_vars, **env_vars}
        
        # Write updated .env file
        with open(self.env_file, 'w') as f:
            f.write("# Quick Start Environment Configuration\n")
            f.write("# Generated by Quick Start setup system\n\n")
            
            for key, value in sorted(merged_vars.items()):
                f.write(f"{key}={value}\n")
    
    def validate_environment(self) -> Tuple[bool, List[str]]:
        """Validate the current environment setup."""
        issues = []
        
        # Check system requirements
        requirements = self.check_system_requirements()
        
        for req, status in requirements.items():
            if not status:
                issues.append(f"System requirement not met: {req}")
        
        # Check environment variables
        required_env_vars = ['IRIS_HOST', 'IRIS_PORT', 'IRIS_NAMESPACE']
        for var in required_env_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")
        
        # Check project structure
        required_dirs = ['common', 'iris_rag', 'quick_start']
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                issues.append(f"Missing project directory: {dir_name}")
        
        return len(issues) == 0, issues
    
    def print_environment_status(self) -> None:
        """Print current environment status."""
        print("\n" + "="*60)
        print("🔍 ENVIRONMENT STATUS")
        print("="*60)
        
        # System information
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Project Root: {self.project_root}")
        
        # System requirements
        requirements = self.check_system_requirements()
        print("\n📋 System Requirements:")
        for req, status in requirements.items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {req.replace('_', ' ').title()}: {'OK' if status else 'FAILED'}")
        
        # Environment validation
        is_valid, issues = self.validate_environment()
        print(f"\n🎯 Environment Status: {'✅ VALID' if is_valid else '❌ ISSUES FOUND'}")
        
        if issues:
            print("\n⚠️ Issues Found:")
            for issue in issues:
                print(f"  • {issue}")
        
        # Recommendations
        if not is_valid:
            print("\n🔧 Recommended Actions:")
            if not requirements.get('docker_running', True):
                print("  • Start Docker: docker-compose up -d")
            if not requirements.get('uv_available', True):
                print("  • Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
            if issues:
                print("  • Run: make quick-start-status for detailed diagnostics")


def main():
    """Main entry point for environment setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Start Environment Setup")
    parser.add_argument('--check', action='store_true', 
                       help='Check environment status')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment variables')
    parser.add_argument('--validate', action='store_true',
                       help='Validate environment configuration')
    
    args = parser.parse_args()
    
    env_setup = EnvironmentSetup()
    
    if args.check or not any([args.setup, args.validate]):
        env_setup.print_environment_status()
    
    if args.setup:
        success = env_setup.setup_environment_variables()
        if success:
            print("✅ Environment setup completed successfully")
        else:
            print("❌ Environment setup failed")
            sys.exit(1)
    
    if args.validate:
        is_valid, issues = env_setup.validate_environment()
        if is_valid:
            print("✅ Environment validation passed")
        else:
            print("❌ Environment validation failed")
            for issue in issues:
                print(f"  • {issue}")
            sys.exit(1)


if __name__ == '__main__':
    main()