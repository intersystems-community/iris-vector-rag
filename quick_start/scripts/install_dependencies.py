#!/usr/bin/env python3
"""
Dependency Installation Script for Quick Start

This script handles dependency installation and management for the Quick Start system,
including Python packages, system dependencies, and Docker containers.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json


class DependencyInstaller:
    """Handles dependency installation for Quick Start."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent.parent.parent
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for dependency installation."""
        logger = logging.getLogger('quick_start.dependencies')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def check_uv_installation(self) -> bool:
        """Check if uv is installed and available."""
        return shutil.which('uv') is not None
    
    def install_uv(self) -> bool:
        """Install uv package manager."""
        try:
            self.logger.info("Installing uv package manager...")
            
            # Use the official installation script
            install_cmd = [
                'curl', '-LsSf', 'https://astral.sh/uv/install.sh'
            ]
            
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Failed to download uv installer: {result.stderr}")
                return False
            
            # Execute the installer
            install_script = result.stdout
            result = subprocess.run(['sh'], input=install_script, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ uv installed successfully")
                return True
            else:
                self.logger.error(f"Failed to install uv: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing uv: {e}")
            return False
    
    def install_python_dependencies(self, profile: str = "standard") -> bool:
        """Install Python dependencies using uv."""
        try:
            if not self.check_uv_installation():
                self.logger.info("uv not found, installing...")
                if not self.install_uv():
                    return False
            
            self.logger.info("Installing Python dependencies with uv...")
            
            # Change to project root
            os.chdir(self.project_root)
            
            # Install dependencies
            cmd = ['uv', 'sync', '--frozen', '--all-extras', '--dev']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Python dependencies installed successfully")
                return True
            else:
                self.logger.error(f"Failed to install Python dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing Python dependencies: {e}")
            return False
    
    def check_docker_installation(self) -> Tuple[bool, bool]:
        """Check if Docker and Docker Compose are installed."""
        docker_available = shutil.which('docker') is not None
        compose_available = shutil.which('docker-compose') is not None
        
        return docker_available, compose_available
    
    def start_docker_services(self, profile: str = "standard") -> bool:
        """Start Docker services for the specified profile."""
        try:
            docker_available, compose_available = self.check_docker_installation()
            
            if not docker_available:
                self.logger.error("Docker is not installed or not available")
                return False
            
            if not compose_available:
                self.logger.error("Docker Compose is not installed or not available")
                return False
            
            self.logger.info("Starting Docker services...")
            
            # Change to project root
            os.chdir(self.project_root)
            
            # Start Docker services
            cmd = ['docker-compose', 'up', '-d']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Docker services started successfully")
                return True
            else:
                self.logger.error(f"Failed to start Docker services: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting Docker services: {e}")
            return False
    
    def check_system_dependencies(self) -> Dict[str, bool]:
        """Check system-level dependencies."""
        dependencies = {}
        
        # Required system commands
        required_commands = [
            'git', 'curl', 'docker', 'docker-compose'
        ]
        
        for cmd in required_commands:
            dependencies[cmd] = shutil.which(cmd) is not None
        
        # Check Python version
        python_version = sys.version_info
        dependencies['python_version'] = python_version >= (3, 8)
        
        return dependencies
    
    def install_system_dependencies_guidance(self) -> Dict[str, str]:
        """Provide guidance for installing missing system dependencies."""
        import platform
        
        system = platform.system().lower()
        guidance = {}
        
        if system == 'darwin':  # macOS
            guidance.update({
                'docker': 'Install Docker Desktop from https://docker.com/products/docker-desktop',
                'git': 'Install via Xcode Command Line Tools: xcode-select --install',
                'curl': 'Usually pre-installed on macOS'
            })
        elif system == 'linux':
            guidance.update({
                'docker': 'Install via package manager: sudo apt-get install docker.io docker-compose (Ubuntu/Debian)',
                'git': 'Install via package manager: sudo apt-get install git (Ubuntu/Debian)',
                'curl': 'Install via package manager: sudo apt-get install curl (Ubuntu/Debian)'
            })
        else:  # Windows or other
            guidance.update({
                'docker': 'Install Docker Desktop from https://docker.com/products/docker-desktop',
                'git': 'Install from https://git-scm.com/downloads',
                'curl': 'Usually available in PowerShell or install via package manager'
            })
        
        return guidance
    
    def validate_installation(self) -> Tuple[bool, List[str]]:
        """Validate that all dependencies are properly installed."""
        issues = []
        
        # Check system dependencies
        sys_deps = self.check_system_dependencies()
        for dep, available in sys_deps.items():
            if not available:
                issues.append(f"Missing system dependency: {dep}")
        
        # Check uv installation
        if not self.check_uv_installation():
            issues.append("uv package manager not installed")
        
        # Check Docker status
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                issues.append("Docker is not running")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Docker is not available or not running")
        
        # Check Python environment
        try:
            result = subprocess.run(['uv', 'run', 'python', '-c', 'import iris_rag'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                issues.append("Python environment not properly configured")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("Cannot validate Python environment")
        
        return len(issues) == 0, issues
    
    def print_installation_status(self) -> None:
        """Print current installation status."""
        print("\n" + "="*60)
        print("📦 DEPENDENCY INSTALLATION STATUS")
        print("="*60)
        
        # System dependencies
        sys_deps = self.check_system_dependencies()
        print("\n🔧 System Dependencies:")
        for dep, available in sys_deps.items():
            emoji = "✅" if available else "❌"
            print(f"  {emoji} {dep}: {'Available' if available else 'Missing'}")
        
        # uv installation
        uv_available = self.check_uv_installation()
        emoji = "✅" if uv_available else "❌"
        print(f"\n📦 Package Manager:")
        print(f"  {emoji} uv: {'Available' if uv_available else 'Missing'}")
        
        # Docker status
        docker_available, compose_available = self.check_docker_installation()
        print(f"\n🐳 Docker:")
        print(f"  {'✅' if docker_available else '❌'} Docker: {'Available' if docker_available else 'Missing'}")
        print(f"  {'✅' if compose_available else '❌'} Docker Compose: {'Available' if compose_available else 'Missing'}")
        
        # Installation validation
        is_valid, issues = self.validate_installation()
        print(f"\n🎯 Installation Status: {'✅ COMPLETE' if is_valid else '❌ INCOMPLETE'}")
        
        if issues:
            print("\n⚠️ Issues Found:")
            for issue in issues:
                print(f"  • {issue}")
        
        # Installation guidance
        if not is_valid:
            print("\n🔧 Installation Guidance:")
            guidance = self.install_system_dependencies_guidance()
            missing_deps = [dep for dep, available in sys_deps.items() if not available]
            
            for dep in missing_deps:
                if dep in guidance:
                    print(f"  • {dep}: {guidance[dep]}")
            
            if not uv_available:
                print("  • uv: Run 'curl -LsSf https://astral.sh/uv/install.sh | sh'")


def main():
    """Main entry point for dependency installation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Start Dependency Installation")
    parser.add_argument('--check', action='store_true', 
                       help='Check dependency status')
    parser.add_argument('--install-python', action='store_true',
                       help='Install Python dependencies')
    parser.add_argument('--install-uv', action='store_true',
                       help='Install uv package manager')
    parser.add_argument('--start-docker', action='store_true',
                       help='Start Docker services')
    parser.add_argument('--profile', default='standard',
                       help='Profile for dependency installation')
    parser.add_argument('--validate', action='store_true',
                       help='Validate installation')
    
    args = parser.parse_args()
    
    installer = DependencyInstaller()
    
    if args.check or not any([args.install_python, args.install_uv, 
                             args.start_docker, args.validate]):
        installer.print_installation_status()
    
    if args.install_uv:
        success = installer.install_uv()
        if not success:
            sys.exit(1)
    
    if args.install_python:
        success = installer.install_python_dependencies(args.profile)
        if not success:
            sys.exit(1)
    
    if args.start_docker:
        success = installer.start_docker_services(args.profile)
        if not success:
            sys.exit(1)
    
    if args.validate:
        is_valid, issues = installer.validate_installation()
        if is_valid:
            print("✅ Installation validation passed")
        else:
            print("❌ Installation validation failed")
            for issue in issues:
                print(f"  • {issue}")
            sys.exit(1)


if __name__ == '__main__':
    main()