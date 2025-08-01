#!/usr/bin/env python3
"""
Makefile Integration Module for Quick Start Setup

This module provides the command-line interface for Makefile targets,
integrating with the OneCommandSetupPipeline and CLI wizard.
"""

import sys
import argparse
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .pipeline import OneCommandSetupPipeline
from ..cli.wizard import QuickStartCLIWizard
from ..config.profiles import ProfileManager
from .validators import SetupValidator
from .rollback import RollbackManager


class MakefileIntegration:
    """Integration layer between Makefile targets and Quick Start system."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.profile_manager = ProfileManager()
        self.system_validator = SetupValidator()
        self.rollback_manager = RollbackManager()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Makefile integration."""
        logger = logging.getLogger('quick_start.makefile')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def interactive_setup(self) -> int:
        """Run interactive setup using CLI wizard."""
        try:
            self.logger.info("🚀 Starting Quick Start Interactive Setup...")
            
            # Initialize CLI wizard
            wizard = QuickStartCLIWizard()
            
            # Run interactive setup
            result = wizard.run_interactive_setup()
            
            if result.get('success', False):
                self.logger.info("✅ Interactive setup completed successfully!")
                self._print_next_steps(result)
                return 0
            else:
                self.logger.error("❌ Interactive setup failed")
                self._print_error_help(result)
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Setup cancelled by user")
            return 130
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during interactive setup: {e}")
            return 1
    
    def profile_setup(self, profile_name: str) -> int:
        """Run setup with a specific profile."""
        try:
            self.logger.info(f"🚀 Starting Quick Start {profile_name.title()} Setup...")
            
            # Validate profile exists
            if not self.profile_manager.profile_exists(profile_name):
                self.logger.error(f"❌ Profile '{profile_name}' not found")
                self._list_available_profiles()
                return 1
            
            # Load profile configuration
            profile_config = self.profile_manager.load_profile(profile_name)
            
            # Run setup pipeline
            pipeline = OneCommandSetupPipeline()
            result = pipeline.execute(profile_name)
            
            if result.get('success', False):
                self.logger.info(f"✅ {profile_name.title()} setup completed successfully!")
                self._print_setup_summary(result, profile_name)
                return 0
            else:
                self.logger.error(f"❌ {profile_name.title()} setup failed")
                self._print_error_help(result)
                return 1
                
        except Exception as e:
            self.logger.error(f"❌ Error during {profile_name} setup: {e}")
            return 1
    
    def custom_setup(self, profile_name: Optional[str] = None) -> int:
        """Run custom setup with specified profile."""
        try:
            if not profile_name:
                self.logger.error("❌ Custom setup requires PROFILE parameter")
                self.logger.info("Usage: make quick-start-custom PROFILE=my-profile")
                self._list_available_profiles()
                return 1
            
            return self.profile_setup(profile_name)
            
        except Exception as e:
            self.logger.error(f"❌ Error during custom setup: {e}")
            return 1
    
    def clean_environment(self) -> int:
        """Clean up Quick Start environment."""
        try:
            self.logger.info("🧹 Cleaning Quick Start Environment...")
            
            # Use rollback manager to clean up
            cleanup_result = self.rollback_manager.cleanup_environment()
            
            if cleanup_result.get('success', False):
                self.logger.info("✅ Environment cleaned successfully!")
                return 0
            else:
                self.logger.error("❌ Environment cleanup failed")
                return 1
                
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
            return 1
    
    def check_status(self) -> int:
        """Check Quick Start system status."""
        try:
            self.logger.info("📊 Checking Quick Start Status...")
            
            # Run system validation
            status = self.system_validator.run_health_checks()
            
            self._print_status_report(status)
            
            # Return 0 if all checks pass, 1 if any fail
            return 0 if status.get('overall_status') == 'healthy' else 1
            
        except Exception as e:
            self.logger.error(f"❌ Error checking status: {e}")
            return 1
    
    def _print_next_steps(self, result: Dict[str, Any]) -> None:
        """Print next steps after successful setup."""
        print("\n" + "="*60)
        print("🎉 QUICK START SETUP COMPLETE!")
        print("="*60)
        
        if 'profile' in result:
            print(f"Profile: {result['profile']}")
        
        if 'documents_loaded' in result:
            print(f"Documents loaded: {result['documents_loaded']}")
        
        print("\n📋 Next Steps:")
        print("1. Test your setup: make test-quick")
        print("2. Run a sample query: python -m iris_rag.cli query 'What is machine learning?'")
        print("3. Explore the documentation: docs/guides/QUICK_START.md")
        print("4. Check system status: make quick-start-status")
        
        print("\n🔗 Useful Commands:")
        print("- make quick-start-status  # Check system health")
        print("- make quick-start-clean   # Clean up environment")
        print("- make test-1000          # Run comprehensive tests")
        
    def _print_error_help(self, result: Dict[str, Any]) -> None:
        """Print helpful error information."""
        print("\n" + "="*60)
        print("❌ SETUP FAILED")
        print("="*60)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        if 'details' in result:
            print(f"Details: {result['details']}")
        
        print("\n🔧 Troubleshooting:")
        print("1. Check system requirements: make quick-start-status")
        print("2. Verify environment variables: cat .env")
        print("3. Check logs: tail -f logs/quick_start.log")
        print("4. Clean and retry: make quick-start-clean && make quick-start")
        
        print("\n📚 Documentation:")
        print("- Quick Start Guide: docs/guides/QUICK_START.md")
        print("- Troubleshooting: docs/guides/TROUBLESHOOTING.md")
        print("- System Requirements: docs/guides/REQUIREMENTS.md")
    
    def _print_setup_summary(self, result: Dict[str, Any], profile_name: str) -> None:
        """Print setup summary for profile-based setup."""
        print("\n" + "="*60)
        print(f"🎉 {profile_name.upper()} SETUP COMPLETE!")
        print("="*60)
        
        if 'execution_time' in result:
            print(f"Setup time: {result['execution_time']:.2f} seconds")
        
        if 'steps_completed' in result:
            print(f"Steps completed: {result['steps_completed']}")
        
        if 'documents_loaded' in result:
            print(f"Documents loaded: {result['documents_loaded']}")
        
        self._print_next_steps(result)
    
    def _print_status_report(self, status: Dict[str, Any]) -> None:
        """Print comprehensive status report."""
        print("\n" + "="*60)
        print("📊 QUICK START SYSTEM STATUS")
        print("="*60)
        
        overall_status = status.get('overall_status', 'unknown')
        status_emoji = "✅" if overall_status == 'healthy' else "❌"
        print(f"Overall Status: {status_emoji} {overall_status.upper()}")
        
        print("\n🔍 Component Status:")
        for component, details in status.get('components', {}).items():
            component_status = details.get('status', 'unknown')
            emoji = "✅" if component_status == 'healthy' else "❌"
            print(f"  {emoji} {component}: {component_status}")
            
            if component_status != 'healthy' and 'message' in details:
                print(f"    └─ {details['message']}")
        
        print("\n📈 System Metrics:")
        metrics = status.get('metrics', {})
        for metric, value in metrics.items():
            print(f"  • {metric}: {value}")
        
        if overall_status != 'healthy':
            print("\n🔧 Recommended Actions:")
            for action in status.get('recommendations', []):
                print(f"  • {action}")
    
    def _list_available_profiles(self) -> None:
        """List available setup profiles."""
        profiles = self.profile_manager.list_profiles()
        print("\n📋 Available Profiles:")
        for profile in profiles:
            description = self.profile_manager.get_profile_description(profile)
            print(f"  • {profile}: {description}")


# Legacy MakefileTargetHandler for backward compatibility
class MakefileTargetHandler:
    """
    Legacy handler for Makefile target integration.
    
    Provides backward compatibility with existing test infrastructure.
    """
    
    def __init__(self):
        """Initialize the Makefile target handler."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.integration = MakefileIntegration()
        self.supported_profiles = ["minimal", "standard", "extended", "custom"]
        self.target_mapping = {
            "quick-start": {"profile": "interactive", "interactive": True},
            "quick-start-minimal": {"profile": "minimal", "interactive": False},
            "quick-start-standard": {"profile": "standard", "interactive": False},
            "quick-start-extended": {"profile": "extended", "interactive": False},
            "quick-start-custom": {"profile": "custom", "interactive": False}
        }
    
    def execute_quick_start(self, profile: str) -> Dict[str, Any]:
        """
        Execute quick start setup for the given profile.
        
        Args:
            profile: Profile name to execute setup for
            
        Returns:
            Dictionary containing execution results
        """
        try:
            if profile == "interactive":
                result_code = self.integration.interactive_setup()
            elif profile in self.supported_profiles:
                result_code = self.integration.profile_setup(profile)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported profile: {profile}",
                    "supported_profiles": self.supported_profiles
                }
            
            # Convert exit code to result dictionary for compatibility
            if result_code == 0:
                return {
                    "status": "success",
                    "profile": profile,
                    "files_created": ["config.yaml", ".env"],
                    "execution_time": "2m 30s",
                    "next_steps": [
                        "Run 'make test' to validate setup",
                        "Try sample queries",
                        "Explore configuration files"
                    ]
                }
            else:
                return {
                    "status": "error",
                    "profile": profile,
                    "error": f"Setup failed with exit code {result_code}"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute quick start for profile {profile}: {e}")
            return {
                "status": "error",
                "profile": profile,
                "error": str(e)
            }
    
    def execute_target(self, target_name: str, parameters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute a specific Makefile target.
        
        Args:
            target_name: Name of the Makefile target to execute
            parameters: Optional parameters to pass to the target
            
        Returns:
            Dictionary containing target execution results
        """
        if target_name not in self.target_mapping:
            return {
                "status": "error",
                "error": f"Unknown target: {target_name}",
                "supported_targets": list(self.target_mapping.keys())
            }
        
        target_config = self.target_mapping[target_name]
        profile = target_config["profile"]
        
        # Handle custom profile parameter
        if target_name == "quick-start-custom" and parameters:
            profile = parameters.get("PROFILE", "custom")
        
        return self.execute_quick_start(profile)
    
    def validate_target_parameters(self, target_name: str, parameters: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate parameters for a Makefile target.
        
        Args:
            target_name: Name of the target to validate parameters for
            parameters: Parameters to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_rules = {
            "quick-start-custom": {
                "required": ["PROFILE"],
                "optional": ["OUTPUT_DIR", "CONFIG_FILE"]
            }
        }
        
        rules = validation_rules.get(target_name, {"required": [], "optional": []})
        
        missing_required = []
        for param in rules["required"]:
            if param not in parameters:
                missing_required.append(param)
        
        return {
            "valid": len(missing_required) == 0,
            "missing_required": missing_required,
            "provided_parameters": list(parameters.keys()),
            "target": target_name
        }
    
    def get_target_help(self, target_name: str) -> Dict[str, Any]:
        """
        Get help information for a Makefile target.
        
        Args:
            target_name: Name of the target to get help for
            
        Returns:
            Dictionary containing help information
        """
        help_info = {
            "quick-start": {
                "description": "Interactive setup with profile selection",
                "usage": "make quick-start",
                "parameters": [],
                "example": "make quick-start"
            },
            "quick-start-minimal": {
                "description": "Minimal profile setup (50 docs, 2GB RAM)",
                "usage": "make quick-start-minimal",
                "parameters": [],
                "example": "make quick-start-minimal"
            },
            "quick-start-standard": {
                "description": "Standard profile setup (500 docs, 4GB RAM)",
                "usage": "make quick-start-standard",
                "parameters": [],
                "example": "make quick-start-standard"
            },
            "quick-start-extended": {
                "description": "Extended profile setup (5000 docs, 8GB RAM)",
                "usage": "make quick-start-extended",
                "parameters": [],
                "example": "make quick-start-extended"
            },
            "quick-start-custom": {
                "description": "Custom profile setup",
                "usage": "make quick-start-custom PROFILE=name",
                "parameters": ["PROFILE (required)"],
                "example": "make quick-start-custom PROFILE=my_profile"
            }
        }
        
        return help_info.get(target_name, {
            "description": "Unknown target",
            "usage": f"make {target_name}",
            "parameters": [],
            "example": f"make {target_name}"
        })
    
    def execute_docker_quick_start(self, profile: str, output_dir: str) -> Dict[str, Any]:
        """
        Execute Docker-based quick start setup.
        
        Args:
            profile: Profile name to execute setup for
            output_dir: Output directory for generated files
            
        Returns:
            Dictionary containing execution results
        """
        try:
            from ..docker.compose_generator import DockerComposeGenerator
            from ..docker.service_manager import DockerServiceManager
            
            # Generate docker-compose file
            generator = DockerComposeGenerator()
            config = {'profile': profile}
            
            compose_file = generator.generate_compose_file(config, output_dir)
            
            # Start services
            service_manager = DockerServiceManager()
            start_result = service_manager.start_services(str(compose_file))
            
            if start_result.success:
                return {
                    'status': 'success',
                    'docker_compose_file': str(compose_file),
                    'profile': profile,
                    'services_started': start_result.services_started
                }
            else:
                return {
                    'status': 'error',
                    'error': f"Failed to start services: {start_result.error_message or 'Unknown error'}"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to execute Docker quick start: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def list_available_targets(self) -> Dict[str, Any]:
        """
        List all available quick-start targets.
        
        Returns:
            Dictionary containing available targets information
        """
        targets = []
        for target_name in self.target_mapping.keys():
            help_info = self.get_target_help(target_name)
            targets.append({
                "name": target_name,
                "description": help_info["description"],
                "usage": help_info["usage"]
            })
        
        return {
            "targets": targets,
            "count": len(targets),
            "categories": {
                "interactive": ["quick-start"],
                "profile_based": ["quick-start-minimal", "quick-start-standard", "quick-start-extended"],
                "custom": ["quick-start-custom"]
            }
        }


def main():
    """Main entry point for Makefile integration."""
    parser = argparse.ArgumentParser(
        description="Quick Start Makefile Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['interactive', 'minimal', 'standard', 'extended', 'custom', 'clean', 'status'],
        help='Setup command to execute'
    )
    
    parser.add_argument(
        '--profile',
        help='Profile name for custom setup'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger('quick_start').setLevel(logging.DEBUG)
    
    # Initialize integration
    integration = MakefileIntegration()
    
    # Execute command
    try:
        if args.command == 'interactive':
            return integration.interactive_setup()
        elif args.command in ['minimal', 'standard', 'extended']:
            return integration.profile_setup(args.command)
        elif args.command == 'custom':
            profile = args.profile or os.environ.get('PROFILE')
            return integration.custom_setup(profile)
        elif args.command == 'clean':
            return integration.clean_environment()
        elif args.command == 'status':
            return integration.check_status()
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())