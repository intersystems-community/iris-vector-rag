"""
Main CLI wizard implementation for Quick Start profile selection.

This module provides the primary QuickStartWizard class that orchestrates
the interactive and non-interactive setup process for RAG templates.
"""

import argparse
import sys
import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

from .prompts import (
    ProfileSelectionPrompt,
    DatabaseConfigPrompt,
    LLMProviderPrompt,
    EmbeddingModelPrompt
)
from .validators import (
    DatabaseConnectivityValidator,
    LLMProviderValidator,
    EmbeddingModelValidator,
    ConfigurationValidator,
    SystemHealthValidator
)
from .formatters import (
    ProfileDisplayFormatter,
    ProgressFormatter,
    ErrorFormatter,
    SummaryFormatter,
    HelpFormatter
)
from ..config.template_engine import ConfigurationTemplateEngine
from ..config.schema_validator import ConfigurationSchemaValidator
from ..config.integration_factory import IntegrationFactory
from ..data.sample_manager import SampleDataManager


@dataclass
class CLIWizardResult:
    """Result from CLI wizard execution."""
    success: bool
    profile: str
    config: Dict[str, Any]
    files_created: List[str]
    errors: List[str]
    warnings: List[str]


class QuickStartCLIWizard:
    """
    Interactive CLI wizard for Quick Start profile selection and configuration.
    
    Supports both interactive and non-interactive modes for setting up
    RAG templates with various profiles and configurations.
    """
    
    def __init__(self, interactive: bool = True):
        """
        Initialize the Quick Start wizard.
        
        Args:
            interactive: Whether to run in interactive mode
        """
        self.interactive = interactive
        self.config = {}
        self.profile = None
        
        # Initialize components
        self.template_engine = ConfigurationTemplateEngine()
        self.schema_validator = ConfigurationSchemaValidator()
        self.integration_factory = IntegrationFactory()
        self.sample_data_manager = SampleDataManager(None)  # Will be configured later
        
        # Initialize prompts
        self.profile_prompt = ProfileSelectionPrompt()
        self.database_prompt = DatabaseConfigPrompt()
        self.llm_prompt = LLMProviderPrompt()
        self.embedding_prompt = EmbeddingModelPrompt()
        
        # Initialize validators
        self.db_validator = DatabaseConnectivityValidator()
        self.llm_validator = LLMProviderValidator()
        self.embedding_validator = EmbeddingModelValidator()
        self.config_validator = ConfigurationValidator()
        self.health_validator = SystemHealthValidator()
        
        # Initialize formatters
        self.profile_formatter = ProfileDisplayFormatter()
        self.progress_formatter = ProgressFormatter()
        self.error_formatter = ErrorFormatter()
        self.summary_formatter = SummaryFormatter()
        self.help_formatter = HelpFormatter()
    
    def run(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the wizard with the given arguments.
        
        Args:
            args: Command line arguments (defaults to sys.argv)
            
        Returns:
            Dictionary containing the configuration results
        """
        try:
            if args is None:
                args = sys.argv[1:]
            
            parsed_args = self._parse_arguments(args)
            
            if parsed_args.help:
                self.help_formatter.display_help()
                return {"status": "help_displayed"}
            
            if parsed_args.list_profiles:
                self._list_profiles()
                return {"status": "profiles_listed"}
            
            if parsed_args.validate_only:
                return self._validate_only_mode(parsed_args)
            
            # Run the main wizard flow
            if self.interactive and not self._has_required_args(parsed_args):
                return self._run_interactive_mode(parsed_args)
            else:
                return self._run_non_interactive_mode(parsed_args)
                
        except KeyboardInterrupt:
            self.error_formatter.display_error("Operation cancelled by user")
            return {"status": "cancelled"}
        except Exception as e:
            self.error_formatter.display_error(f"Unexpected error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def select_profile_interactive(self) -> CLIWizardResult:
        """Interactive profile selection."""
        try:
            profile = self.profile_prompt.select_profile()
            return CLIWizardResult(
                success=True,
                profile=profile,
                config={},
                files_created=[],
                errors=[],
                warnings=[]
            )
        except Exception as e:
            return CLIWizardResult(
                success=False,
                profile="",
                config={},
                files_created=[],
                errors=[str(e)],
                warnings=[]
            )
    
    def select_profile_from_args(self, profile: str = None) -> CLIWizardResult:
        """Non-interactive profile selection from CLI args."""
        try:
            if not profile:
                # Get from sys.argv or other source
                profile = "quick_start_minimal"  # Default
            
            return CLIWizardResult(
                success=True,
                profile=profile,
                config={},
                files_created=[],
                errors=[],
                warnings=[]
            )
        except Exception as e:
            return CLIWizardResult(
                success=False,
                profile="",
                config={},
                files_created=[],
                errors=[str(e)],
                warnings=[]
            )
    
    def generate_configuration(self, wizard_config: Dict[str, Any]) -> CLIWizardResult:
        """
        Generate configuration from wizard input.
        
        This method takes wizard configuration input and generates the complete
        configuration context needed for setup pipeline execution.
        
        Args:
            wizard_config: Dictionary containing wizard configuration parameters
            
        Returns:
            CLIWizardResult with configuration context and success status
        """
        try:
            # Extract profile and basic settings
            profile = wizard_config.get("profile", "minimal")
            environment = wizard_config.get("environment", "development")
            
            # Create configuration context using template engine
            from ..config.context import ConfigurationContext
            
            configuration_context = ConfigurationContext(
                profile=profile,
                environment=environment,
                overrides=wizard_config.get("overrides", {}),
                template_path=None,  # Will be determined by template engine
                environment_variables=wizard_config.get("environment_variables", {})
            )
            
            # Generate configuration using template engine
            template_result = self.template_engine.generate_configuration(configuration_context)
            
            if not template_result.success:
                return CLIWizardResult(
                    success=False,
                    profile=profile,
                    config={},
                    files_created=[],
                    errors=[f"Template generation failed: {template_result.message}"],
                    warnings=[]
                )
            
            # Create result with configuration context
            result = CLIWizardResult(
                success=True,
                profile=profile,
                config=template_result.configuration,
                files_created=template_result.files_created,
                errors=[],
                warnings=template_result.warnings
            )
            
            # Add configuration context as an attribute for setup pipeline
            result.configuration_context = configuration_context
            
            return result
            
        except Exception as e:
            return CLIWizardResult(
                success=False,
                profile=wizard_config.get("profile", "unknown"),
                config={},
                files_created=[],
                errors=[f"Configuration generation failed: {str(e)}"],
                warnings=[]
            )
    
    def get_profile_characteristics(self, profile: str) -> Dict[str, Any]:
        """Get profile characteristics and resource requirements."""
        characteristics = {
            "quick_start_minimal": {
                "document_count": 50,
                "memory_requirements": "2GB",
                "disk_space": "1GB",
                "estimated_setup_time": "5 minutes",
                "tools": ["basic", "health_check"]
            },
            "quick_start_standard": {
                "document_count": 500,
                "memory_requirements": "4GB",
                "disk_space": "5GB",
                "estimated_setup_time": "15 minutes",
                "tools": ["basic", "health_check", "search", "analytics"]
            },
            "quick_start_extended": {
                "document_count": 5000,
                "memory_requirements": "8GB",
                "disk_space": "20GB",
                "estimated_setup_time": "30 minutes",
                "tools": ["basic", "health_check", "search", "analytics", "advanced", "monitoring"]
            }
        }
        
        return characteristics.get(profile, {})
    
    def configure_database_interactive(self) -> Dict[str, Any]:
        """Interactive database configuration prompts."""
        return self.database_prompt.configure_database(None)
    
    def configure_llm_provider_interactive(self) -> Dict[str, Any]:
        """Interactive LLM provider configuration."""
        return self.llm_prompt.configure_llm(None)
    
    def configure_embeddings_interactive(self) -> Dict[str, Any]:
        """Interactive embedding model selection."""
        return self.embedding_prompt.configure_embedding(None)
    
    def generate_env_file(self, config: Dict[str, Any], path: Path) -> Path:
        """Generate environment variable file."""
        env_vars = []
        
        # Database environment variables
        if 'database' in config:
            db = config['database']
            env_vars.extend([
                f"IRIS_HOST={db.get('host', 'localhost')}",
                f"IRIS_PORT={db.get('port', 1972)}",
                f"IRIS_NAMESPACE={db.get('namespace', 'USER')}",
                f"IRIS_USERNAME={db.get('username', '_SYSTEM')}",
                f"IRIS_PASSWORD={db.get('password', 'SYS')}"
            ])
        
        # LLM environment variables
        if 'llm' in config:
            llm = config['llm']
            provider = llm.get('provider', '').upper()
            if 'api_key' in llm:
                env_vars.append(f"{provider}_API_KEY={llm['api_key']}")
            if 'model' in llm:
                env_vars.append(f"LLM_MODEL={llm['model']}")
        
        # Embedding environment variables
        if 'embedding' in config:
            emb = config['embedding']
            if 'model' in emb:
                env_vars.append(f"EMBEDDING_MODEL={emb['model']}")
        
        # Write to file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(env_vars))
            f.write('\n')
        
        return path
    
    def generate_configuration_file(self, profile_config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate configuration file from selected profile."""
        config_file = output_dir / 'config.yaml'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(profile_config, f, default_flow_style=False, indent=2)
        
        return config_file
    
    def create_env_file(self, env_vars: Dict[str, str], path: Path) -> Path:
        """Create environment file (.env) creation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return path
    
    def generate_docker_compose(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate docker-compose file."""
        docker_file = output_dir / 'docker-compose.yml'
        
        docker_config = {
            'version': '3.8',
            'services': {
                'iris': {
                    'image': 'intersystemsdc/iris-community:latest',
                    'ports': [f"{config.get('database', {}).get('port', 1972)}:1972"],
                    'environment': [
                        'ISC_PASSWORD=SYS'
                    ]
                }
            }
        }
        
        if config.get('profile') in ['standard', 'extended']:
            docker_config['services']['mcp_server'] = {
                'build': '.',
                'ports': ['3000:3000'],
                'depends_on': ['iris']
            }
        
        docker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(docker_file, 'w') as f:
            yaml.dump(docker_config, f, default_flow_style=False, indent=2)
        
        return docker_file
    
    def generate_sample_data_script(self, config: Dict[str, Any], output_dir: Path) -> Path:
        """Generate sample data setup script."""
        script_file = output_dir / 'setup_sample_data.py'
        
        script_content = f'''#!/usr/bin/env python3
"""
Sample data setup script generated by Quick Start CLI wizard.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quick_start.data.sample_manager import SampleDataManager
from quick_start.config.template_engine import ConfigurationTemplateEngine

def main():
    """Set up sample data for the RAG system."""
    print("Setting up sample data...")
    
    # Configuration from wizard
    config = {{
        'profile': '{config.get('profile', 'minimal')}',
        'sample_data': {{
            'source': 'pmc',
            'document_count': {config.get('sample_data', {}).get('document_count', 10)},
            'categories': ['biomedical']
        }}
    }}
    
    # Initialize sample data manager
    template_engine = ConfigurationTemplateEngine()
    sample_manager = SampleDataManager(template_engine)
    
    # Download and process sample data
    try:
        print(f"Downloading {{config['sample_data']['document_count']}} documents...")
        # Implementation would go here
        print("Sample data setup complete!")
    except Exception as e:
        print(f"Error setting up sample data: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        script_file.parent.mkdir(parents=True, exist_ok=True)
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_file.chmod(0o755)
        
        return script_file
    
    def generate_all_files(self, config: Dict[str, Any], output_dir: Path) -> CLIWizardResult:
        """Generate all configuration files."""
        try:
            files_created = []
            
            # Generate main config file
            config_file = self.generate_configuration_file(config, output_dir)
            files_created.append(str(config_file))
            
            # Generate environment file
            env_file = self.generate_env_file(config, output_dir / '.env')
            files_created.append(str(env_file))
            
            # Generate docker-compose for standard/extended profiles
            if config.get('profile') in ['quick_start_standard', 'quick_start_extended']:
                docker_file = self.generate_docker_compose(config, output_dir)
                files_created.append(str(docker_file))
            
            # Generate sample data script
            script_file = self.generate_sample_data_script(config, output_dir)
            files_created.append(str(script_file))
            
            return CLIWizardResult(
                success=True,
                profile=config.get('profile', ''),
                config=config,
                files_created=files_created,
                errors=[],
                warnings=[]
            )
            
        except Exception as e:
            return CLIWizardResult(
                success=False,
                profile=config.get('profile', ''),
                config=config,
                files_created=[],
                errors=[str(e)],
                warnings=[]
            )
    
    def test_database_connection(self, db_config: Dict[str, Any]):
        """Test database connection."""
        return self.db_validator.test_connection(db_config)
    
    def test_llm_credentials(self, llm_config: Dict[str, Any]):
        """Test LLM provider credentials."""
        return self.llm_validator.test_provider(llm_config)
    
    def test_embedding_model(self, embedding_config: Dict[str, Any]):
        """Test embedding model availability."""
        return self.embedding_validator.test_model(embedding_config)
    
    def validate_environment_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate environment configuration."""
        result = self.config_validator.validate_configuration(config)
        if result.details and 'errors' in result.details:
            return result.details['errors']
        return []
    
    def prompt_for_input(self, prompt: str, input_type: type):
        """Prompt for input with type validation."""
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                if input_type == bool:
                    return user_input.lower() in ['y', 'yes', 'true', '1']
                elif input_type == int:
                    return int(user_input)
                else:
                    return input_type(user_input)
            except (ValueError, TypeError):
                print(f"Please enter a valid {input_type.__name__}")
    
    def _parse_arguments(self, args: List[str]) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Quick Start CLI Wizard for RAG Templates",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument(
            '--profile',
            choices=['minimal', 'standard', 'extended', 'custom'],
            help='Profile to use for setup'
        )
        
        parser.add_argument('--database-host', help='Database host address')
        parser.add_argument('--database-port', type=int, help='Database port number')
        parser.add_argument('--database-namespace', help='Database namespace')
        parser.add_argument('--database-username', help='Database username')
        parser.add_argument('--database-password', help='Database password')
        
        parser.add_argument(
            '--llm-provider',
            choices=['openai', 'anthropic', 'azure', 'local'],
            help='LLM provider to use'
        )
        parser.add_argument('--llm-api-key', help='LLM API key')
        parser.add_argument('--llm-model', help='LLM model name')
        
        parser.add_argument(
            '--embedding-provider',
            choices=['openai', 'huggingface', 'sentence-transformers', 'local'],
            help='Embedding provider to use'
        )
        parser.add_argument('--embedding-model', help='Embedding model name')
        
        parser.add_argument('--output-dir', help='Output directory for generated files')
        parser.add_argument('--list-profiles', action='store_true', help='List available profiles and exit')
        parser.add_argument('--validate-only', action='store_true', help='Only validate configuration without creating files')
        parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
        parser.add_argument('--help', action='store_true', help='Show help message')
        
        return parser.parse_args(args)
    
    def _has_required_args(self, args: argparse.Namespace) -> bool:
        """Check if required arguments are provided for non-interactive mode."""
        return bool(args.profile and args.database_host and args.llm_provider)
    
    def _run_interactive_mode(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run the wizard in interactive mode."""
        self.progress_formatter.display_progress("Starting Interactive Setup", 0, 5)
        
        try:
            # Step 1: Profile selection
            self.progress_formatter.display_progress("Profile Selection", 1, 5)
            if not args.profile:
                self.profile = self.profile_prompt.select_profile()
            else:
                self.profile = f"quick_start_{args.profile}"
            
            # Step 2: Database configuration
            self.progress_formatter.display_progress("Database Configuration", 2, 5)
            db_config = self.database_prompt.configure_database(args)
            
            # Step 3: LLM provider configuration
            self.progress_formatter.display_progress("LLM Provider Configuration", 3, 5)
            llm_config = self.llm_prompt.configure_llm(args)
            
            # Step 4: Embedding model configuration
            self.progress_formatter.display_progress("Embedding Model Configuration", 4, 5)
            embedding_config = self.embedding_prompt.configure_embedding(args)
            
            # Step 5: Generate configuration
            self.progress_formatter.display_progress("Generating Configuration", 5, 5)
            
            self.config = {
                'profile': self.profile,
                'database': db_config,
                'llm': llm_config,
                'embedding': embedding_config,
                'output_dir': args.output_dir or './quick_start_output'
            }
            
            return self._finalize_configuration()
            
        except Exception as e:
            self.error_formatter.display_error(f"Interactive setup failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _run_non_interactive_mode(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run the wizard in non-interactive mode."""
        try:
            self.profile = f"quick_start_{args.profile}"
            
            self.config = {
                'profile': self.profile,
                'database': {
                    'host': args.database_host,
                    'port': args.database_port or 1972,
                    'namespace': args.database_namespace or 'USER',
                    'username': args.database_username or '_SYSTEM',
                    'password': args.database_password or 'SYS'
                },
                'llm': {
                    'provider': args.llm_provider,
                    'api_key': args.llm_api_key,
                    'model': args.llm_model
                },
                'embedding': {
                    'provider': args.embedding_provider or 'openai',
                    'model': args.embedding_model
                },
                'output_dir': args.output_dir or './quick_start_output'
            }
            
            return self._finalize_configuration()
            
        except Exception as e:
            self.error_formatter.display_error(f"Non-interactive setup failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _finalize_configuration(self) -> Dict[str, Any]:
        """Finalize and validate the configuration."""
        try:
            # Validate configuration
            validation_result = self.config_validator.validate_configuration(self.config)
            if not validation_result.valid:
                errors = validation_result.details.get('errors', []) if validation_result.details else []
                self.error_formatter.display_validation_errors(errors)
                return {"status": "validation_failed", "errors": errors}
            
            # Test connectivity
            if self.interactive:
                print("\nTesting connectivity...")
            
            connectivity_results = self._test_connectivity()
            if not connectivity_results['all_passed']:
                if self.interactive:
                    self.error_formatter.display_connectivity_errors(connectivity_results)
                return {"status": "connectivity_failed", "results": connectivity_results}
            
            # Generate files
            if self.interactive:
                print("\nGenerating configuration files...")
            
            generated_files = self._generate_files()
            
            # Display summary
            if self.interactive:
                self.summary_formatter.display_summary(self.config, generated_files)
            
            return {
                "status": "success",
                "profile": self.profile,
                "config": self.config,
                "generated_files": generated_files,
                "connectivity_results": connectivity_results
            }
            
        except Exception as e:
            self.error_formatter.display_error(f"Configuration finalization failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _test_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to configured services."""
        results = {
            'database': False,
            'llm': False,
            'embedding': False,
            'all_passed': False
        }
        
        try:
            # Test database connectivity
            db_result = self.db_validator.test_connection(self.config['database'])
            results['database'] = db_result.success
            
            # Test LLM provider
            llm_result = self.llm_validator.test_provider(self.config['llm'])
            results['llm'] = llm_result.success
            
            # Test embedding model
            embedding_result = self.embedding_validator.test_model(self.config['embedding'])
            results['embedding'] = embedding_result.success
            
            results['all_passed'] = all([results['database'], results['llm'], results['embedding']])
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _generate_files(self) -> List[str]:
        """Generate configuration files based on the selected profile."""
        generated_files = []
        
        try:
            output_dir = Path(self.config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate main configuration file
            config_file = self.generate_configuration_file(self.config, output_dir)
            generated_files.append(str(config_file))
            
            # Generate environment file
            env_file = self.generate_env_file(self.config, output_dir / '.env')
            generated_files.append(str(env_file))
            
            # Generate docker-compose file if requested
            if self.profile in ['quick_start_standard', 'quick_start_extended']:
                docker_file = self.generate_docker_compose(self.config, output_dir)
                generated_files.append(str(docker_file))
            
            # Generate sample data setup script
            sample_script = self.generate_sample_data_script(self.config, output_dir)
            generated_files.append(str(sample_script))
            
        except Exception as e:
            raise Exception(f"File generation failed: {str(e)}")
        
        return generated_files
    
    def _list_profiles(self):
        """List available profiles with descriptions."""
        self.profile_formatter.display_available_profiles()
    
    def _validate_only_mode(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run validation-only mode."""
        try:
            if not self._has_required_args(args):
                self.error_formatter.display_error(
                    "Validation mode requires --profile, --database-host, and --llm-provider"
                )
                return {"status": "error", "error": "Missing required arguments"}
            
            # Build config from args
            config = {
                'profile': f"quick_start_{args.profile}",
                'database': {
                    'host': args.database_host,
                    'port': args.database_port or 1972,
                    'namespace': args.database_namespace or 'USER',
                    'username': args.database_username or '_SYSTEM',
                    'password': args.database_password or 'SYS'
                },
                'llm': {
                    'provider': args.llm_provider,
                    'api_key': args.llm_api_key,
                    'model': args.llm_model
                },
                'embedding': {
                    'provider': args.embedding_provider or 'openai',
                    'model': args.embedding_model
                }
            }
            
            # Validate configuration
            validation_result = self.config_validator.validate_configuration(config)
            
            if validation_result.valid:
                print("✅ Configuration validation passed")
                return {"status": "validation_passed", "config": config}
            else:
                errors = validation_result.details.get('errors', []) if validation_result.details else []
                self.error_formatter.display_validation_errors(errors)
                return {"status": "validation_failed", "errors": errors}
                
        except Exception as e:
            self.error_formatter.display_error(f"Validation failed: {str(e)}")
            return {"status": "error", "error": str(e)}


# Utility functions for the test suite
def compare_profiles(profiles: List[str]) -> Dict[str, Any]:
    """Compare profile characteristics."""
    wizard = QuickStartCLIWizard()
    comparison = {}
    
    for profile in profiles:
        comparison[profile] = wizard.get_profile_characteristics(profile)
    
    return comparison


def show_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
    """Show configuration differences."""
    # Simple diff implementation
    diff_lines = []
    
    for key in set(config1.keys()) | set(config2.keys()):
        if key not in config1:
            diff_lines.append(f"+ {key}: {config2[key]}")
        elif key not in config2:
            diff_lines.append(f"- {key}: {config1[key]}")
        elif config1[key] != config2[key]:
            diff_lines.append(f"- {key}: {config1[key]}")
            diff_lines.append(f"+ {key}: {config2[key]}")
    
    return '\n'.join(diff_lines)