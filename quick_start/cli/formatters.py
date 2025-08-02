"""
Output formatting and display utilities for Quick Start CLI wizard.

This module provides formatting classes for displaying profile information,
progress indicators, error messages, configuration summaries, and help text
in a user-friendly manner.
"""

import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..cli.prompts import ProfileInfo


class ProfileDisplayFormatter:
    """Formatter for displaying profile information and characteristics."""
    
    def display_available_profiles(self):
        """Display all available profiles with their characteristics."""
        print("\n" + "="*70)
        print("AVAILABLE QUICK START PROFILES")
        print("="*70)
        
        profiles = {
            'minimal': {
                'name': 'Minimal Profile',
                'description': 'Basic setup for testing and development',
                'document_count': '≤ 50 documents',
                'memory_requirements': '2GB RAM',
                'disk_space': '1GB storage',
                'estimated_setup_time': '5 minutes',
                'tools': ['basic', 'health_check'],
                'use_cases': ['Development', 'Testing', 'Learning']
            },
            'standard': {
                'name': 'Standard Profile',
                'description': 'Balanced setup for moderate workloads',
                'document_count': '≤ 500 documents',
                'memory_requirements': '4GB RAM',
                'disk_space': '5GB storage',
                'estimated_setup_time': '15 minutes',
                'tools': ['basic', 'health_check', 'search', 'analytics'],
                'use_cases': ['Small teams', 'Prototyping', 'Demos']
            },
            'extended': {
                'name': 'Extended Profile',
                'description': 'Full-featured setup for production use',
                'document_count': '≤ 5000 documents',
                'memory_requirements': '8GB RAM',
                'disk_space': '20GB storage',
                'estimated_setup_time': '30 minutes',
                'tools': ['basic', 'health_check', 'search', 'analytics', 'advanced', 'monitoring'],
                'use_cases': ['Production', 'Large datasets', 'Enterprise']
            }
        }
        
        for profile_key, profile in profiles.items():
            print(f"\n📋 {profile['name']}")
            print(f"   {profile['description']}")
            print(f"   📊 Documents: {profile['document_count']}")
            print(f"   💾 Memory: {profile['memory_requirements']}")
            print(f"   💿 Storage: {profile['disk_space']}")
            print(f"   ⏱️  Setup time: {profile['estimated_setup_time']}")
            print(f"   🛠️  Tools: {', '.join(profile['tools'])}")
            print(f"   🎯 Use cases: {', '.join(profile['use_cases'])}")
        
        print(f"\n📝 Custom Profile")
        print(f"   Configure your own custom profile")
        print(f"   🎛️  Fully customizable parameters")
        print(f"   ⚙️  Advanced configuration options")
        print()
    
    def display_profile_comparison(self, profiles: List[str]):
        """Display a comparison table of selected profiles."""
        print("\n" + "="*80)
        print("PROFILE COMPARISON")
        print("="*80)
        
        # This would show a detailed comparison table
        # Implementation would depend on the specific profiles being compared
        print("Profile comparison feature coming soon...")
        print()


class ProgressFormatter:
    """Formatter for progress indicators and status updates."""
    
    def display_progress(self, task: str, current: int, total: int, details: str = ""):
        """
        Display progress for a task.
        
        Args:
            task: Name of the current task
            current: Current step number
            total: Total number of steps
            details: Additional details about the current step
        """
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * current // total) if total > 0 else 0
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r🔄 {task} [{bar}] {percentage:.1f}% ({current}/{total})", end='')
        if details:
            print(f" - {details}", end='')
        
        if current == total:
            print(" ✅")
        else:
            print("", end='', flush=True)
    
    def display_step_progress(self, step_name: str, step_number: int, total_steps: int):
        """Display progress for individual steps."""
        print(f"\n[{step_number}/{total_steps}] {step_name}")
        print("-" * (len(step_name) + 10))
    
    def display_spinner(self, message: str):
        """Display a simple spinner for ongoing operations."""
        # In a real implementation, this would show an animated spinner
        print(f"⏳ {message}...")


class ErrorFormatter:
    """Formatter for error messages and recovery options."""
    
    def display_error(self, error_message: str, details: Optional[str] = None):
        """
        Display an error message.
        
        Args:
            error_message: Main error message
            details: Additional error details
        """
        print(f"\n❌ Error: {error_message}", file=sys.stderr)
        if details:
            print(f"   Details: {details}", file=sys.stderr)
        print()
    
    def display_validation_errors(self, errors: List[str]):
        """Display validation errors in a formatted list."""
        print(f"\n❌ Configuration Validation Failed:", file=sys.stderr)
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}", file=sys.stderr)
        print()
    
    def display_connectivity_errors(self, results: Dict[str, Any]):
        """Display connectivity test results with errors highlighted."""
        print(f"\n🔌 Connectivity Test Results:")
        
        services = ['database', 'llm', 'embedding']
        for service in services:
            if service in results:
                status = "✅" if results[service] else "❌"
                print(f"   {status} {service.title()}: {'Connected' if results[service] else 'Failed'}")
        
        if not results.get('all_passed', False):
            print(f"\n💡 Troubleshooting Tips:")
            if not results.get('database', False):
                print(f"   • Check database host and port")
                print(f"   • Verify database credentials")
                print(f"   • Ensure database is running")
            
            if not results.get('llm', False):
                print(f"   • Verify API key is correct")
                print(f"   • Check internet connectivity")
                print(f"   • Confirm API quota/limits")
            
            if not results.get('embedding', False):
                print(f"   • Check embedding model availability")
                print(f"   • Verify provider credentials")
        print()
    
    def display_recovery_options(self, error_type: str):
        """Display recovery options for different error types."""
        recovery_options = {
            'network': [
                "Check your internet connection",
                "Verify firewall settings",
                "Try again in a few moments"
            ],
            'credentials': [
                "Verify your API keys",
                "Check credential format",
                "Ensure account has proper permissions"
            ],
            'configuration': [
                "Review configuration parameters",
                "Check for typos in settings",
                "Validate required fields"
            ]
        }
        
        if error_type in recovery_options:
            print(f"\n🔧 Suggested Solutions:")
            for option in recovery_options[error_type]:
                print(f"   • {option}")
            print()


class SummaryFormatter:
    """Formatter for configuration summaries and final results."""
    
    def display_summary(self, config: Dict[str, Any], generated_files: List[str]):
        """
        Display a summary of the configuration and generated files.
        
        Args:
            config: Final configuration dictionary
            generated_files: List of generated file paths
        """
        print("\n" + "="*70)
        print("🎉 QUICK START SETUP COMPLETE!")
        print("="*70)
        
        # Configuration summary
        print(f"\n📋 Configuration Summary:")
        print(f"   Profile: {config.get('profile', 'Unknown')}")
        
        if 'database' in config:
            db = config['database']
            print(f"   Database: {db.get('host', 'localhost')}:{db.get('port', 1972)}")
        
        if 'llm' in config:
            llm = config['llm']
            print(f"   LLM: {llm.get('provider', 'Unknown')} ({llm.get('model', 'default')})")
        
        if 'embedding' in config:
            emb = config['embedding']
            print(f"   Embeddings: {emb.get('provider', 'Unknown')} ({emb.get('model', 'default')})")
        
        # Generated files
        print(f"\n📁 Generated Files:")
        for file_path in generated_files:
            print(f"   ✅ {file_path}")
        
        # Next steps
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review the generated configuration files")
        print(f"   2. Run the sample data setup script")
        print(f"   3. Test your RAG system")
        print(f"   4. Explore the available tools and features")
        
        print(f"\n📚 Documentation:")
        print(f"   • Configuration guide: ./docs/configuration.md")
        print(f"   • API reference: ./docs/api-reference.md")
        print(f"   • Troubleshooting: ./docs/troubleshooting.md")
        print()
    
    def display_configuration_preview(self, config: Dict[str, Any]):
        """Display a preview of the configuration before finalizing."""
        print(f"\n📋 Configuration Preview:")
        print(f"   Profile: {config.get('profile', 'Unknown')}")
        print(f"   Output Directory: {config.get('output_dir', './quick_start_output')}")
        
        if 'database' in config:
            db = config['database']
            print(f"   Database: {db.get('host')}:{db.get('port')} ({db.get('namespace')})")
        
        if 'llm' in config:
            llm = config['llm']
            print(f"   LLM: {llm.get('provider')} - {llm.get('model')}")
        
        if 'embedding' in config:
            emb = config['embedding']
            print(f"   Embeddings: {emb.get('provider')} - {emb.get('model')}")
        
        print()


class HelpFormatter:
    """Formatter for help messages and usage information."""
    
    def display_help(self):
        """Display comprehensive help information."""
        print("\n" + "="*70)
        print("QUICK START CLI WIZARD HELP")
        print("="*70)
        
        print(f"\nDESCRIPTION:")
        print(f"   Interactive CLI wizard for setting up RAG templates with")
        print(f"   profile-based configuration and automated validation.")
        
        print(f"\nUSAGE:")
        print(f"   Interactive mode:")
        print(f"     python -m quick_start.cli.wizard")
        print(f"   ")
        print(f"   Non-interactive mode:")
        print(f"     python -m quick_start.cli.wizard --profile PROFILE [OPTIONS]")
        
        print(f"\nPROFILES:")
        print(f"   minimal    Basic setup for testing (≤50 docs, 2GB RAM)")
        print(f"   standard   Balanced setup for moderate use (≤500 docs, 4GB RAM)")
        print(f"   extended   Full-featured production setup (≤5000 docs, 8GB RAM)")
        print(f"   custom     Configure your own custom profile")
        
        print(f"\nOPTIONS:")
        print(f"   --profile PROFILE              Profile to use (minimal|standard|extended|custom)")
        print(f"   --database-host HOST           Database host address")
        print(f"   --database-port PORT           Database port number")
        print(f"   --database-namespace NS        Database namespace")
        print(f"   --database-username USER       Database username")
        print(f"   --database-password PASS       Database password")
        print(f"   --llm-provider PROVIDER        LLM provider (openai|anthropic|azure|local)")
        print(f"   --llm-api-key KEY              LLM API key")
        print(f"   --llm-model MODEL              LLM model name")
        print(f"   --embedding-provider PROVIDER  Embedding provider")
        print(f"   --embedding-model MODEL        Embedding model name")
        print(f"   --output-dir DIR               Output directory for generated files")
        print(f"   --list-profiles                List available profiles and exit")
        print(f"   --validate-only                Only validate configuration")
        print(f"   --non-interactive              Run in non-interactive mode")
        print(f"   --help                         Show this help message")
        
        print(f"\nEXAMPLES:")
        print(f"   # Interactive setup")
        print(f"   python -m quick_start.cli.wizard")
        print(f"   ")
        print(f"   # Quick standard setup")
        print(f"   python -m quick_start.cli.wizard --profile standard \\")
        print(f"     --database-host localhost --llm-provider openai")
        print(f"   ")
        print(f"   # List available profiles")
        print(f"   python -m quick_start.cli.wizard --list-profiles")
        print(f"   ")
        print(f"   # Validate configuration only")
        print(f"   python -m quick_start.cli.wizard --profile minimal \\")
        print(f"     --database-host localhost --llm-provider openai --validate-only")
        
        print(f"\nFILES GENERATED:")
        print(f"   config.yaml           Main configuration file")
        print(f"   .env                  Environment variables")
        print(f"   docker-compose.yml    Docker setup (standard/extended profiles)")
        print(f"   setup_sample_data.py  Sample data setup script")
        
        print(f"\nSUPPORT:")
        print(f"   Documentation: ./docs/")
        print(f"   Issues: https://github.com/your-repo/issues")
        print(f"   Community: https://discord.gg/your-community")
        print()
    
    def display_usage(self):
        """Display brief usage information."""
        print(f"\nUsage: python -m quick_start.cli.wizard [OPTIONS]")
        print(f"       python -m quick_start.cli.wizard --help")
        print()
    
    def display_version(self):
        """Display version information."""
        print(f"Quick Start CLI Wizard v2024.1")
        print(f"RAG Templates Framework")
        print()