"""
Interactive prompt utilities for Quick Start CLI wizard.

This module provides interactive prompt classes for gathering user input
during the CLI wizard setup process, including profile selection,
database configuration, LLM provider setup, and embedding model selection.
"""

import getpass
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from ..config.template_engine import ConfigurationTemplateEngine


@dataclass
class ProfileInfo:
    """Information about a profile for display purposes."""
    name: str
    display_name: str
    description: str
    document_count: int
    tools: List[str]
    memory_requirements: str
    disk_space: str
    estimated_setup_time: str


class ProfileSelectionPrompt:
    """Interactive prompt for profile selection."""
    
    def __init__(self):
        """Initialize the profile selection prompt."""
        self.available_profiles = {
            'minimal': ProfileInfo(
                name='quick_start_minimal',
                display_name='Minimal',
                description='Basic setup for testing and development',
                document_count=50,
                tools=['basic', 'health_check'],
                memory_requirements='2GB RAM',
                disk_space='1GB',
                estimated_setup_time='5 minutes'
            ),
            'standard': ProfileInfo(
                name='quick_start_standard',
                display_name='Standard',
                description='Balanced setup for moderate workloads',
                document_count=500,
                tools=['basic', 'health_check', 'search', 'analytics'],
                memory_requirements='4GB RAM',
                disk_space='5GB',
                estimated_setup_time='15 minutes'
            ),
            'extended': ProfileInfo(
                name='quick_start_extended',
                display_name='Extended',
                description='Full-featured setup for production use',
                document_count=5000,
                tools=['basic', 'health_check', 'search', 'analytics', 'advanced', 'monitoring'],
                memory_requirements='8GB RAM',
                disk_space='20GB',
                estimated_setup_time='30 minutes'
            )
        }
    
    def select_profile(self) -> str:
        """
        Interactive profile selection.
        
        Returns:
            Selected profile name
        """
        print("\n" + "="*60)
        print("QUICK START PROFILE SELECTION")
        print("="*60)
        print("\nAvailable profiles:\n")
        
        # Display profile options
        for i, (key, profile) in enumerate(self.available_profiles.items(), 1):
            print(f"{i}. {profile.display_name}")
            print(f"   {profile.description}")
            print(f"   Documents: {profile.document_count}")
            print(f"   Tools: {', '.join(profile.tools)}")
            print(f"   Memory: {profile.memory_requirements}")
            print(f"   Disk: {profile.disk_space}")
            print(f"   Setup time: {profile.estimated_setup_time}")
            print()
        
        print("4. Custom")
        print("   Configure a custom profile")
        print()
        
        while True:
            try:
                choice = input("Select a profile (1-4): ").strip()
                
                if choice == '1':
                    return self.available_profiles['minimal'].name
                elif choice == '2':
                    return self.available_profiles['standard'].name
                elif choice == '3':
                    return self.available_profiles['extended'].name
                elif choice == '4':
                    return self._configure_custom_profile()
                else:
                    print("Invalid choice. Please select 1-4.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                sys.exit(1)
            except EOFError:
                print("\nOperation cancelled.")
                sys.exit(1)
    
    def _configure_custom_profile(self) -> str:
        """Configure a custom profile interactively."""
        print("\n" + "-"*40)
        print("CUSTOM PROFILE CONFIGURATION")
        print("-"*40)
        
        # Get profile name
        while True:
            name = input("Profile name: ").strip()
            if name:
                break
            print("Profile name cannot be empty.")
        
        # Get document count
        while True:
            try:
                doc_count = int(input("Document count: ").strip())
                if doc_count > 0:
                    break
                print("Document count must be positive.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get tools
        print("\nAvailable tools: basic, health_check, search, analytics, advanced, monitoring")
        tools_input = input("Tools (comma-separated): ").strip()
        tools = [tool.strip() for tool in tools_input.split(',') if tool.strip()]
        
        # Confirm configuration
        print(f"\nCustom profile configuration:")
        print(f"  Name: {name}")
        print(f"  Documents: {doc_count}")
        print(f"  Tools: {', '.join(tools)}")
        
        confirm = input("\nConfirm configuration? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return name
        else:
            return self._configure_custom_profile()


class DatabaseConfigPrompt:
    """Interactive prompt for database configuration."""
    
    def configure_database(self, args) -> Dict[str, Any]:
        """
        Configure database connection interactively.
        
        Args:
            args: Command line arguments (may contain pre-filled values)
            
        Returns:
            Database configuration dictionary
        """
        print("\n" + "="*60)
        print("DATABASE CONFIGURATION")
        print("="*60)
        
        config = {}
        
        # Host
        if args and hasattr(args, 'database_host') and args.database_host:
            config['host'] = args.database_host
            print(f"Host: {config['host']} (from command line)")
        else:
            config['host'] = input("Database host [localhost]: ").strip() or 'localhost'
        
        # Port
        if args and hasattr(args, 'database_port') and args.database_port:
            config['port'] = args.database_port
            print(f"Port: {config['port']} (from command line)")
        else:
            while True:
                port_input = input("Database port [1972]: ").strip() or '1972'
                try:
                    config['port'] = int(port_input)
                    break
                except ValueError:
                    print("Please enter a valid port number.")
        
        # Namespace
        if args and hasattr(args, 'database_namespace') and args.database_namespace:
            config['namespace'] = args.database_namespace
            print(f"Namespace: {config['namespace']} (from command line)")
        else:
            config['namespace'] = input("Database namespace [USER]: ").strip() or 'USER'
        
        # Username
        if args and hasattr(args, 'database_username') and args.database_username:
            config['username'] = args.database_username
            print(f"Username: {config['username']} (from command line)")
        else:
            config['username'] = input("Database username [_SYSTEM]: ").strip() or '_SYSTEM'
        
        # Password
        if args and hasattr(args, 'database_password') and args.database_password:
            config['password'] = args.database_password
            print("Password: *** (from command line)")
        else:
            config['password'] = getpass.getpass("Database password [SYS]: ") or 'SYS'
        
        return config


class LLMProviderPrompt:
    """Interactive prompt for LLM provider configuration."""
    
    def __init__(self):
        """Initialize the LLM provider prompt."""
        self.providers = {
            '1': {'name': 'openai', 'display': 'OpenAI (GPT-3.5, GPT-4)'},
            '2': {'name': 'anthropic', 'display': 'Anthropic (Claude)'},
            '3': {'name': 'azure', 'display': 'Azure OpenAI'},
            '4': {'name': 'local', 'display': 'Local LLM (Ollama, etc.)'}
        }
    
    def configure_llm(self, args) -> Dict[str, Any]:
        """
        Configure LLM provider interactively.
        
        Args:
            args: Command line arguments (may contain pre-filled values)
            
        Returns:
            LLM configuration dictionary
        """
        print("\n" + "="*60)
        print("LLM PROVIDER CONFIGURATION")
        print("="*60)
        
        config = {}
        
        # Provider selection
        if args and hasattr(args, 'llm_provider') and args.llm_provider:
            config['provider'] = args.llm_provider
            print(f"Provider: {config['provider']} (from command line)")
        else:
            print("\nAvailable LLM providers:")
            for key, provider in self.providers.items():
                print(f"{key}. {provider['display']}")
            
            while True:
                choice = input("\nSelect LLM provider (1-4): ").strip()
                if choice in self.providers:
                    config['provider'] = self.providers[choice]['name']
                    break
                print("Invalid choice. Please select 1-4.")
        
        # API Key
        if args and hasattr(args, 'llm_api_key') and args.llm_api_key:
            config['api_key'] = args.llm_api_key
            print("API Key: *** (from command line)")
        else:
            if config['provider'] in ['openai', 'anthropic', 'azure']:
                config['api_key'] = getpass.getpass(f"{config['provider'].title()} API key: ")
        
        # Model
        if args and hasattr(args, 'llm_model') and args.llm_model:
            config['model'] = args.llm_model
            print(f"Model: {config['model']} (from command line)")
        else:
            default_models = {
                'openai': 'gpt-3.5-turbo',
                'anthropic': 'claude-3-sonnet',
                'azure': 'gpt-35-turbo',
                'local': 'llama2'
            }
            default_model = default_models.get(config['provider'], 'gpt-3.5-turbo')
            config['model'] = input(f"Model name [{default_model}]: ").strip() or default_model
        
        return config


class EmbeddingModelPrompt:
    """Interactive prompt for embedding model configuration."""
    
    def __init__(self):
        """Initialize the embedding model prompt."""
        self.providers = {
            '1': {'name': 'openai', 'display': 'OpenAI Embeddings'},
            '2': {'name': 'huggingface', 'display': 'Hugging Face'},
            '3': {'name': 'sentence-transformers', 'display': 'Sentence Transformers'},
            '4': {'name': 'local', 'display': 'Local Embeddings'}
        }
    
    def configure_embedding(self, args) -> Dict[str, Any]:
        """
        Configure embedding model interactively.
        
        Args:
            args: Command line arguments (may contain pre-filled values)
            
        Returns:
            Embedding configuration dictionary
        """
        print("\n" + "="*60)
        print("EMBEDDING MODEL CONFIGURATION")
        print("="*60)
        
        config = {}
        
        # Provider selection
        if args and hasattr(args, 'embedding_provider') and args.embedding_provider:
            config['provider'] = args.embedding_provider
            print(f"Provider: {config['provider']} (from command line)")
        else:
            print("\nAvailable embedding providers:")
            for key, provider in self.providers.items():
                print(f"{key}. {provider['display']}")
            
            while True:
                choice = input("\nSelect embedding provider (1-4): ").strip()
                if choice in self.providers:
                    config['provider'] = self.providers[choice]['name']
                    break
                print("Invalid choice. Please select 1-4.")
        
        # Model
        if args and hasattr(args, 'embedding_model') and args.embedding_model:
            config['model'] = args.embedding_model
            print(f"Model: {config['model']} (from command line)")
        else:
            default_models = {
                'openai': 'text-embedding-ada-002',
                'huggingface': 'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers': 'all-MiniLM-L6-v2',
                'local': 'local-embedding-model'
            }
            default_model = default_models.get(config['provider'], 'text-embedding-ada-002')
            config['model'] = input(f"Model name [{default_model}]: ").strip() or default_model
        
        # Auto-detect dimensions for known models
        dimension_map = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768
        }
        
        model_key = config['model'].split('/')[-1]  # Handle huggingface model names
        if model_key in dimension_map:
            config['dimensions'] = dimension_map[model_key]
            print(f"Auto-detected dimensions: {config['dimensions']}")
        
        return config