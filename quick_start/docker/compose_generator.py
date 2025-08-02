"""
Docker-compose file generator for Quick Start system.

This module provides the DockerComposeGenerator class that generates
docker-compose.yml files based on configuration profiles and integrates
with the existing Quick Start template system.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from quick_start.cli.wizard import CLIWizardResult
from quick_start.config.template_engine import ConfigurationTemplateEngine
from quick_start.data.sample_manager import SampleDataManager
from .container_config import ContainerConfigManager
from .volume_manager import VolumeManager

logger = logging.getLogger(__name__)


class DockerComposeGenerator:
    """
    Generator for Docker-compose files based on Quick Start profiles.
    
    Provides profile-based docker-compose.yml generation with integration
    to existing Quick Start components and template system.
    """
    
    def __init__(self, template_engine: Optional[ConfigurationTemplateEngine] = None,
                 container_config_manager: Optional[ContainerConfigManager] = None):
        """
        Initialize the Docker-compose generator.
        
        Args:
            template_engine: Configuration template engine for variable substitution
            container_config_manager: Manager for container configurations
        """
        self.template_engine = template_engine or ConfigurationTemplateEngine()
        self.container_config_manager = container_config_manager or ContainerConfigManager()
        self.volume_manager = VolumeManager()
        
        # Supported profiles
        self.supported_profiles = [
            'minimal', 'standard', 'extended', 'development', 
            'production', 'testing', 'custom'
        ]
    
    def generate_compose_file(self, config: Dict[str, Any],
                            output_dir: Path) -> Path:
        """
        Generate docker-compose.yml file for the given configuration.
        
        Args:
            config: Configuration dictionary containing profile and settings
            output_dir: Directory where the compose file should be created
            
        Returns:
            Path to the generated docker-compose.yml file
        """
        profile = config.get('profile', 'minimal')
        
        if not self.validate_profile(profile):
            raise ValueError(f"Invalid profile: {profile}")
        
        # Generate compose data
        compose_data = self.generate_compose_data(config)
        
        # Write to file
        compose_file = output_dir / 'docker-compose.quick-start.yml'
        
        # Handle template variables if present
        if 'template_variables' in config and self.template_engine:
            # Convert to YAML string first
            yaml_content = yaml.dump(compose_data, default_flow_style=False, indent=2)
            
            # Apply template variables (preserve them as-is)
            template_vars = config['template_variables']
            for var_name, var_value in template_vars.items():
                # Replace actual values with template variables
                if var_name == 'iris_password' and var_value.startswith('${'):
                    yaml_content = yaml_content.replace('ISC_PASSWORD=SYS', f'ISC_PASSWORD={var_value}')
                elif var_name == 'app_port' and var_value.startswith('${'):
                    yaml_content = yaml_content.replace('8000:8000', f'{var_value}:{var_value}')
            
            with open(compose_file, 'w') as f:
                f.write(yaml_content)
        else:
            with open(compose_file, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated docker-compose file: {compose_file}")
        return compose_file
    
    def generate_compose_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the docker-compose data structure.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing docker-compose structure
        """
        profile = config.get('profile', 'minimal')
        
        # Base compose structure
        compose_data = {
            'version': '3.8',
            'services': {},
            'volumes': self.volume_manager.get_volume_config(profile),
            'networks': self.volume_manager.get_network_config(profile)
        }
        
        # Generate services based on profile
        if profile in ['minimal', 'standard', 'extended', 'development', 'production', 'testing']:
            compose_data['services'] = self._generate_profile_services(config)
        elif profile == 'custom':
            compose_data['services'] = self._generate_custom_services(config)
        
        return compose_data
    
    def _generate_profile_services(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate services for standard profiles."""
        profile = config.get('profile', 'minimal')
        services = {}
        
        # IRIS database service (all profiles)
        if profile == 'testing':
            # For testing profile, use iris_test service name
            services['iris_test'] = self.container_config_manager.generate_iris_config(config)
        else:
            services['iris'] = self.container_config_manager.generate_iris_config(config)
        
        # RAG application service (all profiles)
        services['rag_app'] = self.container_config_manager.generate_rag_app_config(config)
        
        # MCP server (standard, extended, development, production)
        if profile in ['standard', 'extended', 'development', 'production']:
            services['mcp_server'] = self.container_config_manager.generate_mcp_server_config(config)
        
        # Extended services (extended, production)
        if profile in ['extended', 'production']:
            services['nginx'] = self.container_config_manager.generate_nginx_config(config)
            services['prometheus'] = self.container_config_manager.generate_prometheus_config(config)
            
            # Add Grafana for extended profile
            if profile == 'extended':
                services['grafana'] = self.container_config_manager.generate_grafana_config(config)
        
        return services
    
    def _generate_custom_services(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate services for custom profile."""
        docker_config = config.get('docker', {})
        custom_services = docker_config.get('services', {})
        
        # If no custom services defined, fall back to minimal
        if not custom_services:
            return self._generate_profile_services({'profile': 'minimal'})
        
        return custom_services
    
    def generate_from_wizard_result(self, wizard_result: CLIWizardResult,
                                  output_dir: Path) -> Path:
        """
        Generate docker-compose file from CLI wizard result.
        
        Args:
            wizard_result: Result from CLI wizard execution
            output_dir: Directory for output file
            
        Returns:
            Path to generated compose file
        """
        if not wizard_result.success:
            raise ValueError("Cannot generate from failed wizard result")
        
        return self.generate_compose_file(wizard_result.config, output_dir)
    
    def generate_with_sample_data(self, config: Dict[str, Any],
                                sample_manager: SampleDataManager,
                                output_dir: Path) -> Path:
        """
        Generate docker-compose file with sample data integration.
        
        Args:
            config: Configuration dictionary
            sample_manager: Sample data manager instance
            output_dir: Directory for output file
            
        Returns:
            Path to generated compose file
        """
        # Enhance config with sample data settings
        enhanced_config = config.copy()
        sample_data_config = config.get('sample_data', {})
        
        if sample_data_config:
            # Add volume mounts for sample data
            enhanced_config.setdefault('docker', {})
            enhanced_config['docker']['sample_data_enabled'] = True
            enhanced_config['docker']['sample_data_source'] = sample_data_config.get('source', 'pmc')
        
        return self.generate_compose_file(enhanced_config, output_dir)
    
    def get_supported_profiles(self) -> List[str]:
        """
        Get list of supported profiles.
        
        Returns:
            List of supported profile names
        """
        return self.supported_profiles.copy()
    
    def validate_profile(self, profile: str) -> bool:
        """
        Validate if profile is supported.
        
        Args:
            profile: Profile name to validate
            
        Returns:
            True if profile is supported, False otherwise
        """
        return profile in self.supported_profiles
    
    def optimize_resources(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource allocation for the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Optimized configuration
        """
        optimized_config = config.copy()
        profile = config.get('profile', 'minimal')
        
        # Apply resource optimizations based on profile
        resource_limits = {}
        memory_optimization = {}
        
        if profile == 'minimal':
            resource_limits = {
                'iris': {'memory': '2g', 'cpus': '1.0'},
                'rag_app': {'memory': '1g', 'cpus': '0.5'}
            }
            memory_optimization = {
                'batch_size': 16,
                'max_workers': 2,
                'memory_limit': '2G'
            }
        elif profile == 'standard':
            resource_limits = {
                'iris': {'memory': '4g', 'cpus': '2.0'},
                'rag_app': {'memory': '2g', 'cpus': '1.0'}
            }
            memory_optimization = {
                'batch_size': 32,
                'max_workers': 4,
                'memory_limit': '4G'
            }
        elif profile == 'extended':
            resource_limits = {
                'iris': {'memory': '2g', 'cpus': '4.0'},
                'rag_app': {'memory': '4g', 'cpus': '2.0'}
            }
            memory_optimization = {
                'batch_size': 64,
                'max_workers': 8,
                'memory_limit': '8G'
            }
        
        optimized_config['resource_limits'] = resource_limits
        optimized_config['memory_optimization'] = memory_optimization
        
        return optimized_config