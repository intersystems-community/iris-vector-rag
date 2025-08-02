"""
Profile Management for Quick Start

This module provides profile management functionality for the Quick Start system,
including loading, validating, and managing different setup profiles.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class ProfileManager:
    """Manages Quick Start setup profiles."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles_dir = Path(__file__).parent / 'templates'
        self.schemas_dir = Path(__file__).parent / 'schemas'
        
    def profile_exists(self, profile_name: str) -> bool:
        """Check if a profile exists."""
        profile_file = self.profiles_dir / f'quick_start_{profile_name}.yaml'
        return profile_file.exists()
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load a profile configuration."""
        profile_file = self.profiles_dir / f'quick_start_{profile_name}.yaml'
        
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_file}")
        
        try:
            with open(profile_file, 'r') as f:
                profile_config = yaml.safe_load(f)
            
            # Add profile name if not present
            if 'name' not in profile_config:
                profile_config['name'] = profile_name
            
            return profile_config
            
        except Exception as e:
            self.logger.error(f"Error loading profile '{profile_name}': {e}")
            raise
    
    def list_profiles(self) -> List[str]:
        """List available profiles."""
        profiles = []
        
        if not self.profiles_dir.exists():
            return profiles
        
        for file_path in self.profiles_dir.glob('quick_start_*.yaml'):
            # Extract profile name from filename
            profile_name = file_path.stem.replace('quick_start_', '')
            profiles.append(profile_name)
        
        return sorted(profiles)
    
    def get_profile_description(self, profile_name: str) -> str:
        """Get profile description."""
        try:
            profile_config = self.load_profile(profile_name)
            return profile_config.get('description', f'{profile_name} profile')
        except Exception:
            return f'{profile_name} profile'
    
    def validate_profile(self, profile_config: Dict[str, Any]) -> bool:
        """Validate profile configuration."""
        required_fields = ['name', 'description']
        
        for field in required_fields:
            if field not in profile_config:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def create_profile(self, profile_name: str, config: Dict[str, Any]) -> bool:
        """Create a new profile."""
        try:
            # Validate configuration
            if not self.validate_profile(config):
                return False
            
            # Ensure profiles directory exists
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            
            # Write profile file
            profile_file = self.profiles_dir / f'quick_start_{profile_name}.yaml'
            with open(profile_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Profile '{profile_name}' created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating profile '{profile_name}': {e}")
            return False
    
    def get_default_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get default profile configurations."""
        return {
            'minimal': {
                'name': 'minimal',
                'description': 'Minimal setup for development (50 docs, 2GB RAM)',
                'requirements': {
                    'memory_gb': 2,
                    'disk_gb': 5,
                    'documents': 50
                },
                'environment': {
                    'IRIS_HOST': 'localhost',
                    'IRIS_PORT': '1972',
                    'LOG_LEVEL': 'INFO'
                },
                'data': {
                    'source': 'pmc_sample',
                    'limit': 50,
                    'embeddings': True
                },
                'pipelines': ['basic', 'hyde']
            },
            'standard': {
                'name': 'standard',
                'description': 'Standard setup for evaluation (500 docs, 4GB RAM)',
                'requirements': {
                    'memory_gb': 4,
                    'disk_gb': 10,
                    'documents': 500
                },
                'environment': {
                    'IRIS_HOST': 'localhost',
                    'IRIS_PORT': '1972',
                    'LOG_LEVEL': 'INFO'
                },
                'data': {
                    'source': 'pmc_sample',
                    'limit': 500,
                    'embeddings': True
                },
                'pipelines': ['basic', 'hyde', 'colbert', 'crag']
            },
            'extended': {
                'name': 'extended',
                'description': 'Extended setup for comprehensive testing (5000 docs, 8GB RAM)',
                'requirements': {
                    'memory_gb': 8,
                    'disk_gb': 20,
                    'documents': 5000
                },
                'environment': {
                    'IRIS_HOST': 'localhost',
                    'IRIS_PORT': '1972',
                    'LOG_LEVEL': 'INFO'
                },
                'data': {
                    'source': 'pmc_sample',
                    'limit': 5000,
                    'embeddings': True
                },
                'pipelines': ['basic', 'hyde', 'colbert', 'crag', 'graphrag', 'noderag', 'hybrid_ifind']
            }
        }
    
    def ensure_default_profiles(self) -> None:
        """Ensure default profiles exist."""
        default_profiles = self.get_default_profiles()
        
        for profile_name, config in default_profiles.items():
            if not self.profile_exists(profile_name):
                self.create_profile(profile_name, config)