"""
Security Configuration Module

This module provides centralized security configuration and validation
to prevent silent fallback vulnerabilities and ensure secure operation.
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class SecurityConfig:
    """Centralized security configuration management"""
    
    def __init__(self):
        self._config = self._load_security_config()
        self._validate_config()
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration from environment variables"""
        return {
            'strict_import_validation': self._get_bool_env('STRICT_IMPORT_VALIDATION', True),
            'disable_silent_fallbacks': self._get_bool_env('DISABLE_SILENT_FALLBACKS', True),
            'enable_audit_logging': self._get_bool_env('ENABLE_AUDIT_LOGGING', True),
            'security_level': SecurityLevel(os.getenv('APP_ENV', 'production')),
            'fail_fast_on_import_error': self._get_bool_env('FAIL_FAST_ON_IMPORT_ERROR', True),
            'allow_mock_implementations': self._get_bool_env('ALLOW_MOCK_IMPLEMENTATIONS', False),
        }
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable with proper parsing"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _validate_config(self) -> None:
        """Validate security configuration for consistency"""
        if self._config['security_level'] == SecurityLevel.PRODUCTION:
            if not self._config['strict_import_validation']:
                logger.warning("SECURITY WARNING: strict_import_validation disabled in production")
            if not self._config['disable_silent_fallbacks']:
                logger.warning("SECURITY WARNING: silent_fallbacks enabled in production")
            if self._config['allow_mock_implementations']:
                logger.warning("SECURITY WARNING: mock_implementations allowed in production")
    
    @property
    def strict_import_validation(self) -> bool:
        """Whether to enforce strict import validation"""
        return self._config['strict_import_validation']
    
    @property
    def disable_silent_fallbacks(self) -> bool:
        """Whether to disable silent fallback mechanisms"""
        return self._config['disable_silent_fallbacks']
    
    @property
    def enable_audit_logging(self) -> bool:
        """Whether to enable audit logging for security events"""
        return self._config['enable_audit_logging']
    
    @property
    def security_level(self) -> SecurityLevel:
        """Current security level"""
        return self._config['security_level']
    
    @property
    def fail_fast_on_import_error(self) -> bool:
        """Whether to fail fast on import errors instead of falling back"""
        return self._config['fail_fast_on_import_error']
    
    @property
    def allow_mock_implementations(self) -> bool:
        """Whether to allow mock implementations (development/testing only)"""
        return self._config['allow_mock_implementations']


class ImportValidationError(Exception):
    """Raised when import validation fails in strict mode"""
    pass


class SilentFallbackError(Exception):
    """Raised when silent fallback is attempted but disabled"""
    pass


class SecurityValidator:
    """Security validation utilities"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
    
    def validate_import(self, module_name: str, import_error: Exception) -> None:
        """Validate import and handle according to security policy"""
        if self.config.enable_audit_logging:
            logger.warning(f"SECURITY AUDIT: Import failed for module '{module_name}': {import_error}")
        
        if self.config.strict_import_validation and self.config.fail_fast_on_import_error:
            raise ImportValidationError(
                f"Import validation failed for '{module_name}' in strict mode: {import_error}"
            )
    
    def check_fallback_allowed(self, component_name: str, fallback_type: str) -> bool:
        """Check if fallback is allowed for a component"""
        if self.config.disable_silent_fallbacks:
            if self.config.enable_audit_logging:
                logger.error(
                    f"SECURITY AUDIT: Silent fallback attempted for '{component_name}' "
                    f"(type: {fallback_type}) but disabled by security policy"
                )
            raise SilentFallbackError(
                f"Silent fallback disabled for '{component_name}' (type: {fallback_type})"
            )
        
        # Allow fallback but log it
        if self.config.enable_audit_logging:
            logger.warning(
                f"SECURITY AUDIT: Silent fallback activated for '{component_name}' "
                f"(type: {fallback_type})"
            )
        
        return True
    
    def validate_mock_usage(self, component_name: str) -> bool:
        """Validate if mock implementations are allowed"""
        if not self.config.allow_mock_implementations:
            if self.config.security_level == SecurityLevel.PRODUCTION:
                raise SilentFallbackError(
                    f"Mock implementation not allowed for '{component_name}' in production"
                )
            
            if self.config.enable_audit_logging:
                logger.warning(
                    f"SECURITY AUDIT: Mock implementation used for '{component_name}' "
                    f"but not explicitly allowed"
                )
        
        return True


# Global security configuration instance
_security_config = None
_security_validator = None


def get_security_config() -> SecurityConfig:
    """Get global security configuration instance"""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def get_security_validator() -> SecurityValidator:
    """Get global security validator instance"""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator(get_security_config())
    return _security_validator


def reset_security_config() -> None:
    """Reset global security configuration (for testing)"""
    global _security_config, _security_validator
    _security_config = None
    _security_validator = None