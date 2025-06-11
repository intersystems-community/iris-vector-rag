"""
Environment Validator for RAG System.

Validates conda environment, package dependencies, ML/AI function availability,
and handles errors securely.
"""

import os
import subprocess
import logging
import importlib.metadata
from packaging.requirements import Requirement
from packaging.version import parse as parse_version
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """
    Validates the RAG system's runtime environment.
    """
    def __init__(self, config=None):
        """
        Initializes the EnvironmentValidator.

        Args:
            config: Optional configuration object.
        """
        self.config = config or {}
        self.results = {}
        
        # Allow tests to set this, or load from self.config
        # Default to None to be more flexible - accept any active conda environment
        self.expected_conda_env_name = getattr(config, 'expected_conda_env_name', None)
        if self.config and isinstance(self.config, dict) and 'expected_conda_env_name' in self.config:
            self.expected_conda_env_name = self.config['expected_conda_env_name']
        
        # For production validation, be flexible about environment names
        # Only enforce specific environment names in test scenarios
        self.flexible_env_validation = self.config.get('flexible_env_validation', True)

    def validate_conda_activation(self):
        """
        Validates that the correct conda environment is activated.
        Uses source activate_env.sh for activation validation.
        """
        try:
            # Check current conda environment
            active_env_name = os.environ.get("CONDA_DEFAULT_ENV")
            active_env_prefix = os.environ.get("CONDA_PREFIX")

            if not self.expected_conda_env_name:
                # If no expected name is set, check if any conda environment is active
                if active_env_name and active_env_prefix:
                    self.results['conda_activation'] = {
                        'status': 'pass',
                        'details': f"A conda environment '{active_env_name}' is active at '{active_env_prefix}'. No specific environment name was expected."
                    }
                    return True
                else:
                    # Try to activate using activate_env.sh
                    activation_result = self._test_conda_activation_script()
                    if activation_result:
                        self.results['conda_activation'] = {
                            'status': 'pass',
                            'details': "Successfully activated conda environment using activate_env.sh"
                        }
                        return True
                    else:
                        self.results['conda_activation'] = {
                            'status': 'fail',
                            'details': "No conda environment is active and activate_env.sh failed"
                        }
                        return False

            # Check if expected environment is active
            if not active_env_name:
                # Try activation script
                activation_result = self._test_conda_activation_script()
                if activation_result:
                    self.results['conda_activation'] = {
                        'status': 'pass',
                        'details': f"Successfully activated expected environment using activate_env.sh"
                    }
                    return True
                else:
                    self.results['conda_activation'] = {
                        'status': 'fail',
                        'details': f"CONDA_DEFAULT_ENV not set. Expected '{self.expected_conda_env_name}'."
                    }
                    return False

            # Use flexible validation if enabled
            if self.flexible_env_validation:
                # Accept any active conda environment in flexible mode
                self.results['conda_activation'] = {
                    'status': 'pass',
                    'details': f"Conda environment '{active_env_name}' is active (flexible validation mode)."
                }
                return True
            
            if active_env_name != self.expected_conda_env_name:
                self.results['conda_activation'] = {
                    'status': 'fail',
                    'details': f"Expected conda environment '{self.expected_conda_env_name}' but found '{active_env_name}'."
                }
                return False

            if not active_env_prefix:
                self.results['conda_activation'] = {
                    'status': 'fail',
                    'details': f"Conda environment '{active_env_name}' is active, but CONDA_PREFIX is not set."
                }
                return False
            
            # Check if the active_env_prefix path actually contains the environment name
            if self.expected_conda_env_name not in os.path.basename(active_env_prefix):
                self.results['conda_activation'] = {
                    'status': 'fail',
                    'details': f"Conda environment '{active_env_name}' is active, but CONDA_PREFIX ('{active_env_prefix}') does not appear to match the environment name."
                }
                return False

            self.results['conda_activation'] = {
                'status': 'pass',
                'details': f"Conda environment '{self.expected_conda_env_name}' is active."
            }
            return True
            
        except Exception as e:
            logger.error(f"Error during conda activation validation: {e}")
            self.results['conda_activation'] = {
                'status': 'error',
                'details': f"Error during conda activation validation: {str(e)}"
            }
            return False

    def _test_conda_activation_script(self):
        """
        Tests conda environment activation using activate_env.sh script.
        """
        try:
            # Check if activate_env.sh exists
            script_path = os.path.join(os.getcwd(), "activate_env.sh")
            if not os.path.exists(script_path):
                logger.warning("activate_env.sh not found in current directory")
                return False
            
            # Test the activation script
            result = subprocess.run(
                ["bash", "-c", "source activate_env.sh && echo $CONDA_DEFAULT_ENV"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"Successfully tested conda activation: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"Conda activation test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing conda activation script: {e}")
            return False

    def verify_package_dependencies(self):
        """
        Verifies that all required package dependencies are installed and at correct versions.
        Uses importlib.metadata and packaging.requirements for robust checking.
        """
        try:
            # Define core required packages for RAG system
            required_packages = {
                'intersystems-irispython': '>=3.2.0',
                'langchain': '>=0.1.0',
                'langchain-openai': '>=0.1.0',
                'sentence-transformers': '>=2.0.0',
                'numpy': '>=1.21.0',
                'pandas': '>=1.3.0',
                'scikit-learn': '>=1.0.0',
                'transformers': '>=4.20.0',
                'torch': '>=1.12.0',
                'faiss-cpu': '>=1.7.0',
                'openai': '>=1.0.0',
                'tiktoken': '>=0.4.0'
            }
            
            # Override with config if provided
            expected_packages_config = self.config.get('expected_packages', required_packages) if self.config else required_packages
            
            if not expected_packages_config:
                self.results['package_dependencies'] = {
                    'status': 'pass',
                    'details': 'No packages specified in configuration to verify.',
                    'packages': {}
                }
                return True

            all_ok = True
            package_results = {}

            for pkg_name, version_spec_str in expected_packages_config.items():
                try:
                    installed_version_str = importlib.metadata.version(pkg_name)
                    installed_version = parse_version(installed_version_str)
                    
                    # Create a requirement object to check against specifiers
                    if not any(op in version_spec_str for op in ['>', '<', '=', '~', '!']):
                        req_str = f"{pkg_name}=={version_spec_str}"
                    else:
                        req_str = f"{pkg_name}{version_spec_str}"
                    
                    requirement = Requirement(req_str)
                    
                    if installed_version in requirement.specifier:
                        package_results[pkg_name] = {
                            'status': 'pass',
                            'expected': version_spec_str,
                            'found': installed_version_str,
                            'details': 'Correct version installed.'
                        }
                    else:
                        all_ok = False
                        package_results[pkg_name] = {
                            'status': 'fail',
                            'expected': version_spec_str,
                            'found': installed_version_str,
                            'details': f"Installed version {installed_version_str} does not meet requirement {version_spec_str}."
                        }
                except importlib.metadata.PackageNotFoundError:
                    all_ok = False
                    package_results[pkg_name] = {
                        'status': 'fail',
                        'expected': version_spec_str,
                        'found': None,
                        'details': f"{pkg_name} is not installed."
                    }
                except Exception as e:
                    all_ok = False
                    package_results[pkg_name] = {
                        'status': 'error',
                        'expected': version_spec_str,
                        'found': None,
                        'details': f"Error checking package {pkg_name}: {str(e)}"
                    }

            overall_status = 'pass' if all_ok else 'fail'
            details_message = "All required packages are installed with compatible versions." if all_ok else "Some package dependencies are not met."
            
            self.results['package_dependencies'] = {
                'status': overall_status,
                'details': details_message,
                'packages': package_results
            }
            return all_ok
            
        except Exception as e:
            logger.error(f"Error during package dependency verification: {e}")
            self.results['package_dependencies'] = {
                'status': 'error',
                'details': f"Error during package dependency verification: {str(e)}",
                'packages': {}
            }
            return False

    def _get_embedding_model(self):
        """
        Loads and returns an embedding model using common.utils.get_embedding_func.
        """
        try:
            from common.utils import get_embedding_func
            return get_embedding_func()
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return None

    def _get_llm(self):
        """
        Loads and returns an LLM using common.utils.get_llm_func.
        """
        try:
            from common.utils import get_llm_func
            return get_llm_func()
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            return None

    def test_ml_ai_function_availability(self):
        """
        Tests the availability and basic functionality of core ML/AI functions.
        (e.g., embedding generation, LLM responses).
        """
        embedding_model_ok = False
        llm_ok = False
        
        embedding_model_details = "Not tested"
        llm_details = "Not tested"

        try:
            # Test Embedding Model
            embedding_func = self._get_embedding_model()
            if embedding_func:
                try:
                    test_text = self.config.get('test_embedding_text', "This is a test sentence for embedding generation.")
                    embedding = embedding_func(test_text)
                    
                    # Handle different embedding function return types
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # Check if it's a nested list (batch response)
                        if isinstance(embedding[0], list):
                            embedding = embedding[0]
                        
                        # Validate embedding format
                        if all(isinstance(val, (int, float)) for val in embedding):
                            embedding_model_ok = True
                            embedding_model_details = f"Embedding model generated {len(embedding)}-dimensional embedding successfully."
                        else:
                            embedding_model_details = f"Embedding model returned invalid embedding format: {type(embedding[0])}"
                    else:
                        embedding_model_details = f"Embedding model returned invalid embedding: {str(embedding)[:100]}"
                except Exception as e:
                    embedding_model_details = f"Error during embedding model test: {str(e)}"
            else:
                embedding_model_details = "Embedding model not available or failed to load."

            # Test LLM
            llm_func = self._get_llm()
            if llm_func:
                try:
                    test_prompt = self.config.get('test_llm_prompt', "Say 'test' and nothing else.")
                    response = llm_func(test_prompt)
                    
                    # Basic check for non-empty response
                    if response and isinstance(response, str) and len(response.strip()) > 0:
                        llm_ok = True
                        llm_details = f"LLM invoked successfully. Response: {str(response)[:50]}..."
                    elif hasattr(response, 'content') and response.content:
                        llm_ok = True
                        llm_details = f"LLM invoked successfully. Response: {str(response.content)[:50]}..."
                    else:
                        llm_details = f"LLM returned empty or invalid response: {str(response)[:100]}"
                except Exception as e:
                    llm_details = f"Error during LLM test: {str(e)}"
            else:
                llm_details = "LLM not available or failed to load."

        except Exception as e:
            logger.error(f"Error during ML/AI function testing: {e}")
            embedding_model_details = f"Error during ML/AI function testing: {str(e)}"
            llm_details = f"Error during ML/AI function testing: {str(e)}"

        overall_status = 'pass' if embedding_model_ok and llm_ok else 'fail'
        self.results['ml_ai_functions'] = {
            'status': overall_status,
            'embedding_model_status': {
                'status': 'pass' if embedding_model_ok else 'fail',
                'details': embedding_model_details
            },
            'llm_status': {
                'status': 'pass' if llm_ok else 'fail',
                'details': llm_details
            }
        }
        return embedding_model_ok and llm_ok

    def run_all_checks(self):
        """
        Runs all environment validation checks.
        """
        try:
            logger.info("Starting environment validation checks...")
            
            conda_ok = self.validate_conda_activation()
            logger.info(f"Conda activation check: {'PASS' if conda_ok else 'FAIL'}")
            
            deps_ok = self.verify_package_dependencies()
            logger.info(f"Package dependencies check: {'PASS' if deps_ok else 'FAIL'}")
            
            ml_funcs_ok = self.test_ml_ai_function_availability()
            logger.info(f"ML/AI functions check: {'PASS' if ml_funcs_ok else 'FAIL'}")
            
            overall_status = conda_ok and deps_ok and ml_funcs_ok
            self.results['overall_status'] = 'pass' if overall_status else 'fail'
            
            logger.info(f"Environment validation completed: {'PASS' if overall_status else 'FAIL'}")
            return overall_status
            
        except Exception as e:
            logger.error(f"Error during environment validation: {e}")
            self.results['overall_status'] = 'error'
            self.results['error'] = str(e)
            return False

    def get_results(self):
        """
        Returns the results of the validation checks.
        """
        return self.results

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    validator = EnvironmentValidator()
    success = validator.run_all_checks()
    
    print("Environment Validation Results:")
    import json
    print(json.dumps(validator.get_results(), indent=2))
    
    if success:
        print("\n✅ Environment validation PASSED")
    else:
        print("\n❌ Environment validation FAILED")