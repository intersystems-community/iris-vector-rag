#!/usr/bin/env python3
"""
Configuration Management for RAG Evaluation Framework
Provides centralized configuration with validation and environment support
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Supported configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 1972
    namespace: str = "USER"
    username: str = "demo"
    password: str = "demo"
    connection_type: str = "dbapi"  # dbapi or jdbc
    schema: str = "RAG"
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables"""
        return cls(
            host=os.getenv("IRIS_HOST", "localhost"),
            port=int(os.getenv("IRIS_PORT", "1972")),
            namespace=os.getenv("IRIS_NAMESPACE", "USER"),
            username=os.getenv("IRIS_USERNAME", "demo"),
            password=os.getenv("IRIS_PASSWORD", "demo"),
            connection_type=os.getenv("CONNECTION_TYPE", "dbapi"),
            schema=os.getenv("IRIS_SCHEMA", "RAG"),
            timeout=int(os.getenv("DB_TIMEOUT", "30"))
        )

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create configuration from environment variables"""
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_length=int(os.getenv("EMBEDDING_MAX_LENGTH", "512")),
            normalize_embeddings=os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true",
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR")
        )

@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openai"  # openai, anthropic, huggingface, local
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create configuration from environment variables"""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30"))
        )
    
    def get_effective_api_key(self) -> Optional[str]:
        """Get the effective API key, checking environment if config value is None"""
        if self.api_key:
            return self.api_key
        # If config api_key is None, check environment variables
        return os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    method: str = "fixed_size"  # fixed_size, semantic, recursive, sentence
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n\n"
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    
    @classmethod
    def from_env(cls) -> 'ChunkingConfig':
        """Create configuration from environment variables"""
        return cls(
            method=os.getenv("CHUNKING_METHOD", "fixed_size"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            separator=os.getenv("CHUNK_SEPARATOR", "\n\n"),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1000"))
        )

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 10
    similarity_threshold: float = 0.1
    rerank: bool = False
    rerank_model: Optional[str] = None
    diversity_threshold: float = 0.7
    max_documents: int = 50
    
    @classmethod
    def from_env(cls) -> 'RetrievalConfig':
        """Create configuration from environment variables"""
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.1")),
            rerank=os.getenv("ENABLE_RERANK", "false").lower() == "true",
            rerank_model=os.getenv("RERANK_MODEL"),
            diversity_threshold=float(os.getenv("DIVERSITY_THRESHOLD", "0.7")),
            max_documents=int(os.getenv("MAX_DOCUMENTS", "50"))
        )

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    enable_ragas: bool = True
    enable_statistical_testing: bool = True
    num_iterations: int = 3
    parallel_execution: bool = False
    max_workers: int = 4
    timeout_per_query: int = 60
    save_individual_results: bool = True
    
    @classmethod
    def from_env(cls) -> 'EvaluationConfig':
        """Create configuration from environment variables"""
        return cls(
            enable_ragas=os.getenv("ENABLE_RAGAS", "true").lower() == "true",
            enable_statistical_testing=os.getenv("ENABLE_STATS", "true").lower() == "true",
            num_iterations=int(os.getenv("NUM_ITERATIONS", "3")),
            parallel_execution=os.getenv("PARALLEL_EXECUTION", "false").lower() == "true",
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            timeout_per_query=int(os.getenv("QUERY_TIMEOUT", "60")),
            save_individual_results=os.getenv("SAVE_INDIVIDUAL", "true").lower() == "true"
        )

@dataclass
class OutputConfig:
    """Output configuration"""
    results_dir: str = "eval_results"
    save_results: bool = True
    create_visualizations: bool = True
    generate_report: bool = True
    export_formats: list = field(default_factory=lambda: ["json", "csv"])
    visualization_formats: list = field(default_factory=lambda: ["png", "pdf"])
    
    @classmethod
    def from_env(cls) -> 'OutputConfig':
        """Create configuration from environment variables"""
        export_formats = os.getenv("EXPORT_FORMATS", "json,csv").split(",")
        viz_formats = os.getenv("VIZ_FORMATS", "png,pdf").split(",")
        
        return cls(
            results_dir=os.getenv("RESULTS_DIR", "eval_results"),
            save_results=os.getenv("SAVE_RESULTS", "true").lower() == "true",
            create_visualizations=os.getenv("CREATE_VIZ", "true").lower() == "true",
            generate_report=os.getenv("GENERATE_REPORT", "true").lower() == "true",
            export_formats=[f.strip() for f in export_formats],
            visualization_formats=[f.strip() for f in viz_formats]
        )

@dataclass
class PipelineConfig:
    """Individual pipeline configuration"""
    enabled: bool = True
    timeout: int = 120
    retry_attempts: int = 3
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComprehensiveConfig:
    """Comprehensive configuration combining all components"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipelines: Dict[str, PipelineConfig] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'ComprehensiveConfig':
        """Create comprehensive configuration from environment variables"""
        return cls(
            database=DatabaseConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            llm=LLMConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            evaluation=EvaluationConfig.from_env(),
            output=OutputConfig.from_env()
        )
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Database validation
        if not self.database.host:
            errors.append("Database host is required")
        if not (1 <= self.database.port <= 65535):
            errors.append("Database port must be between 1 and 65535")
        
        # Embedding validation
        if not self.embedding.model_name:
            errors.append("Embedding model name is required")
        if self.embedding.batch_size <= 0:
            errors.append("Embedding batch size must be positive")
        
        # LLM validation
        if self.llm.provider == "openai" and not self.llm.get_effective_api_key():
            errors.append("OpenAI API key is required for OpenAI provider")
        if not (0 <= self.llm.temperature <= 2):
            errors.append("LLM temperature must be between 0 and 2")
        
        # Chunking validation
        if self.chunking.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        
        # Retrieval validation
        if self.retrieval.top_k <= 0:
            errors.append("Top K must be positive")
        if not (0 <= self.retrieval.similarity_threshold <= 1):
            errors.append("Similarity threshold must be between 0 and 1")
        
        # Evaluation validation
        if self.evaluation.num_iterations <= 0:
            errors.append("Number of iterations must be positive")
        if self.evaluation.max_workers <= 0:
            errors.append("Max workers must be positive")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True

class ConfigManager:
    """Configuration manager for RAG evaluation framework"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager"""
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[ComprehensiveConfig] = None
        
    def load_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   format: Optional[ConfigFormat] = None) -> ComprehensiveConfig:
        """Load configuration from file or environment"""
        
        if config_path:
            self.config_path = Path(config_path)
        
        # If no config file specified, load from environment
        if not self.config_path or not self.config_path.exists():
            logger.info("Loading configuration from environment variables")
            self._config = ComprehensiveConfig.from_env()
        else:
            logger.info(f"Loading configuration from {self.config_path}")
            self._config = self._load_from_file(self.config_path, format)
        
        # Validate configuration
        if not self._config.validate():
            raise ValueError("Configuration validation failed")
        
        return self._config
    
    def _load_from_file(self, 
                       config_path: Path, 
                       format: Optional[ConfigFormat] = None) -> ComprehensiveConfig:
        """Load configuration from file"""
        
        # Auto-detect format if not specified
        if format is None:
            if config_path.suffix.lower() == '.json':
                format = ConfigFormat.JSON
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            else:
                raise ValueError(f"Cannot auto-detect format for {config_path}")
        
        # Load data based on format
        with open(config_path, 'r') as f:
            if format == ConfigFormat.JSON:
                data = json.load(f)
            elif format == ConfigFormat.YAML:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        # Convert to configuration object
        return self._dict_to_config(data)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ComprehensiveConfig:
        """Convert dictionary to configuration object"""
        config = ComprehensiveConfig()
        
        # Update each section if present
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'embedding' in data:
            config.embedding = EmbeddingConfig(**data['embedding'])
        if 'llm' in data:
            config.llm = LLMConfig(**data['llm'])
        if 'chunking' in data:
            config.chunking = ChunkingConfig(**data['chunking'])
        if 'retrieval' in data:
            config.retrieval = RetrievalConfig(**data['retrieval'])
        if 'evaluation' in data:
            config.evaluation = EvaluationConfig(**data['evaluation'])
        if 'output' in data:
            config.output = OutputConfig(**data['output'])
        if 'pipelines' in data:
            config.pipelines = {
                name: PipelineConfig(**params) 
                for name, params in data['pipelines'].items()
            }
        
        return config
    
    def save_config(self, 
                   config: ComprehensiveConfig,
                   config_path: Optional[Union[str, Path]] = None,
                   format: ConfigFormat = ConfigFormat.JSON) -> None:
        """Save configuration to file"""
        
        if config_path:
            save_path = Path(config_path)
        elif self.config_path:
            save_path = self.config_path
        else:
            save_path = Path(f"config.{format.value}")
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        data = asdict(config)
        
        # Save based on format
        with open(save_path, 'w') as f:
            if format == ConfigFormat.JSON:
                json.dump(data, f, indent=2, default=str)
            elif format == ConfigFormat.YAML:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {save_path}")
    
    def get_config(self) -> ComprehensiveConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def create_default_config(self, output_path: Union[str, Path]) -> None:
        """Create a default configuration file"""
        default_config = ComprehensiveConfig()
        
        # Add some example pipeline configurations
        default_config.pipelines = {
            "BasicRAG": PipelineConfig(enabled=True, timeout=60),
            "HyDE": PipelineConfig(enabled=True, timeout=90),
            "CRAG": PipelineConfig(enabled=True, timeout=120),
            "ColBERT": PipelineConfig(enabled=True, timeout=180),
            "NodeRAG": PipelineConfig(enabled=True, timeout=150),
            "GraphRAG": PipelineConfig(enabled=True, timeout=200)
        }
        
        self.save_config(default_config, output_path)
    
    def merge_configs(self, 
                     base_config: ComprehensiveConfig,
                     override_config: ComprehensiveConfig) -> ComprehensiveConfig:
        """Merge two configurations, with override taking precedence"""
        # Convert to dictionaries for easier merging
        base_dict = asdict(base_config)
        override_dict = asdict(override_config)
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        # Convert back to configuration object
        return self._dict_to_config(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


def create_sample_configs():
    """Create sample configuration files"""
    manager = ConfigManager()
    
    # Create default configuration
    manager.create_default_config("eval/config/default_config.json")
    
    # Create development configuration
    dev_config = ComprehensiveConfig.from_env()
    dev_config.evaluation.num_iterations = 1
    dev_config.evaluation.enable_ragas = False
    dev_config.output.create_visualizations = False
    manager.save_config(dev_config, "eval/config/dev_config.json")
    
    # Create production configuration
    prod_config = ComprehensiveConfig.from_env()
    prod_config.evaluation.num_iterations = 5
    prod_config.evaluation.parallel_execution = True
    prod_config.evaluation.max_workers = 8
    manager.save_config(prod_config, "eval/config/prod_config.json")
    
    print("Sample configuration files created in eval/config/")


if __name__ == "__main__":
    create_sample_configs()