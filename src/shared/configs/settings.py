"""
Global Settings and Configuration Management for JJongal-AI

Centralized configuration management with environment-based settings,
validation, and runtime configuration updates.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.shared.utils.logging import get_module_logger

logger = get_module_logger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    vector_db_path: str = "./data/vector_db"
    conversation_db_path: str = "./data/conversations.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    max_backup_files: int = 10

@dataclass
class AIServiceConfig:
    """AI service configuration"""
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    elevenlabs_voice_stability: float = 0.8
    elevenlabs_voice_similarity: float = 0.8
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

@dataclass
class VoiceConfig:
    """Voice processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    max_recording_duration: int = 30
    min_recording_duration: float = 0.5
    noise_reduction_enabled: bool = True
    voice_clone_min_samples: int = 5
    voice_clone_max_samples: int = 15
    audio_quality_threshold: float = 0.6

@dataclass
class WorkflowConfig:
    """Workflow processing configuration"""
    max_concurrent_workflows: int = 5
    workflow_timeout_minutes: int = 30
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 2
    max_conversation_length: int = 50
    story_generation_timeout: int = 120
    multimedia_generation_timeout: int = 300

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests_per_minute: int = 60
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    session_timeout_minutes: int = 30

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    keepalive_timeout: int = 5
    graceful_timeout: int = 30

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80
    connection_pool_size: int = 10
    async_worker_count: int = 4
    response_compression: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    metrics_enabled: bool = True
    logging_enabled: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_file_path: str = "./logs/jjongal-ai.log"
    log_rotation_enabled: bool = True
    log_max_size_mb: int = 100

@dataclass
class ChildSafetyConfig:
    """Child safety and content filtering configuration"""
    content_filtering_enabled: bool = True
    inappropriate_content_threshold: float = 0.8
    age_appropriate_validation: bool = True
    parental_controls_enabled: bool = True
    conversation_monitoring: bool = True
    safe_mode_default: bool = True

class Settings:
    """
    Global settings manager for JJongal-AI
    
    Features:
    - Environment-based configuration loading
    - Runtime configuration updates
    - Settings validation and defaults
    - Configuration file management
    """
    
    def __init__(self, env: Environment = Environment.DEVELOPMENT):
        """
        Initialize settings manager
        
        Args:
            env: Environment type for configuration
        """
        self.environment = env
        self.config_loaded = False
        
        # Configuration sections
        self.database = DatabaseConfig()
        self.ai_services = AIServiceConfig()
        self.voice = VoiceConfig()
        self.workflow = WorkflowConfig()
        self.security = SecurityConfig()
        self.api = APIConfig()
        self.performance = PerformanceConfig()
        self.monitoring = MonitoringConfig()
        self.child_safety = ChildSafetyConfig()
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"Settings initialized for {env.value} environment")
    
    def _load_configuration(self):
        """Load configuration from various sources"""
        try:
            # Load from environment variables first (highest priority)
            self._load_from_environment()
            
            # Load from configuration files
            self._load_from_config_file()
            
            # Apply environment-specific overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            self.config_loaded = True
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_environment(self):
        """Load settings from environment variables"""
        # AI Services
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.ai_services.openai_api_key = api_key
        if api_key := os.getenv("ELEVENLABS_API_KEY"):
            self.ai_services.elevenlabs_api_key = api_key
        if model := os.getenv("OPENAI_MODEL"):
            self.ai_services.openai_model = model
        
        # Security
        if secret := os.getenv("JWT_SECRET_KEY"):
            self.security.jwt_secret_key = secret
        
        # API
        if host := os.getenv("API_HOST"):
            self.api.host = host
        if port := os.getenv("API_PORT"):
            self.api.port = int(port)
        if debug := os.getenv("API_DEBUG"):
            self.api.debug = debug.lower() in ("true", "1", "yes")
        
        # Database
        if db_path := os.getenv("VECTOR_DB_PATH"):
            self.database.vector_db_path = db_path
        if conv_path := os.getenv("CONVERSATION_DB_PATH"):
            self.database.conversation_db_path = conv_path
        
        # Performance
        if memory_limit := os.getenv("MEMORY_LIMIT_MB"):
            self.performance.memory_limit_mb = int(memory_limit)
        if workers := os.getenv("ASYNC_WORKERS"):
            self.performance.async_worker_count = int(workers)
        
        # Monitoring
        if log_level := os.getenv("LOG_LEVEL"):
            self.monitoring.log_level = LogLevel(log_level.upper())
        if log_path := os.getenv("LOG_FILE_PATH"):
            self.monitoring.log_file_path = log_path
    
    def _load_from_config_file(self):
        """Load settings from configuration file"""
        config_files = [
            f"config/{self.environment.value}.json",
            "config/default.json",
            ".jjongal-config.json"
        ]
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    self._merge_config_data(config_data)
                    logger.info(f"Loaded configuration from {config_file}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_file}: {e}")
    
    def _merge_config_data(self, config_data: Dict[str, Any]):
        """Merge configuration data into settings"""
        for section, data in config_data.items():
            if hasattr(self, section) and isinstance(data, dict):
                config_obj = getattr(self, section)
                for key, value in data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            self.api.debug = True
            self.api.reload = True
            self.monitoring.log_level = LogLevel.DEBUG
            self.performance.cache_enabled = False
            
        elif self.environment == Environment.TESTING:
            self.api.debug = True
            self.database.vector_db_path = "./test_data/vector_db"
            self.database.conversation_db_path = "./test_data/conversations.db"
            self.monitoring.log_level = LogLevel.DEBUG
            
        elif self.environment == Environment.STAGING:
            self.api.debug = False
            self.api.workers = 2
            self.monitoring.log_level = LogLevel.INFO
            self.performance.cache_enabled = True
            
        elif self.environment == Environment.PRODUCTION:
            self.api.debug = False
            self.api.reload = False
            self.api.workers = 4
            self.monitoring.log_level = LogLevel.WARNING
            self.performance.cache_enabled = True
            self.security.cors_allowed_origins = []  # Restrict CORS in production
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate required API keys
        if not self.ai_services.openai_api_key:
            errors.append("OpenAI API key is required")
        
        if not self.ai_services.elevenlabs_api_key:
            errors.append("ElevenLabs API key is required")
        
        if not self.security.jwt_secret_key:
            errors.append("JWT secret key is required")
        
        # Validate numeric ranges
        if not (0.0 <= self.ai_services.openai_temperature <= 2.0):
            errors.append("OpenAI temperature must be between 0.0 and 2.0")
        
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")
        
        if not (8000 <= self.voice.sample_rate <= 48000):
            errors.append("Voice sample rate must be between 8000 and 48000")
        
        # Validate paths
        try:
            Path(self.database.vector_db_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.database.conversation_db_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.monitoring.log_file_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create required directories: {e}")
        
        if errors and self.environment == Environment.PRODUCTION:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        elif errors:
            for error in errors:
                logger.warning(f"Configuration warning: {error}")
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings as dictionary"""
        return {
            "environment": self.environment.value,
            "database": asdict(self.database),
            "ai_services": self._sanitize_sensitive_data(asdict(self.ai_services)),
            "voice": asdict(self.voice),
            "workflow": asdict(self.workflow),
            "security": self._sanitize_sensitive_data(asdict(self.security)),
            "api": asdict(self.api),
            "performance": asdict(self.performance),
            "monitoring": asdict(self.monitoring),
            "child_safety": asdict(self.child_safety)
        }
    
    def _sanitize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data from settings"""
        sensitive_keys = ['api_key', 'secret_key', 'password', 'token']
        sanitized = data.copy()
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if value:
                    sanitized[key] = "*" * 8 + value[-4:] if len(value) > 4 else "*" * len(value)
                else:
                    sanitized[key] = "[NOT_SET]"
        
        return sanitized
    
    def update_setting(self, section: str, key: str, value: Any) -> bool:
        """
        Update a specific setting at runtime
        
        Args:
            section: Configuration section name
            key: Setting key
            value: New value
            
        Returns:
            bool: True if update successful
        """
        try:
            if hasattr(self, section):
                config_obj = getattr(self, section)
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"Updated setting {section}.{key}")
                    return True
                else:
                    logger.warning(f"Setting key {key} not found in {section}")
            else:
                logger.warning(f"Configuration section {section} not found")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update setting {section}.{key}: {e}")
            return False
    
    def save_to_file(self, file_path: str):
        """
        Save current configuration to file
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            config_data = self.get_all_settings()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION

# Global settings instance
_settings = None

def get_settings(env: Optional[Environment] = None) -> Settings:
    """
    Get global settings instance
    
    Args:
        env: Environment type (if not already initialized)
        
    Returns:
        Settings: Global settings instance
    """
    global _settings
    if _settings is None:
        if env is None:
            env_name = os.getenv("JJONGAL_ENV", "development").lower()
            env = Environment(env_name)
        _settings = Settings(env)
    return _settings

def reload_settings():
    """Reload settings from configuration sources"""
    global _settings
    if _settings:
        _settings._load_configuration()
        logger.info("Settings reloaded")

def get_env() -> Environment:
    """Get current environment"""
    return get_settings().environment