from typing import Optional, List
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SERVICE_NAME: str = "gillo-ai-service"
    SERVICE_PORT: int = 8000
    ENVIRONMENT: str = "development"

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017/gillo_ai"
    MONGO_DB: str = "gillo_ai"

    # Core Agile Integration Mode (mock | live)
    CORE_AGILE_MODE: str = "mock"
    CORE_AGILE_API_URL: str = "http://localhost:3000"

    # JWT Authentication
    JWT_SECRET: str = "gillo_super_secret_dev_jwt_key_2026"
    JWT_ALGORITHM: str = "HS256"
    ALLOW_MOCK_AUTH: bool = False

    # ML Model Registry Configurations
    MODEL_REGISTRY_DIR: str = "models/artifacts"
    MODEL1_ACTIVE_VERSION: str = "v2_taskove"
    MODEL2_ACTIVE_VERSION: str = "v2_taskove_features"
    # Models 1 & 2 default to deterministic rule_baseline unless a non-synthetic
    # trained model is registered, or synthetic serving is explicitly enabled here:
    ENABLE_SYNTHETIC_MODELS: bool = False

    # LLM Provider Configuration ('template' | 'ollama' | 'cloud')
    LLM_PROVIDER: str = "template"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"  # Aligned with documented Model 3 architecture
    CLOUD_LLM_API_KEY: Optional[str] = None
    CLOUD_LLM_BASE_URL: str = "https://api.openai.com/v1"
    CLOUD_LLM_MODEL: str = "gpt-4o-mini"

    # Background Jobs Switch
    ENABLE_BACKGROUND_JOBS: bool = False

    # Estimation & Risk Parameters
    ESTIMATION_COLD_START_MIN_SAMPLES: int = 5
    DEFAULT_CAPACITY_HOURS: float = 60.0
    SPRINT_OVERLOAD_RATIO: float = 1.00
    SPRINT_TIGHT_RATIO: float = 1.00
    SPRINT_UNDERLOAD_RATIO: float = 0.70

    # Security, CORS & Rate Limiting
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    MAX_REQUEST_SIZE_BYTES: int = 2097152  # 2MB
    RATE_LIMIT_LLM_PER_MINUTE: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        """
        Security validations:
        1. Refuse insecure/unpinned JWT algorithms ('none' or unsupported algorithms).
        2. In production: refuse default, empty, or short (<32 chars) JWT_SECRET.
        3. In production: strictly refuse ALLOW_MOCK_AUTH=True.
        """
        allowed_algorithms = {
            "HS256", "HS384", "HS512",
            "RS256", "RS384", "RS512",
            "ES256", "ES384", "ES512",
        }
        if not self.JWT_ALGORITHM or self.JWT_ALGORITHM.upper() not in allowed_algorithms or self.JWT_ALGORITHM.lower() == "none":
            raise ValueError(
                f"FATAL: Insecure or unsupported JWT_ALGORITHM '{self.JWT_ALGORITHM}'. Allowed: {sorted(allowed_algorithms)}"
            )

        if self.ENVIRONMENT.lower() == "production":
            if "*" in self.CORS_ORIGINS:
                raise ValueError("FATAL: Wildcard CORS origin is forbidden in production.")
            if self.ALLOW_MOCK_AUTH:
                raise ValueError(
                    "FATAL: Production mode detected. ALLOW_MOCK_AUTH must be False in production."
                )
            insecure_defaults = {
                "gillo_super_secret_dev_jwt_key_2026",
                "dev-insecure-secret",
                "secret",
                "changeme",
                ""
            }
            if not self.JWT_SECRET or self.JWT_SECRET in insecure_defaults or len(self.JWT_SECRET) < 32:
                raise ValueError(
                    "FATAL: Production mode detected. You must set a non-default, secure 'JWT_SECRET' (at least 32 characters) in the environment."
                )
        return self


settings = Settings()
