from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from the .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    STORAGE_DIR: str = "./storage"
    # Comma-separated sites allowed to call this API from a browser. The local
    # dev servers are listed by default; add the deployed frontend's address
    # when you deploy. "*" is accepted but should not be used in production:
    # it lets any page on the internet call this API with a user's token.
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    APP_NAME: str = "DataDetective"
    DEBUG: bool = True

    # --- LLM (proposal Sec.16) -------------------------------------------
    # mock = works with no API key at all; everything runs, hypotheses are
    # template-generated. Switch to a real provider when you have a key.
    LLM_PROVIDER: str = "mock"          # mock | anthropic | openai | gemini
    LLM_MODEL: str = ""
    LLM_API_KEY: str = ""
    # Gemini 3.x and similar models reason before answering, and that reasoning
    # is charged against this budget. 2000 was enough for older models but can
    # be consumed entirely by thinking on newer ones.
    LLM_MAX_TOKENS: int = 8000
    # Evaluation runs make hundreds of calls in a row. A free-tier key will
    # rate-limit long before the run finishes, so the client retries with
    # backoff and can be paced to stay under a requests-per-minute cap.
    LLM_MAX_RETRIES: int = 5
    LLM_MIN_INTERVAL_MS: int = 0        # e.g. 4500 for a 15 requests/minute tier

    EMBEDDING_PROVIDER: str = "mock"    # mock | openai | gemini
    EMBEDDING_MODEL: str = ""
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIM: int = 256

    # --- Auth (proposal Sec.19) ------------------------------------------
    # Change SECRET_KEY in .env before any real deployment.
    SECRET_KEY: str = "change-me-in-env-file-before-deploying"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720

    # --- At-rest encryption (proposal Sec.19) ----------------------------
    # Off by default so an existing storage folder keeps working. When you turn
    # it on, generate a key with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    ENCRYPT_AT_REST: bool = False
    ENCRYPTION_KEY: str = ""

    # --- Safety limits (proposal Sec.19) ---------------------------------
    MAX_UPLOAD_MB: int = 100
    MAX_SQL_ROWS: int = 100_000
    TOOL_TIMEOUT_SECONDS: int = 60
    MAX_INVESTIGATION_ROUNDS: int = 5

    # --- Forecasting guard rails (see review note on Sec.12) -------------
    MIN_SERIES_LENGTH: int = 12         # below this: no forecast at all
    MIN_SERIES_FOR_SEASONAL: int = 24   # below this: no ARIMA/seasonal model
    MAX_ACCEPTABLE_MAPE: float = 25.0   # above this: flagged low_confidence


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()