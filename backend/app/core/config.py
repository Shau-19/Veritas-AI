from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)


class Settings(BaseSettings):
    app_name: str = "FineTunePlatform"
    secret_key: str = "change-me-in-production"
    debug: bool = True
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/finetune_db"
    redis_url: str = "redis://localhost:6379/0"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_bucket_name: str = "finetune-datasets"
    aws_region: str = "us-east-1"
    aws_role_arn: str = ""
    runpod_api_key: str = ""
    hf_token: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    groq_api_key: str = ""
    access_token_expire_minutes: int = 1440

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()