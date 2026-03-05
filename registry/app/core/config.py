from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Model Registry"
    env: str = "dev"

    database_url: str

    jwt_secret: str
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 720

    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str = "models"
    s3_region: str = "us-east-1"
    s3_secure: bool = False

    allow_anon_write: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()