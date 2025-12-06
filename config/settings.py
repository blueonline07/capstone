from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Traffic Volume API"
    threshold: int = 128


settings = Settings()
