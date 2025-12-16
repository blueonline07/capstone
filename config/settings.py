from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Traffic Volume API"


settings = Settings()
