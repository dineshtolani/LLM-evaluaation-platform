from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "LLM Eval Platform"
    debug: bool = True

    database_url: str = "postgresql+asyncpg://llmeval:llmeval@localhost:5432/llmevaldb"
    database_url_sync: str = "postgresql://llmeval:llmeval@localhost:5432/llmevaldb"

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "llm_evaluation"

    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"

    redis_url: str = "redis://localhost:6379/0"

    grafana_url: str = "http://localhost:3000"

    prompt_cost_per_token: float = 0.000003
    completion_cost_per_token: float = 0.000015

    class Config:
        env_file = ".env"


settings = Settings()
