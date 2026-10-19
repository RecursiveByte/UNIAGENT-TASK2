from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_base_url: str = Field(alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(alias="OPENAI_MODEL")
    openai_reasoning_effort: str = Field(default="", alias="OPENAI_REASONING_EFFORT")
                                                  
def load_settings() -> LLMSettings:
    return LLMSettings()