
from langchain_openai import ChatOpenAI

from .settings import LLMSettings, load_settings


def build_client(settings: LLMSettings | None = None) -> ChatOpenAI:
    settings = settings or load_settings()

    client_kwargs = {
        "base_url": settings.openai_base_url,
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
    }

    reasoning_effort = settings.openai_reasoning_effort.strip()

    if reasoning_effort:
        client_kwargs["model_kwargs"] = {
            "reasoning_effort": reasoning_effort
        }

    return ChatOpenAI(**client_kwargs)
