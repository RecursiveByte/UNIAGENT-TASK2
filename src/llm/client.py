from langchain_openai import ChatOpenAI

from .settings import LLMSettings, load_settings


def build_client(settings: LLMSettings | None = None) -> ChatOpenAI:
    settings = settings or load_settings()

    model_kwargs = {}
    if settings.openai_reasoning_effort.strip():
        model_kwargs["reasoning_effort"] = settings.openai_reasoning_effort.strip()

    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        model_kwargs=model_kwargs,
    )