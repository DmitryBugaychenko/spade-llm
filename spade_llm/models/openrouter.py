from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel, SecretStr

from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.models import ChatModelFactory, CredentialsUtils

SENSITIVE_KEYS = ["openrouter_api_key"]


class OpenRouterChatModelConfig(BaseModel):
    model: str = Field(description="The model to use")
    openrouter_api_key: str = Field(description="OpenRouter API key")
    openai_api_base: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")
    temperature: float = Field(default=0.7, description="The temperature to use")
    max_tokens: int = Field(default=None, description="Maximum number of tokens to generate")
    timeout: int = Field(default=120, description="Timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum number of retries")


class OpenRouterChatModel(ChatOpenAI):
    def __init__(self, **kwargs):
        # Ensure the base URL is set to OpenRouter
        kwargs.setdefault('openai_api_base', 'https://openrouter.ai/api/v1')
        # Set default headers for OpenRouter
        headers = kwargs.get('default_headers', {})
        headers.setdefault('HTTP-Referer', 'https://github.com/spade-llm/spade-llm')
        headers.setdefault('X-Title', 'SPADE-LLM')
        kwargs['default_headers'] = headers
        
        # Map openrouter_api_key to openai_api_key for compatibility with ChatOpenAI
        if 'openrouter_api_key' in kwargs:
            kwargs['openai_api_key'] = kwargs.pop('openrouter_api_key')
        elif 'api_key' in kwargs:
            kwargs['openai_api_key'] = kwargs.pop('api_key')
            
        super().__init__(**kwargs)


@configuration(OpenRouterChatModelConfig)
class OpenRouterChatModelFactory(ChatModelFactory[OpenRouterChatModel], Configurable[OpenRouterChatModelConfig]):
    def create_model(self) -> OpenRouterChatModel:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True),
        )
        return OpenRouterChatModel(**config_dict)
