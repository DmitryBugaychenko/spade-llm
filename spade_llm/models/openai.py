from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.models import ChatModelFactory, EmbeddingsModelFactory, CredentialsUtils

SENSITIVE_KEYS=["access_token", "password", "key_file_password", "openai_api_key"]

class ChatOpenAIWithExtra(ChatOpenAI, extra="ignore"):
    pass

@configuration(ChatOpenAIWithExtra)
class ChatOpenAIModelFactory(ChatModelFactory[ChatOpenAI], Configurable[ChatOpenAI]):

    def create_model(self) -> ChatOpenAI:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True),
        )
        return ChatOpenAI(**config_dict)


@configuration(OpenAIEmbeddings)
class OpenAIEmbeddingsFactory(EmbeddingsModelFactory[OpenAIEmbeddings], Configurable[OpenAIEmbeddings]):
    def create_model(self) -> OpenAIEmbeddings:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True)
        )
        return config.model_copy(update=config_dict)