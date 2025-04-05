from langchain_gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat

from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.models import ChatModelFactory, EmbeddingsModelFactory, CredentialsUtils

SENSITIVE_KEYS=["access_token", "password", "key_file_password", "credentials"]

class GigaChatWithExtra(GigaChat, extra="ignore"):
    pass

@configuration(GigaChatWithExtra)
class GigaChatModelFactory(ChatModelFactory[GigaChat], Configurable[GigaChatWithExtra]):

    def create_model(self) -> GigaChat:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True),
        )
        return GigaChat(**config_dict)


@configuration(GigaChatEmbeddings)
class GigaChatEmbeddingsFactory(EmbeddingsModelFactory[GigaChatEmbeddings], Configurable[GigaChatEmbeddings]):
    def create_model(self) -> GigaChatEmbeddings:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True),
        )
        return config.model_copy(update=config_dict)