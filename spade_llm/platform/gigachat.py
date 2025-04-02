from spade_llm.platform.conf import configuration, Configurable
from spade_llm.platform.models import ChatModelFactory, ChatModelConfiguration

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat import GigaChatEmbeddings

class GigaChatModelConfig(ChatModelConfiguration):
    #TODO: Add all fields needed to initialize GigaChat using pydantic Field
    pass


@configuration(GigaChatModelConfig)
class GigaChatModelFactory(ChatModelFactory[GigaChat], Configurable[GigaChatModelConfig]):

    def __init__(self, config: GigaChatModelConfig):
        super().__init__()
        self._config = config

    def create_model(self) -> GigaChat:
        config: GigaChatModelConfig = self.config()
        # TODO: Create a new instance of GigaChat using values from config