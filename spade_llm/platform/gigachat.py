import ssl

from spade_llm.platform.conf import configuration, Configurable
from spade_llm.platform.models import ChatModelFactory, ChatModelConfiguration
from pydantic import Field
from typing import List, Optional

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat import GigaChatEmbeddings

class GigaChatModelConfig(ChatModelConfiguration):
    temperature: Optional[float] = Field(default=None, description="What sampling temperature to use.")
    top_p: Optional[float] = Field(default=None, description="Total probability mass of tokens to consider at each step.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=False, description="Whether to stream back partial progress.")
    stop: Optional[List[str]] = Field(default=None, description="Sequences where the API will stop generating further tokens.")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens allowed for the generated answer.")
    presence_penalty: Optional[float] = Field(default=None, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.")
    frequency_penalty: Optional[float] = Field(default=None, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.")
    logit_bias: Optional[dict] = Field(default={}, description="Modify the likelihood of specified tokens appearing in the completion.")
    user: Optional[str] = Field(default="", description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.")

    base_url: Optional[str] = None
    """ Base API URL """
    auth_url: Optional[str] = None
    """ Auth URL """
    credentials: Optional[str] = None
    """ Auth Token """
    scope: Optional[str] = None
    """ Permission scope for access token """

    access_token: Optional[str] = None
    """ Access token for GigaChat """

    model: Optional[str] = None
    """Model name to use."""

    password: Optional[str] = None
    """ Password for authenticate """

    timeout: Optional[float] = None
    """ Timeout for request """
    verify_ssl_certs: Optional[bool] = None
    """ Check certificates for all requests """

    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None

    profanity: bool = True
    """ DEPRECATED: Check for profanity """
    profanity_check: Optional[bool] = None
    """ Check for profanity """
    streaming: bool = False
    """ Whether to stream the results or not. """
    use_api_for_tokens: bool = False
    """ Use GigaChat API for tokens count """
    verbose: bool = False
    """ Verbose logging """
    flags: Optional[List[str]] = None
    """ Feature flags """
    repetition_penalty: Optional[float] = None
    """ The penalty applied to repeated tokens """
    update_interval: Optional[float] = None
    """ Minimum interval in seconds that elapses between sending tokens """

@configuration(GigaChatModelConfig)
class GigaChatModelFactory(ChatModelFactory[GigaChat], Configurable[GigaChatModelConfig]):

    def __init__(self, config: GigaChatModelConfig):
        super().__init__()
        self._config = config

    def create_model(self) -> GigaChat:
        config_dict = self.config().dict(exclude_none=True)
        return GigaChat(**config_dict)
