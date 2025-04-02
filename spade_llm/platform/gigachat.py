import ssl

from spade_llm.platform.conf import configuration, Configurable
from spade_llm.platform.models import ChatModelFactory, ChatModelConfiguration, EmbeddingsModelConfiguration
from pydantic import Field
from typing import List, Optional

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat import GigaChatEmbeddings

class GigaChatModelConfig(ChatModelConfiguration):
    """
    Configuration for GigaChat connection.
    """
    temperature: Optional[float] = Field(default=None, description="Sampling temperature for controlling randomness during generation.")
    top_p: Optional[float] = Field(default=None, description="Probability threshold for nucleus sampling.")
    n: Optional[int] = Field(default=None, description="Number of completions to generate per prompt.")
    stream: Optional[bool] = Field(default=False, description="Enable streaming mode for real-time response updates.")
    stop: Optional[List[str]] = Field(default=None, description="List of sequences marking the end of a completion.")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to include in the generated response.")
    presence_penalty: Optional[float] = Field(default=None, description="Penalty factor for discouraging repetitive responses.")
    frequency_penalty: Optional[float] = Field(default=None, description="Penalty factor for reducing frequent word usage.")
    logit_bias: Optional[dict] = Field(default={}, description="Custom bias adjustments for specific tokens.")
    user: Optional[str] = Field(default="", description="Unique identifier associated with the requesting user.")
    
    base_url: Optional[str] = Field(default=None, description="Root URL for accessing Gigachat services.")
    auth_url: Optional[str] = Field(default=None, description="URL endpoint for authentication purposes.")
    credentials: Optional[str] = Field(default=None, description="Authentication credentials required for authorization.")
    scope: Optional[str] = Field(default=None, description="Scope defining permissions granted upon successful authentication.")
    
    access_token: Optional[str] = Field(default=None, description="Access token obtained after authenticating successfully.")
    
    model: Optional[str] = Field(default=None, description="Name of the pre-trained model being utilized.")
    
    password: Optional[str] = Field(default=None, description="Password string used alongside username for basic authentication.")
    
    timeout: Optional[float] = Field(default=None, description="Timeout duration (in seconds) before cancelling pending requests.")
    verify_ssl_certs: Optional[bool] = Field(default=None, description="Flag indicating whether SSL certificate verification should occur.")
    
    ca_bundle_file: Optional[str] = Field(default=None, description="Path to custom CA bundle file containing trusted root certificates.")
    cert_file: Optional[str] = Field(default=None, description="Path to client-side TLS certificate file.")
    key_file: Optional[str] = Field(default=None, description="Path to private key corresponding to the client-side certificate.")
    key_file_password: Optional[str] = Field(default=None, description="Optional password protecting encrypted private keys.")
    
    profanity: bool = Field(default=True, description="DEPRECATED: Flag enabling automatic detection and filtering of offensive language.")
    profanity_check: Optional[bool] = Field(default=None, description="Enables explicit checking for potentially inappropriate content.")
    streaming: bool = Field(default=False, description="Indicates preference towards receiving incremental data streams instead of complete outputs.")
    use_api_for_tokens: bool = Field(default=False, description="Specifies reliance on external APIs rather than local methods when counting tokens.")
    verbose: bool = Field(default=False, description="Controls level of detail included within logs and diagnostic information.")
    flags: Optional[List[str]] = Field(default=None, description="Collection of feature-specific activation switches.")
    repetition_penalty: Optional[float] = Field(default=None, description="Factor influencing how heavily penalties apply against repeating patterns.")
    update_interval: Optional[float] = Field(default=None, description="Minimum time gap enforced between consecutive updates sent via streaming channels.")

@configuration(GigaChatModelConfig)
class GigaChatModelFactory(ChatModelFactory[GigaChat], Configurable[GigaChatModelConfig]):


    def create_model(self) -> GigaChat:
        config = self.config
        config_dict = config.inject_env_dict(
            keys = ["access_token", "password", "key_file_password", "credentials"],
            conf = config.model_dump(exclude_none=True))
        return GigaChat(**config_dict)

class GigaChatEmbeddingsConf(EmbeddingsModelConfiguration):
    base_url: Optional[str] = Field(None, description="Base API URL")
    auth_url: Optional[str] = Field(None, description="Auth URL")
    credentials: Optional[str] = Field(None, description="Auth Token")
    scope: Optional[str] = Field(None, description="Permission scope for access token")
    access_token: Optional[str] = Field(None, description="Access token for GigaChat")
    model: Optional[str] = Field(None, description="Model name to use.")
    user: Optional[str] = Field(None, description="Username for authenticate")
    password: Optional[str] = Field(None, description="Password for authenticate")
    timeout: Optional[float] = Field(600, description="Timeout for request. By default it works for long requests.")
    verify_ssl_certs: Optional[bool] = Field(None, description="Check certificates for all requests")
    ca_bundle_file: Optional[str] = Field(None, description="CA bundle file for SSL verification")
    cert_file: Optional[str] = Field(None, description="Client-side TLS certificate file")
    key_file: Optional[str] = Field(None, description="Private key corresponding to the client-side certificate")
    key_file_password: Optional[str] = Field(None, description="Password for encrypted private key")
    prefix_query: str = Field("Дано предложение, необходимо найти его парафраз \nпредложение: ", description="Prefix query for paraphrase task")
    use_prefix_query: bool = Field(False, description="Use prefix query flag") 
