import os
from abc import abstractmethod, ABCMeta
from email.policy import default
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import Field, BaseModel, ValidationError

from spade_llm.platform.conf import ConfigurableRecord


class CredentialsUtils:
    @staticmethod
    def inject_env(val: str):
        """
        Utility used to read credentials from environment variables if prefix "env." is provided
        """
        if val.startswith("env."):
            return os.getenv(val[4:])
        else:
            return val

    @staticmethod
    def inject_env_dict(keys: list[str], conf: dict[str, Any]):
        """
        For each key in keys list check if it is mentioned in dict and applies inject_env to its value.
        Used to substitute credentials from environment variables.
        """
        for key in keys:
            if key in conf:
                conf[key] = CredentialsUtils.inject_env(conf[key])
        return conf


class LlmConfiguration(ConfigurableRecord):
    """
    Base configuration class for LLM models
    """


class ChatModelFactory[T: BaseChatModel](metaclass=ABCMeta):
    @abstractmethod
    def create_model(self) -> T:
        pass


class ChatModelConfiguration(LlmConfiguration):
    def create_model(self) -> ChatModelFactory:
        instance = self.create_instance()
        return instance


class EmbeddingsModelFactory[T: Embeddings](metaclass=ABCMeta):
    @abstractmethod
    def create_model(self) -> T:
        pass


class EmbeddingsModelConfiguration(LlmConfiguration):
    def create_model(self) -> EmbeddingsModelFactory:
        instance = self.create_instance()
        return instance

class ModelsProviderConfig(BaseModel):
    chat_models: dict[str,ChatModelConfiguration] = Field(
        default = dict(),
        description="LLM models available to use as chat LLM"
    )

    embeddings_models: dict[str,EmbeddingsModelConfiguration] = Field(
        default = dict(),
        description="LLM models available to use for embeddings"
    )

    def create_chat_model(self, name: str) -> BaseChatModel:
        """
        Lookups configuration with given name and creates chat model using it.
        :param name: Name of the chat model to lookup
        :return: Instance of BaseChatModel configured according to configuration with given name
        """
        # Check if the specified chat model exists
        if name not in self.chat_models:
            raise ValueError(f"No such chat model '{name}' found.")
        
        # Get the corresponding factory and create the model
        factory = self.chat_models[name].create_model()
        return factory.create_model()

    def create_embeddings_model(self, name: str) -> Embeddings:
        """
        Lookups configuration with given name and creates embeddings model using it.
        :param name: Name of the embeddings model to lookup
        :return: Instance of embeddings configured according to configuration with given name
        """
        # Check if the specified embeddings model exists
        if name not in self.embeddings_models:
            raise ValueError(f"No such embeddings model '{name}' found.")
        
        # Get the corresponding factory and create the model
        factory = self.embeddings_models[name].create_model()
        return factory.create_model()
