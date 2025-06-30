import os
from abc import abstractmethod, ABCMeta
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import Field, BaseModel, SecretStr

from spade_llm.core.conf import ConfigurableRecord


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
    def inject_env_dict(keys: list[str], conf: dict[str, Any], extra_params: dict[str, Any] = None):
        """
        For each key in keys list check if it is mentioned in dict and applies inject_env to its value.
        Used to substitute credentials from environment variables.
        You may provide extra arguments into config via extra_params.
        """
        for key in keys:
            if key in conf:
                if isinstance(conf[key], SecretStr):
                    conf[key] = SecretStr(CredentialsUtils.inject_env(conf[key].get_secret_value()))
                else:
                    conf[key] = CredentialsUtils.inject_env(conf[key])

        if extra_params:
            for key in extra_params.keys():
                conf[key] = CredentialsUtils.inject_env(extra_params[key])
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
    def create_model_factory(self) -> ChatModelFactory:
        instance = self.create_configurable_instance()
        return cast(ChatModelFactory, instance)


class EmbeddingsModelFactory[T: Embeddings](metaclass=ABCMeta):


    @abstractmethod
    def create_model(self) -> T:
        pass


class EmbeddingsModelConfiguration(LlmConfiguration):
    def create_model_factory(self) -> EmbeddingsModelFactory:
        instance = self.create_configurable_instance()
        return cast(EmbeddingsModelFactory, instance)


class ModelsProvider(metaclass=ABCMeta):
    @abstractmethod
    def create_chat_model(self, name):
        """
        Lookups configuration with given name and creates chat model using it.
        :param name: Name of the chat model to lookup
        :return: Instance of BaseChatModel configured according to configuration with given name
        """
        pass

    @abstractmethod
    def create_embeddings_model(self, name):
        """
        Lookups configuration with given name and creates embeddings model using it.
        :param name: Name of the embeddings model to lookup
        :return: Instance of embeddings configured according to configuration with given name
        """
        pass


class ModelsProviderConfig(BaseModel, ModelsProvider):
    chat_models: dict[str, ChatModelConfiguration] = Field(
        default=dict(),
        description="LLM models available to use as chat LLM"
    )

    embeddings_models: dict[str, EmbeddingsModelConfiguration] = Field(
        default=dict(),
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
        factory = self.chat_models[name].create_model_factory()
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
        factory = self.embeddings_models[name].create_model_factory()
        return factory.create_model()
