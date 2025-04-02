import os
from abc import abstractmethod, ABCMeta
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import Field

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
    name: str = Field(description="Name of the model used by agents to look up model in the pool")


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

