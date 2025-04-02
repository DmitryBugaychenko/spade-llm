import importlib
from typing import Self

from pydantic import BaseModel, Field


def configuration(config: type[BaseModel]):
    """
    Use this method as decorator to provide reference to pydantic model used to parse
    configuration for the class. Example:

    @configuration(StringConfig)
    class StringConfigurable(Configurable[StringConfig]):
        pass
    """
    def decorator(cls):
        class Configured(cls):
            @classmethod
            def parse_config(cls, json: str) -> BaseModel:
                return config.model_validate_json(json)

        return Configured

    return decorator

class Configurable[T: BaseModel]:
    """
    Mixin used to provide access to the configuration for the configurable object
    """
    def config(self) -> T:
        """
        :return: Configuration to use
        """
        return self._config

    def _configure(self, config: T) -> Self:
        self._config = config
        self.configure()
        return self

    def configure(self):
        """
        Put any logic required to process configuration here. It will be called right
        after constructor.
        """


class ConfigurableRecord(BaseModel, extra="allow"):
    type_name: str = Field(description="Name of the class to instantiate for this record")

    def create_instance(self) -> Configurable:
        module_name, class_name = self.type_name.rsplit(".", 1)
        my_class = getattr(importlib.import_module(module_name), class_name)
        return my_class()._configure(my_class.parse_config(self.model_dump_json()))