import importlib
from typing import TypeVar, Callable, Any, Optional
from pydantic import BaseModel, Field

T = TypeVar('T', bound='Configurable')

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

class Configurable(BaseModel):
    """
    Mixin used to provide access to the configuration for the configurable object
    """
    def config(self) -> BaseModel:
        """
        :return: Configuration to use
        """
        return self._config

    def _configure(self, config: BaseModel) -> 'Configurable':
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
        # Check if type_name follows the proper format
        if '.' not in self.type_name:
            raise ValueError(f"type_name '{self.type_name}' must follow the format '<module>.<class>'")

        module_name, class_name = self.type_name.rsplit('.', 1)

        # Try importing the specified module
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(f"Module '{module_name}' could not be imported.")

        # Get the class from the module
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"Class '{class_name}' does not exist in module '{module_name}'.")

        # Ensure the class is derived from Configurable
        if not issubclass(cls, Configurable):
            raise TypeError(f"Class '{cls.__name__}' must inherit from 'Configurable'.")

        # Verify that the class has a 'parse_config' class method
        if not hasattr(cls, 'parse_config') or not callable(getattr(cls, 'parse_config')):
            raise AttributeError(f"Class '{cls.__name__}' lacks a valid 'parse_config' class method.")

        # Create and configure the instance
        return cls()._configure(cls.parse_config(self.model_dump_json()))
