import importlib
from typing import Self, Any

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

class Configurable[T](BaseModel):
    """
    Mixin used to provide access to the configuration for the configurable object
    """
    @property
    def config(self) -> T:
        """
        :return: Configuration to use
        """
        return self._config

    def _configure(self, config: BaseModel) -> Self:
        self._config = config
        self.configure()
        return self

    def configure(self):
        """
        Put any logic required to process configuration here. It will be called right
        after constructor.
        """

class Args(BaseModel, extra="allow"):
    """
    Placeholder to capture specific configuration attributes as extra attributes
    """

class ConfigurableRecord(BaseModel):
    type_name: str = Field(description="Name of the class to instantiate for this record")
    args: Args = Field(default=Args(), description="Arguments to pass to the created object")

    def _get_args(self):
        return self.args

    def _get_class(self):
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

        return cls

    def create_kwargs_instance(self, expected_class) -> Any:
        cls = self._get_class()

        # Ensure the class is derived from Configurable
        if not issubclass(cls, expected_class):
            raise TypeError(f"Class '{cls.__name__}' must inherit from '{expected_class.__name__}'.")

        # Create and configure the instance
        return cls(**(self._get_args().model_extra))

    def create_basemodel_instance(self) -> BaseModel:
        cls = self._get_class()

        # Ensure the class is derived from Configurable
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Class '{cls.__name__}' must inherit from 'BaseModel'.")

        # Create and configure the instance
        return cls.model_validate_json(self._get_args().model_dump_json())

    def create_configurable_instance(self) -> Configurable:
        cls = self._get_class()

        # Ensure the class is derived from Configurable
        if not issubclass(cls, Configurable):
            raise TypeError(f"Class '{cls.__name__}' must inherit from 'Configurable'.")

        # Verify that the class has a 'parse_config' class method
        if not hasattr(cls, 'parse_config') or not callable(getattr(cls, 'parse_config')):
            raise AttributeError(f"Class '{cls.__name__}' lacks a valid 'parse_config' class method.")

        # Create and configure the instance
        return cls()._configure(cls.parse_config(self._get_args().model_dump_json()))
