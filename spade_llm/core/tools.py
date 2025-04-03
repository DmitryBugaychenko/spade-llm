from abc import ABCMeta, abstractmethod
from typing import Optional, cast

from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool

from spade_llm.core.conf import ConfigurableRecord, Configurable, configuration

class ToolConfigurationRecord(ConfigurableRecord):
    """
    Base configuration record for all tool providers
    """
    def create_factory(self) -> "ToolFactory":
        instance = self.create_configurable_instance()
        return cast(ToolFactory, instance)


class ToolProvider(metaclass=ABCMeta):
    @abstractmethod
    def create_tool(self, name):
        """
        Creates a new instance of tool with given name using configuration
        :param name: Name of the tool to create
        :return: Created tool.
        """
        pass


class ToolProviderConfig(BaseModel, ToolProvider):
    tools: dict[str,ToolConfigurationRecord] = Field(
        default = dict(),
        description="Dictionary with known tools and their configuration."
    )
    
    def create_tool(self, name: str) -> BaseTool:
        """
        Creates a new instance of tool with given name using configuration
        :param name: Name of the tool to create
        :return: Created tool.
        """
        tool_record = self.tools.get(name)
        
        if tool_record is None:
            raise ValueError(f"No tool found with name '{name}'")
            
        factory = tool_record.create_factory()
        return factory.create_tool()

class LangChainApiWrapperTool(ConfigurableRecord):
    """
    Configuration used to create built-in langchain tools. Most of them contains additional
    typed configuration for api_wrapper
    """
    api_wrapper: Optional[ConfigurableRecord] = Field(
        default=None,
        description="Holds extra configuration of API wrapper and its type"
    )

    def _get_args(self):
        if self.api_wrapper:
            if not self.args.__pydantic_extra__:
                self.args.__pydantic_extra__ = dict()
            if not "api_wrapper" in self.args.__pydantic_extra__:
                self.args.__pydantic_extra__["api_wrapper"] = self.api_wrapper.create_basemodel_instance()
        return self.args

class ToolFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_tool(self) -> BaseTool:
        pass

@configuration(LangChainApiWrapperTool)
class LangChainApiWrapperToolFactory(ToolFactory, Configurable[LangChainApiWrapperTool]):
    def create_tool(self) -> BaseTool:
        return self.config.create_kwargs_instance(BaseTool)
