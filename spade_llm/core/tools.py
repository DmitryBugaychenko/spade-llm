import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, cast, Any, Type

from pydantic import Field, BaseModel
from langchain_core.tools import BaseTool, StructuredTool

from spade_llm import consts
from spade_llm.core.api import AgentContext, LocalToolFactory
from spade_llm.core.behaviors import BehaviorsOwner, ReceiverBehavior, MessageTemplate
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


class ArgsSchema(ConfigurableRecord):
    def get_schema(self) -> Type[BaseModel]:
        cls = self._get_class()
        # Ensure the class is derived from Configurable
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Class '{cls.__name__}' must inherit from 'BaseModel'.")

        return cls

logger = logging.getLogger("spade_llm.tools")

class SingleMessage(BaseModel):
    message: str = Field(description="Message to be sent to the agent")

class DelegateToolConfig(BaseModel, LocalToolFactory):
    """
    Tool used to delegate tasks to another agent
    """
    agent_type: str = Field(description="Type of the agent to delegate to")
    description: str = Field(description="Description of the service provided by the agent")
    performative: str = Field(default=consts.REQUEST,description="Performative to use when delegating")
    timeout: float = Field(default=60,description="Timeout for the agent response")
    args_schema: Optional[ArgsSchema] = Field(
        default=None,
        description="Schema of arguments to pass to the agent. If not provided, query will be passed as a plain string.")

    def create_tool(self, agent: BehaviorsOwner, context: AgentContext) -> BaseTool:
        schema = self.args_schema.get_schema() if self.args_schema else SingleMessage

        async def delegate(**kwargs: Any) -> str:
            logger.debug("Delegating message %s to agent %s with schema %s",
                         kwargs, self.agent_type, self.args_schema)

            args = schema(**kwargs)
            logger.debug("Parsed arguments %s", args)

            await (context
             .create_message_builder(self.performative)
             .to_agent(self.agent_type)
             .with_content(args))

            receiver = ReceiverBehavior(MessageTemplate(
                thread_id=context.thread_id,
                validator = MessageTemplate.from_agent(self.agent_type)
            ))
            agent.add_behaviour(receiver)

            try:
                await asyncio.wait_for(receiver.join(), self.timeout)
                return receiver.message.content
            except asyncio.TimeoutError:
                agent.remove_behaviour(receiver)
                return "Failed to receive a response from the agent due to timeout"


        logger.debug("Creating tool for agent %s with description %s and schema %s",
                     self.agent_type, self.description, str(schema))

        return StructuredTool.from_function(
            name=f"delegate_to_{self.agent_type}",
            coroutine=delegate,
            description=self.description,
            args_schema=schema,
            infer_schema=False,
            parse_docstring=False)