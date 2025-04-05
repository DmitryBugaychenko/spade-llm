from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import Field, BaseModel

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import Configurable, configuration
from spade_llm.demo.hierarchy.agents import logger


class ReactAgentConfig(BaseModel):
    system_prompt: str = Field(
        description="System prompt for the agent. It should contain instructions for the agent on "
                    "what goal to persuade and how to behave.")
    model: str = Field(description="Model to use for handling messages.")
    max_iterations: int = Field(
        default=10, description="Maximum number of iterations with tools for handling one message")

class ReactBehaviour(MessageHandlingBehavior):

    def __init__(self,
                 template: MessageTemplate,
                 system_prompt: str,
                 model: BaseChatModel,
                 max_iterations: int):
        super().__init__(template)
        self.max_iterations = max_iterations
        self.model = model
        self.initial_message = [SystemMessage(system_prompt)]

    async def step(self):
        if not self.message:
            return

        self.logger.debug("Handling message %s", self.message)
        model = self.model.bind_tools(self.context.tools)
        tools_dict = {tool.name: tool for tool in self.context.tools}

        answers = []
        user_message = HumanMessage(self.message.content)

        for _ in range(self.max_iterations):
            self.logger.debug("Sending user message %s", user_message)
            answer = await model.ainvoke(
                self.initial_message + [user_message] + answers)
            self.logger.debug("Got answer %s", answer)

            answers.append(answer)

            if isinstance(answer, AIMessage) and answer.tool_calls:
                for tool_call in answer.tool_calls:
                    tool_name = tool_call["name"]
                    logger.debug("Invoking tool %s", tool_name)
                    if tool_name in tools_dict:
                        selected_tool = tools_dict[tool_name]
                        tool_answer = await selected_tool.ainvoke(tool_call["args"])
                        logger.debug("Tool %s answered %s", tool_name, tool_answer)
                        answers.append(ToolMessage(tool_answer, tool_call_id=tool_call["id"]))
                    else:
                        logger.warning("Tool %s not found", tool_name)
                        await self.context.reply_with_failure(self.message).with_content(f"Tool requested by model not found {tool_name}")
                        return
            else:
                logger.debug("Got final answer %s", answer)
                await self.context.reply_with_inform(self.message).with_content(answer.content)
                return

@configuration(ReactAgentConfig)
class ReactAgent(Agent, Configurable[ReactAgentConfig]):
    def setup(self):
        self.add_behaviour(ReactBehaviour(
            template = MessageTemplate.request(),
            system_prompt=self.config.system_prompt,
            model=self.default_context.create_chat_model(self.config.model),
            max_iterations=self.config.max_iterations
        ))
