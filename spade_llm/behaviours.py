import abc
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour
from spade.message import Message
from spade.template import Template

logger = logging.getLogger(__name__)

class SendAndReceiveBehaviour(OneShotBehaviour):
    """
    Simple utility used to send message and wait for response&
    """
    message: Message
    response_template: Template
    response: Message

    def __init__(self, message: Message, response_template: Template):
        super().__init__()
        self.message = message
        self.response_template = response_template
        self.response = None

    async def run(self):
        await self.send(self.message)
        logger.info("Send message %s", self.message)
        self.response = await self.receive(60)
        logger.info("Received response %s", self.response)

class RequestHandlingBehaviour(FSMBehaviour, metaclass=abc.ABCMeta):
    """
    Performs simple cycle for LLM agent
    1. Get prompt to handle
    2. Call LLM
    3. If LLM produced tool calls invoke tools and get back to step 2
    4. If no tool calls produced consider answer as final
    """
    user_message: HumanMessage = None
    model: BaseChatModel = None
    tools: dict[str, BaseTool] = None
    context: list[BaseMessage]
    answers: list[BaseMessage] = []
    final_answer: BaseMessage = None

    def __init__(self, model, context, tools):
        self.user_message = None
        self.model = model
        self.context = context
        self.tools = tools
        self.answers = []
        self.final_answer = None
        super().__init__()

    @abc.abstractmethod
    async def get_prompt(self) -> BaseMessage:
        return None

    @abc.abstractmethod
    async def handle_answer(self, answer: BaseMessage):
        return None

    def setup(self) -> None:
        initialize_handler(self)


class RequestHandlingState(State, metaclass=abc.ABCMeta):
    parent: RequestHandlingBehaviour = None
    def __init__(self, parent: RequestHandlingBehaviour):
        super().__init__()
        self.parent = parent

class GetPrompt(RequestHandlingState):
    prompt: str = ""

    async def run(self):
        prompt = await self.parent.get_prompt()
        if prompt:
            logger.info("Got prompt %s", prompt)
            self.parent.user_message = prompt
            self.set_next_state("SEND_REQUEST")
        else:
            logger.info("Got empty prompt, exiting")
            self.set_next_state("")

class SendRequest(RequestHandlingState):
    async def run(self):
        logger.info("sending request")
        answer = await self.parent.model.ainvoke(
            self.parent.context + [self.parent.user_message] + self.parent.answers)
        self.parent.answers.append(answer)
        if answer.tool_calls:
            self.set_next_state("INVOKE_TOOLS")
        else:
            self.parent.answers = []
            self.parent.final_answer = answer
            # With growing context agent become unstable in tools usage :(
            #self.parent.context += [self.parent.user_message, self.parent.final_answer]
            logger.info("Got final answer %s", answer)
            await self.parent.handle_answer(answer)
            self.set_next_state("GET_PROMPT")

class InvokeTools(RequestHandlingState):
    async def run(self):
        for tool_call in self.parent.answers[-1].tool_calls:
            selected_tool = self.parent.tools[tool_call["name"].lower()]
            logger.info("Invoking tool " + selected_tool.name)
            answer = await selected_tool.ainvoke(tool_call["args"])
            self.parent.answers.append(ToolMessage(answer, tool_call_id=tool_call["id"]))
        self.set_next_state("SEND_REQUEST")


def initialize_handler(fsm):
    fsm.add_state(name="GET_PROMPT", state=GetPrompt(fsm), initial=True)
    fsm.add_state(name="SEND_REQUEST", state=SendRequest(fsm))
    fsm.add_state(name="INVOKE_TOOLS", state=InvokeTools(fsm))
    fsm.add_transition("GET_PROMPT", "SEND_REQUEST")
    fsm.add_transition("SEND_REQUEST", "GET_PROMPT")
    fsm.add_transition("SEND_REQUEST", "INVOKE_TOOLS")
    fsm.add_transition("INVOKE_TOOLS", "SEND_REQUEST")
    return fsm
