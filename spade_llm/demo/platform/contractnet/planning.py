from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import (
    MessageHandlingBehavior,
    MessageTemplate
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import logging
import errno
import os
from pathlib import Path
from typing import Union, Annotated, List, Tuple
import asyncio
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from spade_llm.core.models import ChatModelConfiguration
from spade_llm.core.conf import Args
from spade_llm.demo import models
from langchain_core.vectorstores import VectorStore
from spade_llm.core.conf import ConfigurableRecord, Configurable, configuration
from pydantic import BaseModel, Field, ConfigDict
from spade_llm.core.tools import ToolFactory
from langchain_core.tools import tool, BaseTool, BaseToolkit
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
import operator
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

logger = logging.getLogger(__name__)


class PlanningAgentConfig(BaseModel):
    system_prompt: str = Field(
        description="System prompt for the agent. It should contain instructions for the agent on "
                    "what goal to persuade and how to behave."
    )
    model: str = Field(description="Model to use for handling messages.")
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations with tools for handling one message",
    )
    recursion_limit: int = Field(description="Maximum recursion limit for graph", default=10)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    format_instructions: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user. Должен быть в виде одного числа"""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use esponse. "
                    "If you need to further use tools to get the answer, use Plan."
    )


class PlanningBehaviour(MessageHandlingBehavior):
    parser = PydanticOutputParser(pydantic_object=Plan)
    act_parser = PydanticOutputParser(pydantic_object=Act)
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{system_prompt} 
                For the given objective, come up with a simple step by step plan. (напиши его на русском языке) \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            You can use the following tools: {tools}
            Respond in `json` format\n{format_instructions}""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    replanner_prompt = ChatPromptTemplate.from_template(
        """{system_prompt} 
        For the given objective, come up with a simple step by step plan. \
    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
    The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

    Your objective is this:
    {input}

    Your original plan was this:
    {plan}

    You have currently done the follow steps:
    {past_steps}

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
    Respond in `json` format\n{format_instructions}"""
    )

    def _handle_parser_errors(self):
        """Middleware to handle parser errors gracefully"""

        async def wrapper(msg):
            input_text = msg.content
            try:
                return await self.parser.ainvoke(input_text)
            except Exception as e:
                # Try to extract JSON from response if possible
                try:
                    json_start = input_text.find("{")
                    json_end = input_text.rfind("}") + 1
                    json_str = input_text[json_start:json_end]
                    return await self.parser.ainvoke(json_str)
                except Exception:
                    # Fallback to manual parsing if automatic fails
                    steps = []
                    for line in input_text.split("\n"):
                        steps.append(line.strip())
                    return Plan(steps=steps if steps else ["Could not parse plan"])

        return wrapper

    def __init__(
            self,
            template: MessageTemplate,
            config: PlanningAgentConfig,
            system_prompt: str,
            model: BaseChatModel,
            max_iterations: int,
    ):
        super().__init__(template)
        self.max_iterations = max_iterations
        self.model = model
        self.initial_message = [SystemMessage(system_prompt)]
        self.config = config

    async def step(self):

        self.logger.debug("Handling message %s", self.message)
        tools = self.context.get_tools(self.agent)
        model = self.model.bind_tools(tools)
        tools_dict = {tool.name: tool for tool in tools}
        user_message = HumanMessage(self.message.content)

        planner = self.planner_prompt | model | self._handle_parser_errors()
        replanner = self.replanner_prompt | model

        async def state_to_context(graph):
            config = {"configurable": {"thread_id": self.context.thread_id}}
            snapshot = graph.get_state(config)
            for k, v in snapshot.values.items():
                await self.context.put_item(k, v)

        async def execute_step(state: PlanExecute):

            plan = state["plan"]
            plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
            task = plan[0]

            msg = f"""User query was {state['input']}.
                            You have already done the following steps:
                            {state['past_steps']}
                            For the following plan:
                            {plan_str}\n\nYou are tasked with executing step {1}, {task}.
                            Проводи вычисления только для этого шага."""
            agent_response = ""
            answers = []
            user_message = HumanMessage(msg)

            for _ in range(self.max_iterations):
                mcc_prompt = [SystemMessage("""You are a personal assistant helping with data about mcc codes.
                                    Use provided tools to solve user tasks.""")]
                answer = await model.ainvoke(mcc_prompt + [user_message] + answers)
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
                            agent_response = f"Tool requested by model not found {tool_name}"
                else:
                    logger.debug("Got final answer %s", answer)
                    agent_response = answer.content
                    break

            t = state["past_steps"] + ["На запрос:" + task + "\nОтвет:" + agent_response]
            return {"past_steps": t}

        async def plan_step(state: PlanExecute):
            plan = await planner.ainvoke(
                {
                    "messages": [("user", state["input"])],
                    "system_prompt": self.initial_message,
                    "query": user_message,
                    "format_instructions": self.parser.get_format_instructions(),
                    "tools": tools_dict,
                }
            )
            return {"plan": plan.steps}

        async def replan_step(state: PlanExecute):
            d = state.copy()
            d.update({"format_instructions": self.act_parser.get_format_instructions(),
                      "system_prompt": self.initial_message})
            output = await replanner.ainvoke(d)
            output = self.act_parser.parse(output.content)
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}

        def should_end(state: PlanExecute):
            if "response" in state and state["response"]:
                return END
            else:
                return "agent"

        def create_graph():
            workflow = StateGraph(PlanExecute)
            workflow.add_node("planner", plan_step)
            workflow.add_node("agent", execute_step)
            workflow.add_node("replan", replan_step)
            workflow.add_edge(START, "planner")
            workflow.add_edge("planner", "agent")
            workflow.add_edge("agent", "replan")
            workflow.add_conditional_edges(
                "replan",
                should_end,
                ["agent", END],
            )
            checkpointer = InMemorySaver()
            app = workflow.compile(checkpointer=checkpointer)
            return app

        app = create_graph()
        config = {"recursion_limit": self.config.recursion_limit, "configurable": {"thread_id": self.context.thread_id}}
        inputs = {"input": self.message.content}
        result = ""

        async for event in app.astream(inputs, config=config):
            for k, v in event.items():
                if k != "__end__":
                    if "response" in v:
                        result = v["response"]
                        break
            await state_to_context(app)

        self.logger.info("Got MCC code %s", result)
        await self.context.reply_with_inform(self.message).with_content(result)


@configuration(PlanningAgentConfig)
class PlanningAgent(Agent, Configurable[PlanningAgentConfig]):
    def setup(self):
        self.add_behaviour(
            PlanningBehaviour(
                template=MessageTemplate.request(),
                config=self.config,
                system_prompt=self.config.system_prompt,
                model=self.default_context.create_chat_model(self.config.model),
                max_iterations=self.config.max_iterations,
            )
        )


class MccToolkitConf(ConfigurableRecord):
    pass


class MccDescription(BaseModel):
    description: str = Field(description="Описание категории транзакций")



class MccToolkit(BaseToolkit):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: BaseChatModel = Field(default=None, description="LLM model")
    index: VectorStore = Field(default=None, description="Database with MCC codes")

    def __init__(self):
        super().__init__()
        conf = {
            "max": ChatModelConfiguration(
                type_name="spade_llm.models.gigachat.GigaChatModelFactory",
                args=Args(
                    credentials="env.GIGA_CRED",
                    scope="env.GIGACHAT_SCOPE",
                    model="GigaChat-2-Max",
                    verify_ssl_certs=False,
                ),
            )
        }
        self.model = conf["max"].create_model_factory().create_model()
        self.index = Chroma(
            embedding_function=models.EMBEDDINGS, collection_name="MCC"
        )

    async def prepare_data(self) -> None:
        logger.info("Preparing data")
        file_path = Path("./data").joinpath("mcc_codes.csv")

        if not file_path.is_file() or not file_path.exists():
            logger.error("File with MCC code description not found at %s", file_path)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path)
            )

        logger.info("Loading MCC from %s", file_path)
        loader = CSVLoader(
            file_path=file_path.resolve(),
            metadata_columns=["MCC"],
            csv_args={"delimiter": ",", "quotechar": '"'},
        )

        documents = await loader.aload()
        logger.info("Loaded %i codes", len(documents))
        filtered = [x for x in documents if len(x.page_content) > 100]
        logger.info("%i codes have enough description", len(documents))
        await asyncio.sleep(1)
        await self.index.aadd_documents(filtered)
        logger.info("Codes added to index")

    async def async_setup(self):
        await self.prepare_data()

    @property
    def get_tools(self) -> List[BaseTool]:
        """Returns a list of tools"""
        asyncio.create_task(self.async_setup())
        lst = [
            self._create_mcc_descriptor(),
            self._create_mcc_finder(),
        ]
        return lst

    def _create_mcc_descriptor(self) -> BaseTool:
        parser = PydanticOutputParser(pydantic_object=MccDescription)
        prepare_request_prompt = PromptTemplate.from_template(
            """Your task is to formulate a description of an MCC code for transactions representing spending by people 
            who fall under the characteristic "{query}". The description must meet the following criteria:
            * At least 40 words long
            * No time-specific references
            * Includes names of goods and services related to the activity
            * Includes types of retail outlets and businesses where these goods and services are sold
            Respond in `json` format\n{format_instructions}"""
        )

        @tool
        async def mcc_descriptor(query: str):
            """Returns a detailed description of a spending category based on the query.

            Args:
                query: User query (e.g., 'Restaurant visitors') as a string.

            Returns:
                Detailed description of the spending category based on the query

            Examples:
                >>> mcc_descriptor("People who go to bars")  # Returns a detailed description of the Bars category
                >>> mcc_descriptor("Restaurants")  # Search by category name
            """
            chain = self.model | parser
            request: MccDescription = await chain.ainvoke(
                await prepare_request_prompt.ainvoke(
                    {
                        "query": query,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
            )
            return request.description

        return mcc_descriptor

    def _create_mcc_finder(self) -> BaseTool:
        @tool
        async def mcc_finder(description: str):
            """Finds appropriate MCC codes based on business activity description.
            Before using this tool, call mcc_descriptor to get the MCC code description.

            Args:
                description: Business or activity description (e.g., 'clothing retail') This parameter should be very detailed.

            Returns:
                Appropriate MCC code as a number

            Examples:
                >>> mcc_finder("clothing retail")  # Find codes for clothing retail
                >>> mcc_finder("taxi services")  # Find codes for transportation services
            """
            result = await self.index.asimilarity_search(query=description, k=1)
            doc = result[0]
            return str(doc.metadata["MCC"])

        return mcc_finder


@configuration(MccToolkitConf)
class MccToolFactory(ToolFactory, Configurable[MccToolkitConf]):
    def create_tool(self) -> List[BaseTool]:
        return self.config.create_kwargs_instance(MccToolkit).get_tools
