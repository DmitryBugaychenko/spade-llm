from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentContext
from spade_llm.core.behaviors import (
    MessageHandlingBehavior,
    MessageTemplate, ContextBehaviour
)
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import Union, Annotated, List, Tuple
from spade_llm.core.conf import Configurable, configuration
from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
import operator
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class PlanningAgentConfig(BaseModel):
    system_prompt: str = Field(
        description="System prompt for the agent. It should contain instructions for the agent on "
                    "what goal to persuade and how to behave."
    )
    react_node_prompt: str = Field(description="Prompt for the react node in the planning agent.")
    response_format: str = Field(description="Instructions for the agent on how to respond.")
    model: str = Field(description="Model to use for handling messages.")
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations with tools for handling one message",
    )
    recursion_limit: int = Field(description="Maximum recursion limit for graph", default=10)


class PlanExecute(TypedDict):
    """State container for planning/execution workflow.

    Attributes:
        input: Original user query
        plan: Current step-by-step plan
        past_steps: Completed steps with results
        response: Final response for user
        format_instructions: Output parsing instructions
    """
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
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use response. "
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
                For the given objective, come up with a simple step by step plan.\
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
            You can use the following tools: {tools}
            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text""",
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

    Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that using this format: {response_format}. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
    Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
    )

    def _handle_parser_errors(self):
        """Middleware to handle parser errors gracefully"""

        async def wrapper(msg):
            return await self.parser.ainvoke(msg.content)

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

    async def execute_step(self, state: PlanExecute):

        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        msg = f"""User query was {state['input']}.
                        You have already done the following steps:
                        {state['past_steps']}
                        For the following plan:
                        {plan_str}\n\nNow your task is to execute step '{1}'.
                        Perform calculations only for this step."""
        agent_response = ""
        answers = []
        current_task = HumanMessage(msg)
        for _ in range(self.max_iterations):
            mcc_prompt = [SystemMessage(self.config.react_node_prompt)]
            answer = await self.model.ainvoke(mcc_prompt + [current_task] + answers)
            self.logger.debug("Got answer %s", answer)

            answers.append(answer)

            if isinstance(answer, AIMessage) and answer.tool_calls:
                for tool_call in answer.tool_calls:
                    tool_name = tool_call["name"]
                    logger.debug("Invoking tool %s", tool_name)
                    if tool_name in self.tools_dict:
                        selected_tool = self.tools_dict[tool_name]
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

    async def plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke(
            {
                "messages": [("user", state["input"])],
                "system_prompt": self.initial_message,
                "query": self.user_message,
                "format_instructions": self.parser.get_format_instructions(),
                "tools": self.tools_dict,
            }
        )
        return {"plan": plan.steps}

    async def replan_step(self, state: PlanExecute):
        d = state.copy()
        d.update({"format_instructions": self.act_parser.get_format_instructions(),
                  "system_prompt": self.initial_message,
                  "response_format": self.config.response_format})
        output = await self.replanner.ainvoke(d)
        output = self.act_parser.parse(output.content)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    def should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"

    def create_graph(self):
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("agent", self.execute_step)
        workflow.add_node("replan", self.replan_step)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")
        workflow.add_conditional_edges(
            "replan",
            self.should_end,
            ["agent", END],
        )
        checkpointer = InMemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        return app

    async def step(self):
        tools = self.context.get_tools(self.agent)
        self.tools_dict = {tool.name: tool for tool in tools}
        self.model = self.model.bind_tools(tools)

        self.planner = self.planner_prompt | self.model | self._handle_parser_errors()
        self.replanner = self.replanner_prompt | self.model

        self.user_message = HumanMessage(self.message.content)

        self.logger.debug("Handling message %s", self.message)
        app = self.create_graph()
        graph_config = {"recursion_limit": self.config.recursion_limit,
                        "configurable": {"thread_id": self.context.thread_id}}
        inputs = {"input": self.message.content}

        graph_invoke = ExecuteLangGraphBehaviour(context=self.context,
                                            app=app,
                                            inputs=inputs,
                                            graph_config=graph_config)
        self.agent.add_behaviour(graph_invoke)
        await graph_invoke.join()

        result = await self.context.get_item('response')
        self.logger.info("Got MCC code %s", result)
        await self.context.reply_with_inform(self.message).with_content(result)

class ExecuteLangGraphBehaviour(ContextBehaviour):
    def __init__(self, context: AgentContext, app: CompiledStateGraph, inputs, graph_config):
        super().__init__(context)
        self.app = app
        self.graph_config = graph_config
        self.iterator = app.astream(inputs, config=graph_config).__aiter__()

    async def state_to_context(self, graph):
        snapshot_config = {"configurable": {"thread_id": self.context.thread_id}}
        snapshot = graph.get_state(snapshot_config)
        for k, v in snapshot.values.items():
            await self.context.put_item(k, v)

    async def step(self):
        try:
            event = await self.iterator.__anext__()
            await self.state_to_context(self.app)
        except StopAsyncIteration:
            self.set_is_done()


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
