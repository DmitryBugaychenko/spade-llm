import asyncio
import random
import unittest
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.api import (
    AgentContext,
    AgentId,
    Message,
    MessageBuilder,
    KeyValueStorage,
)
from spade_llm.core.behaviors import (
    Behaviour,
    MessageHandlingBehavior,
    MessageTemplate,
)
from spade_llm.core.storage import InMemoryKeyValueStorage
from spade_llm.core.context import AgentContextImpl
from spade_llm import consts


NUM_RESPONDERS = 5


class ContractNetRequest(BaseModel):
    task: str = Field(description="Description of the task to create proposal for")


class ContractNetProposal(BaseModel):
    author: str = Field(description="Agent created this proposal")
    request: ContractNetRequest = Field(description="Request this proposal is created for")
    estimate: float = Field(description="Estimation for the task")


class AgentRegistry:
    """Simple registry that allows agents to send messages to each other."""

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._contexts: dict[str, AgentContext] = {}

    def register(self, agent_type: str, agent: Agent, context: AgentContext):
        self._agents[agent_type] = agent
        self._contexts[agent_type] = context

    async def send_message(self, msg: Message):
        receiver_type = msg.receiver.agent_type
        if receiver_type in self._agents:
            agent = self._agents[receiver_type]
            context = self._contexts[receiver_type]
            await agent.handle_message(context, msg)

    def get_agent(self, agent_type: str) -> Agent:
        return self._agents[agent_type]


class SimpleContext(AgentContext):
    """Minimal AgentContext implementation for testing."""

    def __init__(self, agent_type: str, agent_id: str, registry: AgentRegistry,
                 thread_id: Optional[uuid.UUID] = None):
        self._agent_type = agent_type
        self._agent_id = agent_id
        self._registry = registry
        self._thread_id = thread_id
        self._kv = InMemoryKeyValueStorage()

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def thread_id(self) -> Optional[uuid.UUID]:
        return self._thread_id

    @property
    def thread_context(self) -> KeyValueStorage:
        return self._kv

    async def fork_thread(self) -> "AgentContext":
        return SimpleContext(self._agent_type, self._agent_id, self._registry,
                             thread_id=uuid.uuid4())

    async def close_thread(self) -> "AgentContext":
        return SimpleContext(self._agent_type, self._agent_id, self._registry)

    async def send(self, msg: Message):
        await self._registry.send_message(msg)

    async def get_item(self, key: str) -> str:
        return await self._kv.get_item(key)

    async def put_item(self, key: str, value: Optional[str]) -> None:
        await self._kv.put_item(key, value)

    async def close(self):
        await self._kv.close()

    def get_tools(self, agent=None) -> list:
        return []

    def chat_model(self):
        return None

    def embeddings_model(self):
        return None


class ResponderBehavior(MessageHandlingBehavior):
    """
    Responder behavior that handles CFP messages and sends back proposals.
    Mimics ContractNetResponderBehavior but uses spade_llm.core classes.
    """

    def __init__(self, agent_type: str, registry: AgentRegistry):
        super().__init__(template=MessageTemplate(performative=consts.REQUEST_PROPOSAL))
        self._agent_type = agent_type
        self._registry = registry

    async def step(self):
        msg = self.message
        if msg is None:
            return

        request = ContractNetRequest.model_validate_json(msg.content)
        estimate = random.uniform(1.0, 100.0)
        proposal = ContractNetProposal(
            author=self._agent_type,
            request=request,
            estimate=estimate,
        )

        response = Message(
            sender=AgentId(agent_type=self._agent_type, agent_id=self._agent_type),
            receiver=msg.sender,
            thread_id=msg.thread_id,
            performative=consts.PROPOSE,
            content=proposal.model_dump_json(),
        )
        await self._registry.send_message(response)

    def is_done(self) -> bool:
        return False


class InitiatorBehavior(Behaviour):
    """
    Initiator behavior that sends CFPs to multiple responders and collects proposals.
    Reproduces the same pattern as ContractNetInitiatorBehavior.get_proposals().
    """

    def __init__(self, task: str, responder_types: list[str], registry: AgentRegistry,
                 initiator_type: str, thread_id: uuid.UUID,
                 time_to_wait: float = 5.0):
        super().__init__()
        self.task = task
        self.responder_types = responder_types
        self.registry = registry
        self.initiator_type = initiator_type
        self.thread_id = thread_id
        self.time_to_wait = time_to_wait
        self.collected_proposals: list[ContractNetProposal] = []
        self._executed = False

    async def step(self):
        if self._executed:
            self.set_is_done()
            return

        self._executed = True

        request = ContractNetRequest(task=self.task)

        # Send CFP to all responders
        for resp_type in self.responder_types:
            cfp = Message(
                sender=AgentId(agent_type=self.initiator_type, agent_id=self.initiator_type),
                receiver=AgentId(agent_type=resp_type, agent_id=resp_type),
                thread_id=self.thread_id,
                performative=consts.REQUEST_PROPOSAL,
                content=request.model_dump_json(),
            )
            await self.registry.send_message(cfp)

        # Now collect proposals one at a time — this is the pattern from
        # ContractNetInitiatorBehavior.get_proposals() that has the race condition.
        # Each receive() creates a new ReceiverBehavior. If messages arrive
        # before the next receive() call, they have no matching behavior and are lost.
        received_count = 0
        template = MessageTemplate(
            thread_id=self.thread_id,
            performative=consts.PROPOSE,
        )

        while received_count < len(self.responder_types):
            response = await self.receive(template, timeout=self.time_to_wait)
            if response is None:
                # Timeout — stop waiting
                break
            received_count += 1
            proposal = ContractNetProposal.model_validate_json(response.content)
            self.collected_proposals.append(proposal)

        self.set_is_done()


class ResponderAgent(Agent):
    """Agent that hosts a ResponderBehavior."""

    def __init__(self, agent_type: str, registry: AgentRegistry):
        super().__init__(agent_type)
        self._registry = registry

    def setup(self):
        self.add_behaviour(ResponderBehavior(self._agent_type, self._registry))


class InitiatorAgent(Agent):
    """Agent that hosts an InitiatorBehavior."""

    def __init__(self, agent_type: str, registry: AgentRegistry,
                 responder_types: list[str], thread_id: uuid.UUID):
        super().__init__(agent_type)
        self._registry = registry
        self._responder_types = responder_types
        self._thread_id = thread_id
        self.initiator_behavior: Optional[InitiatorBehavior] = None

    def setup(self):
        self.initiator_behavior = InitiatorBehavior(
            task="Handle a generic task",
            responder_types=self._responder_types,
            registry=self._registry,
            initiator_type=self._agent_type,
            thread_id=self._thread_id,
            time_to_wait=5.0,
        )
        self.add_behaviour(self.initiator_behavior)


class ContractNetRaceConditionTest(unittest.TestCase):
    """
    Test that reproduces a race condition in the contract net proposal collection pattern.

    When multiple responders reply nearly simultaneously, proposals can be lost
    because the receive() pattern creates a ReceiverBehavior one at a time.
    Messages arriving between receive() calls have no matching behavior and are dropped.
    """

    def test_all_proposals_collected_from_all_responders(self):
        """
        Verify that the initiator collects proposals from ALL responders.

        Creates NUM_RESPONDERS responder agents that each generate a random estimate,
        and one initiator agent that sends CFPs and collects proposals.
        The test asserts that all proposals are received.
        """
        registry = AgentRegistry()
        thread_id = uuid.uuid4()

        responder_types = [f"responder_{i}" for i in range(NUM_RESPONDERS)]

        # Create and start responder agents
        responder_agents = []
        for resp_type in responder_types:
            ctx = SimpleContext(resp_type, resp_type, registry)
            agent = ResponderAgent(resp_type, registry)
            registry.register(resp_type, agent, ctx)
            agent.start(ctx)
            responder_agents.append(agent)

        # Create and start initiator agent
        initiator_type = "initiator"
        initiator_ctx = SimpleContext(initiator_type, initiator_type, registry,
                                      thread_id=thread_id)
        initiator_agent = InitiatorAgent(initiator_type, registry, responder_types, thread_id)
        registry.register(initiator_type, initiator_agent, initiator_ctx)
        initiator_agent.start(initiator_ctx)

        # Wait for initiator to complete
        async def wait_for_completion():
            await initiator_agent.initiator_behavior.join()

        try:
            loop = initiator_agent.loop
            future = asyncio.run_coroutine_threadsafe(wait_for_completion(), loop)
            future.result(timeout=15)
        except Exception as e:
            self.fail(f"Initiator did not complete in time: {e}")
        finally:
            # Stop all agents
            for agent in responder_agents:
                agent.stop()
            initiator_agent.stop()

            async def join_all():
                for agent in responder_agents:
                    await agent.join()
                await initiator_agent.join()

            asyncio.run(join_all())

        collected = initiator_agent.initiator_behavior.collected_proposals

        # THIS IS THE KEY ASSERTION:
        # All responders should have sent proposals, so we expect
        # exactly NUM_RESPONDERS proposals. Due to the race condition,
        # some proposals may be lost because messages arrive when no
        # ReceiverBehavior is registered.
        self.assertEqual(
            len(collected),
            NUM_RESPONDERS,
            f"Expected {NUM_RESPONDERS} proposals but only received "
            f"{len(collected)}. This indicates a race condition "
            f"where proposals were lost.",
        )

        # Verify all responder agents are represented
        proposal_authors = {p.author for p in collected}
        for resp_type in responder_types:
            self.assertIn(
                resp_type,
                proposal_authors,
                f"Missing proposal from responder {resp_type}. "
                f"Race condition caused this proposal to be lost.",
            )

        # Verify all estimates are valid random values
        for proposal in collected:
            self.assertGreaterEqual(proposal.estimate, 1.0)
            self.assertLessEqual(proposal.estimate, 100.0)


if __name__ == "__main__":
    unittest.main()
