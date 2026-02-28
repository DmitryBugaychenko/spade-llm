import asyncio
import logging
import random
import unittest
from typing import Optional

from pydantic import BaseModel
from spade.message import Message

from spade_llm.core.api import AgentContext
from spade_llm.core.behaviors import MessageTemplate
from spade_llm.demo.platform.contractnet.contractnet import (
    ContractNetRequest,
    ContractNetProposal,
    ContractNetResponder,
    ContractNetResponderBehavior,
    ContractNetInitiatorBehavior,
)
from spade_llm.core.agent import Agent
from spade_llm.demo.platform.contractnet.discovery import AgentDescription
from tests.test_utils import TestPlatform, AgentEntry

logging.basicConfig(level=logging.DEBUG)

NUM_RESPONDERS = 5


class RandomEstimateResponderAgent(Agent, ContractNetResponder):
    """An agent that hosts a single ContractNetResponderBehavior with RandomEstimateResponder."""
    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        estimate = random.uniform(1.0, 100.0)
        return ContractNetProposal(
            author=self.agent_type,
            request=request,
            estimate=estimate,
        )

    async def execute(self, proposal: ContractNetProposal, msg: Message) -> BaseModel:
        return ContractNetRequest(task=f"Executed by {self.agent_type}")

    def setup(self) -> None:
        super().setup()
        behavior = ContractNetResponderBehavior(self)
        self.add_behaviour(behavior)


class FixedAgentsInitiatorBehavior(ContractNetInitiatorBehavior):
    """
    A subclass of ContractNetInitiatorBehavior that returns a fixed set of
    responder agents from find_agents instead of querying the DF.
    """

    def __init__(self, task: str, agents: list[AgentDescription], context: AgentContext):
        super().__init__(task, context)
        self.collected_proposals = []
        self._fixed_agents = agents

    async def find_agents(self, task: str) -> list[AgentDescription]:
        return self._fixed_agents

    async def extract_winner_and_notify_losers(self, proposals: list[ContractNetProposal]) -> ContractNetProposal:
        self.collected_proposals = proposals
        return await super().extract_winner_and_notify_losers(proposals)

    async def get_result(self, proposal: ContractNetProposal) -> Optional[MessageTemplate]:
        result = await super().get_result(proposal)
        self.agent.stop()
        return result


class InitiatorAgent(Agent):
    """An agent that hosts a CollectingInitiatorBehavior."""

    def __init__(self, agent_type: str, agents: list[AgentDescription]):
        super().__init__(agent_type=agent_type)
        self._initiator_behavior = None
        self._agents = agents

    def setup(self) -> None:
        super().setup()
        self._initiator_behavior = FixedAgentsInitiatorBehavior(
            "Handle a generic task", self._agents, self.default_context)
        self.add_behaviour(self._initiator_behavior)

    def get_proposals(self) -> list[ContractNetProposal]:
        return self._initiator_behavior.collected_proposals


class ContractNetRaceConditionTest(unittest.TestCase):
    """
    Test that reproduces a race condition in ContractNetInitiatorBehavior.get_proposals().

    When multiple responders reply nearly simultaneously, proposals can be lost
    because of timing issues in the receive loop. This test creates multiple
    responder agents that all generate random estimates, and verifies that the
    initiator collects ALL proposals from ALL responders.
    """

    def test_all_proposals_collected_from_all_responders(self):
        """
        Directly test that get_proposals returns proposals from ALL registered
        responder agents. This is the core race condition test.
        """
        agent_descriptions = []
        responder_entries = []

        for i in range(NUM_RESPONDERS):
            agent_id = f"responder_{i}"
            agent_descriptions.append(
                AgentDescription(
                    id=agent_id,
                    description=f"Responder agent {i} that handles tasks",
                )
            )
            agent = RandomEstimateResponderAgent(agent_type=agent_id)
            responder_entries.append(AgentEntry(agent=agent))


        initiator_agent = InitiatorAgent(
            agent_type="initiator",
            agents=agent_descriptions
        )
        initiator_entry = AgentEntry(agent=initiator_agent)

        all_entries = responder_entries + [initiator_entry]

        platform = TestPlatform(
            agents=all_entries,
            wait_for={"initiator"},
        )

        asyncio.run(platform.run())

        collected_proposals = initiator_agent.get_proposals()

        # THIS IS THE KEY ASSERTION:
        # All responders should have sent proposals, so we expect
        # exactly NUM_RESPONDERS proposals. Due to the race condition,
        # some proposals may be lost.
        self.assertEqual(
            len(collected_proposals),
            NUM_RESPONDERS,
            f"Expected {NUM_RESPONDERS} proposals but only received "
            f"{len(collected_proposals)}. This indicates a race condition "
            f"where proposals were lost.",
        )

        # Verify all responder agents are represented
        proposal_authors = {p.author for p in collected_proposals}
        for i in range(NUM_RESPONDERS):
            agent_id = f"responder_{i}"
            self.assertIn(
                agent_id,
                proposal_authors,
                f"Missing proposal from responder {agent_id}. "
                f"Race condition caused this proposal to be lost.",
            )

        # Verify all estimates are valid random values
        for proposal in collected_proposals:
            self.assertGreaterEqual(proposal.estimate, 1.0)
            self.assertLessEqual(proposal.estimate, 100.0)


if __name__ == "__main__":
    unittest.main()
