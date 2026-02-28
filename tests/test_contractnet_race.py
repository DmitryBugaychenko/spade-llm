import asyncio
import random
import unittest
from typing import Optional

from pydantic import BaseModel
from spade.message import Message

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


NUM_RESPONDERS = 5


class RandomEstimateResponder(ContractNetResponder):
    """A responder that returns a random estimate for any request."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        estimate = random.uniform(1.0, 100.0)
        return ContractNetProposal(
            author=self.agent_id,
            request=request,
            estimate=estimate,
        )

    async def execute(self, proposal: ContractNetProposal, msg: Message) -> BaseModel:
        return ContractNetRequest(task=f"Executed by {self.agent_id}")


class RandomEstimateResponderAgent(Agent):
    """An agent that hosts a single ContractNetResponderBehavior with RandomEstimateResponder."""

    async def setup(self) -> None:
        await super().setup()
        responder = RandomEstimateResponder(self.agent_type)
        behavior = ContractNetResponderBehavior(responder)
        self.add_behaviour(behavior)


class FixedAgentsInitiatorBehavior(ContractNetInitiatorBehavior):
    """
    A subclass of ContractNetInitiatorBehavior that returns a fixed set of
    responder agents from find_agents instead of querying the DF.
    """

    def __init__(self, task: str, agents: list[AgentDescription], **kwargs):
        super().__init__(task, **kwargs)
        self._fixed_agents = agents

    async def find_agents(self, task: str) -> list[AgentDescription]:
        return self._fixed_agents


class CollectingInitiatorBehavior(FixedAgentsInitiatorBehavior):
    """
    An initiator behavior that captures collected proposals for test assertions.
    """

    def __init__(self, task: str, agents: list[AgentDescription], **kwargs):
        super().__init__(task, agents, **kwargs)
        self.collected_proposals: list[ContractNetProposal] = []

    async def run(self):
        agents = await self.find_agents(self.task)
        proposals = await self.get_proposals(agents)
        self.collected_proposals.extend(proposals)

        if len(proposals) > 0:
            self.proposal = await self.extract_winner_and_notify_losers(proposals)
            if self.proposal:
                self.result = await self.get_result(self.proposal)

        self.kill()


class InitiatorAgent(Agent):
    """An agent that hosts a CollectingInitiatorBehavior."""

    def __init__(self, agent_type: str, initiator_behavior: CollectingInitiatorBehavior):
        super().__init__(agent_type=agent_type)
        self._initiator_behavior = initiator_behavior

    async def setup(self) -> None:
        await super().setup()
        self.add_behaviour(self._initiator_behavior)


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

        initiator_behavior = CollectingInitiatorBehavior(
            task="Handle a generic task",
            agents=agent_descriptions,
        )
        initiator_agent = InitiatorAgent(
            agent_type="initiator",
            initiator_behavior=initiator_behavior,
        )
        initiator_entry = AgentEntry(agent=initiator_agent)

        all_entries = responder_entries + [initiator_entry]

        platform = TestPlatform(
            agents=all_entries,
            wait_for={"initiator"},
        )

        asyncio.run(platform.run())

        collected_proposals = initiator_behavior.collected_proposals

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
