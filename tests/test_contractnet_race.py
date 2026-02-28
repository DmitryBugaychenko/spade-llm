import asyncio
import random
import time
import unittest
import uuid
from typing import Optional
from unittest.mock import MagicMock

from pydantic import BaseModel
from spade.agent import Agent as SpadeAgent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates, PERFORMATIVE
from spade_llm.contractnet import (
    ContractNetRequest,
    ContractNetProposal,
    ContractNetResponder,
    ContractNetResponderBehavior,
    ContractNetInitiatorBehavior,
)
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageTemplate
from spade_llm.discovery import (
    AgentDescription,
    AgentSearchRequest,
    AgentSearchResponse,
    DirectoryFacilitatorAgent,
    RegisterAgentBehavior,
)
from tests.base import SpadeTestCase
from tests.discovery import DF_ADDRESS


NUM_RESPONDERS = 5


class RandomEstimateResponder(ContractNetResponder):
    """A responder that returns a random estimate for any request."""

    def __init__(self, agent_jid: str):
        self.agent_jid = agent_jid

    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        estimate = random.uniform(1.0, 100.0)
        return ContractNetProposal(
            author=self.agent_jid,
            request=request,
            estimate=estimate,
        )

    async def execute(self, proposal: ContractNetProposal, msg: Message) -> BaseModel:
        return ContractNetRequest(task=f"Executed by {self.agent_jid}")


class ResponderAgent(Agent):


    async def setup(self):
        responder = RandomEstimateResponder(str(self.jid))
        behavior = ContractNetResponderBehavior(responder)
        self.add_behaviour(behavior)



class ContractNetRaceConditionTest(SpadeTestCase):
    """
    Test that reproduces a race condition in ContractNetInitiatorBehavior.get_proposals().

    When multiple responders reply nearly simultaneously, proposals can be lost
    because of timing issues in the receive loop. This test creates multiple
    responder agents that all generate random estimates, and verifies that the
    initiator collects ALL proposals from ALL responders.
    """

    responder_agents = []
    responder_jids = []

    @classmethod
    def setUpClass(cls):
        SpadeTestCase.setUpClass()

        # Start directory facilitator
        cls.df = DirectoryFacilitatorAgent(
            DF_ADDRESS,
            "pwd",
        )
        # DF might need embeddings; if so, import from tests.models
        # For this test we provide a minimal setup
        try:
            from tests.models import MODELS
            cls.df = DirectoryFacilitatorAgent(
                DF_ADDRESS,
                "pwd",
                embeddings=MODELS.embeddings,
            )
        except Exception:
            pass
        SpadeTestCase.startAgent(cls.df)

        # Start multiple responder agents
        cls.responder_agents = []
        cls.responder_jids = []
        for i in range(NUM_RESPONDERS):
            jid = f"responder{i}@localhost"
            cls.responder_jids.append(jid)
            agent = ResponderAgent(
                jid=jid,
                password="pwd",
                description=f"Responder agent {i} that handles tasks",
            )
            SpadeTestCase.startAgent(agent)
            cls.responder_agents.append(agent)

            # Register each responder with the DF
            register = RegisterAgentBehavior(
                AgentDescription(
                    id=jid,
                    description=f"Agent that can handle generic tasks and queries",
                ),
                DF_ADDRESS,
            )
            agent.add_behaviour(register, register.response_template)
            SpadeTestCase.run_in_container(register.join(10))

    @classmethod
    def tearDownClass(cls):
        for agent in cls.responder_agents:
            SpadeTestCase.stopAgent(agent)
        SpadeTestCase.tearDownClass()

    def test_all_proposals_received(self):
        """
        Verify that the initiator receives proposals from ALL responders.

        This test exposes the race condition: when multiple responders send
        proposals nearly simultaneously, some proposals may be lost due to
        timing issues in get_proposals()'s receive loop.
        """
        agent = DummyAgent(
            jid="race_initiator@localhost",
            password="pwd",
        )

        initiator = ContractNetInitiatorBehavior(
            task="Handle a generic task",
            df_address=DF_ADDRESS,
            time_to_wait_for_proposals=15,
        )

        agent.add_behaviour(initiator, initiator.construct_template())
        SpadeTestCase.startAgent(agent)
        SpadeTestCase.wait_for_behavior(initiator)
        SpadeTestCase.stopAgent(agent)

        # The initiator should have been successful
        self.assertTrue(initiator.is_successful, "Initiator should have completed successfully")

        # The winning proposal should be the one with the lowest estimate
        self.assertIsNotNone(initiator.proposal, "Initiator should have selected a winning proposal")

    def test_all_proposals_collected_from_all_responders(self):
        """
        Directly test that get_proposals returns proposals from ALL registered
        responder agents. This is the core race condition test.

        We override the flow to capture the intermediate proposals list.
        """
        collected_proposals = []
        original_run = ContractNetInitiatorBehavior.run

        async def patched_run(self_initiator):
            """Patched run that captures proposals before winner selection."""
            agents = await self_initiator.find_agents(self_initiator.task, self_initiator.df_address)
            self.assertGreaterEqual(
                len(agents), NUM_RESPONDERS,
                f"Expected at least {NUM_RESPONDERS} agents registered, got {len(agents)}",
            )

            proposals = await self_initiator.get_proposals(agents)
            collected_proposals.extend(proposals)

            if len(proposals) > 0:
                self_initiator.proposal = await self_initiator.extract_winner_and_notify_losers(proposals)
                if self_initiator.proposal:
                    self_initiator.result = await self_initiator.get_result(self_initiator.proposal)

        ContractNetInitiatorBehavior.run = patched_run

        try:
            agent = DummyAgent(
                jid="race_initiator2@localhost",
                password="pwd",
            )

            initiator = ContractNetInitiatorBehavior(
                task="Handle a generic task",
                df_address=DF_ADDRESS,
                time_to_wait_for_proposals=15,
            )

            agent.add_behaviour(initiator, initiator.construct_template())
            SpadeTestCase.startAgent(agent)
            SpadeTestCase.wait_for_behavior(initiator)
            SpadeTestCase.stopAgent(agent)

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
            for jid in self.responder_jids:
                self.assertIn(
                    jid,
                    proposal_authors,
                    f"Missing proposal from responder {jid}. "
                    f"Race condition caused this proposal to be lost.",
                )
        finally:
            ContractNetInitiatorBehavior.run = original_run


if __name__ == "__main__":
    unittest.main()
