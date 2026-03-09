import logging
import unittest
from typing import Optional, Set

from pydantic import BaseModel

from spade_llm.agents.dummy import DummyAgent, ExecuteInContext
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageTemplate, ContextBehaviour
from spade_llm.core.testing import TestPlatform, AgentEntry
from spade_llm.patterns.contractnet import (
    ContractNetRequest,
    ContractNetProposal,
    ContractNetResponder,
    ContractNetResponderBehavior,
    ContractNetInitiatorBehavior,
)
from spade_llm.patterns.discovery import (
    AgentDescription,
    AgentTask,
    DirectoryFacilitatorAgent,
    DirectoryFacilitatorAgentConf,
)
from tests.base import ModelTestCase

logging.basicConfig(level=logging.INFO)
DF_ADDRESS = "df"


# Mock implementations for testing
class MockTaskResult(BaseModel):
    result: str


class MockResponderAgent(Agent, ContractNetResponder):
    """A mock responder agent for testing contract net functionality."""

    def __init__(self, agent_type: str, estimate_value: float, supported_tasks: Set[str]):
        super().__init__(agent_type=agent_type)
        self.estimate_value = estimate_value
        self.supported_tasks = supported_tasks

    async def estimate(self, request: ContractNetRequest, msg: MessageTemplate) -> Optional[ContractNetProposal]:
        # Return proposal only if task is supported
        if request.task in self.supported_tasks:
            return ContractNetProposal(
                author=self.agent_type,
                request=request,
                estimate=self.estimate_value,
            )
        return None

    async def execute(self, proposal: ContractNetProposal, msg: MessageTemplate) -> BaseModel:
        return MockTaskResult(result=f"Executed by {self.agent_type}")

    def setup(self) -> None:
        super().setup()
        behavior = ContractNetResponderBehavior(self)
        self.add_behaviour(behavior)


# Agent descriptions for testing
spendings_agent_desc = AgentDescription(
    description="Agent that analyzes spending patterns",
    id="spendings",
    domain="finance",
    tier=0,
    tasks=[
        AgentTask(
            description="Spending analysis",
            examples=[
                "Analyze monthly spending patterns",
                "Identify categories with highest expenditure",
            ]
        )
    ]
)

transactions_agent_desc = AgentDescription(
    description="Agent that processes transaction data",
    id="transactions",
    domain="finance",
    tier=0,
    tasks=[
        AgentTask(
            description="Transaction processing",
            examples=[
                "Process transaction history",
                "Filter transactions by date range",
            ]
        )
    ]
)


class ContractNetTest(ModelTestCase):

    def test_contract_net_protocol(self):
        # Create dummy agent for testing
        dummy_agent = DummyAgent()

        # Create directory facilitator agent
        df_agent = DirectoryFacilitatorAgent(agent_type=DF_ADDRESS)

        # Create responder agents
        # Spendings agent supports only spending analysis tasks
        spendings_agent = MockResponderAgent(
            "spendings", 
            1.0, 
            {"Analyze monthly spending patterns", "Identify categories with highest expenditure"}
        )
        # Transactions agent supports both tasks
        transactions_agent = MockResponderAgent(
            "transactions", 
            2.0, 
            {
                "Analyze monthly spending patterns", 
                "Identify categories with highest expenditure",
                "Process transaction history",
                "Filter transactions by date range"
            }
        )

        test = self

        class RegisterAgent(ExecuteInContext):
            def __init__(self, description: AgentDescription):
                super().__init__()
                self.description = description

            async def execute(self, beh: ContextBehaviour):
                with test.subTest("Test registration for " + self.description.id):
                    await beh.context.inform(DF_ADDRESS).with_content(self.description)
                    ack = await beh.receive(MessageTemplate.acknowledge(), 10)
                    test.assertIsNotNone(ack)

        class RequestProposal(ExecuteInContext):
            def __init__(self, task: str, expected_winner: str):
                self.task = task
                self.expected_winner = expected_winner

            async def execute(self, beh: ContextBehaviour):
                # Create initiator behavior
                initiator = ContractNetInitiatorBehavior(
                    task=self.task,
                    context=beh.context,
                )
                beh.agent.add_behaviour(initiator)
                await initiator.join()

                with test.subTest("Test contract net for " + self.task):
                    test.assertTrue(initiator.is_successful)
                    test.assertIsNotNone(initiator.result)
                    if self.expected_winner:
                        parsed = MockTaskResult.model_validate_json(initiator.result.content)
                        test.assertIn(self.expected_winner, parsed.result)

        # Register agents
        dummy_agent.as_agent(RegisterAgent(spendings_agent_desc))
        dummy_agent.as_agent(RegisterAgent(transactions_agent_desc))

        # Test contract net protocol
        dummy_agent.as_agent(RequestProposal("Analyze monthly spending patterns", "spendings"))
        dummy_agent.as_agent(RequestProposal("Process transaction history", "transactions"))

        # Run test platform
        TestPlatform.run_test(
            agents=[
                AgentEntry(
                    agent=df_agent,
                    configuration=DirectoryFacilitatorAgentConf(model="test")
                ),
                AgentEntry(agent=spendings_agent),
                AgentEntry(agent=transactions_agent),
                AgentEntry(agent=dummy_agent),
            ],
            wait_for={dummy_agent.agent_type},
            embedding_models={"test": self.embeddings}
        )

    def test_contract_net_responder_behavior(self):
        """Test individual contract net responder behavior."""
        test = self

        class TestResponder(Agent, ContractNetResponder):
            async def estimate(self, request: ContractNetRequest, msg: MessageTemplate) -> Optional[ContractNetProposal]:
                if "spending" in request.task:
                    return ContractNetProposal(
                        author=self.agent_type,
                        request=request,
                        estimate=1.0,
                    )
                return None

            async def execute(self, proposal: ContractNetProposal, msg: MessageTemplate) -> BaseModel:
                return MockTaskResult(result="Task executed successfully")

            def setup(self) -> None:
                super().setup()
                behavior = ContractNetResponderBehavior(self)
                self.add_behaviour(behavior)

        responder = TestResponder("test_responder")
        dummy_agent = DummyAgent()

        class RequestProposal(ExecuteInContext):
            async def execute(self, beh: ContextBehaviour):
                # Send proposal request
                request = ContractNetRequest(task="Analyze spending patterns")
                await beh.context.request_proposal("test_responder").with_content(request)

                # Expect proposal in response
                proposal_msg = await beh.receive(MessageTemplate.propose(), 10)
                test.assertIsNotNone(proposal_msg)

                proposal = ContractNetProposal.model_validate_json(proposal_msg.content)
                test.assertEqual(proposal.author, "test_responder")
                test.assertEqual(proposal.estimate, 1.0)

                # Accept the proposal
                await beh.context.accept("test_responder").with_content(proposal)

                # Expect result
                result_msg = await beh.receive(MessageTemplate.inform(), 10)
                test.assertIsNotNone(result_msg)

                result = MockTaskResult.model_validate_json(result_msg.content)
                test.assertEqual(result.result, "Task executed successfully")

        dummy_agent.as_agent(RequestProposal())

        TestPlatform.run_test(
            agents=[
                AgentEntry(agent=responder),
                AgentEntry(agent=dummy_agent),
            ],
            wait_for={dummy_agent.agent_type}
        )


if __name__ == '__main__':
    unittest.main()
