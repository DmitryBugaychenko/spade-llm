import time
import unittest
import uuid

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates, PERFORMATIVE, REFUSE
from spade_llm.contractnet import ContractNetRequest, ContractNetProposal, ContractNetInitiatorBehavior
from spade_llm.demo.contractnet.agents import MccExpertAgent, SpendingProfileAgent, UsersList, PeriodExpertAgent, \
    Period, TransactionsAgent
from spade_llm.discovery import DirectoryFacilitatorAgent, AgentSearchRequest, AgentSearchResponse, \
    RegisterAgentBehavior
from tests.base import SpadeTestCase, ModelTestCase, DummyAgent
from tests.discovery import DF_ADDRESS
from tests.models import MODELS

PERIOD_AGENT = "period@localhost"
SPENDINGS_AGENT = "spendings@localhost"
TRANSACTIONS_AGENT = "transactions@localhost"
MCC_EXPERT = "mccexpert@localhost"

DATA_PATH = "../data"



class ContractNetTestCase(SpadeTestCase, ModelTestCase):

    @classmethod
    def setUpClass(cls):
        SpadeTestCase.setUpClass()
        expert = MccExpertAgent(
            embeddings=MODELS.embeddings,
            model=MODELS.max,
            data_path=DATA_PATH,
            jid=MCC_EXPERT,
            password="pwd"
        )
        SpadeTestCase.startAgent(expert)

        period = PeriodExpertAgent(
            model=MODELS.max,
            date="2012-04-01",
            jid=PERIOD_AGENT,
            password="pwd"
        )
        SpadeTestCase.startAgent(period)

        df = DirectoryFacilitatorAgent(
            DF_ADDRESS,
            "pwd",
            embeddings=MODELS.embeddings)
        SpadeTestCase.startAgent(df)

        spendings = SpendingProfileAgent(
            data_path=DATA_PATH,
            mcc_expert=MCC_EXPERT,
            df_address=DF_ADDRESS,
            model=MODELS.max,
            jid=SPENDINGS_AGENT,
            password="pwd"
        )
        ContractNetTestCase.spendings = spendings
        SpadeTestCase.startAgent(spendings)
        for b in spendings.behaviours:
            if isinstance(b, RegisterAgentBehavior):
                SpadeTestCase.run_in_container(b.join(10))

        transactions = TransactionsAgent(
            data_path=DATA_PATH,
            mcc_expert=MCC_EXPERT,
            period_expert=PERIOD_AGENT,
            df_address=DF_ADDRESS,
            model=MODELS.max,
            jid=TRANSACTIONS_AGENT,
            password="pwd"
        )
        ContractNetTestCase.transactions = transactions
        SpadeTestCase.startAgent(transactions)
        for b in transactions.behaviours:
            if isinstance(b, RegisterAgentBehavior):
                SpadeTestCase.run_in_container(b.join(10))

    @classmethod
    def tearDownClass(cls):
        SpadeTestCase.stopAgent(ContractNetTestCase.spendings)
        SpadeTestCase.stopAgent(ContractNetTestCase.transactions)
        SpadeTestCase.tearDownClass()

    def test_find_mcc(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(MCC_EXPERT)
                            .with_content("Люди, которые ходят в бары"),
                            Templates.INFORM())
        self.assertEqual("5813", msg.body)

    def test_find_period(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(PERIOD_AGENT)
                                  .with_content("Люди, которые ходили в бары в прошлом месяце"),
                                  Templates.INFORM())
        result = Period.model_validate_json(msg.body)

        self.assertEqual(Period(start = "2012-03-01", end = "2012-03-31"), result)

    def test_find_mcc_in_message_with_period(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(MCC_EXPERT)
                                  .with_content("Люди, которые ходили в бары в прошлом месяце"),
                                  Templates.INFORM())
        self.assertEqual("5813", msg.body)

    def test_from_spendings(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(SPENDINGS_AGENT)
                                  .with_content("Люди, которые ходят в бары"),
                                  Templates.INFORM())
        result = UsersList.model_validate_json(msg.body)

        self.assertGreater(len(result.ids),0)

    def test_from_transactions(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(TRANSACTIONS_AGENT)
                                  .with_content("Люди, которые ходили в бары в прошлом месяце"),
                                  Templates.INFORM())
        result = UsersList.model_validate_json(msg.body)

        self.assertGreater(len(result.ids),0)

    def test_from_transactions_returns_less_users(self):
        msg = self.sendAndReceive(MessageBuilder.request().to_agent(TRANSACTIONS_AGENT)
                                  .with_content("Люди, которые ходили в бары в прошлом месяце"),
                                  Templates.INFORM())
        from_transactions = UsersList.model_validate_json(msg.body)

        msg = self.sendAndReceive(MessageBuilder.request().to_agent(SPENDINGS_AGENT)
                                  .with_content("Люди, которые ходят в бары"),
                                  Templates.INFORM())
        from_spendings = UsersList.model_validate_json(msg.body)

        self.assertGreater(len(from_spendings.ids),len(from_transactions.ids))

    def test_registration(self):
        msg = self.sendAndReceive(
            MessageBuilder.request().to_agent(DF_ADDRESS)
                    .with_content(AgentSearchRequest(
                        task =  "Люди, которые ходят в бары",
                        top_k = 4)),
            Templates.INFORM())

        result = AgentSearchResponse.model_validate_json(msg.body)

        ids = [agent.id for agent in result.agents]
        ids.sort()

        self.assertListEqual([SPENDINGS_AGENT, TRANSACTIONS_AGENT], ids)

    def test_registration_with_period(self):
        msg = self.sendAndReceive(
            MessageBuilder.request().to_agent(DF_ADDRESS)
            .with_content(AgentSearchRequest(
                task =  "Люди, которые ходили в бары в прошлом месяце",
                top_k = 4)),
            Templates.INFORM())

        result = AgentSearchResponse.model_validate_json(msg.body)

        ids = [agent.id for agent in result.agents]
        ids.sort()

        self.assertListEqual([SPENDINGS_AGENT, TRANSACTIONS_AGENT], ids)

    def test_proposal_from_spendings(self):
        request = ContractNetRequest(task = "Люди, которые ходят в бары")
        msg = self.sendAndReceive(MessageBuilder.request_porposal()
                                  .to_agent(SPENDINGS_AGENT)
                                  .with_content(request),
                                  Templates.PROPOSE())
        result = ContractNetProposal.model_validate_json(msg.body)

        self.assertEqual(result.estimate, 1.0)

    def test_proposal_from_spendings_with_period(self):
        request = ContractNetRequest(task = "Люди, которые ходили в бар в прошлом месяце")
        msg = self.sendAndReceive(MessageBuilder.request_porposal()
                                  .to_agent(SPENDINGS_AGENT)
                                  .with_content(request),
                                  Templates.REFUSE())


        self.assertEqual(msg.metadata[PERFORMATIVE], REFUSE)

    def test_proposal_from_transactions(self):
        request = ContractNetRequest(task = "Люди, которые ходят в бары")
        msg = self.sendAndReceive(MessageBuilder.request_porposal()
                                  .to_agent(TRANSACTIONS_AGENT)
                                  .with_content(request),
                                  Templates.PROPOSE())
        result = ContractNetProposal.model_validate_json(msg.body)

        self.assertEqual(result.estimate, 10.0)

    def test_proposal_from_transactions_with_period(self):
        request = ContractNetRequest(task = "Люди, которые ходили в бар в прошлом месяце")
        msg = self.sendAndReceive(MessageBuilder.request_porposal()
                                  .to_agent(TRANSACTIONS_AGENT)
                                  .with_content(request),
                                  Templates.PROPOSE())

        result = ContractNetProposal.model_validate_json(msg.body)

        self.assertEqual(result.estimate, 10.0)

    def test_proposal_from_spendings_with_execution(self):
        request = ContractNetRequest(task = "Люди, которые ходят в бары")
        thread = str(uuid.uuid4())
        proposal = self.sendAndReceive(MessageBuilder.request_porposal()
                                  .to_agent(SPENDINGS_AGENT)
                                  .in_thread(thread)
                                  .with_content(request),
                                  Templates.PROPOSE())
        result = ContractNetProposal.model_validate_json(proposal.body)


        list = self.sendAndReceive(
            MessageBuilder.accept()
            .to_agent(SPENDINGS_AGENT)
            .in_thread(thread)
            .with_content(result),
            Templates.INFORM())

        result = UsersList.model_validate_json(list.body)

        self.assertGreater(len(result.ids),0)

    def test_proposal_from_transactions_with_execution(self):
        request = ContractNetRequest(task = "Люди, которые ходили в бар в прошлом месяце")
        thread = str(uuid.uuid4())
        proposal = self.sendAndReceive(MessageBuilder.request_porposal()
                                       .to_agent(TRANSACTIONS_AGENT)
                                       .in_thread(thread)
                                       .with_content(request),
                                       Templates.PROPOSE())
        result = ContractNetProposal.model_validate_json(proposal.body)


        list = self.sendAndReceive(
            MessageBuilder.accept()
            .to_agent(TRANSACTIONS_AGENT)
            .in_thread(thread)
            .with_content(result),
            Templates.INFORM())

        result = UsersList.model_validate_json(list.body)

        self.assertGreater(len(result.ids),0)

    def test_contractnet_initiator(self):
        agent = DummyAgent(
            jid="initiator@localhost",
            password="pwd")

        initiator = ContractNetInitiatorBehavior(
            task="Люди, которые ходят в бары",
            df_address=DF_ADDRESS)

        agent.add_behaviour(initiator, initiator.construct_template())
        SpadeTestCase.startAgent(agent)
        SpadeTestCase.wait_for_behavior(initiator)
        SpadeTestCase.stopAgent(agent)

        self.assertTrue(initiator.is_successful)
        self.assertEqual(SPENDINGS_AGENT, str(initiator.result.sender))

        result = UsersList.model_validate_json(initiator.result.body)
        self.assertGreater(len(result.ids),0)

    def test_contractnet_initiator_with_period(self):
        agent = DummyAgent(
                jid="initiator@localhost",
                password="pwd")

        initiator = ContractNetInitiatorBehavior(
            task="Люди, которые ходили в бары в прошлом месяце",
            df_address=DF_ADDRESS)

        agent.add_behaviour(initiator, initiator.construct_template())
        SpadeTestCase.startAgent(agent)
        SpadeTestCase.wait_for_behavior(initiator)
        SpadeTestCase.stopAgent(agent)

        self.assertTrue(initiator.is_successful)
        self.assertEqual(TRANSACTIONS_AGENT, str(initiator.result.sender))

        result = UsersList.model_validate_json(initiator.result.body)
        self.assertGreater(len(result.ids),0)


if __name__ == '__main__':
    unittest.main()