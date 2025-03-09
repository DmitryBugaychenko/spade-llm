import unittest
from turtledemo.penrose import start

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates
from spade_llm.demo.contractnet.agents import MccExpertAgent, SpendingProfileAgent, UsersList, PeriodExpertAgent, \
    Period, TransactionsAgent
from tests.base import SpadeTestCase, ModelTestCase
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

        spendings = SpendingProfileAgent(
            data_path=DATA_PATH,
            mcc_expert=MCC_EXPERT,
            jid=SPENDINGS_AGENT,
            password="pwd"
        )
        ContractNetTestCase.spendings = spendings
        SpadeTestCase.startAgent(spendings)

        transactions = TransactionsAgent(
            data_path=DATA_PATH,
            mcc_expert=MCC_EXPERT,
            period_expert=PERIOD_AGENT,
            jid=TRANSACTIONS_AGENT,
            password="pwd"
        )
        ContractNetTestCase.transactions = transactions
        SpadeTestCase.startAgent(transactions)

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


if __name__ == '__main__':
    unittest.main()
