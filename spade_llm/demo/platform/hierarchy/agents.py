from pydantic import BaseModel, Field

from spade_llm import consts
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import configuration, Configurable


class FinancialAgentConf(BaseModel):
    balance: int = Field(description="Initial balance of the agent.")

class FinancialMessage(BaseModel):
    amount: int = Field(description="Amount of the payment.")

@configuration(FinancialAgentConf)
class PaymentsAgent(Agent, Configurable[FinancialAgentConf]):

    class PaymentBehaviour(MessageHandlingBehavior):

        def __init__(self, config: FinancialAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self):
            balance = await self.context.get_item("balance")
            if not balance or balance == "":
                self.logger.info("No balance set, using default: %s", self.config.balance)
                balance = self.config.balance
            else:
                self.logger.debug("Received balance: %s", balance)
                balance = int(balance)

            if self.message:
                payment_request = FinancialMessage.model_validate_json(self.message.content)
                self.logger.debug("Received payment request: %s", payment_request)

                if payment_request.amount > balance:
                    self.logger.warning("Not enough balance to pay: %s", payment_request.amount)
                    deficiency = payment_request.amount - balance
                    await (self.context
                           .reply_with_failure(self.message)
                           .with_content(f"Balance is insufficient for payment, replenish payment account for {deficiency} rubles."))
                else:
                    new_balance = balance - payment_request.amount
                    await self.context.put_item("balance", str(new_balance))
                    self.logger.info("Payment done, new balance: %s", new_balance)
                    await (self.context
                           .reply_with_inform(self.message)
                           .with_content(f"Payment for {payment_request.amount} successfully completed, new balance is {new_balance} rubles."))

    class ReplenishBehaviour(MessageHandlingBehavior):
        def __init__(self, config: FinancialAgentConf):
            super().__init__(MessageTemplate(
                performative=consts.INFORM,
                validator=MessageTemplate.from_agent("savings")))
            self.config = config

        async def step(self):
            balance = await self.context.get_item("balance")
            if not balance or balance == "":
                self.logger.info("No balance set, using default: %s", self.config.balance)
                balance = self.config.balance
            else:
                self.logger.debug("Received balance: %s", balance)
                balance = int(balance)

            if self.message:
                replenish_request = FinancialMessage.model_validate_json(self.message.content)
                self.logger.debug("Received replenish request: %s", replenish_request)
                new_balance = balance + replenish_request.amount
                await self.context.put_item("balance", str(new_balance))
                self.logger.info("Replenish done, new balance: %s", new_balance)

    def setup(self):
        self.add_behaviour(self.PaymentBehaviour(self.config))
        self.add_behaviour(self.ReplenishBehaviour(self.config))

@configuration(FinancialAgentConf)
class SavingsAgent(Agent, Configurable[FinancialAgentConf]):
    class ReplenishBehaviour(MessageHandlingBehavior):

        def __init__(self, config: FinancialAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self):
            balance = await self.context.get_item("balance")
            if not balance or balance == "":
                self.logger.info("No balance set, using default: %s", self.config.balance)
                balance = self.config.balance
            else:
                self.logger.debug("Received balance: %s", balance)
                balance = int(balance)

            if self.message:
                replenish_request = FinancialMessage.model_validate_json(self.message.content)
                self.logger.debug("Received replenish request: %s", replenish_request)

                if replenish_request.amount > balance:
                    self.logger.warning("Not enough balance to replenish: %s", replenish_request.amount)
                    deficiency = replenish_request.amount - balance
                    await (self.context
                           .reply_with_failure(self.message)
                           .with_content(f"Balance is insufficient at savings account, deficiency is {deficiency} rubles."))
                else:
                    await (self.context
                           .request_approval("user")
                           .with_content(f"Replenish current account from savings for {replenish_request.amount}"))

                    self.logger.info("Waiting for user approval...")
                    msg = await self.receive(
                        template = MessageTemplate(
                            thread_id=self.context.thread_id, validator=MessageTemplate.from_agent("user")),
                        timeout=60)
                    self.logger.info("Received message: %s", msg)

                    if not msg or msg.performative != consts.ACKNOWLEDGE:
                        self.logger.warning("User did not approve replenishment.")
                        await (self.context
                               .reply_with_failure(self.message)
                               .with_content(f"User did not approve replenishment."))
                    else:
                        new_balance = balance - replenish_request.amount
                        await self.context.put_item("balance", str(new_balance))
                        await (self.context
                               .inform("payments")
                               .with_content(FinancialMessage(amount=replenish_request.amount)))

                        self.logger.info("Replenish done, new balance at savings account: %s", new_balance)
                        await (self.context
                               .reply_with_inform(self.message)
                               .with_content(f"Replenish of current account for {replenish_request.amount} successfully completed."))

    def setup(self):
        self.add_behaviour(self.ReplenishBehaviour(self.config))