import logging
import asyncio
import unittest
from asyncio import sleep as asleep
from spade_llm.core.api import AgentContext
from pydantic import BaseModel
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.core.conf import configuration, Configurable
from spade_llm import consts

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field


class EmptyConfig(BaseModel):
    timeout: int = Field(default=1)
    agent_number: int = Field(default=0)


@configuration(EmptyConfig)
class SenderAgent(Agent, Configurable[EmptyConfig]):
    class ReceiveMessagesBehaviour(MessageHandlingBehavior):
        def __init__(self,config: EmptyConfig):
            super().__init__(MessageTemplate.request())
            self.config = config
        async def step(self):
            msg = self.message
            print(f"Я АГЕНТ НОМЕР {self.config.agent_number}, ОТПРАВЛЯЮ СООБЩЕНИЕ")
            await self.context.reply_with_inform(msg).with_content(str(self.config.agent_number))

    def setup(self):
        self.add_behaviour(self.ReceiveMessagesBehaviour(self.config))


import time


@configuration(EmptyConfig)
class ReceiverAgent(Agent, Configurable[EmptyConfig]):
    class SendAndReceiveReplyBehaviour(ContextBehaviour):
        def __init__(self, context: AgentContext, config):
            super().__init__(context)
            self.config = config

        async def step(self):
            result = await self.send_and_agregate()

            print("ОБЩАЯ СУММА ПОЛУЧЕННЫХ ЧИСЕЛ = ", result ,"\n Ожидаемое значение = ", 15)
            # assert result == 15
            self.set_is_done()

        async def send_and_agregate(self, time_to_wait_for_responses=10):
            sent: set[str] = set()
            received: set[str] = set()
            received_sum: int = 0
            for i in range(1, 6):
                await self.context.request(f"sender{i}_agent").with_content('Please send me a message')
                sent.add(f"sender{i}_agent")
            started = time.time()
            print(" ОЖИДАЕМ ОТВЕТЫ")
            deadline = started + time_to_wait_for_responses
            while len(received) < len(sent) and time.time() < deadline:
                response = await self.receive(
                    MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
                    max(0.1, deadline - time.time()))
                if response:
                    received.add(str(response.sender))
                    received_sum += int(response.content)
            print("ПОЛУЧИЛИ ОТВЕТ ОТ: ",received)

            return received_sum
    def setup(self):
        self.add_behaviour(self.SendAndReceiveReplyBehaviour(self.default_context, self.config))