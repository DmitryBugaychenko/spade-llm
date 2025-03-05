import os
import sys
import time
import uuid
from collections.abc import Awaitable
from multiprocessing import Process
from unittest import TestCase

from langchain_gigachat import GigaChat
from spade.agent import Agent
from spade.cli import create_cli
from spade.container import Container
from spade.message import Message
from spade.template import Template

from spade_llm.behaviours import SendAndReceiveBehaviour


def start_xmmp():
    sys.argv += ["run", "--memory"]
    cli = create_cli()
    sys.exit(cli())

class DummyAgent(Agent):
    async def setup(self) -> None:
        print("Setting up the agent.")
        await super().setup()

class SpadeTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        print("Starting XMMP Server")
        SpadeTestCase.xmmp = Process(target=start_xmmp, daemon=True)
        SpadeTestCase.xmmp.start()
        SpadeTestCase.container = Container()
        dummy = Agent("{0}@localhost".format(uuid.uuid4()), "qwerty")
        SpadeTestCase.startAgent(dummy)
        SpadeTestCase.dummy = dummy

    @classmethod
    def tearDownClass(cls):
        print("Stopping XMMP Server")
        SpadeTestCase.run_in_container(SpadeTestCase.dummy.stop())
        SpadeTestCase.xmmp.terminate()
        SpadeTestCase.xmmp.join(10)

    @staticmethod
    async def asend(msg: Message):
        await SpadeTestCase.container.send(msg, None)

    @staticmethod
    def send(msg: Message):
        SpadeTestCase.run_in_container(SpadeTestCase.asend(msg))

    @staticmethod
    async def asendAndReceive(msg: Message, template: Template) -> Message:
        behaviour = SendAndReceiveBehaviour(msg,template)
        await SpadeTestCase._asendAndReceive(behaviour, msg, template)
        return behaviour.response

    @staticmethod
    async def _asendAndReceive(behaviour, msg, template):
        dummy: Agent = SpadeTestCase.dummy
        msg.sender = str(dummy.jid)
        dummy.add_behaviour(behaviour, template)
        await behaviour.join(60)

    @staticmethod
    def sendAndReceive(msg: Message, template: Template) -> Message:
        behaviour = SendAndReceiveBehaviour(msg,template)
        SpadeTestCase.run_in_container(SpadeTestCase._asendAndReceive(behaviour, msg, template))
        return behaviour.response

    @staticmethod
    def startAgent(agent: Agent):
        SpadeTestCase.run_in_container(agent.start())
        while not agent.is_alive():
            time.sleep(0.1)


    @staticmethod
    def run_in_container(coro: Awaitable) -> None:
        SpadeTestCase.container.run(coro)

    def test_xmmp_runnig(self):
        self.assertTrue(SpadeTestCase.xmmp.is_alive())

    def test_run_agent(self):
        agent: Agent = SpadeTestCase.dummy
        self.assertTrue(agent.is_alive())


class ModelTestCase(TestCase):
    pro = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        base_url="https://gigachat.devices.sberbank.ru/api/v1",
        model="GigaChat-Pro",
        verify_ssl_certs=False,
    )

    pro_preview = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        base_url="https://gigachat-preview.devices.sberbank.ru/api/v1",
        model="GigaChat-Pro-preview",
        verify_ssl_certs=False,
    )

