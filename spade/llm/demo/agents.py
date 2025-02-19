import logging
import uuid

from aioconsole import ainput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from spade.llm.behaviours import SendAndReceiveBehaviour, RequestHandlingBehaviour


logger = logging.getLogger(__name__)

class ChatAgent(Agent):
    """
    This agent acts as a general assistant. Gets user prompts from stdin
    and uses standard cycle for handling.

    Also provides ability to confirm certain operations with the user.
    """
    model: BaseChatModel = None
    tools: dict[str,BaseTool] = None

    def __init__(self, model: BaseChatModel, tools: dict[str,BaseTool], jid: str, password: str):
        super().__init__(jid, password)
        self.model = model
        self.tools = tools

    class RequestCycle(RequestHandlingBehaviour):
        """
        Cycle for handling user prompts from stdin.
        """
        async def handle_answer(self, answer: BaseMessage):
            print("GigaChat: ", answer.content)

        async def get_prompt(self) -> BaseMessage:
            user_input = await ainput("Пользователь: ")
            if user_input == "пока":
                await self.agent.stop()
                return None
            else:
                return HumanMessage(user_input)

    class AcknowledgmentsCycle(CyclicBehaviour):
        """
        Cycle for getting user acknowledgements.
        """
        async def run(self):
            msg = await self.receive(60)
            if msg:
                user_input = await ainput('Подтвердите операцию "{0}" [y/n]: '.format(msg.body))
                await self.send(Message(
                    to = str(msg.sender),
                    thread = msg.thread,
                    metadata = {"performative" : ("acknowledge" if user_input == "y" else "refuse")}
                ))

    async def setup(self):
        print("Agent starting . . .")
        self.model = self.model.bind_tools(self.tools.values())
        self.add_behaviour(self.RequestCycle(
            model = self.model,
            context= [SystemMessage(
                content="Ты персональный ассистент человека. Используй инструменты для решения его вопросов."
            )],
            tools=self.tools
        ))

        self.add_behaviour(
            self.AcknowledgmentsCycle(),
            Template(metadata={"performative": "request_with_acknowledge"}))

class FinancialAgent(Agent):
    """
    Agent acting like financial assistant. Does not interact with user directly,
    receives messages from chat agent.
    """
    model: BaseChatModel = None
    tools: dict[str,BaseTool] = None

    def __init__(self, model: BaseChatModel, tools: dict[str,BaseTool], jid: str, password: str):
        super().__init__(jid, password)
        self.model = model
        self.tools = tools


    class RequestCycle(RequestHandlingBehaviour):
        msg: Message = None

        async def handle_answer(self, answer: BaseMessage):
            logger.info("Provided financial solution %s", answer)
            await self.send(Message(
                to=str(self.msg.sender),
                thread=self.msg.thread,
                body=answer.content,
                metadata={"performative": "inform"}
            ))

        async def get_prompt(self) -> BaseMessage:
            self.msg = await self.receive(600)
            if self.msg:
                logger.info("Got financial request %s", self.msg)
                return HumanMessage(self.msg.body)
            else:
                return None

    async def setup(self):
        self.model = self.model.bind_tools(self.tools.values())
        self.add_behaviour(self.RequestCycle(
            model = self.model,
            context= [SystemMessage(
                content="Ты финансовый ассистент человека. Используй инструменты для проведения платежей и перевода средств."
            )],
            tools=self.tools
        ))
        print("Financial assistant started.")

    def create_tool(self, sender: Agent) -> BaseTool:
        """
        Create a tool for calling financial agent.
        :param sender: Who will call the agent
        :return: Tool to use
        """
        async def finance_help(request: str) -> str:
            """
            Инструмент позволяет решать финансовые вопросы для человека, в том числе совершать платежи

            Args:
                request: Просьба клиента, которую надо выполнить.
            """
            thread_id = uuid.uuid4().__str__()
            msg = Message(
                to = str(self.jid),
                body = request,
                thread=thread_id,
                metadata= {"performative": "request"}
            )
            response_template = Template(
                thread=thread_id,
                metadata= {"performative": "inform"}
            )
            snd = SendAndReceiveBehaviour(msg,response_template)
            sender.add_behaviour(snd, response_template)
            await snd.join()
            return snd.response.body

        return StructuredTool.from_function(
            name="finance_help",
            coroutine=finance_help,
            infer_schema=True,
            parse_docstring=True
        )

class PaymentAgent(Agent):
    """
    Simple LLM-free agent mocking payments and controlling user balance. Can perform both
    payments and replenish
    """
    balance: int

    def __init__(self, balance: int, jid: str, password: str):
        super().__init__(jid, password)
        self.balance = balance

    class ReceiveReplenishRequests(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(60)
            if msg:
                logger.info("Replenish request received %s", msg)
                amount = int(msg.body)
                self.agent.balance += amount
                logger.info("Balance after replenish is %i", self.agent.balance)

                response = Message(
                    to= str(msg.sender.jid),
                    thread=msg.thread,
                    body="Пополнение успешно зачислено.",
                    metadata={"performative": "acknowledge"}
                )
                await self.send(response)


    class ReceivePaymentRequests(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(60)
            if msg:
                logger.info("Payment request received %s", msg)
                amount = int(msg.body)

                if amount <= self.agent.balance:
                    self.agent.balance = self.agent.balance - int(msg.body)
                    logger.info("Balance after payment is %i", self.agent.balance)

                    response = Message(
                        to= str(msg.sender.jid),
                        thread=msg.thread,
                        body="Платеж проведен успешно.",
                        metadata={"performative": "inform"}
                    )
                    await self.send(response)
                else:
                    logger.info("Balance is insufficient for payment  %i", self.agent.balance)
                    response = Message(
                        to= str(msg.sender.jid),
                        thread=msg.thread,
                        body="Недостаточно средств для совершения платежа, пополните платежный счет на {0} рублей.".format(amount - self.agent.balance),
                        metadata={"performative": "inform"}
                    )
                await self.send(response)

    async def setup(self):
        logger.info("Payment agent is starting with balance %i",  self.balance)
        template = Template()
        template.metadata = {"performative": "request"}
        self.add_behaviour(self.ReceivePaymentRequests(), template)

        template = Template()
        template.metadata = {"performative": "request_with_acknowledge"}
        self.add_behaviour(self.ReceiveReplenishRequests(), template)

    def create_tool(self, sender: Agent) -> BaseTool:
        async def request_payment(amount: int) -> str:
            """
            Инструмент позволяет оплатить счет на указанную сумму в рублях с платежного счета

            Args:
                amount: Количество денег в рублях к оплате
            """
            thread_id = uuid.uuid4().__str__()
            msg = Message(
                to = str(self.jid),
                body = str(amount),
                thread=thread_id,
                metadata= {"performative": "request"}
            )
            response_template = Template(
                thread=thread_id,
                metadata= {"performative": "inform"}
            )
            snd = SendAndReceiveBehaviour(msg,response_template)
            sender.add_behaviour(snd, response_template)
            await snd.join()
            return snd.response.body

        return StructuredTool.from_function(
            name="payment_service",
            coroutine=request_payment,
            infer_schema=True,
            parse_docstring=True
        )

class SavingsAgent(Agent):
    """
    LLM-free agent serving user savings. Can pass some of saved money to payment agent
    if user approves.
    """
    balance: int
    payment_adders: str
    chat_jid: str

    def __init__(self, balance: int, jid: str, password: str, payment_adders: str, chat_jid: str):
        super().__init__(jid, password)
        self.chat_jid = chat_jid
        self.payment_adders = payment_adders
        self.balance = balance

    class RequestCycle(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(60)
            if msg:
                logger.info("Request for savings withdraw received received %s", msg)
                amount = int(msg.body)
                if amount <= self.agent.balance:

                    ack = await self.request_acknowledge(msg, amount)

                    if ack.response.metadata["performative"] != "acknowledge":
                        logger.error("Не получено подтверждение на снятие накоплений")
                        return await self.send(Message(
                            to= str(msg.sender.jid),
                            thread=msg.thread,
                            body="Клиент отказался от снятия накоплений.",
                            metadata={"performative": "inform"}
                        ))

                    await self.inform_payment_agent(amount)

                    self.agent.balance = self.agent.balance - int(msg.body)
                    logger.info("Savings after withdraw is %i", self.agent.balance)

                    response = Message(
                        to= str(msg.sender.jid),
                        thread=msg.thread,
                        body="Платежный счет пополнен из накоплений на {0} рублей.".format(amount),
                        metadata={"performative": "inform"}
                    )
                    await self.send(response)
                else:
                    logger.info("Savings are insufficient for withdrawal  %i", self.agent.balance)
                    response = Message(
                        to= str(msg.sender.jid),
                        thread=msg.thread,
                        body="Недостаточно средств для пополнения платежного счета.".format(amount - self.agent.balance),
                        metadata={"performative": "inform"}
                    )
                await self.send(response)

        async def inform_payment_agent(self, amount):
            logger.info("Informing payment agent about replenish.")
            thread_id = str(uuid.uuid4())
            sndr = SendAndReceiveBehaviour(
                message=Message(
                    to=self.agent.payment_adders,
                    thread=thread_id,
                    body=str(amount),
                    metadata={"performative": "request_with_acknowledge"}
                ),
                response_template=Template(
                    thread=thread_id,
                    metadata={"performative": "acknowledge"}
                )
            )
            self.agent.add_behaviour(sndr, sndr.response_template)
            await sndr.join()

        async def request_acknowledge(self, msg, amount):
            logger.info("Requesting acknowledgement from user.")
            ack = SendAndReceiveBehaviour(
                message=Message(
                    to=self.agent.chat_jid,
                    thread=msg.thread,
                    body="Перевод {0} рублей с накопительного счета на платежный".format(amount),
                    metadata={"performative": "request_with_acknowledge"}
                ),
                response_template=Template(
                    sender=self.agent.chat_jid,
                    thread=msg.thread
                )
            )
            self.agent.add_behaviour(ack, ack.response_template)
            await ack.join()
            return ack

    async def setup(self):
        logger.info("Savings agent is starting with balance %i",  self.balance)
        template = Template()
        template.metadata = {"performative": "request"}
        self.add_behaviour(self.RequestCycle(), template)

    def create_tool(self, sender: Agent) -> BaseTool:
        async def withdraw_savings(amount: int) -> str:
            """
            Инструмент позволяет пополнить платежный счет при недостаточном количестве денег из накоплений

            Args:
                amount: Количество денег в рублях для пополнения платежного счета
            """
            thread_id = uuid.uuid4().__str__()
            msg = Message(
                to = str(self.jid),
                body = str(amount),
                thread=thread_id,
                metadata= {"performative": "request"}
            )
            response_template = Template(
                thread=thread_id,
                metadata= {"performative": "inform"}
            )
            snd = SendAndReceiveBehaviour(msg,response_template)
            sender.add_behaviour(snd, response_template)
            await snd.join()
            return snd.response.body

        return StructuredTool.from_function(
            name="savings_service",
            coroutine=withdraw_savings,
            infer_schema=True,
            parse_docstring=True
        )