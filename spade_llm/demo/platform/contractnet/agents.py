import errno
import logging
import os
from collections import UserList
from pathlib import Path
from typing import Optional
from asyncio import sleep as asleep
import asyncio
import aiosqlite
from spade_llm.core.api import AgentContext
from aioconsole import ainput
from aiosqlite import Cursor, Connection
from async_lru import alru_cache
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from pydantic.fields import Field
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.core.api import Message
from spade_llm.demo.platform.contractnet.contractnet import ContractNetResponder, ContractNetRequest, \
    ContractNetProposal, \
    ContractNetResponderBehavior, ContractNetInitiatorBehavior
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentTask, DF_ADDRESS
from spade_llm.core.conf import configuration, Configurable
from spade_llm.demo import models
from spade_llm import consts

logger = logging.getLogger(__name__)


class Period(BaseModel):
    """Describes a period of time from start to end"""
    start: str = Field(description="Start of the period in the format YYYY-MM-DD")
    end: str = Field(description="End of the period in the format YYYY-MM-DD")


class UsersList(BaseModel):
    ids: list[int] = Field(description="List of clients matching request.")


class MccDescription(BaseModel):
    description: str = Field(description="Описание категории транзакций")


class MccExpertAgentConf(BaseModel):
    model: str = Field(description="Model to use for generating responses.")
    data_path: str = Field(description="Path to data files.", default="./data")


class SegmentAssemblerAgentConf(BaseModel):
    pass


@configuration(SegmentAssemblerAgentConf)
class SegmentAssemblerAgent(Agent, Configurable[SegmentAssemblerAgentConf]):
    @staticmethod
    def cformat(msg: str) -> str:
        """Utility for getting more visible messages in console"""
        return "\033[1m\033[92m{}\033[00m\033[00m".format(msg)

    class SegmentRequestBehaviour(MessageHandlingBehavior):
        def __init__(self, config: SegmentAssemblerAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            request = ContractNetInitiatorBehavior(
                task=self.message.content,
                context=self.context,
                time_to_wait_for_proposals=10
            )

            self.agent.add_behaviour(request)
            await request.join()
            if request.is_successful:
                result = UsersList.model_validate_json(request.result.content)
                head = ",".join([str(id) for id in result.ids[0:min(20, len(result.ids))]])
                await self.context.reply_with_inform(self.message).with_content(SegmentAssemblerAgent.cformat(
                    f"Получен сегмент:\nРазмер {len(result.ids)} ") + f"Первые 20 ids: {head}")
            else:
                await self.context.reply_with_failure(self.message).with_content(SegmentAssemblerAgent.cformat(
                    f"Не удалось собрать сегмент."))

    def setup(self):
        self.add_behaviour(self.SegmentRequestBehaviour(self.config))


@configuration(MccExpertAgentConf)
class MccExpertAgent(Agent, Configurable[MccExpertAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = models.EMBEDDINGS
        self.index = Chroma(embedding_function=self.embeddings, collection_name="MCC")

    class RequestBehaviour(MessageHandlingBehavior):
        index: VectorStore
        model: BaseChatModel
        parser = PydanticOutputParser(pydantic_object=MccDescription)

        prepare_request_prompt = PromptTemplate.from_template(
            """Твоя задача сформулировать описание MCC кода для транзакций, обозначающей траты людей 
            подпадающих под характеристику "{query}". Описание должно соответствовать следующим критериям:
            * Не менее 40 слов
            * Отсутствие привязки ко времени
            * Включает название товаров и услуг связанных с активностью
            * Включает типы торговых точек и предприятий где эти товары и услуги продаются
            Ответь в формате `json`\n{format_instructions}""")

        def __init__(self, config: MccExpertAgentConf, index: VectorStore, embeddings: Embeddings,
                     model: BaseChatModel):
            super().__init__(MessageTemplate.request())
            self.config = config
            self.index = index
            self.embeddings = embeddings
            self.model = model

        async def step(self) -> None:
            msg = self.message
            if msg:
                query = msg.content
                result = await self.find_mcc(query)
                await self.construct_reply(msg, query, result)

        @alru_cache(maxsize=1024)
        async def find_mcc(self, query: str):
            logger.info("Extracting code for segment description %s", query)
            chain = self.model | self.parser
            await asleep(0.5)
            request: MccDescription = await chain.ainvoke(
                await self.prepare_request_prompt.ainvoke(
                    {"query": query,
                     "format_instructions": self.parser.get_format_instructions()}))

            logger.info("Extended description for search %s", request.description)
            result = await self.index.asimilarity_search(query=request.description, k=1)
            return result

        async def construct_reply(self, msg, query, result):
            if len(result) > 0:
                doc = result[0]
                logger.info("Resulting document %s", doc)
                await self.context.reply_with_inform(msg).with_content(str(doc.metadata["MCC"]))
            else:
                logger.error("Failed to find document matching query %s", query)
                await self.context.reply_with_failure(msg).with_content("")

    async def prepare_data(self) -> None:
        logger.info("Preparing data")
        file_path = Path(self.config.data_path).joinpath("mcc_codes.csv")

        if not file_path.is_file() or not file_path.exists():
            logger.error("File with MCC code description not found at %s", file_path)
            await self.stop()
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path))

        logger.info("Loading MCC from %s", file_path)
        loader = CSVLoader(file_path=file_path.resolve(),
                           metadata_columns=["MCC"],
                           csv_args={
                               'delimiter': ',',
                               'quotechar': '"'
                           })

        documents = await loader.aload()
        logger.info("Loaded %i codes", len(documents))
        filtered = [x for x in documents if len(x.page_content) > 100]
        logger.info("%i codes have enough description", len(documents))
        await self.index.aadd_documents(filtered)
        logger.info("Codes added to index")

    async def async_setup(self):
        await self.prepare_data()

    def setup(self) -> None:
        asyncio.create_task(self.async_setup())
        self.add_behaviour(ContractNetResponderBehavior(self.config))
        self.add_behaviour(self.RequestBehaviour(self.config, index=self.index, embeddings=self.embeddings,
                                                 model=self.default_context.create_chat_model(self.config.model)))


class PeriodExpertAgentConf(BaseModel):
    date: str = Field(description="Date to use for calculating period.", default="2023-09-10")
    model: str = Field(description="Model to use for generating responses.")


@configuration(PeriodExpertAgentConf)
class PeriodExpertAgent(Agent, Configurable[PeriodExpertAgentConf]):
    class RequestBehaviour(MessageHandlingBehavior):
        def __init__(self, config: PeriodExpertAgentConf, model: BaseChatModel):
            super().__init__(MessageTemplate.request())
            self.parser = PydanticOutputParser(pydantic_object=Period)
            self.config = config
            self.model = model

        get_period_prompt = PromptTemplate.from_template(
            """Сегодняшняя датa {date}. Определи какой период времени озвучен в запросе "{query}" и ответь 
            в формате `json`\n{format_instructions}""")

        async def step(self) -> None:
            msg = self.message
            if msg:
                query = msg.content
                response = await self.get_period(query)
                if not response or not response.start or not response.end:
                    await self.context.reply_with_failure(msg).with_content('failed to get period')
                else:
                    await self.context.reply_with_inform(msg).with_content(response)

        @alru_cache(maxsize=1024)
        async def get_period(self, query: str) -> Period:
            await asyncio.sleep(1)
            logger.info("Extracting period for query %s", query)
            request = await self.get_period_prompt.ainvoke(
                {
                    "query": query,
                    "date": self.config.date,
                    "format_instructions": self.parser.get_format_instructions()
                }
            )
            chain = self.model | self.parser
            response: Period = await chain.ainvoke(request)
            return response

    def setup(self) -> None:
        self.add_behaviour(
            self.RequestBehaviour(self.config, model=self.default_context.create_chat_model(self.config.model)))


class SpendingProfileAgentConf(BaseModel):
    data_path: str = Field(description="Path to data files.", default="./data")
    table_name: str = Field(description="Table name to use for storing agents.", default="spendings")
    mcc_expert: str = Field(description="Agent ID of MCC expert.", default="mcc_expert")
    model: str = Field(description="Model to use for generating responses.")


@configuration(SpendingProfileAgentConf)
class SpendingProfileAgent(Agent, ContractNetResponder, Configurable[SpendingProfileAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class RequestBehaviour(ContextBehaviour):
        def __init__(self, context: AgentContext, config: SpendingProfileAgentConf, model: BaseChatModel,
                     db: Connection, request):
            super().__init__(context)
            self.db = db
            self.model = model
            self.table_name = config.table_name
            self.mcc_expert = config.mcc_expert
            self.config = config
            self.request = request
            self.result = None

        async def step(self):
            self.result = await self.handle(self.request.request.task)
            self.set_is_done()

        async def handle(self, task: str) -> UserList:
            response = await self.get_mcc(task)
            if response == None or response.performative == consts.FAILURE:
                logging.error("Не удалось определить MCC код для данного запроса.")
                return UserList(ids=[])
            else:
                mcc = int(response.content)
                rows = await self.request_ids(mcc, task)
                return UsersList(ids=[row[0] for row in rows])

        async def request_ids(self, mcc: int, query: str):

            cursor: Cursor = await self.db.execute(
                f'SELECT client_id FROM {self.table_name} WHERE "{mcc}">0')
            rows = await cursor.fetchall()
            await cursor.close()
            return rows

        async def get_mcc(self, task: str):
            await (self.context.request(self.config.mcc_expert).with_content(task))
            receiver = await self.receive(MessageTemplate(self.context.thread_id), timeout=25)
            return receiver

    async def check_table(self, file_path):
        cursor: Cursor = await self.db.execute(f"""SELECT type FROM sqlite_master WHERE
            name='{self.config.table_name}'; """)
        row = await cursor.fetchone()
        if row is None:
            logger.error("Table with data not found %s", self.config.table_name)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path.joinpath(self.table_name)))
        await cursor.close()

    async def connect_database(self):
        file_path = Path(self.config.data_path).joinpath("sqlite.db")
        if not file_path.is_file() or not file_path.exists():
            logger.error("Profile database not found at %s", file_path)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path))
        logger.info("Connecting to the database at %s", file_path)
        self.db = await aiosqlite.connect(file_path.resolve())
        return file_path

    async def make_db(self):
        file_path = await self.connect_database()
        await self.check_table(file_path)

    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        response = await self.check_task(request.task)
        logger.info("For request '%s' model response is '%s'", request.task, response.content)
        if response.content.startswith("Да"):
            return ContractNetProposal(author=str(self.agent_type), request=request, estimate=1.0)
        else:
            return None

    @alru_cache(maxsize=128)
    async def check_task(self, task: str):
        response = await self.model.ainvoke(
            f"""У тебя есть агрегированная информация о тратах людей за прошедшие 10 лет на разные 
            активности. В твоих данных нет деталей когда были траты, есть только общая итоговая сумма.
            Например, ты можешь найти людей которые в ходят в цирк, но не можешь найти тех,
            кто ходил в цирк в определенное время или пойдет в будущем. 
            Относится ли следующий запрос к общим активностям или активностям в конкретное время "{task}"? 

            Ответь Да если речь про общие активности, иначе ответь Нет.""")
        return response

    async def execute(self, proposal: ContractNetProposal, msg: Message):
        thread = await self.default_context.fork_thread()
        handler = self.RequestBehaviour(thread, self.config, self.default_context.create_chat_model(self.config.model),
                                        self.db, proposal)
        self.add_behaviour(handler)
        await handler.join()
        if handler.is_done():
            ret = handler.result
            await thread.close()
            return ret
        else:
            await thread.close()

    def create_description(self) -> AgentDescription:
        return AgentDescription(
            id="spending_agent",
            description="""Агент позволяет собрать сегмент людей с определенным профилем трат.
            Имеет доступ к агрегированным тратам за все время наблюдений.""",
            tasks=[
                AgentTask(
                    description="Найти людей с тратами в определенной категории",
                    examples=[
                        "Люди с тратами в магазинах хозтоваров",
                        "Покупающие продукты на рынках",
                        "Те, кто платит в ресторанах"
                    ]),
                AgentTask(
                    description="Найти людей c активностями, проявляющимися в тратах",
                    examples=[
                        "Люди, которые ходят в бары",
                        "Часто посещающие кафе",
                        "Те, кто пользуется автосервисами"
                    ])
            ]
        )

    async def setup_agent(self):
        await asleep(0.5)
        await self.register_in_df()

    def setup(self) -> None:
        asyncio.create_task(self.setup_agent())
        asyncio.create_task(self.make_db())
        self.add_behaviour(ContractNetResponderBehavior(self))
        self.model = self.default_context.create_chat_model(self.config.model)


    async def register_in_df(self):
        context = self.default_context
        await asleep(2)
        await (context.inform(DF_ADDRESS).with_content(self.create_description()))


class TransactionsAgentConf(BaseModel):
    data_path: str = Field(description="Path to data files.", default="./data")
    mcc_expert: str = Field(description="Agent ID of MCC expert.", default="mcc_expert")
    period_expert: str = Field(description="Agent ID of period expert.", default="period_expert")
    model: str = Field(description="Model to use for generating responses.")
    table_name: str = Field(description="Table name to use for storing agents.", default="transactions")
    date: str = Field(description="Current date.", default="2012-01-01")


@configuration(TransactionsAgentConf)
class TransactionsAgent(SpendingProfileAgent):
    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        response = await self.check_task(request.task)
        logger.info("For request '%s' model response is '%s'", request.task, response.content)
        if response.content.startswith("Да"):
            return ContractNetProposal(author=self.agent_type, estimate=10, request=request)
        else:
            return None

    @alru_cache(maxsize=128)
    async def check_task(self, task: str):
        await asleep(0.3)
        response = await self.model.ainvoke(
            f"""Сегодняшний день это {self.config.date}.
            У тебя есть агрегированная информация о тратах людей на разные активности за прошедшие 10 лет до сегодняшнего дня.
            В твоих данных есть детали когда были траты.
            Например, ты можешь найти людей которые в ходят в цирк, 
            или ходят в цирк конкретно весной, или тех кто ходил в цирк на прошлой неделе,
            но не можешь найти тех,
            кто ходил в цирк в периоды времени, которые нельзя представить в виде одного промежутка времени, 
            и ты не можешь найти тех, кто пойдет в будущем. 
            Можно ли ответить на следующий запрос, имея детали о времени трат "{task}"? 

            Ответь одним словом Да если можно, иначе ответь Нет.""")
        return response

    class RequestBehaviour(SpendingProfileAgent.RequestBehaviour):

        async def request_ids(self, mcc: int, query: str):
            msg = await self.get_period(query)
            if not msg or msg.performative == consts.FAILURE:
                return []
            else:
                period = Period.model_validate_json(msg.content)
                cursor: Cursor = await self.db.execute(
                    f"""SELECT client_id FROM {self.table_name} 
                    WHERE mcc={mcc} AND date BETWEEN '{period.start}' AND '{period.end}'
                    GROUP BY client_id
                    HAVING sum(amount)>0""")
                rows = await cursor.fetchall()
                await cursor.close()
                return rows

        async def get_period(self, query):
            await (self.context.request(self.config.period_expert).with_content(query))
            response = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=10)
            return response

    def create_description(self) -> AgentDescription:
        return AgentDescription(
            id="transaction_agent",
            description="""Агент позволяет собрать сегмент людей с определенным профилем трат.
            Имеет доступ к детальным транзакциям за все время наблюдений, способен делать выборки
            за определенный период.""",
            tasks=[
                AgentTask(
                    description="Найти людей с тратами в определенной категории и период",
                    examples=[
                        "Люди с тратами в магазинах хозтоваров в прошлом месяце",
                        "Покупавшие продукты на рынках в мае 2011-го",
                        "Те, кто платил в ресторанах в этот месяц в прошлом году"
                    ]),
                AgentTask(
                    description="Найти людей c активностями, проявляющимися в тратах в определенном периоде",
                    examples=[
                        "Люди, которые ходили в бар на прошлой неделе",
                        "Часто посещавшие кафе в прошлый вторник",
                        "Те, кто пользовался автосервисами вчера"
                    ])
            ]
        )
