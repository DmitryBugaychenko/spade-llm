import errno
import logging
import os
from async_lru import alru_cache
from pathlib import Path

import aiosqlite
from aiosqlite import Cursor, Connection
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from pydantic.fields import Field
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from spade_llm.behaviours import SendAndReceiveBehaviour
from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates

logger = logging.getLogger(__name__)

class Period(BaseModel):
    """Describes a period of time from start to end"""
    start: str = Field(description="Start of the period in the format YYYY-MM-DD")
    end: str = Field(description="End of the period in the format YYYY-MM-DD")

class UsersList(BaseModel):
    ids : list[int] = Field(description="List of clients matching request.")

class MccDescription(BaseModel):
    description: str = Field(description="Описание категории транзакций")


class MccExpertAgent(Agent):
    _embeddings: Embeddings
    _index: VectorStore
    _model: BaseChatModel
    _data_path: str

    class RequestBehaviour(CyclicBehaviour):
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

        async def on_start(self) -> None:
            self.index = self.agent._index
            self.model = self.agent._model

        async def run(self) -> None:
            msg = await self.receive(10)
            if msg:
                query = msg.body
                result = await self.find_mcc(query)
                reply = await self.construct_reply(msg, query, result)
                await self.send(reply)

        @alru_cache(maxsize=1024)
        async def find_mcc(self, query: str) :
            logger.info("Extracting code for segment description %s", query)
            chain = self.model | self.parser
            request: MccDescription = await chain.ainvoke(
                await self.prepare_request_prompt.ainvoke(
                    {"query": query,
                     "format_instructions" : self.parser.get_format_instructions()}))

            logger.info("Extended description for search %s", request.description)
            result = await self.index.asimilarity_search(query=request.description, k=1)
            return result

        async def construct_reply(self, msg, query, result):
            if len(result) > 0:
                doc = result[0]
                logger.info("Resulting document %s", doc)
                return MessageBuilder.reply_with_inform(msg).with_content(str(doc.metadata["MCC"]))
            else:
                logger.error("Failed to find document matching query %s", query)
                return MessageBuilder.reply_with_failure(msg).with_content("")

    def __init__(self,
                 embeddings: Embeddings,
                 model: BaseChatModel,
                 data_path: str,
                 jid: str,
                 password: str,
                 port: int = 5222,
                 verify_security: bool = False):
        super().__init__(jid,password, port, verify_security)
        self._embeddings = embeddings
        self._model = model
        self._data_path = data_path
        self._index = Chroma(embedding_function=embeddings)

    async def setup(self) -> None:
        file_path = Path(self._data_path).joinpath("mcc_codes.csv")

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
        await self._index.aadd_documents(filtered)
        logger.info("Codes added to index")

        self.add_behaviour(self.RequestBehaviour(), Templates.REQUEST())

class PeriodExpertAgent(Agent):
    _model: BaseChatModel
    _date: str

    def __init__(self,
                 model: BaseChatModel,
                 date: str,
                 jid: str,
                 password: str,
                 port: int = 5222,
                 verify_security: bool = False):
        super().__init__(jid,password, port, verify_security)
        self._date = date
        self._model = model

    class RequestBehaviour(CyclicBehaviour):
        model: BaseChatModel
        date: str
        parser = PydanticOutputParser(pydantic_object=Period)

        get_period_prompt = PromptTemplate.from_template(
            """Сегодняшняя датa {date}. Определи какой период времени озвучен в запросе "{query}" и ответь 
            в формате `json`\n{format_instructions}""")

        async def on_start(self) -> None:
            self.model = self.agent._model
            self.date = self.agent._date

        async def run(self) -> None:
            msg = await self.receive(10)
            if msg:
                query = msg.body

                response = await self.get_period(query)

                await self.send(MessageBuilder.reply_with_inform(msg).with_content(response))

        @alru_cache(maxsize=1024)
        async def get_period(self, query: str) -> Period:
            request = await self.get_period_prompt.ainvoke(
                {
                    "query": query,
                    "date": self.date,
                    "format_instructions": self.parser.get_format_instructions()
                }
            )
            chain = self.model | self.parser
            response: Period = await chain.ainvoke(request)
            return response

    async def setup(self) -> None:
        self.add_behaviour(self.RequestBehaviour(), Templates.REQUEST())

class SpendingProfileAgent(Agent):
    _data_path: str
    _table_name: str = "spendings"
    _db: Connection
    _mcc_expert: str

    def __init__(self, data_path: str, mcc_expert: str,
                 jid: str, password: str,
                 port: int = 5222, verify_security: bool = False):
        super().__init__(jid, password, port, verify_security)
        self._mcc_expert = mcc_expert
        self._data_path = data_path

    class RequestBehaviour(CyclicBehaviour):
        db: Connection
        table_name: str
        mcc_expert: str

        async def on_start(self) -> None:
            self.db = self.agent._db
            self.mcc_expert = self.agent._mcc_expert
            self.table_name = self.agent._table_name

        async def run(self) -> None:
            msg = await self.receive(10)
            if msg:
                mcc_request = (MessageBuilder
                               .request()
                               .follow_or_create_thread(msg)
                               .to_agent(self.mcc_expert)
                               .with_content(msg.body))

                receiver = SendAndReceiveBehaviour(
                    message= mcc_request,
                    response_template= Templates.from_thread(mcc_request.thread) and
                                       (Templates.INFORM() or Templates.FAILURE()))
                self.agent.add_behaviour(receiver, receiver.response_template)

                await receiver.join(10)
                response = receiver.response

                if response == None or Templates.FAILURE().match(response):
                    await self.send(MessageBuilder.reply_with_failure(msg)
                              .with_content("Не удалось определить MCC код для данного запроса."))
                else:
                    mcc = int(response.body)
                    cursor : Cursor = await self.db.execute(
                        f'SELECT client_id FROM {self.table_name} WHERE "{mcc}">0')
                    rows = await cursor.fetchall()
                    await cursor.close()
                    await self.send(MessageBuilder.reply_with_inform(msg).with_content(
                        UsersList(ids = [row[0] for row in rows])
                    ))


    async def setup(self) -> None:
        file_path = Path(self._data_path).joinpath("sqlite.db")

        if not file_path.is_file() or not file_path.exists():
            logger.error("Profile database not found at %s", file_path)
            await self.stop()
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path))

        logger.info("Connecting to the database at %s", file_path)
        self._db = await aiosqlite.connect(file_path.resolve())

        cursor: Cursor = await self._db.execute(f"""SELECT type FROM sqlite_master WHERE
            name='{self._table_name}'; """)
        row = await cursor.fetchone()
        if row is None:
            logger.error("Table with data not found %s", self._table_name)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path.joinpath(self._table_name)))

        await cursor.close()

        self.add_behaviour(self.RequestBehaviour(), Templates.REQUEST())

    async def stop(self) -> None:
        await super().stop()
        await self._db.close()

class SpendingProfileAgent(Agent):
    _data_path: str
    _table_name: str = "spendings"
    _db: Connection
    _mcc_expert: str

    def __init__(self, data_path: str, mcc_expert: str,
                 jid: str, password: str,
                 port: int = 5222, verify_security: bool = False):
        super().__init__(jid, password, port, verify_security)
        self._mcc_expert = mcc_expert
        self._data_path = data_path

    class RequestBehaviour(CyclicBehaviour):
        db: Connection
        table_name: str
        mcc_expert: str

        async def on_start(self) -> None:
            self.db = self.agent._db
            self.mcc_expert = self.agent._mcc_expert
            self.table_name = self.agent._table_name

        async def run(self) -> None:
            msg = await self.receive(10)
            if msg:
                response = await self.get_mcc(msg)

                if response == None or Templates.FAILURE().match(response):
                    await self.send(MessageBuilder.reply_with_failure(msg)
                                    .with_content("Не удалось определить MCC код для данного запроса."))
                else:
                    mcc = int(response.body)

                    rows = await self.request_ids(mcc, response.thread, msg.body)
                    await self.send(MessageBuilder.reply_with_inform(msg).with_content(
                        UsersList(ids = [row[0] for row in rows])
                    ))

        async def request_ids(self, mcc: int, thread: str, query: str):
            cursor: Cursor = await self.db.execute(
                f'SELECT client_id FROM {self.table_name} WHERE "{mcc}">0')
            rows = await cursor.fetchall()
            await cursor.close()
            return rows

        async def get_mcc(self, msg):
            mcc_request = (MessageBuilder
                           .request()
                           .follow_or_create_thread(msg)
                           .to_agent(self.mcc_expert)
                           .with_content(msg.body))
            receiver = SendAndReceiveBehaviour(
                message=mcc_request,
                response_template=Templates.from_thread(mcc_request.thread) and
                                  (Templates.INFORM() or Templates.FAILURE()))
            self.agent.add_behaviour(receiver, receiver.response_template)
            await receiver.join(10)
            response = receiver.response
            return response

    async def setup(self) -> None:
        file_path = await self.connect_database()

        await self.check_table(file_path)

        await self.add_behaviours()

    async def add_behaviours(self):
        self.add_behaviour(self.RequestBehaviour(), Templates.REQUEST())

    async def check_table(self, file_path):
        cursor: Cursor = await self._db.execute(f"""SELECT type FROM sqlite_master WHERE
            name='{self._table_name}'; """)
        row = await cursor.fetchone()
        if row is None:
            logger.error("Table with data not found %s", self._table_name)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path.joinpath(self._table_name)))
        await cursor.close()

    async def connect_database(self):
        file_path = Path(self._data_path).joinpath("sqlite.db")
        if not file_path.is_file() or not file_path.exists():
            logger.error("Profile database not found at %s", file_path)
            await self.stop()
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path))
        logger.info("Connecting to the database at %s", file_path)
        self._db = await aiosqlite.connect(file_path.resolve())
        return file_path

    async def stop(self) -> None:
        await super().stop()
        await self._db.close()

class TransactionsAgent(SpendingProfileAgent):
    _table_name: str = "transactions"
    _period_expert: str

    def __init__(self, data_path: str, mcc_expert: str, period_expert: str,
                 jid: str, password: str, port: int = 5222,
                 verify_security: bool = False):
        super().__init__(data_path, mcc_expert, jid, password, port, verify_security)
        self._period_expert = period_expert


    class RequestBehaviour(SpendingProfileAgent.RequestBehaviour):
        period_expert: str
        async def on_start(self) -> None:
            await super().on_start()
            self.period_expert = self.agent._period_expert

        async def request_ids(self, mcc: int, thread: str, query: str):
            period_request = (MessageBuilder.request()
                              .to_agent(self.period_expert)
                              .in_thread(thread)
                              .with_content(query))
            receiver = SendAndReceiveBehaviour(period_request, Templates.from_thread(thread))
            self.agent.add_behaviour(receiver, receiver.response_template)
            await receiver.join(10)
            response = receiver.response
            if response == None or Templates.FAILURE().match(response):
                return []
            else:
                period = Period.model_validate_json(response.body)
                cursor: Cursor = await self.db.execute(
                    f"""SELECT client_id FROM {self.table_name} 
                    WHERE mcc={mcc} AND date BETWEEN '{period.start}' AND '{period.end}'
                    GROUP BY client_id
                    HAVING sum(amount)>0""")
                rows = await cursor.fetchall()
                await cursor.close()
                return rows




    async def add_behaviours(self):
        self.add_behaviour(self.RequestBehaviour(), Templates.REQUEST())