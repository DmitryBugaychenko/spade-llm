from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import PromptTemplate
import logging
import errno
import os
from pathlib import Path
from typing import List
import asyncio
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from spade_llm.core.models import ChatModelConfiguration
from spade_llm.core.conf import Args
from spade_llm.demo import models
from langchain_core.vectorstores import VectorStore
from spade_llm.core.conf import ConfigurableRecord, Configurable, configuration
from pydantic import BaseModel, Field, ConfigDict
from spade_llm.core.tools import ToolFactory
from langchain_core.tools import tool, BaseTool, BaseToolkit


logger = logging.getLogger(__name__)

class MccToolkitConf(ConfigurableRecord):
    pass


class MccDescription(BaseModel):
    description: str = Field(description="Описание категории транзакций")


class MccToolkit(BaseToolkit):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: BaseChatModel = Field(default=None, description="LLM model")
    index: VectorStore = Field(default=None, description="Database with MCC codes")

    def __init__(self):
        super().__init__()
        conf = {
            "max": ChatModelConfiguration(
                type_name="spade_llm.models.gigachat.GigaChatModelFactory",
                args=Args(
                    credentials="env.GIGA_CRED",
                    scope="env.GIGACHAT_SCOPE",
                    model="GigaChat-2-Max",
                    verify_ssl_certs=False,
                ),
            )
        }
        self.model = conf["max"].create_model_factory().create_model()
        self.index = Chroma(
            embedding_function=models.EMBEDDINGS, collection_name="MCC"
        )

    async def prepare_data(self) -> None:
        logger.info("Preparing data")
        file_path = Path("./data").joinpath("mcc_codes.csv")

        if not file_path.is_file() or not file_path.exists():
            logger.error("File with MCC code description not found at %s", file_path)
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(file_path)
            )

        logger.info("Loading MCC from %s", file_path)
        loader = CSVLoader(
            file_path=file_path.resolve(),
            metadata_columns=["MCC"],
            csv_args={"delimiter": ",", "quotechar": '"'},
        )

        documents = await loader.aload()
        logger.info("Loaded %i codes", len(documents))
        filtered = [x for x in documents if len(x.page_content) > 100]
        logger.info("%i codes have enough description", len(documents))
        await asyncio.sleep(1)
        await self.index.aadd_documents(filtered)
        logger.info("Codes added to index")

    async def async_setup(self):
        await self.prepare_data()

    @property
    def get_tools(self) -> List[BaseTool]:
        """Returns a list of tools"""
        asyncio.create_task(self.async_setup())
        lst = [
            self._create_mcc_descriptor(),
            self._create_mcc_finder(),
        ]
        return lst

    def _create_mcc_descriptor(self) -> BaseTool:
        parser = PydanticOutputParser(pydantic_object=MccDescription)
        prepare_request_prompt = PromptTemplate.from_template(
            """Your task is to formulate a description of an MCC code for transactions representing spending by people 
            who fall under the characteristic "{query}". The description must meet the following criteria:
            * At least 40 words long
            * No time-specific references
            * Includes names of goods and services related to the activity
            * Includes types of retail outlets and businesses where these goods and services are sold
            Respond in `json` format\n{format_instructions}. JSON only, without Markdown and additional text"""
        )

        @tool
        async def mcc_descriptor(query: str):
            """Returns a detailed description of a spending category based on the query.

            Args:
                query: User query (e.g., 'Restaurant visitors') as a string.

            Returns:
                Detailed description of the spending category based on the query

            Examples:
                >>> mcc_descriptor("People who go to bars")  # Returns a detailed description of the Bars category
                >>> mcc_descriptor("Restaurants")  # Search by category name
            """
            chain = self.model | parser
            request: MccDescription = await chain.ainvoke(
                await prepare_request_prompt.ainvoke(
                    {
                        "query": query,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
            )
            return request.description

        return mcc_descriptor

    def _create_mcc_finder(self) -> BaseTool:
        @tool
        async def mcc_finder(description: str):
            """Finds appropriate MCC codes based on business activity description.
            Before using this tool, call mcc_descriptor to get the MCC code description.

            Args:
                description: Business or activity description (e.g., 'clothing retail') This parameter should be very detailed.

            Returns:
                Appropriate MCC code as a number

            Examples:
                >>> mcc_finder("clothing retail")  # Find codes for clothing retail
                >>> mcc_finder("taxi services")  # Find codes for transportation services
            """
            result = await self.index.asimilarity_search(query=description, k=1)
            doc = result[0]
            return str(doc.metadata["MCC"])

        return mcc_finder


@configuration(MccToolkitConf)
class MccToolFactory(ToolFactory, Configurable[MccToolkitConf]):
    def create_tool(self) -> List[BaseTool]:
        return self.config.create_kwargs_instance(MccToolkit).get_tools
