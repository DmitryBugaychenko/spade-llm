import logging

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.demo import models

from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import configuration, Configurable

_ID_SEPARATOR = "/"

logger = logging.getLogger(__name__)


class AgentTask(BaseModel):
    """Represents a single task agent is capable for."""

    description: str = Field(description="Description of a task")
    examples: Optional[list[str]] = Field(description="Examples with a task query")

    def to_string(self) -> str:
        return self.description + "\n" + "\n".join(self.examples)


class AgentDescription(BaseModel):
    """Represents a description of a certain agent with its functions"""

    description: str = Field(description="Description of the agent role and abilities")
    tasks: Optional[list[AgentTask]] = Field(description="Tasks agent is capable for", default=[])
    domain: Optional[str] = Field(description="Domain agent is expert at", default="")
    tier: int = Field(default=0, description="Tier the agent belongs to. During discovery process agents can "
                                             "lookup only agents in the tiers below and in the same domain.")
    id: str = Field(description="Unique identifier of an agent")

    def _to_documents(self) -> list[Document]:
        yield Document(
            page_content=self.description,
            id=_ID_SEPARATOR.join([self.id, '0']),
            metadata={"domain": self.domain, "tier": self.tier}
        )

        counter: int = 1
        for task in self.tasks:
            counter += 1
            yield Document(
                page_content=self.description + "\n" + task.to_string(),
                id=_ID_SEPARATOR.join([self.id, str(counter)]),
                metadata={"domain": self.domain, "tier": self.tier}
            )

    def to_documents(self) -> list[Document]:
        return list(self._to_documents())

    @staticmethod
    def extract_agent_id(doc: Document) -> str:
        return "".join(doc.id.split(_ID_SEPARATOR)[:-1])


class AgentSearchRequest(BaseModel):
    """Search query used to lookup agents for solving tasks"""

    task: str = Field(description="Description of the task we need to find agent for.")
    top_k: int = Field(default=4, description="Maximum number of agents to report.")


class AgentSearchResponse(BaseModel):
    """Response to the search request with list of the agents"""
    agents: list[AgentDescription] = Field(description="List of agent descriptions for agents matching search query.")


class DirectoryFacilitatorAgentConf(BaseModel):
    agents: dict = Field(description="Dictionary of agents to register.", default={})
    threshold: float = Field(default=400, description="Threshold for searching agents.")
    score_ascending: bool = Field(default=True, description="Should the search be ascending or descending.")


class RegisterAgentBehaviour(MessageHandlingBehavior):
    """Behaviour registering agents into the index"""

    def __init__(self, config: DirectoryFacilitatorAgentConf, index: VectorStore, embeddings: Embeddings):
        super().__init__(MessageTemplate.inform())
        self.config = config
        self.index = index
        self.embeddings = embeddings

    async def step(self):
        if self.message:
            # May be problem during souble register request in one time
            description = AgentDescription.model_validate_json(json_data=self.message.content)
            self.config.agents[description.id] = description
            logger.debug("Got description %s", self.message.content)
            logger.info("Registering agent %s", description.id)
            await self.index.aadd_documents(description.to_documents())
            await (self.context.reply_with_acknowledge(self.message).with_content(""))


class SearchForAgentBehaviour(MessageHandlingBehavior):
    """Behaviour handling requests for search looking up th index"""

    def __init__(self, config: DirectoryFacilitatorAgentConf, index: VectorStore, embeddings: Embeddings):
        super().__init__(MessageTemplate.request())
        self.config = config
        self.index = index
        self.embeddings = embeddings

    def filter(self, score: float) -> bool:
        return (self.config.score_ascending and score <= self.config.threshold) or (
                not self.config.score_ascending and score >= self.config.threshold)

    def filter_results(self, result: list[(Document, float)], k: int) -> list[AgentDescription]:
        found = set()
        returned = 0
        for doc, score in result:
            agent_id = AgentDescription.extract_agent_id(doc)
            if agent_id not in found and self.filter(score):
                found.add(agent_id)
                yield self.config.agents.get(agent_id)
            returned += 1
            if returned >= k:
                break

    async def step(self):
        if self.message:
            logger.debug("Got search request %s", self.message.content)
            request = AgentSearchRequest.model_validate_json(json_data=self.message.content)
            result = await self.index.asimilarity_search_with_score(request.task)
            filtered = list(self.filter_results(result, request.top_k))
            await self.context.reply_with_inform(self.message).with_content(
                AgentSearchResponse(agents=filtered))


@configuration(DirectoryFacilitatorAgentConf)
class DirectoryFacilitatorAgent(Agent, Configurable[DirectoryFacilitatorAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = models.EMBEDDINGS
        self.index = Chroma(embedding_function=self.embeddings, collection_name="AGENTS")

    def setup(self):
        self.add_behaviour(RegisterAgentBehaviour(self.config, index=self.index, embeddings=self.embeddings))
        self.add_behaviour(SearchForAgentBehaviour(self.config, index=self.index, embeddings=self.embeddings))
