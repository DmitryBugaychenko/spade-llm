import logging
from typing import Optional, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, Field
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from spade_llm.consts import Templates
from spade_llm.builders import MessageBuilder

_ID_SEPARATOR = "/"

logger = logging.getLogger(__name__)

# Pydantic
class AgentTask(BaseModel):
    """Represents a single task agent is capable for."""

    description: str = Field(description="Description of a task")
    examples: Optional[list[str]] = Field(description="Examples with a task query")

    def to_string(self) -> str:
        return self.description + "\n" + "\n".join(self.examples)

class AgentDescription(BaseModel):
    """Represents a description of a certain agent with its functions"""

    description: str = Field(description="Description of the agent role and abilities")
    tasks: Optional[list[AgentTask]] = Field(description="Tasks agent is capable for", default = [])
    domain: Optional[str] = Field(description="Domain agent is expert at", default="")
    tier: int = Field(default = 0, description="Tier the agent belongs to. During discovery process agents can "
                                               "lookup only agents in the tiers below and in the same domain.")
    id: str = Field(description="Unique identifier of an agent")

    def _to_documents(self) -> list[Document]:
        yield Document(
            page_content=self.description,
            id = _ID_SEPARATOR.join([self.id, '0']),
            metadata = {"domain" : self.domain, "tier": self.tier}
        )

        counter : int = 1
        for task in self.tasks:
            counter += 1
            yield Document(
                page_content=self.description + "\n" + task.to_string(),
                id = _ID_SEPARATOR.join([self.id, str(counter)]),
                metadata = {"domain" : self.domain, "tier": self.tier}
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

class DirectoryFacilitatorAgent(Agent):
     _embeddings: Embeddings
     _index: VectorStore
     _agents: dict[str,AgentDescription] = dict()
     threshold: float = 400
     score_ascending: bool = True

     def __init__(self,
                  jid: str,
                  password: str,
                  embeddings: Embeddings,
                  port: int = 5222,
                  verify_security: bool = False):
         super().__init__(jid,password, port, verify_security)
         self._embeddings = embeddings
         self._index = Chroma(embedding_function=embeddings)

     def filter(self, score: float) -> bool:
         return (self.score_ascending and score <= self.threshold) or (not self.score_ascending and score >= self.threshold)

     class RegisterAgent(CyclicBehaviour):
         """Behaviour listening for agents registration and adding agents to the index"""
         index: VectorStore
         agents: dict[str,AgentDescription]

         async def on_start(self) -> None:
             self.index = self.agent._index
             self.agents = self.agent._agents

         async def run(self) -> None:
             msg = await self.receive(10)
             if msg:
                 description = AgentDescription.model_validate_json(json_data=msg.body)
                 self.agents[description.id] = description
                 logger.debug("Got description %s",msg.body )
                 logger.info("Registering agent %s", description.id)
                 await self.index.aadd_documents(description.to_documents())
                 await self.send(MessageBuilder.reply_with_ack(msg).with_content(""))


     class SearchForAgent(CyclicBehaviour):
         """Behaviour handling requests for search looking up th index"""
         index: VectorStore
         agents: dict[str,AgentDescription]
         filter: Callable[[float], bool]

         async def on_start(self) -> None:
            self.index = self.agent._index
            self.agents = self.agent._agents
            self.filter = self.agent.filter

         def filter_results(self, result: list[(Document,float)], k: int) -> list[AgentDescription]:
            found = set()
            returned = 0
            for doc, score in result:
                agent_id = AgentDescription.extract_agent_id(doc)
                if agent_id not in found and self.filter(score):
                    found.add(agent_id)
                    yield self.agents.get(agent_id)
                returned += 1
                if returned >= k:
                    break

         async def run(self) -> None:
             msg = await self.receive(10)
             if msg:
                 logger.debug("Got search request %s",msg.body )
                 request = AgentSearchRequest.model_validate_json(json_data=msg.body)
                 result = await self.index.asimilarity_search_with_score(request.task)
                 filtered = list(self.filter_results(result, request.top_k))

                 await self.send(MessageBuilder.reply_with_inform(msg).with_content(
                     AgentSearchResponse(agents = filtered)))

     async def setup(self) -> None:
         self.add_behaviour(
             self.RegisterAgent(), Templates.INFORM())
         self.add_behaviour(
             self.SearchForAgent(), Templates.REQUEST())