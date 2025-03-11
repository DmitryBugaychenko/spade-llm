import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates
from spade_llm.discovery import AgentDescription, SearchForAgentBehavior, AgentSearchResponse, AgentSearchRequest

logger = logging.getLogger(__name__)

class ContractNetRequest(BaseModel):
    task: str = Field(description="Description of the task to create proposal for")

class ContractNetProposal(BaseModel):
    author: str = Field(description="Agent created this proposal")
    request: ContractNetRequest = Field(description="Request this proposal is created for")
    estimate: float = Field(description="Estimation for the task")

class ContractNetResponder(ABC):
    @abstractmethod
    async def estimate(self, request: ContractNetRequest, msg: Message) -> Optional[ContractNetProposal]:
        pass

    @abstractmethod
    async def execute(self, proposal: ContractNetProposal, msg: Message) -> BaseModel:
        pass

class ContractNetResponderBehavior(CyclicBehaviour):
    _responder: ContractNetResponder

    def __init__(self, responder: ContractNetResponder):
        super().__init__()
        self._responder = responder


    class RequestHandler(OneShotBehaviour):
        _request: Message
        _responder: ContractNetResponder

        def __init__(self, request: Message, responder: ContractNetResponder):
            super().__init__()
            self._responder = responder
            self._request = request

        async def run(self) -> None:
            request = self._request
            parsed_request = ContractNetRequest.model_validate_json(request.body)
            proposal = await self._responder.estimate(parsed_request, request)

            if proposal:
                await self.send(MessageBuilder.reply_with_propose(request).with_content(proposal))

                response = await self.receive(10)
                if response and Templates.ACCEPT().match(response):
                    result = await(self._responder.execute(proposal, request))
                    await self.send(MessageBuilder.reply_with_inform(request).with_content(result))
            else:
                await self.send(MessageBuilder.reply_with_refuse(request).with_content(""))


    async def run(self) -> None:
        msg = await self.receive(10)
        if msg:
            self.agent.add_behaviour(
                self.RequestHandler(msg, self._responder),
                Templates.from_thread(msg.thread))


class ContractNetInitiatorBehavior(OneShotBehaviour):
    task: str
    df_address: str
    proposal: Optional[ContractNetProposal] = None
    result: Optional[Message]
    time_to_wait_for_proposals: float
    _started_at: float
    thread: str

    def __init__(self,
                 task: str,
                 df_address: str,
                 thread: str = str(uuid.uuid4()),
                 time_to_wait_for_proposals: float = 10):
        super().__init__()
        self.thread = thread
        self.time_to_wait_for_proposals = time_to_wait_for_proposals
        self.df_address = df_address
        self.task = task

    async def on_start(self) -> None:
        self._started_at = time.time()

    @property
    def is_successful(self) -> bool:
        if self.result and Templates.INFORM().match(self.result):
            return True
        else:
            return False

    def construct_template(self) -> Template:
        return Templates.from_thread(self.thread)

    async def run(self) -> None:
        agents : list[AgentDescription] = await self.find_agents(self.task, self.df_address)
        logger.info("Found %i agents for task '%s'", len(agents), self.task)

        if len(agents) > 0:
            proposals: list[ContractNetProposal] = await self.get_proposals(agents)
            logger.info("Got %i proposals for task '%s'", len(proposals), self.task)
            if len(proposals) > 0:
                self.proposal = await self.extract_winner_and_notify_losers(proposals)
                logger.info("Best proposal for task '%s' is %s", self.task, self.proposal)
                if self.proposal:
                    self.result = await self.get_result(self.proposal)
                    to_log = str(self.result)
                    logger.info("Result for task '%s' is %s", self.task, to_log[0:min(256, len(to_log))])
                else:
                    logger.error("Failed to select best proposal for '%s'", self.task)
            else:
                logger.error("Failed to get any proposals for '%s' in time", self.task)
        else:
            logger.error("Failed to find any agents for '%s'", self.task)

    async def find_agents(self, task: str, df_address: str) -> list[AgentDescription]:
        search = SearchForAgentBehavior(
            AgentSearchRequest(task = task, top_k = 10),
            df_address)

        self.agent.add_behaviour(search, search.response_template)

        await search.join(self.time_to_wait_for_proposals)

        if search.response and Templates.INFORM().match(search.response):
            parsed = AgentSearchResponse.model_validate_json(search.response.body)
            return parsed.agents
        else:
            return list()

    async def get_proposals(self, agents: list[AgentDescription]) -> list[ContractNetProposal]:
        sent: set[str] = set()
        received: set[str] = set()
        result: list[ContractNetProposal] = list()

        request = ContractNetRequest(task = self.task)
        for agent in agents:
            sent.add(agent.id)
            await self.send(MessageBuilder
                            .request_porposal()
                            .to_agent(agent.id)
                            .in_thread(self.thread)
                            .with_content(request))
        started = time.time()
        deadline = started + self.time_to_wait_for_proposals
        while len(received) < len(sent) and time.time() < deadline:
            response = await self.receive(max(0.1, deadline - time.time()))
            if response:
                received.add(str(response.sender))
                if Templates.PROPOSE().match(response):
                    result.append(ContractNetProposal.model_validate_json(response.body))

        return result

    async def extract_winner_and_notify_losers(self, proposals : list[ContractNetProposal]) -> ContractNetProposal:
        proposals.sort(key = lambda x: x.estimate)
        winner = proposals[0]

        for agent in proposals[1:]:
            await self.send(MessageBuilder
                      .refuse()
                      .to_agent(agent.author)
                      .in_thread(self.thread)
                      .with_content(""))

        return winner

    async def get_result(self, proposal: ContractNetProposal) -> Optional[Message]:
        await self.send(MessageBuilder
                  .accept()
                  .to_agent(proposal.author)
                  .in_thread(self.thread)
                  .with_content(proposal))

        return await self.receive(10)
