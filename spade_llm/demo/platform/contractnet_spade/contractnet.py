import logging
import time
from abc import ABC, abstractmethod
from typing import Optional
from spade_llm import consts
from pydantic import BaseModel, Field
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.core.api import AgentContext
from spade_llm.demo.platform.contractnet_spade.discovery import AgentDescription, AgentSearchResponse, AgentSearchRequest
from spade_llm.core.api import Message

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


class ContractNetResponderBehavior(MessageHandlingBehavior):
    _responder: ContractNetResponder

    def __init__(self, responder: ContractNetResponder):
        super().__init__(MessageTemplate.request_proposal())
        self._responder = responder

    class AcceptBehavior(MessageHandlingBehavior):
        def __init__(self, responder: ContractNetResponder, proposal):
            super().__init__(MessageTemplate.accept())
            self._responder = responder
            self.proposal = proposal

        async def step(self) -> None:
            msg = self.message
            if msg and msg.performative == consts.ACCEPT:
                result = await(self._responder.execute(self.proposal, msg))
                await self.context.reply_with_inform(msg).with_content(result)
                self.set_is_done()

    async def step(self) -> None:
        if self.message:
            msg = self.message
            parsed_request = ContractNetRequest.model_validate_json(msg.content)
            proposal = await self._responder.estimate(parsed_request, msg)
            if proposal:
                await self.context.reply_with_propose(msg).with_content(proposal)
                handler = self.AcceptBehavior(self._responder, proposal)
                self.agent.add_behaviour(handler)
                await handler.join()

            else:
                await self.context.reply_with_refuse(msg).with_content("")


class ContractNetInitiatorBehavior(ContextBehaviour):
    task: str
    df_address: str
    proposal: Optional[ContractNetProposal] = None
    result: Optional[MessageTemplate]
    time_to_wait_for_proposals: float
    _started_at: float

    def __init__(self,
                 task: str,
                 df_address: str,
                 context: AgentContext,
                 time_to_wait_for_proposals: float = 10):
        super().__init__(context)
        self.time_to_wait_for_proposals = time_to_wait_for_proposals
        self.df_address = df_address
        self.task = task
        self.result = None

    async def on_start(self) -> None:
        self._started_at = time.time()

    @property
    def is_successful(self) -> bool:
        if self.result and self.result.performative == consts.INFORM:
            return True
        else:
            return False

    async def step(self) -> None:
        agents: list[AgentDescription] = await self.find_agents(self.task)
        logger.info("Found %i agents for task '%s'", len(agents), self.task)

        if len(agents) > 0:
            proposals: list[ContractNetProposal] = await self.get_proposals(agents)
            logger.info("Got %i proposals for task '%s'", len(proposals), self.task)
            if len(proposals) > 0:
                self.proposal = await self.extract_winner_and_notify_losers(proposals)
                logger.info("Best proposal for task '%s' is %s", self.task, self.proposal)
                if self.proposal:
                    self.result = await self.get_result(self.proposal)
                    to_log = str(self.result.content)
                    logger.info("Result for task '%s' is %s", self.task, to_log[0:min(256, len(to_log))])
                else:
                    logger.error("Failed to select best proposal for '%s'", self.task)
            else:
                logger.error("Failed to get any proposals for '%s' in time", self.task)
        else:
            logger.error("Failed to find any agents for '%s'", self.task)
        self.set_is_done()

    async def find_agents(self, task: str) -> list[AgentDescription]:
        thread = await self.context.fork_thread()
        await (thread
               .request("df")
               .with_content(AgentSearchRequest(task=task, top_k=10)))

        search = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=10)
        if not search:
            print(f"No search recieved in {10} seconds")

        await thread.close()
        if search and search.performative == consts.INFORM:
            parsed = AgentSearchResponse.model_validate_json(search.content)
            return parsed.agents
        else:
            return list()

    async def get_proposals(self, agents: list[AgentDescription]) -> list[ContractNetProposal]:
        sent: set[str] = set()
        received: set[str] = set()
        result: list[ContractNetProposal] = list()

        request = ContractNetRequest(task=self.task)
        for agent in agents:
            sent.add(agent.id)
            thread = await self.context.fork_thread()
            await (thread.request_proposal(agent.id).with_content(request))
            await thread.close()
        started = time.time()

        deadline = started + self.time_to_wait_for_proposals
        while len(received) < len(sent) and time.time() < deadline:
            response = await self.receive(MessageTemplate(thread_id=self.context.thread_id),
                                          max(0.1, deadline - time.time()))
            if response:
                received.add(str(response.sender))
                if response.performative == consts.PROPOSE:
                    result.append(ContractNetProposal.model_validate_json(response.content))

        return result

    async def extract_winner_and_notify_losers(self, proposals: list[ContractNetProposal]) -> ContractNetProposal:
        proposals.sort(key=lambda x: x.estimate)
        winner = proposals[0]

        for agent in proposals[1:]:
            await self.context.refuse(agent.author).with_content("")

        return winner

    async def get_result(self, proposal: ContractNetProposal) -> Optional[MessageTemplate]:
        thread = await self.context.fork_thread()

        await thread.accept(proposal.author).with_content(proposal)
        reply = await self.receive(MessageTemplate(thread_id=thread.thread_id), 20)

        await thread.close()
        return reply
