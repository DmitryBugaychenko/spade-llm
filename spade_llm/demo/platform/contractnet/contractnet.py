import logging
import time
from abc import ABC, abstractmethod
from typing import Optional
from spade_llm import consts
from pydantic import BaseModel, Field
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour, \
    MessageCollectorBehavior
from spade_llm.core.api import AgentContext
from spade_llm.demo.platform.contractnet.discovery import AgentDescription, AgentSearchResponse, AgentSearchRequest, \
    DF_ADDRESS
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

    class HandleAcceptBehavior(ContextBehaviour):
        def __init__(self, context:AgentContext,responder: ContractNetResponder, proposal):
            super().__init__(context)
            self._responder = responder
            self.proposal = proposal

        async def step(self) -> None:
            msg = await self.receive(MessageTemplate(thread_id=self.context.thread_id), timeout=10)
            if msg and msg.performative == consts.ACCEPT:
                result = await(self._responder.execute(self.proposal, msg))
                await self.context.reply_with_inform(msg).with_content(result)
            elif msg and msg.performative == consts.REFUSE:
                pass
            self.set_is_done()

    async def step(self) -> None:
        if self.message:
            msg = self.message
            parsed_request = ContractNetRequest.model_validate_json(msg.content)
            proposal = await self._responder.estimate(parsed_request, msg)
            if proposal:
                await self.context.reply_with_propose(msg).with_content(proposal)
                handler = self.HandleAcceptBehavior(self.context,self._responder, proposal)
                self.agent.add_behaviour(handler)
                await handler.join()

            else:
                await self.context.reply_with_refuse(msg).with_content("")

class ContractNetInitiatorBehavior(ContextBehaviour):
    task: str
    proposal: Optional[ContractNetProposal] = None
    result: Optional[MessageTemplate]
    time_to_wait_for_proposals: float
    _started_at: float

    def __init__(self,
                 task: str,
                 context: AgentContext,
                 time_to_wait_for_proposals: float = 10,
                 time_to_wait_for_result: float = 60):
        super().__init__(context)
        self.time_to_wait_for_proposals = time_to_wait_for_proposals
        self.time_to_wait_for_result = time_to_wait_for_result
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
                    if self.result:
                        to_log = str(self.result.content)
                        logger.info("Result for task '%s' is %s", self.task, to_log[0:min(256, len(to_log))])
                    else:
                        logger.error("Failed to get result for task '%s'", self.task)
                else:
                    logger.error("Failed to select best proposal for '%s'", self.task)
            else:
                logger.error("Failed to get any proposals for '%s' in time", self.task)
        else:
            logger.error("Failed to find any agents for '%s'", self.task)
        self.set_is_done()

    async def find_agents(self, task: str) -> list[AgentDescription]:
        await (self.context
               .request(DF_ADDRESS)
               .with_content(AgentSearchRequest(task=task, top_k=10)))

        search = await self.receive(MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
                                    timeout=10)

        if search:
            parsed = AgentSearchResponse.model_validate_json(search.content)
            return parsed.agents
        else:
            logger.error(f"No search recieved in {10} seconds")
            return list()

    async def get_proposals(self, agents: list[AgentDescription]) -> list[ContractNetProposal]:
        # Add collector behavior before sending requests to avoid race conditions
        collector = MessageCollectorBehavior(
            MessageTemplate(performative=consts.PROPOSE, thread_id=self.context.thread_id),
            expected_count=len(agents),
        )
        self.agent.add_behaviour(collector)

        # Send all requests for proposals
        request = ContractNetRequest(task=self.task)
        for agent in agents:
            await (self.context.request_proposal(agent.id).with_content(request))

        # Now join the collector with timeout
        try:
            await asyncio.wait_for(collector.join(), self.time_to_wait_for_proposals)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for proposals: collected %d of %d",
                len(collector.messages), len(agents),
            )
            self.agent.remove_behaviour(collector)

        result: list[ContractNetProposal] = []
        for msg in collector.messages:
            result.append(ContractNetProposal.model_validate_json(msg.content))

        return result

    async def extract_winner_and_notify_losers(self, proposals: list[ContractNetProposal]) -> ContractNetProposal:
        proposals.sort(key=lambda x: x.estimate)
        winner = proposals[0]

        for agent in proposals[1:]:
            await self.context.refuse(agent.author).with_content("")

        return winner

    async def get_result(self, proposal: ContractNetProposal) -> Optional[MessageTemplate]:
        await self.context.accept(proposal.author).with_content(proposal)
        reply = await self.receive(MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id,
                                                   validator=MessageTemplate.from_agent(proposal.author)),
                                   self.time_to_wait_for_result)

        return reply
