import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field
from spade.behaviour import FSMBehaviour, State, OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates

logger = logging.getLogger(__name__)

class ContractNetRequest(BaseModel):
    task: str = Field(description="Description of the task to create proposal for")

class ContractNetProposal(BaseModel):
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