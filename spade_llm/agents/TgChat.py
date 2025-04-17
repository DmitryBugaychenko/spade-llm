from asyncio import sleep as asleep
import asyncio
from typing import Optional, Dict

from pydantic import BaseModel, Field
from aiogram.filters import Command

from spade_llm import consts
from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.behaviors import ContextBehaviour, MessageTemplate
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.tools import SingleMessage

from spade_llm.agents.bot import TelegramBot


class TelegramAgentConf(BaseModel):
    bot_token: str = Field(description="Telegram bot token")
    allowed_chat_ids: list[int] = Field(default_factory=list, description="List of allowed chat IDs")
    delegate_type: str = Field(description="Type of the delegate agent to send messages to.")
    delegate_id: str = Field(default="default", description="ID of the delegate agent to send messages to.")
    stop_words: set[str] = Field(default={"exit", "stop"}, description="Stop words used to shut down the agent")
    timeout: float = Field(default=60.0, description="Timeout for receiving messages.")


class TgRequestBehavior(ContextBehaviour):
    # надо чекнуть что bot класса telegramm bot
    def __init__(self, context: AgentContext, config: TelegramAgentConf, request: str, bot):
        super().__init__(context)
        self.config = config
        self.request = request
        self.my_bot = bot

    async def step(self):
        reply = await self.receive(MessageTemplate(thread_id=self.context.thread_id), self.config.timeout)

        if not reply:
            print(f"No response received in {self.config.timeout} seconds")
            return

        # if reply.performative == consts.REQUEST_APPROVAL:
        #     action = self.extract_message(reply)
        #     response = 'n'
        #     # response = await ainput(f"Do you want to approve the action '{action}'? [y/n] ")
        #     if response.lower() == "y":
        #         await (self.context.reply_with_acknowledge(reply).with_content(""))
        #     else:
        #         await (self.context.reply_with_refuse(reply).with_content(""))
        #     return
        msg = "Assistant: " + self.extract_message(reply)
        await self.my_bot.bot_reply(reply_text=msg)
        print(msg)
        # if reply.performative == consts.REQUEST:
        #     # response = await ainput(self.config.prompt)
        #     response = 'response'
        #     await (self.context.reply_with_inform(reply).with_content(response))
        #     return

        self.set_is_done()

    def extract_message(self, reply):
        if reply.content.startswith("{"):
            try:
                return SingleMessage.model_validate_json(reply.content).message
            except Exception as e:
                # Do nothing, just return the content
                pass
        return reply.content


class TgChatBehavior(ContextBehaviour):
    def __init__(self, context: AgentContext, config: TelegramAgentConf):
        super().__init__(context)
        self.config = config
        self.my_bot = TelegramBot(self.config.bot_token)

    async def step(self):
        # Small hack to let all the logs to be printed before prompting
        await asleep(0.5)
        # Вот тут нужно получать user_input
        user_input = await self.my_bot.bot_start()
        print(f"Processing input: {user_input}")
        if user_input is None:
            print('None')
        elif user_input.lower() in self.config.stop_words:
            self.agent.stop()
            self.set_is_done()
        else:

            # нужно перенаправить вывод в self.my_bot.bot_reply(text)
            thread = await self.context.fork_thread()

            await (thread
                   .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                   .with_content(user_input))

            handler = TgRequestBehavior(thread, self.config, user_input, self.my_bot)
            self.agent.add_behaviour(handler)

            await handler.join()
            await thread.close()


@configuration(TelegramAgentConf)
class TelegramAgent(Agent, Configurable[TelegramAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def setup(self):
        self.add_behaviour(TgChatBehavior(self.default_context, self.config))
