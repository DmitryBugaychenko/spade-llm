from spade_llm.core.agent import Agent
from spade_llm.core.api import AgentId, AgentContext
from spade_llm.core.tools import SingleMessage
from spade_llm.core.behaviors import MessageTemplate, ContextBehaviour
from spade_llm.core.conf import configuration, Configurable
from asyncio import sleep as asleep

from pydantic import BaseModel, Field

from spade_llm.demo.platform.tg_example.bot import TelegramBot

"""Есть пример небольшого агента который считывает ввод клиента с консоли 
и прокидывает дальше в МАС https://github.com/DmitryBugaychenko/spade-llm/blob/main/spade_llm/agents/console.py,
 надо сделать такого же, но подключающегося к телеграм боту и пример запуска по аналогии 
 с https://github.com/DmitryBugaychenko/spade-llm/tree/main/spade_llm/demo/platform/echo.
  Оформить пулл реквестом. Для разработки можно использовать любых ИИ-помощников, 
  но желательно доступных в России без ВПН. 
  Если возникнут вопросы - уточнить у Дмитрия в ТГ @dmitrybugaychenko"""


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
        msg = "Assistant: " + self.extract_message(reply)
        await self.my_bot.bot_reply(reply_text=msg)
        print(msg)
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

        user_input = await self.my_bot.get_last_message()
        print(f"Processing input: {user_input}")
        if user_input is None:
            print('Input is none')
        elif user_input.lower() in self.config.stop_words:
            self.agent.stop()
            self.set_is_done()
        else:

            thread = await self.context.fork_thread()

            await (thread
                   .request(AgentId(agent_type=self.config.delegate_type, agent_id=self.config.delegate_id))
                   .with_content(user_input))

            handler = TgRequestBehavior(thread, self.config, user_input, self.my_bot)
            self.agent.add_behaviour(handler)

            await handler.join()
            await thread.close()


@configuration(TelegramAgentConf)
class Bot(Agent, Configurable[TelegramAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.add_behaviour(TgChatBehavior(self.default_context, self.config))
