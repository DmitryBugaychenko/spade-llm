from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import configuration, Configurable, EmptyConfig

"""Есть пример небольшого агента который считывает ввод клиента с консоли 
и прокидывает дальше в МАС https://github.com/DmitryBugaychenko/spade-llm/blob/main/spade_llm/agents/console.py,
 надо сделать такого же, но подключающегося к телеграм боту и пример запуска по аналогии 
 с https://github.com/DmitryBugaychenko/spade-llm/tree/main/spade_llm/demo/platform/echo.
  Оформить пулл реквестом. Для разработки можно использовать любых ИИ-помощников, 
  но желательно доступных в России без ВПН. 
  Если возникнут вопросы - уточнить у Дмитрия в ТГ @dmitrybugaychenko"""


@configuration(EmptyConfig)
class TelegramAgentHandler(Agent, Configurable[EmptyConfig]):
    class TelegramBehaviour(MessageHandlingBehavior):
        async def step(self):
            await (self.context
                   .reply_with_inform(self.message)
                   .with_content(self.message.content))

    def setup(self):
        self.add_behaviour(self.TelegramBehaviour(MessageTemplate()))
