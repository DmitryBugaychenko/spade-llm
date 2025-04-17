from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio


class TelegramBot:
    def __init__(self, TOKEN):
        self.TOKEN = TOKEN

        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.bot = Bot(token=self.TOKEN)
        self.last_message = None
        self._reg_handler()

    def _reg_handler(self):
        @self.dp.message()
        async def handle_message(message: types.Message):
            self.last_message = message
            await self.bot_stop()

    async def bot_start(self):
        await self.dp.start_polling(self.bot)
        return self.last_message.text

    async def bot_stop(self):
        await self.dp.stop_polling()

    async def bot_reply(self, reply_text='No_reply'):
        await self.last_message.answer(f"{reply_text}")
