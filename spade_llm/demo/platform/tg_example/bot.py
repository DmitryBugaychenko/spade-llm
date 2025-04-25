from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from aiogram.filters import Command
import asyncio
import logging.config


class Form(StatesGroup):
    waiting_for_input = State()


class TelegramBot:
    def __init__(self, TOKEN: str):
        self.TOKEN = TOKEN
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.bot = Bot(token=self.TOKEN)
        self.message_queue = asyncio.Queue()
        logging.config.fileConfig('./spade_llm/demo/platform/tg_example/logging_config.ini')
        self._reg_handler()

    def _reg_handler(self):
        @self.dp.message(Command("start"), F.text)
        async def handle_start(message: types.Message, state: FSMContext):
            await message.reply("Бот готов к работе. Отправьте зарос.")
            await state.set_state(Form.waiting_for_input)

        @self.dp.message(Form.waiting_for_input, F.text)
        async def handle_message(message: types.Message):
            await self.message_queue.put(message)

    async def get_last_message(self):
        await self.wait_for_input()
        if not self.message_queue.empty():
            return await self.message_queue.get()
        return None

    async def wait_for_input(self):
        while True:
            if not self.message_queue.empty():
                return
            await self.get_updates()

    async def get_updates(self):
        task = asyncio.create_task(self.stop_polling_after(1))
        await self.dp.start_polling(self.bot)
        await task

    async def stop_polling_after(self, timeout: float):
        await asyncio.sleep(timeout)
        await self.bot_stop()

    async def bot_stop(self):
        await self.dp.stop_polling()

    async def bot_start(self):
        await self.dp.start_polling(self.bot)

    async def bot_reply(self, chat_id: int, reply_text='Empty reply'):
        await self.bot.send_message(chat_id=chat_id, text=reply_text)
