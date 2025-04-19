from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from aiogram.filters import Command
import asyncio


class Form(StatesGroup):
    """Состояния бота для FSM."""
    waiting_for_input = State()


class TelegramBot:
    def __init__(self, TOKEN):
        """Инициализирует бота с указанным токеном и настраивает базовые компоненты."""
        self.TOKEN = TOKEN
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.bot = Bot(token=self.TOKEN)
        self.message_queue = asyncio.Queue()
        self.current_chat_id = None
        self._reg_handler()

    def _reg_handler(self):
        """Регистрирует обработчики команд и сообщений."""

        @self.dp.message(Command("start"), F.text)
        async def handle_start(message: types.Message, state: FSMContext):
            """Обрабатывает команду /start и переводит бота в режим ожидания ввода."""
            self.current_chat_id = message.chat.id
            await message.reply("Бот готов к работе. Отправьте зарос.")
            await state.set_state(Form.waiting_for_input)

        @self.dp.message(Form.waiting_for_input, F.text)
        async def handle_message(message: types.Message):
            """Обрабатывает текстовые сообщения пользователя в состоянии ожидания."""
            self.last_message = message
            await self.message_queue.put(message.text)
            # Управиться без bot_stop
            await self.bot_stop()

    async def bot_start(self):
        """Запускает бота в режиме опроса серверов Telegram."""
        await self.dp.start_polling(self.bot)

    async def get_last_message(self):
        """Возвращает последнее полученное сообщение из очереди."""
        await self.bot_start()
        return await self.message_queue.get()

    async def bot_stop(self):
        """Останавливает процесс опроса серверов Telegram."""
        await self.dp.stop_polling()

    async def bot_reply(self, reply_text='No_reply'):
        """Отправляет ответное сообщение в текущий чат."""
        if self.current_chat_id:
            await self.bot.send_message(self.current_chat_id, reply_text)
