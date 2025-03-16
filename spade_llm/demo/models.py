import os

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat import GigaChatEmbeddings

"""Simplest and fastest model, works well with simple chat, but might
experience problems in tool calling with multiple tools."""
LITE = GigaChat(
    credentials=os.environ['GIGA_CRED'],
    model="GigaChat-Lite",
    streaming=False,
    verify_ssl_certs=False,
)

"""Balanced model capable of structured generation and tool calling, but might
fail on complex reasoning tasks."""
PRO = GigaChat(
    credentials=os.environ['GIGA_CRED'],
    model="GigaChat-Pro",
    streaming=False,
    verify_ssl_certs=False,
)

"""The most powerful model, but also the slowest and the most expensive one."""
MAX = GigaChat(
    credentials=os.environ['GIGA_CRED'],
    model="GigaChat-Max",
    streaming=False,
    verify_ssl_certs=False,
)

"""Model used for vector storages"""
EMBEDDINGS = GigaChatEmbeddings(
    credentials=os.environ['GIGA_CRED'],
    verify_ssl_certs=False,
)