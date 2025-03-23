import os

from langchain_gigachat import GigaChat, GigaChatEmbeddings

class Models:
    lite = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        model="GigaChat-2",
        verify_ssl_certs=False,
    )

    pro = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        model="GigaChat-2-Pro",
        verify_ssl_certs=False,
    )

    max = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
    )

    pro_preview = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        base_url="https://gigachat-preview.devices.sberbank.ru/api/v1",
        model="GigaChat-Pro-preview",
        verify_ssl_certs=False,
    )

    embeddings=GigaChatEmbeddings(
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False,
    )

MODELS = Models()