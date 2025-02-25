import os
from unittest import TestCase

from langchain_gigachat import GigaChat


class ModelTestCase(TestCase):
    model = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        model="GigaChat-Pro",
        verify_ssl_certs=False,
    )