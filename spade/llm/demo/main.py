import logging
import os

import spade
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_gigachat.chat_models import GigaChat
from spade import wait_until_finished

from spade.llm.demo.agents import PaymentAgent, SavingsAgent, ChatAgent, FinancialAgent


async def main():
    logging.basicConfig(level=logging.INFO)
    model = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        streaming=False,
        verify_ssl_certs=False,
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=4096, lang = "ru")
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    chat = ChatAgent(model, {}, "chat@localhost", "your_password")
    payment = PaymentAgent(balance=10000, jid="payment@localhost", password="your_password")
    savings = SavingsAgent(balance=100000, jid="savings@localhost", password="your_password",
                           payment_adders=str(payment.jid), chat_jid=str(chat.jid))

    payment_tool = payment.create_tool(chat)
    savings_tool = savings.create_tool(chat)

    finance = FinancialAgent(
        model = model,
        tools= {
            payment_tool.name: payment_tool,
            savings_tool.name: savings_tool},
        jid="finance@localhost",
        password="your_password"
    )

    financial_tool = finance.create_tool(chat)

    chat.tools = {
        wiki.name : wiki,
        financial_tool.name: financial_tool
    }

    await payment.start()
    await savings.start()
    await finance.start()
    await chat.start()
    print("Chat agent started. Check its console to see the output.")

    print("Starting the WEB interface for")
    chat.web.start(hostname="127.0.0.1", port="10000")

    await wait_until_finished(chat)

if __name__ == "__main__":
    spade.run(main())