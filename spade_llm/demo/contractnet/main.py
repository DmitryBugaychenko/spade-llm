import logging
import os
import sys
from asyncio import sleep as asleep
from multiprocessing import Process

import spade
from aioconsole import ainput
from langchain_gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.cli import create_cli

from spade_llm.contractnet import ContractNetInitiatorBehavior
from spade_llm.demo.hierarchy.agents import ChatAgent
from spade_llm.demo.contractnet.agents import MccExpertAgent, PeriodExpertAgent, SpendingProfileAgent, \
    TransactionsAgent, UsersList
from spade_llm.discovery import DirectoryFacilitatorAgent

PERIOD_AGENT = "period@localhost"
SPENDINGS_AGENT = "spendings@localhost"
TRANSACTIONS_AGENT = "transactions@localhost"
MCC_EXPERT = "mccexpert@localhost"
DF_ADDRESS = "df@localhost"

def start_xmmp():
    sys.argv = [sys.argv[0], "run", "--memory"]
    cli = create_cli()
    sys.exit(cli())

class ChatAgent(Agent):

    @staticmethod
    def cformat(msg:str) -> str:
        """Utility for getting more visible messages in console"""
        return "\033[1m\033[92m{}\033[00m\033[00m".format(msg)

    class ChatBehaviour(CyclicBehaviour):

        async def run(self) -> None:
            # Small hack to let all the logs to be printed before prompting
            await asleep(0.5)
            user_input: str = await ainput(ChatAgent.cformat("Какой сегмент собрать: "))

            if user_input.lower() in {"пока", "bye"}:
                await self.agent.stop()
                return None

            request = ContractNetInitiatorBehavior(
                task=user_input,
                df_address=DF_ADDRESS
            )

            self.agent.add_behaviour(request, request.construct_template())
            await request.join(20)

            # Small hack to let all the logs to be printed before prompting
            await asleep(1)
            if request.is_successful:
                result = UsersList.model_validate_json(request.result.body)
                head = ",".join([str(id) for id in result.ids[0:min(20, len(result.ids))]])
                print(ChatAgent.cformat("Получен сегмент: ") + f"Размер {len(result.ids)} первые 20 ids: {head}")
            else:
                print(ChatAgent.cformat("Не удалось собрать сегмент."))

    async def setup(self) -> None:
        self.add_behaviour(self.ChatBehaviour())


async def main():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("spade_llm").setLevel(logging.INFO)

    data_path = sys.argv[1]
    print("Looking for database at " + data_path)

    xmmp = Process(target=start_xmmp, daemon=True)
    xmmp.start()

    model = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        model="GigaChat-Max",
        verify_ssl_certs=False,
    )

    embeddings=GigaChatEmbeddings(
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False,
    )

    expert = MccExpertAgent(
        embeddings=embeddings,
        model=model,
        data_path=data_path,
        jid=MCC_EXPERT,
        password="pwd"
    )

    period = PeriodExpertAgent(
        model=model,
        date="2012-04-01",
        jid=PERIOD_AGENT,
        password="pwd"
    )

    df = DirectoryFacilitatorAgent(
        DF_ADDRESS,
        "pwd",
        embeddings=embeddings)

    spendings = SpendingProfileAgent(
        data_path=data_path,
        mcc_expert=MCC_EXPERT,
        df_address=DF_ADDRESS,
        model=model,
        jid=SPENDINGS_AGENT,
        password="pwd"
    )

    transactions = TransactionsAgent(
        data_path=data_path,
        mcc_expert=MCC_EXPERT,
        period_expert=PERIOD_AGENT,
        df_address=DF_ADDRESS,
        model=model,
        jid=TRANSACTIONS_AGENT,
        password="pwd"
    )

    chat = ChatAgent("chat@localhost", "pwd")

    await df.start()
    await expert.start()
    await period.start()
    await spendings.start()
    await transactions.start()
    await chat.start()

    await wait_until_finished([chat])
    await df.stop()
    await expert.stop()
    await period.stop()
    await spendings.stop()
    await transactions.stop()

    await wait_until_finished([df,expert,period,spendings,transactions])

    xmmp.terminate()
    xmmp.join()
    print("XMMP Server closed")

if __name__ == "__main__":
    spade.run(main())