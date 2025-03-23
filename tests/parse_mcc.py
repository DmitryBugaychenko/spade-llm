import os
from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import HumanMessage
from langchain_gigachat import GigaChatEmbeddings, GigaChat
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from urllib3 import request

from tests.models import MODELS


def main():
    loader = CSVLoader(file_path='../data/mcc_codes.csv',
                       metadata_columns=["MCC"],
                       csv_args={
                           'delimiter': ',',
                           'quotechar': '"'
                       })

    embeddings=GigaChatEmbeddings(
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False,
    )

    index = Chroma(embedding_function=embeddings)

    index.add_documents([x for x in loader.load() if len(x.page_content) > 100])

    max = MODELS.max

    lite = MODELS.lite

    prepare_request_prompt = PromptTemplate.from_template(
        """Твоя задача сформулировать описание MCC кода для транзакций, обозначающей траты людей 
        подпадающих под описание "{query}". Верни только само описание из не менее 20 слов.""")

    request = lite.invoke(prepare_request_prompt.invoke({"query":"люди, которые ходят в бары"}))
    print(request)
    result = index.similarity_search(query=request.content, k=1)
    print("\n".join([f"{x.metadata["MCC"]}:{x.page_content}" for x in result]))


if __name__ == "__main__":
    main()
