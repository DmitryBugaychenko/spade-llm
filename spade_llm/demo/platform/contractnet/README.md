# Contract Net Demo (spade_llm platform)

This demo shows how agent links might be established in dynamics using two concepts:
1. [Discovery Service](https://ieeexplore.ieee.org/document/4624020) (DF) - a special agent used to search for other agents. It allows others to register in a directory providing description and list of tasks they can handle with examples. Then it allows to search for registered agents. In contrast with traditional DF based on full-text matching and ontologies this one uses vector store and embedings.
2. [Contract Net Protocol](https://en.wikipedia.org/wiki/Contract_Net_Protocol) - a protocol used in open distributed systems to efficiently share tasks between agents. Agent who needs to solve a task (Initiator) first finds others who can help (Responders), ask them for proposals, chose the best proposal and delegate the task to its author.

In the demo we simulate a campaign system where user formulates a description of client segment and system dynamically selects a strategy for collecting it. Segments are collected based on clients [financial transactions](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data) according to two possible strategies:
1. Based on total spending - using aggregated transactions per category identified by [Merchant Category Code](https://en.wikipedia.org/wiki/Merchant_category_code) (MCC). This strategy can handle general requests like "people visiting bars".
2. Based on individual transactions - using detailed informations this strategy can handle segments like "clients who went to a bar last month", but it is more expensive.

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```shell
pip install -r ./requirements.txt
```

Download data from [financial transactions dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data) and place ```transactions_data.csv``` into ```./data``` folder.

Create a SQLite database for the demo%

```shell
python ./spade_llm/demo/platform/contractnet/create_database.py ./data
```

Configure GigaChat credentials. Follow [instruction](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
to get them. You can use any langchain compatible model capable of using tools, change demo/main.py to switch to.
```
export GIGA_CRED=<your access token>
```
Configure API version. Follow [instruction](https://developers.sber.ru/docs/ru/gigachain/overview#langchain-gigachat)
to check them. And export your API version using command
```
export GIGACHAT_SCOPE=<your API version>
```
Run the demo

```shell
python -m spade_llm.boot ./spade_llm/demo/platform/contractnet/config.yaml
```

This demo uses MCC codes and agents description in russian, tested on tasks "Люди, которые ходят в бары" и "Люди, которые ходили в бар в прошлом месяце". 
