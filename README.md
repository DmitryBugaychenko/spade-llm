# spade-llm
Experimental integration of multiagent framework [SPADE](https://github.com/javipalanca/spade) for LLM agents with LangChain, RAG and other perks.

So far following functions are available:
1. REACT-like agent behaviour for SPADE agents
2. Directory Facilitator agent for dynamic agents discovery based on vector database and LLM embeddings
3. Contract Net Protocol implemented as agent behavior

Examples are provided based on GigaChat model. In order to run them your need to get GigaChat credentials using [instruction](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart) and then store access token in environment variable:
```
export GIGA_CRED=<your access token>
```

If you want to try demo with other models, adjust [models.py](https://github.com/DmitryBugaychenko/spade-llm/blob/main/spade_llm/demo/models.py) file.

Following examples are provided:
1. [Hierarchical organization](https://github.com/DmitryBugaychenko/spade-llm/tree/main/spade_llm/demo/hierarchy) - demonstrate a hierarchy of REACT and coded agents delegating tasks top down and escalating problems bottom up.
2. [Contract Net](https://github.com/DmitryBugaychenko/spade-llm/tree/main/spade_llm/demo/contractnet) - an example of "open system" where the user-facing agent does not know others upfront, uses discovery service (directory facilitator) to find them and Contract Net to allocate task efficiently. 