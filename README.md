# spade-llm
Experimental integration of multiagent framework SPADE for LLM agents with LangChain, RAG and other perks.

So far a simple demo is available with four agents:
* Chat agent - general assistant capable of looking up information in Wikipedia and delegating tasks to other assistants, uses a LLM
* Financial assistant - can handle financial task based on mocked payment account and savings account, uses a LLM
* Payment agent - can handle payments if there are enough funds on the payment account, does not use LLM
* Savings agent - manages user savings and is capable of replenish payment account, but requires user confirmation passing message back to chat agent, does not use LLM

Install requirements
```
pip install -r ./requirements.txt
```

Configure GigaChat credentials. Follow [instruction](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
to get them. You can use any langchain compatible model capable of using tools, change demo/main.py to switch to.
```
export GIGA_CRED=<your access token>
```

```
pip install -r ./requirements.txt
python -m spade_llm.demo.main
```

Below is an example of a simple scenario:
1. User asks for population of Moscow and assistant uses Wikipedia to lookup an answer
2. User asks to pay a bill and there are enough money on the payment account, request goes to financial assistant and then to payment agent
3. User asks to pay a bill and there are no enough money on the payment account, payment agent fails the operation and financial assistant asks saving agent to solve the problem, the later gets user confirmation and replenish payment account, then financial assistant goes to payment agent again and at this time its all done
4. Same as above, but this time user refuses to withdraw from savings

Logs are omitted to highlight dialog messages and tool usage decisions.
```
User: What is the population of Moscow?
...
INFO:spade_llm.behaviours:Invoking tool wikipedia
...
GigaChat: The population of Moscow is over 13 million residents within the city limits.
...
User: Pay the bill for 5000 rubles
...
INFO:spade_llm.behaviours:Invoking tool finance_help
...
INFO:spade_llm.behaviours:Invoking tool payment_service
...
GigaChat: The bill for 5000 rubles has been successfully paid.
...
User: Pay the bill for 10000 rubles
...
INFO:spade_llm.behaviours:Invoking tool finance_help
...
INFO:spade_llm.behaviours:Invoking tool payment_service
...
INFO:spade_llm.behaviours:Invoking tool savings_service
...
Confirm operation "Transfer 5000 rubles from savings to payment account" [y/n]: y
...
INFO:spade_llm.behaviours:Invoking tool payment_service
...
GigaChat: The bill has been paid successfully.
...
User: Pay the bill for 5000 rubles
...
INFO:spade_llm.behaviours:Invoking tool finance_help
...
INFO:spade_llm.behaviours:Invoking tool payment_service
...
INFO:spade_llm.behaviours:Invoking tool savings_service
...
Confirm operation "Transfer 5000 rubles from savings to payment account" [y/n]: n
...
GigaChat: I'm sorry, but your payment could not be processed. Please try again later.
...
User: bye
```