# Agent hierarchy demo

A simple demo of hierarchy with four agents:
* Chat agent - general assistant capable of looking up information in Wikipedia and delegating tasks to other assistants, uses a LLM
    * Financial assistant - can handle financial task based on mocked payment account and savings account, uses a LLM
        * Payment agent - can handle payments if there are enough funds on the payment account, does not use LLM
        * Savings agent - manages user savings and is capable of replenish payment account, but requires user confirmation passing message back to chat agent, does not use LLM
    * Wikipedia search - a build on langchain tool

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```
pip install -r ./requirements.txt
```

Configure GigaChat credentials. Follow [instruction](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
to get them. You can use any langchain compatible model capable of using tools, change demo/main.py to switch to.
```
export GIGA_CRED=<your access token>
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/hierarchy/config.yaml
```

Below is an example of a simple scenario:
1. User asks for population of Moscow and assistant uses Wikipedia to lookup an answer
2. User asks to pay a bill and there are enough money on the payment account, request goes to financial assistant and then to payment agent
3. User asks to pay a bill and there are no enough money on the payment account, payment agent fails the operation and financial assistant asks saving agent to solve the problem, the later gets user confirmation and replenish payment account, then financial assistant goes to payment agent again and at this time its all done
4. User asks to replenish current account without specifying amount, assistant asks for clarification and proceed to operation. 

Logs are omitted to highlight dialog messages and agent decisions.
```
User input: Hello!
...
Assistant: Hello! How may I assist you today?.
User input: What is the population of Moscow?
...
Assistant: The population of Moscow is over 13 million residents within the city limits, over 19.1 million residents in the urban area, and over 21.5 million residents in its metropolitan area..
User input: Pay the bill for 1000
...
INFO:payments.PaymentBehaviour:No balance set, using default: 10000
INFO:payments.PaymentBehaviour:Payment done, new balance: 9000
...
Assistant: Your bill has been paid successfully. Your current balance is now 9000 rubles..
User input: Pay the bill for 10000
...
WARNING:payments.PaymentBehaviour:Not enough balance to pay: 10000
...
INFO:savings.ReplenishBehaviour:No balance set, using default: 100000
...
Do you want to approve the action 'Replenish current account from savings for 1000'? [y/n] y
...
Assistant: Your bill for 10000 has been paid successfully. Your new balance is 0 rubles..
User input: Replenish my current account
...
Assistant: Please specify the amount you want to replenish your current account with..
User input: 10000
Do you want to approve the action 'Replenish current account from savings for 10000'? [y/n]y
...
INFO:savings.ReplenishBehaviour:Replenish done, new balance at savings account: 89000
INFO:payments.ReplenishBehaviour:Replenish done, new balance: 10000
...
Assistant: Your current account has been successfully replenished by 10,000 units..
User input: exit
```
