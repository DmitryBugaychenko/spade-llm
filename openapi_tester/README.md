# openapi_tester
Simple utility used to validate OpenAPI specification for LLM compatibility. Generates
tool description from specification and uses inlined examples to generate test prompts
and validate tool calls produced by LLM.

Following extension attributes for OpenAPI used:

* *x-AI-ready* - boolean flag on the level of operation used to indicate that this particular operation should be tested
* *x-examples* - list of examples on the level of operation. Each example has a natural language *prompt* and *args* expected for it.

See cards.yaml for example specification. Follow instructions below to see how the tester
works.

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
python main.py cards.yaml
```
