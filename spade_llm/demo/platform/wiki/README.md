# REACT agent with Wiki tool

This is a simple demo of LLM agent with single tool for Wikipedia search.

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

If you want to specify API version. Follow [instruction](https://developers.sber.ru/docs/ru/gigachain/overview#langchain-gigachat)
to check them. And export your API version using command
```
export GIGACHAT_SCOPE=<your API version>
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/wiki/config.yaml
```

Below is an example of a simple interaction. Note that on call to completions made when handling "Hello!" (one LLM call) and two calls when fetching Bertrand Russell info (first call to identify tool and second call to parse its output).
```
INFO:react:Started agent thread
INFO:console:Started agent thread
User input: Hello!
INFO:httpx:HTTP Request: POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: Hello! How may I assist you today?.
User input: Who is Bertrand Russell and what is he famous for?      
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: Bertrand Russell was a British philosopher, logician, mathematician, and public intellectual known for being one of the founders of analytic philosophy alongside Gottlob Frege and G.E. Moore. He made significant contributions to mathematics, particularly through his work on logic and set theory, and is most famous for co-authoring *Principia Mathematica* with Alfred North Whitehead. Russell's work laid foundational groundwork for modern mathematical logic and influenced many fields of philosophy. Additionally, he played a crucial role in opposing imperialism and advocating for peace, earning him the Nobel Prize in Literature in 1950..
User input: exit
INFO:console:Configured stopped.
INFO:console:Shutting down agent thread...
INFO:console:Exited agent thread.
INFO:react:Configured stopped.
INFO:react:Shutting down agent thread...
INFO:react:Exited agent thread.
```
