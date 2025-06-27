# REACT agent with different trigonometry toolkits

This demo showcases an LLM agent equipped with specialized toolkits
for trigonometric calculations (sine, cosine, tangent, etc.) with different precision.

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

Configure API version. Follow [instruction](https://developers.sber.ru/docs/ru/gigachain/overview#langchain-gigachat)
to check them. And export your API version using command
```
export GIGACHAT_SCOPE=<your API version>
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/trigonometry/config.yaml
```

Below is an example of a simple interaction.
```
Using slower stringprep, consider compiling the faster cython/libidn one.
INFO:react:Started agent thread
INFO:console:Started agent thread
User input: синус 20
INFO:httpx:HTTP Request: POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:spade_llm.demo.platform.trigonometry.trigtoolkit:Calculating sin with 5 digits precision
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: Синус 20 градусов равен примерно 0.34202.
User input: Немецкий синус 20
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:spade_llm.demo.platform.trigonometry.trigtoolkit:Calculating sin with 14 digits precision
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: Результат немецкого синуса от 20 градусов равен 0.34202014332567
User input: Посчитай сумму немецкого синуса 20 и обычного синуса 15
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:spade_llm.demo.platform.trigonometry.trigtoolkit:Calculating sin with 14 digits precision
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
INFO:spade_llm.demo.platform.trigonometry.trigtoolkit:Calculating sin with 5 digits precision
INFO:httpx:HTTP Request: POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: Сумма равна $0.34202014332567 + 0.25882 = 0.60084014332567$.
exitUser input: 
INFO:console:Configured stopped.
INFO:console:Shutting down agent thread...
INFO:console:Exited agent thread.
INFO:react:Configured stopped.
INFO:react:Shutting down agent thread...
INFO:react:Exited agent thread.


```
