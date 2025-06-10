# REACT agent with Wiki tool (Based on openai ChatGPT-4o)

This is a simple demo of LLM agent with single tool for Wikipedia search.

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```
pip install -r ./requirements.txt
pip install langchain_openai
```

Configure OpenAI credentials. Follow [instruction](https://platform.openai.com/api-keys)
to get them. You can use any langchain compatible model capable of using tools, change demo/main.py to switch to.

```
export OPENAI_API_KEY=<your access token>
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/wiki_openai/config.yaml
```

Below is an example of a simple interaction.
```
Using slower stringprep, consider compiling the faster cython/libidn one.
INFO:react:Started agent thread
INFO:console:Started agent thread
User input: What is the tallest building in the world?
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Assistant: The tallest building in the world is the Burj Khalifa in Dubai, United Arab Emirates.
User input: exit
INFO:console:Configured stopped.
INFO:console:Shutting down agent thread...
INFO:console:Exited agent thread.
INFO:react:Configured stopped.
INFO:react:Shutting down agent thread...
INFO:react:Exited agent thread.

```
