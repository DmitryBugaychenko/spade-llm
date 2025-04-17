# Echo agent

This is a simple demo with console input agent and echo agent.

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```
pip install -r ./requirements.txt
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/tg-example/config.yaml
```

Below is an example of a simple interaction.
```
INFO:echo:Started agent thread
INFO:console:Started agent thread
User input: Hello!
Assistant: Hello!.
User input: exit
INFO:console:Configured stopped.
INFO:console:Shutting down agent thread...
INFO:console:Exited agent thread.
INFO:echo:Configured stopped.
INFO:echo:Shutting down agent thread...
INFO:echo:Exited agent thread.
```
