# Telegram Bot Agent

This is a simple demo with Telegram Bot input agent and echo agent.

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```
pip install -r ./requirements.txt
```


# Configure TG bot token

You need token from BotFather to run bot. Follow [instruction](https://core.telegram.org/bots/features#creating-a-new-bot)
to get it.

After that, export this token to the local environment.

## For macOS/linux:
```
export BOT_TOKEN=<your Tg_bot token from BotFather>
```
## For windows cmd
```
set BOT_TOKEN=<your Tg_bot token from BotFather>
```

## For windows PowerShell
```
$env:BOT_TOKEN='<your Tg_bot token from BotFather>'
```

# Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/tg_example/config.yaml
```


