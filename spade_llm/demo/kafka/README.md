# Echo agent with Kafka connection

This is a simple demo with console input agent and echo agent communication with each other using Kafka.

Example is configured to use Kafka at localhost:9092, to setup local Kafka broker follow [quick start instruction](https://kafka.apache.org/quickstart) and create two topics (step 3 in quick start) named "echo" and "console".

To run the demo open a terminal at the root of the GitHub project and follow the steps below.

Install requirements
```
pip install -r ./requirements.txt
```

Run the demo

```
python -m spade_llm.boot ./spade_llm/demo/platform/kafka/config.yaml
```

Below is an example of a simple interaction.
```
INFO:spade_llm.kafka.source:Initializing KafkaMessageSource
INFO:spade_llm.kafka.sink:Initializing Kafka producer...
INFO:spade_llm.kafka.sink:Kafka producer initialized.
INFO:echo:Started agent thread
INFO:console:Started agent thread
INFO:spade_llm.kafka.source:Starting KafkaMessageSource
INFO:spade_llm.kafka.source:Subscribing to topics ['echo', 'console']
INFO:spade_llm.kafka.source:Kafka partitions assigned: [TopicPartition{topic=console,partition=0,offset=-1001,leader_epoch=None,error=None}, TopicPartition{topic=echo,partition=0,offset=-1001,leader_epoch=None,error=None}]
User input: Hello!
Assistant: Hello!.
User input: exit
INFO:console:Configured stopped.
INFO:console:Shutting down agent thread...
INFO:console:Exited agent thread.
INFO:echo:Configured stopped.
INFO:echo:Shutting down agent thread...
INFO:echo:Exited agent thread.
INFO:spade_llm.kafka.source:Stop requested for KafkaMessageSource
INFO:spade_llm.kafka.source:Stopping KafkaMessageSource
INFO:spade_llm.kafka.sink:Closing Kafka producer...
INFO:Configured:Shutting down thread...
INFO:Configured:Exited thread.
INFO:spade_llm.kafka.sink:Kafka producer closed.
```
