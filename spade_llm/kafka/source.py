import threading

from aiologic import Event
from confluent_kafka import Consumer
from pydantic import Field

from spade_llm.core.api import MessageSink, Message
from spade_llm.core.conf import configuration, Configurable
from spade_llm.kafka.sink import KafkaConfig


class KafkaConsumerConfig(KafkaConfig, extra="allow"):
    group_id: str = Field(
        alias="group.id",
        description="Kafka consumer group ID"
    )
    exposed_agents: list[str] = Field(
        description="List of agents to listen for messages for"
    )
    agent_to_topic_mapping: dict[str, str] = Field(
        default={},
        description="Mapping from agent types to Kafka topics. By default, agent type is used as a topic",
    )


@configuration(KafkaConsumerConfig)
class KafkaMessageSource(Configurable[KafkaConsumerConfig]):
    _consumer: Consumer
    _running: bool = False
    _thread: threading.Thread
    _event: Event

    def configure(self):
        self._consumer = Consumer(self.config.model_dump(by_alias=True, exclude={"agent_to_topic_mapping"}))
        self._event = Event()

    def start(self, sink: MessageSink):
        self._running = True
        self._thread = threading.Thread(target=self.consume_messages, args=(sink,))
        self._thread.start()

    def consume_messages(self, sink: MessageSink):
        try:
            topics = [self.config.agent_to_topic_mapping.get(agent, agent) for agent in self.config.exposed_agents]
            self._consumer.subscribe(topics)

            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                elif msg.error():
                    print(f"Kafka error: {msg.error()}")
                else:
                    sink.post_message(Message.model_validate_json(msg.value().decode()))
        finally:
            try:
                self._consumer.close()
            finally:
                self._event.set()

    def stop(self):
        self._running = False

    async def join(self):
        await self._event
