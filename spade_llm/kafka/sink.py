import asyncio
import threading
from typing import Optional

from confluent_kafka import Producer
from pydantic import BaseModel, Field

from spade_llm.core.api import MessageService, MessageSink, Message, MessageSource
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.threading import EventLoopThread


class KafkaConfig(BaseModel):
    bootstrap_servers: str = Field(
        serialization_alias="bootstrap.servers",
        description="Kafka bootstrap servers",
    )


class KafkaProducerConfig(KafkaConfig, extra="allow"):
    linger_ms: int = Field(
        serialization_alias="linger.ms",
        description="Time to wait for messages before sending them to Kafka",
    )
    client_id: str = Field(
        serialization_alias="client.id",
        description="Kafka client ID")
    agent_to_topic_mapping: dict[str, str] = Field(
        default={},
        description="Mapping from agent types to Kafka topics. By default, agent type is used as a topic",
    )

@configuration(KafkaProducerConfig)
class KafkaMessageSink(MessageSink, EventLoopThread, Configurable[KafkaProducerConfig]):
    _producer: Producer

    def configure(self):
        self._producer = Producer(self.config.model_dump(by_alias=True, exclude=["agent_to_topic_mapping"]))


    async def post_message(self, msg: Message):
        self.loop.call_soon_threadsafe(self._post_message_sync, msg)

    def _post_message_sync(self, msg):
        topic = self._config.agent_to_topic_mapping.get(msg.receiver.agent_type, msg.receiver.agent_type)
        self._producer.produce(
            topic=topic,
            key=msg.receiver.agent_id.encode(),
            value=msg.content.encode())

    def close(self):
        self.stop()
        self._producer.flush()