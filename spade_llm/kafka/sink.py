import asyncio
import threading
import logging
from typing import Optional

from confluent_kafka import Producer
from pydantic import BaseModel, Field

from spade_llm.core.api import MessageService, MessageSink, Message, MessageSource
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.threading import EventLoopThread

logger = logging.getLogger(__name__)

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
        logger.info("Initializing Kafka producer...")
        self._producer = Producer(self.config.model_dump(by_alias=True, exclude=["agent_to_topic_mapping"]))
        logger.info("Kafka producer initialized.")

    async def post_message(self, msg: Message):
        logger.debug("Sending message to Kafka", extra={'msg': msg})
        self.loop.call_soon_threadsafe(self._post_message_sync, msg)

    def _post_message_sync(self, msg):
        topic = self._config.agent_to_topic_mapping.get(msg.receiver.agent_type, msg.receiver.agent_type)
        try:
            self._producer.produce(
                topic=topic,
                key=msg.receiver.agent_id.encode(),
                value=msg.content.encode())
        except Exception as e:
            logger.warning("Failed to produce message on topic '%s'", topic, exc_info=e)

    def close(self):
        logger.info("Closing Kafka producer...")
        self.stop()
        self._producer.flush()
        logger.info("Kafka producer closed.")
