import logging

from confluent_kafka import Producer
from pydantic import BaseModel, Field

from spade_llm.core.api import MessageSink, Message
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
        self._producer = Producer(self.config.model_dump(by_alias=True, exclude={"agent_to_topic_mapping"}))
        logger.info("Kafka producer initialized.")

    async def post_message(self, msg: Message):
        logger.debug("Sending message to Kafka %s", msg)
        self.loop.call_soon_threadsafe(self.post_message_sync, msg)

    def post_message_sync(self, msg):
        topic = self._config.agent_to_topic_mapping.get(msg.receiver.agent_type, msg.receiver.agent_type)
        try:
            logger.debug("Producing message on topic '%s'", topic)
            self._producer.produce(
                topic=topic,
                key=msg.receiver.agent_id.encode(),
                value=msg.model_dump_json().encode())
        except Exception as e:
            logger.warning("Failed to produce message on topic '%s' due to %s", topic, e)

    def close(self):
        logger.info("Closing Kafka producer...")
        self.stop()
        self.join_sync()
        self._producer.flush()
        logger.info("Kafka producer closed.")

    def start(self):
        EventLoopThread.start(self)
