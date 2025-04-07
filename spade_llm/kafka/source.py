import threading
import logging

from aiologic import Event
from confluent_kafka import Consumer
from pydantic import Field

from spade_llm.core.api import MessageSink, Message
from spade_llm.core.conf import configuration, Configurable
from spade_llm.kafka.sink import KafkaConfig

logger = logging.getLogger(__name__)

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
        logger.info("Initializing KafkaMessageSource")
        self._consumer = Consumer(self.config.model_dump(by_alias=True, exclude={"agent_to_topic_mapping"}))
        self._event = Event()

    def start(self, sink: MessageSink):
        logger.info("Starting KafkaMessageSource")
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
                    logger.warning("Kafka error", exc_info=msg.error())
                else:
                    logger.debug("Received message %s", msg.value().decode())
                    try:
                        sink.post_message(Message.model_validate_json(msg.value().decode()))
                    except Exception as e:
                        logger.exception("Failed to process message", exc_info=e)
        finally:
            try:
                self._consumer.close()
            finally:
                logger.info("Stopping KafkaMessageSource")
                self._event.set()

    def stop(self):
        logger.info("Stop requested for KafkaMessageSource")
        self._running = False

    async def join(self):
        await self._event
