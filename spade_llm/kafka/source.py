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
        serialization_alias="group.id",
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
    _is_done: Event
    _is_started: Event

    def configure(self):
        logger.info("Initializing KafkaMessageSource")
        self._consumer = Consumer(self.config.model_dump(
            by_alias=True, exclude={"agent_to_topic_mapping", "exposed_agents"}))
        self._is_done = Event()
        self._is_started = Event()

    async def start(self, sink: MessageSink):
        logger.info("Starting KafkaMessageSource")
        self._running = True
        self._thread = threading.Thread(target=self.consume_messages, args=(sink,))
        self._thread.start()
        await self._is_started

    def assignment_callback(self, consumer, partitions):
        logger.info("Kafka partitions assigned: %s", partitions)
        self._is_started.set()

    def consume_messages(self, sink: MessageSink):
        try:
            topics = [self.config.agent_to_topic_mapping.get(agent, agent) for agent in self.config.exposed_agents]
            logger.info("Subscribing to topics %s", topics)
            self._consumer.subscribe(topics, on_assign=self.assignment_callback)

            while self._running:
                logger.debug("Polling Kafka")
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                elif msg.error():
                    logger.warning("Kafka error", exc_info=msg.error())
                else:
                    logger.debug("Received message %s", msg.value().decode())
                    try:
                        sink.post_message_sync(Message.model_validate_json(msg.value().decode()))
                    except Exception as e:
                        logger.exception("Failed to process message %s due to %s", msg, e)
        except Exception as e:
            logger.exception("Exception caught in KafkaMessageSource: %s", e)
        finally:
            try:
                self._consumer.close()
            finally:
                logger.info("Stopping KafkaMessageSource")
                self._is_done.set()

    def stop(self):
        logger.info("Stop requested for KafkaMessageSource")
        self._running = False

    async def await_start(self):
        await self._is_started

    async def join(self):
        await self._is_done
