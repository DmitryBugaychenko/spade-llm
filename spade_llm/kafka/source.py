from confluent_kafka import Consumer
from pydantic import Field

from spade_llm.core.api import MessageSink
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.threading import EventLoopThread
from spade_llm.kafka.sink import KafkaConfig


class KafkaConsumerConfig(KafkaConfig, extra="allow"):
    group_id: str = Field(
        alias="group.id",
        description="Kafka consumer group ID"
    )
    exposed_agents: list[str] = Field(
        description="List of agents to listen for messages for")
    agent_to_topic_mapping: dict[str, str] = Field(
        default={},
        description="Mapping from agent types to Kafka topics. By default, agent type is used as a topic",
    )

@configuration(KafkaConsumerConfig)
class KafkaMessageSource(Configurable[KafkaConsumerConfig]):
    _consumer : Consumer

    def configure(self):
        self._consumer = Consumer(self.config.model_dump(by_alias=True, exclude=["agent_to_topic_mapping"]))

    def start(self, sink: MessageSink):
        self._consumer.subscribe(self.config.exposed_agents)