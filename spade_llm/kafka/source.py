from confluent_kafka import Consumer
from pydantic import Field

from spade_llm.core.api import MessageSink
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.threading import EventLoopThread
from spade_llm.kafka.sink import KafkaConfig
import threading


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

    def configure(self):
        self._consumer = Consumer(self.config.model_dump(by_alias=True, exclude=["agent_to_topic_mapping"]))

    def start(self, sink: MessageSink):
        # Create a new thread to handle consuming messages
        def consume_messages():
            try:
                # Subscribe to topics based on exposed agents
                topics = [self.config.agent_to_topic_mapping.get(agent, agent) for agent in self.config.exposed_agents]
                self._consumer.subscribe(topics)
                
                while True:
                    msg = self._consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        continue
                    elif msg.error():
                        print(f"Kafka error: {msg.error()}")
                    else:
                        # Pass the received message to the sink
                        sink.process_message(msg.value())
            
            except Exception as e:
                print(f"Error occurred during consumption: {e}")
        
        # Start the consumer thread
        consumer_thread = threading.Thread(target=consume_messages)
        consumer_thread.start()
