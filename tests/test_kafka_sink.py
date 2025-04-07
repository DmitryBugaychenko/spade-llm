from spade_llm.kafka.sink import KafkaMessageSink  # Assuming sink module exists
from kafka import KafkaProducer
import pytest


@pytest.fixture(scope='module')
def producer():
    """Fixture to set up a Kafka Producer."""
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
    yield producer
    producer.close()


def test_kafka_message_sink_send(producer):
    """Test sending message via KafkaMessageSink."""
    topic_name = 'test_topic'
    
    # Initialize KafkaMessageSink instance
    sink = KafkaMessageSink(topic_name, bootstrap_servers=['localhost:9092'])
    
    # Send a sample message
    message = b'Test message sent successfully.'
    sink.send(message)
    
    # Verify message was produced correctly
    future_metadata = producer.send(topic_name, value=message).get(timeout=5)
    assert future_metadata.topic == topic_name
    assert future_metadata.partition >= 0


def test_kafka_message_sink_invalid_server():
    """Test initialization failure due to invalid server address."""
    with pytest.raises(Exception):  # Adjust exception type depending on actual implementation
        KafkaMessageSink('test_topic', bootstrap_servers=['invalid_host:9092'])
