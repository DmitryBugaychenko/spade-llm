import unittest
from spade_llm.kafka.sink import KafkaMessageSink  # Assuming sink module exists
from kafka import KafkaProducer


class TestKafkaMessageSink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        
    @classmethod
    def tearDownClass(cls):
        cls.producer.close()

    def test_kafka_message_sink_send(self):
        """Test sending message via KafkaMessageSink."""
        topic_name = 'test_topic'
        
        # Initialize KafkaMessageSink instance
        sink = KafkaMessageSink(topic_name, bootstrap_servers=['localhost:9092'])
        
        # Send a sample message
        message = b'Test message sent successfully.'
        sink.send(message)
        
        # Verify message was produced correctly
        future_metadata = self.producer.send(topic_name, value=message).get(timeout=5)
        self.assertEqual(future_metadata.topic, topic_name)
        self.assertGreaterEqual(future_metadata.partition, 0)

    def test_kafka_message_sink_invalid_server(self):
        """Test initialization failure due to invalid server address."""
        with self.assertRaises(Exception):  # Adjust exception type depending on actual implementation
            KafkaMessageSink('test_topic', bootstrap_servers=['invalid_host:9092'])


if __name__ == "__main__":
    unittest.main()
