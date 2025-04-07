import unittest
from spade_llm.kafka.sink import KafkaMessageSink  # Assuming sink module exists
from kafka import KafkaProducer
from spade_llm.core.api import Message, AgentId  # Import Message and AgentId classes


class TestKafkaMessageSink(unittest.TestCase):

    def test_kafka_message_sink_send(self):
        """Test sending message via KafkaMessageSink."""
        # Define the agent type for the topic name
        receiver = AgentId("receiver", "type")
        topic_name = f"{receiver.agent_type}_messages"

        # Initialize KafkaMessageSink instance
        sink = KafkaMessageSink(topic_name, bootstrap_servers=['localhost:9092'])

        # Prepare a sample Message object
        message_content = {"content": "Test message sent successfully."}
        message = Message(sender="sender_id", receiver="receiver_id", body=message_content)

        # Post the message through the sink
        sink.post_message(message)

    def test_kafka_message_sink_invalid_server(self):
        """Test initialization failure due to invalid server address."""
        with self.assertRaises(Exception):  # Adjust exception type depending on actual implementation
            KafkaMessageSink('test_topic', bootstrap_servers=['invalid_host:9092'])


if __name__ == "__main__":
    unittest.main()
