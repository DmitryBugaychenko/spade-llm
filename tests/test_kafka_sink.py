import unittest
import asyncio

from spade_llm.core.api import Message, AgentId  # Import Message and AgentId classes
from spade_llm.kafka.sink import KafkaMessageSink, KafkaProducerConfig  # Assuming sink module exists


class TestKafkaMessageSink(unittest.TestCase):

    def test_kafka_message_sink_send(self):
        """Test sending message via KafkaMessageSink."""
        asyncio.run(self._run_async_test())

    async def _run_async_test(self):
        # Define the agent type for the topic name
        receiver = AgentId(agent_type="receiver", agent_id="1235")
        sender = AgentId(agent_type="sender", agent_id="12356")

        # Initialize KafkaMessageSink instance
        sink = KafkaMessageSink(KafkaProducerConfig(
            bootstrap_servers='localhost:9092',
            client_id='test_client',
            linger_ms=100))

        # Prepare a sample Message object
        message = Message(sender=sender, receiver=receiver, performative="inform", content="Test message sent successfully.")

        # Post the message through the sink
        await sink.post_message(message)

    def test_kafka_message_sink_invalid_server(self):
        """Test initialization failure due to invalid server address."""
        with self.assertRaises(Exception):  # Adjust exception type depending on actual implementation
            KafkaMessageSink('test_topic', bootstrap_servers=['invalid_host:9092'])


if __name__ == "__main__":
    unittest.main()
