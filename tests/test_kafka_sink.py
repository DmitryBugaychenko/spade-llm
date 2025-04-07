import unittest
import asyncio
from unittest.mock import MagicMock
from spade_llm.core.api import Message, AgentId  
from spade_llm.kafka.sink import KafkaMessageSink, KafkaProducerConfig  
from spade_llm.kafka.source import KafkaMessageSource, KafkaConsumerConfig


class TestKafkaMessageSink(unittest.TestCase):
    
    def test_kafka_message_sink_send(self):
        """Test sending message via KafkaMessageSink."""
        asyncio.run(self._run_async_test())

    async def _run_async_test(self):
        # Define the agent type for the topic name
        receiver = AgentId(agent_type="receiver", agent_id="1235")
        sender = AgentId(agent_type="sender", agent_id="12356")

        # Initialize KafkaMessageSink instance
        sink: KafkaMessageSink = KafkaMessageSink()._configure(KafkaProducerConfig(
            bootstrap_servers='localhost:9092',
            client_id='test_client',
            linger_ms=100))

        sink.start()

        # Prepare a sample Message object
        message = Message(sender=sender, receiver=receiver, performative="inform", content="Test message sent successfully.")

        # Mock MessageSink to verify if the message was received
        mock_sink = MagicMock(spec=Message)

        # Initialize KafkaMessageSource with mocked sink
        source: KafkaMessageSource = KafkaMessageSource()._configure(KafkaConsumerConfig(
            bootstrap_servers='localhost:9092',
            group_id='test_client',
            exposed_agents=['receiver'],
            message_sink=mock_sink))  # Pass the mock sink here

        source.start()

        # Post the message through the sink
        await sink.post_message(message)

        # Wait for the message to propagate
        await asyncio.sleep(1)

        # Verify that the mock sink received the message
        mock_sink.assert_called_once_with(message)

        # Clean up resources
        sink.close()
        await sink.join()
        source.stop()
        await source.join()

if __name__ == "__main__":
    unittest.main()
