import asyncio
import logging
import socket
import unittest
from typing import Optional

import aiologic

from spade_llm.core.api import Message, AgentId, MessageSink
from spade_llm.kafka.sink import KafkaMessageSink, KafkaProducerConfig
from spade_llm.kafka.source import KafkaMessageSource, KafkaConsumerConfig


class SingleMessageSink(MessageSink):
    def post_message_sync(self, msg: Message):
        self.msg = msg
        self.event.set()

    msg: Optional[Message]
    event = aiologic.Event()
    async def post_message(self, msg: Message):
        self.post_message_sync(msg)

class TestKafkaMessageSink(unittest.TestCase):

    def is_port_open(self, host, port):
        """Check if a port is open on a given host."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        finally:
            sock.close()

    def test_kafka_message_sink_send(self):
        logging.basicConfig(level=logging.DEBUG)
        """Test sending message via KafkaMessageSink."""
        
        # Check if Kafka is available on localhost:9092
        if not self.is_port_open('localhost', 9092):
            self.skipTest("Kafka broker not available on localhost:9092")
        
        asyncio.run(self._run_async_test())

    async def _run_async_test(self):
        # Define the agent type for the topic name
        receiver = AgentId(agent_type="receiver", agent_id="1235")
        sender = AgentId(agent_type="sender", agent_id="12356")

        # Initialize KafkaMessageSink instance
        sink: KafkaMessageSink = KafkaMessageSink()._configure(KafkaProducerConfig(
            bootstrap_servers='localhost:9092',
            client_id='test_client',
            linger_ms=0))

        sink.start()

        try:

            # Prepare a sample Message object
            message = Message(sender=sender, receiver=receiver, performative="inform", content="Test message sent successfully.")

            # Mock MessageSink to verify if the message was received
            mock_sink = SingleMessageSink()

            # Initialize KafkaMessageSource with mocked sink
            source: KafkaMessageSource = KafkaMessageSource()._configure(KafkaConsumerConfig(
                bootstrap_servers='localhost:9092',
                group_id='test_client'))

            await source.start(mock_sink,{"receiver"})

            try:
                # Post the message through the sink
                await sink.post_message(message)

                # Wait for the message to propagate
                await mock_sink.event.wait(timeout=60)

                # Verify that the mock sink received the message
                self.assertEqual(mock_sink.msg.sender, message.sender)
                self.assertEqual(mock_sink.msg.receiver, message.receiver)
                self.assertEqual(mock_sink.msg.performative, message.performative)
                self.assertEqual(mock_sink.msg.content, message.content)
            finally:
                source.stop()
                await source.join()
        finally:
            # Clean up resources
            sink.close()
            await sink.join()


if __name__ == "__main__":
    unittest.main()
