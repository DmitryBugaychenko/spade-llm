import asyncio
import time
import unittest

from spade_llm.builders import MessageBuilder
from spade_llm.consts import Templates
from spade_llm.core.behaviors import MessageTemplate, Behaviour
from spade_llm.core.agent import Agent
from tests.base import SpadeTestCase, ModelTestCase


ECHO_AGENT = "echo@localhost"
SENDER_AGENT = "sender@localhost"


class EchoAgent(Agent):
    """
    An agent that echoes back any message it receives.
    """
    def setup(self):
        # Add a behavior to handle and echo messages
        from spade_llm.core.behaviors import MessageHandlingBehavior
        from spade_llm.core.api import AgentContext, Message
        
        class EchoBehavior(MessageHandlingBehavior):
            async def handle_message(self, context: AgentContext, message: Message):
                # Echo back the same message
                reply = MessageBuilder.inform().to_agent(str(message.sender)).with_content(message.body)
                await context.send(reply)
        
        self.add_behaviour(EchoBehavior(MessageTemplate.inform()))


class SenderAgent(Agent):
    """
    An agent that sends a message, waits, and then tries to receive the reply.
    """
    def __init__(self, jid: str, password: str):
        super().__init__(agent_type=jid)
        self.jid = jid
        self.password = password
        self.received_reply = None


class RaceConditionTestCase(SpadeTestCase, ModelTestCase):
    """
    Test case to reproduce the race condition in Behaviour receive method.
    
    Bug: If Agent first sends a message and then calls receive, there is a race
    condition where the reply might arrive before receive is called, and the
    reply will be lost.
    """

    @classmethod
    def setUpClass(cls):
        SpadeTestCase.setUpClass()
        
        # Start the echo agent
        echo = EchoAgent(agent_type=ECHO_AGENT)
        RaceConditionTestCase.echo = echo
        SpadeTestCase.startAgent(echo)
        
        # Start the sender agent
        sender = SenderAgent(jid=SENDER_AGENT, password="pwd")
        RaceConditionTestCase.sender = sender
        SpadeTestCase.startAgent(sender)

    @classmethod
    def tearDownClass(cls):
        SpadeTestCase.stopAgent(RaceConditionTestCase.sender)
        SpadeTestCase.stopAgent(RaceConditionTestCase.echo)
        SpadeTestCase.tearDownClass()

    def test_race_condition_receive_after_send(self):
        """
        Test that reproduces the race condition:
        1. Sender agent sends a message to echo agent
        2. Echo agent immediately replies
        3. Sender agent sleeps for a short time (simulating processing)
        4. Sender agent then tries to receive the reply
        
        The bug is that if the reply arrives before receive is called,
        it gets lost.
        """
        # Create a simple behavior that sends and waits for reply
        class SendAndWaitBehavior(Behaviour):
            def __init__(self):
                self.received_reply = None
                self._is_done = False
            
            async def run(self):
                # Send a message to echo agent
                msg = MessageBuilder.request().to_agent(ECHO_AGENT).with_content("Hello").build()
                await self.context.send_message(msg)
                
                # Wait for a short time (this simulates processing delay)
                # During this time, the echo agent will reply
                await asyncio.sleep(0.5)
                
                # Now try to receive the reply
                # This is where the race condition occurs - if reply arrived
                # before this call, it will be lost
                try:
                    reply = await self.context.receive(timeout=5)
                    self.received_reply = reply
                except Exception as e:
                    print(f"Error receiving reply: {e}")
                
                self._is_done = True
            
            def is_done(self):
                return self._is_done

        # Add the behavior to sender agent
        behavior = SendAndWaitBehavior()
        self.sender.add_behaviour(behavior)
        
        # Wait for the behavior to complete
        self.run_in_container(behavior.join(10))
        
        # Give some time for the reply to be processed
        time.sleep(1)
        
        # Check if we received the reply
        # This test should FAIL if the race condition exists
        self.assertIsNotNone(behavior.received_reply, 
            "Reply was lost due to race condition - it arrived before receive() was called")
        self.assertEqual(behavior.received_reply.body, "Hello")


if __name__ == '__main__':
    unittest.main()
