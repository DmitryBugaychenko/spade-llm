from spade_llm.core.agent import Agent


class DummyAgent(Agent):
    """
    This is a dummy agent for testing purposes. It allows to send messages to over agents
    and collect all incoming messages for assertions
    """