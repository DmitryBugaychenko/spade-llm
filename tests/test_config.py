import asyncio
import unittest

from pydantic import BaseModel, Field

from spade_llm.platform.behaviors import (
    Behaviour,
    BehaviorsOwner,
    MessageTemplate,
    Message,
    MessageHandlingBehavior,
)
from spade_llm.platform.api import AgentId
from spade_llm.platform.conf import Configurable, configuration, ConfigurableRecord


class StringConfig(BaseModel):
    s: str = Field(description="Single field configuration for testing")

@configuration(StringConfig)
class StringConfigurable(Configurable[StringConfig]):
    pass

class IntConfig(BaseModel):
    i: int = Field(description="Single int field configuration for testing")

class MultipleConfig(StringConfig, IntConfig):
    pass

class TestConfig(unittest.TestCase):
    
    def test_configurable_parse_config(self):
        expected = StringConfig(s="str")

        conf = StringConfigurable.parse_config(expected.model_dump_json())

        self.assertEqual(expected, conf)

    def test_configurable_load_string_config(self):
        conf = ConfigurableRecord.model_validate_json(
            '{"type_name": "tests.test_config.StringConfigurable", "s": "str"}'
        )

        parsed = conf.create_instance()

        self.assertEqual("str", parsed.config().s)
        
    # Newly added test case for IntConfig
    def test_configurable_load_int_config(self):
        conf = ConfigurableRecord.model_validate_json(
            '{"type_name": "tests.test_config.IntConfig", "i": 42}'
        )

        parsed = conf.create_instance()

        self.assertEqual(42, parsed.i)
        
    # Newly added test case for MultipleConfig
    def test_configurable_load_multiple_config(self):
        conf = ConfigurableRecord.model_validate_json(
            '{"type_name": "tests.test_config.MultipleConfig", "s": "hello", "i": 100}'
        )

        parsed = conf.create_instance()

        self.assertEqual("hello", parsed.s)
        self.assertEqual(100, parsed.i)


if __name__ == "__main__":
    unittest.main()
