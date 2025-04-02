import json
import unittest

import yaml
from pydantic import BaseModel, Field

from spade_llm.platform.conf import Configurable, configuration, ConfigurableRecord


class StringConfig(BaseModel):
    s: str = Field(description="Single field configuration for testing")

@configuration(StringConfig)
class StringConfigurable(Configurable[StringConfig]):
    pass

class IntConfig(BaseModel):
    i: int = Field(description="Single int field configuration for testing")

@configuration(IntConfig)
class IntConfigurable(Configurable[IntConfig]):
    pass

class MultipleConfig(StringConfig, IntConfig):
    pass

@configuration(MultipleConfig)
class MultipleConfigurable(Configurable[MultipleConfig]):
    pass

class NestedConfig(BaseModel):
    string_part: StringConfig = Field(description="The string part of the nested config.")
    integer_part: IntConfig = Field(description="The integer part of the nested config.")

@configuration(NestedConfig)
class NestedConfigurable(Configurable[NestedConfig]):
    pass

class InvalidConfig(BaseModel):
    x: float = Field(description="Invalid configuration for testing")

# No @configuration decorator applied
class InvalidConfigurable(Configurable[InvalidConfig]):
    pass

class TestConfig(unittest.TestCase):
    def yaml_to_json(self, s: str):
        data = yaml.full_load(s)
        return json.dumps(data)
    
    def test_configurable_parse_config(self):
        expected = StringConfig(s="str")
        conf = StringConfigurable.parse_config(expected.model_dump_json())
        self.assertEqual(expected, conf)

    def test_configurable_load_string_config(self):
        conf_yaml = '''
            type_name: tests.test_config.StringConfigurable
            s: str
            '''
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        parsed = conf.create_instance()

        self.assertEqual("str", parsed.config().s)

    # Newly added test case for IntConfig
    def test_configurable_load_int_config(self):
        conf_yaml = '''
            type_name: tests.test_config.IntConfigurable
            i: 42
            '''
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        parsed = conf.create_instance()

        self.assertEqual(42, parsed.config().i)

        # Newly added test case for MultipleConfig
    def test_configurable_load_multiple_config(self):
        conf_yaml = '''
            type_name: tests.test_config.MultipleConfigurable
            s: hello
            i: 100
            '''
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        parsed = conf.create_instance()

        self.assertEqual("hello", parsed.config().s)
        self.assertEqual(100, parsed.config().i)

        # Unit test for NestedConfigurable
    def test_configurable_load_nested_config(self):
        conf_yaml = '''
            type_name: tests.test_config.NestedConfigurable
            string_part:
              s: nested_string
            integer_part:
              i: 123
            '''
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        parsed = conf.create_instance()

        self.assertEqual("nested_string", parsed.config().string_part.s)
        self.assertEqual(123, parsed.config().integer_part.i)

    def test_configurable_load_invalid_type_name(self):
        conf_yaml = '''
        type_name: non.existent.module.Class
        '''
        with self.assertRaises((ImportError, AttributeError)):
            conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))
            conf.create_instance()

    def test_configurable_missing_required_field(self):
        conf_yaml = '''
        type_name: tests.test_config.IntConfigurable
        '''
        with self.assertRaises(ValueError):  
            conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))
            conf.create_instance()

    def test_configurable_incorrect_field_type(self):
        conf_yaml = '''
        type_name: tests.test_config.IntConfigurable
        i: "not_an_integer"
        '''
        with self.assertRaises(ValueError): 
            conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))
            conf.create_instance()

    def test_configurable_no_configuration_decorator(self):
        conf_yaml = '''
        type_name: tests.test_config.InvalidConfigurable
        x: 3.14
        '''
        with self.assertRaises(AttributeError):  
            conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))
            conf.create_instance()

if __name__ == "__main__":
    unittest.main()
