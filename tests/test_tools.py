import json
import unittest

import yaml
from langchain_community.tools import WikipediaQueryRun

from spade_llm.platform.conf import ConfigurableRecord
from spade_llm.platform.tools import LangChainApiWrapperToolFactory


class TestConfig(unittest.TestCase):
    def yaml_to_json(self, s: str):
        data = yaml.full_load(s)
        return json.dumps(data)
    
    def test_wikipedia_argument_validation(self):
        # Load the YAML configuration string
        conf_yaml = '''
            type_name: spade_llm.platform.tools.LangChainApiWrapperToolFactory
            args:
              type_name: langchain_community.tools.WikipediaQueryRun
              api_wrapper:
                type_name: langchain_community.utilities.WikipediaAPIWrapper
                args:
                  top_k_results: 1
                  doc_content_chars_max: 4096
                  lang: en
                name: wikipedia_api
            '''
        
        # Convert YAML to JSON and load into ConfigurableRecord
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        # Create the LangChainApiWrapperToolFactory instance
        parsed: LangChainApiWrapperToolFactory = conf.create_configurable_instance()

        # Get the WikipediaQueryRun tool
        wiki: WikipediaQueryRun = parsed.create_tool()

        # Validate that the arguments were properly set
        self.assertEqual(wiki.api_wrapper.top_k_results, 1)
        self.assertEqual(wiki.api_wrapper.doc_content_chars_max, 4096)
        self.assertEqual(wiki.api_wrapper.lang, 'en')

    def test_wikipedia_query_run(self):
        # Load the YAML configuration string (same as before)
        conf_yaml = '''
            type_name: spade_llm.platform.tools.LangChainApiWrapperToolFactory
            args:
              type_name: langchain_community.tools.WikipediaQueryRun
              api_wrapper:
                type_name: langchain_community.utilities.WikipediaAPIWrapper
                args:
                  top_k_results: 1
                  doc_content_chars_max: 4096
                  lang: en
                name: wikipedia_api
            '''
        
        # Convert YAML to JSON and load into ConfigurableRecord
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        # Create the LangChainApiWrapperToolFactory instance
        parsed: LangChainApiWrapperToolFactory = conf.create_configurable_instance()

        # Get the WikipediaQueryRun tool
        wiki: WikipediaQueryRun = parsed.create_tool()

        # Test invoking the WikipediaQueryRun tool
        result = wiki.run("Python programming language")
        self.assertIn("Python", result)

if __name__ == "__main__":
    unittest.main()
