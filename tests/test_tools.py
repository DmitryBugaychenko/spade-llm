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

    def test_wikipedia_run(self):
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
        conf = ConfigurableRecord.model_validate_json(self.yaml_to_json(conf_yaml))

        parsed: LangChainApiWrapperToolFactory = conf.create_configurable_instance()

        wiki: WikipediaQueryRun = parsed.create_tool()
        self.assertIsInstance(wiki, WikipediaQueryRun)

if __name__ == "__main__":
    unittest.main()
