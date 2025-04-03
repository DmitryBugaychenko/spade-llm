import json
import unittest
from typing import Optional

import yaml
from langchain_gigachat import GigaChatEmbeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from spade_llm.core.models import LlmConfiguration, EmbeddingsModelConfiguration


class TestGigaChat(unittest.TestCase):
    def yaml_to_json(self, s: str):
        data = yaml.full_load(s)
        return json.dumps(data)

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(
            default=None, description="How funny the joke is, from 1 to 10"
        )

    def test_chat(self):
        conf_yaml = '''
            type_name: spade_llm.platform.gigachat.GigaChatModelFactory
            args:
              credentials: env.GIGA_CRED
              model: GigaChat-2
              verify_ssl_certs: False
            '''
        conf = LlmConfiguration.model_validate_json(self.yaml_to_json(conf_yaml))

        model: BaseChatModel = conf.create_configurable_instance().create_model()

        structured_llm = model.with_structured_output(self.Joke)

        joke = structured_llm.invoke("Tell me a joke about cats")
        print("From structured output: " + str(joke))
        self.assertIsInstance(joke, self.Joke)

    def test_embeddings(self):
        conf_yaml = '''
                type_name: spade_llm.platform.gigachat.GigaChatEmbeddingsFactory
                args:
                  credentials: env.GIGA_CRED
                  verify_ssl_certs: False
                '''
        conf = EmbeddingsModelConfiguration.model_validate_json(self.yaml_to_json(conf_yaml))

        model: GigaChatEmbeddings = conf.create_configurable_instance().create_model()

        embedding = model.embed_query("Some short text")
        print(embedding)
        self.assertGreater(len(embedding), 0)


if __name__ == "__main__":
    unittest.main()
