import unittest
from spade_llm.platform.models import ModelsProviderConfig, ChatModelConfiguration, EmbeddingsModelConfiguration

class TestModelsProviderConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            'chat_models': {'model1': ChatModelConfiguration()},
            'embeddings_models': {'embedding1': EmbeddingsModelConfiguration()}
        }
        self.invalid_chat_model_config = {
            'chat_models': {},
            'embeddings_models': {'embedding1': EmbeddingsModelConfiguration()}
        }
        self.invalid_embedding_model_config = {
            'chat_models': {'model1': ChatModelConfiguration()},
            'embeddings_models': {}
        }

    def test_valid_config(self):
        config = ModelsProviderConfig(**self.valid_config)
        self.assertIsInstance(config, ModelsProviderConfig)

    def test_invalid_chat_model(self):
        with self.assertRaises(ValueError):
            ModelsProviderConfig(**self.invalid_chat_model_config).create_chat_model('nonexistent')

    def test_invalid_embedding_model(self):
        with self.assertRaises(ValueError):
            ModelsProviderConfig(**self.invalid_embedding_model_config).create_embeddings_model('nonexistent')

    def test_create_chat_model(self):
        config = ModelsProviderConfig(**self.valid_config)
        result = config.create_chat_model('model1')
        self.assertIsInstance(result, object)

    def test_create_embeddings_model(self):
        config = ModelsProviderConfig(**self.valid_config)
        result = config.create_embeddings_model('embedding1')
        self.assertIsInstance(result, object)

if __name__ == "__main__":
    unittest.main()
