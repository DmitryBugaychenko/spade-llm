# tests/test_models_provider_config.py

import unittest
from spade_llm.core.models import (
    ModelsProviderConfig,
    ChatModelConfiguration,
    EmbeddingsModelConfiguration,
)

# Implementing mock configurations
class MockChatModelConfiguration(ChatModelConfiguration):
    def create_model_factory(self):
        # Return a dummy factory (for example purposes only)
        class DummyFactory:
            def create_model(self):
                return object()  # Returns a generic object

        return DummyFactory()

class MockEmbeddingsModelConfiguration(EmbeddingsModelConfiguration):
    def create_model_factory(self):
        # Similar approach as above
        class DummyFactory:
            def create_model(self):
                return object()  # Again, returns a generic object

        return DummyFactory()

class TestModelsProviderConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            'chat_models': {'mock_chat_model': MockChatModelConfiguration(type_name='mock_chat_model')},
            'embeddings_models': {'mock_embeddings_model': MockEmbeddingsModelConfiguration(type_name='mock_embeddings_model')}
        }
        
        # Invalid configs remain unchanged since we still want to test error handling
        self.invalid_chat_model_config = {
            'chat_models': {},
            'embeddings_models': {'mock_embeddings_model': MockEmbeddingsModelConfiguration(type_name='mock_embeddings_model')}
        }
        
        self.invalid_embedding_model_config = {
            'chat_models': {'mock_chat_model': MockChatModelConfiguration(type_name='mock_chat_model')},
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
        result = config.create_chat_model('mock_chat_model')
        self.assertIsInstance(result, object)
    
    def test_create_embeddings_model(self):
        config = ModelsProviderConfig(**self.valid_config)
        result = config.create_embeddings_model('mock_embeddings_model')
        self.assertIsInstance(result, object)

if __name__ == "__main__":
    unittest.main()
