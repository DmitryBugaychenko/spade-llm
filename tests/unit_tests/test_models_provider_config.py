import pytest
from spade_llm.platform.models import ModelsProviderConfig, ChatModelConfiguration, EmbeddingsModelConfiguration

@pytest.fixture
def valid_config():
    return {
        'chat_models': {'model1': ChatModelConfiguration()},
        'embeddings_models': {'embedding1': EmbeddingsModelConfiguration()}
    }

@pytest.fixture
def invalid_chat_model_config():
    return {
        'chat_models': {},
        'embeddings_models': {'embedding1': EmbeddingsModelConfiguration()}
    }

@pytest.fixture
def invalid_embedding_model_config():
    return {
        'chat_models': {'model1': ChatModelConfiguration()},
        'embeddings_models': {}
    }

def test_valid_config(valid_config):
    config = ModelsProviderConfig(**valid_config)
    assert isinstance(config, ModelsProviderConfig)
    
def test_invalid_chat_model(invalid_chat_model_config):
    with pytest.raises(ValueError):
        ModelsProviderConfig(**invalid_chat_model_config).create_chat_model('nonexistent')

def test_invalid_embedding_model(invalid_embedding_model_config):
    with pytest.raises(ValueError):
        ModelsProviderConfig(**invalid_embedding_model_config).create_embeddings_model('nonexistent')

def test_create_chat_model(valid_config):
    config = ModelsProviderConfig(**valid_config)
    result = config.create_chat_model('model1')
    assert isinstance(result, object)

def test_create_embeddings_model(valid_config):
    config = ModelsProviderConfig(**valid_config)
    result = config.create_embeddings_model('embedding1')
    assert isinstance(result, object)
