import asyncio
from unittest.mock import AsyncMock, MagicMock
from spade_llm.api import KeyValueStorage
from prefixkeyvaluestorage import PrefixKeyValueStorage

async def test_get_item():
    mock_storage = AsyncMock(spec=KeyValueStorage)
    mock_storage.get_item.return_value = 'mock_value'
    prefix_storage = PrefixKeyValueStorage(mock_storage, 'test:')
    assert await prefix_storage.get_item('key') == 'mock_value'
    mock_storage.get_item.assert_called_once_with('test:key')

async def test_put_item():
    mock_storage = AsyncMock(spec=KeyValueStorage)
    prefix_storage = PrefixKeyValueStorage(mock_storage, 'test:')
    await prefix_storage.put_item('key', 'value')
    mock_storage.put_item.assert_called_once_with('test:key', 'value')

async def test_close():
    mock_storage = AsyncMock(spec=KeyValueStorage)
    prefix_storage = PrefixKeyValueStorage(mock_storage, 'test:')
    await prefix_storage.close()
    mock_storage.put_item.assert_not_called()

async def main():
    await test_get_item()
    await test_put_item()
    await test_close()

if __name__ == "__main__":
    asyncio.run(main())
