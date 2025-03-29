from spade_llm.platform.api import KeyValueStorage

class InMemoryKeyValueStorage(KeyValueStorage):
    def __init__(self):
        self._data = {}
    
    async def get_item(self, key: str) -> str:
        return self._data.get(key)
    
    async def put_item(self, key: str, value: str | None):
        if value is None:
            del self._data[key]
        else:
            self._data[key] = value
