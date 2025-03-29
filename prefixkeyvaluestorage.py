from spade_llm.api import KeyValueStorage

class PrefixKeyValueStorage(KeyValueStorage):
    def __init__(self, wrapped_storage: KeyValueStorage, prefix: str):
        self.wrapped_storage = wrapped_storage
        self.prefix = prefix

    async def get_item(self, key: str) -> str:
        prefixed_key = f"{self.prefix}{key}"
        return await self.wrapped_storage.get_item(prefixed_key)

    async def put_item(self, key: str, value: str | None):
        prefixed_key = f"{self.prefix}{key}"
        await self.wrapped_storage.put_item(prefixed_key, value)
