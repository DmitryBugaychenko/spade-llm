from spade_llm.platform.api import KeyValueStorage

class PrefixKeyValueStorage(KeyValueStorage):
    def __init__(self, wrapped_storage: KeyValueStorage, prefix: str):
        self.wrapped_storage = wrapped_storage
        self.prefix = prefix
        self.tracked_keys = []

    async def get_item(self, key: str) -> str | None:
        prefixed_key = f"{self.prefix}:{key}"
        return await self.wrapped_storage.get_item(prefixed_key)

    async def put_item(self, key: str, value: str | None):
        prefixed_key = f"{self.prefix}:{key}"
        await self.wrapped_storage.put_item(prefixed_key, value)
        if value is not None:
            self.tracked_keys.append(prefixed_key)

    async def close(self):
        for key in self.tracked_keys:
            await self.wrapped_storage.put_item(key, None)
        self.tracked_keys.clear()
