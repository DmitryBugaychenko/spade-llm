from pydantic import BaseModel
from spade_llm.platform.api import (
    KeyValueStorage, StorageFactory,
)

class InMemoryKeyValueStorage(KeyValueStorage):
    async def close(self):
        pass

    def __init__(self):
        self._data = {}
    
    async def get_item(self, key: str) -> str:
        return self._data.get(key)
    
    async def put_item(self, key: str, value: str | None):
        if value is None:
            del self._data[key]
        else:
            self._data[key] = value

class InMemoryStorageFactory(StorageFactory):
    async def create_storage(self, agent_type: str):
        return InMemoryKeyValueStorage()


class TrackedKeys(BaseModel):
    tracked_keys: list[str]

class PrefixKeyValueStorage(KeyValueStorage):
    TRACKED_KEYS_KEY = "_tracked_keys"

    def __init__(self, wrapped_storage: KeyValueStorage, prefix: str):
        self.wrapped_storage = wrapped_storage
        self.prefix = prefix

    async def get_item(self, key: str) -> str | None:
        prefixed_key = f"{self.prefix}:{key}"
        return await self.wrapped_storage.get_item(prefixed_key)

    async def put_item(self, key: str, value: str | None):
        prefixed_key = f"{self.prefix}:{key}"
        await self.wrapped_storage.put_item(prefixed_key, value)
        if value is not None:
            await self._add_tracked_key(prefixed_key)

    async def _get_tracked_keys(self) -> TrackedKeys:
        prefixed_tracked_key = f"{self.prefix}:{self.TRACKED_KEYS_KEY}"
        data = await self.wrapped_storage.get_item(prefixed_tracked_key)
        if data:
            return TrackedKeys.model_validate_json(data)
        else:
            return TrackedKeys(tracked_keys=[])

    async def _set_tracked_keys(self, tracked_keys: TrackedKeys):
        prefixed_tracked_key = f"{self.prefix}:{self.TRACKED_KEYS_KEY}"
        await self.wrapped_storage.put_item(
            prefixed_tracked_key,
            tracked_keys.model_dump_json(exclude_unset=True)
        )

    async def _add_tracked_key(self, key: str):
        tracked_keys = await self._get_tracked_keys()
        if key not in tracked_keys.tracked_keys:
            tracked_keys.tracked_keys.append(key)
            await self._set_tracked_keys(tracked_keys)

    async def close(self):
        tracked_keys = await self._get_tracked_keys()
        for key in tracked_keys.tracked_keys:
            await self.wrapped_storage.put_item(key, None)
        await self.wrapped_storage.put_item(f"{self.prefix}:{self.TRACKED_KEYS_KEY}", None)


