import asyncio
from typing import Any, Optional, List, Dict, cast
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.models import ChatModelFactory


class RateLimitedChatModel(BaseChatModel):
    """
    A wrapper around another chat model that limits the number of concurrent requests.
    If the number of requests exceeds the limit, it waits for running requests to complete.
    """
    wrapped_model: BaseChatModel
    max_concurrent: int = Field(default=1, description="Maximum number of concurrent requests")
    _semaphore: Optional[asyncio.Semaphore] = None

    def __init__(self, wrapped_model: BaseChatModel, max_concurrent: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.wrapped_model = wrapped_model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def _llm_type(self) -> str:
        return f"rate_limited_{self.wrapped_model._llm_type}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # For synchronous calls, we need to handle this differently
        # Since we can't use async directly, we'll use a thread pool
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Create a new event loop in a separate thread
        result = None
        exception = None
        
        def run_in_thread():
            nonlocal result, exception
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self._agenerate(messages, stop, run_manager, **kwargs)
                )
                loop.close()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        async with self._semaphore:
            return await self.wrapped_model._agenerate(messages, stop, run_manager, **kwargs)


class RateLimitedChatModelConfig(BaseModel):
    wrapped_model_config: Dict[str, Any] = Field(
        description="Configuration for the wrapped model"
    )
    max_concurrent: int = Field(
        default=1,
        description="Maximum number of concurrent requests to allow"
    )


@configuration(RateLimitedChatModelConfig)
class RateLimitedChatModelFactory(ChatModelFactory[RateLimitedChatModel], Configurable[RateLimitedChatModelConfig]):
    def create_model(self) -> RateLimitedChatModel:
        config = self.config
        
        # We need to create the wrapped model first
        # This assumes the wrapped_model_config contains a type_name field
        from spade_llm.core.conf import ConfigurableRecord
        
        # Create a ConfigurableRecord from the wrapped_model_config
        wrapped_config_record = ConfigurableRecord(**config.wrapped_model_config)
        wrapped_model = wrapped_config_record.create_configurable_instance()
        
        # Ensure the created instance is a BaseChatModel
        if not isinstance(wrapped_model, BaseChatModel):
            raise ValueError("Wrapped model must be a BaseChatModel")
            
        return RateLimitedChatModel(
            wrapped_model=wrapped_model,
            max_concurrent=config.max_concurrent
        )
