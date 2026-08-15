"""Provider adapters. The protocol is one method wide, on purpose."""

from .http import AnthropicProvider, OpenAICompatibleProvider
from .base import (
    Completion,
    FlakyProvider,
    MockProvider,
    Provider,
    ProviderConfigError,
    ProviderError,
    ProviderPool,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ScriptedProvider,
    mock_pool,
)

__all__ = [
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "Completion",
    "FlakyProvider",
    "MockProvider",
    "Provider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderPool",
    "ProviderRateLimited",
    "ProviderTimeout",
    "ProviderUnavailable",
    "ScriptedProvider",
    "mock_pool",
]
