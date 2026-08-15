"""Real API adapters (Anthropic + OpenAI-compatible), stdlib-only.

Deliberately thin: authentication, request shape, response parsing, and the
retry behaviour every network client needs. Model ids and prices live on the
`Seat` you construct, not here — vendors change both faster than code should.

Failures are translated into the typed errors in `base.py`, and that
translation is the whole point. A 429 or a 529 is an availability problem, and
the session records the participant absent for that round and carries on with
a reduced council. A missing API key is a configuration bug, and marking a
student absent for it would quietly run a two-model session and label it as
one, hiding the real problem behind a protocol feature built for a different
one.

**Keys are read from the environment and nowhere else.** No constructor
default, no file, no config lookup. `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`
live in your shell; nothing in this repo has ever seen them and the test suite
never touches the network.

This duplicates Switchboard's transport rather than importing it, which is a
real cost — two copies drift. It is accepted because the alternative is worse:
a user who wants one real session would have to clone a second repo to make a
single HTTP request, and "separate repo that imports the pattern, not a fork"
means the two projects install apart. The shape is kept identical on purpose,
so an adapter written for either satisfies both.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request

from .base import (
    Completion,
    ProviderConfigError,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderTruncated,
    ProviderUnavailable,
)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

_TRUNCATION_MARKERS = (
    "max_tokens or model output limit was reached",
    "could not finish the message",
    "max_completion_tokens",
)


def _looks_truncated(detail: str) -> bool:
    """Does this error body describe an exhausted completion budget?

    String matching on a vendor message, which is exactly as durable as it
    sounds — but the alternative is reporting a budget problem as a malformed
    request, which sends the reader to debug the wrong thing entirely. When
    the wording drifts, the failure mode is the old behaviour rather than a
    new one."""
    lowered = detail.lower()
    return any(marker in lowered for marker in _TRUNCATION_MARKERS)


class _HTTPProviderBase:
    """Shared transport: retries, backoff, and error translation."""

    name = "http"

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        sleep=time.sleep,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep  # injectable so tests do not actually wait

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_cap)
        delay = min(self.backoff_base * (2**attempt), self.backoff_cap)
        return delay * (0.5 + random.random() / 2)

    @staticmethod
    def _retry_after(headers) -> float | None:
        raw = headers.get("retry-after") if headers else None
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None  # HTTP-date form; fall back to computed backoff

    def _request(self, url: str, body: bytes, headers: dict[str, str], model_id: str) -> dict:
        last: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001 - body is best-effort context
                    pass
                finally:
                    e.close()
                message = f"{self.name}: HTTP {e.code} for {model_id}: {detail or e.reason}"
                if e.code in (401, 403):
                    raise ProviderConfigError(
                        message, provider=self.name, model_id=model_id
                    ) from None
                if e.code == 429:
                    last = ProviderRateLimited(message, provider=self.name, model_id=model_id)
                elif e.code in _RETRYABLE_STATUS:
                    last = ProviderUnavailable(message, provider=self.name, model_id=model_id)
                elif e.code == 400 and _looks_truncated(detail):
                    # Some vendors report an exhausted completion budget as a
                    # 400 rather than a `finish_reason`, which would otherwise
                    # land in the generic bucket below and read as "your
                    # request is malformed". It isn't: the request was fine and
                    # the budget was too small.
                    raise ProviderTruncated(
                        message, provider=self.name, model_id=model_id
                    ) from None
                else:
                    # 400, 404, 422: the request is wrong. Retrying re-sends
                    # the same bad request and hides a real bug.
                    raise ProviderError(
                        message, provider=self.name, model_id=model_id
                    ) from None
                retry_after = self._retry_after(e.headers)
            except socket.timeout:
                last = ProviderTimeout(
                    f"{self.name}: timed out after {self.timeout}s for {model_id}",
                    provider=self.name,
                    model_id=model_id,
                )
                retry_after = None
            except urllib.error.URLError as e:
                reason = getattr(e, "reason", e)
                if isinstance(reason, socket.timeout):
                    last = ProviderTimeout(
                        f"{self.name}: timed out after {self.timeout}s for {model_id}",
                        provider=self.name,
                        model_id=model_id,
                    )
                else:
                    last = ProviderUnavailable(
                        f"{self.name}: connection failed for {model_id}: {reason}",
                        provider=self.name,
                        model_id=model_id,
                    )
                retry_after = None
            except json.JSONDecodeError as e:
                raise ProviderError(
                    f"{self.name}: response was not valid JSON for {model_id}: {e}",
                    provider=self.name,
                    model_id=model_id,
                ) from None

            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, retry_after))

        assert last is not None  # only reachable after a retryable failure
        raise last


class AnthropicProvider(_HTTPProviderBase):
    """Adapter for the Anthropic Messages API.

    Reads `ANTHROPIC_API_KEY` from the environment.
    See https://docs.claude.com/en/api/overview for current API details.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        *,
        version: str = "2023-06-01",
        **transport,
    ) -> None:
        super().__init__(**transport)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.version = version

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY is not set", provider=self.name, model_id=model_id
            )
        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        data = self._request(
            f"{self.base_url}/v1/messages",
            body,
            {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
            },
            model_id,
        )
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        if data.get("stop_reason") == "max_tokens":
            raise ProviderTruncated(
                f"{self.name}: {model_id} hit the {max_tokens}-token completion "
                f"budget; raise SessionConfig.max_tokens",
                provider=self.name,
                model_id=model_id,
            )
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


class OpenAICompatibleProvider(_HTTPProviderBase):
    """Adapter for OpenAI and the many APIs that copy its chat-completions shape.

    Reads `OPENAI_API_KEY` (or whatever `env_var` names) from the environment.
    Point `base_url` at any compatible vendor and set `name` to match the
    provider on your `Seat`.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        *,
        name: str | None = None,
        env_var: str = "OPENAI_API_KEY",
        max_tokens_param: str | None = None,
        **transport,
    ) -> None:
        super().__init__(**transport)
        self.api_key = api_key or os.environ.get(env_var, "")
        self.base_url = base_url.rstrip("/")
        self.env_var = env_var
        if name:
            self.name = name
        # OpenAI's newer models reject `max_tokens`; most compatible vendors
        # only accept it. Pick by host, and let the caller override.
        self._max_tokens_param = max_tokens_param or (
            "max_completion_tokens" if "api.openai.com" in base_url else "max_tokens"
        )

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                f"{self.env_var} is not set", provider=self.name, model_id=model_id
            )
        body = json.dumps(
            {
                "model": model_id,
                self._max_tokens_param: max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        data = self._request(
            f"{self.base_url}/v1/chat/completions",
            body,
            {
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            model_id,
        )
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage", {})
        if choices[0].get("finish_reason") == "length":
            reasoning = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            raise ProviderTruncated(
                f"{self.name}: {model_id} hit the {max_tokens}-token completion budget"
                + (f", spending {reasoning} of it on reasoning" if reasoning else "")
                + "; raise SessionConfig.max_tokens",
                provider=self.name,
                model_id=model_id,
            )
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
