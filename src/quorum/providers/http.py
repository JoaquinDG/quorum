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
import threading
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
        deadline: float = 300.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        sleep=time.sleep,
    ) -> None:
        self.timeout = timeout
        self.deadline = deadline
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep  # injectable so tests do not actually wait

    def _urlopen(self, req):
        """`urlopen` with a wall-clock ceiling the socket layer cannot defeat.

        `timeout=` is a *per-operation* timeout: it bounds each individual
        recv, not the call. A server that dribbles bytes — or a stream that
        stalls between chunks — resets it on every read, so the call can hang
        forever while every single operation looks healthy.

        That is not hypothetical. A benchmark run sat on one ESTABLISHED
        connection for five and a half hours with 2.6 seconds of CPU and no
        output, and the 120-second timeout never fired once.

        The request runs on a daemon thread that the caller stops waiting for
        after `deadline`. The socket is left to the interpreter rather than
        force-closed, which leaks a thread on a genuine hang — acceptable,
        because the alternative is a run that never ends and a bill that never
        stops."""
        box: dict[str, object] = {}

        def call() -> None:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    box["data"] = resp.read()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        worker.join(self.deadline)
        if worker.is_alive():
            raise ProviderTimeout(
                f"{self.name}: no complete response within {self.deadline}s "
                "(the per-read timeout cannot catch a stalled stream)",
                provider=self.name,
            )
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box["data"]

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
                return json.loads(self._urlopen(req))
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
            except ProviderTimeout as e:
                # The deadline fired. Treat it like any other transient
                # failure: a stalled stream is usually the connection, not the
                # request, and the next attempt often succeeds in seconds.
                #
                # It escaped the retry loop before, because ProviderTimeout is
                # a RuntimeError and every handler here catches OSError
                # subclasses. So one stall failed the call outright, and a
                # provider that stalls intermittently on long prompts —
                # measured: 1.4s on three words, indefinite on a full rubric —
                # took down whole runs it should merely have slowed.
                last = e
                retry_after = None
            except json.JSONDecodeError as e:
                raise ProviderError(
                    f"{self.name}: response was not valid JSON for {model_id}: {e}",
                    provider=self.name,
                    model_id=model_id,
                ) from None
            except OSError as e:
                # The catch-all that was missing. urllib wraps most transport
                # problems in URLError, but a socket reset *after* the request
                # is sent surfaces as a bare ConnectionResetError, which is an
                # OSError and slipped straight through every handler above.
                #
                # It escaped as an unhandled exception and killed a 20-task
                # benchmark run partway through — a transient blip taking down
                # the exact long run the retry logic exists to protect. Treated
                # as an outage, because that is what it is.
                last = ProviderUnavailable(
                    f"{self.name}: transport failed for {model_id}: "
                    f"{type(e).__name__}: {e}",
                    provider=self.name,
                    model_id=model_id,
                )
                retry_after = None

            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, retry_after))

        assert last is not None  # only reachable after a retryable failure
        raise last


def _openai_cached_tokens(usage: dict) -> int:
    """Cached prompt tokens, across the dialects of "OpenAI-compatible".

    Two shapes are in use by vendors this adapter already serves:

    - OpenAI (and Google's compatibility layer): a nested
      `prompt_tokens_details.cached_tokens`.
    - DeepSeek: flat `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`.

    Checked in that order and independently, so a vendor that grows the other
    shape starts reporting without an adapter change. A vendor that reports
    neither yields 0, which is indistinguishable from a genuine cache miss —
    the conservative direction, since it prices those tokens at full rate.
    """
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens"):
        return int(details["cached_tokens"])
    if usage.get("prompt_cache_hit_tokens"):
        return int(usage["prompt_cache_hit_tokens"])
    return 0


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

    # Anthropic constrains output by forcing a tool call whose input schema is
    # the contract. Declared as a capability rather than assumed, so a caller
    # can feature-detect instead of catching a 400.
    schema_mode = "tool"

    def complete(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        *,
        schema: tuple[str, dict] | None = None,
    ) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY is not set", provider=self.name, model_id=model_id
            )
        payload: dict[str, object] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            name, definition = schema
            # The prompt is unchanged — the schema is an additional constraint
            # on the reply, not a replacement for the instructions. Keeping the
            # prose identical is what leaves the cacheable prefix intact.
            payload["tools"] = [{
                "name": name,
                "description": "Return the response in the format the prompt describes.",
                "input_schema": definition,
            }]
            payload["tool_choice"] = {"type": "tool", "name": name}
        body = json.dumps(payload).encode()
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
        blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in blocks if block.get("type") == "text"
        )
        if schema is not None:
            # Forced tool use returns the object under `input` rather than as
            # text. Re-serialised so everything downstream — repair, parsing,
            # the trace — sees exactly the shape it always saw.
            for block in blocks:
                if block.get("type") == "tool_use" and isinstance(
                    block.get("input"), dict
                ):
                    text = json.dumps(block["input"], ensure_ascii=False)
                    break
        usage = data.get("usage", {})
        if data.get("stop_reason") == "max_tokens":
            raise ProviderTruncated(
                f"{self.name}: {model_id} hit the {max_tokens}-token completion "
                f"budget; raise SessionConfig.max_tokens",
                provider=self.name,
                model_id=model_id,
            )
        # Anthropic already reports `input_tokens` as the uncached remainder,
        # so it needs no adjustment — the two cache counters sit beside it.
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0) or 0,
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0) or 0,
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
        chat_path: str = "/v1/chat/completions",
        schema_mode: str | None = "auto",
        **transport,
    ) -> None:
        super().__init__(**transport)
        self.api_key = api_key or os.environ.get(env_var, "")
        # Vendors document their base URL inconsistently: some give the origin,
        # some include `/v1`. This adapter appends `/v1/chat/completions`, so a
        # documented URL pasted verbatim produced `/v1/v1/chat/completions` and
        # a 404 that named neither the model nor the mistake. Accept both.
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.env_var = env_var
        # Not every OpenAI-compatible vendor serves the endpoint at
        # `/v1/chat/completions`. Google's compatibility layer, for one, lives
        # under `/v1beta/openai/chat/completions`, and hardcoding the path
        # would have made "OpenAI-compatible" mean "OpenAI-shaped body at
        # OpenAI's URL" — which is a much smaller claim than the one this
        # adapter makes.
        self.chat_path = "/" + chat_path.strip("/")
        if name:
            self.name = name
        # OpenAI's newer models reject `max_tokens`; most compatible vendors
        # only accept it. Pick by host, and let the caller override.
        self._max_tokens_param = max_tokens_param or (
            "max_completion_tokens" if "api.openai.com" in base_url else "max_tokens"
        )
        # Structured output support is not uniform across "OpenAI-compatible".
        # OpenAI itself takes a full JSON schema; DeepSeek takes JSON mode and
        # no schema. Guessing wrong is a 400 on every call, so the default is
        # inferred from the host and can always be set explicitly:
        #
        #   "json_schema"  — constrain generation to the schema
        #   "json_object"  — valid JSON guaranteed, shape is not
        #   None           — no structured support; prompt instructions only
        # "auto" infers from the host; an explicit None means the caller
        # knows this vendor has no structured support and wants the prompt-only
        # path. Those are different intentions and must not collapse into one.
        self.schema_mode = (
            ("json_schema" if "api.openai.com" in base_url else "json_object")
            if schema_mode == "auto"
            else schema_mode
        )

    def complete(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 2048,
        *,
        schema: tuple[str, dict] | None = None,
    ) -> Completion:
        if not self.api_key:
            raise ProviderConfigError(
                f"{self.env_var} is not set", provider=self.name, model_id=model_id
            )
        payload: dict[str, object] = {
            "model": model_id,
            self._max_tokens_param: max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None and self.schema_mode:
            name, definition = schema
            if self.schema_mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name, "schema": definition, "strict": True,
                    },
                }
            else:
                # JSON mode: the reply is guaranteed to parse, not to be the
                # right shape. Worth having anyway — it is the seat that
                # emitted an unterminated object, and this is exactly the
                # failure JSON mode removes. The schema check still runs.
                payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode()
        data = self._request(
            f"{self.base_url}{self.chat_path}",
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
        cache_read = _openai_cached_tokens(usage)
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        # `prompt_tokens` counts the cached part too, unlike Anthropic's
        # `input_tokens`. Subtract it so `input_tokens` means the same thing on
        # every adapter: what is billed at the full input rate. Clamped at zero
        # because a vendor whose two numbers disagree must not be allowed to
        # report negative full-price input and credit the session.
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=max(0, prompt_tokens - cache_read),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=cache_read,
            # Neither OpenAI nor DeepSeek bills a cache *write*: the prefix is
            # cached as a side effect of an ordinary request, at no premium.
            # Zero here is the real number, not a gap in the adapter.
            cache_write_tokens=0,
        )
