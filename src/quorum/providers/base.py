"""Provider abstraction — the pattern imported from Switchboard, not the code.

A Provider is anything that can complete a prompt on a given model. Keeping
the surface to one method means a new vendor is a ~30 line adapter, and it
means the entire protocol runs offline against `MockProvider`, which is how
the test suite and the demo stay free and deterministic.

Quorum defines this itself rather than depending on `switchboard` because the
two projects should be installable apart. The shape is deliberately identical,
so a Switchboard provider satisfies Quorum's protocol structurally and a
`quorum` council can be driven by adapters written for either.

`MockProvider` carries more weight here than its Switchboard counterpart. A
deliberation is only interesting if the participants *disagree*, so the mock
does not echo prompts — it plays the protocol: it takes a persona, writes a
schema-valid answer sheet arguing that persona's line, objects to specific
numbered claims in the sheets it is shown, and revises (or refuses to revise)
its own sheet. That is what makes the offline demo a real session with real
diffs rather than a plumbing check.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..prompts import (
    CRITIQUE_PROMPT_HEADER,
    PROBE_PROMPT_HEADER,
    REVISION_PROMPT_HEADER,
    SHEET_PROMPT_HEADER,
    VERDICT_PROMPT_HEADER,
)


class ProviderError(RuntimeError):
    """A provider call failed. The session marks the participant absent."""

    def __init__(self, message: str, *, provider: str = "", model_id: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model_id = model_id


class ProviderTimeout(ProviderError):
    """The provider did not respond in time."""


class ProviderRateLimited(ProviderError):
    """The provider rejected the call for rate/quota reasons (429)."""


class ProviderUnavailable(ProviderError):
    """The provider is down or overloaded (5xx, connection failure)."""


class ProviderConfigError(ProviderError):
    """Missing key or bad configuration. Not retryable."""


@dataclass(frozen=True)
class Completion:
    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    name: str

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        ...


class ProviderPool:
    """Maps provider names (as named on a `Seat`) to Provider instances."""

    def __init__(self, providers: list[Provider]) -> None:
        self._providers = {p.name: p for p in providers}

    def get(self, name: str) -> Provider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(
                f"no provider registered for {name!r}; available: {sorted(self._providers)}"
            ) from None

    def has(self, name: str) -> bool:
        return name in self._providers

    def names(self) -> list[str]:
        return sorted(self._providers)


# --------------------------------------------------------------------------
# offline personas
# --------------------------------------------------------------------------

_PERSONAS: tuple[dict[str, object], ...] = (
    {
        "position": "Yes, and the main risk is manageable with staged rollout",
        "claims": [
            "The upside is large enough that delay costs more than a partial failure would",
            "The failure modes are reversible within one quarter",
            "Staging the rollout converts an all-or-nothing bet into three cheap experiments",
        ],
        "assumptions": ["The team can staff a staged rollout without pausing other work"],
        "falsifiers": ["Evidence that a partial rollout cannot be rolled back cleanly"],
        "confidence": 0.72,
        "objection": (
            "This claim treats the downside as bounded without saying what bounds it; "
            "if the constraint is reputational rather than technical, staging does not "
            "reduce the exposure, it only spreads it over more weeks"
        ),
    },
    {
        "position": "No, the expected cost outweighs the expected benefit at current scale",
        "claims": [
            "The benefit scales with volume the organisation does not yet have",
            "The switching cost is paid up front and in full regardless of outcome",
            "A cheaper reversible experiment answers the same question within a month",
        ],
        "assumptions": ["Current volume is representative of the next two quarters"],
        "falsifiers": ["A credible forecast showing volume doubling inside six months"],
        "confidence": 0.64,
        "objection": (
            "This claim assumes the cost curve is linear when the argument only holds if "
            "it is; a step change at the point of adoption would invert the conclusion, "
            "and nothing in the sheet rules that out"
        ),
    },
    {
        "position": "It depends on whether the constraint is capacity or coordination",
        "claims": [
            "If the binding constraint is coordination, adding capacity makes it worse",
            "The question is unanswerable without measuring where the queue actually forms",
            "Two weeks of instrumentation costs less than either decision made blind",
        ],
        "assumptions": ["The bottleneck can in fact be measured within two weeks"],
        "falsifiers": ["Instrumentation showing a single unambiguous capacity bottleneck"],
        "confidence": 0.58,
        "objection": (
            "Deferring to measurement is not a neutral act; the two weeks spent measuring "
            "are themselves a decision with a cost, and this claim prices that cost at "
            "zero without arguing for it"
        ),
    },
)


def _persona_index(model_id: str) -> int:
    digest = hashlib.blake2b(model_id.encode("utf-8"), digest_size=4).digest()
    return digest[0] % len(_PERSONAS)


_SHEET_BLOCK = re.compile(r"^--- Sheet ([A-Z]) ---$", re.MULTILINE)
_CLAIM_LINE = re.compile(r"^\s{2}(\d+)\.\s", re.MULTILINE)
_CRITIC_LINE = re.compile(r"^--- Critic ([A-Z]) on your claim (\d+) ---$", re.MULTILINE)
_STUDENT_BLOCK = re.compile(r"^--- (Student \d+) ---$", re.MULTILINE)
_ROSTER_ENTRY = re.compile(r"^  - (\S+)$", re.MULTILINE)
_OWN_SHEET = re.compile(r"YOUR CURRENT SHEET \(JSON\)\s*```json\s*(\{.*?\})\s*```", re.DOTALL)
_QUESTION = re.compile(r"^QUESTION(?: UNDER DISCUSSION)?\s*\n(.+?)\n\n", re.DOTALL | re.MULTILINE)


def _topic(prompt: str) -> str:
    """A short, punctuation-free slug of the question, for readable demos.

    Takes the *last* sentence: a question posed to a council usually opens
    with context and closes with the actual ask, and a slug cut from the
    opening reads like a non sequitur in the demo output.
    """
    match = _QUESTION.search(prompt)
    if not match:
        return "the question"
    text = re.sub(r"\s+", " ", match.group(1)).strip()
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    if sentences:
        text = sentences[-1]
    text = re.sub(r"[.?!]+", "", text)
    if len(text) > 70:
        text = text[:70].rsplit(" ", 1)[0]
    return text.lower() or "the question"


class MockProvider:
    """Deterministic offline participant that actually plays the protocol.

    Which round it is answering is decided by the `*_PROMPT_HEADER` sentinels
    — real lines of the real prompts, so they cannot drift out of sync the way
    a test-only marker would.

    Personas are assigned per model id, so a council of three distinct models
    reliably produces three distinct opening positions and therefore real
    objections, real diffs and a non-empty minority report. Without that, an
    offline session would reach unanimous agreement in round one and prove
    nothing about a protocol whose entire purpose is surfacing disagreement.
    """

    name = "mock"

    def __init__(self, personas: dict[str, int] | None = None) -> None:
        self.personas = dict(personas or {})
        self.calls: list[tuple[str, str]] = []

    # -- round dispatch ----------------------------------------------------

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        self.calls.append((model_id, prompt))
        if SHEET_PROMPT_HEADER in prompt:
            text = self._sheet(model_id, prompt)
        elif CRITIQUE_PROMPT_HEADER in prompt:
            text = self._critique(model_id, prompt)
        elif REVISION_PROMPT_HEADER in prompt:
            text = self._revision(model_id, prompt)
        elif VERDICT_PROMPT_HEADER in prompt:
            text = self._verdict(model_id, prompt)
        elif PROBE_PROMPT_HEADER in prompt:
            text = self._probe(model_id, prompt)
        else:  # pragma: no cover - the engine only sends the four rounds
            text = json.dumps({"note": f"[{model_id}] unrecognised prompt"})
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )

    def _persona(self, model_id: str) -> dict[str, object]:
        index = self.personas.get(model_id, _persona_index(model_id))
        return _PERSONAS[index % len(_PERSONAS)]

    # -- rounds ------------------------------------------------------------

    def _sheet(self, model_id: str, prompt: str) -> str:
        persona = self._persona(model_id)
        topic = _topic(prompt)
        claims = list(persona["claims"])  # type: ignore[arg-type]
        return json.dumps(
            {
                "position": f"{persona['position']}, on {topic}",
                "claims": [{"n": i, "text": t} for i, t in enumerate(claims, start=1)],
                "assumptions": list(persona["assumptions"]),  # type: ignore[arg-type]
                "would_change_my_mind": list(persona["falsifiers"]),  # type: ignore[arg-type]
                "confidence": persona["confidence"],
                "nuance": "",
            }
        )

    def _critique(self, model_id: str, prompt: str) -> str:
        persona = self._persona(model_id)
        objections = []
        blocks = list(_SHEET_BLOCK.finditer(prompt))
        for index, match in enumerate(blocks):
            label = match.group(1)
            end = blocks[index + 1].start() if index + 1 < len(blocks) else len(prompt)
            body = prompt[match.end() : end]
            claim_numbers = [int(n) for n in _CLAIM_LINE.findall(body)]
            if not claim_numbers:  # pragma: no cover - sheets always carry claims
                continue
            digest = hashlib.blake2b(
                f"{model_id}|{label}".encode("utf-8"), digest_size=4
            ).digest()
            target = claim_numbers[digest[0] % len(claim_numbers)]
            objections.append(
                {
                    "sheet": label,
                    "claim_n": target,
                    "argument": f"On claim {target}: {persona['objection']}",
                }
            )
        return json.dumps({"objections": objections})

    def _revision(self, model_id: str, prompt: str) -> str:
        match = _OWN_SHEET.search(prompt)
        if not match:  # pragma: no cover - the engine always embeds the sheet
            raise ProviderError("revision prompt carried no sheet", model_id=model_id)
        sheet = json.loads(match.group(1))
        refs = [(c, int(n)) for c, n in _CRITIC_LINE.findall(prompt)]

        # Whether this persona moves is fixed per model, so a council produces
        # a mix of movers and holdouts — a session where everyone changes and
        # one where nobody does are both protocol failures, and the offline
        # demo should show neither. One mover in three puts the demo's
        # position-change rate inside the band the spec calls healthy.
        index = self.personas.get(model_id, _persona_index(model_id))
        moves = index % len(_PERSONAS) == 0 and bool(refs)
        claims = [dict(c) for c in sheet["claims"]]

        if moves:
            sheet["position"] = (
                f"On reflection, {sheet['position'][0].lower()}{sheet['position'][1:]}"
                " — but only under the narrower reading below"
            )
            if len(claims) > 1:
                claims.pop()
            claims.append(
                {
                    "n": 0,
                    "text": "The objection about the load-bearing assumption stands and "
                    "narrows the claim to the near term",
                }
            )
            sheet["confidence"] = round(max(0.0, float(sheet["confidence"]) - 0.15), 2)
        else:
            claims[0]["text"] = claims[0]["text"] + ", though only where the base rate holds"
            sheet["confidence"] = round(max(0.0, float(sheet["confidence"]) - 0.05), 2)

        for i, claim in enumerate(claims, start=1):
            claim["n"] = i
        sheet["claims"] = claims
        sheet["changed_position"] = moves
        sheet["because"] = (
            [{"critic": refs[0][0], "claim_n": refs[0][1]}] if moves and refs else []
        )
        sheet.pop("compliance_warnings", None)
        return json.dumps(sheet)

    def _verdict(self, model_id: str, prompt: str) -> str:
        students = _STUDENT_BLOCK.findall(prompt)
        topic = _topic(prompt)
        minority = []
        if len(students) > 1:
            minority.append(
                {
                    "source": students[-1],
                    "kind": "objection",
                    "substance": "The dissent that the question is unanswerable without "
                    "measuring where the constraint actually binds was not resolved, "
                    "only outvoted",
                }
            )
        return json.dumps(
            {
                "final_answer": (
                    f"The council did not converge on {topic}. The majority reading is "
                    "conditional: proceed only in a staged form that can be reversed "
                    "within a quarter, and only after the binding constraint is measured."
                ),
                "confidence_note": (
                    "Moderate and contested. Some participants moved under objection and "
                    "at least one held; the disagreement is about which constraint binds, "
                    "not about the evidence."
                ),
                "minority_report": minority,
            }
        )


    def _probe(self, model_id: str, prompt: str) -> str:
        """Guess authorship, deterministically and without cheating.

        The mock genuinely cannot tell the sheets apart — its personas are
        assigned by model id, and nothing in the rendered sheet carries that
        id — so it permutes the roster from a hash of the sheet text. Across
        sessions that lands near chance, which is the *correct* offline
        result: the probe harness must be exercisable end to end without the
        mock manufacturing either a clean bill of health or a fake leak.
        """
        labels = _SHEET_BLOCK.findall(prompt)
        roster = _ROSTER_ENTRY.findall(prompt)
        if not labels or not roster:  # pragma: no cover - the prompt always has both
            return json.dumps({"guesses": {}})
        # Hash the WHOLE prompt, not its tail: the tail is the roster and the
        # JSON rule, identical in every session, which pinned the permutation
        # to one value and made the mock's accuracy land systematically below
        # chance — an artifact that reads as "the blinding is better than
        # random", which is not a thing.
        digest = hashlib.blake2b(prompt.encode(), digest_size=8).digest()
        order = list(range(len(roster)))
        # Fisher-Yates driven by the digest: a permutation, not a fixed shift,
        # so the mock cannot accidentally be systematically right or wrong.
        for i in range(len(order) - 1, 0, -1):
            j = digest[i % len(digest)] % (i + 1)
            order[i], order[j] = order[j], order[i]
        return json.dumps(
            {"guesses": {label: roster[order[i % len(order)]]
                         for i, label in enumerate(labels)}}
        )


def mock_pool(providers: list[str] | object, personas: dict[str, int] | None = None) -> ProviderPool:
    """A ProviderPool of offline mocks, one per provider name.

    A real council names real vendors, so `ProviderPool([MockProvider()])`
    fails on the first seat whose provider is not `mock`. This builds a
    stand-in for every provider a council mentions, which is what lets the
    whole protocol be exercised against a realistic lineup — three labs, one
    arbiter — without a single API key.

    Accepts a list of provider names or a `Council`.
    """
    if hasattr(providers, "seats"):
        council = providers
        names = sorted({seat.provider for seat in council.seats()})  # type: ignore[union-attr]
        if personas is None:
            personas = {
                seat.model_id: index
                for index, seat in enumerate(council.students)  # type: ignore[union-attr]
            }
    else:
        names = sorted(set(providers))  # type: ignore[arg-type]

    pool = []
    for name in names:
        provider = MockProvider(personas=personas)
        provider.name = name  # shadow the class attribute per instance
        pool.append(provider)
    return ProviderPool(pool)


class ScriptedProvider:
    """Offline provider that replays a queued script per model.

    MockProvider always plays the protocol correctly, which means tests
    written against it verify the happy path and nothing else. The interesting
    cases are the ugly ones: a sheet with six claims, a critique that is
    politely agreeable, a model that declares it changed its mind while
    submitting a byte-identical position. ScriptedProvider puts exactly that
    text on the wire.

        provider = ScriptedProvider({
            "model-a": ['{"position": "...", ...}', '{"objections": [...]}'],
            "model-b": [ProviderUnavailable("503")],
        })

    Queue entries are either a string (returned as completion text) or an
    exception instance (raised). The last entry repeats once exhausted, so a
    short script need not anticipate every call.
    """

    def __init__(
        self,
        script: dict[str, list[str | Exception]] | None = None,
        name: str = "mock",
        default: str | Exception | None = None,
    ) -> None:
        self.name = name
        self._script: dict[str, list[str | Exception]] = {
            k: list(v) for k, v in (script or {}).items()
        }
        self._default = default
        self.calls: list[tuple[str, str]] = []  # (model_id, prompt), for assertions

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        self.calls.append((model_id, prompt))
        queue = self._script.get(model_id)
        if queue:
            item = queue.pop(0) if len(queue) > 1 else queue[0]
        elif self._default is not None:
            item = self._default
        else:
            raise ProviderError(
                f"ScriptedProvider has no script for {model_id!r}; "
                f"scripted models: {sorted(self._script)}",
                provider=self.name,
                model_id=model_id,
            )
        if isinstance(item, Exception):
            raise item
        return Completion(
            text=item,
            model_id=model_id,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(item) // 4),
        )


@dataclass
class FlakyProvider:
    """Wraps a provider and fails its first `fail_times` calls.

    Used to prove the fail-closed path does what it claims without reaching
    the network.
    """

    inner: Provider
    fail_times: int = 1
    error: Exception = field(default_factory=lambda: ProviderUnavailable("injected outage"))
    calls: int = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.inner.name

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048) -> Completion:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return self.inner.complete(model_id, prompt, max_tokens)
