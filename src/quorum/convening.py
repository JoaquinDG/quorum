"""When a question is worth a council, and when it very much is not.

Quorum is convened, not default. A session costs several times a single
answer, so the interesting engineering problem is not "can three models
debate" but "how often should they". The target is under 10% of a mixed
workload — most work is extraction, summarisation and code, and none of it
benefits from a debate.

**The gates fail closed, and that is the opposite of Switchboard's router on
purpose.** Switchboard's gates degrade *upward*: when nothing qualifies, it
escalates to the most capable tier, because the cost of an unqualified model
answering is a bad answer. Here the cost of a wrong call is a 7x bill, so an
unrecognised task type or a missing complexity score means *don't convene*.
The default answer is no; convening has to be argued for.

**A council cannot help a question with a checkable answer.** If the output
can be verified — extracted fields, a translation, a passing test — the right
tool is one model plus a verifier, which is exactly what Switchboard already
does. Deliberation is for questions where the disagreement *is* the
information, and running it on a factual task buys three confident answers and
no signal.

The signal extraction here is keyword heuristics over the prompt, and that is
the weakest part of the module. It is honest about being a placeholder: real
deployments should classify with a small model (Switchboard's triage does
exactly this) and pass `task_type` and `complexity` in explicitly, which the
API takes as authoritative whenever they are supplied. The heuristics exist so
the rule is testable offline and so a caller that supplies nothing still gets
a conservative answer rather than an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

JUDGMENT_TYPES = frozenset(
    {"strategy", "pricing", "architecture", "risk", "planning", "reasoning", "policy"}
)
"""Task types where the answer is a judgement call: no ground truth, real
trade-offs, and reasonable models can disagree for reasons worth reading."""

VERIFIABLE_TYPES = frozenset(
    {
        "extraction",
        "summarization",
        "classification",
        "translation",
        "code_generation",
        "code_review",
        "formatting",
        "arithmetic",
        "retrieval",
        "qa",
    }
)
"""Task types with a checkable answer. A council adds cost and no signal."""

# Ambiguity is detected as two *families* rather than a list of phrasings.
#
# The list-of-phrasings version failed twice, the same way both times. The
# convening eval caught "we can either hire two seniors or four juniors" (never
# says "should we"), and then 16 of the 20 benchmark tasks scored zero
# ambiguity because they open "Do we rebuild…" rather than "Should we
# rebuild…". A whitelist of verbs measures dialect, not ambiguity, and patching
# in one more verb each time it misses is not a fix — it is the same bug
# rescheduled.
#
# The families generalise instead: a question is ambiguous when it *asks what
# to do* (a deliberative modal aimed at the reader) or when it *puts
# alternatives on the table* (a choice, however phrased).
_DELIBERATIVE = (
    r"\b(should|shall|do|does|can|could|would|must|ought)\s+(we|i|you|they|it)\b|"
    r"\bwhether\b|\bwhich\b|\bwhat should\b|\bhow (should|do|fast)\b|"
    r"\bmake the case\b|\bargue\b"
)
_CHOICE = (
    # "or" inside a question is offering alternatives; "or" inside a statement
    # is usually a list ("positive, negative or mixed").
    r"\bor\b(?=[^.?!]*\?)|\beither\b[^.?!]{0,120}\bor\b|\bversus\b|\bvs\.?\b|"
    r"\btrade[- ]?offs?\b|\bpros and cons\b|\bdecide between\b|\bworth (it|the)\b|"
    r"\bis it better\b|\bwhich is (the )?better\b|\bbetter (bet|choice|option|use of)\b"
)
_STAKES = (
    r"\bstrateg|\bpricing\b|\bprice\b|\bmigrat|\bre-?architect|\brebuild\b|"
    r"\bacquisition\b|\blayoff|\bhead ?count\b|\bruntime risk\b|\bcompliance\b|"
    r"\bbet the\b|\broadmap\b|\binvest|\bbudget\b|\bcontract\b|\bhiring\b|"
    r"\bshut down\b|\bdeprecat|\bone[- ]way door\b|\birreversible\b"
)
_VERIFIABLE = (
    r"\bextract\b|\bsummari[sz]e\b|\btranslate\b|\bconvert\b|\bparse\b|\bcount\b|"
    r"\blist all\b|\bformat\b|\brename\b|\bfix (this|the) (bug|test|typo)\b|"
    r"\bwrite a (function|script|query|regex|test)\b|\bwhat is the\b|\bcompute\b|"
    r"\bdeserial|\bvalidate the schema\b|\brefactor this function\b"
)


@dataclass(frozen=True)
class Task:
    """A question, optionally pre-classified.

    `task_type` and `complexity` are authoritative when supplied — a caller
    with a real classifier should not have its judgement overridden by keyword
    matching. Shape-compatible with Switchboard's `Task` so a router can hand
    one straight over.
    """

    prompt: str
    task_type: str = ""
    complexity: float | None = None

    def __post_init__(self) -> None:
        if self.complexity is not None and not 0.0 <= self.complexity <= 1.0:
            raise ValueError(f"complexity {self.complexity} outside [0, 1]")


@dataclass(frozen=True)
class ConveningPolicy:
    enabled: bool = True
    min_complexity: float = 0.6
    min_score: float = 0.60
    judgment_types: frozenset[str] = JUDGMENT_TYPES
    verifiable_types: frozenset[str] = VERIFIABLE_TYPES
    weights: dict[str, float] = field(
        default_factory=lambda: {"complexity": 0.45, "ambiguity": 0.35, "stakes": 0.20}
    )
    """Used when the task arrived unclassified and the signals had to be
    guessed from the prompt. Keyword markers carry most of the weight because
    the complexity number is itself a guess."""

    classified_weights: dict[str, float] = field(
        default_factory=lambda: {"complexity": 0.70, "ambiguity": 0.20, "stakes": 0.10}
    )
    """Used when the caller supplied `task_type` and `complexity`.

    A caller that has classified the task has already done the work the
    keyword markers exist to approximate, and the module's stated contract is
    that a supplied classification is authoritative. Scoring it under the same
    weights as a guess quietly overrides it: 16 of the 20 benchmark tasks —
    hand-written judgement calls, explicitly typed and scored 0.75-0.9 —
    were declined because their phrasing missed a regex. When the answer is
    supplied, trust it and let the markers adjust rather than decide."""
    force: bool = False
    """Convene regardless. For the benchmark harness, which needs Quorum to run
    on tasks the rule would decline, and for a caller that has already decided.
    Never a default — a policy that always says yes is not a policy."""


CONVENE_ALWAYS = ConveningPolicy(force=True)
CONVENE_NEVER = ConveningPolicy(enabled=False)
CONVENE_CONSERVATIVE = ConveningPolicy(min_complexity=0.7, min_score=0.72)
CONVENE_DEFAULT = ConveningPolicy()

CONVENING_PRESETS = {
    "default": CONVENE_DEFAULT,
    "conservative": CONVENE_CONSERVATIVE,
    "always": CONVENE_ALWAYS,
    "never": CONVENE_NEVER,
}


@dataclass(frozen=True)
class TaskSignals:
    complexity: float
    task_type: str
    ambiguity: float
    stakes: float
    verifiable: bool
    inferred: tuple[str, ...] = ()
    """Which fields were guessed rather than supplied. Carried onto the
    decision so a low-confidence call is visibly low-confidence."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": round(self.complexity, 3),
            "task_type": self.task_type,
            "ambiguity": round(self.ambiguity, 3),
            "stakes": round(self.stakes, 3),
            "verifiable": self.verifiable,
            "inferred": list(self.inferred),
        }


def _hits(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.IGNORECASE))


def _saturate(count: int, full: int = 2) -> float:
    """Diminishing returns: one marker is a signal, five is not five signals."""
    return min(1.0, count / full)


def signals(task: Task) -> TaskSignals:
    """Extract convening signals, preferring what the caller supplied."""
    text = task.prompt
    inferred: list[str] = []

    ambiguity = _saturate(_hits(_DELIBERATIVE, text) + _hits(_CHOICE, text))
    stakes = _saturate(_hits(_STAKES, text))
    verifiable_markers = _hits(_VERIFIABLE, text)

    task_type = task.task_type.strip().lower()
    if not task_type:
        inferred.append("task_type")
        if verifiable_markers:
            task_type = "extraction"
        elif ambiguity and stakes:
            task_type = "strategy"
        else:
            task_type = ""

    complexity = task.complexity
    if complexity is None:
        inferred.append("complexity")
        # Length is a weak proxy, so it is capped low and the ambiguity and
        # stakes markers carry most of the weight.
        length = min(1.0, len(text) / 600)
        complexity = min(1.0, 0.25 * length + 0.45 * ambiguity + 0.30 * stakes)

    verifiable = task_type in VERIFIABLE_TYPES or (
        verifiable_markers > 0 and task_type not in JUDGMENT_TYPES
    )

    return TaskSignals(
        complexity=complexity,
        task_type=task_type,
        ambiguity=ambiguity,
        stakes=stakes,
        verifiable=verifiable,
        inferred=tuple(inferred),
    )


@dataclass(frozen=True)
class ConveneDecision:
    convene: bool
    score: float
    reason: str
    gates: tuple[str, ...]
    signals: TaskSignals

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.convene

    def to_dict(self) -> dict[str, Any]:
        return {
            "convene": self.convene,
            "score": round(self.score, 3),
            "reason": self.reason,
            "gates": list(self.gates),
            "signals": self.signals.to_dict(),
        }


def should_convene(
    task: Task | str, policy: ConveningPolicy | None = None
) -> ConveneDecision:
    """Decide whether this question earns a council.

    Returns a decision with its reasoning attached, never a bare bool: the
    point of a convening rule is that somebody can look at a month of them and
    argue with the calls, which requires the calls to have said why.
    """
    if isinstance(task, str):
        task = Task(prompt=task)
    policy = policy or CONVENE_DEFAULT
    sig = signals(task)
    gates: list[str] = []

    def no(reason: str) -> ConveneDecision:
        return ConveneDecision(False, 0.0, reason, tuple(gates), sig)

    if policy.force:
        gates.append("force")
        return ConveneDecision(
            True, 1.0, "policy forces convening (benchmark or explicit caller)",
            tuple(gates), sig,
        )
    if not policy.enabled:
        gates.append("disabled")
        return no("convening is disabled by policy")

    if sig.verifiable:
        gates.append("verifiable")
        return no(
            f"task_type={sig.task_type or 'unknown'} has a checkable answer; "
            "one model plus a verifier is cheaper and better suited"
        )

    if sig.task_type and sig.task_type not in policy.judgment_types:
        gates.append("not_a_judgment_type")
        return no(
            f"task_type={sig.task_type!r} is not a judgement call; "
            f"the council is for {sorted(policy.judgment_types)}"
        )
    if not sig.task_type:
        # Fail closed. An unclassified task under Switchboard's router
        # escalates upward because a bad answer is the risk; here the risk is
        # a 7x bill on a task nobody could even name.
        gates.append("unclassified")
        return no("task type could not be determined; not convening on a guess")

    if sig.complexity < policy.min_complexity:
        gates.append("complexity")
        return no(
            f"complexity {sig.complexity:.2f} is below the bar of "
            f"{policy.min_complexity:.2f}; a single model handles this"
        )

    weights = (
        policy.weights if "complexity" in sig.inferred else policy.classified_weights
    )
    score = (
        weights["complexity"] * sig.complexity
        + weights["ambiguity"] * sig.ambiguity
        + weights["stakes"] * sig.stakes
    ) / sum(weights.values())

    if score < policy.min_score:
        gates.append("score")
        return no(
            f"score {score:.2f} below {policy.min_score:.2f} "
            f"(complexity {sig.complexity:.2f}, ambiguity {sig.ambiguity:.2f}, "
            f"stakes {sig.stakes:.2f})"
        )

    caveat = (
        f" (inferred: {', '.join(sig.inferred)})" if sig.inferred else ""
    )
    return ConveneDecision(
        True,
        score,
        f"score {score:.2f} clears {policy.min_score:.2f} on a "
        f"{sig.task_type} question with ambiguity {sig.ambiguity:.2f} "
        f"and stakes {sig.stakes:.2f}{caveat}",
        tuple(gates),
        sig,
    )
