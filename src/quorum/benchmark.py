"""Quorum vs one model vs one model that critiques itself.

The claim under test is *not* "the council writes better answers" — this
project has been careful never to make that claim. The benchmark exists to
find out whether it is true, and to publish the result either way. The most
likely outcome, on the published evidence for multi-model deliberation, is
that the arms score close together and Quorum's value stays where the README
says it is: in the transcript, not the paragraph.

**Rubrics that don't hand the council the win.** This is the hard part, and
pretending otherwise would make the whole exercise theatre. It is very easy to
write a rubric that Quorum cannot lose — score "acknowledges uncertainty" and
"presents multiple perspectives" and a debate transcript wins by construction
against a single confident answer, having demonstrated nothing except that it
was scored on its own format.

Two defences, neither of them complete:

1. Every criterion is tagged `favours_deliberation`. Criteria that a
   deliberative process gets more easily are *marked as such*, and the harness
   reports the total both with and without them. If Quorum only leads on the
   tagged half, that is visible in the output rather than buried in the
   headline.
2. Criteria are about the *answer*, not the process. "Names a specific course
   of action" and "states what the recommendation depends on" are properties
   of a good answer to a judgement question whoever wrote it. "Shows its
   deliberation" is not a criterion at all.

**The three arms.** `single` is the strongest seat answering once — the thing
a user would actually do instead. `self_critique` is that same model asked to
attack its own answer and revise, which is the honest sceptic's question: is
the value in *multiple models*, or just in a second pass? An arm that Quorum
cannot beat is the most useful arm in the harness.

**Judging.** A model that took no part scores each answer against the rubric,
blind to which arm produced it, in a randomised order. Same reasoning as the
arbiter: an evaluator that knows which answer came from the council has a
reason to prefer it that has nothing to do with the answer.

**Mock runs are not results.** Run against `MockProvider`, every arm emits
canned text and the scores measure nothing. `BenchmarkReport.is_mock` is set
in that case and every rendering says so in the first line. The harness is
built to be exercised offline in CI; the numbers are only meaningful with real
providers and real keys.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .council import Council, Seat
from .costs import pick_baseline_seat
from .providers.base import ProviderError, ProviderPool
from .session import Session, SessionConfig
from .sheets import SheetError, extract_json

ARMS = ("quorum", "single", "self_critique")


# --------------------------------------------------------------------------
# tasks and rubrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    key: str
    text: str
    favours_deliberation: bool = False
    """True when a deliberative process gets this criterion more easily than a
    single answer does. Not a reason to drop the criterion — a good answer
    really should do these things — but a reason to report the score twice."""

    weight: float = 1.0


@dataclass(frozen=True)
class JudgmentTask:
    key: str
    prompt: str
    task_type: str
    complexity: float
    criteria: tuple[Criterion, ...]
    note: str = ""

    @property
    def neutral_criteria(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if not c.favours_deliberation)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

SINGLE_TEMPLATE = """Answer this question as well as you can, for a decision-maker
who has to act on it.

QUESTION
{task}

Give a specific recommendation, say what it depends on, and name the strongest
consideration against it. Prose, no headings, under 400 words.
"""

SELF_CRITIQUE_TEMPLATE = """Here is a question and your own first answer to it.

QUESTION
{task}

YOUR FIRST ANSWER
{answer}

Attack that answer as hard as you can: what is it assuming, where is it weakest,
what would change it? Then write your final answer, revised or unchanged.

Reply with a single JSON object: {{"final_answer": "..."}}
"""

JUDGE_TEMPLATE = """You are scoring answers to a hard judgement question. You did
not write any of them and you do not know who did.

QUESTION
{task}

{answers}

Score each answer against every criterion, 0 to 1, where 0 means the criterion
is not met at all and 1 means it is fully met. Judge the answer in front of you,
not the style it is written in. Length is not quality.

CRITERIA
{criteria}

Reply with a single JSON object mapping each answer label to its scores:
{{"A": {{"criterion_key": 0.0}}, "B": {{"criterion_key": 0.0}}}}
"""


@dataclass
class ArmResult:
    arm: str
    answer: str
    cost_est: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    error: str = ""
    session: Any = None

    def total(self, criteria: Iterable[Criterion]) -> float:
        criteria = list(criteria)
        weight = sum(c.weight for c in criteria)
        if not weight:
            return 0.0
        return sum(self.scores.get(c.key, 0.0) * c.weight for c in criteria) / weight


def run_single(task: JudgmentTask, seat: Seat, providers: ProviderPool) -> ArmResult:
    try:
        completion = providers.get(seat.provider).complete(
            seat.model_id, SINGLE_TEMPLATE.format(task=task.prompt), 1024
        )
    except ProviderError as exc:
        return ArmResult("single", "", error=str(exc))
    return ArmResult(
        "single",
        completion.text,
        cost_est=seat.cost.estimate(completion.input_tokens, completion.output_tokens),
        tokens_in=completion.input_tokens,
        tokens_out=completion.output_tokens,
    )


def run_self_critique(
    task: JudgmentTask, seat: Seat, providers: ProviderPool
) -> ArmResult:
    first = run_single(task, seat, providers)
    if first.error:
        return ArmResult("self_critique", "", error=first.error)
    prompt = SELF_CRITIQUE_TEMPLATE.format(task=task.prompt, answer=first.answer)
    try:
        completion = providers.get(seat.provider).complete(seat.model_id, prompt, 1024)
    except ProviderError as exc:
        return ArmResult("self_critique", "", error=str(exc))
    try:
        data = extract_json(completion.text, actor="self_critique")
        answer = data.get("final_answer", "") if isinstance(data, dict) else ""
    except SheetError:
        answer = completion.text
    return ArmResult(
        "self_critique",
        answer or completion.text,
        cost_est=first.cost_est
        + seat.cost.estimate(completion.input_tokens, completion.output_tokens),
        tokens_in=first.tokens_in + completion.input_tokens,
        tokens_out=first.tokens_out + completion.output_tokens,
    )


def run_quorum(
    task: JudgmentTask,
    council: Council,
    providers: ProviderPool,
    *,
    config: SessionConfig | None = None,
    session_id: str | None = None,
) -> ArmResult:
    result = Session(council, providers, config=config).run(
        task.prompt, session_id=session_id
    )
    if not result.verdict:
        return ArmResult(
            "quorum", "", cost_est=result.cost_est,
            error=result.failed_reason or "no verdict", session=result,
        )
    return ArmResult(
        "quorum",
        result.verdict.final_answer,
        cost_est=result.cost_est,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        session=result,
    )


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------


def judge(
    task: JudgmentTask,
    arms: list[ArmResult],
    judge_seat: Seat,
    providers: ProviderPool,
    *,
    shuffle_seed: str = "",
) -> None:
    """Score every arm in one call, blind to which arm is which.

    One call rather than three: a judge that sees the answers side by side
    scores them on a common scale, where three independent calls drift. The
    order is shuffled per task so a judge with a position bias cannot
    systematically favour the same arm.
    """
    scored = [a for a in arms if a.answer and not a.error]
    if not scored:
        return
    order = list(range(len(scored)))
    random.Random(f"{task.key}|{shuffle_seed}").shuffle(order)
    labelled = [(chr(ord("A") + i), scored[j]) for i, j in enumerate(order)]

    answers = "\n\n".join(
        f"--- Answer {label} ---\n{arm.answer}" for label, arm in labelled
    )
    criteria = "\n".join(f"  {c.key}: {c.text}" for c in task.criteria)
    prompt = JUDGE_TEMPLATE.format(task=task.prompt, answers=answers, criteria=criteria)

    try:
        completion = providers.get(judge_seat.provider).complete(
            judge_seat.model_id, prompt, 1024
        )
        data = extract_json(completion.text, actor="judge")
    except (ProviderError, SheetError) as exc:
        for arm in scored:
            arm.error = f"judging failed: {exc}"
        return
    if not isinstance(data, dict):
        for arm in scored:
            arm.error = "judging failed: reply was not an object"
        return

    valid = {c.key for c in task.criteria}
    for label, arm in labelled:
        raw = data.get(label, {})
        if not isinstance(raw, dict):
            arm.error = f"judge returned no scores for answer {label}"
            continue
        arm.scores = {
            key: min(1.0, max(0.0, float(value)))
            for key, value in raw.items()
            if key in valid and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    task: JudgmentTask
    arms: dict[str, ArmResult]

    def winner(self, criteria: Iterable[Criterion] | None = None) -> str:
        criteria = list(criteria if criteria is not None else self.task.criteria)
        scored = {
            name: arm.total(criteria)
            for name, arm in self.arms.items()
            if arm.answer and not arm.error
        }
        if not scored:
            return "none"
        best = max(scored.values())
        tied = [name for name, value in scored.items() if abs(value - best) < 1e-9]
        return tied[0] if len(tied) == 1 else "tie"


@dataclass
class BenchmarkReport:
    outcomes: list[TaskOutcome] = field(default_factory=list)
    is_mock: bool = False
    council_models: tuple[str, ...] = ()
    single_model: str = ""
    judge_model: str = ""

    def mean(self, arm: str, *, neutral_only: bool = False) -> float:
        values = []
        for outcome in self.outcomes:
            result = outcome.arms.get(arm)
            if not result or result.error or not result.answer:
                continue
            criteria = (
                outcome.task.neutral_criteria if neutral_only else outcome.task.criteria
            )
            values.append(result.total(criteria))
        return sum(values) / len(values) if values else 0.0

    def cost(self, arm: str) -> float:
        return sum(
            outcome.arms[arm].cost_est
            for outcome in self.outcomes
            if arm in outcome.arms
        )

    def wins(self, *, neutral_only: bool = False) -> dict[str, int]:
        tally = {arm: 0 for arm in ARMS}
        tally["tie"] = 0
        tally["none"] = 0
        for outcome in self.outcomes:
            criteria = (
                outcome.task.neutral_criteria if neutral_only else outcome.task.criteria
            )
            tally[outcome.winner(criteria)] += 1
        return tally

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_mock": self.is_mock,
            "tasks": len(self.outcomes),
            "council": list(self.council_models),
            "single_model": self.single_model,
            "judge_model": self.judge_model,
            "mean_score": {arm: round(self.mean(arm), 4) for arm in ARMS},
            "mean_score_neutral_criteria": {
                arm: round(self.mean(arm, neutral_only=True), 4) for arm in ARMS
            },
            "cost": {arm: round(self.cost(arm), 6) for arm in ARMS},
            "wins": self.wins(),
            "wins_neutral_criteria": self.wins(neutral_only=True),
        }

    def render_markdown(self) -> str:
        out: list[str] = ["# Quorum benchmark", ""]
        if self.is_mock:
            out += [
                "> **THIS IS A MOCK RUN. THESE ARE NOT RESULTS.**",
                ">",
                "> Every arm was answered by `MockProvider`, which emits canned text. "
                "The numbers below demonstrate that the harness runs end to end and "
                "measure nothing else. Real numbers require real providers and keys.",
                "",
            ]
        out += [
            f"- tasks: **{len(self.outcomes)}**",
            f"- council: {', '.join(self.council_models)}",
            f"- single-model arm: {self.single_model}",
            f"- judge: {self.judge_model}",
            "",
            "## Mean rubric score",
            "",
            "| arm | all criteria | neutral criteria only | total cost |",
            "| --- | --- | --- | --- |",
        ]
        for arm in ARMS:
            out.append(
                f"| `{arm}` | {self.mean(arm):.3f} | "
                f"{self.mean(arm, neutral_only=True):.3f} | ${self.cost(arm):.4f} |"
            )
        out += [
            "",
            "*Neutral criteria* exclude the ones tagged `favours_deliberation` — the "
            "ones a deliberative process gets more easily. If an arm leads on all "
            "criteria but not on neutral ones, it is winning on format rather than "
            "on substance, and that column is where you would see it.",
            "",
            "## Wins per task",
            "",
            "| arm | all criteria | neutral criteria only |",
            "| --- | --- | --- |",
        ]
        wins, neutral = self.wins(), self.wins(neutral_only=True)
        for arm in ARMS + ("tie", "none"):
            out.append(f"| `{arm}` | {wins.get(arm, 0)} | {neutral.get(arm, 0)} |")
        out += [
            "",
            "## Per task",
            "",
            "| task | quorum | single | self-critique | winner |",
            "| --- | --- | --- | --- | --- |",
        ]
        for outcome in self.outcomes:
            cells = []
            for arm in ARMS:
                result = outcome.arms.get(arm)
                cells.append(
                    "err" if (not result or result.error)
                    else f"{result.total(outcome.task.criteria):.2f}"
                )
            out.append(
                f"| `{outcome.task.key}` | " + " | ".join(cells)
                + f" | {outcome.winner()} |"
            )
        out.append("")
        return "\n".join(out)


def run_benchmark(
    tasks: Iterable[JudgmentTask],
    council: Council,
    providers: ProviderPool,
    judge_seat: Seat,
    *,
    single_seat: Seat | None = None,
    config: SessionConfig | None = None,
    is_mock: bool = False,
    on_task: Callable[[JudgmentTask], None] | None = None,
) -> BenchmarkReport:
    """Run every arm on every task and score them.

    `judge_seat` must not hold a seat on the council, for the same reason the
    arbiter must not: an evaluator that debated has a position to defend.
    """
    if judge_seat.model_id in [s.model_id for s in council.seats()]:
        raise ValueError(
            f"judge {judge_seat.model_id!r} sits on the council; "
            "an evaluator that took part is not an evaluator"
        )
    single_seat = single_seat or pick_baseline_seat(council)
    report = BenchmarkReport(
        is_mock=is_mock,
        council_models=tuple(s.model_id for s in council.students),
        single_model=single_seat.model_id,
        judge_model=judge_seat.model_id,
    )

    for task in tasks:
        if on_task:
            on_task(task)
        arms = {
            "quorum": run_quorum(
                task, council, providers, config=config, session_id=f"bench-{task.key}"
            ),
            "single": run_single(task, single_seat, providers),
            "self_critique": run_self_critique(task, single_seat, providers),
        }
        judge(task, list(arms.values()), judge_seat, providers, shuffle_seed=task.key)
        report.outcomes.append(TaskOutcome(task=task, arms=arms))

    return report


def load_tasks(path: str) -> list[JudgmentTask]:
    """Load tasks and rubrics from JSON, so the set can be edited without code."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        JudgmentTask(
            key=entry["key"],
            prompt=entry["prompt"],
            task_type=entry.get("task_type", "strategy"),
            complexity=float(entry.get("complexity", 0.8)),
            note=entry.get("note", ""),
            criteria=tuple(
                Criterion(
                    key=c["key"],
                    text=c["text"],
                    favours_deliberation=bool(c.get("favours_deliberation", False)),
                    weight=float(c.get("weight", 1.0)),
                )
                for c in entry["criteria"]
            ),
        )
        for entry in data["tasks"]
    ]


# --------------------------------------------------------------------------
# offline arms
# --------------------------------------------------------------------------


class BenchmarkMockProvider:
    """Plays the benchmark's three extra prompt shapes, delegates the rest.

    Kept here rather than folded into `MockProvider` so the core offline
    participant stays a participant: the benchmark's templates live in this
    module, and teaching `providers.base` about them would invert the
    dependency and put evaluation logic inside the thing being evaluated.

    Its scores are a hash. They are not opinions, and `is_mock` exists so no
    rendering of them can be mistaken for one.
    """

    def __init__(self, inner: Any, name: str = "mock") -> None:
        self.inner = inner
        self.name = name
        self.calls: list[tuple[str, str]] = []  # (model_id, prompt), for assertions

    def complete(self, model_id: str, prompt: str, max_tokens: int = 2048):
        from .providers.base import Completion

        self.calls.append((model_id, prompt))

        if prompt.startswith(JUDGE_TEMPLATE.split("\n", 1)[0]):
            text = self._judge(prompt)
        elif "YOUR FIRST ANSWER" in prompt:
            text = json.dumps(
                {"final_answer": f"[{model_id}] revised after self-critique: the "
                                 "recommendation stands, narrowed to the near term."}
            )
        elif prompt.startswith("Answer this question as well as you can"):
            text = (
                f"[{model_id}] A specific recommendation, what it depends on, and the "
                "strongest consideration against it."
            )
        else:
            return self.inner.complete(model_id, prompt, max_tokens)
        return Completion(
            text=text,
            model_id=model_id,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )

    @staticmethod
    def _judge(prompt: str) -> str:
        import hashlib
        import re

        labels = re.findall(r"^--- Answer ([A-Z]) ---$", prompt, re.MULTILINE)
        keys = re.findall(r"^  (\w+): ", prompt, re.MULTILINE)
        scores = {}
        for label in labels:
            digest = hashlib.blake2b(
                f"{label}|{prompt[:200]}".encode(), digest_size=16
            ).digest()
            scores[label] = {
                key: round(0.4 + (digest[i % len(digest)] % 61) / 100, 2)
                for i, key in enumerate(keys)
            }
        return json.dumps(scores)
