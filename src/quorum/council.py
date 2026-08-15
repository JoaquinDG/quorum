"""Who is in the room, and what they cost.

A council is three students and an arbiter. Both numbers are deliberate.

**Three students, not five.** Diversity of priors is what the protocol is
buying, and it comes from distinct model *families*, not from headcount — a
fourth and fifth seat multiply cost linearly while adding little that the
first three did not already cover. The cap is enforced rather than advised,
because "just add one more model" is the easiest way to turn a 5x session into
a 10x one without anybody deciding to.

**An arbiter that did not debate.** No-self-grading is enforced here, in the
type, rather than by convention in the engine: a `Council` whose arbiter also
holds a student seat cannot be constructed. Switchboard learned this the hard
way in its auditor — independence that lives in a code path is independence
that a later refactor can quietly remove.

Prices are per million tokens and are the caller's to maintain, exactly as
Switchboard's catalog is. The demo council's numbers are synthetic and say so;
a cost figure copied from a README is a cost figure that was true once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIN_STUDENTS = 2
MAX_STUDENTS = 5
"""The spec set this at 3 and called more a v1 non-goal, on the grounds that
larger councils "multiply cost and add little diversity beyond 3 distinct
model families". Both halves have since been measured, and they came apart.

*Cost multiplies* — confirmed, though not for the stated reason. Objections
scale n(n-1) and the arbiter reads every one of them, so a fourth student
inflates the line that is already ~69% of the bill. Four students costs
roughly 1.7x a session, five roughly 2.4x.

*Diversity saturates at three* — this depends entirely on something the spec
did not distinguish: whether the fourth seat is a fourth **lab** or a second
model from a lab already seated. The second case is close to the worst
possible trade, buying correlated priors at 1.7x. The first is the thing the
protocol is actually for.

So the cap moves to 5 to make the question testable, and the default council
stays at 3. Anything above 3 raises a warning naming the cost, and any lab
holding two seats raises another — because the number that matters is distinct
labs, not headcount, and the code should say so rather than assume it."""


@dataclass(frozen=True)
class ModelCost:
    """USD per million tokens. Zero means "unpriced", not "free"."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def estimate(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok
        ) / 1_000_000

    @property
    def priced(self) -> bool:
        """False when this seat was never given a price.

        Worth asking explicitly, because an unpriced seat contributes exactly
        `0.0` to the session total and a report would print that as though the
        seat were free. A cost guardrail that silently omits a participant is
        the failure mode a cost guardrail exists to prevent, so callers can ask
        rather than discover it from an implausibly cheap invoice.
        """
        return bool(self.input_per_mtok or self.output_per_mtok)


@dataclass(frozen=True)
class Seat:
    model_id: str
    provider: str
    cost: ModelCost = field(default_factory=ModelCost)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("a seat needs a model_id")
        if not self.provider.strip():
            raise ValueError(f"seat {self.model_id!r} needs a provider name")


class CouncilError(ValueError):
    """The council as configured cannot run a valid session."""


@dataclass(frozen=True)
class Council:
    students: tuple[Seat, ...]
    arbiter: Seat

    def __post_init__(self) -> None:
        if not MIN_STUDENTS <= len(self.students) <= MAX_STUDENTS:
            raise CouncilError(
                f"a council seats {MIN_STUDENTS}-{MAX_STUDENTS} students, "
                f"got {len(self.students)}"
            )
        models = [s.model_id for s in self.students]
        duplicates = {m for m in models if models.count(m) > 1}
        if duplicates:
            # Two seats on one model is one opinion billed twice: the sheets
            # correlate, the objections are self-critique wearing a label, and
            # the disagreement signal the session exists to produce is fake.
            raise CouncilError(
                f"students must be distinct models; duplicated: {sorted(duplicates)}"
            )
        if self.arbiter.model_id in models:
            raise CouncilError(
                f"arbiter {self.arbiter.model_id!r} also holds a student seat; "
                "the arbiter must not grade a debate it took part in"
            )

    def seats(self) -> tuple[Seat, ...]:
        return self.students + (self.arbiter,)

    def student_seats(self) -> tuple[int, ...]:
        """Seat numbers, 1-indexed, matching the 'Student N' labels."""
        return tuple(range(1, len(self.students) + 1))

    def student(self, seat: int) -> Seat:
        return self.students[seat - 1]

    def seat_of(self, model_id: str) -> int:
        for index, student in enumerate(self.students, start=1):
            if student.model_id == model_id:
                return index
        raise KeyError(f"{model_id!r} holds no student seat")

    def labs(self) -> tuple[str, ...]:
        return tuple(sorted({s.provider for s in self.students}))

    @property
    def single_lab(self) -> bool:
        """True when every student comes from one provider.

        Not an error — comparing tiers of one family is a legitimate thing to
        want — but it changes what a session *means*, so nothing may show a
        verdict from one without saying so.

        Diversity of priors is the entire product. Three models from one lab
        share training data and alignment, so their blind spots correlate: they
        tend to miss the same things, and the agreement they reach is worth
        much less than the same agreement between families. Switchboard learned
        this in its auditor, where a same-lab pass was being counted as full
        independence. The failure here is worse, because a single-lab council
        can look *healthier* than a mixed one — three similar models converge
        faster, which reads as consensus and is actually correlation.

        It also changes what the deanonymization probe measures. Telling Opus
        from Haiku is a different and easier question than telling Claude from
        GPT from Gemini, so a probe accuracy from a single-lab council does not
        transfer to the mixed case in either direction.
        """
        return len(self.labs()) == 1

    @property
    def arbiter_shares_lab(self) -> bool:
        """The arbiter comes from a lab that also holds a student seat.

        Legal — the arbiter never debated, which is the rule that matters —
        but a weaker form of independence than it looks. An arbiter and a
        student from one lab share training data and alignment, so the arbiter
        is disposed to find its sibling's reasoning natural. It cannot tell
        *which* sheet is the sibling (it sees seat labels only), which limits
        the damage but does not remove it: the disposition applies to the
        argument, not to the label.

        Switchboard makes the same distinction, flagging `cross_lab=False` on
        a verdict rather than refusing it. Same call here, same reason: a
        single-vendor deployment should still get graded, the number just means
        less and has to say so.
        """
        return self.arbiter.provider in {s.provider for s in self.students}

    @property
    def lab_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for student in self.students:
            counts[student.provider] = counts.get(student.provider, 0) + 1
        return counts

    @property
    def doubled_labs(self) -> tuple[str, ...]:
        """Labs holding more than one student seat.

        `single_lab` only fires when *every* student shares a provider, which
        misses the more common shape: four students across three labs, where
        one lab quietly holds two of them. Those two seats share training data
        and alignment, so the council has three independent priors and pays for
        four."""
        return tuple(sorted(lab for lab, n in self.lab_counts.items() if n > 1))

    @property
    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if len(self.students) > 3:
            warnings.append(
                f"{len(self.students)}-student council: objections scale with n(n-1) "
                "and the arbiter reads all of them, so this costs roughly "
                f"{1 + 0.35 * (len(self.students) - 3):.1f}x a three-student session"
            )
        if self.doubled_labs and not self.single_lab:
            warnings.append(
                "lab(s) holding more than one seat: "
                + ", ".join(f"{lab} x{self.lab_counts[lab]}" for lab in self.doubled_labs)
                + " — those seats share priors, so the council has fewer independent "
                "views than students"
            )
        if self.single_lab:
            warnings.append(
                f"single-lab council: all {len(self.students)} students are from "
                f"{self.labs()[0]!r}, so their blind spots correlate and agreement "
                "between them is weaker evidence than it looks"
            )
        if self.arbiter_shares_lab:
            warnings.append(
                f"same-lab arbiter: {self.arbiter.model_id!r} comes from "
                f"{self.arbiter.provider!r}, which also holds a student seat, so the "
                "synthesis is less independent than a cross-lab arbiter's"
            )
        return tuple(warnings)


def demo_council() -> Council:
    """A synthetic three-lab council for offline runs.

    The model names and prices are invented. They are shaped like a real
    lineup — three providers, an arbiter from a fourth — so the offline demo
    exercises the cross-lab case, but nothing here should be mistaken for a
    price list.

    The arbiter is the most capable seat by default. The spec leaves this open
    (a neutral mid-tier arbiter is cheaper and less likely to impose its own
    answer over the transcript's); it is a one-line change here and a question
    for the benchmark to settle, not for a README to assert.
    """
    return Council(
        students=(
            Seat("atlas-3", "atlas", ModelCost(3.0, 15.0)),
            Seat("borealis-2", "borealis", ModelCost(2.5, 10.0)),
            Seat("cinder-1", "cinder", ModelCost(1.0, 5.0)),
        ),
        arbiter=Seat("delphi-frontier", "delphi", ModelCost(5.0, 25.0)),
    )
