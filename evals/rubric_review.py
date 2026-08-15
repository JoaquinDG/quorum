"""Have an independent model attack the benchmark rubrics.

    PYTHONPATH=src python3 evals/rubric_review.py --run

The rubrics decide the benchmark's outcome before a single arm runs. They were
written in one pass by the same author as the protocol they score, which is the
textbook way to produce a benchmark that cannot lose. `judgment_tasks.json`
says so in its own header; this is the attempt to do something about it.

The reviewer is told what to hunt for and given permission to be hostile. It
never sees Quorum's results, its README, or which arm is expected to win — only
the question and the criteria — so it cannot reverse-engineer the answer the
author wanted.

Three failure modes it is asked to find:

- **Process smuggled in as substance.** "Names the strongest counter-argument"
  looks like a property of a good answer and is one a debate reaches more
  easily. Those criteria are already tagged `favours_deliberation`; the
  question is whether the tagging is *complete*.
- **Mis-tagging.** A criterion marked neutral that a deliberation gets for
  free is worse than an untagged one, because the harness reports a "neutral
  criteria only" column that would then be quietly contaminated.
- **Unmeasurable wording.** A criterion a judge cannot apply consistently adds
  noise and lets whichever arm is longer win on rater fatigue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quorum import ModelCost, OpenAICompatibleProvider, ProviderError, Seat  # noqa: E402
from quorum import extract_json, load_tasks  # noqa: E402

HERE = os.path.dirname(__file__)
TASKS = os.path.join(HERE, "judgment_tasks.json")

REVIEWER = Seat("gemini-3.1-pro-preview", "google", ModelCost(1.25, 10.0))

PROMPT = """You are auditing a scoring rubric for bias. Be hostile: your job is to
find the ways this rubric could decide its own outcome.

The rubric scores written answers to a hard judgement question. Some answers
will come from a single model answering once. Others will come from a process
where several models answer independently, critique each other's specific
claims, revise, and have a separate model synthesise the result.

A criterion is "deliberation-favouring" if that multi-model process would
satisfy it more easily than a single good answer would — regardless of whether
the underlying answer is better. The rubric already tags some criteria this
way. Your job is to check whether the tagging is honest and complete.

QUESTION BEING SCORED
{prompt}

CRITERIA
{criteria}

For each criterion, decide:
- "verdict": "fair" (measures answer quality, reachable by either),
             "favours_deliberation" (the tagged property, whether or not it is
             currently tagged),
             "unmeasurable" (a judge could not apply it consistently)
- "mistagged": true if your verdict disagrees with the current tag
- "why": one sentence, concrete
- "rewrite": a better wording, or null if the criterion is fine as written

Then judge the rubric as a whole:
- "overall_bias": "favours_deliberation" | "neutral" | "favours_single"
- "worst_criterion": the key of the single most problematic criterion, or null
- "comment": two sentences maximum

Reply with a single JSON object and nothing else:
{{"criteria": [{{"key": "...", "verdict": "...", "mistagged": false,
  "why": "...", "rewrite": null}}],
  "overall_bias": "...", "worst_criterion": null, "comment": "..."}}
"""


def review(task, provider) -> dict:
    criteria = "\n".join(
        f'  {c.key}: "{c.text}"  [currently tagged: '
        f'{"favours_deliberation" if c.favours_deliberation else "neutral"}]'
        for c in task.criteria
    )
    completion = provider.complete(
        REVIEWER.model_id, PROMPT.format(prompt=task.prompt, criteria=criteria), 8192
    )
    data = extract_json(completion.text, actor="reviewer")
    data["_tokens"] = completion.output_tokens
    data["_cost"] = REVIEWER.cost.estimate(completion.input_tokens, completion.output_tokens)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adversarially review the rubrics.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(HERE, "RUBRIC_REVIEW.md"))
    args = parser.parse_args(argv)

    tasks = load_tasks(TASKS)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} tasks, {sum(len(t.criteria) for t in tasks)} criteria")
    print(f"reviewer: {REVIEWER.model_id} (wrote none of them, sees no results)")
    if not args.run:
        print("\n  re-run with --run")
        return 0
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2

    provider = OpenAICompatibleProvider(
        name="google", base_url="https://generativelanguage.googleapis.com",
        chat_path="/v1beta/openai/chat/completions", env_var="GEMINI_API_KEY")

    reviews, cost = {}, 0.0
    for task in tasks:
        try:
            data = review(task, provider)
        except (ProviderError, ValueError) as exc:
            print(f"  {task.key:22s} FAILED {str(exc)[:70]}")
            continue
        reviews[task.key] = data
        cost += data.get("_cost", 0)
        flags = sum(1 for c in data.get("criteria", []) if c.get("mistagged"))
        print(f"  {task.key:22s} bias={data.get('overall_bias','?'):22s} mistagged={flags}")

    # Aggregate: the tagging's honesty is the thing being audited.
    tagged_now = {(t.key, c.key): c.favours_deliberation for t in tasks for c in t.criteria}
    mistagged, unmeasurable, bias_counts = [], [], {}
    for task_key, data in reviews.items():
        bias_counts[data.get("overall_bias", "?")] = bias_counts.get(data.get("overall_bias", "?"), 0) + 1
        for c in data.get("criteria", []):
            key = (task_key, c.get("key"))
            if c.get("verdict") == "unmeasurable":
                unmeasurable.append((task_key, c.get("key"), c.get("why", "")))
            if c.get("mistagged") and key in tagged_now:
                mistagged.append((task_key, c.get("key"), tagged_now[key],
                                  c.get("verdict"), c.get("why", ""), c.get("rewrite")))

    total = sum(len(t.criteria) for t in tasks)
    print(f"\n=== verdict ===")
    print(f"  rubrics reviewed:   {len(reviews)}/{len(tasks)}")
    print(f"  overall bias:       {bias_counts}")
    print(f"  mistagged criteria: {len(mistagged)} of {total}")
    print(f"  unmeasurable:       {len(unmeasurable)}")
    print(f"  cost:               ${cost:.4f}")

    lines = [
        "# Adversarial rubric review", "",
        f"Reviewer: `{REVIEWER.model_id}`. It wrote none of these rubrics, never",
        "saw Quorum's README or results, and was told to be hostile.", "",
        f"- rubrics reviewed: **{len(reviews)} of {len(tasks)}**",
        f"- overall bias verdicts: **{bias_counts}**",
        f"- criteria the reviewer says are mistagged: **{len(mistagged)} of {total}**",
        f"- criteria it calls unmeasurable: **{len(unmeasurable)}**", "",
    ]
    if mistagged:
        lines += ["## Mistagged criteria", "",
                  "| task | criterion | tagged | reviewer says | why |",
                  "| --- | --- | --- | --- | --- |"]
        for t, k, was, verdict, why, _ in mistagged:
            lines.append(f"| `{t}` | `{k}` | "
                         f"{'favours_deliberation' if was else 'neutral'} | {verdict} | {why} |")
        lines.append("")
    if unmeasurable:
        lines += ["## Criteria a judge could not apply consistently", "",
                  "| task | criterion | why |", "| --- | --- | --- |"]
        for t, k, why in unmeasurable:
            lines.append(f"| `{t}` | `{k}` | {why} |")
        lines.append("")
    lines += ["## Per-task comments", ""]
    for task_key, data in reviews.items():
        lines.append(f"- **`{task_key}`** — {data.get('overall_bias','?')}: "
                     f"{data.get('comment','')}")
    lines.append("")
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(reviews, handle, indent=2)
    print(f"  written to {os.path.relpath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
