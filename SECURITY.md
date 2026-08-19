# Security Policy

Quorum is a library you run yourself, with your own keys, against your own
providers. There is no Quorum service, no hosted endpoint, and nothing here
holds your data. That shapes what a vulnerability looks like in this project:
not a breached server, but a mechanism quietly failing to do the thing the
README says it does.

## Reporting

Use GitHub's private vulnerability reporting:
**[Report a vulnerability](https://github.com/JoaquinDG/quorum/security/advisories/new)**.
That opens a thread visible only to you and me.

If that is not available to you, email **joaquin.diaz@newryglobal.com** with
`[quorum security]` in the subject line.

Please do not open a public issue for something you believe is exploitable.
For everything else, including bugs that are merely embarrassing, a public
issue is the right venue and is genuinely welcome.

## What to expect

One maintainer, no security team, no bug bounty. What I can commit to:

- **Acknowledgement within 72 hours.** If you have not heard back by then,
  assume the message went astray and chase it.
- An assessment, with reasoning, within two weeks.
- Credit in the advisory and in the changelog, unless you would rather not
  be named.

I would much rather hear about something that turns out to be nothing than
not hear about it.

## In scope

The protocol claims are the security surface. A way around any of these is a
real finding:

- **Blinding failures.** Any route by which a critic learns which model, lab
  or seat wrote the text it is grading. The schema blinding invariant forbids
  it and `tests/test_invariants.py` asserts it.
- **Independence failures.** Any path by which a round-1 prompt sees peer
  content before it answers.
- **Prompt injection through debate content.** Round 2 puts one model's
  output into another model's prompt. A construction that steers the
  receiving model, suppresses a critique, or escapes into instruction
  context is in scope.
- **Self-grading.** An arbiter that debated, or a student critiquing itself.
- **Fail-closed violations.** A malformed provider reply coerced into
  something valid-looking instead of failing.
- **Key exposure.** Keys are read from the environment and nowhere else. A
  key reaching a trace, a report, a log line, or a provider it was not
  destined for is a serious bug.
- **Artefacts that reach the network.** Reports are self-contained by design
  and CI asserts it. An HTML report that loads anything remote when opened
  turns a local artefact into a beacon, and that is a finding.

## Out of scope

- **The published deanonymization figure.** The prober scores +9.7 points
  over chance and the README says so plainly. Sharpening that measurement is
  a contribution rather than a vulnerability report. See `CONTRIBUTING.md`.
- Models producing wrong, biased or unpleasant answers. That is the subject
  matter, not a defect in the mechanism.
- Spend incurred by your own configuration. The accounting is tested to
  report what was spent, not to stop you spending it.
- Vulnerabilities in the model providers themselves. Those belong to the
  provider.
- Anything that already assumes the attacker holds your API keys or has
  write access to your machine.

## Supported versions

`main` and the most recent release. There are no backports to older tags.
