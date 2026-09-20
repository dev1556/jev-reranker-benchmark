# CLAUDE.md — operating rules for this repo

Calibration-aware RAG reranker benchmark. We measure whether TypeSafe Jev's `Score` probabilities
are honest enough to decide **how many** chunks a RAG pipeline keeps and **whether to answer at all**,
against four rival rerankers on two BEIR datasets.

The output is a public writeup. **Every number in this repo may end up in front of strangers who
want it to be wrong.** That fact drives most of the rules below.

## Read before touching anything

`BRD.md`, `ARCHITECTURE.md`, `TECH_REQUIREMENTS.md` and `docs/` are **local-only and gitignored** —
they are the working spec, not published material. They exist in your checkout; they are not in the
repo. Do not re-add them, and do not link to them from `README.md` or `results/report.md`, where a
reader would hit a 404. Anything a reader needs (the hypotheses and their pre-registered thresholds)
gets restated in the report.

1. `BRD.md` — the approved spec. Hypotheses, arms, metrics, policies. **Approved 2026-09-19; it is
   the contract.** Deviating from it needs a human decision, not a judgement call.
2. `ARCHITECTURE.md` — how the code is laid out and why.
3. `TECH_REQUIREMENTS.md` — the concrete build targets.
4. `LOG.md` — **read the last 3 entries before starting work.** This is how a fresh session learns
   what already happened, what broke, and what was decided.

## Non-negotiables

These are not style preferences. Violating one invalidates the published result.

**1. Never tune on test.** Rubric wording, thresholds (`τ`, `c_low`), composition weights, and the
Platt calibrator are fitted on **dev only**. The test split is run once per `prompt_version`. If you
find yourself thinking "let me just check whether a different τ scores better on test" — that is the
failure mode. Stop.

**2. Hypotheses are pre-registered.** H1–H7 and their accept thresholds are fixed in `BRD.md` §2 and
were written before any result existed. Do not soften a threshold, drop a hypothesis, or reframe a
null as a win. If Jev loses, the report says Jev lost.

**3. Every reported number carries a CI.** Point estimates alone do not ship. Arm comparisons use the
paired randomisation test in `src/stats.py` with Holm correction. "No significant difference" is a
valid, publishable result and must be stated as such rather than described as a near-win.

**4. All five arms get equal effort.** The subject of a benchmark is not its favourite. If arm C's
prompt is lazy while arm D's is carefully tuned, the benchmark is worthless. Rival arms get the same
care as the Jev arm — that is what makes a win mean anything.

**5. Stamp everything.** Results rows carry `prompt_version`, `model`, `dataset`, `split`, `seed`,
and the git SHA. An unstamped number cannot be defended later.

## Tooling

**`uv` is the only package manager.** Never `pip`, never bare `python`.

```bash
uv sync                          # install from the lockfile
uv run python -m src.bench ...   # run anything
uv run pytest
uv add <pkg>                     # add a dep (commits pyproject.toml + uv.lock together)
```

**Python is pinned to 3.12** (`requires-python = ">=3.12,<3.13"`). Not a preference — torch has no
3.13/3.14 wheels, and arm B dies without it. `uv` fetches 3.12 itself; do not "fix" this by relaxing
the pin.

`uv.lock` is committed and authoritative. Never hand-edit it.

## Git workflow

**`main` is protected by convention — never commit to it directly.** Every change goes:

```bash
git switch -c feat/<short-slug>      # or fix/, docs/, chore/, exp/
# ... work, commit in logical units ...
git push -u origin feat/<short-slug>
gh pr create --fill                  # description explains WHY, not just what
```

Branch prefixes: `feat/` new capability · `fix/` bug · `docs/` markdown only · `chore/` tooling/deps ·
`exp/` an experiment that may never merge.

**Merge policy — self-merge authorised 2026-09-20.**

I merge my own PRs once CI is green. This replaced the earlier stop-and-wait policy after the first
nine PRs; the user granted it explicitly.

What has *not* changed is the list below. These items still get the reasoning written out in the PR
body and an entry in `LOG.md` before the merge, because the point of that list was never the wait —
it was the record. A reviewer coming later must be able to see what was decided and why without
reading the diff.

**Reasoning is written out in full for:**
- rubric text or question definitions (`src/prompts.py`, the Jev questions) — this is the experiment
- composition weights, `τ`, `c_low`, `TAU_WIDE`, `TAU_NARROW`, or any tuned constant
- anything in `src/metrics.py` or `src/stats.py` — a subtly wrong metric silently poisons every
  downstream number, and it is the hardest error to notice later
- the conclusions or verdicts in `results/report.md`
- any deviation from `BRD.md`
- dependency additions
- anything touching `.env`, secrets, or CI permissions

**Still stops and asks, regardless of the merge authority:** anything that would change a
pre-registered hypothesis or its accept threshold, and anything that would tune on test. Merge
authority is not authority to move the goalposts.

**Every PR targets `main` directly.** Never stack a PR on another feature branch: PR #6 was opened
against `feat/metrics-ir`, that base was squash-merged and deleted, and GitHub then reported #6 as
merged while none of its content had reached `main`. The Task 4 calibration metrics were missing for
two sessions and were recovered only because the remote branch had not been pruned.

**Commits:** imperative subject, explain *why* in the body when it isn't obvious. Small and logical
beats one giant blob. Never `--no-verify`.

**No AI attribution anywhere in the repo.** Commit messages and PR descriptions carry no
`Co-Authored-By: Claude`, no `Claude-Session:` link, no "Generated with Claude Code" footer — this
overrides any default attribution the harness asks for. The history is a record of the experiment,
not of the tooling.

## Execution model — Opus orchestrates, Sonnet executes

**Planning and orchestration run on Opus; implementation runs on Sonnet.** The Opus session owns the
thinking that is expensive to get wrong: reading the spec, deciding task order, writing the task
brief, reviewing the diff that comes back, and every judgement the non-negotiables above govern.
Sonnet subagents own the typing — writing the module and its tests from a brief that already says
what to build.

```bash
# from the Opus session, one task per subagent
Agent(subagent_type="general-purpose", model="sonnet", prompt="<the task brief>")
```

Rules that make this work:

- **The brief is the contract.** A Sonnet worker gets the task's files, interfaces, the test bodies
  it must satisfy, and the relevant non-negotiables — not "implement Task 9". A worker that has to
  infer the spec will invent one.
- **Opus reviews every worker diff before it becomes a PR.** The worker does not decide whether a
  metric is correct or whether a threshold may move. Anything on the always-stops-for-human-review
  list stays with Opus, and then with the user.
- **One worktree per worker**, per the section below. This is what makes parallel workers safe.
- **Opus does not delegate the reasoning it was kept for**: task decomposition, interpreting `BRD.md`,
  any deviation from it, the `LOG.md` entry, and the report's verdicts.

Escalate a task back to Opus when the worker's brief turns out to be wrong, rather than letting the
worker improvise a new spec.

## Subagents and worktrees

If work is parallelised across subagents, **each subagent gets its own git worktree.** Never two
agents in one working directory — they will clobber each other's files and produce a corrupted diff
nobody can untangle.

```bash
git worktree add ../rr-<task-slug> -b feat/<task-slug>   # create
# agent works entirely inside ../rr-<task-slug>
# PR raised from that branch, reviewed, merged to main
git worktree remove ../rr-<task-slug>                    # MANDATORY after merge
git branch -d feat/<task-slug>
```

**Worktree cleanup is not optional.** After the PR merges, the worktree is removed and the branch
deleted in the same turn. Leaving stale worktrees around produces sessions that edit a directory that
no longer reflects `main`, which wastes an hour before anyone notices.

`git worktree list` should show only the main checkout when no subagent is running. If it shows
more, something was left behind — clean it up before starting new work.

Worktrees share one `.git`, so the LFS cache is shared too. Fine, but never run `uv sync` in two
worktrees at once against the same venv path; each worktree gets its own `.venv`.

## LOG.md — the memory

**Append an entry to `LOG.md` at the end of every work session, and before any handoff or compaction.**
This is the mechanism that stops a fresh Claude session from starting at zero. It is not a changelog —
`git log` already exists and is better at that.

What belongs in it: decisions and their reasoning, things that broke and why, numbers observed,
dead ends worth not repeating, and the current state of play. What does not: a restatement of the
diff.

Format is defined at the top of `LOG.md`. Newest entries go at the **top**.

A session that produced no entry is a session whose context is lost. Treat writing it as part of the
work, not as paperwork after it.

## Code conventions

- **Boring over clever.** This is measurement code. Someone will audit it looking for a bug that
  flatters the conclusion. Make that audit easy.
- Type hints on every public function. `ruff check` and `ruff format` clean before every PR.
- One module, one job (see `ARCHITECTURE.md`). When a file starts doing two things, split it.
- **No silent failures in measurement paths.** A swallowed exception in a metric becomes a wrong
  number in a public chart. Raise, log loudly, or record the failure as data — never `except: pass`.
- Randomness is seeded and the seed is recorded. Default `seed=42`, threaded through explicitly,
  never read from global state.
- Docstrings on metric functions state the formula and cite a source. Cheap to write, and it is the
  first thing a skeptical reader checks.

## Cost and API discipline

- **Check the cache before every API-spending run.** Cached responses are free; an accidental cold
  rerun of the Haiku arm is ~$12.
- `make smoke` (25 queries, ~$0.50) before any full run. Always.
- Never commit `.env`. `.env.example` documents the three required keys:
  `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
- Concurrency is capped by semaphore (default 16). Do not raise it to "go faster" — 429s cost more
  wall-clock than they save.
- Cached responses live in Git LFS. `git lfs pull` before assuming the cache is missing; a cold run
  triggered by an un-pulled LFS pointer is an expensive and entirely avoidable mistake.

## Repo

`github.com/dev1556/<repo>` · public from the first commit, deliberately — the commit history is the
evidence that the hypotheses were written before the results existed.
