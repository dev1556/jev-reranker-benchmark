# LOG.md — project memory

**If you are a fresh Claude session: read the top 3 entries before doing anything.**
Then read `BRD.md` → `ARCHITECTURE.md` → `TECH_REQUIREMENTS.md`, and `CLAUDE.md` for the rules.

This is **not** a changelog. `git log` already lists what changed and is better at it. This file
records what a diff cannot: why a decision went the way it did, what broke and what the cause turned
out to be, what numbers were observed, and which dead ends are not worth walking again.

## Rules

- **Newest entry at the top**, directly under this section.
- One entry per work session. Write it **before** the session ends or context compacts — a session
  with no entry is a session whose reasoning is gone.
- Record: decisions + reasoning, failures + root cause, observed numbers, dead ends, open threads.
- Do not record: a restatement of the diff, or a list of files touched.
- Numbers get their stamp: `prompt_version`, dataset, split, seed.
- Be honest in here. A log that only records successes is useless to the next session, which will
  hit the same wall and have no idea it was already hit.

## Template

```markdown
## YYYY-MM-DD — <short title>
**State:** <where the project stands in one line>
**Did:** <what was accomplished>
**Decided:** <decision + the reasoning, especially anything that constrains later work>
**Broke / learned:** <failures, root causes, surprises, dead ends>
**Numbers:** <any measurements, with stamps>
**Next:** <the immediate next thing>
**Open:** <unresolved questions or anything waiting on the user>
```

---

## 2026-09-20 — Lost calibration metrics, repo hygiene, Task 9

**State:** Tasks 1-3 and 5-8 on main. Task 4 (calibration metrics) was **not** on main and had to be
recovered — PR #11. Task 9 (arms A and E) is implemented and green but unpushed, waiting on #11.
PR #10 (untracking the spec docs) is green and waiting on the user.

**Did:** Untracked `BRD.md`, `ARCHITECTURE.md`, `TECH_REQUIREMENTS.md` and `docs/` and gitignored them
plus `.superpowers/`. Added two CLAUDE.md rules: no AI attribution in commits or PRs, and the
Opus-orchestrates / Sonnet-executes execution model. Removed the four stale worktrees. Dispatched the
first Sonnet worker (Task 9) and reviewed its diff.

**Decided:**

- **Spec docs stay local.** They remain in history at `b2cb4b8`, so the pre-registration claim is still
  checkable from the log, but nothing in the repo links to them. Consequence recorded in CLAUDE.md: the
  hypotheses and their accept thresholds must be restated in `results/report.md`, because a reader can
  no longer be pointed at `BRD.md` §2.
- **Worker briefs must be self-contained from now on.** A new worktree contains only tracked files, so
  a worker no longer sees the plan or the BRD at all. The Task 9 brief was an extracted copy of the
  plan section written to the scratchpad.
- **One ECE in the repo.** The Task 9 worker wrote a local `_ece` in its test file because
  `src.metrics.ece` did not exist. Replaced with the real import once #11 restored it — a second ECE
  would have let arm E be measured by a metric the report never uses.

**Broke / learned:**

- **The Task 4 calibration metrics were silently missing from main, and nothing flagged it.** Root
  cause: PR #6 was opened against `feat/metrics-ir` instead of `main`. PR #3 was squash-merged and its
  branch deleted, which left GitHub reporting #6 as "MERGED" — into a branch that no longer exists.
  `gh pr list` showed six merged PRs and main had five of their diffs. The work survived only on
  `origin/feat/metrics-calibration` at `db0b9d7`; had that remote branch been pruned too, ECE, Brier,
  `reliability_bins` and AUROC would have been gone, and H2/H3 depend on all four.
  **Rule going forward: every PR targets `main` directly. No stacked PRs.**
- **The gap was found by a Sonnet worker refusing to improvise.** It tried `from src.metrics import
  ece`, did not find it, and stopped rather than inventing a metric or editing `metrics.py`. Worth
  noting because it is evidence the brief-plus-review split works: an orchestrator reading a plan that
  says "import ece" would likely have assumed it existed.
- Earlier session note stands: `gh pr merge` is blocked by the environment's classifier. All merges
  need the user.

**Numbers:** 81 tests passing on the Task 9 branch (which includes #11's 8 calibration tests). No API
call has been made yet; nothing measured.

**Next:** merge #10 and #11, rebase the Task 9 branch onto the new main, re-verify, then PR it. Then
Task 10 (arm B, cross-encoder) to a Sonnet worker.

**Open:**
- Still unanswered: FiQA 300-sampled vs full 648, and the 0.35/0.65 composition weights.
- `results/report.md` owes a restatement of H1-H7 now that `BRD.md` is unpublished.

---

## 2026-09-20 — Tasks 5 and 8: stats and embeddings

**State:** Tasks 1, 2, 6, 7 merged. Task 3 (IR metrics, PR #3), Task 4 (calibration metrics, PR #6),
Task 5 (stats, PR #7) and Task 8 (embeddings, PR #8) are all open. #3/#6/#7 are held for human
review by rule (metrics/stats). #8 is green and routine but the merge command was blocked by the
sandbox classifier, so it is waiting on the user too.

**Did:** Finished the half-written `src/stats.py` + tests found uncommitted in the `rr-stats`
worktree (10 tests, all passing), rebased it onto main, raised PR #7. Then implemented Task 8
(`src/embed.py`, 10 tests) in `rr-embed` and raised PR #8.

**Decided:**

- **`FakeEmbedder` seeds from a `blake2b` digest, not `hash(t)`.** The plan's code used Python's
  `hash()`, which is salted per process, so every fixture candidate pool would silently differ
  between runs. The plan's own determinism test would not have caught it — it compares two calls
  inside one process. Added `test_fake_embedder_is_stable_across_instances`.
- **`test_cosine_is_magnitude_invariant` now asserts the real order** (`["long", "short"]`, fixed by
  the doc_id tiebreak). As written in the plan it asserted `result[0] in {"long", "short"}`, which is
  true for every possible output — a test that could never fail.
- **Dropped a `cfg_pool_hint()` helper** that existed only to interpolate the pool size into an
  assertion message. A function per error string is not worth the read.
- **`assert_pools_identical` stays a runtime check.** FR-2: one divergent document turns this from a
  reranking benchmark into a retrieval benchmark, and nothing downstream would notice.

**Broke / learned:**

- **PR #6 shows "no checks reported" and that is correct, not a CI failure.** It targets
  `feat/metrics-ir`, and the workflow triggers only on `pull_request: branches: [main]`. It will run
  once #3 merges and #6 retargets. Do not go hunting for a broken workflow again.
- `gh pr merge` is blocked by the auto-mode classifier in this environment. Any merge needs the user.
- Long heredocs through Bash still work for Python and file writes; the earlier failure was markdown
  specific. Writing LOG entries via a small inline Python script is reliable.

**Numbers:** None measured — no arm exists yet and no API call has been made. Test suite: 36 passing
on both branches.

**Next:** Merge #3 → #6 → #7 → #8 (human review on the first three), then Task 9 (reranker protocol,
arm A cosine, arm E Platt) in a fresh worktree.

**Open:**
- Four PRs waiting on the user; three of them by design, #8 only because of the merge block.
- `rr-stats`, `rr-metrics-ir`, `rr-metrics-cal`, `rr-embed` worktrees must be removed after their PRs
  merge — `git worktree list` currently shows five entries and should show one.
- Still unanswered from the last session: FiQA 300-sampled vs full 648, and the 0.35/0.65 composition
  weights, which remain a guess until the dev tuning pass.

---

## 2026-09-19 — Implementation plan written

**State:** BRD approved. `CLAUDE.md`, `ARCHITECTURE.md`, `TECH_REQUIREMENTS.md` and a 19-task
implementation plan all written. Still **no repo and no code** — nothing has been created on GitHub.

**Did:** Wrote `docs/superpowers/plans/2026-09-19-jev-reranker-benchmark.md`. Verified the toolchain:
`gh` authed as `dev1556`, uv 0.12.12, git 2.48.1, git-lfs 3.6.1 all present.

**Decided:**

- **Repo name `jev-reranker-benchmark`**, public from the first commit.
- **Python pinned to 3.12.** The dev machine runs 3.14.3; torch has no wheels for it and arm B
  (cross-encoder) would die at install. A test asserts the version so the pin cannot silently drift.
- **Added `src/types.py`**, a refinement of `ARCHITECTURE.md` §8, which listed `Scored` inside
  `rerankers.py`. `data.py` and `embed.py` need `Query`/`Doc` and must not import the arms. Task 1
  updates the architecture doc.
- **Task order puts `metrics.py` and `stats.py` third and fifth**, before any arm exists. They are
  pure functions with hand-computed tests, they gate every downstream number, and building them
  early means no arm result is ever produced against unverified scoring code.
- **Test-split protection is a runtime guard, not a convention.** `tuning_context()` makes
  `get_split(corpus, "test")` raise. Discipline that depends on remembering eventually fails.
- **`--split dev` is the CLI default**, so an accidental invocation cannot burn the
  once-per-`prompt_version` test budget.
- **Errored API calls are recorded and excluded, never scored `0.0`.** Tested explicitly for arms C
  and D. A silent zero would depress that arm's metrics in a way nobody would spot in a CSV — the
  most dangerous bug class in this repo.
- **`confidence=None` must not be coerced to 1.0.** Arms A/B/E have no confidence signal; treating
  silence as certainty would hand them free narrowing under policy 4 and flatter them against Jev.
  There is a test for this in `tests/test_gating.py`.

**Broke / learned:**

- Heredocs through the Bash tool failed again on long markdown — appending to the plan via `Edit`
  anchored on the file's unique tail works reliably. Use that pattern.
- Writing the plan surfaced three interface gaps that the BRD had glossed: cost/latency aggregation
  for the Pareto chart, the shape of the `sims` dict feeding arm A, and where arm E's dev-split
  `(score, label)` training pairs come from. All three are recorded in the plan's Self-Review §4 with
  the task and step that resolves them.

**Numbers:** None yet. Plan-level estimates unchanged: ~$0.50 smoke, ~$20 full cold run.

**Next:** Task 1 — `gh repo create dev1556/jev-reranker-benchmark --public`, `uv init` on 3.12,
LFS tracking for `cache/**`, CI workflow, then Tasks 2–19 in order.

**Open:**
- Execution mode not chosen yet (subagent-driven vs inline).
- FiQA 300 sampled vs full 648 — still the user's call.
- Composition weights 0.35/0.65 remain a guess until the dev tuning pass.

---

## 2026-09-19 — Planning complete, BRD approved

**State:** Spec and docs done. No code, no repo yet. Nothing has been measured.

**Did:** Researched TypeSafe's docs from scratch (Jev was unknown at session start — it launched
2026-09-15). Wrote `BRD.md` through two drafts, got approval, then wrote `CLAUDE.md`,
`ARCHITECTURE.md`, `TECH_REQUIREMENTS.md`, and this file.

**Decided:**

- **One `(query, chunk)` pair per Jev `state`, parallelism via ~50 concurrent calls** — not 50
  chunks stuffed into one state. TypeSafe's own weakness list says accuracy falls as the state fills
  with content unrelated to the decision. The stuffed variant would be cheaper in call count and
  directly against documented guidance.
- **Five arms, not three.** Added a cross-encoder (`bge-reranker-base`) because benchmarking only
  against cosine and an LLM is a rigged fight, and cosine+Platt as arm E because otherwise the
  obvious rebuttal to any calibration win is "twenty lines of logistic regression does that."
- **Two datasets** (SciFact + FiQA). TypeSafe publishes a "jaggedness" page admitting uneven
  cross-domain performance; a single-domain result would be correctly dismissed as anecdote.
- **Adversarial injection probe added.** TypeSafe documents that injected instructions can steer
  outputs. A reranker reads attacker-controlled documents in any real RAG system, and nobody has
  published what that means for reranking. Cheapest high-value addition in the project.
- **Hypotheses pre-registered with accept thresholds before any number exists.** The output is a
  public post; this is what separates it from vendor-flattering benchmarks. A null result ships.
- **Python pinned to 3.12.** Dev machine runs 3.14, torch has no wheels for it, arm B would die at
  install time.
- Workflow, per user: uv only · feature branch → PR → merge · one git worktree per subagent, removed
  immediately after merge · public repo from the first commit so history proves the hypotheses
  predate the results · self-merge on green CI except for rubric/metrics/stats/conclusions/deps,
  which stop for human review.

**Broke / learned:**

- A `cat > file <<'EOF'` heredoc through the Bash tool failed with an unmatched-quote parse error on
  a long markdown document. Used the `Write` tool instead. Don't retry heredocs for large markdown.
- Several doc URLs guessed from the nav sidebar 404. The ones that actually resolve:
  `docs.typesafe.ai/introduction`, `/primitives/score`, `/confidence`,
  `/model-jaggedness/jev-1.13`, and `/llms-full.txt` (most useful — one page, most of the API).
  `/quickstart`, `/quick-start`, `/cookbooks/*` all 404 as guessed.
- **The finding that shaped the whole framing:** TypeSafe's model card calls Jev "fast, calibrated",
  but the confidence page explicitly declines a formal claim — *"We provide `confidence` as a
  convenient measure that fits most use-cases, but you are never locked into our definition."* So
  there is a published adjective with no published measurement behind it. That gap is the
  contribution, and the writeup should lead with it.
- Jev's documented weakness #8 says there is no guarantee that `P(noul) = 1 − P(not noul)`. Worth
  measuring directly (FR-9) — a model sold as calibrated failing its own structural invariant is an
  interesting result either way.

**Numbers:** None. Nothing measured yet. Budget modelled at ~$20 for a full cold run, dominated by
the Haiku arm (~$10–14); Jev is ~$0.40 of it.

**Next:** Implementation plan, then repo creation (`gh repo create dev1556/<name> --public`), LFS
init for `cache/**`, `uv init` on 3.12, CI workflow, then build order per the plan —
`config`/`data`/`embed` first, `metrics`/`stats` next with their hand-computed tests, arms after.

**Open:**
- Repo name not chosen yet.
- FiQA at 300 sampled queries vs. the full 648 — user's call, deferred.
- Composition weights (0.35/0.65) are a guess; one dev tuning pass, then frozen.
- Listwise Jev variant using `Choice` deliberately deferred to future work.

## Task 7: data.py (BEIR loading, splits, tuning guard)

Implemented `src/data.py` and `tests/test_data.py` per task-7-brief.md.
`tuning_context()` uses a ContextVar reset in `finally`, so an exception
inside the context cannot leave the test-split guard latched on.

**SciFact real-load check (2026-09-19):** `load_corpus("scifact", CONFIG)`
against live HuggingFace (`BeIR/scifact`) returned exactly
**5183 docs / 300 queries / 300 qrels** — matches the expected count with no
drift. Field names (`_id`, `text`, `title`, `query-id`, `corpus-id`, `score`)
matched the brief's assumptions exactly; no adaptation needed.
