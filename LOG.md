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

## 2026-09-20 - H5/H6 modules merged; Task 16 briefed but NOT started

**State:** Tasks 1-15 are merged to `main` (PRs #1-#23). 162 tests passing. Every measurement module
exists: metrics, calibration metrics, stats, cache, data, embeddings, six arms (A cosine, B
cross-encoder, C Haiku, D Jev composed, D-prime jev_cookbook, E Platt), gating, unanswerable sets,
robustness probe. **No API call has ever been made. Nothing has been measured. `.env` does not exist
on this machine.**

**Task 16 is split in two and neither half is started.** Both workers were killed by the session
ending before they wrote a single file; the `../rr-run-infra` and `../rr-bench-cli` worktrees exist,
are clean, and sit at `9bfd9e7`. Nothing to salvage, nothing lost. The two briefs were written to a
temp scratchpad, which is not durable, so the substance of both is recorded below - **re-dispatch
from this section if the briefs are gone.**

### Task 16a - run infrastructure (`embed.py`, `rerankers.py`, `data.py`)

Four things the real run cannot happen without:

1. **`build_pools_and_sims(corpus, queries, embedder, cfg) -> (pools, sims)`** in `embed.py`, where
   `sims[query_id][doc_id]` is the cosine similarity that picked the pool. `build_pools` keeps its
   current signature as a thin wrapper. Arm A consumes `sims` and nothing produces it today.
2. **`OpenAIEmbedder` has no cache** - the original plan's docstring claimed `.npz` caching that was
   never implemented. FiQA is 57,638 passages re-embedded on every cold run, and the published
   reproduce path cannot work without a key at all. Cache to `cfg.DATA_DIR/"embeddings"/{key}.npz`,
   key = sha256 over `EMBED_MODEL` plus the exact texts in order, atomic write (temp file then
   `os.replace`, matching `cache.py`), zero API calls on a hit.
3. **Arm C is sequential.** 50 candidates x ~210 test queries x 2 datasets is ~21,000 serial Haiku
   calls, roughly six hours per dataset, while arm D already runs concurrently - the arms are not
   even comparable on latency. `ThreadPoolExecutor(max_workers=cfg.SEMAPHORE)` inside `rerank`,
   output order unchanged (`(-score, doc_id)`), one failure must not abort its siblings. Do not raise
   `SEMAPHORE`.
4. **FiQA sampling is not wired.** `sample_query_ids` exists and nothing calls it, so FiQA would run
   all 648 judged queries. Apply `cfg.FIQA_N_QUERIES` sampling inside `get_split`, **before** the
   dev/test partition, so every consumer sees the same 300 and no caller can forget. SciFact
   unaffected.

### Task 16b - the CLI (`bench.py`). Five defects in the plan's draft

Each one changes the results CSV, which every chart and verdict is built from:

1. **`sims` was a dict of zeros.** Arm A ranks on those values, so every document ties, the ranking
   collapses to doc_id order, `p_relevant` is 0.5 everywhere - and it still gets published as "the
   cosine baseline". Use real sims, plus a runtime assertion that a query's sims are not all
   identical, so a regression to placeholders fails loudly instead of looking plausible.
2. **Arm E is never fitted** and raises until `.fit()` runs. `run()` must fit it on the **dev** split
   even when the run is `--split test` (non-negotiable #1). Labels are qrels grade > 0.
3. **Ranking metrics were emitted once per policy** - four identical rows differing only in the
   `policy` column, which a chart reads as a policy effect. Ranking metrics go out once with
   `policy="none"`; per-policy rows carry `mean_kept`, `abstention_rate`, `mean_words_forwarded`.
4. **Every row was stamped `model=JEV_MODEL`**, so arm B's and arm C's rows would claim to come from
   the Jev model. Each arm stamps its own: cosine/platt to `EMBED_MODEL`, cross_encoder to
   `CROSS_ENCODER`, llm to `LLM_MODEL`, jev/jev_cookbook to `JEV_MODEL`.
5. **`ARM_NAMES` was missing `jev_cookbook`.**

Plus: latency and token columns (`latency_p50`, `latency_p95`, `input_tokens_total`,
`output_tokens_total`, `n_calls`) aggregated from `Scored.meta`, **with no dollar conversion** - Jev's
price per token is not known to this repo and an invented price in a public chart is worse than no
chart. Task 17 does money from a documented table.

**Decided:**

- **`mean_words_forwarded`, not "tokens forwarded".** BRD's headline table says tokens; a real
  tokenizer is a new dependency for a number only ever compared *between arms over identical text*.
  Whitespace words are the honest proxy, so the metric is named for what it measures and the docstring
  says why. Task 17 can label it "tokens (whitespace proxy)".
- **H6 targets only genuinely irrelevant documents**, 5 per query per BRD section 8, not one uniform
  pick from the pool. A random pick can land on an already-relevant, already-top document that has
  nowhere to rise; it contributes ~0 inflation and drags the mean down, reading as robustness no arm
  earned.
- **H6's nDCG delta is at k=10**, matching the headline table, not at pool length, which is not
  comparable across pools of different size. `N_INJECTED_PER_QUERY=5` and `NDCG_K=10` are now named
  constants in `config.py`.
- **Self-merge is live** and works: PRs #21-#23 were merged by me on green CI. `gh pr merge` is no
  longer blocked by the sandbox classifier.

**Broke / learned:**

- **A cost bug in H5 worth remembering as a pattern.** `build_gold_removed` stripped the corpus and
  called `build_pools` per query, which re-embeds the whole corpus each time: 300 FiQA queries over
  57,638 passages is ~17M embedding calls instead of 57,638. Fixed to embed once and drop gold by
  index; verified byte-identical pools against the naive version with no gold leakage, plus a test
  asserting two embed calls total regardless of query count. **The pattern: any per-query loop that
  calls a corpus-level function is a cost bug until proven otherwise.**
- **A worktree created before a merge does not contain that merge.** `rr-robustness` predated the
  `FIQA_N_QUERIES` commit, so a config edit anchored on that line silently no-op'd and three tests
  failed with `AttributeError`. Rebase a worker's worktree onto current `main` before editing config.
- Every spec bug found so far is the same shape: **a test double that does not match production.**
  Four for four across arms B, C, D and the H5/H6 modules.

**Numbers:** 162 tests. bge logits `[2.848, -7.643]` on a relevant/irrelevant pair, still the only
real model output observed. Zero API spend to date.

**Next, in order:**

1. Re-dispatch Tasks 16a and 16b (briefs above). They touch disjoint files and run in parallel; 16b
   codes against 16a's `build_pools_and_sims` signature without waiting for it.
2. `make smoke` - **the first real spend.** Needs `.env` with `TYPESAFE_API_KEY`,
   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. 25 queries, ~$0.50. Record actual spend, wall-clock, cache
   hit rate, any 429s, and the resolved `response.model` for Haiku. **If spend exceeds $2, stop.**
3. Task 17 (report, charts, H1-H7 verdicts), Task 18 (ablations), Task 19 (Streamlit and README).

**Open:**
- `.env` absent. Ask the user for it when the smoke run is actually ready, not before.
- Tuned constants all still placeholders pending the dev tuning pass: `W_TOPICAL`/`W_ANSWERS`, `TAU`,
  `C_LOW`, `MASS_TARGET`, `TAU_WIDE`, `TAU_NARROW`.
- `results/report.md` owes a restatement of H1-H7 (the BRD is unpublished) and both approved BRD
  deviations: the claim-vs-question rubric rewording, and the added `jev_cookbook` arm.
- OneDrive intermittently locks `.git` and fails object writes. After **any** failed git command,
  check `git branch --show-current` before retrying - that is how a commit landed on `main` earlier.

---

## 2026-09-20 — Arms C and D, and a rubric that only fitted one dataset

**State:** All five arms exist plus a sixth variant. Open: #17 (haiku model id), #18 (arm C), #19
(arm D + cookbook variant, rebased on #18). 132 tests passing with every arm present together.
Nothing has hit a live API yet.

**Did:** Ran two more Sonnet workers (Tasks 11 and 12), reviewed both, installed the
`typesafe@typesafe-ai` plugin and used its skill for arm D. Resolved the `src/prompts.py` /
`src/rerankers.py` collision between the two arm branches by hand.

**Decided:**

- **Arm D gains a fourth question, `cookbook_relevant`, and a second arm `jev_cookbook`.** TypeSafe's
  own reranking cookbook recommends a *single Noul*, no composition, no weights — different from
  BRD §4.4's composed design. Running their recipe as a fourth question **in the same call** costs no
  extra request. If the composed arm wins, this answers "the tuned weights did the work"; if it
  loses, it answers "you did not follow their documented recipe". The cookbook also independently
  confirms our per-pair `state` choice: "one request per candidate · no request sees another".
- **That Noul carries `criteria: {true, false}`.** The cookbook states both conditions explicitly;
  omitting them would have run a weaker version of TypeSafe's own recommendation than the one they
  publish, which under the equal-effort rule biases the comparison toward the arm being checked.
  Verified offline that the SDK accepts them.
- **Both rubrics are now neutral between a claim and a question** — a deviation from BRD §4.4, taken
  before any number existed. The pre-registered levels said "the claim the query is making", which
  fits SciFact (queries *are* claims) and not FiQA (colloquial questions). Both arms C and D would
  have been answering a slightly wrong question on one of two datasets, which is exactly what H7
  measures. Every level describes the same situation as before; only the framing noun changed, and the
  identical change went into arm C. `PROMPT_VERSION` stays `v1` because v1 has never been executed.
- **Failures are cached (reversing the arm C worker).** It argued a transient 503 should not be locked
  in permanently — fair, but the published reproduce path is `git lfs pull && make report` with no API
  keys. An uncached failure makes a warm-cache run attempt a live call, and the arm raises without a
  key, so one historical 503 would kill the whole reproduction and NFR-5 with it. Retrying is now an
  explicit human action: delete the entry, re-run the arm.
- **`LLM_MODEL` is the undated `claude-haiku-4-5`** (user's call). An alias can be repointed, but the
  cache is keyed on the model stamp, so a repoint surfaces as a cache miss rather than silently mixed
  results. Record the resolved `response.model` during the smoke run.

**Broke / learned:**

- **The same cache-shard bug appeared in both arm C's and arm D's draft code, and both workers found
  it independently.** `cache.get(key)` defaults to `arm="misc"` while `put(key, payload, arm=name)`
  writes to the arm's own shard, and `_path` shards on arm — so every "hit" would have missed and
  re-called the paid API. Note the obvious cache-hit test still passes with the bug present, because a
  miss just re-calls the same fake.
- **Probability keys: int in the SDK, string over the wire, string after a JSON cache round-trip.** So
  a cache *miss* and a cache *hit* would have disagreed on `p_relevant` for the same document — a
  second run quietly reporting different numbers. Normalised to int at the boundary.
- **The `questions`-as-plain-dicts question is settled:** the SDK types the values as model objects,
  but `TypeAdapter(Mapping[str, Noul | Score]).validate_python({...})` coerces dicts cleanly. Checked
  offline rather than left as an unknown for the first live run.
- **I committed directly to `main`.** A git object write failed mid-command (OneDrive locking this
  folder — the second occurrence), the failing command's trailing `git switch main` still ran, and my
  retry of `add`+`commit` therefore landed on `main`. Nothing was pushed; the commit was moved to its
  branch and `main` reset. **Lesson: after any failed git command, check `git branch --show-current`
  before retrying.** Do not chain `git switch` after a commit in the same command.
- **One of my own tests passed vacuously** — the `Broken` fake raised before incrementing its call
  counter, so `assert calls == 0` could never fail. Fixed to count before raising, plus an assertion
  that the first run really did attempt the calls. Same class of error as the plan's
  `assert result[0] in {"long", "short"}` from the embeddings task.
- Rebasing arm D onto arm C required resolving `prompts.py` and `rerankers.py` by hand (both branches
  create/append the same files). My first scripted merge silently dropped `LLM_RERANK_PROMPT` and
  mangled UTF-8 (`§` → mojibake) because it split on `"""` and let subprocess pick the encoding.
  Redone reading blobs as bytes and stripping only the leading docstring. **Check for the constants by
  name after any scripted merge.**

**Numbers:** 132 tests. Real bge logits from the previous session still the only observed model
output. No API spend to date.

**Next:** merge #17 → #18 → #19, then Task 14 (unanswerable query sets, H5) and Task 15 (adversarial
probe, H6), which are independent of each other. Task 16 (bench CLI) after those.

**Open:**
- `.env` does not exist locally, so no live call has ever been made. `make smoke` is the first real
  test of arm C, arm D, and the model id in #17.
- FiQA 300-sampled vs full 648 — still unanswered, and Task 14 onward will assume 300 unless told.
- Tuned constants still at placeholder values: `W_TOPICAL`/`W_ANSWERS`, `TAU`, `C_LOW`, `MASS_TARGET`,
  `TAU_WIDE`, `TAU_NARROW`.
- `results/report.md` owes a restatement of H1–H7 and now also this rubric deviation.

---

## 2026-09-20 — Arms A/B/E, gating, and two spec bugs the workers caught

**State:** PRs #13 (arms A+E), #15 (arm B), #14 (gating policies) all green and waiting on review.
Tasks 11 (arm C) and 12 (arm D) are next and both touch `src/rerankers.py`, so they are serial.

**Did:** Ran the first three Sonnet workers under the new execution model (Task 9 inline earlier, then
Tasks 10 and 13 in parallel worktrees). Reviewed all three diffs. Installed the `typesafe@typesafe-ai`
plugin and validated the planned arm D integration against the live docs and the installed SDK.

**Decided:**

- **`TAU_WIDE = 0.3` and `TAU_NARROW = 0.7` are now named constants in `config.py`.** The draft used
  `TAU * 0.6`, a tuned value hiding inside `gating.py`; a worker's alternative, `TAU * C_LOW`, avoided
  the new constant but reused a *confidence* threshold as a *probability* scale. Both are worse than
  naming the thing. `test_gated_bars_are_ordered` pins `TAU_WIDE < TAU < TAU_NARROW` so a dev tuning
  pass cannot silently swap the meanings of widen and narrow. The user was asked and chose this.
- **The draft's `confidence_gated` implemented only half of BRD §6** — it widened on low confidence and
  never narrowed. Both directions now exist.
- **`mass` sums raw `p_relevant`, not renormalised.** BRD §6 says "cumulative expected-relevance mass".
  Renormalising makes an arm claiming 0.9 on everything indistinguishable from one claiming 0.1, which
  destroys the exact property the policy exists to exploit.
- **Every policy orders by the arm's own score.** A policy decides how many chunks to keep, never the
  order. Sorting policies 2-4 by `p_relevant` would move nDCG for reasons unrelated to selection and
  confound H1 with H3. It only differs for an arm where score and `p_relevant` are not monotone in
  each other — arm D exactly.

**Broke / learned:**

- **Arm B would have shipped a double sigmoid.** `CrossEncoder.predict()` applies `Sigmoid` by default,
  so the spec's "returns raw logits" was wrong and the implementation would have squashed an
  already-squashed probability: a true logit of 2.85 (P=0.945) published as 0.72. Verified on the real
  model — default `predict()` gives `[0.945, 0.00048]`, `activation_fn=Identity()` gives
  `[2.848, -7.643]`. Fixed at construction with a stubbed-import regression test. Arm B is the
  strongest rival, so this error ran in the direction that would have flattered the subject.
- **Two more of the same class are waiting in Task 12**, found before writing any code and recorded in
  the scratchpad at `task-12-api-delta.md`:
  1. `ScoreAnswer.probabilities` is `dict[int, float]` in the SDK but string-keyed over the wire, and
     **JSON object keys are always strings, so a cache hit and a cache miss would disagree**. The
     plan's `probs["1"] + probs["2"]` breaks on a live response. Normalise with
     `{int(k): float(v) ...}` at the boundary and test both key types.
  2. `response.answers[key]` holds `ScoreAnswer` / `NoulAnswer` pydantic objects, not dicts. The
     plan's `FakeJev` returns dicts, so composition written against dicts passes every test and fails
     on the first live call. `JevReranker` should `model_dump()` at the client boundary.
  Also confirmed `noul` is a probability (0.95), not a boolean, despite the SDK page's wording.
- **Pattern worth naming: every spec bug so far was a fake that did not match production.** Arm B's
  fake returned logits the real model does not return; arm D's fake returns dicts the real SDK does
  not return. Any new arm's brief must include "load the real thing once and compare".
- A `Permission denied` writing a git object during the arm B commit — OneDrive locking the repo
  directory. Commit, push and tests all verified clean afterwards. Watch for it; the repo lives in a
  synced folder.
- The plugin's skill does not register until the session restarts, so its docs were read directly.
- Switching branches after the untracking PR merged **deleted the local `BRD.md`, `ARCHITECTURE.md`,
  `TECH_REQUIREMENTS.md` and `docs/`** — git removes files tracked in the old HEAD but absent from the
  new one. Restored from `90a3061`. They are ignored now, so it cannot recur.

**Numbers:** 87 tests passing on each of the three branches. Still no API call, nothing measured.
Real-model check on bge: logits `[2.848, -7.643]` for a relevant/irrelevant pair — sane.

**Next:** merge #13 → #15 → #14, then Task 11 (arm C, Haiku) and Task 12 (arm D, Jev) with the API
deltas folded into the brief. Invoke the `typesafe:typesafe-ai` skill when writing arm D.

**Open:**
- `TAU_WIDE`/`TAU_NARROW` join `W_TOPICAL`/`W_ANSWERS` and `TAU`/`C_LOW`/`MASS_TARGET` as placeholders
  awaiting the dev tuning pass.
- FiQA 300-sampled vs full 648 — still unanswered.
- `results/report.md` owes a restatement of H1-H7 now that `BRD.md` is unpublished.

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
