# Meza Scholars satgen — project notes

Read this first. It records how the generator works, why each rule exists, and how we improve it.
Update it in the same pull request whenever you add a rule or change how something works.

## What this repo is

- `src/satgen/` is the open-source satgen pipeline (MIT license, upstream: github.com/Dulkinsar/sat-question-pipeline).
  It writes Digital SAT questions, solves each one 5 times without the key, has a stronger model review it with the
  key hidden, confirms any key change, and removes near-duplicates. **Don't edit `src/satgen/`.** Pull upstream
  fixes with `git fetch upstream && git merge upstream/main`.
- Everything Meza-specific lives in the `meza_*` files, `data/vocab_bank.json`, and `tests/test_meza_quality.py`.
- Scope right now: **Reading and Writing only.**

| File | What it does |
|---|---|
| `meza_run.sh` | Entry point. `./meza_run.sh 20` = 20 questions in the full-test mix; `./meza_run.sh 8 words_in_context_completion` = only that type. Each run goes to `runs/<date_time>/`. |
| `meza_satgen.py` | Wraps satgen: assigns subject areas, clue structures, vocabulary, and answer-choice plans in code; runs the quality gate; rewrites failing drafts. |
| `meza_quality.py` | Free, deterministic checks (lint, rhythm, vocabulary) and the passage-free probe. `python meza_quality.py runs/<run>/questions.json` lints an old run with no API calls. |
| `meza_context.txt` | The item-writing rulebook the writer model follows. |
| `meza_export.py` | Builds `review.html` and `questions.csv` for human review. |
| `data/vocab_bank.json` | 499 words from Sebastian's Barron's deck (words and parts of speech only), tiered by word frequency: 233 advanced, 209 mid, 57 common. |

Run tests with `pytest` (no network). Offline smoke run: `python meza_satgen.py run --offline --count 6`.
Optional but recommended: `pip install wordfreq` so the checker can judge the level of words outside the bank.

## How a question is made

1. Code assigns the slot: subject area (10-area rotation), and for Words in Context a clue structure, passage
   vocabulary, and an answer-choice plan.
2. The writer (Claude Sonnet) drafts it from `meza_context.txt` plus those assignments, plus a summary of the
   last 5 questions of the same type.
3. Quality gate (before satgen's solvers): lint → rhythm check → vocabulary check → passage-free probe.
   A failing draft goes back to the writer with the exact problems, up to `MEZA_MAX_REPAIRS` (default 2) times.
4. satgen's normal pipeline: 5 blind solves, key-hidden review, re-check, dedupe, export.
5. Every draft attempt is logged to `runs/<run>/quality.jsonl`.

Questions of the same type are written one at a time (other types run in parallel) so each one can see the
ones before it.

## Rules and why they exist (decisions log)

Each rule below came from auditing output or from Sebastian's review rounds against real practice sets.

- **The key must not stand out without the passage.** Audit of the first 30 questions: the key was the longest
  choice in 22/30 (chance ≈ 7), and absolutes (only, best, entirely) appeared only in wrong answers. → lint
  checks length and absolutes; the passage-free probe rejects items a test-taker can solve from the choices alone.
- **Vary sentence structure (Words in Context).** Round 1: 3 of 5 items used "word, colon, explanation"; real
  tests use it maybe 5–10% of the time. → 8 clue structures assigned in code by weight (contrast 3, exposition 3,
  evaluation 2, consequence 2, delayed 1, inferred_trait 1, extension 1, restatement 1). Only `restatement` may
  put a colon/semicolon/dash after the blank.
- **Don't define the answer.** Round 1: clues restated the key ("exacting" followed by a list of exacting acts).
  → rulebook bans definitional clues; hard items need the meaning to come from how sentences relate.
- **Don't fall into a rhythm.** Round 2: every passage was "belief, then twist" even with different labels.
  → the writer sees its last 5 questions; rhythm check rejects a third twist in a row, a repeated opening, the
  blank in the same place three times, or a repeated answer/tested word. Only `contrast` items may twist.
- **Harder passages need harder words, but only a sprinkle.** Round 3: everything was readable by any student.
  Round 4: three bank words per passage was too much. → medium: 1 advanced word max; hard: 1–2 max.
- **Balance answer-choice difficulty.** Round 4: all four hard choices were obscure, which real tests rarely do.
  → hard WIC choices = 2 advanced + 1 mid + 1 common; medium = 1/2/1; easy = 0/1/3. The key's tier is assigned
  (hard: usually advanced, sometimes mid or common) so "pick the word I don't know" stops working.
- **Underlined-word items get the same treatment.** Exact stem, one `<u></u>` matching the stem, choices of
  ≤4 words, at least one wrong choice is the word's most common meaning, ≤1 advanced word among choices
  (difficulty comes from the uncommon sense). Tested words come from the bank's common/mid tiers.

## Legal and content rules (do not break)

- Never paste official College Board / Bluebook / Question Bank questions into prompts, the rulebook, tests, or
  this repo. Study them yourself and write the **pattern** in your own words.
- The vocabulary bank holds words and parts of speech only — no Barron's definitions or example sentences.
- Never invent a quote, study, or statistic and present it as real. Use generic researchers ("a team of
  ecologists") for invented findings.
- Never commit `.env` (API keys). Each person uses their own key.

## How we improve it (the review loop)

1. Work a real practice set for one skill.
2. Generate 3–5 questions of that skill (or ask Claude to write a round by hand to the current rules).
3. Note every difference from the real test: structure, clue directness, vocabulary level, trap quality, rhythm.
4. Turn each difference into (a) a rule in `meza_context.txt`, (b) a check in `meza_quality.py` if it can be
   checked in code, and (c) a test in `tests/test_meza_quality.py`. Add the reason to the decisions log above.

## Working together

- `main` is the official version. Work on a branch (`sebastian/...`, `<name>/...`), open a pull request, have the
  other person review, then merge.
- To avoid edit conflicts in `meza_context.txt`, split work by question type.
- Run `pytest` before opening a pull request.

## Open next steps

- **Trap labels (proposed, not built):** a Words in Context trap list (most common meaning, other real meaning,
  wrong connotation, wrong degree, reversal, borrowed from elsewhere in the passage, grammatical but imprecise);
  the writer tags every wrong answer; easy may have one filler, medium/hard none; tags shown in `review.html`.
  The audit of rounds 5–6 found 9 of 30 wrong answers were filler.
- First real 20-question run with the new gate, then compare the key-longest rate against the 22/30 baseline.
- Check the ratio of fill-in-the-blank to underlined-word items in `meza_run.sh` (currently 5:4) against real tests.
- Bring the same review loop to the other R&W skills (transitions, inferences, central ideas, ...).
- Later: calibrate difficulty with real student results from the portal; a web tool for tutors.
