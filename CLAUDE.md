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
| `meza_satgen.py` | Wraps satgen: assigns subject areas, clue structures, passage shapes, question wordings, vocabulary, and answer-choice plans in code; runs the quality gate; rewrites failing drafts. |
| `meza_quality.py` | Free, deterministic checks (lint, rhythm, vocabulary) and the passage-free probe. `python meza_quality.py runs/<run>/questions.json` lints an old run with no API calls. |
| `meza_context.txt` | The item-writing rulebook the writer model follows. |
| `meza_export.py` | Builds `review.html` and `questions.csv` for human review. |
| `data/vocab_bank.json` | 499 words from Sebastian's Barron's deck (words and parts of speech only), tiered by word frequency: 233 advanced, 209 mid, 57 common. |

Run tests with `pytest` (no network). Offline smoke run: `python meza_satgen.py run --offline --count 6`.
Optional but recommended: `pip install wordfreq` so the checker can judge the level of words outside the bank.

## How a question is made

1. Code assigns the slot: subject area (10-area rotation), and for Words in Context a clue structure, passage
   vocabulary, and an answer-choice plan. Central Ideas and Details items get a passage shape and, for main idea,
   a stem; detail items also get a question wording family and a focus. Inference items get the ending's job, a
   lead-in, a blank form, and a passage shape. Command of Evidence items get a variation (finding stem and
   logic, table task, or quotation genre). Every question also gets the letter its correct answer will
   sit at, planned for the whole batch.
2. The writer (Claude Sonnet) drafts it from `meza_context.txt` plus those assignments, plus a summary of the
   last 5 questions of the same type.
3. Quality gate (before satgen's solvers): lint → rhythm check → vocabulary check → passage-free probe.
   A failing draft goes back to the writer with the exact problems, up to `MEZA_MAX_REPAIRS` (default 2) times.
4. satgen's normal pipeline: the key is moved to its planned letter (the wrong answers are shuffled around it),
   then 5 blind solves, key-hidden review, re-check, dedupe, export.
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
- **Main idea items follow the real test's shape (Information and Ideas, round 1).** A review of about 20 real
  medium main-idea items: the main idea is often pieced together from the first and last sentences; the correct
  answer often keeps a concession ("Although X, Y"); the tempting wrong answer often copies the key's structure
  with one half swapped; "always" and "never" basically never appear. The first hand-written round ran 91–130
  words in 5–6 sentences. → the `central_ideas` rules were rewritten, lint rejects "always"/"never" in choices, and
  the stem is assigned in code: in every 20, 16 use "Which choice best states the main idea of the text?", 3 use a
  question form ("Which question does the text most directly attempt to answer?" or "...which research question
  was [the] study most likely intended to answer?"; choices must then be questions), and 1 a small variation.
  Science researchers stay generic, and literary intros stay plain (no invented author, title, or year).
- **Central Ideas and Details follow a 33-item review at all three levels (round 2).** Detail questions outnumber
  main idea about 4 to 1 (26 vs 7). Passages run 66–109 words (usually about 85) at every level; difficulty comes
  from density (easy about 4–5 sentences of ~21 words, hard about 2–3 of ~31) and from vocabulary (easy 0–2 harder
  words, hard 4–8). "According to the text" appears only on easy and medium items; hard items use "most likely,"
  "most strongly suggest," "inferred," or "best supports." In about 30 of 33 items all four choices share an
  opening. About 6 in 10 wrong answers are "Never said," then Opposite and Mix-up; none was "too narrow." The lure
  is a Word echo on easy/medium items and a Twin on hard ones. 7 of 8 literary excerpts came from books published
  1811–1926. → shared rules for `central_ideas` and `details` in the rulebook and a full `details` section; code
  assigns a passage shape by difficulty (accepted view then challenge is most common, mostly medium/hard) and, for
  detail items, a wording family by difficulty (easy 6 literal : 1 "Based on the text"; medium 3 : 4 : 3 other
  inference wordings; hard 0 : 5 : 4) and a focus (trait 7, viewpoint 6, reason/cause/effect 6, conclusion 3,
  other 4). Lint fails passages over 110 words, "According to the text" on hard detail items, and "always"/"never"
  in choices; it warns about too many sentences for the level and about choices that don't share an opening.
  Bank words are optional (at most one) at easy and medium, and hard items use one or two, as in the real test.
  Literary excerpts are written in an older style (about 1810–1930). `meza_run.sh` now has 4 detail slots to 1
  main idea.
- **Inferences follow a 21-item review, 7 per level (round 1).** Every real item ends its last sentence with
  the blank; no passage was a literary excerpt; passages run about 50–120 words (usually 85–95) at every level,
  with hard items denser (2–4 long sentences, formal logic phrasing) rather than longer. Hard items make the
  student build a link (a premise carried from the first sentence, or only one side of a test stated). Numbers
  are background. About half of correct answers hedge (a mild tell, unlike Central Ideas), and correct answers
  never rest on absolutes. More than half of wrong answers are Never said; the lure is Real-world true on
  easy/medium, the restated old view on medium, and a direction-flipped Twin or Half right on hard. The
  hand-written round 1 laid out both outcomes of a test in its hard items, used one hard template twice, leaned on
  Opposite lures, and made answers hinge on numbers. → `inferences` rewritten in the rulebook; code assigns the
  ending's job by difficulty (what a finding means, explain a puzzle, predict, what a view leaves out, what an
  example shows, motive or takeaway), a lead-in ("suggests" 3 : logic word 2 : other 1, checked in code), a blank
  form (clause 11 : verb 9), and a shape (elimination puzzles hard only); literary slots become informational
  passages about an author or work. Lint fails passages over 120 words, a blank that isn't at the very end, a
  literary excerpt, and absolutes in the correct answer; it warns about too many sentences for the level. On hard
  inference items only, a correct answer that is clearly the longest is a warning, not a failure. Researchers stay
  generic ("a team of historians").
- **Command of Evidence follows a 21-item review, 7 per level (round 1).** About 40% findings, a third tables or
  graphs, a quarter quotations (easy and medium only); no "weaken" item appeared. Unlike the other Information and
  Ideas skills, text length grows with difficulty (about 50 words easy, 70 medium, 100–120 hard). Correct findings
  are flat facts that don't hedge; well over half of wrong answers are "True but irrelevant," and misreads appear
  only in data items. Medium findings test the claim head-on; a stated rival explanation, a chain, or a gap filled
  by a general mechanism appears only on hard. Real quotation items name the work, author, year, and genre and end
  the claim with a colon and the blank. The hand-written round 1 put a rival explanation in a medium item,
  explained a table's logic in its own text, made students subtract, used obvious misreads, and gave its quotation
  item no named work. → the three `command_of_evidence_*` sections and a shared section rewritten; code assigns
  the finding stem (support most often; underlined claim, expert quote, and rare weaken variations) and logic by
  difficulty, the table task by difficulty (easy lookup, complete, support, hard apply-a-proposal), and the
  quotation genre (poems and prose); a quotation slot that lands on hard is written as medium. Quotation items use
  a made-up title and author (allowed by the legal rules below) because the generator can't check real quotes.
  Tables only for now: bar graphs need image support in the export and the portal. Lint checks length by kind
  (findings up to 125 words, quotations 15–60, table text up to 125), the quotation stem and colon-plus-blank
  ending, and no longer counts absolute words inside literary quotes as a tell. `meza_run.sh` now has 3 finding
  slots, 2 table slots, and 1 quotation slot.
- **Spread the correct answers across A–D.** A batch of 20 should not stack up one letter or put the same
  letter several times in a row; satgen shuffles each question on its own, which allows both. → `key_plan`
  assigns letters in blocks of four (each letter once per block), and `MezaPipeline.shuffle` puts each key at its
  planned letter, so a batch is evenly mixed and no letter appears three times in a row. Questions screened out
  later leave small gaps, so a shipped batch is close to even rather than exact. `python meza_quality.py
  runs/<run>/questions.json` reports the letter mix and the longest run.
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
- Information and Ideas: rules for all three skills (Central Ideas and Details, Inferences, Command of
  Evidence) come from hand-written rounds; all still need a real generator run.
- Bar graphs for Command of Evidence (real items use them nearly as often as tables, including grouped bars on
  hard items): the writer would output chart data, and the export would draw an image; check the portal first.
- Central Ideas and Details literature is mostly prose (7 of 8), but the subject rotation gives poems half of the
  literary slots. Easy items are mostly literature (6 of 11), but the rotation ignores difficulty.
- Bring the same review loop to the other R&W skills (transitions, text structure, ...).
- Later: calibrate difficulty with real student results from the portal; a web tool for tutors.
