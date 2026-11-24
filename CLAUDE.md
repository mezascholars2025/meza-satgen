# Meza Scholars satgen — project notes

Read this first. It records how the generator works, why each rule exists, and how we improve it.
Update it in the same pull request whenever you add a rule or change how something works.

## What this repo is

- `src/satgen/` is the open-source satgen pipeline (MIT license, upstream: github.com/Dulkinsar/sat-question-pipeline).
  It writes Digital SAT questions, solves each one 5 times without the key, has a stronger model review it with the
  key hidden, confirms any key change, and removes near-duplicates. **Don't edit `src/satgen/`.** Pull upstream
  fixes with `git fetch upstream && git merge upstream/main`.
- Everything Meza-specific lives in the `meza_*` files, `data/meza_vocab_bank.json`, and `tests/test_meza_quality.py`.
- Scope right now: **Reading and Writing only.**

| File | What it does |
|---|---|
| `meza_run.sh` | Entry point. `./meza_run.sh 20` = 20 questions in the full-test mix; `./meza_run.sh 8 words_in_context_completion` = only that type. Each run goes to `runs/<date_time>/`. |
| `meza_satgen.py` | Wraps satgen: assigns subject areas, clue structures, vocabulary, and answer-choice plans in code; runs the quality gate; rewrites failing drafts. |
| `meza_quality.py` | Free, deterministic checks (lint, rhythm, vocabulary) and the passage-free probe. `python meza_quality.py runs/<run>/questions.json` lints an old run with no API calls. |
| `meza_context.txt` | The item-writing rulebook the writer model follows. |
| `meza_export.py` | Builds `review.html` and `questions.csv` for human review. |
| `data/meza_vocab_bank.json` | 1,961 unique words merged from three public Anki decks Sebastian chose (words and parts of speech only), tiered by word frequency: 1,069 advanced, 691 mid, 201 common; 103 flagged as having a second meaning (used for underlined-word items). |

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
- **Harder passages need harder words, but only a sprinkle** (bank since replaced; see Source rule). Round 3: everything was readable by any student.
  Round 4: three bank words per passage was too much. → medium: 1 advanced word max; hard: 1–2 max.
- **Balance answer-choice difficulty.** Round 4: all four hard choices were obscure, which real tests rarely do.
  → hard WIC choices = 2 advanced + 1 mid + 1 common; medium = 1/2/1; easy = 0/1/3. The key's tier is assigned
  (hard: usually advanced, sometimes mid or common) so "pick the word I don't know" stops working.
- **Underlined-word items get the same treatment.** Exact stem, one `<u></u>` matching the stem, choices of
  ≤4 words, at least one wrong choice is the word's most common meaning, ≤1 advanced word among choices
  (difficulty comes from the uncommon sense). Tested words come from the bank's common/mid tiers.

- **Text Structure and Purpose: what gets underlined (Sebastian's note).** Function items usually underline a
  sentence, but real tests occasionally underline a question the text poses ("the underlined question") or a
  phrase/quotation ("the underlined portion"). → assigned in code by weight (sentence 8, question 1, portion 1);
  lint checks that the stem word matches what is underlined.

- **Text Structure and Purpose: choice verbs and length (Sebastian's note from practice sets).** Choices open with
  "To [verb]" (purpose) or "It [verbs]" (function, overall), mostly from: suggest, indicate, emphasize, convey,
  describe, illustrate, summarize, present, reflect, represent, introduce, establish, demonstrate, provide, discuss,
  evaluate, compare, explain, capture. Real items come in two length styles: ~9-12 words or ~15-22 words.
  → code assigns a band per item (overall structure = long; purpose/function = short or long at random) and lint
  enforces it (±1 word) plus ≥2 preferred verbs; the correct answer's opening verb rotates (can't repeat the last 3).

- **Literary passages: specific intros and denser poetry (Sebastian's note).** Round 1's poem intro ("The speaker
  addresses an old clock") was too vague and the language too plain. → intros must name an author, a year
  (1820-1925), the form, and a quoted title, then one specific sentence on who the speaker/character is and their
  situation. Authors and titles are INVENTED (never a real writer or work, so no invented lines are attributed to a
  real author) and the explanation ends with [Invented work]. Poems/drama use period diction (inversion,
  apostrophe, extended metaphor) and get more bank words: medium up to 2 advanced, hard 2-3.

- **Text Structure wrong answers must be near misses (Sebastian, round 2).** Distractors like "to describe the
  plants" (no plants in the text) or "compares two engineers' competing plans" (only one plan) can be crossed out
  without reading. → the writer must build wrong answers from the passage's own content and label each with a trap
  type (too_narrow, too_broad, wrong_action, wrong_relationship, wrong_order, partial, wrong_target,
  misread_stance); medium/hard need 2+ types. Then the reviewer model rates each wrong answer (on topic? tempting
  1-5?) and off-topic or untempting ones go back for a rewrite (easy may keep one plain trap). A word-overlap check
  was tried and rejected: correct answers are worded abstractly, so overlap can't separate good traps from bad.
  Turn the critic off with MEZA_TRAP_CRITIC=off.

- **The skim test (Sebastian, round 3).** Even "on-topic" traps were weak when they leaned on a word the text barely
  touches ("compares the godwit with other animals" from "of any animal") or contradicted an explicit statement.
  Real wrong answers are statements a student who skimmed would accept as accurate: their facts match the passage,
  only the purpose/scope/order/role is wrong. → rule + writer prompt; the critic now also rates accurate_content,
  requires 2+ of 3 wrong answers to be accurate, and scores anything contradicting the text or built on a passing
  word as tempting 1-2. Also: hand-written review rounds must shuffle key positions (satgen already shuffles real runs).

## Source rule (Sebastian, Nov 24; do not break)

Allowed as source or training material, and nothing else:
- original questions written by the Meza team or by this generator
- public test SPECIFICATIONS (the prose of College Board's Assessment Framework and Skills Insight documents),
  but NOT the sample questions printed inside them
- openly licensed content (e.g. OpenStax, public-domain passages, Creative Commons word lists), with attribution
- the MIT-licensed satgen code (its bundled eval questions are hand-written by its author, not official items)

Never allowed, directly or indirectly: College Board Question Bank items, Bluebook or other official practice-test
questions, sample questions from College Board documents, and commercial prep material (books, decks, courses).
"Indirectly" includes rules or statistics derived from studying those questions.

Also: never invent a quote, study, or statistic and present it as real (use generic researchers for invented
findings); never commit `.env`; each person uses their own API key.

Audit decisions (Nov 24):
- Vocabulary bank: the Barron's-based list was removed. `data/meza_vocab_bank.json` merges three publicly shared Anki
  decks Sebastian chose; only words and parts of speech are kept (no definitions, sentences, or audio). The decks
  state no license; Sebastian's call is that a list of English words from public decks is acceptable.
- Question stems: kept. Sebastian's call: they are the standard wording used across SAT prep, not taken from the
  Question Bank.
- Review-round rules (Nov 24): kept. Sebastian built them with Claude, without College Board material.
- Bundled satgen eval questions: checked; hand-written by the satgen author, no official items.

## How we improve it (the review loop)

1. Pick one skill. Read what the public spec text says about it (skills, formats, difficulty factors).
2. Generate 3-5 questions of that skill (or ask Claude to write a round by hand to the current rules).
3. Review them against the spec and against item-writing quality: clue directness, trap quality (skim test),
   rhythm, vocabulary level, answer-position balance. Do not compare them against official questions.
4. Turn each finding into (a) a rule in `meza_context.txt`, (b) a check in `meza_quality.py` if it can be checked in
   code, and (c) a test in `tests/test_meza_quality.py`. Add the reason and its source to the decisions log.

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
