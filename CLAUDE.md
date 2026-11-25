# Meza Scholars satgen — project notes

Read this first. It records how the generator works, why each rule exists, and how we improve it.
Update it in the same pull request whenever you add a rule or change how something works.

## What this repo is

- `src/satgen/` is the open-source satgen pipeline (MIT license, upstream: github.com/Dulkinsar/sat-question-pipeline).
  It writes Digital SAT questions, solves each one 5 times without the key, has a stronger model review it with the
  key hidden, confirms any key change, and removes near-duplicates. **Don't edit `src/satgen/`.** Pull upstream
  fixes with `git fetch upstream && git merge upstream/main`.
- Everything Meza-specific lives in the `meza_*` files, `data/meza_vocab_bank.json`, and `tests/test_meza_quality.py`.
- Scope right now: **Reading and Writing**, and **Math, starting with Algebra** (`meza_math.py`, Nov 25).

| File | What it does |
|---|---|
| `meza_run.sh` | Entry point. `./meza_run.sh 20` = 20 questions in the full-test mix; `./meza_run.sh 8 words_in_context_completion` = only that type. Each run goes to `runs/<date_time>/`. |
| `meza_satgen.py` | Wraps satgen: assigns subject areas, clue structures, passage shapes, question wordings, vocabulary, and answer-choice plans in code; runs the quality gate; rewrites failing drafts. |
| `meza_quality.py` | Free, deterministic checks (lint, rhythm, vocabulary) and the passage-free probe. `python meza_quality.py runs/<run>/questions.json` lints an old run with no API calls. |
| `meza_context.txt` | The item-writing rulebook the writer model follows. |
| `meza_export.py` | Builds `review.html` and `questions.csv` for human review. |
| `meza_math.py` | Math questions built in code: numbers, answer, and wrong choices are computed, so every key is right. `python meza_math.py run --count 12` writes `runs/math-<date_time>/` with review.html and makes no API calls (template stories); `--ai-stories` lets the writer model word the word problems (uses credits); `--families linear_rate`. See "Math" below. |
| `data/meza_vocab_bank.json` | 1,961 unique words merged from three public Anki decks Sebastian chose (words and parts of speech only), tiered by word frequency: 1,069 advanced, 691 mid, 201 common; 103 flagged as having a second meaning (used for underlined-word items). |

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
- **Harder passages need harder words, but only a sprinkle** (bank since replaced; see Source rule). Round 3: everything was readable by any student.
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

- **First real run (Nov 24, 10 questions): 5/5 vocabulary shipped, 0/5 Text Structure.** Fixes:
  - Words in Context passages were built around the tested word (every sentence hinting at "a sound mentor").
    → code assigns a word role (detail 4:1 at medium/hard, 1:1 at easy); for "detail" the word sits in one sentence
    and only that sentence gives the clue; a reviewer-model focus check rejects passages built around the word
    (MEZA_FOCUS_CRITIC=off to skip).
  - Literary introductions counted toward passage length, leaving one-sentence excerpts. → the intro is excluded
    from all length checks; excerpts need 40+ words of their own.
  - Text Structure: the trap critic failed any item with one weak wrong answer and rewrites redid every choice, so
    nothing converged. → medium may keep one wrong answer rated 2 (never 1); hard still needs all >= 3; rewrites
    change only the flagged choices; Text Structure gets one extra rewrite; broken-JSON drafts are retried for free.
  - The quality log now records every problem in full ("problems" field).

- **Second real run: 1 of 10 shipped.** Stacked judgment checks rejected nearly everything. → the reviewer-model
  critics and the passage-free probe are now "soft": after the last rewrite, a question whose only remaining
  problems are soft ships with a yellow Review flag in review.html (and "FLAGGED: ..." in the CSV's Review status)
  instead of being discarded. Format, length, stems, vocabulary, and rhythm rules still block. Also fixed: bank
  phrases like "grapple with" could never be matched in a passage.

## Math (Sebastian, Nov 25)

Sebastian paused the reading work to make the math generator really good, starting with Algebra. Sources: his own
"Mastering SAT Algebra" lesson deck (Meza's own material: 8 topics, the golden rule "know exactly what the question is
asking," Desmos tips) and example questions he described.

- **Built in code, not written by the AI.** Reading answers need judgment; math answers can be computed. Each question
  family picks numbers that come out even, computes the answer exactly (fractions, no rounding), and builds each wrong
  choice from a named student mistake, so the explanation can say which mistake each wrong answer comes from.
  `tests/test_meza_math.py` re-solves every equation with a separate parser, so the key is checked twice.
- **Families so far** (skill: Linear equations in one variable):
  - `expression_value` ("If 4x + 5 = 29, what is the value of 9x − 7?"). Easy: solve, then substitute. Medium: x is
    an awkward fraction and the target is a multiple of the given side plus a constant, so scaling is the fast route.
    Hard: parentheses and x on both sides. Main trap at every level: the value of x (the golden rule); others: the
    constant dropped or with the wrong sign, a distribution error, a sign error when moving a term.
  - `linear_rate` (Sebastian's trucker: a job done at a constant rate). Easy: rate given; when will N be left?
    Medium: start amount and one later reading, so the rate must be worked out; how many hours after starting will
    N be left? Hard: two readings, start not given; when will the job be finished? Main trap: "how many MORE hours"
    versus "how many hours after starting"; others: time to do the remaining part, the whole job, the rate itself,
    using the amount left as the amount done.
  - Added from Sebastian's list (Nov 25):
    - `solve_simple` ("r + 23 = 61. What is the value of r?"; medium "45 − r = 17"): traps are the wrong operation,
      subtracting in the wrong order, stopping at −r, and copying the right side.
    - `solve_combine` ("7x − 3x + 5 = 29. What value of x is the solution to the given equation?"; medium: the
      combined coefficient is negative): adding the coefficients, sign slips, not dividing, leaving out a term. The
      constant is a multiple of the combined coefficient so the mistakes come out as whole numbers.
    - `one_solution_cannot` ("x(r + 3) + 5 = 7x − 2 ... exactly one solution ... CANNOT be the value of r?"; hard
      adds another x-term): r that makes the x-coefficients equal; traps ignore the number in the parentheses,
      flip its sign, or flip the answer's sign. Multiple choice only.
    - `infinite_constant` ("3(2x + 5) − x = kx + 15 ... infinitely many solutions ... value of k?"; hard: "= 5(x + k)",
      so k is inside parentheses): traps leave out a term, skip distributing, or forget to divide.
    - `three_kinds_equation` (three kinds of one thing; the first is m times the second, the third is known, the
      total is known; "which equation represents this situation?"; medium: x is the first kind, so the second is
      x/m): choices are equations; traps add m instead of multiplying, reverse which is m times the other, leave
      out a kind, or add the known amount to the total.
    - `payment_equation` (a down payment, sign-up fee, or first deposit plus equal repeated payments equals a total;
      "which equation?"; medium: the payment amount is unknown and the number of payments is given).
    - `interpret_parts` (the same situations as an equation: "What does 625 / 25m / 1,075 represent?"; medium asks
      about the product, the part students mislabel most). Wrong choices are what the other parts represent.
    - `in_terms_of` ("3(2x + 10) − 4 = 5(x + 5) + 8. If x is the solution to the given equation, what is the value
      of x + 5?"): both sides are built from the same group, so students can let k = x + 5 (2x + 10 = 2k) and solve
      for k directly, or solve for x and add. Medium: the left group is 2(x + c); hard: 2 or 3 times, c and the
      answer can be negative. Traps: the value of x (the golden rule), the value of 2x + 10, forgetting the factor
      of 2, a sign slip on a constant, the opposite sign.
    - `reciprocal_ratio` ("If x/4 = 6, what is the value of 4/x?"; medium asks c/x with a different numerator, so
      the flip alone is not enough): traps are the value of x/4, the value of x, flipping when it doesn't apply.
      Fraction choices are natural here.
    - `same_product` ("If 6(4x) = 72, what is the value of 4x?"; medium adds a constant): no need to find x; traps
      are x itself, dividing by the wrong number, multiplying again, not dividing.
    - `no_solution_parens` ("(r − 4)x = 9 ... no solution ... value of r?"; medium "(r + 3)x = 7x + 5"): the
      x-coefficient must be 0 (or match the other side); traps are the wrong sign, setting the coefficient equal to
      the constant or to 1, ignoring the number in the parentheses.
    - `no_solution_k` ("3(kx − 2) = 5x + 1 ... no solution ... value of k?"): set the x-coefficients equal, 3k = 5;
      medium k is a whole number, hard a fraction; traps forget the 3, flip the fraction, multiply, or flip the sign.
    - `how_many_solutions` (both sides built up; "How many solutions does the given equation have?"): choices are
      always Zero / Exactly one / Exactly two / Infinitely many in that order; the key is decided by whether the
      x-coefficients and constants match. Hard adds an x-term to each side.
    - `infinite_two_constants` ("a(2x + 3) + 4x = 14x + b ... infinitely many solutions ... value of b / a / a + b?"):
      match the x-terms to find a, then the constants to find b. Medium has no extra x-term; hard adds it and can
      ask for a + b.
    - `fraction_decimal`: easy "0.4x + 1.2 = 3.6" (choices shown as decimals; traps move the decimal point,
      stop before dividing, or divide too early), medium "x/3 + x/4 = 7" (traps add the denominators, leave out a
      term, multiply only one side by the LCD), hard "(x + 4)/3 = (x − 2)/5" (traps cross multiply the wrong way or
      without distributing, or give the value of each fraction).
    - `word_solve` (word problems that ask for a number): easy a fixed fee plus a cost per unit (phone texts, taxi
      miles, plumber hours, parking hours; each setting has a realistic fee range, and the fee is at least two units
      of the rate); medium a rectangle whose length is m times or k more than its width, given the perimeter, asking
      for the width or the length; hard the sum of three or four consecutive (even/odd) integers, asking for the
      least or greatest. Traps: leaving out the fee, the other dimension, treating it as a square, counting each
      side once, the middle integer, the other end, counting by 1 instead of 2. Stories can be AI-worded like the
      rest (`--ai-stories`); story checks now handle decimals such as $0.10.
    - `which_has_property` ("Which equation has no solution?", or infinitely many, or exactly one): four equations
      with the same left side; each choice's right side changes the x-coefficient, the constant, or both, and each
      wrong choice's explanation names the property it does have. Hard builds up the left side and factors some
      right sides.
  - Equations are shown on their own line above the question (the `passage` field), as on the SAT. Families with
    choices that are equations or phrases put the key at the letter used least so far in the batch.
    `tests/test_meza_math.py` checks every choice by substitution: only the key solves the equation, makes the
    coefficients match, or holds for the true value in the story.
- **Linear functions** (Sebastian, Nov 25; skill tag "Linear functions"):
  - `interpret_models` ("Which of the following is the best interpretation of 2.5 in this context?"): real-world
    models in y = mx + b form (two plants growing, two candles burning, two tanks draining, two friends saving, two
    hikers descending; decreasing models print a negative slope, and the question asks about the number as printed).
    Hard, as Sebastian described: two models in about four long sentences, so the wrong answers can also put the
    number in the other thing's equation (always one such choice) or turn it into a comparison ("grew 2.5 cm more
    each week than plant B"). Medium: one model. Other traps: slope vs. starting value, a total instead of a rate,
    the rate flipped ("took 2.5 weeks to grow 1 cm"), the value at the end instead of the start. Choice explanations
    for phrase or equation choices now give only the letter and the reason.
  - **Graphs** (Sebastian: "definitely super important"): `graph_svg` draws an SAT-style graph in code: a light
    grid with at least 10 lines each way (10-14), dark axes, numbered ticks (every other line when crowded), italic
    axis titles with units, the line clipped to the grid, optional dots. Steps are 1, 2, 2.5, or 5 times a power of
    10, chosen per problem. Lines pass through grid intersections, so the value a student reads is exact. The SVG is
    stored in the question's `figure` field and shown in review.html and the copy-for-slides block.
  - `graph_estimate` ("Based on the graph, what is the estimated value of y when x = 160?"; hard asks for x given
    y): six settings (van fuel, pool filling, phone battery, tree height, plane altitude, kayak cost). Traps: the
    grid line above or below, one vertical grid line over, the starting value, the wrong axis.
  - `graph_slope` ("what is the change in the amount of fuel for each additional mile driven?"; hard: "each
    additional 40 minutes"): choices are phrases with a direction and an amount; traps are the opposite direction,
    the rate flipped, the starting value, the change over the whole graph, the change for one grid step or one unit.
  - `rate_total_function` ("$18 per hour; 6 hours Saturday and x hours Sunday. Which function f gives the total?"):
    f(x) = 18x + 108 or 18(x + 6); traps add the hours instead of the earnings, add to the rate, leave out a day.
  - `intercept_meaning` (best interpretation of the x- or y-intercept of y = f(x) for a draining tank, candle, gift
    card, phone battery): traps include the other intercept's true statement and the right number in the wrong role.
  - `two_points_function` ("f(5) = 7 and f(2) = 19. Which equation defines f?"): traps flip the slope's sign, use an
    output as the y-intercept, or flip the slope.
  - `intercept_from_slope` ("slope 3, y-intercept (0, −12): x-coordinate of the x-intercept?", or the reverse;
    hard uses fraction slopes).
  - `cost_function`: easy a one-time fee plus a monthly charge; medium Sebastian's "first 2 (or 3) hours at one rate,
    then another rate, x = hours past the first 2," so the first hours' cost becomes the y-intercept; hard the same
    with x = total hours (x ≥ 2), simplified (C(x) = 10(x − 3) + 105 = 10x + 75).
  - `function_change` (F(c) = (9/5)c + 32, a print shop's cost, K(f) = (5/9)(f − 32) + 273.15, ocean pressure
    0.43d + 14.7: "If the input increases by 18, by how much does the output increase?"): slope × change; traps add a
    visible constant (273.15) or give the input change.
  - `table_equation` (a titled table of real-life values: hours studied and test scores, oven temperature, ticket
    costs, a draining tank; "Which equation represents the relationship?"): rows step by 2 or 3 on hard, so the
    change between rows is a trap; the first y-value (x ≠ 0) as the y-intercept is another.
  - Sebastian's alternatives (Nov 25): `table_mb` (same tables, "f(x) = mx + b ... What is the value of m / b?"),
    `table_shift` (a table of f, then g(x) = f(x) ± c (medium) or f(x − c) / f(x + c) (hard); "Which equation
    defines g?"; traps shift the wrong way or forget to multiply the inside shift by the slope), and `sum_functions`
    (f and g given, h(x) = f(x) + g(x) or f(x) − g(x); the slope, the y-intercept, or the x-intercept of h; traps
    multiply, use only f, or combine the two x-intercepts).
  - More from Sebastian (Nov 25): `graph_which_equation` (a line in the xy-plane; "What is an equation of the line
    shown?"; hard choices in standard form; traps flip the slope's or intercept's sign or flip the slope),
    `evaluate_function` (easy f(6); medium f(k) = 31, find k; hard f(x) = 4x + c with f(3) = 19, find f(5)),
    `unit_conversion_function` (inches a week, height in feet: h(w) = (2w + 18)/12 on medium, simplified on hard;
    traps leave it in inches, convert only the rate or only the start, or multiply), and `two_lines_graph` (two
    labeled lines on one real-world graph: when are they equal, or how many more per 4-week period, so the
    per-week difference is a trap).
- **Linear equations in two variables** (Sebastian, Nov 25):
  - `two_var_equation`: easy "3x + 5y = 240" from a story; medium "at least"/"at most" makes it ≥/≤ (the flipped
    sign is the main trap); hard the limit is in a bigger unit (pounds of flour, tons, hours) and must be converted.
  - `table_for_equation`: the choices are x/y tables (review.html draws them as tables); traps flip the slope's or
    intercept's sign, swap x and y, or swap slope and intercept; hard gives the equation in standard form.
  - `tradeoff` (6x + 8y = 600: buy 4 more pounds of chicken, how many fewer pounds of beef? a·k/b; medium the other
    way; hard per single unit, a fraction): k is chosen so the flipped ratio, the main trap, is also whole.
  - `slope_point_line` (slope and a point; hard: parallel to a standard-form line), `isosceles_perimeter` (2x + y = P;
    medium perimeter in feet; hard the third side in feet: 2x + 12y = P), `perpendicular_vertical` (perpendicular to
    x = 4: slope 0, or y = −3 through a point), `perpendicular_lines` (negative reciprocal; medium the full
    equation through a point; hard the given line in standard form), `interpret_coefficients` (6x + 15y = 300:
    what does 15 / 300 / the point (20, 0) mean?), `table_unknowns` (a linear table with a missing k; hard puts k
    between unevenly spaced x-values, so "halfway between" is the trap).
- **Systems of two linear equations** (Sebastian, Nov 25; skill "Systems of two linear equations in two variables"):
  - `system_which_property` ("Which system has no solution / infinitely many / exactly one?"). Every choice shares
    the first equation and the second equations are near misses of each other (the same line scaled, the same slope
    with another constant, a sign flipped, the coefficients swapped), per Sebastian: wrong answers must be close.
  - `system_coordinate` (x or y of the solution; medium sometimes asks for 2x − y; hard sometimes asks for the sum
    or difference of the two left sides, so adding the equations is the shortcut), `system_cancel_letters` (medium
    "2x + ky = 6 + 5k; 7x + ky = 21 + 5k": subtract and k cancels; hard "ax + by = 3a + 5b; 2x − by = 6 − 5b" with a
    and b positive: add, factor (a + 2)x = 3(a + 2), so x = 3, then plug back in for y), `system_constant_r` (the
    constant for no solution or infinitely many; hard a non-whole ratio or "exactly one solution ... CANNOT be r"),
    `system_solve_pair` (choices are ordered pairs; easy can be a graph; traps swap x and y, flip signs, or are on
    only one line), `system_word_setup` (pick the system: a total x + y = 120 plus a value or percent equation,
    0.3x + 0.25y = 33; hard uses mixtures, interest, survey percents, nickels and dimes; traps swap the totals or
    rates, write percents as whole numbers, or move the decimal point), `system_graph_equations` (two lines on a
    graph, choices are systems: easy the xy-plane, medium a real-world graph, hard "which system has the same
    solution as the system graphed?", where the traps' solutions are the point with x and y swapped or a sign
    flipped), and `system_times_more` ("three times as many chose soccer as tennis, and 40 more chose soccer: how
    many chose tennis?"; easy gives a total; hard "4 more than three times"; traps divide by 3 instead of 2, give
    the other group, or leave out the extra).
  - Graphs now also come in the full xy-plane (axes through the origin, −8 to 8, "x"/"y" and O), and two lines on a
    real-world graph get a key below it (solid and dashed). A choice that is a table or a system renders as one in
    review.html. Tests solve every system and check every choice independently.
- **Choices.** Numbers in ascending order, as on the SAT. A whole-number answer gets whole-number wrong choices (a
  stray fraction stands out), except medium `expression_value`, where x itself is the trap. Keys: among the letters
  the real mistakes can produce, each question takes the letter used least so far in the batch, so keys spread over
  A-D without filler; a labeled near miss fills in only when fewer than three real mistakes exist (under 6% of
  questions). About 1 in 4 questions is grid-in (student-produced response).
- **Stories.** Word problems get 4-5 sentences about a named person in a rotating setting (trucker, food bank
  volunteer, print shop, warehouse, baker, farmworker; a batch uses every setting before repeating). The writer model
  (Sonnet; MEZA_MATH_WRITER) wraps the fixed numbers; code checks that every planned number appears exactly once, in
  the planned order (first story run: one draft relabeled the numbers and kept the same answer by coincidence), with
  no other numbers and thousands commas, and a blind solver (Haiku; MEZA_MATH_SOLVER) must reach the computed
  answer in 2 of 3 tries. Otherwise the plain template story is used. Runs are free by default (template stories,
  no API calls); AI stories only with `--ai-stories` (Sebastian, Nov 25: work on the generator, not on generating
  batches, to save API credits; test changes offline). First runs: 15 of 16 stories passed on the
  first try. Runs: `runs/math-first`, `runs/math-stories`.

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
- Information and Ideas rules (Central Ideas and Details, Inferences, Command of Evidence): kept. The "real
  practice items" they were built from (33, 21, and 21 items) are Meza's own questions, not College Board material
  (Sebastian, Nov 24).
- Literary introductions differ by skill on purpose (for now): Words in Context and Text Structure use a specific
  invented author/title/year intro; Central Ideas and Details use a plain intro ("from a novel"). The lint enforces
  the specific intro only outside Central Ideas and Details.

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

- Math: solving a formula for one variable ("A = ½h(b + c); which equation gives h?") belongs with Advanced Math
  (equivalent expressions), per Sebastian; build it there.
- Math: graphs need image support in the portal when questions move there (the review page shows the SVG).
- Math: the rest of Algebra from his deck (linear inequalities, absolute value, function notation and composite
  functions), then the other math domains. He describes question types; Claude builds them in code and tests offline
  (no API batches).
- Math: more two-variable and systems types worth adding: the x- and y-intercepts of ax + by = c, which point lies on
  a line, the line through two points, parallel lines (same slope, the system has no solution), systems of
  inequalities (which point is in the solution region; needs shaded graphs).

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
