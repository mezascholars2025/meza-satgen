"""Run satgen with the Meza Scholars rulebook, a balanced subject spread, and a quality gate.

Subject spread: satgen writes each question separately, so left alone the writer keeps
drifting back to the same science topics. This wrapper assigns every question a subject
area in code, before writing starts:
  - every batch cycles through all 10 subject areas before any repeats (balanced),
  - the order is shuffled with a new random seed each run (so it differs every time),
  - types that need informational text never get a fiction or poetry excerpt.

Quality gate (meza_quality.py), run on every draft before satgen's solvers see it:
  1. lint  - free checks for format errors and answer "tells" (key is the longest choice,
             absolutes only in wrong answers, wrong stem, too long, overused topic, ...)
  2. probe - a cheap model tries to find the key WITHOUT the passage. If it can, the
             item gives its answer away.
  A draft that fails is sent back to the writer with the exact problems to fix, up to
  MEZA_MAX_REPAIRS times. If it still fails it ends as generation_failed, with the
  reasons in the report. Every attempt is logged to MEZA_QUALITY_LOG (quality.jsonl).

Usage is identical to `python -m satgen run ...`; meza_run.sh calls it for you.

Settings (environment variables):
  MEZA_SUBJECT_SEED   repeat a run's subject order exactly
  MEZA_QUALITY        on (default) / lint (no probe) / off
  MEZA_MAX_REPAIRS    rewrites allowed per question (default 2)
  MEZA_PROBE_RUNS     passage-free attempts per question (default 3)
  MEZA_PROBE_FAIL     hits that count as a giveaway (default 3, i.e. every time)
  MEZA_QUALITY_LOG    where to write the per-attempt log (meza_run.sh sets runs/<run>/quality.jsonl)
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
from pathlib import Path

from meza_quality import (PROBE_TYPES, base_type, describe, difficulty_of, features, lint, probe_feedback,
                          probe_votes, rhythm_check, vocab_used)
from meza_traps import CRITIC_GUIDANCE, CRITIC_SYSTEM, TRAPS, critic_prompt, family_of, format_examples
from meza_traps import load_bank, pick_examples, reused_topic
from meza_quality import FOCUS_CRITIC_PROMPT, FOCUS_CRITIC_SYSTEM, fingerprint, focus_problems
from meza_quality import (TSP_BANDS, TSP_TRAPS, TSP_TYPES, TSP_VERBS,
                          lead_verb, trap_critic_problems, trap_label_problems)
from meza_quality import (advanced_in, central_stem_kind, choice_mix_problems, detail_stem_family, leadin_kinds,
                          word_tier)
from satgen import cli, pipeline
from satgen.answers import CHOICES, normalize_spr
from satgen.llm import SimulatedClient
from satgen.models import MCQ, SPR, Question

SUBJECTS = [
    "LIFE SCIENCE - biology, ecology, zoology, or botany",
    "HISTORY - an event, person, institution, or development from any era or region of the world",
    "LITERATURE: PROSE FICTION - an excerpt from a short story or novel",
    "PHYSICAL SCIENCE - physics, chemistry, or materials science",
    "SOCIAL SCIENCE - economics, psychology, linguistics, sociology, or political science",
    "EARTH AND SPACE SCIENCE - geology, climate, oceans, or astronomy",
    "HUMANITIES - visual art, music, architecture, dance, film, or philosophy",
    "LITERATURE: POETRY OR DRAMA - an excerpt from a poem or a play",
    "ARCHAEOLOGY AND ANTHROPOLOGY - ancient societies, artifacts, or cultural practices",
    "TECHNOLOGY AND ENGINEERING HISTORY - inventions, infrastructure, or how a technology developed",
]
LITERARY = {2, 7}
INFORMATIONAL_STAND_IN = (
    "LITERATURE STUDIES - an informational passage about an author, a literary work, "
    "or a literary movement (not an excerpt)"
)
# Types whose passages may be literary excerpts. Every other type gets an informational
# passage, so a literary slot becomes literature studies for it.
EXCERPT_OK = (
    "details", "central_ideas", "words_in_context", "text_structure",
    "cross_text_connections",
)  # not inferences: a review of 21 real inference items found no literary excerpts

SUBJECT_LINE = (
    "\n\nREQUIRED SUBJECT AREA FOR THIS QUESTION: {subject}.\n"
    "Write the passage on this subject area and no other, and choose a specific, less common "
    "topic within it. Do not substitute a different subject, even if another topic seems easier "
    "for this question type."
)
# Words in Context: each question gets a clue structure assigned in code, because the writer never
# sees its other questions and otherwise reuses one pattern (a colon after the blank) most of the time.
# Weights approximate the real mix; "restatement" (colon/dash explanation) is rare on the test.
NO_TWIST = (" There is NO contrast and NO 'expected, but actually' setup: do not use however, yet, but, "
            "although, though, instead, despite, or rather than, and do not open with what people once believed.")
WIC_CLUES = [
    ("contrast", 3, "a contrast or concession (although, while, yet, rather than, despite) between two ideas; "
                    "the blank sits on one side of it, and the other side tells the student the word's direction"),
    ("exposition", 3, "plain explanation: the passage simply explains a process, finding, or work, and the blank is "
                      "the precise word (often a verb or adjective) that those facts call for" + NO_TWIST),
    ("evaluation", 2, "the blank is an evaluative word about a claim, piece of evidence, method, or work (how strong, "
                      "limited, original, or useful it is); its meaning follows from what the passage reports about "
                      "that thing" + NO_TWIST),
    ("consequence", 2, "cause and effect: the blank names a cause or a result, and the passage states the other half, "
                       "so the student reasons from one to the other" + NO_TWIST),
    ("delayed", 1, "the blank is in the FIRST sentence and the only clues come in later sentences" + NO_TWIST),
    ("inferred_trait", 1, "the blank describes a person's attitude, manner, or stance, which the student must infer "
                          "from what the person does or says; the actions imply the trait without listing its features"),
    ("extension", 1, "the sentence with the blank builds on or intensifies an earlier point (not only... but; indeed; "
                     "in fact), so the word goes a step further in the same direction" + NO_TWIST),
    ("restatement", 1, "restatement: a colon or dash after the blank leads into an explanation of it (this is the "
                       "only structure allowed to do that)"),
]
# Function items: what gets underlined. A question the text poses shows up occasionally on the real test.
FUNCTION_KINDS = [
    ("sentence", 8, "one complete declarative sentence"),
    ("question", 1, "one question that the text itself poses (it ends with a question mark); the stem must say "
                    "\"underlined question\""),
    ("portion", 1, "a phrase or a quotation within a sentence, not a whole sentence; the stem must say "
                   "\"underlined portion\""),
]
TRAP_LINE = (
    "\nWRONG ANSWERS FOR THIS ITEM: build every wrong answer from the passage's own content (its real subject, "
    "details, people, and moves). Each must be wrong only about the text's purpose, structure, or the role of a part, "
    "never because it mentions something the passage does not. Use these trap types:\n{traps}\n"
    "SKIM TEST: at least two wrong answers must be accurate statements about what the passage says, wrong only in "
    "purpose, scope, order, or role, so a student who skimmed would accept them. Never build a wrong answer on a word "
    "the passage barely touches, and never let one contradict an explicit statement.\n"
    "{level_rule} Add a \"traps\" field to the JSON mapping each wrong answer's letter to its trap type."
)
TRAP_LEVEL_RULE = {
    "easy": "Even at easy, every wrong answer stays on topic; easy items just use plainer traps (too narrow, wrong target).",
    "medium": "Use at least two different trap types; at least one wrong answer should need a second look.",
    "hard": "Use at least two different trap types; every wrong answer should be a near miss that is right about "
            "most of the text.",
}
# Every other skill with a trap list (vocabulary, Central Ideas and Details, Inferences, Command of Evidence).
TRAP_LINE_GENERAL = (
    "\nWRONG ANSWERS FOR THIS ITEM: {guidance} Build each wrong answer on one of these trap types:\n{traps}\n"
    "No filler: every wrong answer must target a real mistake a careful student could make.\n"
    "{level_rule} Add a \"traps\" field to the JSON mapping each wrong answer's letter to its trap type."
)
TRAP_LEVEL_RULE_GENERAL = {
    "easy": "Easy items may use plainer traps, but each must still be on the passage's topic.",
    "medium": "Use at least two different trap types; at least one wrong answer should need a second look.",
    "hard": "Use at least two different trap types; every wrong answer should be a near miss.",
}
TRAP_EXAMPLES = int(os.environ.get("MEZA_TRAP_EXAMPLES", "2"))  # proven examples shown per question (0 = none)
TRAP_CRITIC = os.environ.get("MEZA_TRAP_CRITIC", "on").lower()  # on / off
TSP_LINE = (
    "\nANSWER CHOICE LENGTH FOR THIS ITEM: every choice is {lo}-{hi} words ({style})."
    "\nOPENING VERBS: give each choice a different opening verb, mostly from: {verbs}.{avoid}"
)
FUNCTION_LINE = "\nREQUIRED UNDERLINED PORTION FOR THIS ITEM: {desc}."
# Where the tested word sits (Sebastian's note): on real items it is usually a detail in one sentence, not the
# passage's main idea. Easy items may build the passage around it more often.
WORD_ROLE_WEIGHTS = {"easy": {"detail": 1, "central": 1}, "medium": {"detail": 4, "central": 1},
                     "hard": {"detail": 4, "central": 1}}
ROLE_LINE = {
    "detail": ("\nROLE OF THE TESTED WORD: a DETAIL. The passage's main idea is about something else (a finding, an "
               "event, a person's work). The tested word sits in ONE sentence as part of that larger point, and only "
               "that sentence, plus at most one nearby phrase, gives the clue. No other sentence should exist mainly to "
               "hint at the word's meaning, and do not attach the word to an appositive that sums up the passage."),
    "central": ("\nROLE OF THE TESTED WORD: central. The passage may be built around the idea the word names, but "
                "still give only one or two clues, never a sentence that restates the word's meaning."),
}
FOCUS_CRITIC = os.environ.get("MEZA_FOCUS_CRITIC", "on").lower()
CLUE_LINE = "\nREQUIRED CLUE STRUCTURE FOR THIS WORDS IN CONTEXT ITEM: {desc}."


def clue_plan(count: int, seed: int) -> list[int]:
    """Weighted blocks: every 14 questions use each structure its weight's number of times, shuffled."""
    rng = random.Random(f"clue:{seed}")
    block_src = [i for i, (_, w, _) in enumerate(WIC_CLUES) for _ in range(w)]
    plan: list[int] = []
    while len(plan) < count:
        block = block_src[:]
        rng.shuffle(block)
        plan.extend(block)
    return plan[:count]


# Main idea items: the stem is assigned in code so the real mix holds (about 16 in 20 standard, 3 in 20
# "Which question...", 1 in 20 a small variation), instead of the writer always using the standard one.
CENTRAL_STEM_PLAN = [
    ("standard", 16, "Which choice best states the main idea of the text?"),
    ("question", 3, "Which question does the text most directly attempt to answer? (If the passage centers on one "
                    "study, you may use instead: Based on the text, which research question was [the researchers'] "
                    "study most likely intended to answer?)"),
    ("variant", 1, "Which choice best describes the main idea of the text? (If the passage centers on one person, "
                   "you may use instead: Which choice best states the text's main idea about [that person]?)"),
]
STEM_LINE = "\nREQUIRED STEM FOR THIS ITEM: {stem}"


def stem_plan(count: int, seed: int) -> list[int]:
    """Weighted blocks, as for clue structures: every 20 questions use each stem its weight's number of times."""
    rng = random.Random(f"stem:{seed}")
    block_src = [i for i, (_, w, _) in enumerate(CENTRAL_STEM_PLAN) for _ in range(w)]
    plan: list[int] = []
    while len(plan) < count:
        block = block_src[:]
        rng.shuffle(block)
        plan.extend(block)
    return plan[:count]


# Central Ideas and Details (main idea + detail items). Weights come from a review of 33 real items at all three
# levels; the shape and the detail question are assigned in code so the writer doesn't settle into one pattern.
CID_SHAPES = {
    "challenge": "an accepted view or expectation, then a challenge to it or a new finding that adds to or changes it",
    "description": "a description or claim, then supporting details or a conclusion (no twist needed)",
    "contrast": "a contrast of two things (two places, two people, two approaches, or two groups' views), kept "
                "clearly apart",
    "finding": "a study's setup, then its finding (it may end by ruling out another explanation)",
    "cause": "a problem, then its cause (and possibly a proposed fix), or a claim followed by a chain of reasons",
    "scene": "one short scene that shows a character's reaction to something",
    "sketch": "a sketch of one character's nature or habits",
    "metaphor": "an extended metaphor or image that carries the excerpt's feeling",
}
TWIST_SHAPES = ("challenge", "contrast")  # built on a contrast, so the no-third-twist rule is waived
CID_SHAPE_WEIGHTS = {
    "easy": {"description": 3, "finding": 1, "contrast": 1, "challenge": 1},
    "medium": {"challenge": 3, "contrast": 2, "description": 1, "finding": 1, "cause": 1},
    "hard": {"challenge": 4, "finding": 2, "contrast": 1, "cause": 1},
    "literary": {"scene": 4, "sketch": 3, "metaphor": 1},
}
SHAPE_LINE = "\nREQUIRED PASSAGE SHAPE FOR THIS ITEM: {desc}."
DETAIL_FAMILIES = {
    "literal": 'Begin the question with "According to the text," (for example: what is true about [a character]? / '
               "why are [a group] worried about [X]? / how do [a group] view [X]? / what is a difference between how "
               "[a group] view [X] and how they view [Y]? / what does [a character] do as [an action]?)",
    "based": 'Begin the question with "Based on the text," and ask what is likely or implied (for example: [a person '
             "or team] would most likely agree with which statement about [X]? / which choice best describes [X] or "
             "the relationship between [X] and [Y]? / what can be concluded about [a trend]? / what would have been "
             "the most likely consequence if [X] had not happened?)",
    "suggest": 'Use one of these wordings: "What does the text most strongly suggest about [X]?" / "What does the '
               'text indicate about [X]?" / "The text makes which point about [X]?" / "It can most reasonably be '
               'inferred from the text that [a finding] was important for which reason?" / "Information in the text '
               'best supports which statement about [X]?"',
}
DETAIL_FAMILY_WEIGHTS = {
    "easy": {"literal": 6, "based": 1},
    "medium": {"literal": 3, "based": 4, "suggest": 3},
    "hard": {"based": 5, "suggest": 4},  # never "According to the text" on hard items
}
DETAIL_FOCUS = {
    "trait": "a trait or description of a person, character, or thing",
    "viewpoint": "a person's or group's viewpoint (what they think, how they regard something, what they would "
                 "agree with); wrong answers flip the view, give it to the wrong person or group, or state the view "
                 "the person rejects",
    "reason": "a reason, cause, or effect (why something happened, what it led to, or why a finding mattered)",
    "conclusion": "a conclusion that follows from the evidence in the passage",
    "other": "another specific detail (what a study was about, what a character does, a challenge a group faces, or "
             "how two things are related)",
}
DETAIL_FOCUS_WEIGHTS = {
    "informational": {"trait": 7, "viewpoint": 6, "reason": 6, "conclusion": 3, "other": 4},
    "literary": {"trait": 3, "viewpoint": 1, "other": 1},
}
DETAIL_LINE = "\nREQUIRED QUESTION FOR THIS DETAIL ITEM: {family}. Ask about {focus}."


def cid_assignment(qtype: str, level: str, literary: bool, seed: int, index: int) -> dict:
    """Passage shape (both types) and, for detail items, the question family and focus. Seeded per item."""
    rng = random.Random(f"cid:{seed}:{index}")
    pick = lambda weights: rng.choices(list(weights), weights=list(weights.values()))[0]  # noqa: E731
    out = {"shape": pick(CID_SHAPE_WEIGHTS["literary" if literary else level])}
    if base_type(qtype) == "details":
        out["family"] = pick(DETAIL_FAMILY_WEIGHTS[level])
        out["focus"] = pick(DETAIL_FOCUS_WEIGHTS["literary" if literary else "informational"])
    return out


# Inferences, from a review of 21 real items (7 per level). The ending's job, the lead-in, the blank's form, and the
# passage shape are assigned in code; "puzzle" and "example" endings bring their own shape.
INF_SHAPES = {
    "pushback": "an old or common view, then a finding that pushes back against it, then the blank",
    "study": "a study's setup, then one or two findings (often two that must be put together), then the blank",
    "tradeoff": "a trade-off or a chain of causes (one thing helps, another hurts), and the blank weighs them",
    "approaches": "two approaches or views contrasted, and the blank applies one of them to a specific case",
    "puzzle": "a puzzle: two facts that don't fit and one explanation ruled out, and the blank gives what is left",
    "example": "a general claim, then one person or case as an example, and the blank says what the example shows",
}
INF_SHAPE_WEIGHTS = {  # "puzzle" and "example" come only with the ending of the same name
    "easy": {"study": 3, "tradeoff": 2, "pushback": 1},
    "medium": {"pushback": 4, "study": 3, "tradeoff": 1},
    "hard": {"pushback": 2, "study": 2, "approaches": 2},
}
INF_TWIST_SHAPES = ("pushback", "approaches", "puzzle")
INF_ENDINGS = {
    "meaning": "say what a finding or set of facts means",
    "puzzle": "explain a puzzle: why two facts don't line up",
    "predict": "predict what should happen, or what researchers expected to happen",
    "leaves_out": "point out what a view or approach leaves out",
    "example": "say what an example shows",
    "motive": "give someone's motive, or what a group should take away",
}
INF_ENDING_WEIGHTS = {
    "easy": {"meaning": 4, "predict": 1, "example": 1, "motive": 1},
    "medium": {"meaning": 3, "predict": 1, "motive": 1, "leaves_out": 1},
    "hard": {"meaning": 2, "puzzle": 3, "leaves_out": 2, "predict": 1},
}
INF_LEADINS = {
    "suggests": 'a "this finding suggests"-type phrase right before the blank (for example, "These findings suggest '
                'that ______." or "...which indicates that ______.")',
    "logic": "a logic word in the last sentence (therefore, thus, consequently, in this way, taken together), with the "
             "blank finishing that sentence",
    "other": 'a prediction ("...researchers would expect ______"), an "if this view is right" conditional, or a '
             'purpose ("...in order to ______")',
}
INF_LEADIN_WEIGHTS = {"suggests": 3, "logic": 2, "other": 1}
INF_BLANKS = {
    "clause": "a full clause (each choice is a complete clause)",
    "verb": "the sentence's verb phrase (all four choices start with a verb)",
}
INF_BLANK_WEIGHTS = {"clause": 11, "verb": 9}
INF_LINE = ("\nREQUIRED ENDING FOR THIS INFERENCE ITEM: the completion must {ending}."
            "\nREQUIRED LEAD-IN: {leadin}.\nREQUIRED BLANK FORM: the blank completes {blank}.")


def inf_assignment(level: str, seed: int, index: int) -> dict:
    """Ending, passage shape, lead-in, and blank form for one inference item. Seeded per item."""
    rng = random.Random(f"inf:{seed}:{index}")
    pick = lambda weights: rng.choices(list(weights), weights=list(weights.values()))[0]  # noqa: E731
    ending = pick(INF_ENDING_WEIGHTS[level])
    shape = ending if ending in ("puzzle", "example") else pick(INF_SHAPE_WEIGHTS[level])
    return {"ending": ending, "shape": shape, "leadin": pick(INF_LEADIN_WEIGHTS), "blank": pick(INF_BLANK_WEIGHTS)}


# Command of Evidence, from a review of 21 real items (7 per level). Each item gets a variation by difficulty;
# quotation items exist only at easy and medium, so a hard quotation slot is written as medium.
COE_CLAIM_VARIATIONS = {
    "support": 'Use the stem "Which finding, if true, would most directly support the [researchers\' or student\'s] '
               '[conclusion/hypothesis/claim]?"',
    "underlined": "Underline the claim with <u></u> and use the stem \"Which [detail/statement], if true, would most "
                  "strongly support the claim in the underlined sentence?\"",
    "expert": 'Make the claim a student\'s argument and use the stem "Which quotation from [an expert, described '
              'generically] would most effectively support the student\'s argument?"; the choices are short quotations',
    "weaken": 'Use the stem "Which finding, if true, would most directly undermine the [claim]?" (this variation is '
              "rare on the real test)",
}
COE_CLAIM_VARIATION_WEIGHTS = {
    "easy": {"support": 8, "underlined": 1, "expert": 1},
    "medium": {"support": 7, "underlined": 1, "expert": 1, "weaken": 1},
    "hard": {"support": 8, "underlined": 1, "weaken": 1},
}
COE_CLAIM_LOGIC = {
    "direct": "a straight match: a student's claim after a few plain sentences, and a correct finding that basically "
              "restates it",
    "head_on": "test the claim head-on with its own groups and comparison, after one translation step (a threshold, "
               "a code, or a hypothesis that turns measurements into categories); no stated rival explanation",
    "rival": "state a rival explanation before the conclusion; the correct finding isolates the claimed cause from "
             "the rival",
    "chain": "make the claim a chain (X affects Y, and Y affects Z) or a two-part claim; the correct finding gives "
             "both links or covers both halves, and a tempting wrong answer gives only one",
    "mechanism": "leave a gap in the reasoning; the correct finding is a general fact or mechanism that fills it and "
                 "may not mention the specific case at all",
}
COE_CLAIM_LOGIC_WEIGHTS = {
    "easy": {"direct": 1},
    "medium": {"head_on": 1},
    "hard": {"rival": 2, "chain": 1, "mechanism": 1},
}
COE_TABLE_TASKS = {
    "lookup": 'a one-value lookup: end the text with a sentence and "______." to be completed with one value from the '
              "table; the choices are single words or numbers (one is a real value from the wrong row)",
    "complete": 'complete a statement: end the text with "______." and read one or two values straight off the table',
    "support": "support a claim: compare two rows against the claim after a translation step from the text (a "
               "threshold or a definition); choices may be long multi-part statements with one wrong detail buried "
               "inside",
    "apply": "apply a definition, proposal, or hypothesis from the text to the data, tracking change over time; "
             "sometimes the data run opposite to what a proposal predicts, so the student reasons backward",
}
COE_TABLE_TASK_WEIGHTS = {
    "easy": {"lookup": 1, "complete": 2, "support": 1},
    "medium": {"support": 2, "complete": 1},
    "hard": {"apply": 3, "support": 1},
}
COE_GENRES = {"poem": 2, "novel": 1, "short story": 1, "collection of letters": 1, "travel narrative": 1, "play": 1}
COE_LINE = "\nREQUIRED VARIATION FOR THIS EVIDENCE ITEM: {desc}."


def coe_assignment(qtype: str, level: str, seed: int, index: int) -> dict:
    """Variation (and logic or genre) for one Command of Evidence item. Seeded per item."""
    rng = random.Random(f"coe:{seed}:{index}")
    pick = lambda weights: rng.choices(list(weights), weights=list(weights.values()))[0]  # noqa: E731
    base = base_type(qtype)
    if base == "command_of_evidence_claim":
        return {"variation": pick(COE_CLAIM_VARIATION_WEIGHTS[level]), "logic": pick(COE_CLAIM_LOGIC_WEIGHTS[level])}
    if base == "command_of_evidence_quantitative":
        return {"task": pick(COE_TABLE_TASK_WEIGHTS[level])}
    if base == "command_of_evidence_quotation":
        return {"genre": pick(COE_GENRES)}
    return {}


def coe_line(a: dict) -> str:
    if "variation" in a:
        desc = f"{COE_CLAIM_VARIATIONS[a['variation']]}. Logic: {COE_CLAIM_LOGIC[a['logic']]}"
    elif "task" in a:
        desc = f"a table item whose task is {COE_TABLE_TASKS[a['task']]}"
    else:
        desc = (f"the invented work is a {a['genre']}; mix quotation lengths from a short line to about 40 words, "
                f"and end the text with a colon and the blank")
    return COE_LINE.format(desc=desc)


def key_plan(count: int, seed: int) -> list[int]:
    """Where each question's correct answer goes (0-3 for A-D). Every block of 4 uses each letter once, so a batch
    is evenly mixed and no letter appears more than twice in a row (satgen alone shuffles each question on its own,
    which can stack up B's or put three C's in a row)."""
    rng = random.Random(f"keys:{seed}")
    plan: list[int] = []
    while len(plan) < count:
        block = list(range(len(CHOICES)))
        rng.shuffle(block)
        plan.extend(block)
    return plan[:count]


REPAIR_BLOCK = (
    "\n\nYOUR PREVIOUS DRAFT OF THIS QUESTION FAILED QUALITY CHECKS.\n"
    "Previous draft (JSON):\n{draft}\n\n"
    "Problems to fix:\n{problems}\n\n"
    "Write a corrected version. Keep the same question type, difficulty, and subject area. Change ONLY what the "
    "problems require: if specific answer choices are named, rewrite just those and keep the passage, the correct "
    "answer, and every other choice word for word. Return the full JSON again, with any double quotation marks "
    "inside strings escaped as \\\"."
)

QUALITY = os.environ.get("MEZA_QUALITY", "on").lower()
MAX_REPAIRS = int(os.environ.get("MEZA_MAX_REPAIRS", "2"))
PROBE_RUNS = int(os.environ.get("MEZA_PROBE_RUNS", "3"))
PROBE_FAIL = int(os.environ.get("MEZA_PROBE_FAIL", "3"))
LOG_PATH = os.environ.get("MEZA_QUALITY_LOG", "")
_log_lock = threading.Lock()
VOCAB_PATH = Path(os.environ.get("MEZA_VOCAB", Path(__file__).with_name("data") / "meza_vocab_bank.json"))
RECENT_KEEP = 5
RECENT_BLOCK = (
    "\n\nYOUR MOST RECENT QUESTIONS OF THIS TYPE (newest last):\n{items}\n"
    "Make this question clearly different from them: a different opening, a different sentence rhythm, the blank "
    "(if any) in a different place, and a different answer word.{extra}"
)


# Harder passages draw advanced words from the vocabulary bank (data/meza_vocab_bank.json: 1,961 unique words
# merged from three public Anki decks; words and parts of speech only). Words are dealt like cards, so a run uses
# the whole bank before repeating one.
VOCAB_RULES = {
    # difficulty: (bank words offered, minimum to use, most advanced bank words allowed in the passage, tiers dealt)
    "easy": (0, 0, 1, ()),
    "medium": (2, 1, 1, ("mid", "advanced")),
    "hard": (3, 1, 2, ("advanced", "mid", "advanced")),
}
# Poems and plays get denser period language (Sebastian's note): more bank words allowed.
POETRY_VOCAB_RULES = {
    "easy": (1, 0, 1, ("mid",)),
    "medium": (3, 1, 2, ("advanced", "mid", "advanced")),
    "hard": (4, 2, 3, ("advanced", "advanced", "mid", "advanced")),
}
# Words in Context answer choices: how many of each tier, and how often the key comes from each tier.
# Real hard items usually mix two words a typical student doesn't know, one they half know, one they know.
CHOICE_MIX = {
    "easy": {"mid": 1, "common": 3},
    "medium": {"advanced": 1, "mid": 2, "common": 1},
    "hard": {"advanced": 2, "mid": 1, "common": 1},
}
KEY_TIER_WEIGHTS = {
    "easy": {"common": 3, "mid": 1},
    "medium": {"mid": 3, "advanced": 1, "common": 1},
    "hard": {"advanced": 3, "mid": 2, "common": 1},
}
TIER_LABEL = {"advanced": "advanced (most students won't know it)", "mid": "mid-level (students half know it)",
              "common": "common (students know it)"}
MEANING_ADV_CAP = {"easy": 0, "medium": 1, "hard": 1}
MEANING_LINE = (
    "\nTESTED WORD FOR THIS ITEM: pick ONE of these words that has more than one common meaning and use it in a "
    "less common sense (if none has two common meanings, use your own everyday word that does): {words}. "
    "Underline it with <u></u>. Choices are short meanings; at least one wrong choice is the word's most common "
    "meaning, and every choice must fit the sentence grammatically. Use at most {cap} advanced word{s} among the choices."
)
NO_VOCAB_TYPES = ("rhetorical_synthesis",)  # notes are plain factual bullets
# Central Ideas and Details test reading, not vocabulary: at easy and medium, bank words are optional (at most
# one) and passages lean on subject terms a student can work out from context. Real hard passages are dense with
# academic vocabulary, so hard items use one or two bank words as usual (review of 33 real items, Nov 2026).
OPTIONAL_VOCAB_TYPES = ("central_ideas", "details")  # optional at easy and medium; hard uses one or two
OPTIONAL_VOCAB_LINE = (
    "\nVOCABULARY FOR THIS PASSAGE ({level}, optional): this item tests the main idea, not vocabulary. You may use "
    "at most one of these words if it fits naturally, or none: {words}. Rely instead on subject terms a student can "
    "work out from the sentence around them."
)
VOCAB_LINE = (
    "\nVOCABULARY FOR THIS PASSAGE ({level}): sprinkle in {need_text} of these words, used naturally and correctly "
    "(any form is fine): {words}. Use no more than {cap} advanced word{s} in the whole passage; the rest of the "
    "passage stays in clear academic English. Do not use them as the answer or let them give the answer away."
)
POOL_LINE = (
    "\nANSWER CHOICES FOR THIS ITEM: all four choices are the same part of speech ({pos}), with this mix: {mix}. "
    "Word lists to pick from (you may change a word's form, or use your own word of the same level if none fits):\n"
    "{lists}\nThe CORRECT answer must be the {key_tier} word."
)


class VocabBank:
    def __init__(self, path: Path, seed: int):
        data = json.loads(path.read_text(encoding="utf-8"))["words"] if path.exists() else []
        self.rng = random.Random(f"vocab:{seed}")
        self.by_tier: dict[str, list[dict]] = {}
        for w in data:
            self.by_tier.setdefault(w["tier"], []).append(w)
            if w.get("multi_meaning") and " " not in w["word"]:
                self.by_tier.setdefault("multi", []).append(w)
        self.decks: dict[str, list[dict]] = {}
        self.lock = threading.Lock()

    def deal(self, tier: str, n: int, pos: str | None = None) -> list[str]:
        out: list[str] = []
        with self.lock:
            key = f"{tier}:{pos or ''}"
            source = [w for w in self.by_tier.get(tier, []) if not pos or w["pos"].startswith(pos)]
            guard = 0
            while len(out) < n and source and guard < 4 * n + 10:
                guard += 1
                if not self.decks.get(key):
                    self.decks[key] = source[:]
                    self.rng.shuffle(self.decks[key])
                w = self.decks[key].pop()["word"]
                if w not in out:
                    out.append(w)
        return out

    def passage_words(self, difficulty: str, poetry: bool = False) -> tuple[list[str], int, int]:
        rules = POETRY_VOCAB_RULES if poetry else VOCAB_RULES
        n, need, cap, tiers = rules.get(difficulty, (0, 0, 1, ()))
        words: list[str] = []
        for i in range(n):
            words += self.deal(tiers[i % len(tiers)], 1)
        return words, (need if words else 0), cap

    def targets(self, n: int) -> list[str]:
        """Candidate tested words for meaning items: mostly words the decks mark as having a second meaning,
        topped up with everyday and mid-level words."""
        out = self.deal("multi", n - 2) if self.by_tier.get("multi") else []
        out += [w for w in self.deal("common", 1) + self.deal("mid", 2) if w not in out]
        return out[:n]

    def choice_plan(self, difficulty: str) -> dict:
        """Which tiers the four choices come from, which tier the key is, and word lists to pick from."""
        mix = CHOICE_MIX.get(difficulty, CHOICE_MIX["medium"])
        weights = KEY_TIER_WEIGHTS.get(difficulty, KEY_TIER_WEIGHTS["medium"])
        with self.lock:
            key_tier = self.rng.choices(list(weights), weights=list(weights.values()))[0]
            pos = self.rng.choice(["adjective", "verb", "noun"])
        lists = {t: self.deal(t, n + 2, pos) for t, n in mix.items()}
        return {"mix": mix, "key_tier": key_tier, "pos": pos, "lists": lists}


def subject_plan(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    plan: list[int] = []
    while len(plan) < count:
        block = list(range(len(SUBJECTS)))
        rng.shuffle(block)
        if plan and block[0] == plan[-1]:  # no back-to-back repeat across blocks
            block.append(block.pop(0))
        plan.extend(block)
    return plan[:count]


def subject_for(qtype: str, slot: int) -> str:
    base = base_type(qtype)
    if base == "command_of_evidence_quotation":
        return "LITERATURE - an invented literary work (see the quotation-item rules)"
    if slot in LITERARY and not base.startswith(EXCERPT_OK):
        return INFORMATIONAL_STAND_IN
    return SUBJECTS[slot]


def log_attempt(entry: dict) -> None:
    if not LOG_PATH:
        return
    with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class MezaPipeline(pipeline.Pipeline):
    plan: list[int] = []
    clues: list[int] = []
    stems: list[int] = []
    keys: list[int] = []
    seed: int = 0
    # Questions of the same type are written one at a time (other types still run in parallel), so each
    # new question can see the shape of the ones before it and is told to do something different.
    _type_locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def _lock_for(self, t: str) -> threading.Lock:
        with self._locks_guard:
            return self._type_locks.setdefault(t, threading.Lock())

    def run(self, qtypes, count, *args, **kwargs):
        seed = int(os.environ.get("MEZA_SUBJECT_SEED") or random.SystemRandom().randrange(10**9))
        type(self).plan = subject_plan(count, seed)
        type(self).clues = clue_plan(count, seed)
        type(self).stems = stem_plan(count, seed)
        type(self).seed = seed
        type(self).keys = key_plan(count, seed)
        self.vocab = VocabBank(VOCAB_PATH, seed)
        self.trap_bank = load_bank()
        self._recent = {}
        print(f"Subject seed: {seed} (set MEZA_SUBJECT_SEED={seed} to repeat this order)", file=sys.stderr)
        if QUALITY != "off":
            print(f"Quality gate: {QUALITY}, up to {MAX_REPAIRS} rewrites per question", file=sys.stderr)
        return super().run(qtypes, count, *args, **kwargs)

    def shuffle(self, q: Question, index: int) -> None:
        """Put the correct answer at this question's planned letter and shuffle the wrong answers around it."""
        if q.format != MCQ or index >= len(self.keys):
            return super().shuffle(q, index)
        right = q.choices[q.answer]
        wrong = [q.choices[k] for k in CHOICES if k != q.answer]
        random.Random(f"keys:{self.seed}:{index}").shuffle(wrong)
        wrong.insert(self.keys[index], right)
        q.choices = dict(zip(CHOICES, wrong, strict=True))
        q.answer = CHOICES[self.keys[index]]

    def _draft(self, prompt: str, qtype: str, fmt: str) -> tuple[Question, dict]:
        for tries in range(3):  # a draft with broken JSON is retried without using up a rewrite
            text = self._call(self.generator, "generate", prompt, system=pipeline.GEN_SYSTEM,
                              temperature=0.9, json_mode=True)
            try:
                data = pipeline.parse_json(text)
                break
            except (ValueError, json.JSONDecodeError):
                if tries == 2:
                    raise
                prompt += ("\n\nYour last reply was not valid JSON. Return valid JSON only, escaping any double "
                           "quotation marks inside strings as \\\".")
        q = Question(
            qtype=qtype,  # keep the requested name (with its difficulty) rather than the writer's echo
            stem=str(data["stem"]),
            choices={str(k).strip().upper(): str(v) for k, v in (data.get("choices") or {}).items()},
            answer=str(data["answer"]).strip(),
            explanation=str(data.get("explanation") or ""),
            passage=str(data.get("passage") or ""),
            format=fmt,
        )
        q.answer = q.answer.upper() if fmt == MCQ else normalize_spr(q.answer)
        return q, data

    def _focus_critic(self, q: Question) -> dict:
        """The reviewer model judges whether a Words in Context passage is built around the tested word."""
        prompt = FOCUS_CRITIC_PROMPT.format(question=q.to_prompt())
        try:
            return pipeline.parse_json(self._call(self.reviewer, "focus_critic", prompt, system=FOCUS_CRITIC_SYSTEM,
                                                  json_mode=True))
        except (ValueError, KeyError, TypeError):
            return {}

    def _trap_critic(self, q: Question, family: str = "tsp") -> dict:
        """The reviewer model rates each wrong answer: on topic? how tempting? which trap?"""
        prompt = critic_prompt(family, q.to_prompt(), q.answer)
        try:
            return pipeline.parse_json(self._call(self.reviewer, "trap_critic", prompt, system=CRITIC_SYSTEM,
                                                  json_mode=True))
        except (ValueError, KeyError, TypeError):
            return {}

    def _probe(self, q: Question, index: int, attempt: int) -> list[bool]:
        def ask(prompt: str, system: str) -> str:
            return self._call(self.solver, "probe", prompt, system=system, json_mode=True)
        return probe_votes(ask, q.stem, q.choices, q.answer, runs=PROBE_RUNS,
                           rng=random.Random(f"probe:{index}:{attempt}"))

    def generate(self, index: int, qtype: str) -> Question:
        if QUALITY == "off" or isinstance(self.generator, SimulatedClient):
            return self._generate(index, qtype)
        with self._lock_for(base_type(qtype)):
            return self._generate(index, qtype)

    def _generate(self, index: int, qtype: str) -> Question:
        if base_type(qtype) == "command_of_evidence_quotation" and difficulty_of(qtype, index) == "hard":
            qtype = base_type(qtype) + "_medium"  # real quotation items are easy or medium only
        slot = self.plan[index] if index < len(self.plan) else index % len(SUBJECTS)
        subject = subject_for(qtype, slot)
        fmt = SPR if qtype.startswith("spr_") else MCQ
        base_prompt = pipeline.GEN_PROMPT.format(
            qtype=qtype,
            index=index,
            context=self.context,
            format_rules=pipeline.SPR_RULES if fmt == SPR else pipeline.MCQ_RULES,
        ) + SUBJECT_LINE.format(subject=subject)
        clue, role = "", ""
        if base_type(qtype).startswith("words_in_context"):
            ci = self.clues[index] if index < len(self.clues) else index % len(WIC_CLUES)
            clue, _, desc = WIC_CLUES[ci]
            if base_type(qtype) == "words_in_context_meaning":
                if clue == "restatement":  # a colon explaining the word would give the meaning away
                    clue, _, desc = next(c for c in WIC_CLUES if c[0] == "exposition")
                desc = desc.replace("the blank", "the underlined word")
            base_prompt += CLUE_LINE.format(desc=desc)
            weights = WORD_ROLE_WEIGHTS.get(difficulty_of(qtype, index) or "medium", WORD_ROLE_WEIGHTS["medium"])
            role = random.Random(f"role:{index}:{len(self.plan)}").choices(list(weights), list(weights.values()))[0]
            base_prompt += ROLE_LINE[role]
        want_kind, band = "", None
        if base_type(qtype) in TSP_TYPES:
            rng_b = random.Random(f"tsp:{index}:{len(self.plan)}")
            style = "long" if base_type(qtype) == "text_structure_overall" else rng_b.choice(["short", "long"])
            band = TSP_BANDS[style]
            verbs = TSP_VERBS[:]
            rng_b.shuffle(verbs)
            prior = [r["key"] for r in self._recent.get(base_type(qtype), [])[-3:] if r.get("key")] \
                if "_recent" in self.__dict__ else []
            avoid = (f" The correct answer must NOT open with {', '.join(sorted(set(prior)))} "
                     f"(recent correct answers did)." if prior else "")
            base_prompt += TSP_LINE.format(lo=band[0], hi=band[1], style=f"{style} style", verbs=", ".join(verbs[:9]),
                                           avoid=avoid)
            base_prompt += TRAP_LINE.format(
                traps="\n".join(f"  - {k}: {v}" for k, v in TSP_TRAPS.items()),
                level_rule=TRAP_LEVEL_RULE.get(difficulty_of(qtype, index) or "medium", ""))
        if base_type(qtype) == "text_structure_function":
            rng = random.Random(f"function:{index}:{len(self.plan)}")
            kinds = [k for k, w, _ in FUNCTION_KINDS for _ in range(w)]
            want_kind = rng.choice(kinds)
            base_prompt += FUNCTION_LINE.format(desc=next(d for k, _, d in FUNCTION_KINDS if k == want_kind))
        stem_kind, stem_text = "", ""
        if base_type(qtype) == "central_ideas":
            si = self.stems[index] if index < len(self.stems) else 0
            stem_kind, _, stem_text = CENTRAL_STEM_PLAN[si]
            base_prompt += STEM_LINE.format(stem=stem_text)
        if "_recent" not in self.__dict__:
            self._recent = {}
        if "trap_bank" not in self.__dict__:
            self.trap_bank = load_bank()
        trap_fam = family_of(base_type(qtype)) if fmt == MCQ and QUALITY != "off" else None
        if trap_fam and trap_fam != "tsp":
            lvl = difficulty_of(qtype, index) or "medium"
            base_prompt += TRAP_LINE_GENERAL.format(
                guidance=CRITIC_GUIDANCE[trap_fam], traps="\n".join(f"  - {k}: {v}" for k, v in TRAPS[trap_fam].items()),
                level_rule=TRAP_LEVEL_RULE_GENERAL.get(lvl, ""))
        examples = []
        if trap_fam and TRAP_EXAMPLES:
            examples = pick_examples(self.trap_bank, base_type(qtype), difficulty_of(qtype, index) or "medium",
                                     n=TRAP_EXAMPLES, seed=f"{index}:{len(self.plan)}")
            base_prompt += format_examples(examples)
        if "vocab" not in self.__dict__:
            self.vocab = VocabBank(VOCAB_PATH, index)
        level = difficulty_of(qtype, index) or "medium"
        cid: dict = {}
        if base_type(qtype) in ("central_ideas", "details"):
            cid = cid_assignment(qtype, level, subject.startswith("LITERATURE:"), self.seed, index)
            base_prompt += SHAPE_LINE.format(desc=CID_SHAPES[cid["shape"]])
            if "family" in cid:
                base_prompt += DETAIL_LINE.format(family=DETAIL_FAMILIES[cid["family"]],
                                                  focus=DETAIL_FOCUS[cid["focus"]])
        inf: dict = {}
        if base_type(qtype) == "inferences":
            inf = inf_assignment(level, self.seed, index)
            base_prompt += SHAPE_LINE.format(desc=INF_SHAPES[inf["shape"]])
            base_prompt += INF_LINE.format(ending=INF_ENDINGS[inf["ending"]], leadin=INF_LEADINS[inf["leadin"]],
                                           blank=INF_BLANKS[inf["blank"]])
        coe: dict = {}
        if base_type(qtype).startswith("command_of_evidence"):
            coe = coe_assignment(qtype, level, self.seed, index)
        if coe:
            base_prompt += coe_line(coe)
        twist_ok = cid.get("shape") in TWIST_SHAPES or inf.get("shape") in INF_TWIST_SHAPES
        vocab, need, cap, plan = [], 0, 9, None
        if fmt == MCQ and QUALITY != "off" and not base_type(qtype).startswith(NO_VOCAB_TYPES):
            poetry = "POETRY" in subject.upper()
            vocab, need, cap = self.vocab.passage_words(level, poetry=poetry)
            optional = base_type(qtype).startswith(OPTIONAL_VOCAB_TYPES) and level != "hard"
            if optional:
                need, cap = 0, 1
            if vocab:
                line = OPTIONAL_VOCAB_LINE if optional else VOCAB_LINE
                base_prompt += line.format(level=level, need_text="one or two" if cap > 1 else "one",
                                           words=", ".join(vocab), cap=cap, s="s" if cap > 1 else "")
            if base_type(qtype) == "words_in_context_meaning":
                targets = self.vocab.targets(6)
                adv_cap = MEANING_ADV_CAP.get(level, 1)
                base_prompt += MEANING_LINE.format(words=", ".join(targets), cap=adv_cap,
                                                   s="" if adv_cap == 1 else "s")
                plan = {"mix": {"advanced": adv_cap}, "key_tier": None, "meaning": True, "targets": targets}
            if base_type(qtype) == "words_in_context_completion":
                plan = self.vocab.choice_plan(level)
                mix = ", ".join(f"{n} {TIER_LABEL[t]}" for t, n in plan["mix"].items())
                lists = "\n".join(f"  {t}: {', '.join(ws)}" for t, ws in plan["lists"].items() if ws)
                base_prompt += POOL_LINE.format(pos=plan["pos"], mix=mix, lists=lists,
                                                key_tier=TIER_LABEL[plan["key_tier"]].split(" (")[0])
        recent = self._recent.get(base_type(qtype), [])
        if QUALITY != "off" and fmt == MCQ and recent:
            twisty = sum(r["twist"] for r in recent[-3:]) >= 2 and clue != "contrast" and not twist_ok
            extra = (" Several of them used a belief-then-twist rhythm, so this one must not use a contrast at all."
                     if twisty else "")
            items = "\n".join(f"{i}. {describe(r)}" for i, r in enumerate(recent, 1))
            base_prompt += RECENT_BLOCK.format(items=items, extra=extra)

        # The offline demo's simulated writer can't follow the rulebook; skip the gate there.
        gate = QUALITY != "off" and fmt == MCQ and not isinstance(self.generator, SimulatedClient)
        prompt, problems = base_prompt, []
        repairs = MAX_REPAIRS + (1 if base_type(qtype) in TSP_TYPES else 0)
        for attempt in range(repairs + 1 if gate else 1):
            try:
                q, data = self._draft(prompt, qtype, fmt)
                q.validate()
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                if not gate or attempt == repairs:
                    raise
                problems = [f"The draft could not be used: {exc}. Follow the JSON format exactly."]
                log_attempt({"index": index, "qtype": qtype, "attempt": attempt, "invalid": str(exc)})
                prompt = base_prompt + REPAIR_BLOCK.format(draft="(unusable)", problems="\n".join(problems))
                continue
            if not gate:
                return q

            res = lint(qtype, q.passage, q.stem, q.choices, q.answer, q.explanation, subject=subject, clue=clue,
                       want_kind=want_kind, choice_band=band, difficulty=level)
            key_mark = (q.choices.get(q.answer, "") if base_type(qtype).startswith(("words_in_context", "transitions"))
                        else lead_verb(q.choices.get(q.answer, "")) if base_type(qtype) in TSP_TYPES else "")
            shape = features(qtype, q.passage, key_mark, clue)
            problems = list(res.fails) + rhythm_check(shape, recent, twist_ok=twist_ok)
            got = central_stem_kind(q.stem) if stem_kind else None
            if got and (got if got in ("standard", "question") else "variant") != stem_kind:
                problems.append(f"Use the required stem for this item, word for word: {stem_text}")
            if inf.get("leadin") in ("suggests", "logic") and inf["leadin"] not in leadin_kinds(q.passage):
                problems.append(f"Use the required lead-in into the blank: {INF_LEADINS[inf['leadin']]}.")
            fam = detail_stem_family(q.stem) if "family" in cid else None
            if fam and fam != cid["family"]:
                problems.append(f"Use the required question wording for this item: {DETAIL_FAMILIES[cid['family']]}.")
            used = vocab_used(q.passage, vocab)
            if len(used) < need:
                problems.append(f"Use at least {need} of these words naturally in the passage (any form): "
                                f"{', '.join(vocab)}. You used {len(used)}. Keep the meaning and the answer the same.")
            heavy = advanced_in(q.passage)
            if len(heavy) > cap:
                problems.append(f"The passage uses {len(heavy)} advanced words ({', '.join(heavy)}); the most for this "
                                f"level is {cap}. Swap the extras for plainer words so the passage isn't overloaded.")
            if plan and plan.get("meaning"):
                heavy_choices = [v for v in q.choices.values() if word_tier(v) == "advanced"]
                if len(heavy_choices) > plan["mix"]["advanced"]:
                    problems.append(f"Too many advanced words among the choices ({', '.join(heavy_choices)}); use at "
                                    f"most {plan['mix']['advanced']}. Meaning choices should be readable; the difficulty "
                                    f"comes from the uncommon sense of the tested word.")
            elif plan:
                problems += choice_mix_problems(q.choices, q.answer, plan["mix"], plan["key_tier"])
            hits: list[bool] = []
            critic: dict = {}
            # Judgment calls (the reviewer-model critics and the passage-free probe) are "soft": after the last rewrite
            # they flag the question for human review instead of discarding it. Everything else must pass.
            soft: list[str] = []
            problems += reused_topic(q.passage, examples)
            if trap_fam:
                problems += trap_label_problems(data.get("traps"), q.answer, level, trap_fam)
                if not problems and QUALITY == "on" and TRAP_CRITIC == "on":
                    critic = self._trap_critic(q, trap_fam)
                    soft += trap_critic_problems(critic, q.answer, level, trap_fam)
            focus: dict = {}
            if role == "detail" and not problems and QUALITY == "on" and FOCUS_CRITIC == "on":
                focus = self._focus_critic(q)
                soft += focus_problems(focus, role)
            if not problems and QUALITY == "on" and base_type(qtype).startswith(PROBE_TYPES):
                hits = self._probe(q, index, attempt)
                if sum(hits) >= PROBE_FAIL:
                    soft.append(probe_feedback(hits))
            last = attempt == repairs
            flagged = last and not problems and bool(soft)
            problems += soft
            log_attempt({"index": index, "qtype": qtype, "subject": subject, "clue": clue, "attempt": attempt,
                         "fails": res.fails, "warns": res.warns, "probe_hits": sum(hits),
                         "probe_runs": len(hits), "shape": shape, "underlined": want_kind,
                         "traps": data.get("traps"), "trap_critic": critic, "vocab": vocab, "vocab_used": used,
                         "choice_plan": plan, "word_role": role, "focus": focus,
                         "stem": stem_kind, "cid": cid, "inf": inf, "coe": coe,
                         "choices": q.choices, "answer": q.answer,  # before shuffling; maps trap labels later
                         "problems": problems, "passed": not problems or flagged,
                         "flagged": soft if flagged else [], "fingerprint": fingerprint(q.passage, q.stem)})
            if not problems or flagged:
                self._recent.setdefault(base_type(qtype), []).append(shape)
                del self._recent[base_type(qtype)][:-RECENT_KEEP]
                return q
            draft = json.dumps({k: data.get(k) for k in ("passage", "stem", "choices", "answer", "explanation", "traps")},
                               ensure_ascii=False, indent=1)
            prompt = base_prompt + REPAIR_BLOCK.format(draft=draft, problems="\n".join(f"- {p}" for p in problems))
        raise ValueError("quality gate: " + " | ".join(p[:120] for p in problems))


def install() -> None:
    """Make satgen's CLI build a MezaPipeline. Done only when this file is run, not on import, so importing it
    (for example from tests) leaves satgen's own Pipeline untouched."""
    pipeline.Pipeline = MezaPipeline


if __name__ == "__main__":
    install()
    args = sys.argv[1:]
    if args[:1] != ["run"]:
        args = ["run", *args]
    sys.exit(cli.main(args))
