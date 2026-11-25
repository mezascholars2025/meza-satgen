"""Meza Scholars quality gate for satgen questions.

Two checks run on every freshly written question, before satgen's solvers see it:

1. lint(q)  - free, deterministic checks for format errors and "tells" that a test-wise
              student could use to guess the answer without reading (the key is the longest
              choice, absolutes only in wrong answers, wrong stem, wrong length, ...).
2. probe(q) - the "test-wise student": a cheap model sees ONLY the question and the four
              choices, never the passage. If it still finds the key every time, the item
              gives its answer away and gets rewritten.

Problems come back as plain sentences, so meza_satgen.py can hand them to the writer and
ask for a fixed version instead of throwing the question away.

    python meza_quality.py runs/<folder>/questions.json     # lint an existing run (no API calls)
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

BLANK = "______"
CHOICE_KEYS = ("A", "B", "C", "D")
DIFFICULTIES = ("easy", "medium", "hard")

# ----------------------------------------------------------------------------- type facts

SEC_STEM = "Which choice completes the text so that it conforms to the conventions of Standard English?"
FIXED_STEMS = {
    "words_in_context_completion": "Which choice completes the text with the most logical and precise word or phrase?",
    "transitions": "Which choice completes the text with the most logical transition?",
    "central_ideas": "Which choice best states the main idea of the text?",
    "inferences": "Which choice most logically completes the text?",
    "text_structure_purpose": "Which choice best states the main purpose of the text?",
    "text_structure_overall": "Which choice best describes the overall structure of the text?",
}
FUNCTION_STEM = re.compile(r"^Which choice best describes the function of the underlined (sentence|question|portion) "
                           r"in the text as a whole\?$")
TSP_TYPES = ("text_structure_function", "text_structure_overall", "text_structure_purpose")
TSP_VERBS = ["suggest", "indicate", "emphasize", "convey", "describe", "illustrate", "summarize", "present", "reflect",
             "represent", "introduce", "establish", "demonstrate", "provide", "discuss", "evaluate", "compare",
             "explain", "capture"]
TSP_OTHER_VERBS = ["argue", "challenge", "criticize", "question", "propose", "qualify", "refute", "contrast", "offer",
                   "raise", "note", "identify", "trace", "examine", "defend", "dispute", "acknowledge", "highlight",
                   "praise", "express", "analyze", "recount", "admire", "celebrate", "lament", "advocate", "reveal",
                   "outline", "detail", "address", "recommend", "warn", "speculate", "consider", "pose", "give",
                   "state", "show", "list", "assert", "wonder", "reject", "dismiss", "mock"]
TSP_BANDS = {"short": (9, 12), "long": (15, 22)}


def lead_verb(choice: str) -> str:
    """'To describe how...' -> describe; 'It presents...' -> present; '' if the choice doesn't open that way."""
    m = re.match(r"^(?:To|It)\s+([A-Za-z]+)", choice.strip())
    if not m:
        return ""
    v = m.group(1).lower()
    for base in TSP_VERBS + TSP_OTHER_VERBS:
        if v in (base, base + "s", base + "es", base[:-1] + "ies" if base.endswith("y") else base + "s"):
            return base
    return v


MEANING_STEM = re.compile(r'^As used in the text, what does the (word|phrase) "([^"]+)" most nearly mean\?$')
SHORT_CHOICE_TYPES = ("words_in_context", "transitions", "boundaries_", "form_")
NEEDS_BLANK = ("words_in_context_completion", "inferences", "transitions", "boundaries_", "form_")
NO_BLANK = ("details", "central_ideas", "text_structure", "cross_text_connections",
            "command_of_evidence_claim", "command_of_evidence_quotation", "rhetorical_synthesis",
            "words_in_context_meaning")
# Types the passage-free probe is fair for. Grammar, vocabulary and transition items cannot be
# answered without the text anyway, and rhetorical synthesis puts its goal in the stem on purpose.
PROBE_TYPES = ("details", "central_ideas", "command_of_evidence", "inferences", "text_structure",
               "cross_text_connections")

# (real words) lengths that read like the real test; outside these is a warning, not a failure
TARGET_WORDS = {
    "short": (25, 75),      # SEC, transitions, words in context
    "notes": (55, 125),     # rhetorical synthesis notes
    "default": (60, 150),
}

OVERUSED_TOPICS = ["bee", "pollinat", "sunflower", "bean seedling", "sea turtle", "mangrove", "octopus",
                   "crow", "coral reef", "urban heat island"]
ABSOLUTES = ["only", "always", "never", "all", "every", "none", "entirely", "completely", "proves",
             "proved", "impossible", "best", "certainly", "solely", "exclusively"]
TRANSITION_GROUPS = {
    "contrast": ["however", "nevertheless", "nonetheless", "by contrast", "in contrast", "still", "even so",
                 "on the other hand", "conversely", "instead", "rather", "yet"],
    "cause": ["therefore", "thus", "consequently", "as a result", "hence", "accordingly", "for this reason"],
    "addition": ["moreover", "furthermore", "additionally", "in addition", "besides", "likewise", "similarly",
                 "also"],
    "example": ["for example", "for instance", "specifically", "to illustrate"],
    "clarify": ["in other words", "that is", "put differently", "namely"],
    "time": ["subsequently", "afterward", "afterwards", "later", "meanwhile", "finally", "previously",
             "eventually", "first", "then", "next"],
    "emphasis": ["in fact", "indeed", "of course", "granted", "admittedly", "to be sure"],
}
STOP = set("""a an the and or but of to in on for with by from as at is are was were be been being that this
these those it its their they them he she his her which who whom what when where while than then into
about over under not no can could would should may might will has have had do does did such also more
most other some""".split())


def base_type(qtype: str) -> str:
    for d in DIFFICULTIES:
        qtype = qtype.removesuffix("_" + d)
    return qtype


def difficulty_of(qtype: str, index: int | None = None) -> str | None:
    for d in DIFFICULTIES:
        if qtype.endswith("_" + d):
            return d
    return DIFFICULTIES[index % 3] if index is not None else None


def std_words(text: str) -> float:
    """College Board counts a 'word' as 6 characters, spaces and punctuation included."""
    return len(text) / 6


def strip_intro(passage: str) -> str:
    """Drop a literary introduction ("The following text is from ...") so it never counts toward length."""
    p = passage.lstrip()
    if p.startswith("The following text is from"):
        parts = re.split(r"\n\s*\n", p, maxsplit=1)
        return parts[1] if len(parts) > 1 else ""
    return passage


def reading_text(passage: str) -> str:
    """Passage without the literary introduction, the data table, and the table's title line."""
    lines = strip_intro(passage).splitlines()
    keep = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("|"):
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if nxt.startswith("|") and len(s) < 120:  # the title line sitting on top of a table
            continue
        keep.append(s)
    text = " ".join(x for x in keep if x)
    return re.sub(r"</?u>", "", text)


def fingerprint(passage: str, stem: str) -> str:
    """Short key that matches a question in questions.json back to its quality-log entry."""
    return " ".join((re.sub(r"</?u>", "", passage) + " " + stem).lower().split())[:160]


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return " ".join(s.split()).strip()


def has_word(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text.lower()) is not None


# ----------------------------------------------------------------------------- lint


@dataclass
class LintResult:
    fails: list[str] = field(default_factory=list)   # must be fixed before the question can ship
    warns: list[str] = field(default_factory=list)   # logged for the human reviewer

    @property
    def ok(self) -> bool:
        return not self.fails


def lint(qtype: str, passage: str, stem: str, choices: dict[str, str], answer: str,
         explanation: str = "", subject: str = "", clue: str = "", want_kind: str = "",
         choice_band: tuple[int, int] | None = None) -> LintResult:
    r = LintResult()
    t = base_type(qtype)
    text = reading_text(passage)
    n_std = std_words(text)
    n_real = len(text.split())
    key = choices.get(answer, "")
    wrong = [v for k, v in choices.items() if k != answer]

    # --- length (the 25-150 rule uses standardized words) ---
    if n_std > 150:
        r.fails.append(f"The passage is {n_std:.0f} standardized words (characters / 6); the limit is 150. "
                       f"Cut about {len(text) - 900} characters (roughly {(len(text) - 900) // 6 + 1} words).")
    elif n_std < 25:
        r.fails.append(f"The passage is only {n_std:.0f} standardized words; the minimum is 25.")
    band = ("short" if t.startswith(SHORT_CHOICE_TYPES) else "notes" if t == "rhetorical_synthesis" else "default")
    lo, hi = TARGET_WORDS[band]
    if not lo <= n_real <= hi:
        r.warns.append(f"Passage has {n_real} words; real items of this type usually run {lo}-{hi}.")
    if passage.lstrip().startswith("The following text is from") and n_real < 40:
        r.fails.append(f"The excerpt itself is only {n_real} words; the introduction does not count toward length. "
                       f"Give the excerpt at least 40 words of its own.")

    # --- blanks and underlines ---
    blanks = passage.count(BLANK)
    if t.startswith(NEEDS_BLANK) and blanks != 1:
        r.fails.append(f"This type needs exactly one blank written as {BLANK}; the passage has {blanks}.")
    if t.startswith(NO_BLANK) and blanks:
        r.fails.append("This type must not contain a blank.")
    if t == "command_of_evidence_quantitative" and "complete" in stem.lower() and blanks != 1:
        r.fails.append(f"A 'complete the statement' data item must end its text with {BLANK}.")
    if t == "words_in_context_completion" and clue != "restatement" and re.search(re.escape(BLANK) + r"\s*[:;\u2014\u2013]|" + re.escape(BLANK) + r"\s+-", passage):
        r.fails.append("Do not follow the blank with a colon, semicolon, or dash that explains the word. Build the "
                       "passage around the required clue structure so the meaning must be inferred.")
    if t == "text_structure_function":
        unders = re.findall(r"<u>(.*?)</u>", passage, flags=re.S)
        m = FUNCTION_STEM.match(norm(stem))
        if len(unders) != 1:
            r.fails.append("Underline exactly one portion of the passage with <u></u>.")
        if not m:
            r.fails.append('Use the exact stem: "Which choice best describes the function of the underlined '
                           'sentence (or question, or portion) in the text as a whole?"')
        if len(unders) == 1 and m:
            u = norm(unders[0]).rstrip('"\u201d')
            kind = m.group(1)
            is_q = u.endswith("?")
            is_sentence = bool(re.match(r"^[A-Z\"\u201c]", u)) and u[-1:] in ".!?"
            if kind == "question" and not is_q:
                r.fails.append('The stem says "underlined question", but the underlined text is not a question.')
            if kind != "question" and is_q:
                r.fails.append('The underlined text is a question, so the stem must say "underlined question".')
            if kind == "sentence" and not is_sentence:
                r.fails.append('The stem says "underlined sentence"; underline one complete sentence.')
            if kind == "portion" and is_sentence:
                r.fails.append('The stem says "underlined portion"; underline a phrase or quotation, not a whole sentence.')
            if want_kind and kind != want_kind:
                r.fails.append(f'This item must underline a {want_kind} (the stem says "{kind}").')
    if t == "words_in_context_meaning":
        m = MEANING_STEM.match(norm(stem))
        unders = re.findall(r"<u>(.*?)</u>", passage, flags=re.S)
        if not m:
            r.fails.append('Use the exact stem: As used in the text, what does the word "[word]" most nearly mean? '
                           '(write "phrase" instead of "word" for a two- or three-word expression).')
        if len(unders) != 1:
            r.fails.append("Underline the target word or phrase exactly once in the passage with <u></u>.")
        elif m and norm(unders[0]).lower().strip(".,;:") != m.group(2).lower().strip(".,;:"):
            r.fails.append(f'The stem quotes "{m.group(2)}" but the passage underlines "{unders[0]}"; they must match.')
        elif m and (m.group(1) == "word") != (len(unders[0].split()) == 1):
            r.fails.append('Say "word" in the stem for one word and "phrase" for more than one.')
        if any(len(v.split()) > 4 for v in choices.values()):
            r.fails.append("Meaning choices must be short: a word or a phrase of at most four words.")

    if t in TSP_TYPES and choices:
        opener = "To" if t == "text_structure_purpose" else "It"
        bad_open = [v for v in choices.values() if not v.strip().startswith(opener + " ")]
        if bad_open:
            r.fails.append(f'Every choice must open with "{opener} [verb]..." (for example "{opener} '
                           f'{"describe" if opener == "To" else "describes"}...").')
        verbs = [lead_verb(v) for v in choices.values()]
        preferred = [v for v in verbs if v in TSP_VERBS]
        if len(preferred) < 2:
            r.fails.append("At least two choices should open with a verb from the preferred list (suggest, indicate, "
                           "emphasize, convey, describe, illustrate, summarize, present, reflect, represent, introduce, "
                           "establish, demonstrate, provide, discuss, evaluate, compare, explain, capture).")
        if len(set(v for v in verbs if v)) < len([v for v in verbs if v]):
            r.warns.append("Two choices open with the same verb; real items usually use four different verbs.")
        if choice_band:
            lo, hi = choice_band
            off = {k: len(v.split()) for k, v in choices.items() if not lo - 1 <= len(v.split()) <= hi + 1}
            if off:
                r.fails.append(f"Every choice must be {lo}-{hi} words for this item; these are not: "
                               + ", ".join(f"{k} ({n} words)" for k, n in sorted(off.items())) + ".")

    # --- stems ---
    want = SEC_STEM if t.startswith(("boundaries_", "form_")) else FIXED_STEMS.get(t)
    if want and norm(stem) != norm(want):
        r.fails.append(f'Use the exact stem for this type: "{want}"')

    # --- structure by type ---
    if t == "command_of_evidence_quantitative":
        rows = [ln for ln in passage.splitlines() if ln.strip().startswith("|")]
        if len(rows) < 4:
            r.fails.append("Include a titled Markdown table with a header row and at least two data rows.")
    if t == "cross_text_connections" and not ("Text 1" in passage and "Text 2" in passage):
        r.fails.append('Label the two passages "Text 1" and "Text 2".')
    if t == "rhetorical_synthesis":
        notes = [ln for ln in passage.splitlines() if ln.strip().startswith("- ")]
        if not 4 <= len(notes) <= 6:
            r.fails.append(f"Rhetorical synthesis needs 4-6 notes, each on a line starting with '- '; found {len(notes)}.")
        if "student has taken the following notes" not in passage:
            r.fails.append('Begin with "While researching a topic, a student has taken the following notes:"')
    if "LITERATURE" in subject.upper() and "STUDIES" not in subject.upper() and "invented" not in subject:
        intro = passage.lstrip()
        if not intro.startswith("The following text is from"):
            r.fails.append('A literary excerpt must open with "The following text is from ..."')
        else:
            first = re.split(r"(?<=[.!?][\"\u201d])\s+|(?<=[.!?])\s+(?=[A-Z])", intro, maxsplit=1)
            head = first[0]
            if not re.search(r"\b(18[2-9]\d|19[01]\d|192[0-5])\b", head) or not re.search(r"['\u2019]s\b", head) \
                    or not re.search(r"[\"\u201c][^\"\u201d]{3,}[\"\u201d]", head):
                r.fails.append('Make the introduction specific: author\'s full name, year (1820-1925), form, and '
                               'title in quotation marks, e.g. The following text is from [Author]\'s 1874 poem '
                               '"[Title]." Author and title are invented.')
            context = first[1] if len(first) > 1 else ""
            context = context.split("\n")[0]
            if len(context.split()) < 8:
                r.fails.append("After the title, add one specific sentence of context: who the speaker or character "
                               "is and what situation they are in.")
        if "[Invented work]" not in explanation:
            r.fails.append('End the explanation with "[Invented work]".')

    # --- tells a test-wise student can exploit (reading-style items only) ---
    if not t.startswith(SHORT_CHOICE_TYPES) and key and wrong:
        lens = {k: len(v) for k, v in choices.items()}
        longest = max(lens.values())
        mean_wrong = sum(len(w) for w in wrong) / len(wrong)
        if lens[answer] == longest and lens[answer] > 1.25 * mean_wrong:
            r.fails.append(f"The correct answer is the longest choice ({lens[answer]} characters vs. an average of "
                           f"{mean_wrong:.0f} for the wrong ones). Make the choices similar in length, and make at "
                           f"least one wrong answer as long and detailed as the key.")
        elif lens[answer] == longest:
            r.warns.append("The correct answer is the longest choice (small gap).")
        if min(lens.values()) and longest / min(lens.values()) > 2.6:
            r.warns.append("Choice lengths vary a lot; real choices look parallel.")

        abs_wrong = [w for w in wrong if any(has_word(w, a) for a in ABSOLUTES)]
        abs_key = any(has_word(key, a) for a in ABSOLUTES)
        if len(abs_wrong) >= 2 and not abs_key:
            found = sorted({a for w in abs_wrong for a in ABSOLUTES if has_word(w, a)})
            r.fails.append(f"Absolute words ({', '.join(found)}) appear only in wrong answers, so students can "
                           f"eliminate them without reading. Keep at most one overstated distractor and make the "
                           f"others wrong for reasons that need the passage.")

        content = lambda s: {w for w in words(s) if w not in STOP and len(w) > 3}  # noqa: E731
        ptext = content(text)
        def overlap(s: str) -> float:
            c = content(s)
            return len(c & ptext) / len(c) if c else 0.0
        if t in ("details", "central_ideas") and overlap(key) - max(overlap(w) for w in wrong) > 0.35:
            r.warns.append("The key copies far more passage wording than any wrong answer; paraphrase the key "
                           "or add a word-trap distractor.")

    # --- short-choice types ---
    if t.startswith(("boundaries_lists", "boundaries_independent", "boundaries_dependent",
                     "boundaries_supplements", "boundaries_unnecessary", "boundaries_transition")):
        bare = {" ".join(words(v)) for v in choices.values()}
        if len(bare) > 1:
            r.fails.append("Boundaries choices must use the same words and differ only in punctuation.")
    if t == "transitions":
        groups = []
        for v in choices.values():
            g = next((name for name, items in TRANSITION_GROUPS.items()
                      if norm(v).lower().rstrip(",") in items), None)
            if g:
                groups.append(g)
        dup = {g for g in groups if groups.count(g) > 1}
        if dup:
            r.fails.append(f"Two choices come from the same transition family ({', '.join(sorted(dup))}), so both "
                           f"could fit. Use four different families.")

    # --- content rules ---
    low = text.lower()
    hits = [w for w in OVERUSED_TOPICS
            if re.search(r"(?<![a-z])" + re.escape(w) + (r"[a-z]*" if w == "pollinat" else r"(?:s|es)?(?![a-z])"), low)]
    if hits:
        r.fails.append(f"Overused topic ({', '.join(hits)}). Pick a less common topic in the same subject area.")
    if re.search(r"\byou(r)?\b", low) and "LITERATURE" not in subject.upper() and '"' not in passage:
        r.warns.append('Informational passages should not address the reader as "you".')

    # --- explanation must survive shuffling ---
    if re.search(r"\b(?:[Cc]hoice|[Oo]ption|[Aa]nswer)\s+\(?[A-D]\)?(?![a-z])|\([A-D]\)", explanation):
        r.fails.append("The explanation refers to a choice by letter; choices are shuffled, so quote its first words.")
    if "[Domain:" not in explanation:
        r.warns.append("Explanation is missing the [Domain] [Skill] [Difficulty] [Subject] header.")
    return r


# ----------------------------------------------------------------------------- rhythm
# Each question is written on its own, so the writer settles into one rhythm (usually "a belief,
# then a twist"). These helpers describe a passage's shape so meza_satgen.py can show the writer
# its recent questions and refuse a draft that repeats them.

CONTRAST_MARKERS = ["however", "yet", "but", "although", "though", "even though", "instead", "despite",
                    "in spite of", "whereas", "nevertheless", "nonetheless", "rather than", "in contrast",
                    "by contrast", "on the other hand", "conversely", "contrary to", "even so", "still,"]
SETUP_MARKERS = ["long believed", "long assumed", "long cited", "long thought", "once believed", "once assumed",
                 "had expected", "had assumed", "traditionally", "conventional wisdom", "commonly assumed",
                 "widely assumed", "widely believed", "it was thought", "previously thought", "many assume",
                 "assumption that"]
# Types where a contrast is part of the design, so the rhythm rule doesn't apply.
CONTRAST_NATIVE = ("cross_text_connections", "command_of_evidence_claim", "transitions", "boundaries_", "form_")


def _sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text.strip()) if x]


def features(qtype: str, passage: str, key_text: str = "", clue: str = "") -> dict:
    """A short description of a passage's shape: how it opens, whether it twists, where the blank sits."""
    text = reading_text(passage)
    body = text
    low = " " + body.lower() + " "
    low_nb = re.sub(r"not only(.*?)\bbut\b", r"\1", low)  # "not only ... but" is an extension, not a contrast
    contrast = [m for m in CONTRAST_MARKERS if has_word(low_nb, m.rstrip(","))
                and (m != "still," or "still," in low_nb) and (m != "but" or " but " in low_nb)]
    setup = [m for m in SETUP_MARKERS if m in low]
    sents = _sentences(body)
    pos = next((i for i, x in enumerate(sents) if BLANK in x), None)
    target = re.search(r"<u>(.*?)</u>", passage, flags=re.S)
    if base_type(qtype) == "words_in_context_meaning" and target:
        word = " ".join(target.group(1).split())
        pos = next((i for i, x in enumerate(sents) if word in x), None)
        key_text = word  # for meaning items, the word being tested is what must not repeat
    where = None if pos is None else ("first" if pos == 0 else "last" if pos == len(sents) - 1 else "middle")
    return {
        "type": base_type(qtype),
        "opening": " ".join(body.split()[:4]),
        "twist": bool(contrast or setup),
        "markers": contrast + setup,
        "blank": where,
        "marker": "underlined word" if base_type(qtype) == "words_in_context_meaning" else "blank",
        "sentences": len(sents),
        "key": key_text,
        "clue": clue,
    }


def describe(f: dict) -> str:
    rhythm = f"a belief or expectation, then a twist ({', '.join(f['markers'][:3])})" if f["twist"] else "no twist"
    parts = [f'opens "{f["opening"]}..."', f"rhythm: {rhythm}", f"{f['sentences']} sentences"]
    if f["blank"]:
        parts.append(f"{f.get('marker', 'blank')} in the {f['blank']} sentence")
    if f["clue"]:
        parts.append(f"clue structure: {f['clue']}")
    if f["key"]:
        label = ("tested word" if f["type"] == "words_in_context_meaning" else
                 "correct answer opens with" if f["type"] in TSP_TYPES else "answer")
        parts.append(f'{label}: "{f["key"][:40]}"')
    return "; ".join(parts)


def rhythm_check(f: dict, recent: list[dict]) -> list[str]:
    """Fails for a draft that repeats the shape of the same type's recent questions (newest last)."""
    fails = []
    t = f["type"]
    if t.startswith("words_in_context") and f["clue"] and f["clue"] != "contrast" and f["twist"]:
        fails.append(f'This item\'s clue structure is "{f["clue"]}", which must NOT use a contrast or an '
                     f'"expected/believed, then actually" setup (you used: {", ".join(f["markers"])}). Rewrite the '
                     f"passage with no contrast words and no twist; the clue must come from the structure you were given.")
    last2 = recent[-2:]
    if (not t.startswith(CONTRAST_NATIVE) and f["clue"] != "contrast" and f["twist"]
            and len(last2) == 2 and all(r["twist"] for r in last2)):
        fails.append("Your last two questions of this type both used a belief-then-twist rhythm. This one must not: "
                     "remove the contrast (however, yet, but, although, instead...) and build the passage another way.")
    for r in recent[-4:]:
        if f["opening"] and " ".join(f["opening"].lower().split()[:2]) == " ".join(r["opening"].lower().split()[:2]):
            fails.append(f'The passage opens the same way as a recent question ("{r["opening"]}..."). Open differently.')
            break
    if t.startswith("words_in_context") and f["blank"] and len(recent) >= 2 and all(
            r["blank"] == f["blank"] for r in recent[-2:]):
        fails.append(f"The {f.get('marker', 'blank')} is in the {f['blank']} sentence again, like your last two "
                     f"questions of this type. Put it somewhere else.")
    if f["key"] and any(f["key"].strip().lower() == r["key"].strip().lower() for r in recent):
        what = ("tested word" if t == "words_in_context_meaning" else
                "opening verb of the correct answer" if t in TSP_TYPES else "answer")
        fails.append(f'"{f["key"]}" was the {what} in a recent question of this type. Use a different word.')
    return fails


# ----------------------------------------------------------------------------- vocabulary


def _forms(word: str) -> set[str]:
    """Common forms of a word (abate/abated/abating, cogent/cogency, belie/belied, laud/laudable)."""
    w = word.lower()
    f = {w, w + "s", w + "es", w + "ed", w + "d", w + "ing", w + "ly", w + "ness", w + "al", w + "ally"}
    if w.endswith("e"):
        f |= {w[:-1] + "ing", w[:-1] + "ation", w[:-1] + "ion", w[:-1] + "able", w[:-1] + "ive", w[:-1] + "ity"}
    if w.endswith("y"):
        f |= {w[:-1] + "ies", w[:-1] + "ied", w[:-1] + "ily", w[:-1] + "iness"}
    if w.endswith("ie"):
        f |= {w[:-2] + "ying"}
    for a, b in (("ent", "ence"), ("ent", "ency"), ("ant", "ance"), ("ant", "ancy"), ("ence", "ent"),
                 ("ance", "ant"), ("ous", "ously"), ("ous", "ousness"), ("ic", "ically"), ("ate", "ation"),
                 ("ity", "ous"), ("ion", "e"), ("able", "ably"), ("ible", "ibly"), ("ize", "ization")):
        if w.endswith(a):
            f.add(w[: -len(a)] + b)
    if not w.endswith("e"):
        f |= {w + "able", w + "ation", w + "ment"}
        if len(w) > 2 and w[-1] in "bdglmnprt" and w[-2] in "aeiou" and w[-3] not in "aeiou":
            f |= {w + w[-1] + "ed", w + w[-1] + "ing"}  # mar -> marred, marring
    return f


def vocab_used(text: str, vocab: list[str]) -> list[str]:
    low = re.sub(r"</?u>", "", text).lower()
    tokens = set(re.findall(r"[^\W\d_]+", low))
    out = []
    for w in vocab:
        if " " in w:  # a phrase such as "grapple with": any form of the first word, then the rest
            first, rest = w.lower().split(" ", 1)
            alts = "|".join(sorted((re.escape(f) for f in _forms(first)), key=len, reverse=True))
            if re.search(r"(?<![a-z])(?:" + alts + r")\s+" + re.escape(rest) + r"(?![a-z])", low):
                out.append(w)
        elif _forms(w) & tokens:
            out.append(w)
    return out


_BANK: dict[str, str] | None = None  # every form of every bank word -> tier
try:  # optional: precise word frequency for words that aren't in the bank (pip install wordfreq)
    from wordfreq import zipf_frequency as _zipf
except ImportError:  # pragma: no cover
    _zipf = None


def _bank() -> dict[str, str]:
    global _BANK
    if _BANK is None:
        _BANK = {}
        path = Path(__file__).with_name("data") / "meza_vocab_bank.json"
        if path.exists():
            for w in json.loads(path.read_text(encoding="utf-8"))["words"]:
                for f in _forms(w["word"]):
                    _BANK.setdefault(f, w["tier"])
    return _BANK


def word_tier(word: str) -> str | None:
    """advanced / mid / common, or None when unknown (multi-word phrases, or not in the bank without wordfreq)."""
    w = word.strip().lower().strip(".,;:")
    if not w or " " in w:
        return None
    if _zipf is not None:
        z = _zipf(w, "en")
        return "advanced" if z < 3.0 else "mid" if z < 3.8 else "common"
    return _bank().get(w)


def advanced_in(passage: str) -> list[str]:
    """Advanced bank words that appear in a passage (only the bank is used, so names and technical terms don't count)."""
    bank = _bank()
    tokens = set(re.findall(r"[^\W\d_]+", reading_text(passage).lower()))
    return sorted({t for t in tokens if bank.get(t) == "advanced"})


def choice_mix_problems(choices: dict[str, str], answer: str, mix: dict[str, int], key_tier: str) -> list[str]:
    """Checks the level mix of single-word choices. Words whose level is unknown (phrases, or words outside the
    bank when wordfreq isn't installed) are skipped rather than guessed."""
    tiers = {k: word_tier(v) for k, v in choices.items()}
    known = [t for t in tiers.values() if t]
    out = []
    n_adv = known.count("advanced")
    want_adv = mix.get("advanced", 0)
    if n_adv > want_adv or (len(known) == 4 and n_adv < want_adv):
        out.append(f"The choices include {n_adv} advanced word(s); this item needs exactly {want_adv}. Real items mix "
                   f"levels: {', '.join(f'{n} {t}' for t, n in mix.items())}.")
    if tiers[answer] and tiers[answer] != key_tier:
        out.append(f'The correct answer "{choices[answer]}" is a {tiers[answer]} word, but this item\'s correct answer '
                   f"must be the {key_tier} word.")
    return out


# ----------------------------------------------------------------------------- traps (Text Structure and Purpose)
# Real wrong answers are built from the passage's own content and are wrong about its PURPOSE or SHAPE.
# A wrong answer that brings in a topic, comparison, or claim the passage never touches is too easy to cross out.

TSP_TRAPS = {
    "too_narrow": "states the role of one detail or one sentence as if it were the purpose of the whole text",
    "too_broad": "claims the text does more, or covers more, than it actually does",
    "wrong_action": "right content, wrong action or strength (argues, criticizes, or proves where the text only "
                    "describes or suggests; or the reverse)",
    "wrong_relationship": "gets how the parts relate backwards (says a detail supports a claim it actually "
                          "complicates, or that a view is rejected when it is only qualified)",
    "wrong_order": "lists moves the text really makes, but in the wrong sequence",
    "partial": "describes only some of the text's moves and leaves out a key one",
    "wrong_target": "describes the role of a different sentence or part of the text",
    "misread_stance": "misjudges the author's or speaker's attitude (too strong, too weak, or the wrong feeling)",
}
TRAP_CRITIC_SYSTEM = "You are a veteran Digital SAT item reviewer who judges how well wrong answers are built."
TRAP_CRITIC_PROMPT = """Here is a Reading and Writing question with its correct answer marked.

{question}

Correct answer: {answer}

Judge each WRONG answer strictly, the way a veteran Digital SAT reviewer would. On real items, most wrong answers
are statements a student who SKIMMED the passage would accept as accurate: their facts match what the text says,
and only the purpose, scope, order, or role is wrong. Weak wrong answers lean on a word the text barely touches,
contradict an explicit statement, or describe something the text never does.

For each wrong answer give:
- accurate_content: true only if every fact or detail it mentions is something the passage actually says, stated
  the way the passage states it (it may still get the purpose or role wrong). False if it adds, exaggerates, or
  contradicts anything, or builds on a passing word (for example "compares X with other animals" when the text only
  says "of any animal").
- on_topic: false if it brings in a topic, comparison, event, or claim the passage never mentions.
- tempting: 1-5 for a student who read the passage once, quickly. 1 = crossed out instantly; 2 = clearly off on a
  second look; 3 = needs real thought; 4-5 = very hard to rule out. Be strict: anything that contradicts the text
  or rests on a passing word is 1 or 2.
- trap: one of: {traps}, or "none".

Reply with JSON only, keyed by letter, for the three wrong answers:
{{"<letter>": {{"accurate_content": true, "on_topic": true, "tempting": 3, "trap": "<type>", "why": "<one short sentence>"}}}}"""


def trap_label_problems(traps: object, answer: str, level: str) -> list[str]:
    """Checks the writer's own labels: one valid trap type per wrong answer; harder items mix types."""
    wrong = [k for k in CHOICE_KEYS if k != answer]
    if not isinstance(traps, dict) or any(str(traps.get(k, "")).strip() not in TSP_TRAPS for k in wrong):
        return ['Add a "traps" field to the JSON that maps each WRONG answer\'s letter to one trap type from: '
                + ", ".join(TSP_TRAPS) + "."]
    kinds = {str(traps[k]).strip() for k in wrong}
    if level in ("medium", "hard") and len(kinds) < 2:
        return ["Use at least two different trap types among the wrong answers."]
    return []


def trap_critic_problems(verdict: dict, answer: str, level: str) -> list[str]:
    """Off-topic or 'none' traps always fail. At least two wrong answers must be accurate about the passage's
    content (wrong only in role/scope). Medium/hard: every wrong answer tempting >= 3; easy: at most one below 3."""
    wrong = [k for k in CHOICE_KEYS if k != answer]
    if not isinstance(verdict, dict) or not any(isinstance(verdict.get(k), dict) for k in wrong):
        return []  # critic unavailable: don't block on it
    off, weak, inaccurate, very_weak = [], [], [], []
    for k in wrong:
        v = verdict.get(k)
        if not isinstance(v, dict):
            continue
        why = str(v.get("why", "")).strip()
        label = f"{k} ({why})" if why else k
        if v.get("on_topic") is False or str(v.get("trap", "")).lower() == "none":
            off.append(label)
            continue
        if v.get("accurate_content") is False:
            inaccurate.append(label)
        try:
            t = int(v.get("tempting", 3))
        except (TypeError, ValueError):
            t = 3
        if t <= 2:
            weak.append(label)
            if t <= 1:
                very_weak.append(label)
    out = []
    if off:
        out.append("These wrong answers bring in content the passage never mentions or target no real mistake, so "
                   "students can cross them out without understanding the text: " + "; ".join(off) + ". Rebuild each "
                   "from the passage's own content, wrong only about its purpose, structure, or the role of a part.")
    if len(inaccurate) > 1:
        out.append("Too many wrong answers misstate the passage's content: " + "; ".join(inaccurate) + ". At least two "
                   "wrong answers must describe what the passage really says, accurately, and be wrong only about its "
                   "purpose, scope, order, or role, so a student who skimmed would accept them.")
    allowed = {"easy": 1, "medium": 1, "hard": 0}.get(level, 0)
    if len(weak) > allowed or (level != "easy" and very_weak):
        out.append("These wrong answers are too easy to rule out: " + "; ".join(weak) + ". Make each a near miss: an "
                   "accurate statement about the passage with the wrong scope, action, relationship, order, or target.")
    return out


# ----------------------------------------------------------------------------- word focus (Words in Context)
# On real items the tested word is usually a detail in one sentence; the passage's main idea is about something else.
# A passage built entirely around the word (every sentence hinting at its meaning) is too easy to solve.

FOCUS_CRITIC_SYSTEM = "You are a veteran Digital SAT item reviewer."
FOCUS_CRITIC_PROMPT = """Here is a Words in Context question. The tested word is the blank or the underlined word.

{question}

Answer with JSON only:
{{"main_idea": "<the passage's main idea in one sentence>",
  "word_is_main_idea": <true if the passage is mainly ABOUT the quality, action, or idea the tested word names>,
  "hint_sentences": <how many sentences or clauses exist mainly to point to the tested word's meaning>}}"""


def focus_problems(verdict: dict, role: str) -> list[str]:
    if role != "detail" or not isinstance(verdict, dict) or "hint_sentences" not in verdict:
        return []
    try:
        hints = int(verdict.get("hint_sentences", 0))
    except (TypeError, ValueError):
        hints = 0
    if verdict.get("word_is_main_idea") is True or hints > 1:
        return [f"The passage is built around the tested word (main idea: {verdict.get('main_idea', '?')}; "
                f"{hints} sentence(s) mainly hint at its meaning). For this item the word must be a DETAIL: write a "
                "passage whose main idea is about something else, put the word in one sentence, and let only that "
                "sentence (and at most one nearby phrase) supply the clue."]
    return []


# ----------------------------------------------------------------------------- probe

PROBE_SYSTEM = ("You are an expert test-taker who knows every trick for guessing on multiple-choice "
                "reading tests: answer length, absolute words, hedged language, choices that sound "
                "most 'academic', choices that echo the question, and so on.")
PROBE_PROMPT = """The reading passage for this SAT question has been hidden from you. Using ONLY the question
and the four choices, pick the choice most likely to be correct. You must pick one.
Reply with JSON only: {{"answer": "<A, B, C or D>"}}

{stem}

{choices}"""


def probe_votes(ask, stem: str, choices: dict[str, str], answer: str, runs: int = 3,
                rng: random.Random | None = None) -> list[bool]:
    """Ask a passage-free solver `runs` times, reshuffling the choices each time so letter bias
    can't help. `ask(prompt, system)` returns the model's text. Returns hits (True = found the key)."""
    rng = rng or random.Random()
    key_text = choices[answer]
    hits = []
    for _ in range(runs):
        texts = list(choices.values())
        rng.shuffle(texts)
        shown = dict(zip(CHOICE_KEYS, texts))
        block = "\n".join(f"{k}) {v}" for k, v in shown.items())
        reply = ask(PROBE_PROMPT.format(stem=stem, choices=block), PROBE_SYSTEM)
        m = re.search(r'"answer"\s*:\s*"?\s*([A-D])', reply) or re.search(r"\b([A-D])\b", reply)
        hits.append(bool(m) and shown.get(m.group(1)) == key_text)
    return hits


def probe_feedback(hits: list[bool]) -> str:
    return (f"A test-taker who could NOT see the passage still picked the correct answer {sum(hits)} of "
            f"{len(hits)} times, using only the choices. The key stands out on its own (it may be the most "
            f"balanced, the most hedged, the most complete, or the only one that sounds reasonable). Rewrite "
            f"the wrong answers so each one sounds just as reasonable as the key until you check the passage.")


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    total = failing = 0
    tallies: dict[str, int] = {}
    key_longest = 0
    for path in argv:
        for i, q in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))):
            qtype = q.get("type") or q.get("qtype") or ""
            m = re.search(r"\[Subject:\s*([^\]]+)\]", q.get("explanation", ""))
            res = lint(qtype, q.get("passage", ""), q["stem"], q["choices"], q["answer"],
                       q.get("explanation", ""), subject=m.group(1) if m else "")
            total += 1
            lens = {k: len(v) for k, v in q["choices"].items()}
            key_longest += lens[q["answer"]] == max(lens.values())
            if res.fails or res.warns:
                print(f"\n{path} #{i} [{qtype}]")
                for f in res.fails:
                    print("  FAIL", f)
                    tallies[f.split(" (")[0].split(".")[0][:60]] = tallies.get(f.split(" (")[0].split(".")[0][:60], 0) + 1
                for w in res.warns:
                    print("  warn", w)
            failing += not res.ok
    print(f"\n{failing} of {total} questions fail at least one check.")
    print(f"Correct answer is the longest choice in {key_longest} of {total} questions "
          f"(about {total // 4} expected by chance).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
