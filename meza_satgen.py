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
from meza_quality import advanced_in, choice_mix_problems, word_tier
from satgen import cli, pipeline
from satgen.answers import normalize_spr
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
    "details", "central_ideas", "inferences", "words_in_context", "text_structure",
    "cross_text_connections",
)

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


REPAIR_BLOCK = (
    "\n\nYOUR PREVIOUS DRAFT OF THIS QUESTION FAILED QUALITY CHECKS.\n"
    "Previous draft (JSON):\n{draft}\n\n"
    "Problems to fix:\n{problems}\n\n"
    "Write a corrected version. Keep the same question type, difficulty, and subject area. "
    "You may keep the passage if it was not part of the problem. Return the full JSON again."
)

QUALITY = os.environ.get("MEZA_QUALITY", "on").lower()
MAX_REPAIRS = int(os.environ.get("MEZA_MAX_REPAIRS", "2"))
PROBE_RUNS = int(os.environ.get("MEZA_PROBE_RUNS", "3"))
PROBE_FAIL = int(os.environ.get("MEZA_PROBE_FAIL", "3"))
LOG_PATH = os.environ.get("MEZA_QUALITY_LOG", "")
_log_lock = threading.Lock()
VOCAB_PATH = Path(os.environ.get("MEZA_VOCAB", Path(__file__).with_name("data") / "vocab_bank.json"))
RECENT_KEEP = 5
RECENT_BLOCK = (
    "\n\nYOUR MOST RECENT QUESTIONS OF THIS TYPE (newest last):\n{items}\n"
    "Make this question clearly different from them: a different opening, a different sentence rhythm, the blank "
    "(if any) in a different place, and a different answer word.{extra}"
)


# Harder passages draw advanced words from the vocabulary bank (data/vocab_bank.json, built from the
# Barron's deck: words and parts of speech only). Words are dealt like cards, so a run uses the whole
# bank before repeating one.
VOCAB_RULES = {
    # difficulty: (bank words offered, minimum to use, most advanced bank words allowed in the passage, tiers dealt)
    "easy": (0, 0, 1, ()),
    "medium": (2, 1, 1, ("mid", "advanced")),
    "hard": (3, 1, 2, ("advanced", "mid", "advanced")),
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

    def passage_words(self, difficulty: str) -> tuple[list[str], int, int]:
        n, need, cap, tiers = VOCAB_RULES.get(difficulty, (0, 0, 1, ()))
        words: list[str] = []
        for i in range(n):
            words += self.deal(tiers[i % len(tiers)], 1)
        return words, (need if words else 0), cap

    def targets(self, n: int) -> list[str]:
        """Candidate tested words for meaning items: everyday and mid-level words, many with several senses."""
        out = self.deal("common", n // 2 + 1) + self.deal("mid", n - n // 2 - 1)
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
        self.vocab = VocabBank(VOCAB_PATH, seed)
        self._recent = {}
        print(f"Subject seed: {seed} (set MEZA_SUBJECT_SEED={seed} to repeat this order)", file=sys.stderr)
        if QUALITY != "off":
            print(f"Quality gate: {QUALITY}, up to {MAX_REPAIRS} rewrites per question", file=sys.stderr)
        return super().run(qtypes, count, *args, **kwargs)

    def _draft(self, prompt: str, qtype: str, fmt: str) -> tuple[Question, dict]:
        text = self._call(self.generator, "generate", prompt, system=pipeline.GEN_SYSTEM,
                          temperature=0.9, json_mode=True)
        data = pipeline.parse_json(text)
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
        slot = self.plan[index] if index < len(self.plan) else index % len(SUBJECTS)
        subject = subject_for(qtype, slot)
        fmt = SPR if qtype.startswith("spr_") else MCQ
        base_prompt = pipeline.GEN_PROMPT.format(
            qtype=qtype,
            index=index,
            context=self.context,
            format_rules=pipeline.SPR_RULES if fmt == SPR else pipeline.MCQ_RULES,
        ) + SUBJECT_LINE.format(subject=subject)
        clue = ""
        if base_type(qtype).startswith("words_in_context"):
            ci = self.clues[index] if index < len(self.clues) else index % len(WIC_CLUES)
            clue, _, desc = WIC_CLUES[ci]
            if base_type(qtype) == "words_in_context_meaning":
                if clue == "restatement":  # a colon explaining the word would give the meaning away
                    clue, _, desc = next(c for c in WIC_CLUES if c[0] == "exposition")
                desc = desc.replace("the blank", "the underlined word")
            base_prompt += CLUE_LINE.format(desc=desc)
        if "_recent" not in self.__dict__:
            self._recent = {}
        if "vocab" not in self.__dict__:
            self.vocab = VocabBank(VOCAB_PATH, index)
        level = difficulty_of(qtype, index) or "medium"
        vocab, need, cap, plan = [], 0, 9, None
        if fmt == MCQ and QUALITY != "off" and not base_type(qtype).startswith(NO_VOCAB_TYPES):
            vocab, need, cap = self.vocab.passage_words(level)
            if vocab:
                base_prompt += VOCAB_LINE.format(level=level, need_text="one or two" if cap > 1 else "one",
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
            twisty = sum(r["twist"] for r in recent[-3:]) >= 2 and clue != "contrast"
            extra = (" Several of them used a belief-then-twist rhythm, so this one must not use a contrast at all."
                     if twisty else "")
            items = "\n".join(f"{i}. {describe(r)}" for i, r in enumerate(recent, 1))
            base_prompt += RECENT_BLOCK.format(items=items, extra=extra)

        # The offline demo's simulated writer can't follow the rulebook; skip the gate there.
        gate = QUALITY != "off" and fmt == MCQ and not isinstance(self.generator, SimulatedClient)
        prompt, problems = base_prompt, []
        for attempt in range(MAX_REPAIRS + 1 if gate else 1):
            try:
                q, data = self._draft(prompt, qtype, fmt)
                q.validate()
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                if not gate or attempt == MAX_REPAIRS:
                    raise
                problems = [f"The draft could not be used: {exc}. Follow the JSON format exactly."]
                log_attempt({"index": index, "qtype": qtype, "attempt": attempt, "invalid": str(exc)})
                prompt = base_prompt + REPAIR_BLOCK.format(draft="(unusable)", problems="\n".join(problems))
                continue
            if not gate:
                return q

            res = lint(qtype, q.passage, q.stem, q.choices, q.answer, q.explanation, subject=subject, clue=clue)
            shape = features(qtype, q.passage, q.choices.get(q.answer, "") if base_type(qtype).startswith(
                ("words_in_context", "transitions")) else "", clue)
            problems = list(res.fails) + rhythm_check(shape, recent)
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
            if not problems and QUALITY == "on" and base_type(qtype).startswith(PROBE_TYPES):
                hits = self._probe(q, index, attempt)
                if sum(hits) >= PROBE_FAIL:
                    problems.append(probe_feedback(hits))
            log_attempt({"index": index, "qtype": qtype, "subject": subject, "clue": clue, "attempt": attempt,
                         "fails": res.fails, "warns": res.warns, "probe_hits": sum(hits),
                         "probe_runs": len(hits), "shape": shape, "vocab": vocab, "vocab_used": used,
                         "choice_plan": plan, "passed": not problems})
            if not problems:
                self._recent.setdefault(base_type(qtype), []).append(shape)
                del self._recent[base_type(qtype)][:-RECENT_KEEP]
                return q
            draft = json.dumps({k: data.get(k) for k in ("passage", "stem", "choices", "answer", "explanation")},
                               ensure_ascii=False, indent=1)
            prompt = base_prompt + REPAIR_BLOCK.format(draft=draft, problems="\n".join(f"- {p}" for p in problems))
        raise ValueError("quality gate: " + " | ".join(p[:120] for p in problems))


pipeline.Pipeline = MezaPipeline

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] != ["run"]:
        args = ["run", *args]
    sys.exit(cli.main(args))
