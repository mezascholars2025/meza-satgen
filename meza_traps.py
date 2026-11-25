"""Trap answers: per-skill trap types, a bank of proven examples, and the learning loop.

Every multiple-choice skill with reading content gets:
  - a list of trap types (the mistake each wrong answer is built on), which the writer must label;
  - guidance for the reviewer model that rates each wrong answer (the "trap critic");
  - a few proven examples from data/trap_bank.json, shown to the writer as models of strong wrong answers.

The bank only holds ORIGINAL questions (Meza's own or this generator's; see the Source rule in CLAUDE.md).
It grows through review:

    python meza_traps.py learn runs/<run>      add questions marked GOOD TRAPS in the run's questions.csv
    python meza_traps.py candidates runs/<run> list questions whose traps scored high, worth a look
    python meza_traps.py list                  what the bank holds, per skill
    python meza_traps.py approve <id> ...      mark starter examples as reviewed (approved ones are shown first)
    python meza_traps.py remove <id> ...       take examples out
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

BANK_PATH = Path(__file__).with_name("data") / "trap_bank.json"
CHOICE_KEYS = ("A", "B", "C", "D")
DIFFICULTIES = ("easy", "medium", "hard")

# ----------------------------------------------------------------------------- trap types per skill family

TRAPS = {
    "tsp": {
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
    },
    "wic": {
        "most_common_meaning": "the word's everyday meaning, which does not fit this context",
        "other_real_meaning": "another real meaning of the word (or a near-synonym of one) that a hurried reader could "
        "fit into the sentence",
        "wrong_connotation": "close in meaning but with the wrong feeling (implies blame, praise, or guilt the passage "
        "does not)",
        "wrong_degree": "right direction, wrong strength (too weak or too strong for what the passage says)",
        "reversal": "the opposite of what the context requires, tempting to a student who misreads the logic",
        "borrowed_from_passage": "fits a different idea elsewhere in the passage, not the blank or the tested word",
        "imprecise_fit": "grammatical and on topic, but not the precise action or quality the context calls for",
    },
    "cid": {  # Central Ideas and Details (central_ideas, details)
        "never_said": "a real detail from the passage plus something it does not support (a cause, comparison, "
        "preference, reaction, or timeline)",
        "opposite": "the passage's own words with the meaning flipped",
        "mix_up": "the wrong person, factor, or time order, or something already known presented as a result",
        "half_right": "one accurate part and one wrong part",
        "goes_too_far": "a reasonable idea pushed past what the passage supports",
        "word_echo": "repeats a striking word or detail from the passage but changes the claim",
        "twin": "keeps the correct answer's structure but changes one element",
        "wrong_tone": "misreads a character's or speaker's feeling (literary excerpts only)",
    },
    "inf": {  # Inferences
        "never_said": "a claim on the same topic that the passage does not get into",
        "real_world_true": "plausible from general knowledge but not supported by the passage",
        "old_view": "restates the view the passage pushes against",
        "overreach": "sounds like the natural takeaway but goes further than the evidence",
        "twin": "keeps the correct answer's structure but flips one direction or element",
        "half_right": "the first part checks out; the second does not",
        "opposite": "reverses the passage's logic",
    },
    "coe": {  # Command of Evidence (claim, quotation, quantitative)
        "true_but_irrelevant": "on topic and accurate-sounding, but misses the claim's specific point",
        "half_right": "supports part of the claim (half a chain, one of two conditions), not all of it",
        "misread_data": "a wrong value, row, or comparison from the data (data items only)",
        "opposite": "would weaken the claim (or support it, if the stem asks to weaken)",
        "wrong_part": "addresses a different claim or a different part of the passage",
    },
}
FAMILY_OF = {
    "text_structure": "tsp",
    "words_in_context": "wic",
    "central_ideas": "cid",
    "details": "cid",
    "inferences": "inf",
    "command_of_evidence": "coe",
}


def family_of(base: str) -> str | None:
    """'text_structure_function' -> 'tsp'; None for skills without a trap list (grammar, transitions, ...)."""
    for prefix, fam in FAMILY_OF.items():
        if base.startswith(prefix):
            return fam
    return None


# What a strong wrong answer looks like, per family, for the writer and for the critic.
CRITIC_GUIDANCE = {
    "tsp": (
        "On real items, most wrong answers are statements a student who SKIMMED the passage would accept as "
        "accurate: their facts match what the text says, and only the purpose, scope, order, or role is wrong. Weak "
        "wrong answers lean on a word the text barely touches, contradict an explicit statement, or describe "
        "something the text never does."
    ),
    "wic": (
        "Here the choices are words or short phrases. A strong wrong answer fits the sentence grammatically and "
        "would tempt a student who half-knows the words or reads the clue hastily: the word's everyday meaning, "
        "another real meaning, a near-synonym with the wrong connotation or strength, or a word that fits another "
        "idea in the passage. A weak wrong answer is a word no student would consider (unrelated meaning, filler)."
    ),
    "cid": (
        "On real items, most wrong answers stay on the passage's topic and reuse its real details, but add something "
        "the passage does not support, flip a meaning, or mix up who did what. The most tempting one echoes a "
        "striking word (easy, medium) or copies the correct answer's structure with one element changed (hard). A "
        "weak wrong answer is off topic, obviously false, or so extreme that no reader would consider it."
    ),
    "inf": (
        "On real items, wrong answers continue the passage plausibly but are not supported: a real-world truth the "
        "passage never establishes, the old view it argues against, an overreach, or a near-twin of the correct "
        "answer with one direction flipped. A weak wrong answer is clearly unrelated or contradicts the passage "
        "outright. A wrong answer should not stand out by tone alone (for example, the only bland or the only "
        "sweeping choice)."
    ),
    "coe": (
        "On real items, most wrong answers are accurate-sounding and on topic but miss the claim's specific point "
        "(true but irrelevant), or support only part of it. In data items, misreads hide one wrong detail inside an "
        "otherwise accurate statement. A weak wrong answer is obviously off topic or checkable as false in a second."
    ),
}

CRITIC_SYSTEM = "You are a veteran Digital SAT item reviewer who judges how well wrong answers are built."
CRITIC_PROMPT = """Here is a Reading and Writing question with its correct answer marked.

{question}

Correct answer: {answer}

Judge each WRONG answer strictly, the way a veteran Digital SAT reviewer would. {guidance}

For each wrong answer give:
- accurate_content: true only if every fact or detail it mentions is something the passage actually says, stated
  the way the passage states it (it may still be wrong about what the question asks). False if it adds, exaggerates,
  or contradicts anything. For single-word choices, use true.
- on_topic: false if it brings in a topic, comparison, event, or claim unrelated to the passage. For single-word
  choices, use true unless the word has nothing to do with the passage.
- tempting: 1-5 for a student who read the passage once, quickly. 1 = crossed out instantly; 2 = clearly off on a
  second look; 3 = needs real thought; 4-5 = very hard to rule out. Be strict.
- trap: one of: {traps}, or "none" if it targets no real mistake (filler).

Reply with JSON only, keyed by letter, for the three wrong answers:
{{"<letter>": {{"accurate_content": true, "on_topic": true, "tempting": 3, "trap": "<type>", "why": "<one short sentence>"}}}}"""


def critic_prompt(family: str, question_text: str, answer: str) -> str:
    return CRITIC_PROMPT.format(question=question_text, answer=answer, guidance=CRITIC_GUIDANCE[family],
                                traps=", ".join(TRAPS[family]))


# ----------------------------------------------------------------------------- the bank of proven examples


def load_bank(path: Path = BANK_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("examples", [])


def save_bank(examples: list[dict], path: Path = BANK_PATH) -> None:
    data = {
        "about": "Proven wrong-answer sets, shown to the writer as models. Original questions only (see the Source "
        "rule in CLAUDE.md). Grow it with `python meza_traps.py learn runs/<run>`.",
        "examples": examples,
    }
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def pick_examples(bank: list[dict], base: str, level: str, n: int = 2, seed: str = "") -> list[dict]:
    """Up to n examples of the same skill, preferring the same difficulty and Sebastian-approved entries, rotated by
    seed so different questions see different examples (and none gets copied over and over)."""
    fam = family_of(base)
    same_skill = [e for e in bank if e.get("skill") == base]
    pool = same_skill or [e for e in bank if family_of(e.get("skill", "")) == fam]
    if not pool:
        return []
    rng = random.Random(f"examples:{seed}")
    rng.shuffle(pool)
    pool.sort(key=lambda e: (e.get("level") != level, not str(e.get("status", "")).startswith("approved")))
    return pool[:n]


def format_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    parts = ["\nMODEL WRONG ANSWERS: these are reviewed questions with strong wrong answers. Study HOW each wrong "
             "answer is built (which mistake it targets, why a careful student might still pick it). Never reuse "
             "their topics, sentences, or choices; write something new."]
    for i, e in enumerate(examples, 1):
        lines = [f"\nExample {i} ({e['skill']}, {e['level']}):", e["passage"].strip(), e["stem"]]
        lines += [f"{k}) {e['choices'][k]}" for k in CHOICE_KEYS if k in e["choices"]]
        lines.append(f"Correct: {e['answer']}")
        for k in CHOICE_KEYS:
            t = e.get("traps", {}).get(k)
            if t and k != e["answer"]:
                lines.append(f"  {k} ({t['type']}): {t['why']}")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def next_id(bank: list[dict]) -> str:
    nums = [int(m.group(1)) for e in bank if (m := re.match(r"t(\d+)$", str(e.get("id", ""))))]
    return f"t{max(nums, default=0) + 1:03d}"


# ----------------------------------------------------------------------------- learning from reviewed runs


def _fp(passage: str, stem: str) -> str:
    return " ".join((re.sub(r"</?u>", "", passage) + " " + stem).lower().split())[:160]


def _final_entries(run: Path) -> tuple[list[dict], dict]:
    questions = json.loads((run / "questions.json").read_text(encoding="utf-8"))
    log = {}
    if (run / "quality.jsonl").exists():
        for line in (run / "quality.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("passed") and entry.get("fingerprint"):
                log[entry["fingerprint"]] = entry
    return questions, log


def _traps_for(q: dict, entry: dict | None) -> dict:
    """Map the writer's trap labels (made before choices were shuffled) onto the final letters, adding the critic's
    reason for each wrong answer when there is one."""
    if not entry or not entry.get("traps") or not entry.get("choices"):
        return {}
    by_text = {v: k for k, v in entry["choices"].items()}
    critic = entry.get("trap_critic") or {}
    out = {}
    for k, text in q["choices"].items():
        old = by_text.get(text)
        if old and old != entry.get("answer") and old in entry["traps"]:
            why = (critic.get(old) or {}).get("why", "") if isinstance(critic.get(old), dict) else ""
            out[k] = {"type": entry["traps"][old], "why": why}
    return out


def learn(run: Path, path: Path = BANK_PATH) -> list[str]:
    """Add every question whose CSV 'Review status' or 'Reviewer notes' says GOOD TRAPS."""
    questions, log = _final_entries(run)
    marked = set()
    with open(run / "questions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            note = f"{row.get('Review status', '')} {row.get('Reviewer notes', '')}".lower()
            if "good trap" in note:
                marked.add(int(row["#"]))
    bank = load_bank(path)
    have = {_fp(e["passage"], e["stem"]) for e in bank}
    added = []
    for i, q in enumerate(questions, 1):
        if i not in marked or _fp(q.get("passage", ""), q["stem"]) in have:
            continue
        entry = log.get(_fp(q.get("passage", ""), q["stem"]))
        qtype = (entry or {}).get("qtype") or q.get("type") or q.get("qtype") or ""
        base = re.sub(r"_(easy|medium|hard)$", "", qtype)
        level = next((d for d in DIFFICULTIES if qtype.endswith(d)), "medium")
        ex = {"id": next_id(bank), "skill": base, "level": level, "passage": q.get("passage", ""),
              "stem": q["stem"], "choices": q["choices"], "answer": q["answer"],
              "traps": _traps_for(q, entry), "status": "approved (Sebastian)",
              "source": f"{run.name} #{i}", "added": date.today().isoformat()}
        bank.append(ex)
        added.append(ex["id"])
    if added:
        save_bank(bank, path)
    return added


def candidates(run: Path) -> list[str]:
    """Questions whose wrong answers all scored 4+ with the critic: good ones to review and mark GOOD TRAPS."""
    questions, log = _final_entries(run)
    out = []
    for i, q in enumerate(questions, 1):
        entry = log.get(_fp(q.get("passage", ""), q["stem"])) or {}
        critic = entry.get("trap_critic") or {}
        scores = [v.get("tempting") for v in critic.values() if isinstance(v, dict)]
        if len(scores) == 3 and all(isinstance(s, int) and s >= 4 for s in scores) and not entry.get("flagged"):
            out.append(f"#{i} {entry.get('qtype', '')}: wrong answers rated {scores}")
    return out


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("learn", "candidates", "list", "approve", "remove") or (
            argv[0] != "list" and len(argv) < 2):
        print(__doc__)
        return 1
    if argv[0] == "list":
        bank = load_bank()
        for fam in TRAPS:
            items = [e for e in bank if family_of(e.get("skill", "")) == fam]
            print(f"{fam}: {len(items)} example(s)")
            for e in items:
                print(f"  {e['id']} {e['skill']} {e['level']} [{e.get('status', '')}] {e.get('source', '')}")
        return 0
    if argv[0] == "remove":
        bank = load_bank()
        keep = [e for e in bank if e.get("id") not in argv[1:]]
        save_bank(keep)
        print(f"removed {len(bank) - len(keep)} example(s)")
        return 0
    if argv[0] == "approve":
        bank = load_bank()
        hits = [e for e in bank if e.get("id") in argv[1:]]
        for e in hits:
            e["status"] = "approved (Sebastian)"
        save_bank(bank)
        print(f"approved {len(hits)} example(s)")
        return 0
    run = Path(argv[1])
    if argv[0] == "learn":
        added = learn(run)
        print(f"added {len(added)} example(s): {', '.join(added) or '-'}")
        if not added:
            print('Mark questions with "GOOD TRAPS" in the Review status column of questions.csv first.')
    else:
        found = candidates(run)
        print("\n".join(found) or "no candidates in this run")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
