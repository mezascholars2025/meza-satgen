"""Meza Math: SAT Math questions built in code, so every answer is computed, never guessed.

Algebra first (Sebastian, Nov 25). Each question family picks numbers that come out even, computes the answer, and
builds the wrong answer choices from real student mistakes (the main one: solving for x when the question asks for
something else, the deck's golden rule "know exactly what the question is asking"). Word problems get a story: the
writer model wraps the fixed numbers in 4-5 sentences about a named person; code checks every number, and a blind
solver must reach the computed answer, or the plain template story is used instead.

    python meza_math.py run --count 8                              # all algebra families, runs/math-<date_time>/
    python meza_math.py run --count 6 --families linear_rate --offline   # no API calls (template stories)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path

LEVELS = ("easy", "medium", "hard")
MINUS = "−"  # the SAT prints a true minus sign


# ----------------------------------------------------------------------------- formatting


def num(n) -> str:
    """11500 -> '11,500'; -3 -> '−3'; Fraction(7, 2) -> '7/2'."""
    n = Fraction(n)
    if n.denominator != 1:
        return (MINUS if n < 0 else "") + f"{abs(n.numerator)}/{n.denominator}"
    return (MINUS if n < 0 else "") + f"{abs(n.numerator):,}"


def term(coef: int, var: str = "x") -> str:
    if coef == 1:
        return var
    if coef == -1:
        return MINUS + var
    return f"{num(coef)}{var}"


def lin(a: int, b: int, var: str = "x") -> str:
    """lin(4, 5) -> '4x + 5'; lin(4, -5) -> '4x − 5'; lin(4, 0) -> '4x'."""
    out = term(a, var)
    if b > 0:
        out += f" + {num(b)}"
    elif b < 0:
        out += f" {MINUS} {num(-b)}"
    return out


def spr_answer(n) -> str:
    """Grid-in answer string (ASCII minus, fraction if needed)."""
    n = Fraction(n)
    return str(n.numerator) if n.denominator == 1 else f"{n.numerator}/{n.denominator}"


# ----------------------------------------------------------------------------- the item


@dataclass
class MathItem:
    family: str
    skill: str
    level: str
    fmt: str  # "mcq" or "spr"
    stem: str
    answer: Fraction
    steps: list[str]
    mistakes: list[tuple[Fraction, str]] = field(default_factory=list)  # (wrong value, the mistake behind it)
    story: dict | None = None  # facts for the story writer (word problems only)
    choices: dict[str, str] = field(default_factory=dict)
    key: str = ""

    def build_choices(self, rng: random.Random, used: dict[str, int] | None = None) -> None:
        """Four numeric choices in ascending order (as on the SAT): the answer plus three distinct real mistakes. The
        key's letter follows from where the mistakes fall; among the letters the real mistakes can produce, take the
        one used least so far in the batch (`used`), so keys spread over A-D without filler. A labeled near miss
        fills in only when a question has fewer than three real mistakes."""
        if self.fmt != "mcq":
            return
        whole = not (self.family == "expression_value" and self.level == "medium")  # there, x itself is the trap
        seen, real = {self.answer}, []
        for value, why in self.mistakes:
            if value not in seen and value > 0 and (value.denominator == 1 or not whole):
                seen.add(value)
                real.append((value, why))
        below = sorted([m for m in real if m[0] < self.answer], key=lambda m: m[0])
        above = sorted([m for m in real if m[0] > self.answer], key=lambda m: m[0])
        reachable = [k for k in range(4) if k <= len(below) and 3 - k <= len(above)]
        if reachable:
            used = used if used is not None else {}
            k = min(reachable, key=lambda k: (used.get("ABCD"[k], 0), rng.random()))
            # keep the most common mistakes (listed first) where there is a choice on a side
            pick_b = [m for m in real if m in below][:k]
            pick_a = [m for m in real if m in above][: 3 - k]
            wrong = pick_b + pick_a
        else:
            wrong = list(real[:3])
            step = max(1, abs(int(self.answer)) // 10)
            for v in (self.answer + o * step for n in range(1, 60) for o in (n, -n)):
                if len(wrong) >= 3:
                    break
                if v not in seen and v > 0:
                    seen.add(v)
                    wrong.append((v, "an arithmetic slip near the correct value"))
        values = sorted([(self.answer, "")] + wrong[:3], key=lambda v: v[0])
        self.choices = {k: num(v) for k, (v, _) in zip("ABCD", values, strict=True)}
        self.key = next(k for k, (v, _) in zip("ABCD", values, strict=True) if v == self.answer)
        self.mistakes = list(wrong[:3])
        if used is not None:
            used[self.key] = used.get(self.key, 0) + 1

    def explanation(self) -> str:
        lines = [f"[Domain: Algebra] [Skill: {self.skill}] [Difficulty: {self.level.title()}] [Subject: Math]"]
        lines += self.steps
        if self.fmt == "mcq":
            for k, text in self.choices.items():
                if k == self.key:
                    continue
                why = next(w for v, w in self.mistakes if num(v) == text)
                lines.append(f"Choice {k} ({text}): {why}.")
        return "\n".join(lines)

    def to_record(self) -> dict:
        return {
            "type": f"{self.family}_{self.level}",
            "format": self.fmt,
            "passage": "",
            "stem": self.stem,
            "choices": self.choices,
            "answer": self.key if self.fmt == "mcq" else spr_answer(self.answer),
            "explanation": self.explanation(),
            "family": self.family,
        }


# ----------------------------------------------------------------------------- family 1: value of an expression


def expression_value(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'If 4x + 5 = 29, what is the value of 9x − 7?'  The main trap is stopping at x."""
    skill = "Linear equations in one variable"
    if level == "easy":
        while True:
            x, a = rng.randint(2, 12), rng.randint(2, 9)
            b = rng.choice([n for n in range(-15, 21) if n])
            p, q = rng.randint(2, 9), rng.choice([n for n in range(-12, 13) if n])
            c, ans = a * x + b, p * x + q
            if p != a and c > 0 and ans > 0 and ans != x:
                break
        stem = f"If {lin(a, b)} = {num(c)}, what is the value of {lin(p, q)}?"
        slip = Fraction(c + b, a)
        mistakes = [
            (Fraction(x), f"this is the value of x; the question asks for {lin(p, q)}"),
            (
                Fraction(p * x),
                f"this is {term(p)} without the {num(q)}"
                if q > 0
                else f"this is {term(p)} before subtracting {num(-q)}",
            ),
            (
                p * slip + q,
                f"this comes from {'adding' if b > 0 else 'subtracting'} {num(abs(b))} instead of "
                f"{'subtracting' if b > 0 else 'adding'} it when solving for x",
            ),
            (
                Fraction(p * x - q),
                f"this {'subtracts' if q > 0 else 'adds'} {num(abs(q))} instead of "
                f"{'adding' if q > 0 else 'subtracting'} it",
            ),
            (Fraction(c), f"this is the value of {lin(a, b)}, given in the equation, not of {lin(p, q)}"),
            (p * Fraction(c, a) + q, f"this divides {num(c)} by {a} without first removing the {num(abs(b))}"),
            (Fraction(a * x + q), f"this uses {a}, the coefficient from the equation, instead of {p}"),
        ]
        steps = [
            f"Solve for x: {term(a)} = {num(c)} {MINUS + ' ' + num(b) if b > 0 else '+ ' + num(-b)} = {num(c - b)}, "
            f"so x = {num(x)}.",
            f"Substitute: {num(p)}({num(x)}) {'+ ' + num(q) if q > 0 else MINUS + ' ' + num(-q)} = {num(ans)}.",
        ]
        return MathItem("expression_value", skill, level, fmt, stem, Fraction(ans), steps, mistakes)

    if level == "medium":
        # x is an awkward fraction, so the fast route is to scale the whole equation: k(ax + b) + d.
        while True:
            a, b = rng.randint(2, 9), rng.choice([n for n in range(-12, 13) if n])
            c = rng.randint(5, 40)
            k = rng.choice([2, 3, 4, 5])
            d = rng.choice([0, 0] + [n for n in range(-9, 10) if n])
            x = Fraction(c - b, a)
            ans = k * c + d
            if x.denominator != 1 and ans > 0:
                break
        stem = f"If {lin(a, b)} = {num(c)}, what is the value of {lin(k * a, k * b + d)}?"
        mistakes = [
            (x, "this is the value of x; the question asks for the value of the whole expression"),
            (Fraction(c), f"this is the value of {lin(a, b)}, not of the new expression"),
            (Fraction(k * c), f"this multiplies {num(c)} by {k} but forgets that the constant changed too")
            if d
            else (Fraction(c + k), f"this adds {k} instead of multiplying by {k}"),
            (
                Fraction(k * c - d) if d else Fraction(k * c + k * b),
                f"this {'subtracts' if d > 0 else 'adds'} "
                f"{num(abs(d))} instead of {'adding' if d > 0 else 'subtracting'} it"
                if d
                else f"this multiplies {num(c)} by {k} and then adds {k} times {num(b)} again",
            ),
        ]
        rest = k * b + d - k * b
        steps = [
            f"Notice that {lin(k * a, k * b + d)} = {k}({lin(a, b)})"
            + (f" {'+ ' + num(rest) if rest > 0 else MINUS + ' ' + num(-rest)}" if rest else "")
            + ".",
            f"So its value is {k}({num(c)})"
            + (f" {'+ ' + num(d) if d > 0 else MINUS + ' ' + num(-d)}" if d else "")
            + f" = {num(ans)}. (Solving for x first also works, but x = {num(x)} is messier.)",
        ]
        return MathItem(
            "expression_value", "Linear equations in one variable", level, fmt, stem, Fraction(ans), steps, mistakes
        )

    # hard: parentheses and x on both sides
    while True:
        a, b = rng.randint(2, 6), rng.choice([n for n in range(-9, 10) if n])
        cc = rng.choice([n for n in range(1, 8) if n != a])
        x = rng.randint(-6, 12)
        d = a * (x + b) - cc * x
        p, q = rng.randint(2, 9), rng.choice([n for n in range(-15, 16) if n])
        ans = p * x + q
        wrong_dist = Fraction(d - b, a - cc)  # took a(x + b) as ax + b
        if x and ans > 0 and ans != x and p != a and d:
            break
    left = f"{a}(x {'+ ' + num(b) if b > 0 else MINUS + ' ' + num(-b)})"
    stem = f"If {left} = {lin(cc, d)}, what is the value of {lin(p, q)}?"
    mistakes = [
        (Fraction(x), f"this is the value of x; the question asks for {lin(p, q)}"),
        (
            p * wrong_dist + q,
            f"this comes from multiplying only x by {a} (writing {lin(a, b)} instead of {lin(a, a * b)})",
        ),
        (Fraction(p * x), f"this is {term(p)} without the constant term"),
        (
            p * Fraction(d + a * b, a - cc) + q,
            f"this comes from moving {num(a * b)} to the other side without changing its sign",
        ),
        (
            Fraction(p * x - q),
            f"this {'subtracts' if q > 0 else 'adds'} {num(abs(q))} instead of "
            f"{'adding' if q > 0 else 'subtracting'} it",
        ),
        (p * Fraction(d - a * b, a + cc) + q, f"this adds {term(cc)} to both sides instead of subtracting it"),
    ]
    collect = f"{term(a - cc)} = {num(d - a * b)}" + (f", so x = {num(x)}" if a - cc != 1 else "")
    steps = [
        f"Distribute: {lin(a, a * b)} = {lin(cc, d)}.",
        f"Collect the x-terms on one side: {collect}.",
        f"Substitute: {num(p)}({num(x)}) {'+ ' + num(q) if q > 0 else MINUS + ' ' + num(-q)} = {num(ans)}.",
    ]
    return MathItem("expression_value", skill, level, fmt, stem, Fraction(ans), steps, mistakes)


# ----------------------------------------------------------------------------- family 2: constant-rate word problem

FRAMES = [
    dict(
        role="a truck driver loading bags of rice for today's delivery",
        item="bags of rice",
        base="load",
        s="loads",
        ing="loading",
        past="loaded",
    ),
    dict(
        role="a volunteer packing food boxes for a weekend food drive",
        item="food boxes",
        base="pack",
        s="packs",
        ing="packing",
        past="packed",
    ),
    dict(
        role="the owner of a print shop printing flyers for a city festival",
        item="flyers",
        base="print",
        s="prints",
        ing="printing",
        past="printed",
    ),
    dict(
        role="a warehouse worker labeling packages for shipment",
        item="packages",
        base="label",
        s="labels",
        ing="labeling",
        past="labeled",
    ),
    dict(
        role="a baker boxing cookies for a large catering order",
        item="cookies",
        base="box",
        s="boxes",
        ing="boxing",
        past="boxed",
    ),
    dict(
        role="a farmworker picking strawberries for a market order",
        item="strawberries",
        base="pick",
        s="picks",
        ing="picking",
        past="picked",
    ),
]
NAMES = [
    "Marcus",
    "Sofia",
    "Diego",
    "Aaliyah",
    "Mateo",
    "Priya",
    "Jamal",
    "Camila",
    "Ethan",
    "Valeria",
    "Kenji",
    "Isabela",
    "Andre",
    "Lucia",
    "Omar",
    "Gabriela",
]
_FRAME_DECK: list[dict] = []


def _next_frame(rng: random.Random) -> dict:
    """Deal settings like cards, so a batch uses every setting before repeating one."""
    if not _FRAME_DECK:
        _FRAME_DECK.extend(rng.sample(FRAMES, len(FRAMES)))
    return _FRAME_DECK.pop()


RATES = [125, 150, 200, 250, 300, 375, 400, 450, 500, 600, 625, 750, 800]


def linear_rate(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A job done at a constant rate: S items at the start, r per hour, T hours in all."""
    skill = "Linear equations in one variable"
    frame, name = _next_frame(rng), rng.choice(NAMES)
    f = dict(frame, name=name)
    while True:
        r, T = rng.choice(RATES), rng.randint(8, 20)
        t1 = rng.randint(2, max(2, T // 2 - 1))
        t2 = rng.randint(t1 + 2, T - 1)
        S = r * T
        R1, R2 = S - r * t1, S - r * t2
        if S <= 15000 and t2 - t1 >= 2 and T - t2 >= 2 and r not in (R1, R2):
            break
    it = f["item"]
    facts = {
        "name": name,
        "role": f["role"],
        "item": it,
        "verb": f["base"],
        "total": S,
        "rate": r,
        "t1": t1,
        "left1": R1,
        "t2": t2,
        "left2": R2,
        "finish": T,
    }
    if level == "easy":
        answer = Fraction(t2)
        stem = (
            f"{name} is {f['role']}. {name} has to {f['base']} {num(S)} {it} in all and {f['s']} {num(r)} {it} "
            f"every hour. How many hours after starting will {name} have exactly {num(R2)} {it} left to "
            f"{f['base']}?"
        )
        ask = f"how many hours after starting {name} will have exactly {num(R2)} {it} left (answer: {t2})"
        given = [S, r, R2]
        mistakes = [
            (
                Fraction(R2, r),
                f"this is how long it takes to {f['base']} the last {num(R2)} {it}, not how "
                f"long until only {num(R2)} are left",
            ),
            (Fraction(T), f"this is the time to finish all {num(S)} {it}"),
            (Fraction(S + R2, r), f"this adds the {num(R2)} {it} instead of subtracting them"),
        ]
        steps = [
            f"{name} must {f['base']} {num(S)} {MINUS} {num(R2)} = {num(S - R2)} {it} before {num(R2)} are left.",
            f"At {num(r)} per hour, that takes {num(S - R2)} ÷ {num(r)} = {t2} hours.",
        ]
    elif level == "medium":
        answer = Fraction(t2)
        stem = (
            f"{name} is {f['role']}. At the start of the day, {name} had {num(S)} {it} to {f['base']}. "
            f"{name} {f['s']} the same number of {it} every hour. After {t1} hours, {num(R1)} {it} were still "
            f"left to {f['base']}. At this rate, how many hours after starting will {name} have exactly "
            f"{num(R2)} {it} left to {f['base']}?"
        )
        ask = (
            f"how many hours after starting (total, counted from the start) {name} will have exactly {num(R2)} "
            f"{it} left (answer: {t2})"
        )
        given = [S, t1, R1, R2]
        wrong_rate = Fraction(R1, t1)  # divided the bags still LEFT by the hours, instead of the bags done
        mistakes = [
            (
                Fraction(t2 - t1),
                f"this is how many MORE hours after the first {t1}; the question asks for "
                f"the total time since {name} started",
            ),
            (Fraction(R2, r), f"this is the time to {f['base']} the last {num(R2)} {it}"),
            (
                (S - R2) / wrong_rate,
                f"this uses {num(R1)} ÷ {t1} as the hourly rate (the {it} left, not the {it} done)",
            ),
            (Fraction(t1 + t2), f"this adds the first {t1} hours to the total a second time"),
            (Fraction(t1), f"this is the {t1} hours already worked"),
            (Fraction(R1 - R2, r) + t2, f"this adds the {t2 - t1} more hours to {t2} instead of to {t1}"),
            (Fraction(T), f"this is the time to {f['base']} all {num(S)} {it}"),
        ]
        steps = [
            f"In {t1} hours, {name} {f['past']} {num(S)} {MINUS} {num(R1)} = {num(S - R1)} {it}, so the rate is "
            f"{num(S - R1)} ÷ {t1} = {num(r)} {it} per hour.",
            f"To have {num(R2)} left, {name} must {f['base']} {num(S)} {MINUS} {num(R2)} = {num(S - R2)} {it}.",
            f"That takes {num(S - R2)} ÷ {num(r)} = {t2} hours after starting.",
        ]
    else:  # hard: two observations, starting amount not given; when will the job be finished?
        answer = Fraction(T)
        stem = (
            f"{name} is {f['role']} and {f['s']} the same number of {it} every hour. After {t1} hours of work, "
            f"{num(R1)} {it} were still left to {f['base']}. After {t2} hours of work, {num(R2)} {it} were still "
            f"left. If {name} keeps working at this rate, how many hours after starting will {name} finish?"
        )
        ask = f"how many hours after starting {name} will finish all of the {it} (answer: {T})"
        given = [t1, R1, t2, R2]
        mistakes = [
            (
                Fraction(R2, r),
                f"this is how many more hours are needed after the first {t2}; the question "
                f"asks for the total time since {name} started",
            ),
            (
                Fraction(r),
                f"this is the hourly rate ({num(R1)} {MINUS} {num(R2)}) ÷ ({t2} {MINUS} {t1}), not the number of hours",
            ),
            (Fraction(R1, r), f"this is the time left after the first {t1} hours"),
            (t2 + Fraction(R1, r), f"this adds the time left after hour {t1} to {t2} hours"),
            (Fraction(t2 - t1), f"this is the {t2 - t1} hours between the two readings"),
            (Fraction(R1 + R2, r), "this adds the two amounts left instead of using their difference"),
        ]
        steps = [
            f"Between hour {t1} and hour {t2}, {name} {f['past']} {num(R1)} {MINUS} {num(R2)} = {num(R1 - R2)} "
            f"{it} in {t2 - t1} hours, so the rate is {num(r)} {it} per hour.",
            f"After {t2} hours, {num(R2)} are left, which takes {num(R2)} ÷ {num(r)} = {num(Fraction(R2, r))} "
            f"more hour{'s' if R2 != r else ''}.",
            f"Total: {t2} + {num(Fraction(R2, r))} = {T} hours after starting.",
        ]
    facts.update(level=level, ask=ask, given=given)
    return MathItem("linear_rate", skill, level, fmt, stem, answer, steps, mistakes, story=facts)


FAMILIES = {"expression_value": expression_value, "linear_rate": linear_rate}


# ----------------------------------------------------------------------------- stories (word problems)

STORY_PROMPT = """Write the word problem for a Digital SAT Math question (Algebra, {level}).

Situation: {name} is {role}. The work happens at a constant rate, measured in {item}.
Numbers to use, each exactly once and written exactly like this: {numbers}.
Do not add any other numbers (no dates, times of day, or extra quantities).
What the question must ask: {ask}.

Write 4 or 5 short sentences in plain, natural English: introduce {name} and the job (one concrete, natural detail
about why the work is being done is welcome), give the numbers in the order listed above as the situation unfolds,
and end with the question as the last sentence. Each number keeps the meaning it has in the plan below; do not
restate a number as a different quantity or add facts that relate the numbers to each other. Say once that the rate
is constant; do not pad with sentences like "she does not take breaks." Keep the wording of the question exact about
what is being counted (total hours since starting versus more hours). Do not explain how to solve it.

The plan, as a plain template: {template}
Reply with JSON only: {{"stem": "<the full word problem>"}}"""
SOLVE_PROMPT = 'Solve this SAT math question. Reply with JSON only: {{"answer": "<number>"}}\n\n{stem}'


def story_numbers_ok(stem: str, given: list[int]) -> list[str]:
    """Every planned number exactly once, in the planned order (so each keeps its role), and nothing else."""
    found = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", stem)]
    problems = []
    if sorted(found) != sorted(given):
        problems.append(f"numbers in the story {found} must be exactly {given}")
    elif found != list(given):
        problems.append(
            f"the numbers appear in the order {found}; keep the planned order {list(given)} so each "
            f"number keeps its role"
        )
    for n in given:
        if n >= 1000 and num(n) not in stem:
            problems.append(f"write {n} as {num(n)}")
    if not stem.strip().endswith("?"):
        problems.append("the story must end with the question")
    return problems


class Story:
    """Writes the story with the writer model; falls back to the template stem if a check fails."""

    def __init__(self, writer=None, solver=None, tries: int = 2):
        self.writer, self.solver, self.tries = writer, solver, tries
        self.log: list[dict] = []

    def tell(self, item: MathItem) -> None:
        if not item.story or not self.writer:
            return
        f = item.story
        numbers = ", ".join(num(n) for n in f["given"])
        for attempt in range(self.tries):
            text = self.writer(STORY_PROMPT.format(numbers=numbers, template=item.stem, **f))
            try:
                stem = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["stem"].strip()
            except (AttributeError, ValueError, KeyError):
                continue
            problems = story_numbers_ok(stem, f["given"])
            votes = []
            if not problems and self.solver:
                for _ in range(3):
                    reply = self.solver(SOLVE_PROMPT.format(stem=stem))
                    m = re.search(r'"answer"\s*:\s*"?\s*(-?[\d,./]+)', reply)
                    try:
                        votes.append(Fraction(m.group(1).replace(",", "")) if m else None)
                    except (ValueError, ZeroDivisionError):
                        votes.append(None)
                if sum(v == item.answer for v in votes) < 2:
                    problems.append(f"a blind solver got {votes}, not {item.answer}")
            self.log.append(
                {"family": item.family, "level": item.level, "attempt": attempt, "stem": stem, "problems": problems}
            )
            if not problems:
                item.stem = stem
                return


def claude_caller(model: str, temperature: float = 1.0):
    """A prompt -> text function using satgen's client (it handles models that fix their own sampling)."""
    from satgen.llm import call_with_retry, make_client

    client = make_client(model)

    def call(prompt: str) -> str:
        return call_with_retry(client, prompt, temperature=temperature, json_mode=True)[0]

    return call


# ----------------------------------------------------------------------------- batches


def plan(count: int, families: list[str], rng: random.Random) -> list[tuple[str, str, str]]:
    """Spread families and levels evenly; about 3 in 4 multiple choice (the rest grid-in), as on the test."""
    slots = []
    for i in range(count):
        fam = families[i % len(families)]
        level = LEVELS[(i // len(families)) % 3]
        fmt = "spr" if rng.random() < 0.25 else "mcq"
        slots.append((fam, level, fmt))
    return slots


def generate(count: int, families: list[str], seed: int, story: Story | None = None) -> list[MathItem]:
    rng = random.Random(seed)
    _FRAME_DECK.clear()
    items = []
    used: dict[str, int] = {}
    for fam, level, fmt in plan(count, families, rng):
        item = FAMILIES[fam](rng, level, fmt)
        item.build_choices(rng, used)
        if story:
            story.tell(item)
        items.append(item)
    return items


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--count", type=int, default=8)
    run.add_argument("--families", default=",".join(FAMILIES))
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--offline", action="store_true", help="no API calls: template stories")
    run.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    out = args.out or Path("runs") / f"math-{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    story = None
    if not args.offline:
        story = Story(
            claude_caller(os.environ.get("MEZA_MATH_WRITER", "claude-sonnet-5-5")),
            claude_caller(os.environ.get("MEZA_MATH_SOLVER", "claude-haiku-5-5"), temperature=0.0),
        )
    items = generate(args.count, args.families.split(","), seed, story)
    (out / "questions.json").write_text(
        json.dumps([i.to_record() for i in items], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if story:
        (out / "stories.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in story.log), encoding="utf-8"
        )
    print(f"wrote {len(items)} questions to {out} (seed {seed})")
    import meza_export

    meza_export.build(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
