"""Meza Math: SAT Math questions built in code, so every answer is computed, never guessed.

Algebra first (Sebastian, Nov 25). Each question family picks numbers that come out even, computes the answer, and
builds the wrong answer choices from real student mistakes (the main one: solving for x when the question asks for
something else, the deck's golden rule "know exactly what the question is asking"). Word problems get a story: the
writer model wraps the fixed numbers in 4-5 sentences about a named person; code checks every number, and a blind
solver must reach the computed answer, or the plain template story is used instead.

    python meza_math.py run --count 8                          # free: no API calls, template stories
    python meza_math.py run --count 8 --ai-stories             # the writer model words the word problems (uses credits)
    python meza_math.py run --count 6 --families linear_rate   # one family only
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
    passage: str = ""  # the equation, shown on its own line above the question (as on the SAT)
    positive_only: bool = True  # numeric choices: keep wrong answers positive (word problems, counts)
    key_text: str = ""  # for choices that are equations or phrases instead of numbers

    def build_choices(self, rng: random.Random, used: dict[str, int] | None = None) -> None:
        """Four numeric choices in ascending order (as on the SAT): the answer plus three distinct real mistakes. The
        key's letter follows from where the mistakes fall; among the letters the real mistakes can produce, take the
        one used least so far in the batch (`used`), so keys spread over A-D without filler. A labeled near miss
        fills in only when a question has fewer than three real mistakes."""
        if self.fmt != "mcq":
            return
        if self.key_text:
            self._build_text_choices(rng, used)
            return
        whole = not (self.family == "expression_value" and self.level == "medium")  # there, x itself is the trap
        seen, real = {self.answer}, []
        for value, why in self.mistakes:
            ok_sign = value > 0 if self.positive_only else value != 0 or self.answer == 0
            if value not in seen and ok_sign and (value.denominator == 1 or not whole):
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
                if v not in seen and (v > 0 or not self.positive_only):
                    seen.add(v)
                    wrong.append((v, "an arithmetic slip near the correct value"))
        values = sorted([(self.answer, "")] + wrong[:3], key=lambda v: v[0])
        self.choices = {k: num(v) for k, (v, _) in zip("ABCD", values, strict=True)}
        self.key = next(k for k, (v, _) in zip("ABCD", values, strict=True) if v == self.answer)
        self.mistakes = list(wrong[:3])
        if used is not None:
            used[self.key] = used.get(self.key, 0) + 1

    def _build_text_choices(self, rng: random.Random, used: dict[str, int] | None) -> None:
        """Equations or phrases: the key goes to the letter used least so far; the three most common mistakes fill
        the other letters in random order."""
        wrong, seen = [], {self.key_text}
        for text, why in self.mistakes:
            if text not in seen:
                seen.add(text)
                wrong.append((text, why))
        assert len(wrong) >= 3, f"{self.family}: needs three distinct wrong choices"
        wrong = rng.sample(wrong, 3)
        used = used if used is not None else {}
        key = min("ABCD", key=lambda k: (used.get(k, 0), rng.random()))
        rng.shuffle(wrong)
        others = iter(wrong)
        self.choices = {k: self.key_text if k == key else next(others)[0] for k in "ABCD"}
        self.key = key
        self.mistakes = wrong
        used[key] = used.get(key, 0) + 1

    def explanation(self) -> str:
        lines = [f"[Domain: Algebra] [Skill: {self.skill}] [Difficulty: {self.level.title()}] [Subject: Math]"]
        lines += self.steps
        if self.fmt == "mcq":
            for k, text in self.choices.items():
                if k == self.key:
                    continue
                why = next(w for v, w in self.mistakes if (v if self.key_text else num(v)) == text)
                lines.append(f"Choice {k} ({text}): {why}.")
        return "\n".join(lines)

    def to_record(self) -> dict:
        return {
            "type": f"{self.family}_{self.level}",
            "format": self.fmt,
            "passage": self.passage,
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


# ------------------------------------------------------------------- more linear equations (Sebastian, Nov 25)

SKILL = "Linear equations in one variable"
UNKNOWNS = ["x", "r", "n", "p", "t", "w", "k"]


def plus(n: int) -> str:
    """' + 5' or ' − 5' (for building equations)."""
    return f" + {num(n)}" if n >= 0 else f" {MINUS} {num(-n)}"


def solve_simple(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'r + 23 = 61. What is the value of r?'  Medium: the unknown is subtracted ('45 − r = 17')."""
    v = rng.choice(UNKNOWNS)
    ask = rng.choice([f"What is the value of {v}?", f"What value of {v} is the solution to the given equation?"])
    if level == "easy":
        while True:
            c, ans = rng.choice([n for n in range(-60, 61) if abs(n) >= 3]), rng.randint(-20, 90)
            d = ans + c
            if ans and d and abs(d) < 150 and len({ans, d, d + c, c - d}) == 4:
                break
        passage = f"{v}{plus(c)} = {num(d)}"
        mistakes = [
            (
                Fraction(d + c),
                f"this {'adds' if c > 0 else 'subtracts'} {num(abs(c))} instead of "
                f"{'subtracting' if c > 0 else 'adding'} it",
            ),
            (Fraction(c - d), "this subtracts in the wrong order"),
            (Fraction(d), "this is the right side of the equation, not the value of " + v),
        ]
        steps = [
            f"{'Subtract' if c > 0 else 'Add'} {num(abs(c))} {'from' if c > 0 else 'to'} both sides: "
            f"{v} = {num(d)} {MINUS + ' ' + num(c) if c > 0 else '+ ' + num(-c)} = {num(ans)}."
        ]
    else:
        while True:
            c, d = rng.randint(10, 90), rng.choice([n for n in range(-40, 80) if n])
            ans = c - d
            if ans and len({ans, d - c, c + d, d}) == 4:
                break
        passage = f"{num(c)} {MINUS} {v} = {num(d)}"
        mistakes = [
            (Fraction(d - c), f"this stops at {MINUS}{v} = {num(d - c)} and forgets to divide by {MINUS}1"),
            (Fraction(c + d), f"this adds {num(d)} to {num(c)} instead of subtracting"),
            (Fraction(d), "this is the right side of the equation, not the value of " + v),
        ]
        steps = [
            f"Subtract {num(c)} from both sides: {MINUS}{v} = {num(d - c)}.",
            f"Divide both sides by {MINUS}1: {v} = {num(ans)}.",
        ]
    return MathItem(
        "solve_simple", SKILL, level, fmt, ask, Fraction(ans), steps, mistakes, passage=passage, positive_only=False
    )


def solve_combine(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'7x − 3x + 5 = 29. What value of x is the solution to the given equation?'  Combine the x-terms first.
    Medium: the second x-term is bigger, so the combined coefficient is negative."""
    v = rng.choice(UNKNOWNS)
    while True:
        a, b = rng.randint(2, 14), rng.randint(1, 13)
        if level == "easy" and a - b < 2 or level != "easy" and b - a < 2:
            continue
        x = rng.randint(1, 15) if level == "easy" else rng.choice([n for n in range(-12, 13) if n])
        g = abs(a - b)  # a constant that is a multiple of the combined coefficient keeps the mistakes whole numbers
        c = g * rng.choice([n for n in range(-30 // g - 1, 30 // g + 2) if n])
        d = (a - b) * x + c
        if d and abs(c) >= 2:
            break
    passage = f"{term(a, v)} {MINUS} {term(b, v)}{plus(c)} = {num(d)}"
    ask = rng.choice([f"What value of {v} is the solution to the given equation?", f"What is the value of {v}?"])
    mistakes = [
        (Fraction(d - c, a + b), f"this adds {term(a, v)} and {term(b, v)} instead of subtracting"),
        (
            Fraction(d + c, a - b),
            f"this {'adds' if c > 0 else 'subtracts'} {num(abs(c))} instead of "
            f"{'subtracting' if c > 0 else 'adding'} it",
        ),
        (Fraction(d - c), f"this stops before dividing by {num(a - b)}"),
        (Fraction(d - c, a), f"this leaves out the {term(b, v)} term"),
        (
            Fraction(d, a - b),
            f"this divides {num(d)} by {num(a - b)} without first {'subtracting' if c > 0 else 'adding'} {num(abs(c))}",
        ),
        (Fraction(d - c, b - a), f"this combines the {v}-terms as {term(b - a, v)} (the wrong sign)"),
    ]
    steps = [
        f"Combine the {v}-terms: {term(a - b, v)}{plus(c)} = {num(d)}.",
        f"{'Subtract' if c > 0 else 'Add'} {num(abs(c))}: {term(a - b, v)} = {num(d - c)}.",
        f"Divide by {num(a - b)}: {v} = {num(x)}.",
    ]
    return MathItem(
        "solve_combine", SKILL, level, fmt, ask, Fraction(x), steps, mistakes, passage=passage, positive_only=False
    )


def one_solution_cannot(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'x(r + 3) + 5 = 7x − 2 ... If the equation has exactly one solution, which of the following CANNOT be the value
    of r?'  Exactly one solution means the x-coefficients differ, so r cannot make them equal.
    Hard adds another x-term on the left."""
    while True:
        a = rng.choice([n for n in range(-9, 10) if n])
        cc = rng.randint(2, 12)
        e = rng.choice([n for n in range(-6, 7) if n]) if level == "hard" else 0
        b, d = rng.choice([n for n in range(-15, 16) if n]), rng.choice([n for n in range(-15, 16) if n])
        bad = cc - a - e  # the value of r that makes the x-coefficients equal
        if bad and b != d:
            break
    left = f"x(r{plus(a)})" + (f" + {term(e)}" if e > 0 else f" {MINUS} {term(-e)}" if e else "") + plus(b)
    passage = f"{left} = {lin(cc, d)}"
    stem = (
        "In the given equation, r is an integer constant. If the equation has exactly one solution, which of the "
        "following CANNOT be the value of r?"
    )
    mistakes = [
        (
            Fraction(cc),
            f"this sets r equal to {cc}, ignoring the {num(a)} inside the parentheses"
            + (f" and the {term(e)} term" if e else ""),
        ),
        (Fraction(cc + a - e), f"this moves the {num(a)} to the other side without changing its sign"),
        (Fraction(-bad), "this has the right size but the wrong sign"),
    ]
    if e:
        mistakes.insert(1, (Fraction(cc - a), f"this leaves out the {term(e)} term on the left"))
    coef = f"r{plus(a)}" + (plus(e) if e else "")
    steps = [
        f"Collect the x-terms on the left: the coefficient of x is {coef}"
        + (f" = r{plus(a + e)}" if e else "")
        + f"; on the right it is {cc}.",
        "The equation has exactly one solution unless the two coefficients are equal.",
        f"They are equal when r{plus(a + e)} = {cc}, that is, r = {num(bad)}. Then the x-terms cancel and the "
        f"equation has no solution, so r cannot be {num(bad)}.",
    ]
    return MathItem(
        "one_solution_cannot",
        SKILL,
        level,
        "mcq",
        stem,
        Fraction(bad),
        steps,
        mistakes,
        passage=passage,
        positive_only=False,
    )


def infinite_constant(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'3(2x + 5) − x = kx + 15 ... The equation has infinitely many solutions. What is the value of k?'
    Medium: k is a coefficient. Hard: k is inside parentheses on the right, '5(x + k)'."""
    k = rng.choice(["k", "n", "m", "c"])
    while True:
        a, p_, q = rng.randint(2, 6), rng.randint(1, 5), rng.choice([n for n in range(-9, 10) if n])
        e = rng.choice([n for n in range(-5, 6) if n])
        A, B = a * p_ - e, a * q  # left side = Ax + B
        if level != "hard" and A > 0 and A != a * p_:
            break
        if level == "hard" and A > 1 and B % A == 0 and B // A not in (0, q, B):
            break
    inner = lin(p_, q)
    left = f"{a}({inner})" + (f" {MINUS} {term(e)}" if e > 0 else f" + {term(-e)}")
    stem = (
        f"In the given equation, {k} is a constant. The equation has infinitely many solutions. What is the value "
        f"of {k}?"
    )
    if level != "hard":
        passage = f"{left} = {k}x{plus(B)}"
        ans = Fraction(A)
        mistakes = [
            (Fraction(a * p_), f"this leaves out the {MINUS if e > 0 else '+'}{term(abs(e))} term"),
            (
                Fraction(a * p_ + e),
                f"this {'adds' if e > 0 else 'subtracts'} the {term(abs(e))} term "
                f"instead of {'subtracting' if e > 0 else 'adding'} it",
            ),
            (Fraction(p_ - e), f"this forgets to multiply {term(p_)} by {a}"),
            (Fraction(-A), "this has the right size but the wrong sign"),
        ]
        steps = [
            f"Distribute and combine on the left: {lin(a * p_, B)} {MINUS if e > 0 else '+'} {term(abs(e))} = "
            f"{lin(A, B)}.",
            f"Infinitely many solutions means both sides are the same expression: {lin(A, B)} = {k}x{plus(B)}.",
            f"So {k} = {A}.",
        ]
    else:
        passage = f"{left} = {A}(x + {k})"
        ans = Fraction(B, A)
        mistakes = [
            (Fraction(B), f"this matches {num(B)} with {k} and forgets that {k} is multiplied by {A}"),
            (Fraction(q), f"this matches the constants before distributing the {a}"),
            (Fraction(-B, A), "this has the right size but the wrong sign"),
            (Fraction(B, a * p_), f"this divides by {a * p_}, leaving out the {term(abs(e))} term"),
            (Fraction(B - A), f"this subtracts {A} instead of dividing by {A}"),
        ]
        steps = [
            f"Distribute and combine on the left: {lin(A, B)}.",
            f"Distribute on the right: {lin(A, 0)} + {A}{k}.",
            f"Infinitely many solutions means the constants match too: {A}{k} = {num(B)}, so {k} = {num(ans)}.",
        ]
    return MathItem(
        "infinite_constant", SKILL, level, fmt, stem, ans, steps, mistakes, passage=passage, positive_only=False
    )


THREE_KINDS = [
    dict(who="A bakery", short="the bakery", verb="baked", item="muffins", kinds=("blueberry", "banana", "corn")),
    dict(who="A furniture workshop", short="the workshop", verb="built", item="chairs", kinds=("oak", "pine", "maple")),
    dict(who="A florist", short="the florist", verb="arranged", item="bouquets", kinds=("rose", "tulip", "lily")),
    dict(who="A school club", short="the club", verb="sold", item="T-shirts", kinds=("small", "medium", "large")),
    dict(who="A food truck", short="the food truck", verb="made", item="tacos", kinds=("chicken", "beef", "vegetable")),
    dict(who="A bike shop", short="the shop", verb="repaired", item="bikes", kinds=("road", "mountain", "kids'")),
]


def three_kinds_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Three kinds of one thing: the first is m times the second, the third is a known number, and the total is known.
    'Which equation represents this situation?'  Medium: x is the FIRST kind, so the second is x/m."""
    f = rng.choice(THREE_KINDS)
    m, s, k3 = rng.randint(2, 6), rng.randint(8, 60), rng.randint(12, 150)
    T = m * s + s + k3
    k1, k2, k3n = f["kinds"]
    it = f["item"]
    setup = (
        f"{f['who']} {f['verb']} three kinds of {it}: {k1}, {k2}, and {k3n}. The number of {k1} {it} was {m} "
        f"times the number of {k2} {it}, and {k3} of the {it} were {k3n}. In all, {f['short']} {f['verb']} {T} "
        f"{it}."
    )
    if level == "easy":
        question = f"If x represents the number of {k2} {it}, which equation represents this situation?"
        key = f"{m}x + x + {k3} = {T}"
        mistakes = [
            (f"x + {m} + {k3} = {T}", f"this adds {m} instead of multiplying by {m}"),
            (f"{m}x + {k3} = {T}", f"this leaves out the {k2} {it}"),
            (f"x/{m} + x + {k3} = {T}", f"this makes the {k2} {it} {m} times the {k1} {it}, the reverse"),
            (f"{m}x + x = {T} + {k3}", f"this adds the {k3} {k3n} {it} to the total instead of subtracting"),
        ]
        steps = [
            f"The number of {k2} {it} is x, so the number of {k1} {it} is {m}x.",
            f"Add all three kinds and set the sum equal to the total: {m}x + x + {k3} = {T}.",
        ]
    else:
        question = f"If x represents the number of {k1} {it}, which equation represents this situation?"
        key = f"x + x/{m} + {k3} = {T}"
        mistakes = [
            (f"x + {m}x + {k3} = {T}", f"this makes the {k2} {it} {m} times the {k1} {it}, the reverse"),
            (f"x/{m} + {k3} = {T}", f"this leaves out the {k1} {it}"),
            (f"x + x/{m} = {T} + {k3}", f"this adds the {k3} {k3n} {it} to the total instead of subtracting"),
            (f"x + x {MINUS} {m} + {k3} = {T}", f"this subtracts {m} instead of dividing by {m}"),
        ]
        steps = [
            f"The number of {k1} {it} is x. It is {m} times the number of {k2} {it}, so there are x/{m} {k2} {it}.",
            f"Add all three kinds and set the sum equal to the total: x + x/{m} + {k3} = {T}.",
        ]
    item = MathItem(
        "three_kinds_equation", SKILL, level, "mcq", f"{setup} {question}", Fraction(0), steps, mistakes, key_text=key
    )
    item.story = dict(
        kind="setup",
        level=level,
        setup=setup,
        question=question,
        given=[m, k3, T],
        truth=s if level == "easy" else m * s,
        situation=f"{f['who'].lower()} {f['verb']} three kinds of {it} ({k1}, {k2}, {k3n})",
    )
    return item


PLANS = [
    dict(
        name="laptop",
        var="p",
        each_var="a",
        setup="{n} bought a laptop on a payment plan. {n} paid a down payment of ${D} and then made monthly payments "
        "of ${M} each. In all, {n} paid ${T} for the laptop.",
        setup_n="{n} bought a laptop on a payment plan. {n} paid a down payment of ${D} and then made {k} equal "
        "monthly payments. In all, {n} paid ${T} for the laptop.",
        var_def="p is the number of monthly payments {n} made",
        each_def="a is the amount, in dollars, of each monthly payment",
        start="The down payment, in dollars",
        each="The amount, in dollars, of each monthly payment",
        product="The total amount, in dollars, of all of {n}'s monthly payments",
        total="The total amount, in dollars, {n} paid for the laptop",
        eq="the total amount, in dollars, {n} paid for a laptop",
    ),
    dict(
        name="gym",
        var="m",
        each_var="f",
        setup="{n} joined a gym that charges a one-time sign-up fee of ${D} plus a membership fee of ${M} each "
        "month. {n} paid a total of ${T} in fees.",
        setup_n="{n} joined a gym that charges a one-time sign-up fee of ${D} plus the same membership fee each "
        "month. After {k} months, {n} had paid a total of ${T} in fees.",
        var_def="m is the number of months {n} paid for",
        each_def="f is the monthly membership fee, in dollars",
        start="The sign-up fee, in dollars",
        each="The membership fee, in dollars, for one month",
        product="The total amount, in dollars, of the monthly membership fees {n} paid",
        total="The total amount, in dollars, {n} paid in fees",
        eq="the total amount, in dollars, {n} paid in gym fees",
    ),
    dict(
        name="savings",
        var="w",
        each_var="d",
        setup="{n} opened a savings account with a deposit of ${D}. After that, {n} deposited ${M} into the account "
        "every week and made no withdrawals. The account then held ${T}.",
        setup_n="{n} opened a savings account with a deposit of ${D}. After that, {n} deposited the same amount every "
        "week for {k} weeks and made no withdrawals. The account then held ${T}.",
        var_def="w is the number of weekly deposits {n} made",
        each_def="d is the amount, in dollars, of each weekly deposit",
        start="The first deposit, in dollars",
        each="The amount, in dollars, of each weekly deposit",
        product="The total amount, in dollars, of {n}'s weekly deposits",
        total="The amount, in dollars, in the account after the weekly deposits",
        eq="the amount, in dollars, in {n}'s savings account",
    ),
]


def _plan_numbers(rng: random.Random) -> tuple[dict, str, int, int, int, int]:
    f, n = rng.choice(PLANS), rng.choice(NAMES)
    D = rng.choice(range(50, 1001, 25))
    M = rng.choice(range(15, 301, 5))
    k = rng.randint(4, 24)
    return f, n, D, M, k, D + M * k


def payment_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A start amount plus a repeated payment equals a total. 'Which equation represents this situation?'
    Easy: the number of payments is unknown. Medium: the amount of each payment is unknown."""
    f, n, D, M, k, T = _plan_numbers(rng)
    Ds, Ms, Ts = num(D), num(M), num(T)
    if level == "easy":
        v = f["var"]
        setup = f["setup"].format(n=n, D=Ds, M=Ms, T=Ts)
        question = f"Which equation represents this situation, where {f['var_def'].format(n=n)}?"
        key = f"{Ds} + {Ms}{v} = {Ts}"
        mistakes = [
            (f"{Ds}{v} + {Ms} = {Ts}", "this swaps the one-time amount and the repeated amount"),
            (f"({Ds} + {Ms}){v} = {Ts}", f"this multiplies the one-time ${Ds} by {v} too"),
            (f"{Ms}{v} {MINUS} {Ds} = {Ts}", f"this subtracts the ${Ds} instead of adding it"),
            (f"{Ds} + {Ms} + {v} = {Ts}", f"this adds {v} instead of multiplying ${Ms} by {v}"),
        ]
        steps = [
            f"The one-time amount is ${Ds}, and each of the {v} repeated amounts is ${Ms}, so they add up to {Ms}{v}.",
            f"Together they equal the total: {Ds} + {Ms}{v} = {Ts}.",
        ]
        given = [D, M, T]
    else:
        v = f["each_var"]
        setup = f["setup_n"].format(n=n, D=Ds, k=k, T=Ts)
        question = f"Which equation represents this situation, where {f['each_def']}?"
        key = f"{Ds} + {k}{v} = {Ts}"
        mistakes = [
            (f"{Ds}{v} + {k} = {Ts}", f"this multiplies the ${Ds} by {v} and adds {k}"),
            (f"{k}({Ds} + {v}) = {Ts}", f"this counts the ${Ds} {k} times instead of once"),
            (f"{k}{v} = {Ts} + {Ds}", f"this adds the ${Ds} to the total instead of subtracting it"),
            (f"{Ds} + {v} = {Ts}", f"this leaves out that there were {k} payments"),
        ]
        steps = [
            f"There were {k} payments of {v} dollars each, which add up to {k}{v}.",
            f"Together with the one-time ${Ds}, they equal the total: {Ds} + {k}{v} = {Ts}.",
        ]
        given = [D, k, T]
    item = MathItem(
        "payment_equation", SKILL, level, "mcq", f"{setup} {question}", Fraction(0), steps, mistakes, key_text=key
    )
    item.story = dict(
        kind="setup",
        level=level,
        setup=setup,
        question=question,
        given=given,
        truth=k if level == "easy" else M,
        situation=f"{n}: {f['name']} plan with a one-time amount plus equal repeated amounts",
    )
    return item


def interpret_parts(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'The equation 150 + 45p = 690 represents ... What does 45 represent?'  Easy asks about one number; medium asks
    about the product (45p), the part students most often mislabel."""
    f, n, D, M, k, T = _plan_numbers(rng)
    v = f["var"]
    Ds, Ms, Ts = num(D), num(M), num(T)
    eq = f"{Ds} + {Ms}{v} = {Ts}"
    phrases = {part: f[part].format(n=n) for part in ("start", "each", "product", "total")}
    target = rng.choice(["start", "each", "total"]) if level == "easy" else "product"
    shown = {"start": Ds, "each": Ms, "product": f"{Ms}{v}", "total": Ts}
    why = {
        "start": f"this is what {Ds} represents",
        "each": f"this is what {Ms} represents",
        "product": f"this is what {Ms}{v} represents",
        "total": f"this is what {Ts} represents",
    }
    stem = (
        f"The equation {eq} represents {f['eq'].format(n=n)}, where {f['var_def'].format(n=n)}. What does "
        f"{shown[target]} represent in this equation?"
    )
    mistakes = [(phrases[p], why[p]) for p in ("product", "each", "start", "total") if p != target]
    steps = [
        f"In {eq}, {Ds} is added once ({phrases['start'].lower()}), {Ms} is multiplied by {v} "
        f"({phrases['each'].lower()}), so {Ms}{v} is {phrases['product'][0].lower() + phrases['product'][1:]}, "
        f"and {Ts} is {phrases['total'][0].lower() + phrases['total'][1:]}.",
        f"So {shown[target]} represents {phrases[target][0].lower() + phrases[target][1:]}.",
    ]
    return MathItem(
        "interpret_parts", SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=phrases[target]
    )


def in_terms_of(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'3(2x + 10) − 4 = 5(x + 5) + 8. If x is the solution to the given equation, what is the value of x + 5?'
    Both sides are built from the same group (x + 5), so a student can let k = x + 5 (2x + 10 = 2k) and solve for k
    directly, or solve for x and add 5. Hard: the left group is 2 or 3 times (x + c), c can be negative, and the
    answer can be negative."""
    while True:
        c = rng.randint(2, 12) if level != "hard" else rng.choice([n for n in range(-12, 13) if abs(n) >= 2])
        m = 2 if level != "hard" else rng.choice([2, 3])
        a, d = rng.randint(2, 7), rng.randint(2, 9)
        u = rng.randint(2, 15) if level != "hard" else rng.choice([n for n in range(-9, 13) if n not in (0, 1)])
        b = rng.choice([n for n in range(-20, 21) if abs(n) >= 2])
        e = (a * m - d) * u + b
        if a * m != d and e and abs(e) >= 2 and u - c != u and (u - c) != 0:
            break
    group = lin(1, c)
    passage = f"{a}({lin(m, m * c)}){plus(b)} = {d}({group}){plus(e)}"
    stem = f"If x is the solution to the given equation, what is the value of {group}?"
    mistakes = [
        (Fraction(u - c), f"this is the value of x; the question asks for the value of {group}"),
        (Fraction(m * u), f"this is the value of {lin(m, m * c)}, not {group}"),
        (
            Fraction(e - b, a - d) if a != d else Fraction(0),
            f"this treats {lin(m, m * c)} as if it were {group}, forgetting the factor of {m}",
        ),
        (
            Fraction(e + b, a * m - d),
            f"this {'adds' if b > 0 else 'subtracts'} {num(abs(b))} instead of "
            f"{'subtracting' if b > 0 else 'adding'} it",
        ),
        (Fraction(-u), "this has the right size but the wrong sign"),
    ]
    steps = [
        f"Let k = {group}. Then {lin(m, m * c)} = {m}k, so the equation becomes {a}({m}k){plus(b)} = "
        f"{d}k{plus(e)}, or {a * m}k{plus(b)} = {d}k{plus(e)}.",
        f"Solve for k: {term(a * m - d, 'k')} = {num(e - b)}, so k = {num(u)}.",
        f"So {group} = {num(u)}. (Solving for x first also works: x = {num(u - c)}, and {num(u - c)}{plus(c)} = "
        f"{num(u)}.)",
    ]
    return MathItem(
        "in_terms_of", SKILL, level, fmt, stem, Fraction(u), steps, mistakes, passage=passage, positive_only=False
    )


FAMILY_LEVELS = {
    "expression_value": LEVELS,
    "linear_rate": LEVELS,
    "solve_simple": ("easy", "medium"),
    "solve_combine": ("easy", "medium"),
    "one_solution_cannot": ("medium", "hard"),
    "infinite_constant": ("medium", "hard"),
    "three_kinds_equation": ("easy", "medium"),
    "payment_equation": ("easy", "medium"),
    "interpret_parts": ("easy", "medium"),
    "in_terms_of": ("medium", "hard"),
}

FAMILIES = {
    "expression_value": expression_value,
    "linear_rate": linear_rate,
    "solve_simple": solve_simple,
    "solve_combine": solve_combine,
    "one_solution_cannot": one_solution_cannot,
    "infinite_constant": infinite_constant,
    "three_kinds_equation": three_kinds_equation,
    "payment_equation": payment_equation,
    "interpret_parts": interpret_parts,
    "in_terms_of": in_terms_of,
}


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
SETUP_PROMPT = """Write the setup for a Digital SAT Math word problem (Algebra, {level}).

Situation: {situation}.
Numbers to use, each exactly once, in this order, written exactly like this: {numbers}. Do not add any other numbers.
Plain version (keep every fact, the role of every number, the people's names, and the names of the items, because the
question refers to them; you may reword it and add one natural detail): {setup}

Write 2 to 4 short, natural sentences. Do NOT write the question; this exact question will follow your text:
"{question}"
Reply with JSON only: {{"setup": "<the sentences>"}}"""
SOLVE_MCQ_PROMPT = 'Answer this SAT math question. Reply with JSON only: {{"answer": "<A, B, C, or D>"}}\n\n{question}'


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
        setup_kind = f.get("kind") == "setup"
        for attempt in range(self.tries):
            if setup_kind:  # the writer words the setup; the exact question is added by code
                text = self.writer(SETUP_PROMPT.format(numbers=numbers, **f))
            else:
                text = self.writer(STORY_PROMPT.format(numbers=numbers, template=item.stem, **f))
            try:
                got = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                stem = (got["setup"].strip() + " " + f["question"]) if setup_kind else got["stem"].strip()
            except (AttributeError, ValueError, KeyError):
                continue
            problems = story_numbers_ok(stem, f["given"])
            votes = []
            if not problems and self.solver and item.key_text:
                shown = "\n".join([item.passage, stem] + [f"{k}) {v}" for k, v in item.choices.items()]).strip()
                for _ in range(3):
                    m = re.search(r'"answer"\s*:\s*"?\s*([A-D])', self.solver(SOLVE_MCQ_PROMPT.format(question=shown)))
                    votes.append(m.group(1) if m else None)
                if sum(v == item.key for v in votes) < 2:
                    problems.append(f"a blind solver chose {votes}, not {item.key}")
            elif not problems and self.solver:
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
        levels = FAMILY_LEVELS.get(fam, LEVELS)
        level = levels[(i // len(families)) % len(levels)]
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
    run.add_argument(
        "--ai-stories",
        action="store_true",
        help="let the writer model word the word problems (uses API credits); default: free templates",
    )
    run.add_argument("--offline", action="store_true", help="kept for old commands; runs are offline by default")
    run.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    out = args.out or Path("runs") / f"math-{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    story = None
    if args.ai_stories and not args.offline:
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
