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


def dec(n) -> str:
    """Fraction(2, 5) -> '0.4'; Fraction(3) -> '3' (for decimal equations)."""
    n = Fraction(n)
    if n.denominator == 1:
        return num(n)
    text = f"{abs(float(n)):,.4f}".rstrip("0").rstrip(".")
    return (MINUS if n < 0 else "") + text


def money(n) -> str:
    """Fraction(1, 10) -> '$0.10'; Fraction(25) -> '$25'; 1250 -> '$1,250'."""
    n = Fraction(n)
    return f"${n.numerator:,}" if n.denominator == 1 else f"${float(n):,.2f}"


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
    fractions_ok: bool = False  # fraction wrong choices are natural here (fraction answers)
    fixed_choices: list[str] | None = None  # text choices in a set order ("Zero", "Exactly one", ...)
    truth: dict | None = None  # values of the constants behind the question (for tests)
    decimals: bool = False  # show numeric choices as decimals (0.4) instead of fractions (2/5)

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
        # there, x itself is the trap
        whole = not (self.fractions_ok or self.family == "expression_value" and self.level == "medium")
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
        show = dec if self.decimals else num
        self.choices = {k: show(v) for k, (v, _) in zip("ABCD", values, strict=True)}
        self.key = next(k for k, (v, _) in zip("ABCD", values, strict=True) if v == self.answer)
        self.mistakes = list(wrong[:3])
        if used is not None:
            used[self.key] = used.get(self.key, 0) + 1

    def _build_text_choices(self, rng: random.Random, used: dict[str, int] | None) -> None:
        """Equations or phrases: the key goes to the letter used least so far; the three most common mistakes fill
        the other letters in random order."""
        if self.fixed_choices:  # a set list in a set order; the key's letter follows from the list
            self.choices = dict(zip("ABCD", self.fixed_choices, strict=True))
            self.key = next(k for k, v in self.choices.items() if v == self.key_text)
            why = dict(self.mistakes)
            self.mistakes = [(v, why[v]) for v in self.fixed_choices if v != self.key_text]
            if used is not None:
                used[self.key] = used.get(self.key, 0) + 1
            return
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
                if self.key_text:
                    why = next(w for v, w in self.mistakes if v == text)
                    lines.append(f"Choice {k}: {why}.")
                    continue
                show = dec if self.decimals else num
                why = next(w for v, w in self.mistakes if (v if self.key_text else show(v)) == text)
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


def nz(lo: int, hi: int) -> list[int]:
    """The nonzero integers from lo to hi."""
    return [n for n in range(lo, hi + 1) if n]


def reciprocal_ratio(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'If x/4 = 6, what is the value of 4/x?' (easy: the flip shortcut gives 1/6). Medium asks for c/x with a different
    numerator, so the flip alone is not enough."""
    while True:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        x = a * b
        c = a if level == "easy" else rng.choice([n for n in range(2, 31) if n != a])
        ans = Fraction(c, x)
        if level == "easy" or (ans != Fraction(1, b) and ans.denominator != 1):
            break
    stem = f"If x/{a} = {b}, what is the value of {c}/x?"
    mistakes = [
        (Fraction(b), f"this is the value of x/{a}, not {c}/x"),
        (Fraction(x), f"this is the value of x; the question asks for {c}/x"),
        (Fraction(a, b), f"this divides {a} by {b} instead of finding x first"),
        (Fraction(b, a), f"this divides {b} by {a}, as if x/{a} = {b} meant x = {b}/{a}"),
        (Fraction(c * b, a) if c != a else Fraction(c, b * b), "this mixes up which number to divide by"),
    ]
    if c != a:
        mistakes.insert(0, (Fraction(1, b), f"this flips x/{a} = {b}, which gives {a}/x, not {c}/x"))
        mistakes.insert(2, (Fraction(c, b), f"this divides {c} by {b} instead of by x"))
    steps = [f"Multiply both sides by {a}: x = {a}({b}) = {x}.", f"So {c}/x = {c}/{x} = {num(ans)}."]
    if c == a:
        steps.append(f"Shortcut: {a}/x is x/{a} flipped, so it equals 1/{b}.")
    return MathItem("reciprocal_ratio", SKILL, level, fmt, stem, ans, steps, mistakes, fractions_ok=True)


def same_product(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'If 6(4x) = 72, what is the value of 4x?'  No need to find x. Medium adds a constant: '6(4x) − 9 = 63'."""
    while True:
        a, b, x = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 12)
        d = rng.choice(nz(-20, 20)) if level == "medium" else 0
        u = b * x
        c = a * u + d
        if a != b and c > 0 and abs(d) != a:
            break
    left = f"{a}({b}x)" + (plus(d) if d else "")
    stem = f"If {left} = {c}, what is the value of {b}x?"
    mistakes = [
        (Fraction(x), f"this is the value of x; the question asks for {b}x"),
        (Fraction(c - d, b), f"this divides by {b} instead of by {a}"),
        (Fraction(u * b), f"this multiplies by {b} again"),
        (
            Fraction(c - a) if not d else Fraction(c + d, a),
            "this subtracts instead of dividing"
            if not d
            else f"this {'adds' if d > 0 else 'subtracts'} {num(abs(d))} instead of "
            f"{'subtracting' if d > 0 else 'adding'} it",
        ),
        (Fraction(c - d), f"this stops before dividing by {a}"),
    ]
    steps = ([f"{'Subtract' if d > 0 else 'Add'} {num(abs(d))}: {a}({b}x) = {num(c - d)}."] if d else []) + [
        f"Divide both sides by {a}: {b}x = {num(c - d)} ÷ {a} = {u}. There is no need to find x."
    ]
    return MathItem("same_product", SKILL, level, fmt, stem, Fraction(u), steps, mistakes)


def no_solution_parens(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'(r − 4)x = 9 ... r is a constant ... no solution ... value of r?'  No solution means the x-coefficient is 0
    (and the constant is not). Medium has x on both sides: '(r + 3)x = 7x + 5' means r + 3 = 7."""
    r = rng.choice(["r", "k", "a", "m", "c"])
    while True:
        a = rng.choice(nz(-12, 12))
        b = rng.choice(nz(-20, 30))
        cc = rng.randint(2, 12) if level == "medium" else 0
        ans = cc - a
        if ans and abs(a) >= 2 and len({ans, -ans, b, cc}) == 4:
            break
    inner = f"{r}{plus(a)}"
    if level == "easy":
        passage = f"({inner})x = {num(b)}"
        mistakes = [
            (Fraction(-ans), "this has the right size but the wrong sign"),
            (Fraction(b), "this is the right side of the equation"),
            (Fraction(b - a), f"this sets {inner} equal to {num(b)} instead of 0"),
            (Fraction(1 - a), f"this sets {inner} equal to 1 instead of 0"),
        ]
        steps = [
            f"If {inner} = 0, the left side is 0 for every x, but the right side is {num(b)}, so there is no "
            f"solution. Any other value of {r} gives exactly one solution.",
            f"{inner} = 0 means {r} = {num(ans)}.",
        ]
    else:
        passage = f"({inner})x = {lin(cc, b)}"
        mistakes = [
            (Fraction(cc), f"this sets {r} equal to {cc}, ignoring the {num(a)}"),
            (Fraction(cc + a), f"this moves the {num(a)} without changing its sign"),
            (Fraction(-ans), "this has the right size but the wrong sign"),
            (Fraction(b), "this is the constant on the right side"),
        ]
        steps = [
            f"No solution means the x-terms must cancel while the constants differ (0 = {num(b)}).",
            f"The x-coefficients are equal when {inner} = {cc}, so {r} = {num(ans)}.",
        ]
    stem = f"In the given equation, {r} is a constant. If the equation has no solution, what is the value of {r}?"
    return MathItem(
        "no_solution_parens",
        SKILL,
        level,
        fmt,
        stem,
        Fraction(ans),
        steps,
        mistakes,
        passage=passage,
        positive_only=False,
    )


def no_solution_k(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'3(kx − 2) = 5x + 1 ... k is a constant ... no solution ... value of k?'  Set the x-coefficients equal:
    3k = 5, so k = 5/3. Medium: k is a whole number. Hard: k is a fraction."""
    k = rng.choice(["k", "a", "b", "p"])
    while True:
        a, b = rng.randint(2, 7), rng.choice(nz(-12, 12))
        cc = a * rng.randint(1, 6) if level == "medium" else rng.choice([n for n in range(2, 20) if n % a])
        d = rng.choice(nz(-20, 20))
        ans = Fraction(cc, a)
        if d != -a * b and ans not in (cc, a * cc):
            break
    passage = f"{a}({k}x{plus(-b)}) = {lin(cc, d)}"
    stem = f"In the given equation, {k} is a constant. The equation has no solution. What is the value of {k}?"
    mistakes = [
        (Fraction(cc), f"this forgets that {k}x is multiplied by {a}"),
        (Fraction(a, cc), f"this divides {a} by {cc} instead of {cc} by {a}"),
        (Fraction(a * cc), f"this multiplies {cc} by {a} instead of dividing"),
        (Fraction(cc - a), f"this subtracts {a} instead of dividing by {a}"),
        (-ans, "this has the right size but the wrong sign"),
    ]
    steps = [
        f"Distribute: {a}{k}x{plus(-a * b)} = {lin(cc, d)}.",
        f"No solution means the x-terms cancel but the constants differ ({num(-a * b)} ≠ {num(d)}), so the "
        f"x-coefficients are equal: {a}{k} = {cc}.",
        f"So {k} = {num(ans)}.",
    ]
    return MathItem(
        "no_solution_k",
        SKILL,
        level,
        fmt,
        stem,
        ans,
        steps,
        mistakes,
        passage=passage,
        positive_only=False,
        fractions_ok=ans.denominator != 1,
    )


SOLUTION_COUNTS = ["Zero", "Exactly one", "Exactly two", "Infinitely many"]


def how_many_solutions(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Both sides built up; 'How many solutions does the given equation have?'  The answer is zero, exactly one, or
    infinitely many, depending on whether the x-coefficients and the constants match. Hard adds an x-term to each
    side, so more simplifying is needed."""
    case = rng.choice(["Zero", "Exactly one", "Infinitely many"])
    while True:
        a, p_, q = rng.randint(2, 6), rng.randint(1, 4), rng.choice(nz(-9, 9))
        r_ = rng.choice(nz(-5, 5)) if level == "hard" else 0
        s_ = rng.choice(nz(-12, 12))
        A, B = a * p_ + r_, a * q + s_
        A2 = A + rng.choice(nz(-4, 4)) if case == "Exactly one" else A
        B2 = B + rng.choice(nz(-9, 9)) if case == "Zero" else (B if case == "Infinitely many" else rng.randint(-20, 20))
        t, u = rng.randint(2, 5), rng.randint(1, 3)
        w = A2 - t * u if level == "hard" else 0
        v = rng.choice(nz(-9, 9))
        z = B2 - t * v
        if A and A2 and (level != "hard" or w) and z and (level == "hard" or A2 % t == 0):
            if level != "hard":
                u = A2 // t
            if u:
                break
    left = f"{a}({lin(p_, q)})" + (f" + {term(r_)}" if r_ > 0 else f" {MINUS} {term(-r_)}" if r_ else "") + plus(s_)
    right = f"{t}({lin(u, v)})" + (f" + {term(w)}" if w > 0 else f" {MINUS} {term(-w)}" if w else "") + plus(z)
    passage = f"{left} = {right}"
    why = {
        "Zero": (
            f"the x-coefficients ({A} and {A2}) are different, so there is a solution"
            if A != A2
            else "the constants match too, so every x works"
        ),
        "Exactly one": (f"the x-terms cancel ({A}x on both sides), so x can't be solved for" if A == A2 else ""),
        "Exactly two": "an equation like this, with no x², can't have exactly two solutions",
        "Infinitely many": (
            f"the x-coefficients ({A} and {A2}) are different"
            if A != A2
            else f"the constants are different ({num(B)} and {num(B2)})"
        ),
    }
    mistakes = [(c, why[c]) for c in SOLUTION_COUNTS if c != case]
    simple = f"{lin(A, B)} = {lin(A2, B2)}"
    verdict = {
        "Exactly one": f"The x-coefficients differ ({A} and {A2}), so there is exactly one solution.",
        "Zero": f"The x-terms cancel, leaving {num(B)} = {num(B2)}, which is false: no solution.",
        "Infinitely many": "Both sides are the same expression, so every x is a solution.",
    }[case]
    steps = [f"Simplify each side: {simple}.", verdict]
    return MathItem(
        "how_many_solutions",
        SKILL,
        level,
        "mcq",
        "How many solutions does the given equation have?",
        Fraction(0),
        steps,
        mistakes,
        passage=passage,
        key_text=case,
        fixed_choices=SOLUTION_COUNTS,
    )


PAIRS = [("a", "b"), ("h", "k"), ("m", "n"), ("p", "q")]


def infinite_two_constants(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'a(2x + 3) + 4x = 14x + b ... a and b are constants ... infinitely many solutions ... value of b?'  Match the
    x-terms (2a + 4 = 14, so a = 5), then the constants (b = 3a = 15). Medium has no extra x-term and asks for a or
    b; hard adds the x-term and can ask for a + b."""
    c1, c2 = rng.choice(PAIRS)
    while True:
        a_, p_, q = rng.randint(2, 9), rng.randint(1, 4), rng.choice(nz(-9, 9))
        r_ = rng.choice(nz(-6, 6)) if level == "hard" else 0
        C = p_ * a_ + r_
        b_ = q * a_
        if C > 0:
            break
    ask = rng.choice([c1, c2]) if level != "hard" else rng.choice([c2, f"{c1} + {c2}"])
    value = {c1: a_, c2: b_, f"{c1} + {c2}": a_ + b_}[ask]
    left = f"{c1}({lin(p_, q)})" + (f" + {term(r_)}" if r_ > 0 else f" {MINUS} {term(-r_)}" if r_ else "")
    passage = f"{left} = {lin(C, 0)} + {c2}"
    stem = (
        f"In the given equation, {c1} and {c2} are constants. If the equation has infinitely many solutions, "
        f"what is the value of {ask}?"
    )
    a_wrong = Fraction(C, p_)  # leaves out the r term
    mistakes = {
        c1: [
            (Fraction(b_), f"this is the value of {c2}, not {c1}"),
            (
                a_wrong if r_ else Fraction(C),
                f"this leaves out the {term(r_)} term" if r_ else f"this forgets to divide by {p_}",
            ),
            (
                Fraction(C + r_, p_) if r_ else Fraction(C - p_),
                f"this moves the {term(r_)} with the wrong sign" if r_ else f"this subtracts {p_} instead of dividing",
            ),
            (Fraction(-a_), "this has the right size but the wrong sign"),
        ],
        c2: [
            (Fraction(a_), f"this is the value of {c1}, not {c2}"),
            (Fraction(q), f"this matches the constants before multiplying by {c1}"),
            (
                q * a_wrong if r_ else Fraction(q * C),
                f"this finds {c1} without the {term(r_)} term"
                if r_
                else f"this multiplies {num(q)} by {C} instead of by {c1}",
            ),
            (Fraction(-b_), "this has the right size but the wrong sign"),
        ],
    }
    mistakes[f"{c1} + {c2}"] = [
        (Fraction(a_), f"this is only {c1}"),
        (Fraction(b_), f"this is only {c2}"),
        (Fraction(a_ * b_), f"this multiplies {c1} and {c2} instead of adding"),
        (Fraction(a_ - b_), f"this subtracts {c2} instead of adding it"),
    ]
    xr = f" + {term(r_)}" if r_ > 0 else f" {MINUS} {term(-r_)}" if r_ else ""
    const = f" + {num(q)}{c1}" if q > 0 else f" {MINUS} {num(-q)}{c1}"
    steps = [
        f"Distribute: {p_ if p_ != 1 else ''}{c1}x{const}{xr} = {lin(C, 0)} + {c2}.",
        f"Infinitely many solutions means the x-terms match: {p_ if p_ != 1 else ''}{c1}{plus(r_) if r_ else ''} "
        f"= {C}, so {c1} = {a_}.",
        f"The constants match too: {c2} = {num(q)}{c1} = {num(q)}({a_}) = {num(b_)}.",
    ]
    if ask == f"{c1} + {c2}":
        steps.append(f"So {c1} + {c2} = {a_} + {num(b_)} = {num(a_ + b_)}.")
    item = MathItem(
        "infinite_two_constants",
        SKILL,
        level,
        fmt,
        stem,
        Fraction(value),
        steps,
        mistakes[ask],
        passage=passage,
        positive_only=False,
    )
    item.truth = {c1: a_, c2: b_}
    return item


def terminating(n: Fraction) -> bool:
    """True if n can be written as a short decimal (denominator built from 2s and 5s)."""
    d = n.denominator
    for f in (2, 5):
        while d % f == 0:
            d //= f
    return d == 1 and n.denominator <= 100


def fraction_decimal(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: decimals, '0.4x + 1.2 = 3.6'. Medium: fractions, 'x/3 + x/4 = 7' (clear them with 12). Hard: cross
    multiplying, '(x + 4)/3 = (x − 2)/5'."""
    if level == "easy":
        while True:
            a = Fraction(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 25]), 10)
            x = rng.randint(2, 20)
            b = Fraction(rng.choice(nz(-40, 40)), 10)
            c = a * x + b
            if c > 0 and b.denominator != 1:
                break
        passage = f"{dec(a)}x{' + ' + dec(b) if b > 0 else ' ' + MINUS + ' ' + dec(-b)} = {dec(c)}"
        stem = "What value of x is the solution to the given equation?"
        mistakes = [
            (Fraction(x * 10), "this moves the decimal point one place too far"),
            (Fraction(x, 10), "this moves the decimal point one place the wrong way"),
            (
                (c + b) / a,
                f"this {'adds' if b > 0 else 'subtracts'} {dec(abs(b))} instead of "
                f"{'subtracting' if b > 0 else 'adding'} it",
            ),
            (c - b, f"this stops before dividing by {dec(a)}"),
            (c / a - b, f"this divides by {dec(a)} before removing the {dec(abs(b))}"),
        ]
        mistakes = [(v, w) for v, w in mistakes if terminating(v)]
        steps = [
            f"{'Subtract' if b > 0 else 'Add'} {dec(abs(b))}: {dec(a)}x = {dec(c - b)}.",
            f"Divide by {dec(a)}: x = {dec(c - b)} ÷ {dec(a)} = {x}. (Or multiply everything by 10 first: "
            f"{dec(10 * a)}x{' + ' if b > 0 else ' ' + MINUS + ' '}{dec(abs(10 * b))} = {dec(10 * c)}.)",
        ]
        return MathItem(
            "fraction_decimal",
            SKILL,
            level,
            fmt,
            stem,
            Fraction(x),
            steps,
            mistakes,
            passage=passage,
            fractions_ok=True,
            decimals=True,
        )
    if level == "medium":
        while True:
            p_, q = rng.sample(range(2, 9), 2)
            L = p_ * q // __import__("math").gcd(p_, q)
            x = L * rng.randint(1, 6)
            r = x // p_ + x // q
            if len({x, r * (p_ + q), r * p_, r * L}) == 4:
                break
        passage = f"x/{p_} + x/{q} = {r}"
        stem = "What value of x is the solution to the given equation?"
        mistakes = [
            (Fraction(r * (p_ + q)), f"this adds the fractions as x/{p_ + q}, adding the denominators"),
            (Fraction(r * p_), f"this leaves out the x/{q} term"),
            (Fraction(r * L), f"this multiplies only the right side by {L}"),
            (Fraction(r * p_ * q), f"this multiplies by {p_} and by {q} but forgets that the x-terms add up"),
        ]
        steps = [
            f"Multiply every term by {L}, the least common denominator: {L // p_}x + {L // q}x = {r * L}.",
            f"Combine: {L // p_ + L // q}x = {r * L}, so x = {x}.",
        ]
        return MathItem("fraction_decimal", SKILL, level, fmt, stem, Fraction(x), steps, mistakes, passage=passage)
    while True:  # hard: (x + a)/p = (x − b)/q
        p_, q = rng.sample(range(2, 10), 2)
        k = rng.choice(nz(-6, 6))
        a = rng.choice(nz(-9, 9))
        b = k * (p_ - q) - a
        x = k * p_ - a
        if b and x and abs(b) <= 15:
            break
    passage = f"(x{plus(a)})/{p_} = (x{plus(-b)})/{q}"
    stem = "What value of x is the solution to the given equation?"
    mistakes = [
        (Fraction(-b - a, q - p_), f"this multiplies only x by {q} and by {p_}, not the whole numerators"),
        (Fraction(-q * b - p_ * a, p_ - q), f"this cross multiplies the wrong way ({p_} with x{plus(a)})"),
        (Fraction(k), "this is the value of each fraction, not of x"),
        (Fraction(-x), "this has the right size but the wrong sign"),
    ]
    steps = [
        f"Cross multiply: {q}(x{plus(a)}) = {p_}(x{plus(-b)}).",
        f"Distribute: {lin(q, q * a)} = {lin(p_, -p_ * b)}.",
        f"Collect terms: {term(q - p_)} = {num(-p_ * b - q * a)}" + (f", so x = {num(x)}." if q - p_ != 1 else "."),
    ]
    return MathItem(
        "fraction_decimal", SKILL, level, fmt, stem, Fraction(x), steps, mistakes, passage=passage, positive_only=False
    )


FEE_FRAMES = [
    dict(
        setup="{n}'s phone plan charges a monthly fee of {F} plus {r} for each text message sent. {n}'s bill for "
        "one month was {T}.",
        q="How many text messages did {n} send that month?",
        rates=[Fraction(1, 10), Fraction(1, 20), Fraction(1, 4), Fraction(3, 20)],
        counts=range(40, 400, 10),
        fees=(15, 60),
    ),
    dict(
        setup="A taxi company charges {F} to start a ride plus {r} for each mile driven. {n} paid {T} for one ride.",
        q="How many miles long was {n}'s ride?",
        rates=[Fraction(3, 2), 2, Fraction(5, 2), 3, Fraction(7, 4)],
        counts=range(3, 25),
        fees=(3, 10),
    ),
    dict(
        setup="A plumber charges {F} for a house call plus {r} for each hour of work. {n}'s bill for one house call "
        "was {T}.",
        q="For how many hours did the plumber work?",
        rates=[45, 60, 75, 80, 90],
        counts=range(2, 9),
        fees=(50, 180),
    ),
    dict(
        setup="A parking garage charges {F} to enter plus {r} for each hour a car is parked. {n} paid {T} to park "
        "one day.",
        q="For how many hours was {n}'s car parked?",
        rates=[Fraction(5, 2), 3, 4, Fraction(7, 2)],
        counts=range(2, 12),
        fees=(4, 20),
    ),
]


def word_solve(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Word problems that ask for a number. Easy: a fixed fee plus a cost per unit ('How many text messages?').
    Medium: a rectangle's perimeter ('What is the width?'). Hard: consecutive integers ('What is the greatest?')."""
    name = rng.choice(NAMES)
    if level == "easy":
        f = rng.choice(FEE_FRAMES)
        r = Fraction(rng.choice(f["rates"]))
        while True:
            # the fee is a whole number of units, so the mistakes come out whole, in this setting's usual range
            lo, hi = f["fees"]
            fee_units = rng.randint(max(2, -(-lo // r)), max(2, int(hi // r)))  # never fee = rate
            F = r * fee_units
            k = rng.choice(f["counts"])
            T = F + r * k
            if lo <= F <= hi and T.denominator in (1, 2, 4, 5, 10, 20, 100) and fee_units != k:
                break
        setup = f["setup"].format(n=name, F=money(F), r=money(r), T=money(T))
        question = f["q"].format(n=name)
        mistakes = [
            (T / r, f"this divides the whole bill by {money(r)}, leaving out the {money(F)}"),
            ((T + F) / r, f"this adds the {money(F)} instead of subtracting it"),
            (T - F, f"this stops before dividing by {money(r)}"),
            (Fraction(fee_units), f"this divides the {money(F)} by {money(r)}"),
        ]
        steps = [
            f"Subtract the fixed {money(F)}: {money(T)} {MINUS} {money(F)} = {money(T - F)}.",
            f"Divide by {money(r)} each: {money(T - F)} ÷ {money(r)} = {k}.",
        ]
        item = MathItem("word_solve", SKILL, level, fmt, f"{setup} {question}", Fraction(k), steps, mistakes)
        item.story = dict(
            kind="setup",
            level=level,
            setup=setup,
            question=question,
            given=[F, r, T],
            shown=[money(F), money(r), money(T)],
            situation="a fixed fee plus a cost for each unit",
        )
        item.truth = {"F": F, "r": r, "T": T}
        return item
    if level == "medium":
        unit = rng.choice(["feet", "meters", "yards"])
        times = rng.random() < 0.5
        ask = rng.choice(["width", "length"])
        while True:
            m = rng.randint(2, 5)
            # an even "longer by" amount, and a width that is an even multiple of m, keep the mistakes whole numbers
            w = rng.randint(3, 40) if not times else 2 * m * rng.randint(1, 6)
            more = 2 * rng.randint(1, 10)
            L_ = m * w if times else w + more
            P = 2 * (w + L_)
            if L_ != w:
                break
        rel = f"{m} times its width" if times else f"{more} {unit} longer than its width"
        setup = f"The length of a rectangular garden is {rel}. The perimeter of the garden is {P} {unit}."
        question = f"What is the {ask} of the garden, in {unit}?"
        ans, other = (w, L_) if ask == "width" else (L_, w)
        if times:  # the true width is P / (2(m + 1))
            once = Fraction(P, m + 1)  # added the length and width once instead of twice
            no_width = Fraction(P, 2 * m)  # left the width out of the perimeter
            scale = m if ask == "length" else 1
            mistakes = [
                (once * scale, "this adds the length and the width once instead of twice"),
                (no_width * scale, f"this leaves the width out: it solves 2({m}w) = {P}"),
            ]
        else:  # the true width is (P − 2·more) / 4
            once = Fraction(P - more, 2)
            extra_once = Fraction(P - more, 4)
            shift = more if ask == "length" else 0
            mistakes = [
                (once + shift, "this adds the length and the width once instead of twice"),
                (extra_once + shift, f"this subtracts the {more} extra {unit} only once instead of twice"),
            ]
        mistakes = [
            (
                Fraction(other),
                f"this is the {'length' if ask == 'width' else 'width'}; the question asks for the {ask}",
            ),
            (Fraction(P, 4), "this treats the garden as a square"),
        ] + mistakes
        steps = (
            [
                f"Let w be the width. The length is {m}w, so the perimeter is 2w + 2({m}w) = {2 + 2 * m}w = {P}.",
                f"So w = {w}" + (f" and the length is {m}({w}) = {L_}." if ask == "length" else "."),
            ]
            if times
            else [
                f"Let w be the width. The length is w + {more}, so the perimeter is 2w + 2(w + {more}) = 4w + "
                f"{2 * more} = {P}.",
                f"So 4w = {P - 2 * more} and w = {w}"
                + (f"; the length is {w} + {more} = {L_}." if ask == "length" else "."),
            ]
        )
        item = MathItem("word_solve", SKILL, level, fmt, f"{setup} {question}", Fraction(ans), steps, mistakes[:3])
        item.story = dict(
            kind="setup",
            level=level,
            setup=setup,
            question=question,
            given=[m, P] if times else [more, P],
            situation="a rectangle's length, width, and perimeter",
        )
        item.truth = {"w": w, "L": L_}
        return item
    # hard: consecutive integers
    step = rng.choice([1, 2, 2])
    count = rng.choice([3, 3, 4]) if step == 1 else 3
    kind = {1: "consecutive integers", 2: rng.choice(["consecutive even integers", "consecutive odd integers"])}[step]
    while True:
        first = rng.randint(5, 60)
        if step == 2 and (first % 2 == 0) != (kind == "consecutive even integers"):
            continue
        nums = [first + step * i for i in range(count)]
        break
    S = sum(nums)
    ask = rng.choice(["greatest", "least"])
    ans = nums[-1] if ask == "greatest" else nums[0]
    other = nums[0] if ask == "greatest" else nums[-1]
    words = {3: "three", 4: "four"}[count]
    setup = f"The sum of {words} {kind} is {S}."
    question = f"What is the {ask} of these integers?"
    mistakes = [
        (
            Fraction(other),
            f"this is the {'least' if ask == 'greatest' else 'greatest'} of the integers; the "
            f"question asks for the {ask}",
        ),
        (Fraction(S, count), "this is the average of the integers" + (", the middle one" if count == 3 else "")),
        (
            Fraction(nums[0] + step * count if ask == "greatest" else nums[-1] - step * count),
            "this counts one integer too many",
        ),
        (Fraction(nums[1] if count > 2 else nums[0]), "this is the second integer"),
    ]
    if step == 2:
        up_by_one = nums[0] + (count - 1) if ask == "greatest" else nums[-1] - (count - 1)
        mistakes.insert(2, (Fraction(up_by_one), "this counts up by 1 instead of by 2"))
    n0 = "n" + "".join(f" + (n + {step * i})" for i in range(1, count))
    steps = [
        "Call the least integer n. The integers are "
        + ", ".join(["n"] + [f"n + {step * i}" for i in range(1, count)])
        + ".",
        f"Their sum is {n0} = {count}n + {step * sum(range(count))} = {S}, so n = {nums[0]}.",
        f"The {ask} integer is {ans}.",
    ]
    item = MathItem("word_solve", SKILL, level, fmt, f"{setup} {question}", Fraction(ans), steps, mistakes)
    item.story = dict(
        kind="setup", level=level, setup=setup, question=question, given=[S], situation=f"the sum of {words} {kind}"
    )
    item.truth = {"nums": nums}
    return item


PROPERTIES = {
    "no solution": "Zero",
    "infinitely many solutions": "Infinitely many",
    "exactly one solution": "Exactly one",
}


def which_has_property(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Which equation has no solution?' (or infinitely many, or exactly one), with four equations as the choices.
    All four share the left side; the right sides differ in the x-coefficient, the constant, or both. Hard: the left
    side is built up and some right sides are factored."""
    prop = rng.choice(list(PROPERTIES))
    while True:
        a, p_, q = rng.randint(2, 7), (rng.randint(2, 4) if level == "hard" else 1), rng.choice(nz(-9, 9))
        s_ = rng.choice(nz(-9, 9)) if level == "hard" else 0
        A, B = a * p_, a * q + s_
        d, d2 = rng.sample(nz(-6, 6), 2)
        e, e2 = rng.sample([n for n in nz(-3, 3) if A + n], 2)
        if B + d and B - d and B + d2:
            break
    left = f"{a}({lin(p_, q)})" + (plus(s_) if s_ else "")

    def rhs(A2: int, B2: int) -> str:
        g = __import__("math").gcd(A2, B2)
        if level == "hard" and g > 1 and rng.random() < 0.5:
            return f"{g}({lin(A2 // g, B2 // g)})"
        return lin(A2, B2)

    cases = {
        "Infinitely many": (A, B),
        "Zero": (A, B + d),
        "Zero2": (A, B - d),
        "Exactly one": (A + e, B),
        "Exactly one2": (A + e2, B + d2),
    }
    say = {
        "Infinitely many": "this equation has infinitely many solutions: both sides are the same expression",
        "Zero": "this equation has no solution: same x-coefficient, different constants",
        "Exactly one": "this equation has exactly one solution: the x-coefficients are different",
    }
    pick = {
        "no solution": ("Zero", ["Infinitely many", "Exactly one", "Exactly one2"]),
        "infinitely many solutions": ("Infinitely many", ["Zero", "Exactly one", "Exactly one2"]),
        "exactly one solution": ("Exactly one", ["Infinitely many", "Zero", "Zero2"]),
    }[prop]
    key = f"{left} = {rhs(*cases[pick[0]])}"
    mistakes = [(f"{left} = {rhs(*cases[c])}", say[c.rstrip("2")]) for c in pick[1]]
    steps = [
        f"The left side simplifies to {lin(A, B)}.",
        {
            "no solution": "No solution means the same x-coefficient with a different constant.",
            "infinitely many solutions": "Infinitely many solutions means both sides are the same expression.",
            "exactly one solution": "Exactly one solution means the x-coefficients are different.",
        }[prop],
        f"Only {key} does that: its right side is {lin(*cases[pick[0]])}.",
    ]
    return MathItem(
        "which_has_property",
        SKILL,
        level,
        "mcq",
        f"Which equation has {prop}?",
        Fraction(0),
        steps,
        mistakes,
        key_text=key,
    )


# ----------------------------------------------------------------------------------------- linear functions (Nov 25)

FUNC_SKILL = "Linear functions"

MODEL_FRAMES = [
    dict(
        intro="A botanist is comparing how two bean plants grow in a greenhouse. The botanist began measuring both "
        "plants on the same day.",
        intro1="A botanist is measuring how a bean plant, plant A, grows in a greenhouse.",
        extra="Each plant grew at a constant rate during the study.",
        objs=("plant A", "plant B"),
        vars_=("a", "b"),
        t="w",
        t_unit="week",
        qty="height",
        model="the height {v}, in centimeters, of {o} {t} weeks after the botanist began measuring",
        slopes=[0.5, 1.5, 2, 2.5, 3, 3.5, 4],
        starts=range(5, 31),
        sign=1,
        rate="{O} grew {k} centimeters each week",
        start="{O}'s height was {k} centimeters when the botanist began measuring",
        total="{O} grew {k} centimeters in all during the weeks the botanist measured",
        more="{O} grew {k} centimeters more each week than {o2}",
        invert="It took {o} {k} weeks to grow 1 centimeter",
        final="{O}'s height was {k} centimeters when the botanist finished measuring",
    ),
    dict(
        intro="Two candles, a red candle and a white candle, were lit at the same time and burned at constant rates.",
        intro1="A red candle was lit and burned at a constant rate.",
        extra="Their heights were recorded until the first candle burned out.",
        objs=("the red candle", "the white candle"),
        vars_=("r", "s"),
        t="t",
        t_unit="hour",
        qty="height",
        model="the height {v}, in centimeters, of {o} {t} hours after it was lit",
        slopes=[0.4, 0.5, 0.6, 0.8, 1.2, 1.5],
        starts=range(12, 31),
        sign=-1,
        rate="{O}'s height decreased by {k} centimeters each hour",
        start="{O}'s height was {k} centimeters when it was lit",
        total="{O}'s height decreased by {k} centimeters in all while it burned",
        more="{O}'s height decreased by {k} centimeters more each hour than {o2}'s height",
        invert="It took {k} hours for {o}'s height to decrease by 1 centimeter",
        final="{O}'s height was {k} centimeters when it burned out",
    ),
    dict(
        intro="A city is draining two water tanks, tank P and tank Q, for cleaning. Both tanks began draining at "
        "the same time.",
        intro1="A city is draining a water tank, tank P, for cleaning.",
        extra="The water level in each tank was recorded every minute until the tank was empty.",
        objs=("tank P", "tank Q"),
        vars_=("p", "q"),
        t="m",
        t_unit="minute",
        qty="amount of water",
        model="the amount of water {v}, in gallons, in {o} {t} minutes after draining began",
        slopes=[12, 15, 18, 20, 25, 30],
        starts=range(300, 1001, 25),
        sign=-1,
        rate="The amount of water in {o} decreased by {k} gallons each minute",
        start="{O} held {k} gallons of water when draining began",
        total="{O} lost {k} gallons of water in all while it drained",
        more="{O} lost {k} more gallons of water each minute than {o2}",
        invert="It took {k} minutes for 1 gallon of water to drain from {o}",
        final="{O} held {k} gallons of water when draining ended",
    ),
    dict(
        intro="Two friends, {n1} and {n2}, started saving money for a trip in the same week. Each of them adds "
        "the same amount to a savings account every week.",
        intro1="{n1} started saving money for a trip and adds the same amount to a savings account every week.",
        extra="Neither friend withdrew any money during this time.",
        objs=("{n1}", "{n2}"),
        vars_=("j", "k"),
        t="w",
        t_unit="week",
        qty="savings",
        model="the amount of money {v}, in dollars, in {o}'s account {t} weeks after they started saving",
        slopes=[15, 20, 25, 30, 40, 45],
        starts=range(50, 401, 10),
        sign=1,
        money=True,
        rate="{O} added ${k} to the account each week",
        start="{O} had ${k} in the account when they started saving",
        total="{O} saved ${k} in all during the weeks they saved",
        more="{O} added ${k} more to the account each week than {o2}",
        invert="It took {o} {k} weeks to save $1",
        final="{O} had ${k} in the account when they stopped saving",
    ),
    dict(
        intro="Two hikers, {n1} and {n2}, began descending a mountain at the same time, each at a steady pace.",
        intro1="A hiker, {n1}, began descending a mountain at a steady pace.",
        extra="Each hiker's elevation was recorded by a GPS watch.",
        objs=("{n1}", "{n2}"),
        vars_=("e", "f"),
        t="t",
        t_unit="minute",
        qty="elevation",
        model="the elevation {v}, in feet, of {o} {t} minutes after the descent began",
        slopes=[20, 25, 30, 40, 50],
        starts=range(1500, 4001, 100),
        sign=-1,
        rate="{O}'s elevation decreased by {k} feet each minute",
        start="{O}'s elevation was {k} feet when the descent began",
        total="{O} descended {k} feet in all",
        more="{O} descended {k} more feet each minute than {o2}",
        invert="It took {o} {k} minutes to descend 1 foot",
        final="{O}'s elevation was {k} feet when the descent ended",
    ),
]


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def interpret_models(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Real-world linear models, y = mx + b. 'What is the best interpretation of 2.5 in this context?'
    Medium: one model. Hard: two models in about four long sentences, so the wrong answers can also mix up the two
    things or turn the number into a comparison. Traps: slope vs. starting value, the wrong object, a comparison,
    a total instead of a rate, the rate flipped (units per thing instead of things per unit)."""
    f = dict(rng.choice(MODEL_FRAMES))
    n1, n2 = rng.sample(NAMES, 2)
    objs = [o.format(n1=n1, n2=n2) for o in f["objs"]]
    intro = f["intro"].format(n1=n1, n2=n2)
    intro1 = f["intro1"].format(n1=n1, n2=n2)
    while True:
        m = [Fraction(str(x)) for x in rng.sample(f["slopes"], 2)]
        b = rng.sample(list(f["starts"]), 2)
        shown = {dec(m[0]), dec(m[1]), num(b[0]), num(b[1])}
        if len(shown) == 4:
            break
    sign = f["sign"]
    two = level == "hard"
    eqs = [f"{f['vars_'][i]} = {MINUS if sign < 0 else ''}{dec(m[i])}{f['t']} + {num(b[i])}" for i in range(2)]
    models = [f["model"].format(v=f["vars_"][i], o=objs[i], t=f["t"]) for i in range(2)]
    if two:
        body = (
            f"{intro} The equation {eqs[0]} models {models[0]}, and the equation {eqs[1]} models {models[1]}. "
            f"{f['extra']}"
        )
        role, i = rng.choice(["rate", "start"]), rng.randint(0, 1)
    else:
        body = f"{intro1} The equation {eqs[0]} models {models[0]}."
        role, i = rng.choice(["rate", "start"]), 0
    j = 1 - i
    val = m[i] if role == "rate" else Fraction(b[i])
    k_text = f"{num(b[i])}" if role == "start" else dec(m[i])
    asked = f"{MINUS}{dec(m[i])}" if role == "rate" and sign < 0 else k_text
    money_k = k_text  # money frames already put the $ in their phrases

    def say(kind: str, o: int, o2: int | None = None) -> str:
        return (
            _cap(
                f[kind].format(
                    O=_cap(objs[o]),
                    o=objs[o],
                    O2=_cap(objs[o2]) if o2 is not None else "",
                    o2=objs[o2] if o2 is not None else "",
                    k=money_k,
                )
            )
            + "."
        )

    key = say(role, i)
    if role == "rate":
        pool = [
            (
                say("start", i),
                f"this describes a starting amount; {asked} is the coefficient of {f['t']}, the "
                f"change each {f['t_unit']}",
            ),
            (say("total", i), f"{asked} is the change each {f['t_unit']}, not the total change"),
            (
                say("invert", i),
                f"this flips the rate: {asked} is the change in {f['qty']} each "
                f"{f['t_unit']}, not the {f['t_unit']}s needed for a change of 1",
            ),
        ]
        if two:
            pool += [
                (say("rate", j), f"{asked} is in the equation for {objs[i]}, not {objs[j]}"),
                (
                    say("more", i, j),
                    f"this compares the two {'rates' if sign > 0 else 'models'}; {asked} is {objs[i]}'s own rate",
                ),
            ]
    else:
        pool = [
            (say("rate", i), f"this describes a rate; {asked} is the constant term, the value when {f['t']} = 0"),
            (say("final", i), f"{asked} is the value at the start ({f['t']} = 0), not at the end"),
            (say("total", i), f"{asked} is the starting value, not a total change"),
        ]
        if two:
            pool += [(say("start", j), f"{asked} is in the equation for {objs[i]}, not {objs[j]}")]
    mistakes = rng.sample(pool, 3) if two else pool
    if two:  # the two-model trap is the reason this is hard: always keep one wrong-object choice
        wrong_obj = pool[3]
        if wrong_obj not in mistakes:
            mistakes[0] = wrong_obj
    stem = f"{body} Which of the following is the best interpretation of {asked} in this context?"
    what = (
        f"the coefficient of {f['t']}, so it is the change in {f['qty']} for each {f['t_unit']}"
        if role == "rate"
        else f"the constant term, so it is the value when {f['t']} = 0, at the start"
    )
    steps = [
        f"In {eqs[i]}, {asked} is {what}.",
        f"So: {key}" + (f" (The {MINUS} sign means the amount decreases.)" if role == "rate" and sign < 0 else ""),
    ]
    item = MathItem("interpret_models", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"role": role, "obj": objs[i], "value": val}
    return item


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
    "reciprocal_ratio": ("easy", "medium"),
    "same_product": ("easy", "medium"),
    "no_solution_parens": ("easy", "medium"),
    "no_solution_k": ("medium", "hard"),
    "how_many_solutions": ("medium", "hard"),
    "infinite_two_constants": ("medium", "hard"),
    "fraction_decimal": LEVELS,
    "word_solve": LEVELS,
    "which_has_property": ("medium", "hard"),
    "interpret_models": ("medium", "hard"),
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
    "reciprocal_ratio": reciprocal_ratio,
    "same_product": same_product,
    "no_solution_parens": no_solution_parens,
    "no_solution_k": no_solution_k,
    "how_many_solutions": how_many_solutions,
    "infinite_two_constants": infinite_two_constants,
    "fraction_decimal": fraction_decimal,
    "word_solve": word_solve,
    "which_has_property": which_has_property,
    "interpret_models": interpret_models,
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


def story_numbers_ok(stem: str, given: list) -> list[str]:
    """Every planned number exactly once, in the planned order (so each keeps its role), and nothing else. Decimals
    such as $0.10 count as one number."""
    found = [Fraction(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", stem)]
    given = [Fraction(n) for n in given]
    problems = []
    if sorted(found) != sorted(given):
        problems.append(f"numbers in the story {found} must be exactly {given}")
    elif found != list(given):
        problems.append(
            f"the numbers appear in the order {found}; keep the planned order {list(given)} so each "
            f"number keeps its role"
        )
    for n in given:
        if n >= 1000 and n.denominator == 1 and num(n) not in stem:
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
        numbers = ", ".join(f.get("shown") or [num(n) for n in f["given"]])
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
