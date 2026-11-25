"""Meza Math: SAT Math questions built in code, so every answer is computed, never guessed.

Algebra first (Sebastian, Nov 25). Each question family picks numbers that come out even, computes the answer, and
builds the wrong answer choices from real student mistakes (the main one: solving for x when the question asks for
something else, the deck's golden rule "know exactly what the question is asking"). Word problems get a story: the
writer model wraps the fixed numbers in 4-5 sentences about a named person; code checks every number, and a blind
solver must reach the computed answer, or the plain template story is used instead.

    python meza_math.py run --count 8                          # free: no API calls, template stories
    python meza_math.py run --count 8 --ai-stories             # the writer model words the word problems (uses credits)
    python meza_math.py run --count 6 --families linear_rate   # one family only
    python meza_math.py run --count 20 --order ascending       # number choices least to greatest, as on the SAT
                                                               # (default: random, so the answer can be any letter)
"""

from __future__ import annotations

import argparse
import json
import math
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
CHOICE_ORDER = "random"  # number choices: "random" (any letter can be the answer) or "ascending" (as on the SAT)


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


def _share(cands: list, key: int, score: list) -> float:
    """The chance a student using this rule lands on the key: 1 / (number of choices tied for the rule's pick)."""
    best = [i for i, v in enumerate(score) if v == cands]
    return (key in best) / len(best) if best else 0.0


def _parts(t: str):
    import collections

    return collections.Counter(re.findall(r"\d+(?:[.,/]\d+)*|[A-Za-z]+|[=<>≤≥+()−-]", t))


def _slip(a, b) -> bool:
    near = {-a, a + 1, a - 1, a * 2, a / 2, a * 10, a / 10}
    if a:
        near |= {1 / a, -1 / a}
    return b in near


def tell_score(
    texts: list[str], key: int, values: list[Fraction] | None = None, parts: list | None = None, related=None
) -> float:
    """How well the four choices alone point to the key (Sebastian, Nov 25: like the Reading passage-free probe).
    Rules a student might use without reading the question: pick the choice that shares the most parts with the
    others (or the fewest), the longest or shortest, and for numbers the one most others are a simple slip away from
    (sign, flip, ±1, ×/÷2, ×/÷10), the one nearest the middle, the smallest, or the largest. Returns the best of
    these rules' chances of landing on the key (0.25 is pure guessing)."""
    P = parts or [_parts(t) for t in texts]
    shared = [sum(sum((P[i] & P[j]).values()) for j in range(4) if j != i) for i in range(4)]
    length = [len(t) for t in texts]
    rules = [(max(shared), shared), (min(shared), shared), (max(length), length), (min(length), length)]
    if values is not None:
        if related is None:
            related = [
                sum(_slip(values[i], values[j]) + _slip(values[j], values[i]) for j in range(4) if j != i)
                for i in range(4)
            ]
        mean = sum(values) / 4
        middle = [-abs(v - mean) for v in values]
        rules += [(max(related), related), (max(middle), middle), (min(values), values), (max(values), values)]
    return max(_share(best, key, score) for best, score in rules)


def grid2(render, right1, wrong1, why1: str, right2, wrong2, why2: str) -> list[tuple[str, str]]:
    """Wrong choices from two decisions, each with a right option and one tempting wrong one (Sebastian, Nov 25):
    the four choices are then equally alike (each shares one part with two others), so 'pick the choice most like
    the others' no longer finds the key. Returns the three wrong choices with their reasons."""
    return [
        (render(wrong1, right2), why1),
        (render(right1, wrong2), why2),
        (render(wrong1, wrong2), f"{why1}; also, {why2[0].lower() + why2[1:]}"),
    ]


def least_telling(key_text: str, options: list, show=str, key_value=None) -> list:
    """From the real wrong answers (most common first), the three that let the choices alone give the key away the
    least; ties go to the more common mistakes."""
    import itertools

    pool = options[:7]
    shown = [show(v) for v, _ in pool]
    parts = [_parts(t) for t in shown]
    key_parts = _parts(key_text)
    rel = None
    if key_value is not None:  # slips between every pair, worked out once
        vals = [key_value] + [v for v, _ in pool]
        rel = [[_slip(a, b) + _slip(b, a) for b in vals] for a in vals]
    best, best_rank = None, None
    for combo in itertools.combinations(range(len(pool)), 3):
        picked = [pool[i] for i in combo]
        texts = [key_text] + [shown[i] for i in combo]
        values = [key_value] + [v for v, _ in picked] if key_value is not None else None
        related = None
        if rel is not None:
            idx = [0] + [i + 1 for i in combo]
            related = [sum(rel[a][b] for b in idx if b != a) for a in idx]
        tell = tell_score(texts, 0, values, [key_parts] + [parts[i] for i in combo], related)
        rank = (round(tell, 6), sum(combo))
        if best_rank is None or rank < best_rank:
            best, best_rank = picked, rank
    return best


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
    figure: str = ""  # an SVG graph shown between the passage and the question

    def build_choices(self, rng: random.Random, used: dict[str, int] | None = None, order: str | None = None) -> None:
        """Four numeric choices: the answer plus three distinct real mistakes (the most common ones). A labeled near
        miss fills in only when a question has fewer than three real mistakes.

        order="random" (the default, Sebastian, Nov 25): the key goes to the letter used least so far in the batch
        (`used`, ties broken at random), and the wrong answers fill the other letters in random order, so the answer
        can land on any letter. order="ascending": numbers from least to greatest, as on the SAT (better for full
        mock tests and for plugging in the choices); the key's letter then follows from the values, and among the
        letters the real mistakes can produce, the least-used one is taken."""
        if self.fmt != "mcq":
            return
        if self.key_text:
            self._build_text_choices(rng, used)
            return
        order = order or CHOICE_ORDER
        # there, x itself is the trap
        whole = not (self.fractions_ok or self.family == "expression_value" and self.level == "medium")
        seen, real = {self.answer}, []
        for value, why in self.mistakes:
            ok_sign = value > 0 if self.positive_only else value != 0 or self.answer == 0
            if value not in seen and ok_sign and (value.denominator == 1 or not whole):
                seen.add(value)
                real.append((value, why))
        show = dec if self.decimals else num
        if order == "random":
            pool = real + self._double_slips(real, seen, whole)
            wrong = least_telling(show(self.answer), pool, show, self.answer) if len(pool) > 3 else list(pool[:3])
            step = max(1, abs(int(self.answer)) // 10)
            for v in (self.answer + o * step for n in range(1, 60) for o in (n, -n)):
                if len(wrong) >= 3:
                    break
                if v not in seen and (v > 0 or not self.positive_only):
                    seen.add(v)
                    wrong.append((v, "an arithmetic slip near the correct value"))
            used = used if used is not None else {}
            key = min("ABCD", key=lambda k: (used.get(k, 0), rng.random()))
            rng.shuffle(wrong)
            others = iter(wrong)
            self.choices = {k: show(self.answer) if k == key else show(next(others)[0]) for k in "ABCD"}
            self.key = key
            self.mistakes = wrong
            used[key] = used.get(key, 0) + 1
            return
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
        self.choices = {k: show(v) for k, (v, _) in zip("ABCD", values, strict=True)}
        self.key = next(k for k, (v, _) in zip("ABCD", values, strict=True) if v == self.answer)
        self.mistakes = list(wrong[:3])
        if used is not None:
            used[self.key] = used.get(self.key, 0) + 1

    def _double_slips(self, real: list, seen: set, whole: bool) -> list:
        """When a real mistake is a plain slip of the answer (the sign, the decimal point, a factor of 2, the
        fraction flipped), the same slip made on top of another real mistake is also a real (double) mistake. Adding
        those lets the wrong answers pair up with each other, so the answer stops being the one every other choice is
        a slip away from (Sebastian, Nov 25: answers must not be guessable from the choices alone)."""
        a = self.answer
        slips = []
        for v, why in real:
            for _slip, f in (
                ("sign", lambda t: -t),
                ("x10", lambda t: t * 10),
                ("/10", lambda t: t / 10),
                ("x2", lambda t: t * 2),
                ("/2", lambda t: t / 2),
                ("flip", lambda t: 1 / t if t else None),
            ):
                if a and v == f(a):
                    slips.append((f, why))
        extra = []
        for f, why_slip in slips:
            for v, why in real:
                if v == f(self.answer):
                    continue
                d = f(v)
                ok_sign = d is not None and (d > 0 if self.positive_only else d != 0)
                if (
                    ok_sign
                    and d not in seen
                    and (d.denominator == 1 or not whole)
                    and (terminating(d) or not self.decimals)
                ):
                    seen.add(d)
                    tail = why_slip[5:] if why_slip.startswith("this ") else why_slip
                    extra.append((d, f"{why}; it also {tail}"))
        return extra

    def _build_text_choices(self, rng: random.Random, used: dict[str, int] | None) -> None:
        """Equations or phrases: the key goes to the letter used least so far; of the real mistakes, the three that
        give the key away least from the choices alone (see tell_score) fill the other letters in random order."""
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
        wrong = least_telling(self.key_text, wrong) if len(wrong) > 3 else wrong
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
            "figure": self.figure,
            "stem": self.stem,
            "choices": self.choices,
            "answer": self.key if self.fmt == "mcq" else spr_answer(self.answer),
            "explanation": self.explanation(),
            "family": self.family,
        }


# ----------------------------------------------------------------------------------------------------------- graphs


def nice_axis(top: Fraction, lines: int = 10) -> tuple[Fraction, Fraction]:
    """A grid step from 1, 2, 2.5, or 5 times a power of 10, so the axis reaches `top` with at least `lines` grid
    lines; returns (step, axis maximum)."""
    for exp in range(-2, 7):
        for base in (1, 2, Fraction(5, 2), 5):
            step = Fraction(base) * Fraction(10) ** exp
            n = -(-top // step)  # grid lines needed to reach top
            if n <= max(lines, 12):
                n = max(n, lines)
                return step, step * n
    raise ValueError("axis too large")


def graph_svg(
    x_label: str,
    y_label: str,
    x_step,
    x_n: int,
    y_step,
    y_n: int,
    lines=(),
    points=(),
    label_every: tuple[int, int] = (1, 1),
    x_lo: int = 0,
    y_lo: int = 0,
    plane: bool = False,
    line_names=(),
    shade=(),
    dashed=(),
) -> str:
    """An SAT-style graph drawn in code, so every number is exact. A light grid (x_n grid steps right of 0 and x_lo
    left of it; likewise y), dark axes through the origin, numbered ticks, and solid lines given as (slope,
    intercept); dots given as (x, y). Real-world graphs (plane=False) start at 0 and carry axis titles with units;
    xy-plane graphs (plane=True) run into the negatives, with "x" and "y" at the ends of the axes and O at the
    origin. Two lines can carry names (line_names), shown in a key under the graph; the second line is dashed.
    Inequality graphs: shade=[(m, b, "above" or "below")] fills that side of y = mx + b in light gray, and the lines
    whose positions are in dashed are drawn dashed (a strict inequality)."""
    W, H = 460, 380 if not plane else 420
    L, B, T, R = (70, 55, 15, 15) if not plane else (30, 20, 20, 20)
    if line_names:  # a key under the graph: a solid line for the first name, a dashed line for the second
        H += 24
        B += 24
    pw, ph = W - L - R, H - B - T
    xs_, ys_ = Fraction(x_step), Fraction(y_step)
    x_min, x_max = -xs_ * x_lo, xs_ * x_n
    y_min, y_max = -ys_ * y_lo, ys_ * y_n

    def X(v) -> float:
        return L + float((Fraction(v) - x_min) / (x_max - x_min)) * pw

    def Y(v) -> float:
        return T + ph - float((Fraction(v) - y_min) / (y_max - y_min)) * ph

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Georgia, serif" font-size="12" role="img" aria-label="{y_label} versus {x_label}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for i in range(-x_lo, x_n + 1):
        xv = xs_ * i
        out.append(
            f'<line x1="{X(xv):.1f}" y1="{Y(y_min):.1f}" x2="{X(xv):.1f}" y2="{Y(y_max):.1f}" '
            f'stroke="#cfcfcf" stroke-width="1"/>'
        )
        if i % label_every[0] == 0 and (not plane or i and -x_lo < i < x_n):
            out.append(
                f'<text x="{X(xv) + (0 if not plane else -3):.1f}" y="{Y(0) + 16:.1f}" '
                f'text-anchor="{"middle" if not plane else "end"}">{dec(xv)}</text>'
            )
    for j in range(-y_lo, y_n + 1):
        yv = ys_ * j
        out.append(
            f'<line x1="{X(x_min):.1f}" y1="{Y(yv):.1f}" x2="{X(x_max):.1f}" y2="{Y(yv):.1f}" '
            f'stroke="#cfcfcf" stroke-width="1"/>'
        )
        if j % label_every[1] == 0 and j and (not plane or -y_lo < j < y_n):
            out.append(f'<text x="{X(0) - 6:.1f}" y="{Y(yv) + 4:.1f}" text-anchor="end">{dec(yv)}</text>')
    out.append(
        f'<line x1="{X(x_min):.1f}" y1="{Y(0):.1f}" x2="{X(x_max):.1f}" y2="{Y(0):.1f}" stroke="black" '
        f'stroke-width="1.5"/>'
    )
    out.append(
        f'<line x1="{X(0):.1f}" y1="{Y(y_min):.1f}" x2="{X(0):.1f}" y2="{Y(y_max):.1f}" stroke="black" '
        f'stroke-width="1.5"/>'
    )
    for m, b, side in shade:  # clip the grid rectangle to the half-plane (Sutherland-Hodgman, one edge)
        m, b = Fraction(m), Fraction(b)
        box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

        def inside(pt, m=m, b=b, side=side) -> bool:
            d = pt[1] - (m * pt[0] + b)
            return d >= 0 if side == "above" else d <= 0

        poly = []
        for i, cur in enumerate(box):
            prev = box[i - 1]
            if inside(cur) != inside(prev):  # the edge crosses the line
                (x1, y1), (x2, y2) = prev, cur
                if x1 == x2:
                    cross = (x1, m * x1 + b)
                else:
                    cross = ((y1 - b) / m, y1) if m else (x1, b)
                poly.append(cross)
            if inside(cur):
                poly.append(cur)
        if len(poly) >= 3:
            pts_ = " ".join(f"{X(px):.1f},{Y(py):.1f}" for px, py in poly)
            out.append(f'<polygon points="{pts_}" fill="#8c8c8c" fill-opacity="0.35" stroke="none"/>')
    for n, (m, b) in enumerate(lines):  # clip y = mx + b to the grid
        m, b = Fraction(m), Fraction(b)
        pts = []
        for xv in (x_min, x_max):
            yv = m * xv + b
            if y_min <= yv <= y_max:
                pts.append((xv, yv))
        if m:
            for yv in (y_min, y_max):
                xv = (yv - b) / m
                if x_min <= xv <= x_max:
                    pts.append((xv, yv))
        pts = sorted(set(pts))
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            dash = ' stroke-dasharray="7 5"' if (n == 1 and line_names) or n in dashed else ""
            out.append(
                f'<line x1="{X(x1):.1f}" y1="{Y(y1):.1f}" x2="{X(x2):.1f}" y2="{Y(y2):.1f}" stroke="black" '
                f'stroke-width="2.2"{dash}/>'
            )
    for xv, yv in points:
        out.append(f'<circle cx="{X(xv):.1f}" cy="{Y(yv):.1f}" r="3.5" fill="black"/>')
    for n, name in enumerate(line_names):
        cx = W / 2 - 110 + n * 140
        dash = ' stroke-dasharray="7 5"' if n == 1 else ""
        out.append(
            f'<line x1="{cx:.1f}" y1="{H - 14}" x2="{cx + 34:.1f}" y2="{H - 14}" stroke="black" '
            f'stroke-width="2.2"{dash}/>'
        )
        out.append(f'<text x="{cx + 42:.1f}" y="{H - 10}" font-weight="bold">{name}</text>')
    if plane:
        out.append(
            f'<text x="{X(x_max) - 4:.1f}" y="{Y(0) - 6:.1f}" text-anchor="end" font-style="italic" '
            f'font-size="14">x</text>'
        )
        out.append(f'<text x="{X(0) + 6:.1f}" y="{Y(y_max) + 12:.1f}" font-style="italic" font-size="14">y</text>')
        out.append(f'<text x="{X(0) - 5:.1f}" y="{Y(0) + 15:.1f}" text-anchor="end" font-style="italic">O</text>')
    else:
        out.append(
            f'<text x="{L + pw / 2:.1f}" y="{T + ph + 45}" text-anchor="middle" font-style="italic">{x_label}</text>'
        )
        out.append(
            f'<text transform="translate(16 {T + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
            f'font-style="italic">{y_label}</text>'
        )
    out.append("</svg>")
    return "".join(out)


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
    M = rng.choice([v for v in range(15, 301, 5) if v != D])  # equal amounts would make the swapped choice true
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
            (T / (r + F), f"this adds the {money(F)} to the cost of each unit"),
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
        item = MathItem("word_solve", SKILL, level, fmt, f"{setup} {question}", Fraction(ans), steps, mistakes)
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
    mistakes = pool
    if two:  # a 2x2 of (this object or the other) x (the right role or its twin), so no choice is the "middle" one
        twin = "more" if role == "rate" else "final"
        twin_why = (
            f"this compares the two {'rates' if sign > 0 else 'models'}; {asked} is {objs[i]}'s own rate"
            if role == "rate"
            else f"{asked} is the value at the start ({f['t']} = 0), not at the end"
        )

        def cell(obj: int, kind: str) -> str:
            return say(kind, obj, 1 - obj) if kind == "more" else say(kind, obj)

        mistakes = grid2(cell, i, j, f"{asked} is in the equation for {objs[i]}, not {objs[j]}", role, twin, twin_why)
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


def coef(m, var: str = "x", decimal: bool = False) -> str:
    """3 -> '3x', -1 -> '−x', 3/4 -> '(3/4)x', 0.5 -> '0.5x' (decimal=True)."""
    m = Fraction(m)
    if m == 1:
        return var
    if m == -1:
        return MINUS + var
    if m.denominator == 1 or decimal:
        return f"{dec(m) if decimal else num(m)}{var}"
    return f"{MINUS if m < 0 else ''}({num(abs(m))}){var}"


def linf(m, b, var: str = "x", decimal: bool = False) -> str:
    """'18x + 108', '−0.05x + 13', '(5/9)x − 17' (b may be 0)."""
    out = coef(m, var, decimal)
    b = Fraction(b)
    show = dec if decimal else num
    if b > 0:
        out += f" + {show(b)}"
    elif b < 0:
        out += f" {MINUS} {show(-b)}"
    return out


def unit(n, one: str, many: str) -> str:
    return f"{dec(n)} {one if Fraction(n) == 1 else many}"


GRAPH_FRAMES = [
    dict(
        intro="A delivery van's fuel tank was filled before a trip. The graph shows the amount of fuel y, in gallons, "
        "left in the tank after the van had been driven x miles.",
        x="Distance driven (miles)",
        y="Fuel in tank (gallons)",
        sign=-1,
        xs=[10, 20, 25, 50],
        ys=[1, 2],
        xname="the van had been driven",
        xu=("mile", "miles"),
        per="mile driven",
        yq="the amount of fuel left in the tank",
        yu=("gallon", "gallons"),
        start="the van started the trip with",
    ),
    dict(
        intro="A swimming pool is being filled with water at a constant rate. The graph shows the amount of water y, "
        "in gallons, in the pool x minutes after filling began.",
        x="Time (minutes)",
        y="Water in pool (gallons)",
        sign=1,
        xs=[5, 10, 20],
        ys=[100, 200, 250, 500],
        xname="filling had gone on for",
        xu=("minute", "minutes"),
        per="minute",
        yq="the amount of water in the pool",
        yu=("gallon", "gallons"),
        start="the pool held",
    ),
    dict(
        intro="A phone's battery drains at a steady rate during a long video call. The graph shows the battery charge "
        "y, as a percent of a full charge, x minutes after the call began.",
        x="Time (minutes)",
        y="Battery charge (percent)",
        sign=-1,
        xs=[10, 20, 25],
        ys=[5, 10],
        xname="the call had lasted",
        xu=("minute", "minutes"),
        per="minute",
        yq="the battery charge",
        yu=("percentage point", "percentage points"),
        start="the battery charge was",
    ),
    dict(
        intro="A tree nursery measured the height of a young maple tree once a month. The graph shows a linear model "
        "of the tree's height y, in inches, x months after the first measurement.",
        x="Time (months)",
        y="Height (inches)",
        sign=1,
        xs=[1, 2],
        ys=[2, 4, 5],
        xname="the first measurement had been taken",
        xu=("month", "months"),
        per="month",
        yq="the tree's height",
        yu=("inch", "inches"),
        start="the tree's height was",
    ),
    dict(
        intro="A small plane descends at a constant rate as it approaches an airport. The graph shows the plane's "
        "altitude y, in feet, x minutes after the descent began.",
        x="Time (minutes)",
        y="Altitude (feet)",
        sign=-1,
        xs=[1, 2],
        ys=[500, 1000],
        xname="the descent had lasted",
        xu=("minute", "minutes"),
        per="minute",
        yq="the plane's altitude",
        yu=("foot", "feet"),
        start="the plane's altitude was",
    ),
    dict(
        intro="A kayak rental shop charges a fixed fee plus an hourly rate. The graph shows the total cost y, in "
        "dollars, of renting a kayak for x hours.",
        x="Rental time (hours)",
        y="Total cost (dollars)",
        sign=1,
        xs=[1],
        ys=[5, 10, 20],
        xname="the kayak had been rented for",
        xu=("hour", "hours"),
        per="hour",
        yq="the total cost",
        yu=("dollar", "dollars"),
        start="the fixed fee was",
    ),
]


def _graph_line(rng: random.Random, f: dict):
    """A line through grid points: slope = (dy grid steps)/(dx grid steps), starting on a grid line, staying on the
    grid. Returns (xs, x_n, ys, y_n, m, b, dx)."""
    while True:
        xs, ys = Fraction(rng.choice(f["xs"])), Fraction(rng.choice(f["ys"]))
        x_n, y_n = rng.choice([10, 12]), rng.choice([10, 12, 14])
        dx, dy = rng.choice([1, 2, 4, 5]), rng.choice([1, 2, 3])
        m = f["sign"] * dy * ys / (dx * xs)
        if f["sign"] > 0:
            b = ys * rng.randint(0 if f["sign"] > 0 and "cost" not in f["y"] else 1, 3)
            end = b + m * xs * x_n
            ok = end <= ys * y_n and end >= ys * (y_n - 4)
        else:
            b = ys * rng.randint(y_n - 3, y_n)
            end = b + m * xs * x_n
            ok = end >= 0 and end <= ys * 4 or (end < 0 and (-b / m) >= xs * (x_n - 2))
        if ok and (xs * x_n) / (dx * xs) >= 2:
            return xs, x_n, ys, y_n, m, b, dx


def graph_estimate(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A graph of a real-world linear relationship. 'Based on the graph, what is the estimated amount of fuel left
    when the van has been driven 160 miles?'  Hard asks the other way: 'what is the estimated distance when 5 gallons
    are left?'  Traps: reading one grid line too high or low, the starting value, reading the wrong axis."""
    f = rng.choice(GRAPH_FRAMES)
    xs, x_n, ys, y_n, m, b, dx = _graph_line(rng, f)
    fig = graph_svg(f["x"], f["y"], xs, x_n, ys, y_n, lines=[(m, b)], label_every=(2 if x_n > 10 else 1, 2))
    lattice = [
        xs * dx * j for j in range(1, x_n // dx + 1) if 0 <= m * xs * dx * j + b <= ys * y_n and xs * dx * j <= xs * x_n
    ]
    x0 = rng.choice(lattice[: max(1, len(lattice) - 1)] or lattice)
    y0 = m * x0 + b
    if level != "hard":
        stem = f"{f['intro']} Based on the graph, what is the estimated value of y when x = {dec(x0)}?"
        ans = y0
        mistakes = [
            (y0 + ys, "this reads the grid line just above the line"),
            (y0 - ys, "this reads the grid line just below the line"),
            (b, f"this is the value of y when x = 0 ({f['start']} this)"),
            (m * (x0 + xs) + b, "this reads the line one vertical grid line too far right"),
            (m * (x0 - xs) + b, "this reads the line one vertical grid line too far left"),
        ]
        steps = [
            f"Find x = {dec(x0)} on the horizontal axis and go straight up to the line.",
            f"The line there is at y = {dec(y0)} on the vertical axis.",
        ]
    else:
        stem = f"{f['intro']} Based on the graph, what is the estimated value of x when y = {dec(y0)}?"
        ans = x0
        mistakes = [
            (x0 + xs, "this reads one vertical grid line too far right"),
            (x0 - xs, "this reads one vertical grid line too far left"),
            (y0, "this is the value of y, not x"),
            ((y0 + ys - b) / m, "this starts from the grid line above y = " + dec(y0)),
        ]
        steps = [
            f"Find y = {dec(y0)} on the vertical axis and go across to the line.",
            f"Then go straight down: the line is at x = {dec(x0)}.",
        ]
    mistakes = [(v, w) for v, w in mistakes if v >= 0 and terminating(v)]
    item = MathItem(
        "graph_estimate",
        FUNC_SKILL,
        level,
        fmt,
        stem,
        ans,
        steps,
        mistakes,
        figure=fig,
        fractions_ok=True,
        decimals=True,
    )
    item.truth = {"m": m, "b": b, "x0": x0, "y0": y0}
    return item


def graph_slope(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'According to the graph, what is the change in the amount of fuel for each additional mile driven?'  Hard asks
    for each additional block of units ('each additional 40 miles'). Choices are phrases: the right size and
    direction; the opposite direction; the rate flipped; the starting value; the total change shown on the graph."""
    f = rng.choice(GRAPH_FRAMES)
    xs, x_n, ys, y_n, m, b, dx = _graph_line(rng, f)
    fig = graph_svg(f["x"], f["y"], xs, x_n, ys, y_n, lines=[(m, b)], label_every=(2 if x_n > 10 else 1, 2))
    k = Fraction(1) if level != "hard" else xs * dx
    change = abs(m) * k
    up, down = "increases", "decreases"
    word = up if m > 0 else down
    other = down if m > 0 else up
    per = f["per"] if k == 1 else f"{dec(k)} {f['xu'][1]}" + (" driven" if "driven" in f["per"] else "")
    stem = f"{f['intro']} According to the graph, what is the change in {f['yq']} for each additional {per}?"

    def say(w: str, n) -> str:
        return f"{_cap(f['yq'])} {w} by {unit(n, *f['yu'])}."

    key = say(word, change)
    x_end = min(xs * x_n, (-b / m) if m < 0 else xs * x_n)
    amounts = []  # the tempting wrong amounts, most common first
    if terminating(1 / abs(m)):
        amounts.append((1 / abs(m), f"this flips the rate: {dec(1 / abs(m))} {f['xu'][1]} for each {f['yu'][0]}"))
    if xs * dx != k:
        amounts.append(
            (abs(m) * xs * dx, f"this is the change for each {dec(xs * dx)} {f['xu'][1]}, not for each {per}")
        )
    if k != 1:
        amounts.append((abs(m), f"this is the change for each single {f['xu'][0]}, not for each {per}"))
    amounts += [
        (abs(m * x_end), f"this is the change over the whole graph, not for each {per}"),
        (abs(b), f"{dec(b)} is the value when x = 0, not the change for each {f['xu'][0]}"),
        (change * 2, f"this doubles the change for each {per}"),
    ]
    amounts = [(v, w) for v, w in amounts if v != change and v != 0]
    wrong_amount, why_amount = amounts[0] if rng.random() < 0.7 or len(amounts) == 1 else amounts[1]
    mistakes = grid2(
        say,
        word,
        other,
        f"the line goes {'up' if m > 0 else 'down'} from left to right, so it {word}",
        change,
        wrong_amount,
        why_amount,
    )
    steps = [
        f"Pick two points on the line where it crosses grid lines, such as (0, {dec(b)}) and "
        f"({dec(xs * dx)}, {dec(b + m * xs * dx)}).",
        f"Slope = ({dec(b + m * xs * dx)} {MINUS} {dec(b)}) ÷ {dec(xs * dx)} = {dec(m)}"
        + (f"; for {dec(k)} {f['xu'][1]}, the change is {dec(m)} × {dec(k)} = {dec(m * k)}." if k != 1 else "."),
        f"So {key[0].lower() + key[1:]}",
    ]
    item = MathItem(
        "graph_slope", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, figure=fig
    )
    item.truth = {"m": m, "b": b, "k": k}
    return item


RATE_JOBS = [
    dict(
        text="{n} earns ${r} per hour babysitting. On Saturday, {n} babysat for {a} hours, and on Sunday, {n} babysat "
        "for x hours.",
        total="the total amount, in dollars, {n} earned babysitting on Saturday and Sunday",
        rates=[12, 15, 16, 18, 20, 22, 25],
    ),
    dict(
        text="{n} reads at a rate of {r} pages per hour. On Monday, {n} read for {a} hours, and on Tuesday, {n} read "
        "for x hours.",
        total="the total number of pages {n} read on Monday and Tuesday",
        rates=[20, 24, 30, 35, 40, 45],
    ),
    dict(
        text="A truck driver drives at an average speed of {r} miles per hour. On the first day of a trip, the driver "
        "drove for {a} hours, and on the second day, the driver drove for x hours.",
        total="the total distance, in miles, the driver drove on the two days",
        rates=[50, 55, 60, 62, 65],
    ),
    dict(
        text="A printer prints {r} pages per minute. On Friday, it printed for {a} minutes, and on Saturday, it "
        "printed for x minutes.",
        total="the total number of pages printed on Friday and Saturday",
        rates=[18, 20, 24, 30, 36, 40],
    ),
]


def rate_total_function(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'... earns $18 per hour ... babysat for 6 hours on Saturday and x hours on Sunday. Which function f gives the
    total amount earned on the two days?'  f(x) = 18x + 108 (or 18(x + 6), the form chosen at random). Traps add the
    hours instead of the earnings, multiply the hours into the rate, or leave out the first day."""
    n = rng.choice(NAMES)
    while True:  # two-digit numbers throughout, so no choice is longer than the others
        f = rng.choice(RATE_JOBS)
        r = rng.choice(f["rates"])
        a = rng.randint(2, 9)
        if 10 <= r * a <= 99 and r + a <= 99:
            break
    stem = f"{f['text'].format(n=n, r=r, a=a)} Which function f gives {f['total'].format(n=n)}?"
    key = f"f(x) = {r}x + {num(r * a)}"
    mistakes = grid2(
        lambda c, k_: f"f(x) = {c}x + {k_}",
        r,
        r + a,
        f"this adds the {a} hours to the rate",
        r * a,
        r + a,
        f"for the first day this adds {r} and {a} instead of multiplying them",
    )
    steps = [
        f"The first day gives {r} × {a} = {num(r * a)}; the second gives {r}x.",
        f"Together: f(x) = {r}x + {num(r * a)}.",
    ]
    item = MathItem("rate_total_function", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"r": r, "a": a}
    return item


INTERCEPT_FRAMES = [
    dict(
        desc="the amount of water, in gallons, in a tank x minutes after the tank began draining",
        start="The tank held {k} gallons of water when it began draining.",
        end="The tank is empty {k} minutes after it began draining.",
        rate="The amount of water in the tank decreases by {k} gallons each minute.",
        slopes=[10, 12, 15, 20, 25],
        xs=range(12, 61),
    ),
    dict(
        desc="the height, in centimeters, of a candle x hours after it was lit",
        start="The candle was {k} centimeters tall when it was lit.",
        end="The candle burns out completely {k} hours after it was lit.",
        rate="The candle's height decreases by {k} centimeters each hour.",
        slopes=[Fraction(1, 2), Fraction(3, 4), 1, Fraction(5, 4), 2],
        xs=range(8, 41),
    ),
    dict(
        desc="the balance, in dollars, on a gift card x weeks after it was first used, if the same amount is "
        "spent each week",
        start="The gift card had a balance of ${k} when it was first used.",
        end="The gift card is used up {k} weeks after it was first used.",
        rate="The balance decreases by ${k} each week.",
        slopes=[5, 8, 10, 12, 15, 20],
        xs=range(4, 16),
    ),
    dict(
        desc="a phone's battery charge, as a percent of a full charge, x hours after the phone was unplugged",
        start="The battery charge was {k}% when the phone was unplugged.",
        end="The battery is fully drained {k} hours after the phone was unplugged.",
        rate="The battery charge decreases by {k} percentage points each hour.",
        slopes=[4, 5, 8, 10],
        xs=range(8, 26),
        max_b=100,  # a battery starts at 100% or less
    ),
]


def intercept_meaning(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'The function f(x) = −25x + 600 gives the amount of water in a tank x minutes after it began draining. What is
    the best interpretation of the x-intercept of the graph of y = f(x) in this context?'  Traps: the other
    intercept (a true statement, but not the one asked), the right number in the wrong role, the rate."""
    f = rng.choice(INTERCEPT_FRAMES)
    while True:
        m = Fraction(rng.choice(f["slopes"]))
        x0 = rng.choice(list(f["xs"]))
        b = m * x0
        if b.denominator == 1 and b != x0 and b <= f.get("max_b", b):
            break
    which = rng.choice(["x", "y"]) if level == "hard" else "y"
    decimal = m.denominator != 1
    eq = f"f(x) = {linf(-m, b, decimal=decimal)}"
    stem = (
        f"The function {eq} gives {f['desc']}. What is the best interpretation of the {which}-intercept of the "
        f"graph of y = f(x) in the xy-plane in this context?"
    )
    say = {k: f[k] for k in ("start", "end", "rate")}
    if which == "y":
        key = say["start"].format(k=num(b))
        mistakes = [
            (say["end"].format(k=num(b)), f"{num(b)} is the starting amount (x = 0), not a time"),
            (say["end"].format(k=num(x0)), "this is true, but it describes the x-intercept"),
            (say["rate"].format(k=num(b)), f"{num(b)} is the starting amount, not the change each unit"),
        ]
        steps = [f"The y-intercept is where x = 0: f(0) = {num(b)}.", f"So: {key}"]
    else:
        key = say["end"].format(k=num(x0))
        mistakes = [
            (say["start"].format(k=num(x0)), f"{num(x0)} is a time (the x-value), not a starting amount"),
            (say["start"].format(k=num(b)), "this is true, but it describes the y-intercept"),
            (say["rate"].format(k=num(x0)), f"{num(x0)} is the x-value where f(x) = 0, not a rate"),
        ]
        steps = [
            f"The x-intercept is where f(x) = 0: {linf(-m, b, decimal=decimal)} = 0, so x = {num(x0)}.",
            f"So: {key}",
        ]
    item = MathItem("intercept_meaning", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": -m, "b": b, "x0": x0, "which": which}
    return item


def two_points_function(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'For the linear function f, f(5) = 7 and f(2) = 19. Which equation defines f?'  Traps: the slope with the wrong
    sign, a given output used as the y-intercept, the slope flipped (run over rise). Hard gives the two values in a
    table-free sentence with negative inputs."""
    letter = rng.choice(["f", "g", "h", "p"])
    while True:
        m = rng.choice(nz(-6, 6))
        b = rng.choice(nz(-20, 30))
        lo, hi = (-6, 9) if level == "hard" else (1, 9)
        x1, x2 = rng.sample([v for v in range(lo, hi + 1) if v], 2)
        y1, y2 = m * x1 + b, m * x2 + b
        if len({y1, y2, b, -b}) == 4 and (y1 - b) and abs(m) != 1:
            break
    stem = (
        f"For the linear function {letter}, {letter}({num(x1)}) = {num(y1)} and {letter}({num(x2)}) = "
        f"{num(y2)}. Which equation defines {letter}?"
    )
    key = f"{letter}(x) = {linf(m, b)}"
    mistakes = [
        (f"{letter}(x) = {linf(-m, y1 + m * x1)}", "this uses the slope with the wrong sign"),
        (f"{letter}(x) = {linf(m, y1)}", f"this uses {num(y1)}, an output, as the y-intercept"),
        (f"{letter}(x) = {linf(Fraction(1, m), b)}", "this flips the slope (change in x over change in y)"),
        (f"{letter}(x) = {linf(m, y2)}", f"this uses {num(y2)}, an output, as the y-intercept"),
    ]
    steps = [
        f"Slope = ({num(y2)} {MINUS} {num(y1)}) / ({num(x2)} {MINUS} {num(x1)}) = {num(m)}.",
        f"Use one point: {num(y1)} = {num(m)}({num(x1)}) + b, so b = {num(b)}.",
        f"So {key}.",
    ]
    item = MathItem("two_points_function", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b, "letter": letter}
    return item


def intercept_from_slope(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'In the xy-plane, a line has a slope of 3 and passes through (0, −12). What is the x-coordinate of the
    x-intercept?'  Or the other way: slope and x-intercept given, find the y-coordinate of the y-intercept. Hard
    uses fraction slopes."""
    while True:
        m = (
            Fraction(rng.choice(nz(-8, 8)))
            if level != "hard"
            else Fraction(rng.choice(nz(-5, 5)), rng.choice([2, 3, 4]))
        )
        a = Fraction(rng.choice(nz(-12, 12)))  # x-intercept
        b = -m * a  # y-intercept
        if b.denominator == 1 and b and m.denominator <= 4 and a != b and abs(m) != 1:
            break
    if rng.random() < 0.5:
        stem = (
            f"In the xy-plane, a line has a slope of {num(m)} and a y-intercept of (0, {num(b)}). What is the "
            f"x-coordinate of the x-intercept of the line?"
        )
        ans = a
        mistakes = [
            (-a, "this has the right size but the wrong sign"),
            (m / b, "this divides the slope by the y-intercept"),
            (b, "this is the y-intercept"),
            (-b * m, "this multiplies by the slope instead of dividing"),
        ]
        steps = [
            f"The line is y = {linf(m, b)}. At the x-intercept, y = 0: 0 = {linf(m, b)}.",
            f"So x = {MINUS}({num(b)}) ÷ {num(m)} = {num(a)}.",
        ]
    else:
        stem = (
            f"In the xy-plane, a line has a slope of {num(m)} and an x-intercept of ({num(a)}, 0). What is the "
            f"y-coordinate of the y-intercept of the line?"
        )
        ans = b
        mistakes = [
            (-b, "this has the right size but the wrong sign"),
            (a, "this is the x-intercept"),
            (a / m, "this divides the x-intercept by the slope instead of multiplying"),
            (m * a + m, "this adds the slope again"),
        ]
        steps = [
            f"Use y = {coef(m)} + b with the point ({num(a)}, 0): 0 = ({num(m)})({num(a)}) + b.",
            f"So b = {num(b)}.",
        ]
    item = MathItem(
        "intercept_from_slope",
        FUNC_SKILL,
        level,
        fmt,
        stem,
        ans,
        steps,
        mistakes,
        positive_only=False,
        fractions_ok=True,
    )
    item.truth = {"m": m, "a": a, "b": b}
    return item


MONTHLY_PLANS = [
    dict(
        what="A gym charges a one-time sign-up fee of ${a} plus ${r} per month.",
        c="the total cost, in dollars, of a gym membership for m months",
        fees=range(25, 151, 5),
        rates=range(20, 61, 5),
    ),
    dict(
        what="A streaming service charges a one-time setup fee of ${a} plus ${r} per month.",
        c="the total cost, in dollars, of the service for m months",
        fees=range(10, 51, 5),
        rates=range(8, 21),
    ),
    dict(
        what="A storage company charges a one-time move-in fee of ${a} plus ${r} per month to rent a storage unit.",
        c="the total cost, in dollars, of renting the unit for m months",
        fees=range(20, 101, 10),
        rates=range(45, 151, 5),
    ),
]
HOURLY_PLANS = [
    dict(
        what="A shop rents kayaks for ${a} per hour for the first {h} hours and ${r} per hour after that.",
        thing="renting a kayak",
    ),
    dict(
        what="A tutor charges ${a} per hour for the first {h} hours of tutoring each week and ${r} per hour for any "
        "additional hours that week.",
        thing="a week of tutoring",
    ),
    dict(
        what="A parking garage charges ${a} per hour for the first {h} hours and ${r} per hour after that.",
        thing="parking a car in the garage",
    ),
    dict(
        what="A bike rental shop charges ${a} per hour for the first {h} hours and ${r} per hour after that.",
        thing="renting a bike",
    ),
]


def cost_function(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: a one-time fee plus a monthly charge, 'Which function gives the cost C(m)?' Medium: ${a} per hour for the
    first 2 (or 3) hours and ${r} per hour after that, where x is the number of hours AFTER the first 2, so the first
    hours' cost goes into the y-intercept: C(x) = rx + 2a. Hard: the same, but x is the TOTAL number of hours (x ≥ 2),
    so C(x) = r(x − 2) + 2a, shown simplified."""
    if level == "easy":
        f = rng.choice(MONTHLY_PLANS)
        a, r = rng.choice(list(f["fees"])), rng.choice(list(f["rates"]))
        if a == r:
            a += 5
        stem = f"{f['what'].format(a=a, r=r)} Which function C gives {f['c']}?"
        key = f"C(m) = {r}m + {a}"
        mistakes = [
            (f"C(m) = {a}m + {r}", "this swaps the one-time fee and the monthly charge"),
            (f"C(m) = {r + a}m", "this charges the one-time fee every month"),
            (f"C(m) = {r}m {MINUS} {a}", "this subtracts the fee instead of adding it"),
            (f"C(m) = {r}(m + {a})", f"this multiplies the fee by ${r}"),
        ]
        steps = [f"Each month costs ${r}, so m months cost {r}m. The ${a} fee is paid once.", f"So {key}."]
        truth = {"fee": a, "rate": r}
    else:
        f = rng.choice(HOURLY_PLANS)
        h = rng.choice([2, 2, 3])
        while True:
            a, r = rng.randint(4, 15) * 5, rng.randint(2, 12) * 5
            if a != r and h * a != r:
                break
        first = h * a
        if level == "medium":
            q = (
                f"Which function C gives the cost, in dollars, of {f['thing']} for x hours more than the first {h} "
                f"hours?"
            )
            key = f"C(x) = {r}x + {first}"
            mistakes = [
                (f"C(x) = {r}x + {a}", f"this adds the cost of only one of the first {h} hours"),
                (f"C(x) = {a}x + {h * r}", "this swaps the two hourly rates"),
                (f"C(x) = {r}x + {r * h}", f"this charges the first {h} hours at the later rate"),
                (
                    f"C(x) = {r}(x {MINUS} {h}) + {first}",
                    f"x already counts only the hours after the first {h}, so nothing more is subtracted",
                ),
            ]
            steps = [
                f"The first {h} hours always cost {h} × ${a} = ${first}; that is the constant term.",
                f"Each of the x extra hours costs ${r}, so {key}.",
            ]
        else:
            q = f"Which function C gives the cost, in dollars, of {f['thing']} for x hours, where x ≥ {h}?"
            key = f"C(x) = {linf(r, first - r * h)}"
            mistakes = [
                (f"C(x) = {r}x + {first}", f"this charges ${r} for all x hours as well as the first {h}"),
                (f"C(x) = {linf(r, first + r * h)}", f"this uses x + {h} instead of x {MINUS} {h}"),
                (
                    f"C(x) = {linf(a, r * h - a * h)}" if r * h != a * h else f"C(x) = {a}x",
                    "this swaps the two hourly rates",
                ),
                (f"C(x) = {r}x", f"this leaves out the first {h} hours' cost"),
            ]
            steps = [
                f"The first {h} hours cost {h} × ${a} = ${first}; the other x {MINUS} {h} hours cost ${r} each.",
                f"C(x) = {r}(x {MINUS} {h}) + {first} = {linf(r, first - r * h)}.",
            ]
        stem = f"{f['what'].format(a=a, r=r, h=h)} {q}"
        truth = {"a": a, "r": r, "h": h}
    item = MathItem("cost_function", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = truth
    return item


FORMULAS = [
    dict(
        text="The function F(c) = (9/5)c + 32 gives the temperature, in degrees Fahrenheit, for a temperature of c "
        "degrees Celsius.",
        ask="If a temperature increases by {d} degrees Celsius, by how many degrees Fahrenheit does it increase?",
        m=Fraction(9, 5),
        b=Fraction(32),
        shown=Fraction(32),
        deltas=[5, 10, 15, 20, 25, 30, 40],
        level="medium",
    ),
    dict(
        text="A print shop charges C(n) = 0.15n + 12.5 dollars to print n copies of a flyer.",
        ask="If an order increases by {d} copies, by how many dollars does the cost increase?",
        m=Fraction(3, 20),
        b=Fraction(25, 2),
        shown=Fraction(25, 2),
        deltas=[20, 40, 60, 100, 200],
        level="medium",
    ),
    dict(
        text="The function K(f) = (5/9)(f − 32) + 273.15 gives the temperature, in kelvins, for a temperature of f "
        "degrees Fahrenheit.",
        ask="If a temperature increases by {d} degrees Fahrenheit, by how many kelvins does it increase?",
        m=Fraction(5, 9),
        b=Fraction(27315, 100) - Fraction(160, 9),
        shown=Fraction(27315, 100),
        deltas=[9, 18, 27, 36, 45, 54],
        level="hard",
    ),
    dict(
        text="The function P(d) = 0.43d + 14.7 gives the approximate pressure, in pounds per square inch, at a depth "
        "of d feet below the surface of the ocean.",
        ask="If a diver descends {d} feet deeper, by how many pounds per square inch does the pressure increase?",
        m=Fraction(43, 100),
        b=Fraction(147, 10),
        shown=Fraction(147, 10),
        deltas=[10, 20, 30, 50, 100],
        level="hard",
    ),
]


def function_change(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'K(f) = (5/9)(f − 32) + 273.15 ... If a temperature increases by 18 degrees Fahrenheit, by how many kelvins does
    it increase?'  The change is the slope times the change in input; the constants cancel. Traps: the input change
    itself, the slope flipped, the constant term added back in, one output value instead of the change."""
    f = rng.choice([x for x in FORMULAS if x["level"] == level])
    d = rng.choice(f["deltas"])
    m = f["m"]
    ans = m * d
    stem = f"{f['text']} {f['ask'].format(d=d)}"
    mistakes = [
        (Fraction(d), "this is the change in the input, not in the output"),
        (d / m, "this divides by the rate instead of multiplying (the rate is flipped)"),
        (ans + f["shown"], f"this adds {dec(f['shown'])}, a constant that cancels when you subtract two outputs"),
        (d + f["shown"], f"this adds {dec(f['shown'])} to the change in the input"),
        (m, "this is the change for an increase of 1, not of " + str(d)),
        (Fraction(d, m.denominator), f"this divides by {m.denominator} but leaves out the {m.numerator} in the rate"),
        (Fraction(d * m.numerator), f"this multiplies by {m.numerator} but does not divide by {m.denominator}"),
    ]
    mistakes = [(v, w) for v, w in mistakes if not (m.numerator == 1 and "leaves out the 1" in w) and v != ans]
    mistakes = [(v, w) for v, w in mistakes if terminating(v)]
    steps = [
        f"The output changes by the slope, {dec(m)}, for each 1 the input increases; the constant terms cancel.",
        f"So the change is {dec(m)} × {d} = {dec(ans)}.",
    ]
    item = MathItem(
        "function_change", FUNC_SKILL, level, fmt, stem, ans, steps, mistakes, fractions_ok=True, decimals=True
    )
    item.truth = {"m": m, "d": d}
    return item


TABLE_FRAMES = [
    dict(
        title="Hours studied and test scores",
        x="Hours studied",
        y="Test score (points)",
        xd="the number of hours a student studied",
        yd="the student's test score, in points",
        xstart=[1, 2],
        xstep=[1, 2],
        m=range(4, 10),
        b=range(40, 61),
    ),
    dict(
        title="Oven temperature while preheating",
        x="Time (minutes)",
        y="Temperature (°F)",
        xd="the time, in minutes, since the oven was turned on",
        yd="the oven's temperature, in degrees Fahrenheit",
        xstart=[2, 3],
        xstep=[2, 3],
        m=range(15, 41, 5),
        b=range(70, 81),
    ),
    dict(
        title="Cost of concert tickets",
        x="Number of tickets",
        y="Total cost (dollars)",
        xd="the number of tickets bought",
        yd="the total cost, in dollars, including a one-time service fee",
        xstart=[1, 2],
        xstep=[1, 2],
        m=range(18, 46),
        b=range(5, 16),
    ),
    dict(
        title="Water in a storage tank",
        x="Time (days)",
        y="Water in tank (liters)",
        xd="the number of days since the tank was filled",
        yd="the amount of water in the tank, in liters",
        xstart=[2, 3],
        xstep=[2, 3],
        m=range(-30, -9, 5),
        b=range(400, 601, 20),
    ),
]


def md_table(title: str, xh: str, yh: str, rows: list) -> str:
    lines = [title, f"| {xh} | {yh} |", "|---|---|"]
    lines += [f"| {num(x)} | {num(y)} |" for x, y in rows]
    return "\n".join(lines)


def _table(rng: random.Random, f: dict):
    x0, dx = rng.choice(f["xstart"]), rng.choice(f["xstep"])
    m, b = rng.choice(list(f["m"])), rng.choice(list(f["b"]))
    xs = [x0 + dx * i for i in range(4)]
    return xs, dx, m, b, [(x, m * x + b) for x in xs]


def table_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A table of real-life values ('Hours studied' and 'Test score'). 'Which equation represents the relationship
    between x and y?'  Traps: the change between rows used as the slope (rows step by 2), the first y-value used as
    the y-intercept (the table starts at x = 1 or 2, not 0), the slope flipped, the slope with the wrong sign."""
    f = rng.choice(TABLE_FRAMES)
    while True:
        xs, dx, m, b, rows = _table(rng, f)
        if level == "medium" or dx > 1:
            break
    passage = md_table(f["title"], f["x"], f["y"], rows)
    stem = (
        f"The table shows values of x, {f['xd']}, and y, {f['yd']}. The relationship between x and y is linear. "
        f"Which equation represents this relationship?"
    )
    key = f"y = {linf(m, b)}"
    if dx > 1:
        m_wrong, why_m = (
            m * dx,
            f"this uses the change between rows ({num(m * dx)}) as the slope; the rows are {dx} apart",
        )
    elif terminating(Fraction(1) / m) and Fraction(1) / m != m:
        m_wrong, why_m = Fraction(1) / m, "this flips the slope"
    else:
        m_wrong, why_m = -m, "this uses the slope with the wrong sign"
    b_wrong = rows[0][1] if rows[0][1] != b else b + m
    mistakes = grid2(
        lambda mm_, bb: f"y = {linf(mm_, bb)}",
        m,
        m_wrong,
        why_m,
        b,
        b_wrong,
        f"this uses {num(b_wrong)}, the y-value at x = {xs[0]}, as the y-intercept",
    )
    steps = [
        f"Slope: ({num(rows[1][1])} {MINUS} {num(rows[0][1])}) ÷ ({xs[1]} {MINUS} {xs[0]}) = {num(m)}.",
        f"y-intercept: {num(rows[0][1])} = {num(m)}({xs[0]}) + b, so b = {num(b)}.",
        f"So {key}.",
    ]
    item = MathItem(
        "table_equation", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, passage=passage
    )
    item.truth = {"m": m, "b": b, "rows": rows}
    return item


def table_mb(rng: random.Random, level: str, fmt: str) -> MathItem:
    """The same kind of table, but: 'The function f is defined by f(x) = mx + b ... What is the value of m?' (or b).
    Traps: the change between rows, the first y-value, the slope flipped, m and b mixed up."""
    f = rng.choice(TABLE_FRAMES)
    while True:
        xs, dx, m, b, rows = _table(rng, f)
        if level == "medium" or dx > 1:
            break
    ask = rng.choice(["m", "b"])
    passage = md_table(f["title"], f["x"], f["y"], rows)
    stem = (
        f"The table shows values of x, {f['xd']}, and f(x), {f['yd']}, for a linear function f defined by "
        f"f(x) = mx + b, where m and b are constants. What is the value of {ask}?"
    )
    if ask == "m":
        ans = Fraction(m)
        mistakes = [
            (Fraction(m * dx), "this is the change between rows, not per 1 unit of x"),
            (Fraction(b), "this is the value of b, not m"),
            (Fraction(1, m), "this flips the slope"),
            (
                Fraction(rows[1][1] - rows[0][1], rows[1][0]),
                f"this divides the change in y by x = {xs[1]} instead of by the change in x",
            ),
        ]
    else:
        ans = Fraction(b)
        mistakes = [
            (Fraction(rows[0][1]), f"this is the y-value at x = {xs[0]}, not at x = 0"),
            (Fraction(m), "this is the value of m, not b"),
            (Fraction(rows[0][1] + m * xs[0]), "this adds instead of subtracting when working back to x = 0"),
            (Fraction(rows[0][1] - m * dx * xs[0]), "this works back using the change between rows"),
        ]
    steps = [
        f"m = ({num(rows[1][1])} {MINUS} {num(rows[0][1])}) ÷ ({xs[1]} {MINUS} {xs[0]}) = {num(m)}.",
        f"b: {num(rows[0][1])} = {num(m)}({xs[0]}) + b, so b = {num(b)}.",
        f"So {ask} = {num(ans)}.",
    ]
    item = MathItem(
        "table_mb",
        FUNC_SKILL,
        level,
        fmt,
        stem,
        ans,
        steps,
        mistakes,
        passage=passage,
        positive_only=False,
        fractions_ok=ask == "m",
    )
    item.truth = {"m": m, "b": b, "ask": ask, "rows": rows}
    return item


def table_shift(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A table of x and f(x) for a linear function f, then 'g(x) = f(x) + 5' (medium) or 'g(x) = f(x − 3)' /
    'g(x) = f(x + 3)' (hard). 'Which equation defines g?'  Traps: shifting the wrong way, adding the shift to the
    slope, forgetting to multiply the inside shift by the slope, leaving g the same as f."""
    while True:
        m, b = rng.choice(nz(-6, 7)), rng.choice(nz(-12, 15))
        x0, dx = rng.choice([-2, -1, 0, 1, 2]), rng.choice([1, 2])
        c = rng.randint(2, 9)
        if abs(m) != 1 and m * c != c:
            break
    rows = [(x0 + dx * i, m * (x0 + dx * i) + b) for i in range(4)]
    passage = md_table("Values of the linear function f", "x", "f(x)", rows)
    kind = "outside" if level == "medium" else rng.choice(["minus", "plus"])
    sign = rng.choice([1, -1]) if kind == "outside" else 1
    if kind == "outside":
        g = f"g(x) = f(x){plus(sign * c)}"
        key = f"g(x) = {linf(m, b + sign * c)}"
        mistakes = [
            (f"g(x) = {linf(m, b - sign * c)}", f"this {'subtracts' if sign > 0 else 'adds'} {c} instead"),
            (f"g(x) = {linf(m + sign * c, b)}", f"this adds the {num(sign * c)} to the slope"),
            (f"g(x) = {linf(m, b)}", "this is f itself, unchanged"),
            (
                f"g(x) = {linf(m * c, b)}" if sign > 0 else f"g(x) = {linf(m, sign * c)}",
                "this multiplies the slope by the shift" if sign > 0 else "this leaves out f's own constant",
            ),
        ]
        steps = [
            f"From the table, f(x) = {linf(m, b)} (slope {num(m)}, f(0) = {num(b)}).",
            f"g(x) = f(x){plus(sign * c)} = {linf(m, b + sign * c)}.",
        ]
    else:
        shift = -c if kind == "minus" else c  # g(x) = f(x + shift)
        g = f"g(x) = f(x{plus(shift)})"
        key = f"g(x) = {linf(m, b + m * shift)}"
        mistakes = [
            (f"g(x) = {linf(m, b - m * shift)}", "this shifts the wrong way"),
            (f"g(x) = {linf(m, b + shift)}", f"this forgets to multiply the {num(shift)} by the slope, {num(m)}"),
            (f"g(x) = {linf(m + shift, b)}", f"this adds the {num(shift)} to the slope"),
            (f"g(x) = {linf(m, b)}", "this is f itself, unchanged"),
        ]
        steps = [
            f"From the table, f(x) = {linf(m, b)}.",
            f"g(x) = f(x{plus(shift)}) = {num(m)}(x{plus(shift)}){plus(b)} = {linf(m, b + m * shift)}.",
        ]
    stem = (
        f"The table shows several values of x and their corresponding values of f(x) for the linear function f. "
        f"The function g is defined by {g}. Which equation defines g?"
    )
    item = MathItem(
        "table_shift", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, passage=passage
    )
    item.truth = {"m": m, "b": b, "g": g}
    return item


def sum_functions(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'f(x) = 3x − 8 and g(x) = 5x + 20. The function h is defined by h(x) = f(x) + g(x). What is the slope of the
    graph of y = h(x)?' (or the y-coordinate of the y-intercept, or the x-coordinate of the x-intercept). Hard can
    use h(x) = f(x) − g(x) and asks for the x-intercept more often. Traps: the slopes multiplied, only f's value, the
    wrong sign from subtracting, f's own x-intercept."""
    while True:
        m1, m2 = rng.sample(nz(-7, 9), 2)
        b1, b2 = rng.sample(nz(-20, 25), 2)
        op = 1 if level == "medium" else rng.choice([1, -1])
        M, Bc = m1 + op * m2, b1 + op * b2
        if M and Bc and Fraction(-Bc, M).denominator <= 4 and m1 * m2 != M and b1 != Bc:
            break
    ask = rng.choice(
        ["slope", "y-intercept"] if level == "medium" else ["slope", "y-intercept", "x-intercept", "x-intercept"]
    )
    h = f"h(x) = f(x) {'+' if op > 0 else MINUS} g(x)"
    q = {
        "slope": "What is the slope of the graph of y = h(x) in the xy-plane?",
        "y-intercept": "What is the y-coordinate of the y-intercept of the graph of y = h(x) in the xy-plane?",
        "x-intercept": "What is the x-coordinate of the x-intercept of the graph of y = h(x) in the xy-plane?",
    }[ask]
    stem = (
        f"The functions f and g are defined by f(x) = {linf(m1, b1)} and g(x) = {linf(m2, b2)}. The function h is "
        f"defined by {h}. {q}"
    )
    if ask == "slope":
        ans = Fraction(M)
        mistakes = [
            (Fraction(m1 * m2), "this multiplies the slopes"),
            (Fraction(m1), "this is the slope of f only"),
            (Fraction(m1 - op * m2), f"this {'subtracts' if op > 0 else 'adds'} the slopes"),
            (Fraction(Bc), "this is the y-intercept of h"),
        ]
    elif ask == "y-intercept":
        ans = Fraction(Bc)
        mistakes = [
            (Fraction(b1 * b2), "this multiplies the constants"),
            (Fraction(b1), "this is f's y-intercept only"),
            (Fraction(b1 - op * b2), f"this {'subtracts' if op > 0 else 'adds'} the constants"),
            (Fraction(M), "this is the slope of h"),
        ]
    else:
        ans = -Fraction(Bc, M)
        mistakes = [
            (Fraction(Bc, M), "this has the right size but the wrong sign"),
            (-Fraction(b1, m1), "this is the x-intercept of f only"),
            (
                -Fraction(b1, m1) - op * Fraction(b2, m2),
                "this combines the x-intercepts of f and g, which does not give h's x-intercept",
            ),
            (Fraction(Bc), "this is the y-intercept of h"),
        ]
    steps = [
        f"{h} = ({linf(m1, b1)}) {'+' if op > 0 else MINUS} ({linf(m2, b2)}) = {linf(M, Bc)}.",
        {
            "slope": f"The slope is {num(M)}.",
            "y-intercept": f"The y-intercept is (0, {num(Bc)}).",
            "x-intercept": f"Set h(x) = 0: {linf(M, Bc)} = 0, so x = {num(ans)}.",
        }[ask],
    ]
    item = MathItem(
        "sum_functions", FUNC_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False, fractions_ok=True
    )
    item.truth = {"M": M, "B": Bc, "ask": ask}
    return item


# ---------------------------------------------------------------- linear functions: more from Sebastian's list (Nov 25)


def std_form(p: int, q: int, b) -> str:
    """The line y = (p/q)x + b in standard form, 'px − qy = −qb', with whole numbers and a positive x-coefficient."""
    import math

    A, Bc, C = p, -q, -q * Fraction(b)
    den = C.denominator
    A, Bc, C = A * den, Bc * den, int(C * den)
    g = math.gcd(math.gcd(abs(A), abs(Bc)), abs(C)) or 1
    A, Bc, C = A // g, Bc // g, C // g
    if A < 0:
        A, Bc, C = -A, -Bc, -C
    yterm = f" + {coef(Bc, 'y')}" if Bc > 0 else f" {MINUS} {coef(-Bc, 'y')}"
    return f"{coef(A)}{yterm} = {num(C)}"


def graph_which_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """The graph of a line in the xy-plane. 'Which equation defines the line shown?'  Easy: whole-number slope;
    medium: a fraction slope; hard: the choices are in standard form (ax + by = c). Traps: the slope's sign, the
    y-intercept's sign, the slope flipped (run over rise)."""
    while True:
        p_ = rng.choice(nz(-4, 4))
        q = 1 if level == "easy" else rng.choice([2, 3])
        b = rng.choice(nz(-5, 5))
        m = Fraction(p_, q)
        if m.denominator == q and abs(m) != 1 and abs(b) != abs(m):
            break
    fig = graph_svg("x", "y", 1, 8, 1, 8, lines=[(m, b)], label_every=(2, 2), x_lo=8, y_lo=8, plane=True)
    stem = "What is an equation of the line shown?"

    def show(slope, intercept) -> str:
        slope = Fraction(slope)
        if level == "hard":
            return std_form(slope.numerator, slope.denominator, intercept)
        return f"y = {linf(slope, intercept)}"

    key = show(m, b)
    mistakes = [
        (show(-m, b), "this uses the slope with the wrong sign"),
        (show(m, -b), "this uses the y-intercept with the wrong sign"),
        (show(1 / m, b), "this flips the slope (run over rise)"),
        (show(-m, -b), "this uses both signs backward"),
    ]
    steps = [
        f"The line crosses the y-axis at (0, {num(b)}), so the y-intercept is {num(b)}.",
        f"From (0, {num(b)}), the line goes {'up' if m > 0 else 'down'} {abs(m.numerator)} for every "
        f"{m.denominator} to the right, so the slope is {num(m)}.",
        f"So the line is y = {linf(m, b)}" + (f", or {key}." if level == "hard" else "."),
    ]
    item = MathItem(
        "graph_which_equation", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, figure=fig
    )
    item.truth = {"m": m, "b": b}
    return item


def evaluate_function(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'f(x) = 4x − 9. What is the value of f(6)?'  Medium: 'If f(k) = 31, what is k?' (work backward).
    Hard: 'f(x) = 4x + c, where c is a constant. If f(3) = 19, what is the value of f(5)?'"""
    letter = rng.choice(["f", "g", "h"])
    a, c = rng.choice(nz(-9, 9)), rng.choice(nz(-15, 20))
    if abs(a) == 1:
        a = 3 * a
    if level == "easy":
        x = rng.randint(2, 12)
        ans = Fraction(a * x + c)
        stem = f"The function {letter} is defined by {letter}(x) = {linf(a, c)}. What is the value of {letter}({x})?"
        mistakes = [
            (Fraction(a * x), f"this leaves out the {num(c)}"),
            (Fraction(a * x - c), f"this uses {num(-c)} instead of {num(c)}"),
            (Fraction(a + x + c), f"this adds {x} instead of multiplying it by {num(a)}"),
            (Fraction(a * (x + c)), f"this combines {x} and {num(c)} before multiplying by {num(a)}"),
        ]
        steps = [f"{letter}({x}) = {num(a)}({x}){plus(c)} = {num(ans)}."]
    elif level == "medium":
        k = rng.randint(-6, 12) or 5
        v = a * k + c
        ans = Fraction(k)
        stem = (
            f"The function {letter} is defined by {letter}(x) = {linf(a, c)}. If {letter}(k) = {num(v)}, what is "
            f"the value of k?"
        )
        mistakes = [
            (Fraction(a * v + c), f"this finds {letter}({num(v)}), putting the output in as the input"),
            (Fraction(v - c), f"this stops before dividing by {num(a)}"),
            (
                Fraction(v + c, a),
                f"this {'adds' if c > 0 else 'subtracts'} {num(abs(c))} instead of "
                f"{'subtracting' if c > 0 else 'adding'} it",
            ),
            (Fraction(v), f"this is {letter}(k), not k"),
        ]
        steps = [f"Set {linf(a, c)} = {num(v)} with x = k: {coef(a, 'k')} = {num(v - c)}.", f"So k = {num(k)}."]
    else:
        x1, x2 = rng.sample(range(1, 9), 2)
        y1 = a * x1 + c
        ans = Fraction(a * x2 + c)
        stem = (
            f"The function {letter} is defined by {letter}(x) = {coef(a)} + c, where c is a constant. If "
            f"{letter}({x1}) = {num(y1)}, what is the value of {letter}({x2})?"
        )
        mistakes = [
            (Fraction(c), "this is the value of c"),
            (Fraction(a * x2), "this leaves out c"),
            (Fraction(y1 + x2 - x1), f"this changes {letter}({x1}) by the change in x, not {num(a)} times it"),
            (Fraction(y1 + a * x2), f"this adds {num(a)}({x2}) to {letter}({x1})"),
        ]
        steps = [
            f"{num(y1)} = {num(a)}({x1}) + c, so c = {num(c)}.",
            f"{letter}({x2}) = {num(a)}({x2}){plus(c)} = {num(ans)}.",
        ]
    item = MathItem("evaluate_function", FUNC_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
    item.truth = {"a": a, "c": c}
    return item


CONVERSIONS = [
    dict(
        text="A bamboo plant is {s} inches tall and grows {r} inches each week.",
        var="w",
        f="h",
        q="the plant's height, in feet, w weeks from now",
        k=12,
        u="inches",
        per="week",
        rs=[2, 3, 4, 6],
        ss=[18, 24, 30, 42, 54],
    ),
    dict(
        text="A sunflower is {s} centimeters tall and grows {r} centimeters each week.",
        var="w",
        f="H",
        q="the sunflower's height, in meters, w weeks from now",
        k=100,
        u="centimeters",
        per="week",
        rs=[5, 10, 15, 20, 25],
        ss=[20, 40, 60, 80],
    ),
    dict(
        text="A video has been watched for a total of {s} minutes so far, and viewers add {r} minutes of watching "
        "each day.",
        var="d",
        f="W",
        q="the total watching time, in hours, d days from now",
        k=60,
        u="minutes",
        per="day",
        rs=[30, 45, 90, 120],
        ss=[90, 150, 180, 240],
    ),
    dict(
        text="A puppy weighs {s} ounces and gains {r} ounces each week.",
        var="w",
        f="P",
        q="the puppy's weight, in pounds, w weeks from now",
        k=16,
        u="ounces",
        per="week",
        rs=[4, 6, 8, 12],
        ss=[24, 40, 48, 56],
    ),
]


def unit_conversion_function(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'A bamboo plant is 18 inches tall and grows 2 inches each week. Which function gives the plant's height, in
    FEET, w weeks from now?'  Medium shows the choices unsimplified, h(w) = (2w + 18)/12; hard simplifies them,
    h(w) = (1/6)w + 3/2. Traps: no conversion, converting only the rate or only the start, multiplying instead of
    dividing."""
    f = rng.choice(CONVERSIONS)
    k, r, s = f["k"], rng.choice(f["rs"]), rng.choice(f["ss"])
    while s == r:  # the swapped choice must differ
        s = rng.choice(f["ss"])
    v, fn = f["var"], f["f"]
    stem = f"{f['text'].format(s=s, r=r)} Which function {fn} gives {f['q']}?"
    if level == "medium":

        def show(conv: str, order: bool) -> str:
            inner = f"{coef(r, v)} + {s}" if order else f"{coef(s, v)} + {r}"
            return f"{fn}({v}) = ({inner})/{k}" if conv == "div" else f"{fn}({v}) = {k}({inner})"
    else:

        def show(conv: str, order: bool) -> str:
            rate, start = (Fraction(r), Fraction(s)) if order else (Fraction(s), Fraction(r))
            if conv == "div":
                return f"{fn}({v}) = {linf(rate / k, start / k, v)}"
            return f"{fn}({v}) = {linf(rate * k, start * k, v)}"

    key = show("div", True)
    mistakes = grid2(
        show,
        "div",
        "mul",
        f"this multiplies by {k} instead of dividing",
        True,
        False,
        f"this swaps the {r} {f['u']} each {f['per']} and the starting {s} {f['u']}",
    )
    steps = [
        f"In {f['u']}, the amount after {v} {f['per']}s is {coef(r, v)} + {s}.",
        f"There are {k} {f['u']} in each new unit, so divide by {k}: {key}.",
    ]
    item = MathItem(
        "unit_conversion_function", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key
    )
    item.truth = {"r": r, "s": s, "k": k, "var": v}
    return item


TWO_LINE_FRAMES = [
    dict(
        intro="Two plants, plant A and plant B, were measured each week. The graph shows linear models of their "
        "heights y, in centimeters, x weeks after the first measurement.",
        x="Time (weeks)",
        y="Height (centimeters)",
        names=("Plant A", "Plant B"),
        xs=[1, 2],
        ys=[2, 4, 5],
        when="weeks after the first measurement",
        what="the two plants have the same height",
        yu="centimeters",
        per="week",
        rate="grow",
    ),
    dict(
        intro="Two friends, {n1} and {n2}, each put money into a savings account every week. The graph shows the "
        "amount of money y, in dollars, in each account x weeks after they started.",
        x="Time (weeks)",
        y="Amount saved (dollars)",
        names=("{n1}", "{n2}"),
        xs=[1, 2],
        ys=[20, 25, 50],
        when="weeks after they started",
        what="the two accounts hold the same amount",
        yu="dollars",
        per="week",
        rate="save",
    ),
    dict(
        intro="Two tanks, tank P and tank Q, are being filled with water at constant rates. The graph shows the "
        "amount of water y, in gallons, in each tank x minutes after filling began.",
        x="Time (minutes)",
        y="Water (gallons)",
        names=("Tank P", "Tank Q"),
        xs=[2, 5],
        ys=[10, 20, 25],
        when="minutes after filling began",
        what="the two tanks hold the same amount of water",
        yu="gallons",
        per="minute",
        rate="gain",
    ),
]


def two_lines_graph(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Two lines on one real-world graph. Medium: 'how many weeks after the first measurement do the two plants have
    the same height?' (read where the lines cross). Hard: 'how many more centimeters per week does plant A grow than
    plant B?' (the difference of the two slopes, each read from the graph). Traps: the y-value of the crossing, one
    grid line over, one line's rate only, the difference in starting values, the rates added."""
    f = rng.choice(TWO_LINE_FRAMES)
    n1, n2 = rng.sample(NAMES, 2)
    names = [x.format(n1=n1, n2=n2) for x in f["names"]]
    said = [n if "{" in f["names"][0] else n[0].lower() + n[1:] for n in names]  # "plant A" inside a sentence
    intro = f["intro"].format(n1=n1, n2=n2)
    x_n, y_n = 10, rng.choice([10, 12])
    while True:
        xs, ys = Fraction(rng.choice(f["xs"])), Fraction(rng.choice(f["ys"]))
        k = rng.randint(3, 7)  # the lines cross k grid lines to the right of the y-axis
        d1, d2 = sorted(rng.sample([1, 2, 3], 2), reverse=True)
        j = rng.randint(4, y_n - 2)
        if j - d1 * k >= 0:
            break
    xi, yi = xs * k, ys * j
    m1, m2 = d1 * ys / xs, d2 * ys / xs
    b1, b2 = yi - m1 * xi, yi - m2 * xi
    fig = graph_svg(f["x"], f["y"], xs, x_n, ys, y_n, lines=[(m1, b1), (m2, b2)], label_every=(1, 2), line_names=names)
    if level != "hard":
        ans = xi
        stem = f"{intro} According to the graph, how many {f['when']} do {f['what']}?"
        mistakes = [
            (yi, f"this is the number of {f['yu']} where the lines cross, not the time"),
            (xi + xs, "this reads one grid line too far to the right"),
            (xi - xs, "this reads one grid line too far to the left"),
            (b2 - b1, "this is the difference between the two starting values"),
            ((b2 - b1) / m1, f"this is when {said[0]} reaches the starting value of {said[1]}"),
            (m1 - m2, f"this is how many more {f['yu']} per {f['per']} {said[0]} gains, not a time"),
        ]
        steps = [f"The two lines cross at ({dec(xi)}, {dec(yi)}).", f"So {f['what']} {dec(xi)} {f['when']}."]
    else:
        p_ = xs * rng.choice([2, 3, 4])  # "during each 4-week period": the per-week difference becomes a trap
        ans = p_ * (m1 - m2)
        stem = (
            f"{intro} According to the graph, how many more {f['yu']} does {said[0]} {f['rate']} than {said[1]} "
            f"during each {dec(p_)}-{f['per']} period?"
        )
        mistakes = [
            (m1 - m2, f"this is the difference for one {f['per']}, not {dec(p_)} {f['per']}s"),
            (p_ * m1, f"this is the amount for {said[0]} alone"),
            (p_ * m2, f"this is the amount for {said[1]} alone"),
            (b2 - b1, "this is the difference between the two starting values"),
            (p_ * (m1 + m2), "this adds the two amounts"),
        ]
        steps = [
            f"{names[0]}: from (0, {dec(b1)}) to ({dec(xi)}, {dec(yi)}), it changes {dec(yi - b1)} in "
            f"{dec(xi)}, a rate of {dec(m1)} per {f['per']}.",
            f"{names[1]}: from (0, {dec(b2)}) to ({dec(xi)}, {dec(yi)}), it changes {dec(yi - b2)} in "
            f"{dec(xi)}, a rate of {dec(m2)} per {f['per']}.",
            f"The difference is {dec(m1 - m2)} per {f['per']}, so over {dec(p_)} {f['per']}s it is "
            f"{dec(p_)} × {dec(m1 - m2)} = {dec(ans)}.",
        ]
    mistakes = [(v, w) for v, w in mistakes if v > 0 and terminating(v)]
    item = MathItem(
        "two_lines_graph",
        FUNC_SKILL,
        level,
        fmt,
        stem,
        ans,
        steps,
        mistakes,
        figure=fig,
        fractions_ok=True,
        decimals=True,
    )
    item.truth = {"m": (m1, m2), "b": (b1, b2), "xi": xi, "yi": yi}
    return item


# ---------------------------------------------------------- linear equations in two variables (Sebastian, Nov 25)

TWO_SKILL = "Linear equations in two variables"

TWO_VAR_FRAMES = [
    dict(
        setup="A bakery sells muffins for ${a} each and loaves of bread for ${b} each.",
        eq="On Saturday, the bakery earned ${c} from selling x muffins and y loaves of bread.",
        ineq="The bakery's goal is to earn {dir} ${c} on Saturday from selling x muffins and y loaves of bread.",
        dirs=("at least", "≥"),
        a=(2, 5),
        b=(4, 9),
        what="dollars earned",
        xs="muffins",
        ys="loaves of bread",
    ),
    dict(
        setup="A theater sells adult tickets for ${a} each and student tickets for ${b} each.",
        eq="The theater earned ${c} from selling x adult tickets and y student tickets for one show.",
        ineq="The theater needs to earn {dir} ${c} from selling x adult tickets and y student tickets for one show.",
        dirs=("at least", "≥"),
        a=(12, 20),
        b=(6, 11),
        what="dollars earned",
        xs="adult tickets",
        ys="student tickets",
    ),
    dict(
        setup="Ana's training plan includes running workouts that take {a} minutes each and swimming workouts that "
        "take {b} minutes each.",
        eq="Last month, Ana spent a total of {c} minutes on x running workouts and y swimming workouts.",
        ineq="Next month, Ana plans to spend {dir} {c} minutes on x running workouts and y swimming workouts.",
        dirs=("at most", "≤"),
        a=(25, 45),
        b=(30, 60),
        what="minutes",
        xs="running workouts",
        ys="swimming workouts",
    ),
    dict(
        setup="A freight elevator carries boxes of books that weigh {a} pounds each and boxes of tools that weigh "
        "{b} pounds each.",
        eq="On one trip, the elevator carried x boxes of books and y boxes of tools with a total weight of {c} pounds.",
        ineq="On one trip, the elevator can safely carry {dir} {c} pounds of x boxes of books and y boxes of tools.",
        dirs=("at most", "≤"),
        a=(30, 50),
        b=(55, 80),
        what="pounds",
        xs="boxes of books",
        ys="boxes of tools",
    ),
]

CONVERT_FRAMES = [  # hard: the limit is given in a larger unit than the items
    dict(
        text="Each batch of muffins uses {a} ounces of flour, and each batch of bread uses {b} ounces of flour. A "
        "bakery has {big} pounds of flour (1 pound = 16 ounces). Which inequality represents the numbers of "
        "batches of muffins, x, and batches of bread, y, the bakery can make with this flour?",
        k=16,
        a=(8, 14),
        b=(18, 28),
        bigs=(20, 25, 30, 40),
        small="ounces",
        big="pounds",
    ),
    dict(
        text="A delivery truck can carry at most {big} tons (1 ton = 2,000 pounds). Each washing machine weighs {a} "
        "pounds, and each refrigerator weighs {b} pounds. Which inequality represents the numbers of washing "
        "machines, x, and refrigerators, y, the truck can carry?",
        k=2000,
        a=(150, 200),
        b=(250, 350),
        bigs=(2, 3, 4),
        small="pounds",
        big="tons",
    ),
    dict(
        text="A volunteer has at most {big} hours to spend at a food bank this month. Sorting one crate of cans "
        "takes {a} minutes, and packing one box of groceries takes {b} minutes. Which inequality represents the "
        "numbers of crates sorted, x, and boxes packed, y, the volunteer can complete this month?",
        k=60,
        a=(10, 20),
        b=(5, 9),
        bigs=(6, 8, 10, 12),
        small="minutes",
        big="hours",
    ),
]


def two_var_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'A bakery sells muffins for $3 and bread for $5 ... earned $240 ... Which equation represents this
    situation?' (3x + 5y = 240). Medium: 'at least' / 'at most' makes it an inequality. Hard: the limit is in a
    bigger unit (pounds of flour, ounces per batch), so it must be converted. Traps: the prices swapped, the counts
    added (x + y = 240), the sign of the inequality flipped, the limit not converted."""
    ge, le = "≥", "≤"
    if level != "hard":
        f = rng.choice(TWO_VAR_FRAMES)
        a = rng.randint(*f["a"])
        b = rng.choice([v for v in range(f["b"][0], f["b"][1] + 1) if v != a])
        c = a * rng.randint(8, 30) + b * rng.randint(8, 30)
        if level == "medium":  # a goal or a limit is a round number
            c = -(-c // 50) * 50
        setup = f["setup"].format(a=a, b=b)
        if level == "easy":
            rel = "="
            stem = f"{setup} {f['eq'].format(c=num(c))} Which equation represents this situation?"
        else:
            rel = f["dirs"][1]
            stem = f"{setup} {f['ineq'].format(c=num(c), dir=f['dirs'][0])} Which inequality represents this situation?"
        flip = le if rel == ge else ge
        key = f"{coef(a)} + {coef(b, 'y')} {rel} {num(c)}"
        mistakes = [
            (f"{coef(b)} + {coef(a, 'y')} {rel} {num(c)}", "this swaps the amounts for the two items"),
            (f"x + y {rel} {num(c)}", f"this adds the numbers of {f['xs']} and {f['ys']}, not the {f['what']}"),
            (
                f"{coef(a + b)}(x + y) {rel} {num(c)}".replace(f"{a + b}x(", f"{a + b}("),
                f"this uses {a} + {b} for every item",
            ),
        ]
        if level == "easy":
            mistakes.append((f"{num(a)}x {MINUS} {coef(b, 'y')} = {num(c)}", "this subtracts instead of adding"))
        else:
            mistakes = [
                (f"{coef(a)} + {coef(b, 'y')} {flip} {num(c)}", "this has the inequality sign backward"),
                *mistakes[:1],
                (f"{coef(b)} + {coef(a, 'y')} {flip} {num(c)}", "this swaps the amounts and flips the sign"),
                (f"x + y {rel} {num(c)}", f"this counts {f['xs']} and {f['ys']} instead of {f['what']}"),
            ]
        truth = {"a": a, "b": b, "c": c, "rel": rel}
    else:
        f = rng.choice(CONVERT_FRAMES)
        a = rng.randint(*f["a"])
        b = rng.choice([v for v in range(f["b"][0], f["b"][1] + 1) if v != a])
        if f["k"] == 2000:
            a, b = a // 10 * 10, b // 10 * 10
        big = rng.choice(f["bigs"])
        c = big * f["k"]
        stem = f["text"].format(a=a, b=b, big=big)
        key = f"{coef(a)} + {coef(b, 'y')} ≤ {num(c)}"
        mistakes = [
            (f"{coef(a)} + {coef(b, 'y')} ≤ {num(big)}", f"this leaves the limit in {f['big']}"),
            (f"{coef(a)} + {coef(b, 'y')} ≥ {num(c)}", "this has the inequality sign backward"),
            (f"{coef(b)} + {coef(a, 'y')} ≤ {num(c)}", "this swaps the two amounts"),
            (
                f"{coef(a * f['k'])} + {coef(b * f['k'], 'y')} ≤ {num(big)}",
                f"this converts the wrong way (multiplies the {f['small']} by {num(f['k'])})",
            ),
        ]
        truth = {"a": a, "b": b, "c": c, "rel": le}
    steps = [
        f"Each of the x items accounts for {num(a)} and each of the y items for {num(b)}, so the total is "
        f"{coef(a)} + {coef(b, 'y')}.",
        f"So {key}.",
    ]
    if level == "hard":
        steps.insert(1, f"The limit in {f['small']} is {big} × {num(f['k'])} = {num(c)}.")
    item = MathItem("two_var_equation", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = truth
    return item


def par(n) -> str:
    """A number ready to be multiplied: '3', '(1/3)', '(−2)'."""
    n = Fraction(n)
    return num(n) if n > 0 and n.denominator == 1 else f"({num(n)})"


def xy_table(rows) -> str:
    return "\n".join(["| x | y |", "|---|---|"] + [f"| {num(x)} | {num(y)} |" for x, y in rows])


def table_for_equation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Which table gives three values of x and their corresponding values of y for the equation y = 3x − 2?'
    The choices are tables. Medium: a fraction slope with x-values that make whole numbers. Hard: the equation is in
    standard form (2x + 3y = 12). Traps: the slope's sign, the intercept's sign, x and y swapped, the slope and
    intercept swapped."""
    while True:
        m = Fraction(rng.choice(nz(-5, 5)))
        if level == "medium":
            m = Fraction(rng.choice(nz(-3, 3)), rng.choice([2, 3]))
        b = rng.choice(nz(-9, 9))
        if level == "hard":
            A, Bc = rng.choice([(2, 3), (3, 2), (4, 3), (2, 5), (3, 4), (5, 2)])
            sgn = rng.choice([1, -1])
            Bc *= sgn
            C = Bc * rng.choice(nz(-4, 4))
            m, b = Fraction(-A, Bc), Fraction(C, Bc)
        if abs(m) != 1 and m.denominator <= 5 and abs(b) != abs(m) and m != b:
            break
    step = m.denominator
    x0 = rng.choice([0, step, -step]) if level != "easy" else rng.randint(0, 2)
    xs = [x0, x0 + step, x0 + 2 * step] if level != "easy" else [x0, x0 + 1, x0 + 3]
    eq = f"y = {linf(m, b)}" if level != "hard" else std_form(m.numerator, m.denominator, b).replace("x", "x", 1)
    if level == "hard":  # std_form writes the line y = mx + b as Ax + By = C
        eq = std_form(m.numerator, m.denominator, b)

    def tab(mm_, bb, swap=False):
        rows = [(x, mm_ * x + bb) for x in xs]
        return xy_table([(y, x) for x, y in rows] if swap else rows)

    key = tab(m, b)
    mistakes = [
        (tab(-m, b), "these values use the slope with the wrong sign"),
        (tab(m, -b), "these values use the y-intercept with the wrong sign"),
        (tab(m, b, swap=True), "this table swaps the x- and y-values"),
        (tab(b, m), "these values swap the slope and the y-intercept"),
    ]
    mistakes = [(t, w) for t, w in mistakes if all(Fraction(v).denominator == 1 for v in _table_values(t))]
    stem = f"Which table gives three values of x and their corresponding values of y for the equation {eq}?"
    steps = [f"Substitute each x-value into {eq} and solve for y."]
    if level == "hard":
        steps.insert(0, f"Solving for y gives y = {linf(m, b)}.")
    steps.append(
        "Only one table has every pair correct: " + ", ".join(f"({num(x)}, {num(m * x + b)})" for x in xs) + "."
    )
    item = MathItem("table_for_equation", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b}
    return item


def _table_values(t: str) -> list[Fraction]:
    out = []
    for line in t.splitlines()[2:]:
        for cell in line.strip("|").split("|"):
            out.append(Fraction(cell.strip().replace(MINUS, "-").replace(",", "")))
    return out


TRADEOFF_FRAMES = [
    dict(
        text="A caterer has a budget of ${c} for meat. The equation {eq} represents the possible numbers of pounds "
        "of chicken, x, and pounds of beef, y, the caterer can buy.",
        more="If the caterer buys {k} more pounds of chicken, how many fewer pounds of beef can the caterer buy?",
        less="If the caterer buys {k} fewer pounds of beef, how many more pounds of chicken can the caterer buy?",
        each="For each additional pound of chicken the caterer buys, how many fewer pounds of beef can the caterer "
        "buy?",
        a=(4, 6, 8),
        b=(8, 10, 12),
        x1="pound of chicken",
        xn="pounds of chicken",
        y1="pound of beef",
        yn="pounds of beef",
        use="costs ${v}",
    ),
    dict(
        text="Rosa earns money tutoring and working at a bookstore. The equation {eq} represents the numbers of hours "
        "she can tutor, x, and work at the bookstore, y, to earn ${c} in a month.",
        more="If Rosa tutors {k} more hours, how many fewer hours does she need to work at the bookstore to still "
        "earn ${c}?",
        less="If Rosa works {k} fewer hours at the bookstore, how many more hours does she need to tutor to still "
        "earn ${c}?",
        each="For each additional hour Rosa tutors, how many fewer hours does she need to work at the bookstore to "
        "still earn ${c}?",
        a=(18, 20, 24, 25),
        b=(12, 15, 16),
        x1="hour of tutoring",
        xn="hours of tutoring",
        y1="hour at the bookstore",
        yn="hours at the bookstore",
        use="earns ${v}",
    ),
    dict(
        text="A printer has enough ink for {c} units of printing. The equation {eq} represents the numbers of color "
        "pages, x, and black-and-white pages, y, it can print with this ink.",
        more="If {k} more color pages are printed, how many fewer black-and-white pages can be printed?",
        less="If {k} fewer black-and-white pages are printed, how many more color pages can be printed?",
        each="For each additional color page printed, how many fewer black-and-white pages can be printed?",
        a=(6, 8, 9, 12),
        b=(2, 3, 4),
        x1="color page",
        xn="color pages",
        y1="black-and-white page",
        yn="black-and-white pages",
        use="uses {v} units",
    ),
]


def _trade_k(rng: random.Random, a: int, b: int, need: int) -> int:
    """A change k that makes the answer and, when it stays small, the flipped ratio (the main trap) whole numbers."""
    import math

    g = math.gcd(a, b)
    both = (a // g) * (b // g)
    return both if both <= 15 else need // g * rng.randint(1, 2)


def tradeoff(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'6x + 8y = 600 ... If the caterer buys 4 more pounds of chicken, how many fewer pounds of beef can it buy?'
    The extra x use a·k, which must come out of y at b each: a·k/b. Medium goes the other way (k fewer y: b·k/a more
    x). Hard asks per single unit, a fraction. Traps: the ratio flipped, the amount the change uses up, one for one,
    the most y possible."""

    f = rng.choice(TRADEOFF_FRAMES)
    a = rng.choice(f["a"])
    b = rng.choice([v for v in f["b"] if v != a])
    if level == "hard":
        while a % b == 0:
            b = rng.choice([v for v in range(2, 16) if v != a])
    c = a * b * rng.randint(4, 9)
    eq = f"{coef(a)} + {coef(b, 'y')} = {num(c)}"
    intro = f["text"].format(c=num(c), eq=eq)
    if level == "hard":
        ans = Fraction(a, b)
        stem = f"{intro} {f['each'].format(c=num(c))}"
        mistakes = [
            (Fraction(b, a), f"this divides {b} by {a}, the ratio flipped"),
            (Fraction(a), f"this is the {a} for one {f['x1']}"),
            (Fraction(1), "this trades one for one"),
            (Fraction(c, b), f"this is the most {f['yn']} possible (when x = 0)"),
        ]
        steps = [
            f"Each {f['x1']} {f['use'].format(v=a)}, and each {f['y1']} {f['use'].format(v=b)}.",
            f"One more {f['x1']} must be balanced by {a} ÷ {b} = {num(ans)} fewer {f['yn']}.",
        ]
    elif level == "easy":
        k = _trade_k(rng, a, b, b)
        ans = Fraction(a * k, b)
        stem = f"{intro} {f['more'].format(k=k, c=num(c))}"
        mistakes = [
            (Fraction(b * k, a), f"this flips the ratio ({b} × {k} ÷ {a})"),
            (Fraction(a * k), f"this is {a} × {k}, the amount the extra {f['xn']} account for"),
            (Fraction(k), "this trades one for one"),
        ]
        steps = [
            f"{k} more {f['xn']} account for {a} × {k} = {a * k} of the {num(c)}.",
            f"Each {f['y1']} {f['use'].format(v=b)}, so that is {a * k} ÷ {b} = {num(ans)} fewer {f['yn']}.",
        ]
    else:
        k = _trade_k(rng, a, b, a)
        ans = Fraction(b * k, a)
        stem = f"{intro} {f['less'].format(k=k, c=num(c))}"
        mistakes = [
            (Fraction(a * k, b), f"this flips the ratio ({a} × {k} ÷ {b})"),
            (Fraction(b * k), f"this is {b} × {k}, the amount the {k} {f['yn']} account for"),
            (Fraction(k), "this trades one for one"),
        ]
        steps = [
            f"{k} fewer {f['yn']} free up {b} × {k} = {b * k} of the {num(c)}.",
            f"Each {f['x1']} {f['use'].format(v=a)}, so that is {b * k} ÷ {a} = {num(ans)} more {f['xn']}.",
        ]
    item = MathItem("tradeoff", TWO_SKILL, level, fmt, stem, ans, steps, mistakes, fractions_ok=level == "hard")
    item.truth = {"a": a, "b": b, "c": c}
    return item


def slope_point_line(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'A line in the xy-plane has a slope of 3 and passes through the point (2, 5). Which equation defines the
    line?'  Medium: a fraction slope or negative coordinates. Hard: the line is parallel to a line given in standard
    form. Traps: the point's y-value as the y-intercept, the wrong sign when solving for b, the slope's sign."""
    while True:
        if level == "easy":
            m = Fraction(rng.randint(2, 6))
            x1, y1 = rng.randint(1, 5), rng.randint(1, 12)
        else:
            m = Fraction(rng.choice(nz(-4, 4)), rng.choice([1, 2, 3]))
            x1 = m.denominator * rng.choice(nz(-3, 3))
            y1 = rng.choice(nz(-9, 9))
        b = y1 - m * x1
        if b and abs(m) != 1 and b != y1 and y1 + m * x1 != b and b != x1:
            break
    pt = f"({num(x1)}, {num(y1)})"
    if level == "hard":
        given = std_form(m.numerator, m.denominator, rng.choice([v for v in nz(-6, 6) if v != b]))
        stem = f"Line k is parallel to the line {given} and passes through the point {pt}. Which equation defines k?"
    else:
        stem = (
            f"A line in the xy-plane has a slope of {num(m)} and passes through the point {pt}. Which equation "
            f"defines the line?"
        )
    key = f"y = {linf(m, b)}"
    mistakes = [
        (f"y = {linf(m, y1)}", f"this uses {num(y1)}, the point's y-value, as the y-intercept"),
        (f"y = {linf(m, y1 + m * x1)}", "this adds the slope times x instead of subtracting it when finding b"),
        (f"y = {linf(-m, y1 + m * x1)}", "this uses the slope with the wrong sign"),
        (f"y = {linf(m, x1)}", f"this uses {num(x1)}, the point's x-value, as the y-intercept"),
    ]
    if level == "hard":
        mistakes[2] = (
            f"y = {linf(-m, y1 + m * x1)}",
            "this reads the slope from the standard form with the wrong sign",
        )
    steps = [
        f"Write y = {coef(m)} + b and substitute {pt}: {num(y1)} = {par(m)}({num(x1)}) + b, so b = {num(b)}.",
        f"So {key}.",
    ]
    if level == "hard":
        steps.insert(0, f"Solving {given} for y gives a slope of {num(m)}; parallel lines have the same slope.")
    item = MathItem("slope_point_line", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b, "pt": (x1, y1)}
    return item


def isosceles_perimeter(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'An isosceles triangle has two sides of length x inches and a third side of length y inches. Its perimeter is
    36 inches. Which equation represents this situation?'  Medium: the perimeter is given in feet. Hard: the third side
    is in feet and the perimeter in inches. Traps: counting the equal sides once, doubling the wrong side, doubling
    both, not converting."""
    if level == "easy":
        P = rng.randint(12, 40)
        stem = (
            f"An isosceles triangle has two sides that are each x inches long and a third side that is y inches "
            f"long. The perimeter of the triangle is {P} inches. Which equation represents this situation?"
        )
        key = f"2x + y = {P}"
        mistakes = [
            (f"x + y = {P}", "this counts the two equal sides once"),
            (f"x + 2y = {P}", "this doubles the wrong side"),
            (f"2x + 2y = {P}", "this doubles both sides, as for a rectangle"),
        ]
        steps = [f"The perimeter is x + x + y = 2x + y, so 2x + y = {P}."]
        truth = {"cx": 2, "cy": 1, "P": P}
    elif level == "medium":
        ft = rng.randint(2, 6)
        P = 12 * ft
        stem = (
            f"An isosceles triangle has two sides that are each x inches long and a third side that is y inches "
            f"long. The perimeter of the triangle is {ft} feet. Which equation represents this situation? "
            f"(1 foot = 12 inches)"
        )
        key = f"2x + y = {P}"
        mistakes = [
            (f"2x + y = {ft}", "this leaves the perimeter in feet"),
            (f"x + y = {P}", "this counts the two equal sides once"),
            (f"x + 2y = {P}", "this doubles the wrong side"),
            (f"2x + 2y = {P}", "this doubles both sides, as for a rectangle"),
        ]
        steps = [f"The perimeter in inches is {ft} × 12 = {P}.", f"x + x + y = 2x + y, so 2x + y = {P}."]
        truth = {"cx": 2, "cy": 1, "P": P}
    else:
        P = rng.randint(30, 90)
        stem = (
            f"An isosceles triangle has two sides that are each x inches long and a third side that is y feet "
            f"long. The perimeter of the triangle is {P} inches. Which equation represents this situation? "
            f"(1 foot = 12 inches)"
        )
        key = f"2x + 12y = {P}"
        mistakes = [
            (f"2x + y = {P}", "this leaves the third side in feet"),
            (f"2x + y/12 = {P}", "this divides by 12 instead of multiplying"),
            (f"x + 12y = {P}", "this counts the two equal sides once"),
            (f"2x + 24y = {P}", "this doubles the third side too"),
        ]
        steps = ["A side of y feet is 12y inches.", f"The perimeter in inches is x + x + 12y, so 2x + 12y = {P}."]
        truth = {"cx": 2, "cy": 12, "P": P}
    item = MathItem("isosceles_perimeter", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = truth
    return item


def perpendicular_vertical(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Lines perpendicular to a vertical or horizontal line. Easy: 'Line k is perpendicular to the line x = 4. What is
    the slope of line k?' (0; perpendicular to y = 4: undefined). Medium: 'Line k is perpendicular to the line x = 4
    and passes through (2, −3). Which equation defines line k?' (y = −3). Traps: the other orientation, the point's
    other coordinate, the number in the given line."""
    a = rng.choice([v for v in nz(-6, 6) if abs(v) > 1])
    vertical = rng.random() < 0.6
    given = f"x = {num(a)}" if vertical else f"y = {num(a)}"
    if level == "easy":
        stem = f"In the xy-plane, line k is perpendicular to the line {given}. What is the slope of line k?"
        und = "The slope is undefined."
        fixed = ["0", f"{num(a)}", f"{num(Fraction(-1, a))}", und]
        key = "0" if vertical else und
        why = {
            "0": "a horizontal line has slope 0, but it is perpendicular only to a vertical line",
            und: "a vertical line has an undefined slope, but it is perpendicular only to a horizontal line",
            num(a): f"{num(a)} is where the given line crosses the {'x' if vertical else 'y'}-axis, not a slope",
            num(Fraction(-1, a)): f"this treats {num(a)} as a slope and takes the negative reciprocal",
        }
        mistakes = [(v, why[v]) for v in fixed if v != key]
        steps = [
            f"The line {given} is {'vertical' if vertical else 'horizontal'}.",
            "A line perpendicular to it is "
            + ("horizontal, with a slope of 0." if vertical else "vertical, so its slope is undefined."),
        ]
        item = MathItem(
            "perpendicular_vertical",
            TWO_SKILL,
            level,
            "mcq",
            stem,
            Fraction(0),
            steps,
            mistakes,
            key_text=key,
            fixed_choices=fixed,
        )
    else:
        px, py = rng.choice(nz(-7, 7)), rng.choice(nz(-7, 7))
        while abs(px) == abs(py) or a in (px, py):
            px, py = rng.choice(nz(-7, 7)), rng.choice(nz(-7, 7))
        stem = (
            f"In the xy-plane, line k is perpendicular to the line {given} and passes through the point "
            f"({num(px)}, {num(py)}). Which equation defines line k?"
        )
        key = f"y = {num(py)}" if vertical else f"x = {num(px)}"
        other = f"x = {num(px)}" if vertical else f"y = {num(py)}"
        mistakes = [
            (other, "this line passes through the point but is parallel to the given line"),
            (f"y = {num(px)}" if vertical else f"x = {num(py)}", "this uses the wrong coordinate of the point"),
            (
                f"x = {num(py)}" if vertical else f"y = {num(px)}",
                "this uses the wrong coordinate and the wrong direction",
            ),
            (f"y = {num(-py)}" if vertical else f"x = {num(-px)}", "this uses the coordinate with the wrong sign"),
        ]
        steps = [
            f"The line {given} is {'vertical' if vertical else 'horizontal'}, so line k is "
            f"{'horizontal' if vertical else 'vertical'}.",
            f"A {'horizontal' if vertical else 'vertical'} line through ({num(px)}, {num(py)}) is {key}.",
        ]
        item = MathItem(
            "perpendicular_vertical", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key
        )
    item.truth = {"vertical": vertical, "a": a}
    return item


COEF_FRAMES = [
    dict(
        text="Jada earns ${a} per hour tutoring and ${b} per hour babysitting. The equation {eq} represents the "
        "numbers of hours, x, she tutors and the number of hours, y, she babysits to earn ${c} in a month.",
        a=(15, 25),
        b=(9, 14),
        forms={
            "a": "the amount of money, in dollars, Jada earns for each hour she tutors",
            "b": "the amount of money, in dollars, Jada earns for each hour she babysits",
            "x": "the number of hours Jada tutors",
            "c": "the total amount of money, in dollars, Jada earns in the month",
            "ab": "the amount of money, in dollars, Jada earns for each hour of tutoring and babysitting combined",
            "xi": "if Jada does not babysit, she must tutor {v} hours to earn ${c}",
            "yi": "if Jada does not tutor, she must babysit {v} hours to earn ${c}",
            "xi_bad": "Jada tutors {v} hours each week",
            "yi_bad": "Jada earns ${v} each hour she babysits",
        },
    ),
    dict(
        text="A store sells notebooks for ${a} each and binders for ${b} each. The equation {eq} represents the "
        "numbers of notebooks, x, and binders, y, a teacher can buy for ${c}.",
        a=(2, 4),
        b=(5, 9),
        forms={
            "a": "the price, in dollars, of each notebook",
            "b": "the price, in dollars, of each binder",
            "x": "the number of notebooks the teacher buys",
            "c": "the total amount, in dollars, the teacher spends",
            "ab": "the price, in dollars, of one notebook and one binder together",
            "xi": "if the teacher buys no binders, the teacher can buy {v} notebooks",
            "yi": "if the teacher buys no notebooks, the teacher can buy {v} binders",
            "xi_bad": "the teacher buys {v} notebooks and {v} binders",
            "yi_bad": "each binder costs ${v}",
        },
    ),
    dict(
        text="A hiking club plans trips on two trails. The equation {eq} represents the numbers of trips, x, on a "
        "trail that is {a} miles long and the number of trips, y, on a trail that is {b} miles long that "
        "the club can take to hike a total of {c} miles.",
        a=(4, 8),
        b=(9, 13),
        forms={
            "a": "the length, in miles, of each trip on the first trail",
            "b": "the length, in miles, of each trip on the second trail",
            "x": "the number of trips on the first trail",
            "c": "the total number of miles the club hikes",
            "ab": "the total length, in miles, of one trip on each trail",
            "xi": "if the club hikes only the first trail, it takes {v} trips",
            "yi": "if the club hikes only the second trail, it takes {v} trips",
            "xi_bad": "the club hikes {v} miles on each trip",
            "yi_bad": "the second trail is {v} miles long",
        },
    ),
]


def interpret_coefficients(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'The equation 15x + 9y = 300 ... What is the best interpretation of 15 in this context?'  Medium asks about
    the total; hard about an intercept: 'the graph of the equation passes through (20, 0). What is the best
    interpretation of this point?'. Wrong choices are what the other numbers stand for or the number in the wrong
    role."""
    f = rng.choice(COEF_FRAMES)
    while True:
        a = rng.randint(*f["a"])
        b = rng.randint(*f["b"])
        c = a * b * rng.randint(2, 6)
        if a != b:
            break
    eq = f"{coef(a)} + {coef(b, 'y')} = {num(c)}"
    stem0 = f["text"].format(a=a, b=b, c=num(c), eq=eq)
    F_ = {k: v for k, v in f["forms"].items()}
    if level == "easy":
        which = rng.choice(["a", "b"])
        shown = a if which == "a" else b
        stem = f"{stem0} What is the best interpretation of {shown} in this context?"
        other = "b" if which == "a" else "a"
        key = _cap(F_[which])
        mistakes = [
            (_cap(F_[other]), "this describes the other coefficient"),
            (_cap(F_["x"]), "this describes x, the variable, not its coefficient"),
            (_cap(F_["c"]), f"this describes {num(c)}"),
            (_cap(F_["ab"]), f"this would be {a + b}"),
        ]
    elif level == "medium":
        stem = f"{stem0} What is the best interpretation of {num(c)} in this context?"
        key = _cap(F_["c"])
        mistakes = [
            (_cap(F_["a"]), f"this describes {a}"),
            (_cap(F_["b"]), f"this describes {b}"),
            (_cap(F_["x"]), "this describes x"),
            (_cap(F_["ab"]), f"this would be {a + b}"),
        ]
    else:
        on_x = rng.random() < 0.5
        v = Fraction(c, a) if on_x else Fraction(c, b)
        pt = f"({num(v)}, 0)" if on_x else f"(0, {num(v)})"
        stem = (
            f"{stem0} The graph of this equation in the xy-plane passes through the point {pt}. What is the best "
            f"interpretation of this point in this context?"
        )
        key = _cap(F_["xi" if on_x else "yi"].format(v=num(v), c=num(c)))
        mistakes = [
            (_cap(F_["yi" if on_x else "xi"].format(v=num(v), c=num(c))), "this switches the two variables"),
            (_cap(F_["xi_bad"].format(v=num(v), c=num(c))), f"this gives {num(v)} the wrong meaning"),
            (_cap(F_["yi_bad"].format(v=num(v), c=num(c))), f"this treats {num(v)} as a rate"),
        ]
    steps = [f"In {eq}, x and y are counts, {a} and {b} are the amounts for each one, and {num(c)} is the total."]
    if level == "hard":
        steps.append(f"At {pt}, one of the two counts is 0, so the other alone makes the total of {num(c)}.")
    steps.append(f"So the best interpretation is: {key[0].lower() + key[1:]}.")
    item = MathItem("interpret_coefficients", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"a": a, "b": b, "c": c}
    return item


def perpendicular_lines(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'Line k is perpendicular to y = 2x + 5. What is the slope of line k?' (−1/2). Medium: perpendicular to
    y = (3/4)x − 2 and passes through (6, 1); which equation defines k? Hard: the given line is in standard form
    (3x − 5y = 10). Traps: the parallel slope, the reciprocal without the sign change, the opposite without the
    reciprocal, the point's y-value as the y-intercept."""
    while True:
        m = Fraction(rng.choice(nz(-5, 5)), rng.choice([1, 2, 3, 4]))
        if abs(m) != 1 and m.denominator <= 4:
            break
    p = -1 / m
    b0 = rng.choice(nz(-8, 8))
    given = f"y = {linf(m, b0)}" if level != "hard" else std_form(m.numerator, m.denominator, b0)
    if level in ("easy", "hard"):
        stem = f"In the xy-plane, line k is perpendicular to the line {given}. What is the slope of line k?"
        ans = p
        mistakes = [
            (m, "this is the slope of a parallel line"),
            (1 / m, "this flips the slope but keeps its sign"),
            (-m, "this changes the sign but does not flip the slope"),
        ]
        steps = (
            [f"Solving for y gives y = {linf(m, b0)}, so the slope is {num(m)}."]
            if level == "hard"
            else [f"The given line has slope {num(m)}."]
        )
        steps.append(f"A perpendicular slope is the negative reciprocal: {num(ans)}.")
        item = MathItem(
            "perpendicular_lines",
            TWO_SKILL,
            level,
            fmt,
            stem,
            ans,
            steps,
            mistakes,
            positive_only=False,
            fractions_ok=True,
        )
    else:
        while True:
            x1 = p.denominator * rng.choice(nz(-3, 3))
            y1 = rng.choice(nz(-8, 8))
            b = y1 - p * x1
            if b and b != y1 and b != y1 + p * x1:
                break
        stem = (
            f"In the xy-plane, line k is perpendicular to the line {given} and passes through the point "
            f"({num(x1)}, {num(y1)}). Which equation defines line k?"
        )
        key = f"y = {linf(p, b)}"
        mistakes = [
            (f"y = {linf(m, y1 - m * x1)}", "this line is parallel to the given line, not perpendicular"),
            (f"y = {linf(1 / m, y1 - x1 / m)}", "this flips the slope but keeps its sign"),
            (f"y = {linf(p, y1)}", f"this uses {num(y1)}, the point's y-value, as the y-intercept"),
            (f"y = {linf(-m, y1 + m * x1)}", "this changes the sign of the slope but does not flip it"),
        ]
        steps = [
            f"The given slope is {num(m)}, so line k has slope {num(p)}.",
            f"{num(y1)} = {par(p)}({num(x1)}) + b, so b = {num(b)}: {key}.",
        ]
        item = MathItem(
            "perpendicular_lines", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key
        )
    item.truth = {"m": m, "b0": b0}
    return item


def table_unknowns(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'The table shows three values of x and their corresponding values of y. There is a linear relationship
    between x and y. What is the value of k?'  Easy: the x-values step evenly. Medium: uneven steps, k at the end.
    Hard: k in the middle, between unevenly spaced x-values (the average is the trap). Traps: adding the slope once,
    adding the change in x, repeating the last change in y, the halfway value."""
    while True:
        m = rng.choice(nz(-6, 6))
        b = rng.randint(-10, 15)
        if abs(m) != 1:
            break
    if level == "easy":
        x0, d = rng.randint(0, 4), rng.randint(2, 3)
        xs = [x0, x0 + d, x0 + 2 * d]
        hide = 2
    elif level == "medium":
        x0 = rng.randint(0, 4)
        d1, d2 = rng.sample([1, 2, 3, 4, 5], 2)
        xs = [x0, x0 + d1, x0 + d1 + d2]
        hide = 2
    else:
        x0 = rng.randint(-2, 3)
        d1, d2 = rng.sample([1, 2, 3, 4, 5, 6], 2)
        xs = [x0, x0 + d1, x0 + d1 + d2]
        hide = 1
    ys = [m * x + b for x in xs]
    ans = Fraction(ys[hide])
    rows = [(num(x), "k" if i == hide else num(y)) for i, (x, y) in enumerate(zip(xs, ys, strict=True))]
    passage = "\n".join(["| x | y |", "|---|---|"] + [f"| {x} | {y} |" for x, y in rows])
    stem = (
        "The table shows three values of x and their corresponding values of y. There is a linear relationship "
        "between x and y. What is the value of k?"
    )
    known = [i for i in range(3) if i != hide]
    (xa, ya), (xb, yb) = [(xs[i], ys[i]) for i in known]
    if hide == 2 and level == "easy":
        mistakes = [
            (Fraction(yb + m), "this adds the slope once instead of once for each step in x"),
            (Fraction(yb + (xs[2] - xb)), "this adds the change in x instead of the change in y"),
            (Fraction(m * xs[2]), "this multiplies x by the slope but leaves out the y-intercept"),
            (Fraction(yb - m * (xs[2] - xb)), "this uses the slope with the wrong sign"),
        ]
    elif hide == 2:
        mistakes = [
            (Fraction(yb + m), "this adds the slope once instead of once for each step in x"),
            (Fraction(yb + (xs[2] - xb)), "this adds the change in x instead of the change in y"),
            (
                Fraction(yb + (yb - ya)),
                "this repeats the last change in y, but the x-values do not change by the same amount",
            ),
            (Fraction(yb - m * (xs[2] - xb)), "this uses the slope with the wrong sign"),
        ]
    else:
        mistakes = [
            (Fraction(ya + yb, 2), "this is halfway between the y-values, but the x-value is not halfway"),
            (Fraction(ya + m), "this adds the slope once"),
            (Fraction(ya + m * (xs[2] - xs[1])), "this uses the wrong change in x"),
            (Fraction(yb - m), "this subtracts the slope once from the last value"),
        ]
    slope = Fraction(yb - ya, xb - xa)
    steps = [
        f"The slope is ({num(yb)} {MINUS} {num(ya)}) / ({num(xb)} {MINUS} {num(xa)}) = {num(slope)}.",
        f"From x = {num(xa)} to x = {num(xs[hide])}, x changes by {num(xs[hide] - xa)}, so y changes by "
        f"{num(slope)} × {num(xs[hide] - xa)} = {num(slope * (xs[hide] - xa))}.",
        f"k = {num(ya)}{plus(slope * (xs[hide] - xa))} = {num(ans)}.",
    ]
    item = MathItem(
        "table_unknowns", TWO_SKILL, level, fmt, stem, ans, steps, mistakes, passage=passage, positive_only=False
    )
    item.truth = {"xs": xs, "ys": ys, "hide": hide}
    return item


# ------------------------------------------- systems of two linear equations in two variables (Sebastian, Nov 25)

SYS_SKILL = "Systems of two linear equations in two variables"


def axby(A, B, C, decimal: bool = False) -> str:
    """'3x + 2y = 16', '−x + 4y = 0', '0.3x + 0.25y = 33' (A or B may be 0)."""
    A, B = Fraction(A), Fraction(B)
    show = dec if decimal else num
    if A:
        out = coef(A, "x", decimal)
        if B > 0:
            out += f" + {coef(B, 'y', decimal)}"
        elif B < 0:
            out += f" {MINUS} {coef(-B, 'y', decimal)}"
    else:
        out = coef(B, "y", decimal)
    return f"{out} = {show(C)}"


def _line_eq(rng: random.Random, level: str, m, b, scale=1, shift=0) -> str:
    """One line written the way the level calls for. Easy: y = mx + b. Medium: Ax + By = C, scaled. Hard: either."""
    m, b = Fraction(m), Fraction(b)
    if level == "easy":
        return f"y = {linf(m, b)}" if shift == 0 else f"y = {linf(m, b + shift)}"
    A, B, C = -m * m.denominator, Fraction(m.denominator), b * m.denominator  # -mx + y = b, cleared of fractions
    if A < 0 or A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return axby(A * scale, B * scale, C * scale + shift)


def system_which_property(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Which system of linear equations has no solution?' (or infinitely many, or exactly one). Every choice shares
    the first equation; the second equations are near misses of each other (the same line scaled, the same slope
    with another constant, a sign flipped, the coefficients swapped), so a student has to compare slopes and
    constants, not just spot the odd one out. Easy: y = mx + b forms. Medium: standard forms. Hard: the first
    equation in y = mx + b form, the others in standard form."""
    want = rng.choice(["none", "infinite", "one"])
    names = {"none": "no solution", "infinite": "infinitely many solutions", "one": "exactly one solution"}
    why = {
        "none": "these lines are parallel (same slope, different y-intercepts), so the system has no solution",
        "infinite": "these equations describe the same line, so the system has infinitely many solutions",
        "one": "these lines have different slopes, so they cross once: exactly one solution",
    }
    while True:
        m = Fraction(rng.choice(nz(-5, 5)), 1 if level == "easy" else rng.choice([1, 2, 3]))
        b = rng.choice(nz(-9, 9))
        if abs(m) != 1 and m.denominator <= 3:
            break
    k = rng.choice([2, 3])
    first = f"y = {linf(m, b)}" if level != "medium" else _line_eq(rng, "medium", m, b)
    A, B, C = -m * m.denominator, Fraction(m.denominator), b * m.denominator  # -mx + y = b without fractions
    if A < 0:
        A, B, C = -A, -B, -C
    d1, d2 = rng.sample([v for v in nz(-7, 7) if v], 2)

    def second(kind: str, d: int = 0) -> str:
        if level == "easy":
            return {
                "none": f"y = {linf(m, b + d)}",
                "infinite": f"{k}y = {linf(k * m, k * b)}",
                "flip": f"y = {linf(-m, b)}",
                "swap": f"y = {linf(b, m)}" if b != m else f"y = {linf(-m, b + d)}",
            }[kind]
        if kind == "none":
            return axby(k * A, k * B, k * C + d)
        if kind == "infinite":
            return axby(k * A, k * B, k * C)
        if kind == "flip":
            return axby(k * A, -k * B, k * C)
        return axby(k * B, k * A, k * C) if abs(A) != abs(B) else axby(k * A, -k * B, k * C + d)

    pool = {"none": [("none", d1), ("none", d2)], "infinite": [("infinite", 0)], "one": [("flip", 0), ("swap", d1)]}
    kind_of = {"none": "none", "infinite": "infinite", "flip": "one", "swap": "one"}
    key_kind = rng.choice(pool[want])
    options = [kd for w, kds in pool.items() if w != want for kd in kds]
    key = f"{first}\n{second(*key_kind)}"
    mistakes, seen = [], {key}
    for kd in options:
        text = f"{first}\n{second(*kd)}"
        if text not in seen:
            seen.add(text)
            mistakes.append((text, why[kind_of[kd[0]]]))
    stem = f"Which system of linear equations has {names[want]}?"
    steps = [
        f"The first equation is the line y = {linf(m, b)}: slope {num(m)}, y-intercept {num(b)}.",
        "Rewrite each second equation as y = mx + b. The same slope and the same intercept mean the same line "
        "(infinitely many solutions); the same slope and a different intercept mean parallel lines (no "
        "solution); different slopes mean exactly one solution.",
        f"Only the correct choice has {names[want]}.",
    ]
    item = MathItem("system_which_property", SYS_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"want": want}
    return item


def _pick_solution(rng: random.Random) -> tuple[int, int]:
    while True:
        x0, y0 = rng.choice(nz(-9, 9)), rng.choice(nz(-9, 9))
        if abs(x0) != abs(y0):
            return x0, y0


def system_coordinate(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'If (x, y) is the solution to the system, what is the value of x?' (or y). Easy: substitution (y = 3x;
    2x + y = 25). Medium: one variable cancels when the equations are added. Hard: both equations must be multiplied
    first. Traps: the other coordinate (the golden rule), the opposite sign, stopping before dividing."""
    ask = rng.choice(["x", "y"])
    if level == "easy":
        x0 = rng.randint(2, 9)
        m, c = rng.choice([2, 3, 4, 5]), rng.choice([0, 0, rng.randint(-6, 6)])
        y0 = m * x0 + c
        A, B = rng.randint(1, 5), rng.choice([1, 2, 3])
        C = A * x0 + B * y0
        passage = f"y = {linf(m, c)}\n{axby(A, B, C)}"
        steps = [
            f"Substitute y = {linf(m, c)} into {axby(A, B, C)}: {num(A)}x + {num(B)}({linf(m, c)}) = {num(C)}.",
            f"So {coef(A + B * m)} = {num(C - B * c)}, x = {num(x0)}, and y = {num(y0)}.",
        ]
        extra = [(Fraction(C - B * c), f"this stops before dividing by {A + B * m}")]
        if B > 1 and (C - c) % (A + m) == 0:
            extra.append((Fraction(C - c, A + m), f"this substitutes without multiplying by {B}"))
    elif level == "medium":
        x0, y0 = _pick_solution(rng)
        a1, a2 = rng.sample(range(1, 8), 2)
        b = rng.randint(1, 6)
        c1, c2 = a1 * x0 + b * y0, a2 * x0 - b * y0
        passage = f"{axby(a1, b, c1)}\n{axby(a2, -b, c2)}"
        steps = [
            f"Add the equations; the y-terms cancel: {coef(a1 + a2)} = {num(c1 + c2)}, so x = {num(x0)}.",
            f"Substitute into the first equation: {num(a1)}({num(x0)}) + {coef(b, 'y')} = {num(c1)}, so y = {num(y0)}.",
        ]
        extra = [(Fraction(c1 + c2), f"this stops before dividing by {a1 + a2}")]
    else:
        while True:
            x0, y0 = _pick_solution(rng)
            a1, b1, a2, b2 = (rng.choice(nz(-7, 7)) for _ in range(4))
            if a1 * b2 - a2 * b1 and abs(a1) != abs(a2) and abs(b1) != abs(b2) and min(map(abs, (a1, b1, a2, b2))) > 1:
                break
        if a1 < 0 and b1 < 0:  # write −2x − 4y = −46 as 2x + 4y = 46
            a1, b1 = -a1, -b1
        if a2 < 0 and b2 < 0:
            a2, b2 = -a2, -b2
        c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
        passage = f"{axby(a1, b1, c1)}\n{axby(a2, b2, c2)}"
        steps = [
            f"Multiply the first equation by {num(b2)} and the second by {num(b1)}, then subtract to cancel y:",
            f"{num(a1 * b2 - a2 * b1)}x = {num(c1 * b2 - c2 * b1)}, so x = {num(x0)}.",
            f"Substitute to get y = {num(y0)}.",
        ]
        extra = [(Fraction(c1 * b2 - c2 * b1), "this stops before dividing")]
    ans, other = (Fraction(x0), Fraction(y0)) if ask == "x" else (Fraction(y0), Fraction(x0))
    mistakes = [
        (other, f"this is the value of {'y' if ask == 'x' else 'x'}, not {ask}"),
        (-ans, "this has the wrong sign"),
        *extra,
        (Fraction(x0 + y0), "this is the value of x + y"),
    ]
    stem = f"The solution to the given system of equations is (x, y). What is the value of {ask}?"
    roll = rng.random()
    if level == "medium" and roll < 0.4:  # an expression in x and y
        p_, q_ = rng.choice([(1, -1), (2, 1), (1, 2), (-1, 1), (3, -1), (2, -1)])
        ask = expr_xy(p_, q_)
        ans = Fraction(p_ * x0 + q_ * y0)
        stem = f"The solution to the given system of equations is (x, y). What is the value of {ask}?"
        mistakes = [
            (Fraction(x0), "this is the value of x alone"),
            (Fraction(y0), "this is the value of y alone"),
            (Fraction(p_ * y0 + q_ * x0), "this swaps the values of x and y"),
            (-ans, "this has the wrong sign"),
            (Fraction(x0 + y0), "this is the value of x + y"),
        ]
        steps.append(
            f"So {ask} = {num(p_)}({num(x0)}) + {par(q_)}({num(y0)}) = {num(ans)}.".replace("1(", "(")
            .replace("+ (−1)", f"{MINUS} 1")
            .replace("(−1)(", f"{MINUS}(")
        )
    elif level == "hard" and roll < 0.5:  # the shortcut: add or subtract the equations; no need to solve
        add = rng.random() < 0.6
        p_, q_ = (a1 + a2, b1 + b2) if add else (a1 - a2, b1 - b2)
        if p_ and q_ and abs(p_) != 1 and abs(q_) != 1:
            ask = expr_xy(p_, q_)
            ans = Fraction(c1 + c2 if add else c1 - c2)
            stem = f"The solution to the given system of equations is (x, y). What is the value of {ask}?"
            mistakes = [
                (
                    Fraction(c1 - c2 if add else c1 + c2),
                    f"this {'subtracts' if add else 'adds'} the equations' right sides",
                ),
                (Fraction(x0 + y0), "this is the value of x + y"),
                (Fraction(x0), "this is the value of x"),
                (Fraction(y0), "this is the value of y"),
                (-ans, "this has the wrong sign"),
            ]
            steps = [
                f"{'Add' if add else 'Subtract'} the equations: the left sides give {ask}, and the right sides "
                f"give {num(c1)} {'+' if add else MINUS} {par(c2)} = {num(ans)}.",
                f"There is no need to find x and y first (they are {num(x0)} and {num(y0)}).",
            ]
    item = MathItem(
        "system_coordinate", SYS_SKILL, level, fmt, stem, ans, steps, mistakes, passage=passage, positive_only=False
    )
    item.truth = {"sol": (x0, y0), "ask": ask}
    return item


def expr_xy(p, q) -> str:
    """'2x + y', 'x − y', '−x + 3y'."""
    return axby(p, q, 0)[: -len(" = 0")]


GRAPH_SYS_SETTINGS = TWO_LINE_FRAMES


def system_graph_equations(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A graph of two lines; the choices are systems of equations. Easy: lines in the xy-plane, 'Which system of
    equations is represented by the lines shown?' Medium: a real-world graph (two plants, two savings accounts, two
    tanks), 'Which system of equations could represent the two lines?' Hard: 'Which system of equations has the same
    solution as the system graphed?' (read the crossing point, then check each system). Wrong choices are near
    misses: a sign flipped, the intercepts traded between the lines, slope and intercept swapped, the rate per grid
    line instead of per unit; on hard, systems whose solution is the point with x and y swapped or a sign flipped."""

    def sys_text(l1, l2, decimal=False) -> str:
        return f"y = {linf(l1[0], l1[1], decimal=decimal)}\ny = {linf(l2[0], l2[1], decimal=decimal)}"

    if level == "medium":
        f = rng.choice(TWO_LINE_FRAMES)
        n1, n2 = rng.sample(NAMES, 2)
        names = [x.format(n1=n1, n2=n2) for x in f["names"]]
        intro = f["intro"].format(n1=n1, n2=n2)
        x_n, y_n = 10, rng.choice([10, 12])
        while True:
            xs, ys = Fraction(rng.choice(f["xs"])), Fraction(rng.choice(f["ys"]))
            kx = rng.randint(3, 7)
            d1, d2 = sorted(rng.sample([1, 2, 3], 2), reverse=True)
            j = rng.randint(4, y_n - 2)
            if j - d1 * kx >= 1:
                break
        xi, yi = xs * kx, ys * j
        m1, m2 = d1 * ys / xs, d2 * ys / xs
        b1, b2 = yi - m1 * xi, yi - m2 * xi
        fig = graph_svg(
            f["x"], f["y"], xs, x_n, ys, y_n, lines=[(m1, b1), (m2, b2)], label_every=(1, 2), line_names=names
        )
        dm = any(v.denominator > 1 for v in (m1, m2))
        key = sys_text((m1, b1), (m2, b2), dm)
        traps = [
            (sys_text((m1, b2), (m2, b1), dm), "this trades the starting values between the two lines"),
            (sys_text((b1, m1), (b2, m2), dm), "this swaps each line's rate and starting value"),
        ]
        if xs > 1:
            traps.append(
                (
                    sys_text((m1 * xs, b1), (m2 * xs, b2), dm),
                    f"this uses the change for each grid line ({dec(xs)} units of x), not for 1 unit",
                )
            )
        else:
            traps.append(
                (
                    sys_text((ys * d1 / (xs * 2), b1), (ys * d2 / (xs * 2), b2), dm),
                    "this reads each rise over two grid lines as the change for one",
                )
            )
        traps.append((sys_text((-m1, b1), (-m2, b2), dm), "this gives both lines negative slopes, but both rise"))
        stem = f"{intro} Which system of equations could represent the two lines shown?"
        steps = [
            f"{names[0]} starts at {dec(b1)} and rises {dec(m1)} for each 1 unit of x: y = {linf(m1, b1, decimal=dm)}.",
            f"{names[1]} starts at {dec(b2)} and rises {dec(m2)} for each 1 unit of x: y = {linf(m2, b2, decimal=dm)}.",
        ]
        truth = {"lines": [(m1, b1), (m2, b2)]}
    else:
        while True:
            x0, y0 = rng.choice(nz(-5, 5)), rng.choice(nz(-5, 5))
            m1 = Fraction(rng.choice(nz(-3, 3)))
            m2 = Fraction(rng.choice(nz(-3, 3)), rng.choice([1, 2]))
            b1, b2 = y0 - m1 * x0, y0 - m2 * x0
            if (
                m1 != m2
                and abs(x0) != abs(y0)
                and b1.denominator == 1
                and b2.denominator == 1
                and 0 < abs(b1) <= 7
                and 0 < abs(b2) <= 7
                and b1 != b2
                and abs(m1) != abs(m2)
            ):
                break
        fig = graph_svg(
            "x", "y", 1, 8, 1, 8, lines=[(m1, b1), (m2, b2)], label_every=(2, 2), x_lo=8, y_lo=8, plane=True
        )
        if level == "easy":
            key = sys_text((m1, b1), (m2, b2))
            traps = [
                (sys_text((-m1, b1), (m2, b2)), "this gives one line's slope the wrong sign"),
                (sys_text((m1, -b1), (m2, b2)), "this gives one line's y-intercept the wrong sign"),
                (sys_text((m1, b2), (m2, b1)), "this trades the y-intercepts between the two lines"),
                (sys_text((m1, b1), (-m2, -b2)), "this gives the other line the opposite slope and intercept"),
            ]
            stem = "Which system of equations is represented by the lines shown in the xy-plane?"
            steps = [
                f"One line crosses the y-axis at {num(b1)} and has slope {num(m1)}: y = {linf(m1, b1)}.",
                f"The other crosses at {num(b2)} and has slope {num(m2)}: y = {linf(m2, b2)}.",
            ]
        else:
            while True:
                a1, c1_, a2, c2_ = (rng.choice(nz(-4, 4)) for _ in range(4))
                if a1 * c2_ - a2 * c1_ and abs(a1 * c2_ - a2 * c1_) > 1:
                    break

            def sys_for(px, py) -> str:
                return f"{axby(a1, c1_, a1 * px + c1_ * py)}\n{axby(a2, c2_, a2 * px + c2_ * py)}"

            key = sys_for(x0, y0)
            traps = [
                (sys_for(y0, x0), f"this system's solution is ({num(y0)}, {num(x0)}), with x and y swapped"),
                (sys_for(-x0, y0), f"this system's solution is ({num(-x0)}, {num(y0)})"),
                (sys_for(x0, -y0), f"this system's solution is ({num(x0)}, {num(-y0)})"),
                (sys_for(-x0, -y0), f"this system's solution is ({num(-x0)}, {num(-y0)})"),
            ]
            stem = "Which system of equations has the same solution as the system of equations graphed in the xy-plane?"
            steps = [
                f"The lines cross at ({num(x0)}, {num(y0)}), so that is the solution of the graphed system.",
                f"Only one choice is true for x = {num(x0)} and y = {num(y0)} in both equations.",
            ]
        truth = {"lines": [(m1, b1), (m2, b2)], "sol": (x0, y0)}
    mistakes, seen = [], {key}
    for t, w in traps:
        if t not in seen:
            seen.add(t)
            mistakes.append((t, w))
    item = MathItem(
        "system_graph_equations", SYS_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, figure=fig
    )
    item.truth = truth
    return item


def system_cancel_letters(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Letters that cancel. Medium: '2x + ky = 6 + 5k; 7x + ky = 21 + 5k' (k a nonzero constant): subtract, and ky
    cancels, so 5x = 15; then y = 5. Hard: two letters, 'ax + by = 3a + 5b; 2x − by = 6 − 5b' (a and b positive):
    add, (a + 2)x = 3(a + 2), so x = 3; then 2(3) − by = 6 − 5b gives y = 5. Asks for y on hard (find x first, then
    plug back in). Traps: the other coordinate, the opposite sign, x + y, stopping before dividing."""
    x0, y0 = rng.sample(range(2, 10), 2)
    if level == "medium":
        letter = rng.choice(["k", "c", "r"])
        m1, m2 = sorted(rng.sample(range(1, 10), 2))
        if rng.random() < 0.5:
            m1, m2 = m2, m1
        e1 = f"{coef(m1)} + {letter}y = {num(m1 * x0)} + {y0}{letter}"
        e2 = f"{coef(m2)} + {letter}y = {num(m2 * x0)} + {y0}{letter}"
        passage = f"{e1}\n{e2}"
        stem = (
            f"In the given system of equations, {letter} is a nonzero constant. If (x, y) is the solution to the "
            f"system, what is the value of x?"
        )
        ans = Fraction(x0)
        d = abs(m1 - m2)
        mistakes = [
            (Fraction(y0), "this is the value of y, not x"),
            (Fraction(d * x0), f"this stops before dividing by {d}"),
            (Fraction(x0 + y0), "this is the value of x + y"),
            (Fraction(-x0), "this subtracts in opposite orders on the two sides, so the sign is wrong"),
            (Fraction((m1 + m2) * x0, d), "this adds the constants instead of subtracting them"),
        ]
        steps = [
            f"Subtract one equation from the other; {letter}y and {y0}{letter} cancel: {coef(d)} = {num(d * x0)}.",
            f"So x = {x0} (and then {letter}y = {y0}{letter} gives y = {y0}).",
        ]
        truth = {"sol": (x0, y0), "ask": "x", "letters": (letter,)}
    else:
        c = rng.randint(2, 7)
        e1 = f"ax + by = {x0}a + {y0}b"
        e2 = f"{coef(c)} {MINUS} by = {num(c * x0)} {MINUS} {y0}b"
        passage = f"{e1}\n{e2}"
        stem = (
            "In the given system of equations, a and b are positive constants. If (x, y) is the solution to the "
            "system, what is the value of y?"
        )
        ans = Fraction(y0)
        mistakes = [
            (Fraction(x0), "this is the value of x, which comes first, not y"),
            (
                Fraction(-y0),
                "this has the wrong sign (−by = −5b means y = 5)".replace("5b", f"{y0}b").replace("y = 5", f"y = {y0}"),
            ),
            (Fraction(c * x0), f"this is the value of {coef(c)}, not y"),
            (Fraction(x0 + y0), "this is the value of x + y"),
        ]
        steps = [
            f"Add the equations; by cancels: ax + {coef(c)} = {x0}a + {num(c * x0)}.",
            f"Factor: (a + {c})x = {x0}(a + {c}). Since a is positive, a + {c} is not 0, so x = {x0}.",
            f"Substitute into the second equation: {num(c)}({x0}) {MINUS} by = {num(c * x0)} {MINUS} {y0}b, so "
            f"by = {y0}b and y = {y0}.",
        ]
        truth = {"sol": (x0, y0), "ask": "y", "letters": ("a", "b"), "c": c}
    item = MathItem(
        "system_cancel_letters", SYS_SKILL, level, fmt, stem, ans, steps, mistakes, passage=passage, positive_only=False
    )
    item.truth = truth
    return item


def system_constant_r(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'In the given system, r is a constant. For what value of r does the system have no solution?' ('3x + 5y = 7;
    rx + 10y = 9': r = 6). Infinitely many: '3x + 5y = 7; 6x + 10y = r' (r = 14). Hard: the ratio is not a whole
    number ('4x − ry = 3; 6x − 9y = 5': r = 6), or 'exactly one solution ... which CANNOT be r?'. Traps: the
    coefficient not scaled, the opposite sign, the ratio flipped, the other coefficient."""
    letter = rng.choice(["r", "k", "a"])
    while True:
        A, B = rng.randint(2, 7), rng.choice(nz(-8, 8))
        C = rng.choice(nz(-9, 12))
        if abs(B) > 1 and A != abs(B):
            break
    if level == "medium":
        k = rng.choice([2, 3, 4])
        want = rng.choice(["none", "infinite"])
        if want == "none":
            C2 = k * C + rng.choice(nz(-5, 5))
            passage = f"{axby(A, B, C)}\n{letter}x{plus(k * B)[:3]}{coef(abs(k * B), 'y')} = {num(C2)}"
            ans = Fraction(k * A)
            stem = (
                f"In the given system of equations, {letter} is a constant. For what value of {letter} does the "
                f"system have no solution?"
            )
            mistakes = [
                (Fraction(A), f"this copies {A}, but the y-coefficient was multiplied by {k}"),
                (Fraction(-k * A), "this has the wrong sign"),
                (Fraction(k * B), "this is the y-coefficient, not the x-coefficient"),
                (Fraction(k), "this is the multiplier, not the coefficient"),
            ]
            steps = [
                f"The y-coefficient {num(k * B)} is {k} times {num(B)}.",
                f"For no solution, the x-coefficients must match the same way: {letter} = {k} × {A} = {k * A}.",
                f"Then the constants {num(C2)} and {k} × {num(C)} = {num(k * C)} differ, so the lines are parallel.",
            ]
        else:
            passage = f"{axby(A, B, C)}\n{axby(k * A, k * B, 0)[: -len(' = 0')]} = {letter}"
            ans = Fraction(k * C)
            stem = (
                f"In the given system of equations, {letter} is a constant. For what value of {letter} does the "
                f"system have infinitely many solutions?"
            )
            mistakes = [
                (Fraction(C), f"this copies {num(C)}, but the second equation is {k} times the first"),
                (Fraction(-k * C), "this has the wrong sign"),
                (Fraction(k), "this is the multiplier, not the constant"),
                (Fraction(k * A), "this is the x-coefficient"),
            ]
            steps = [
                f"The second equation's coefficients are {k} times the first's.",
                f"For the same line, the constant must be {k} times as well: {letter} = {k} × {num(C)} = {num(k * C)}.",
            ]
        fr = False
    else:
        p_, q = rng.choice([(2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2)])
        mult = rng.randint(1, 3)
        A1, A2 = p_ * mult, q * mult
        B2 = q * rng.choice(nz(-3, 3))
        if abs(B2) == 1:
            B2 = 2 * q
        r = Fraction(A1 * B2, A2)
        C1, C2 = rng.choice(nz(-9, 9)), rng.choice(nz(-9, 9))
        while Fraction(C1, C2) == Fraction(A1, A2):
            C2 += 1
        passage = f"{coef(A1)} + {letter}y = {num(C1)}\n{axby(A2, B2, C2)}"
        ans = r
        if rng.random() < 0.5:
            stem = (
                f"In the given system of equations, {letter} is a constant. For what value of {letter} does the "
                f"system have no solution?"
            )
        else:
            stem = (
                f"In the given system of equations, {letter} is a constant. The system has exactly one solution. "
                f"Which of the following CANNOT be the value of {letter}?"
            )
        mistakes = [
            (Fraction(A2 * B2, A1), f"this flips the ratio ({num(B2)} × {A2} ÷ {A1})"),
            (-r, "this has the wrong sign"),
            (Fraction(B2), f"this copies {num(B2)}"),
            (Fraction(A1), f"this copies {A1}, the x-coefficient"),
        ]
        steps = [
            f"No solution means the lines are parallel: the coefficients are in the same ratio, "
            f"{A1}/{A2} = {letter}/{num(B2)}.",
            f"So {letter} = {num(B2)} × {A1}/{A2} = {num(r)}; the constants {num(C1)} and {num(C2)} are not in "
            "that ratio, so the lines are parallel, not the same.",
        ]
        if "CANNOT" in stem:
            steps.append(f"Any other value gives exactly one solution, so {letter} cannot be {num(r)}.")
        fr = True
    item = MathItem(
        "system_constant_r",
        SYS_SKILL,
        level,
        fmt if "CANNOT" not in stem else "mcq",
        stem,
        ans,
        steps,
        mistakes,
        passage=passage,
        positive_only=False,
        fractions_ok=fr,
    )
    item.truth = {"letter": letter}
    return item


def system_solve_pair(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'What is the solution (x, y) to the given system of equations?' The choices are ordered pairs. Easy: the
    system is graphed in the xy-plane (read where the lines cross) or is x + y = 10; x − y = 4. Medium: one equation
    must be multiplied first. Hard: fraction coefficients. Traps: x and y swapped, a point on only one of the lines,
    the y-intercept of a line, the opposite signs."""
    x0, y0 = _pick_solution(rng)
    fig = ""
    if level == "easy" and rng.random() < 0.5:  # graphed
        while True:
            x0, y0 = rng.choice(nz(-5, 5)), rng.choice(nz(-5, 5))
            m1 = Fraction(rng.choice(nz(-3, 3)))
            m2 = Fraction(rng.choice(nz(-3, 3)), rng.choice([1, 2]))
            b1, b2 = y0 - m1 * x0, y0 - m2 * x0
            if (
                m1 != m2
                and abs(x0) != abs(y0)
                and b1.denominator == 1
                and b2.denominator == 1
                and abs(b1) <= 8
                and abs(b2) <= 8
                and b1 != b2
            ):
                break
        fig = graph_svg(
            "x", "y", 1, 8, 1, 8, lines=[(m1, b1), (m2, b2)], label_every=(2, 2), x_lo=8, y_lo=8, plane=True
        )
        passage = ""
        stem = "The graph of a system of two linear equations is shown. What is the solution (x, y) to the system?"
        wrong = [
            ((y0, x0), "this swaps x and y"),
            ((0, b1), "this is where one line crosses the y-axis"),
            ((0, b2), "this is where the other line crosses the y-axis"),
            ((-x0, -y0), "this reverses both signs"),
        ]
        steps = [
            f"The lines cross at ({num(x0)}, {num(y0)}).",
            "The point where they cross is on both lines, so it is the solution.",
        ]
    else:
        if level == "easy":
            s_, d_ = x0 + y0, x0 - y0
            passage = f"x + y = {num(s_)}\nx {MINUS} y = {num(d_)}"
            on_first = (x0 + 1, y0 - 1)
            steps = [
                f"Add the equations: 2x = {num(s_ + d_)}, so x = {num(x0)}.",
                f"Then y = {num(s_)} {MINUS} {par(x0)} = {num(y0)}.",
            ]
        elif level == "medium":
            while True:  # multiplying the first equation by k makes the y-terms cancel
                a1, b1 = rng.choice(nz(-5, 5)), rng.choice(nz(-5, 5))
                k = rng.choice([2, 3, -2, -3])
                a2, b2 = rng.choice(nz(-6, 6)), -k * b1
                if abs(a1) != abs(a2) and a1 * k + a2 and abs(b1) > 1:
                    break
            c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
            passage = f"{axby(a1, b1, c1)}\n{axby(a2, b2, c2)}"
            g = math.gcd(a1, b1)
            on_first = (x0 + b1 // g, y0 - a1 // g)
            steps = [
                f"Multiply the first equation by {num(k)}: {axby(k * a1, k * b1, k * c1)}.",
                f"Add it to the second; y cancels: {coef(k * a1 + a2)} = {num(k * c1 + c2)}, so x = {num(x0)}.",
                f"Substitute to get y = {num(y0)}.",
            ]
        else:
            while True:
                p1, q1 = rng.choice([2, 3, 4]), rng.choice([2, 3, 5])
                x0 = p1 * rng.choice(nz(-3, 3))
                y0 = q1 * rng.choice(nz(-3, 3))
                a2, b2 = rng.choice(nz(-4, 4)), rng.choice(nz(-4, 4))
                if p1 != q1 and abs(x0) != abs(y0) and Fraction(1, p1) * b2 != Fraction(1, q1) * a2:
                    break
            c1 = Fraction(x0, p1) + Fraction(y0, q1)
            c2 = a2 * x0 + b2 * y0
            passage = f"x/{p1} + y/{q1} = {num(c1)}\n{axby(a2, b2, c2)}"
            on_first = (x0 + p1, y0 - q1)
            steps = [
                f"Multiply the first equation by {p1 * q1}: {axby(q1, p1, c1 * p1 * q1)}.",
                f"Solve the system: x = {num(x0)} and y = {num(y0)}.",
            ]
        stem = "What is the solution (x, y) to the given system of equations?"
        wrong = [
            ((y0, x0), "this swaps x and y"),
            ((-x0, -y0), "this reverses both signs"),
            ((x0, -y0), "this has the wrong sign on y"),
            (on_first, "this point makes the first equation true but not the second"),
        ]

    def pair(pt) -> str:
        return f"({num(pt[0])}, {num(pt[1])})"

    key = pair((x0, y0))
    mistakes = [(pair(pt), w) for pt, w in wrong if pair(pt) != key]
    item = MathItem(
        "system_solve_pair",
        SYS_SKILL,
        level,
        "mcq",
        stem,
        Fraction(0),
        steps,
        mistakes,
        passage=passage,
        key_text=key,
        figure=fig,
    )
    item.truth = {"sol": (x0, y0)}
    return item


SETUP_FRAMES = [
    dict(
        level="medium",
        kind="value",
        text="A school club sold {n} tickets to a show. Student tickets cost ${a} each, and adult tickets cost ${b} "
        "each. The club collected a total of ${t} from ticket sales. If x is the number of student tickets sold "
        "and y is the number of adult tickets sold, which system of equations represents this situation?",
        a=(3, 6),
        b=(8, 12),
        n=(80, 200),
        unit=1,
    ),
    dict(
        level="medium",
        kind="value",
        text="A farm stand sold {n} baskets of fruit. Small baskets cost ${a} each, and large baskets cost ${b} each. "
        "The stand earned ${t} from the baskets. If x is the number of small baskets sold and y is the number "
        "of large baskets sold, which system of equations represents this situation?",
        a=(4, 7),
        b=(9, 14),
        n=(40, 120),
        unit=1,
    ),
    dict(
        level="hard",
        kind="percent",
        text="A chemist mixes x liters of a {a}% acid solution with y liters of a {b}% acid solution. The mixture is "
        "{n} liters and contains {t} liters of acid. Which system of equations represents this situation?",
        a=(10, 30),
        b=(35, 60),
        n=(20, 120),
        step=5,
    ),
    dict(
        level="hard",
        kind="percent",
        text="Maria invested a total of ${n} in two accounts. One account earns {a}% simple interest per year, and "
        "the other earns {b}% simple interest per year. After one year, she earned ${t} in interest. If x is "
        "the amount, in dollars, invested in the first account and y is the amount, in dollars, invested in the "
        "second account, which system of equations represents this situation?",
        a=(2, 4),
        b=(5, 7),
        n=(2000, 10000),
        step=500,
    ),
    dict(
        level="hard",
        kind="percent",
        text="In a survey of {n} people at a library, {a}% of the x adults surveyed and {b}% of the y teenagers "
        "surveyed said they read every day. A total of {t} people surveyed said they read every day. Which "
        "system of equations represents this situation?",
        a=(20, 40),
        b=(45, 80),
        n=(100, 300),
        step=20,
    ),
    dict(
        level="hard",
        kind="coins",
        text="A jar contains {n} coins, all nickels and dimes, worth a total of ${t}. If x is the number of nickels "
        "and y is the number of dimes, which system of equations represents this situation?",
        a=(5, 5),
        b=(10, 10),
        n=(20, 80),
        step=1,
    ),
]


def system_word_setup(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A word problem that is a system: one equation for the total count (x + y = 120) and one for the value or the
    percent (3x + 8y = 940; 0.3x + 0.25y = 33). 'Which system of equations represents this situation?' Medium: prices
    (whole numbers). Hard: percents or coins, written as decimals. Traps: the two totals swapped, percents written as
    whole numbers, the rates swapped, cents instead of dollars."""
    f = rng.choice([fr for fr in SETUP_FRAMES if fr["level"] == level])
    a = (
        rng.randint(*f["a"])
        if f["kind"] != "percent"
        else rng.randrange(f["a"][0], f["a"][1] + 1, 5 if f["a"][1] > 10 else 1)
    )
    b = (
        rng.randint(*f["b"])
        if f["kind"] != "percent"
        else rng.randrange(f["b"][0], f["b"][1] + 1, 5 if f["b"][1] > 10 else 1)
    )
    step = f.get("step", 1)
    while True:
        n = rng.randrange(f["n"][0], f["n"][1] + 1, step)
        x = rng.randrange(step, n, step) if step > 1 else rng.randint(1, n - 1)
        y = n - x
        if f["kind"] == "value":
            ra, rb, t = Fraction(a), Fraction(b), a * x + b * y
        elif f["kind"] == "percent":
            ra, rb = Fraction(a, 100), Fraction(b, 100)
            t = ra * x + rb * y
        else:
            ra, rb = Fraction(5, 100), Fraction(10, 100)
            t = ra * x + rb * y
        if x != y and (Fraction(t).denominator == 1 or f["kind"] == "coins" and (Fraction(t) * 100).denominator == 1):
            break
    dm = f["kind"] != "value"
    tshow = money(t)[1:] if f["kind"] == "coins" else num(t)
    stem = f["text"].format(n=num(n), a=a, b=b, t=tshow)
    total = f"x + y = {num(n)}"

    def val(p, q, rhs) -> str:
        return axby(p, q, 0, decimal=dm)[: -len(" = 0")] + f" = {rhs}"

    key = f"{total}\n{val(ra, rb, tshow)}"
    mistakes = [
        (f"x + y = {tshow}\n{val(ra, rb, num(n))}", "this swaps the two totals"),
        (f"{total}\n{val(rb, ra, tshow)}", "this swaps the two rates"),
    ]
    if f["kind"] == "value":
        mistakes += [
            (
                f"{total}\n{axby(a + b, 0, 0)[:-4].replace('x', '(x + y)')} = {tshow}",
                f"this uses {a} + {b} for each one",
            ),
            (f"x + y = {num(n)}\n{coef(a)} = {tshow}", "this leaves out the second kind"),
        ]
    elif f["kind"] == "percent":
        mistakes += [
            (f"{total}\n{axby(a, b, 0)[:-4]} = {tshow}", "this writes the percents as whole numbers"),
            (f"{total}\n{val(ra * 10, rb * 10, tshow)}", "this moves the decimal point one place too far"),
        ]
    else:
        mistakes += [
            (f"{total}\n5x + 10y = {tshow}", "this uses cents for the coins but dollars for the total"),
            (f"{total}\n0.1x + 0.05y = {tshow}", "this swaps the values of nickels and dimes"),
        ]
        mistakes = [m for m in mistakes if "swaps the two rates" not in m[1]]
    steps = [
        f"The two parts add up to {num(n)}: {total}.",
        f"Each one contributes its own amount to the total of {tshow}: {val(ra, rb, tshow)}.",
    ]
    if f["kind"] == "percent":
        steps.insert(1, f"{a}% is {dec(ra)} and {b}% is {dec(rb)}.")
    item = MathItem("system_word_setup", SYS_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"n": n, "t": t, "rates": (ra, rb), "sol": (x, y)}
    return item


TIMES_FRAMES = [
    dict(
        times="In a survey, {kw} as many students chose soccer as their favorite sport as chose tennis.",
        plus_="In a survey about favorite sports, the number of students who chose soccer was {e} more than {kw} the "
        "number who chose tennis.",
        total="A total of {n} students chose soccer or tennis.",
        diff="The number of students who chose soccer was {d} more than the number who chose tennis.",
        qa="How many students chose soccer?",
        qb="How many students chose tennis?",
        la="students who chose soccer",
        lb="students who chose tennis",
    ),
    dict(
        times="At a bake sale, {kw} as many cookies as brownies were sold.",
        plus_="At a bake sale, the number of cookies sold was {e} more than {kw} the number of brownies sold.",
        total="A total of {n} cookies and brownies were sold.",
        diff="The bake sale sold {d} more cookies than brownies.",
        qa="How many cookies were sold?",
        qb="How many brownies were sold?",
        la="cookies",
        lb="brownies",
    ),
    dict(
        times="A plant nursery has {kw} as many pine seedlings as oak seedlings.",
        plus_="At a plant nursery, the number of pine seedlings is {e} more than {kw} the number of oak seedlings.",
        total="The nursery has {n} pine and oak seedlings in all.",
        diff="The nursery has {d} more pine seedlings than oak seedlings.",
        qa="How many pine seedlings does the nursery have?",
        qb="How many oak seedlings does the nursery have?",
        la="pine seedlings",
        lb="oak seedlings",
    ),
    dict(
        times="In a class election, Andre received {kw} as many votes as Bianca.",
        plus_="In a class election, the number of votes Andre received was {e} more than {kw} the number Bianca "
        "received.",
        total="Andre and Bianca received a total of {n} votes.",
        diff="Andre received {d} more votes than Bianca.",
        qa="How many votes did Andre receive?",
        qb="How many votes did Bianca receive?",
        la="votes for Andre",
        lb="votes for Bianca",
    ),
]
TIMES_WORDS = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}


def system_times_more(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Three times as many students chose soccer as chose tennis, and 40 more students chose soccer than tennis.
    How many students chose tennis?' (a = 3b and a − b = 40, so b = 20). Easy: a total instead of a difference
    (a + b = 120). Medium: the difference. Hard: '4 more than three times the number' (a = 3b + 4), with the
    difference. Students can write the system or plug in the choices. Traps: the other group, dividing by k instead
    of k − 1 (or k + 1), the given number itself, leaving out the extra amount."""
    f = rng.choice(TIMES_FRAMES)
    k = rng.choice([2, 3, 4, 5]) if level != "hard" else rng.choice([2, 3, 4])
    b = k * rng.randint(2, 10)  # a multiple of k, so dividing by k (the main trap) comes out even
    extra = (k - 1) * rng.randint(2, 6) if level == "hard" else 0
    a = k * b + extra
    kw = TIMES_WORDS[k]
    intro = f["plus_"].format(e=extra, kw=kw) if level == "hard" else f["times"].format(kw=kw)
    ask_small = rng.random() < 0.6
    if level == "easy":
        total = a + b
        info = f["total"].format(n=num(total))
        system = f"a = {k}b and a + b = {num(total)}"
        steps = [
            f"Let a be the number of {f['la']} and b the number of {f['lb']}: {system}.",
            f"Substitute: {k}b + b = {num(total)}, so {k + 1}b = {num(total)} and b = {b}; then a = {a}.",
        ]
        traps = [
            (Fraction(total, k), f"this divides {num(total)} by {k} instead of {k + 1}"),
            (Fraction(total, 2), "this splits the total in half"),
            (Fraction(total - k), f"this subtracts {k} from the total"),
        ]
    else:
        diff = a - b
        info = f["diff"].format(d=num(diff))
        system = f"a = {k}b{plus(extra) if extra else ''} and a {MINUS} b = {num(diff)}"
        steps = [
            f"Let a be the number of {f['la']} and b the number of {f['lb']}: {system}.",
            f"Substitute: {k}b{plus(extra) if extra else ''} {MINUS} b = {num(diff)}, so {coef(k - 1, 'b')} = "
            f"{num(diff - extra)} and b = {b}; then a = {a}.",
        ]
        traps = [
            (Fraction(diff, k), f"this divides {num(diff)} by {k} instead of {k - 1}"),
            (Fraction(diff), f"this is the difference, {num(diff)}"),
            (Fraction(diff, k + 1), f"this divides by {k + 1}, as if {num(diff)} were the total"),
        ]
        if extra:
            traps.insert(0, (Fraction(diff, k - 1), f"this leaves out the extra {extra}"))
            traps.append((Fraction(diff + extra, k - 1), f"this adds the extra {extra} instead of subtracting it"))
    ans = Fraction(b if ask_small else a)
    other = Fraction(a if ask_small else b)
    if not ask_small:  # the traps above are values of b; for a, use what each would make a
        traps = [
            (k * v + extra, f"{w}, which gives {num(k * v + extra)} {f['la']}")
            for v, w in traps
            if Fraction(v).denominator == 1
        ]
    mistakes = [(other, f"this is the number of {f['la'] if ask_small else f['lb']}"), *traps]
    stem = f"{intro} {info} {f['qb'] if ask_small else f['qa']}"
    item = MathItem("system_times_more", SYS_SKILL, level, fmt, stem, ans, steps, mistakes)
    item.truth = {"k": k, "extra": extra, "a": a, "b": b, "ask": "b" if ask_small else "a", "level": level}
    return item


# ---------------------------------------------------- linear inequalities in one or two variables (Sebastian, Nov 25)

INEQ_SKILL = "Linear inequalities in one or two variables"
SYMS = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}
FLIP = {"<": ">", ">": "<", "≤": "≥", "≥": "≤"}
LOOSE = {"<": "≤", ">": "≥", "≤": "<", "≥": ">"}  # strict <-> not strict

MAX_FRAMES = [
    dict(
        text="{name} has ${B} to spend at an arcade. Admission costs ${F}, and each game costs ${r}.",
        q="What is the greatest number of games {name} can play?",
        F=(8, 15),
        r=(Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3)),
        B=(40, 80),
        unit="games",
        money=True,
    ),
    dict(
        text="A delivery van can carry at most {B} pounds. The driver and equipment weigh {F} pounds, and each box "
        "of supplies weighs {r} pounds.",
        q="What is the greatest number of boxes of supplies the van can carry?",
        F=(180, 320),
        r=(35, 40, 45, 55),
        B=(1200, 2000),
        unit="boxes",
        money=False,
    ),
    dict(
        text="{name}'s phone plan costs ${F} each month plus ${r} for each gigabyte of data used. {name} wants to "
        "spend at most ${B} on the plan this month.",
        q="What is the greatest number of whole gigabytes of data {name} can use this month?",
        F=(20, 35),
        r=(Fraction(9, 4), Fraction(7, 2), Fraction(4), Fraction(9, 2)),
        B=(45, 75),
        unit="gigabytes",
        money=True,
    ),
]
MIN_FRAMES = [
    dict(
        text="A school club needs to raise at least ${B} for a trip. It has already raised ${F} and sells candles "
        "for ${r} each.",
        q="What is the least number of candles the club must sell to reach its goal?",
        F=(60, 150),
        r=(6, 7, 8, 9),
        B=(400, 700),
        unit="candles",
        money=True,
    ),
    dict(
        text="{name} needs at least {B} points to earn a badge in a game. {name} already has {F} points and earns "
        "{r} points for each level completed.",
        q="What is the least number of levels {name} must complete to earn the badge?",
        F=(100, 300),
        r=(15, 18, 24, 35),
        B=(700, 1200),
        unit="levels",
        money=False,
    ),
]


def approx(q: Fraction) -> str:
    """'40.5', or '40.22...' for a repeating decimal."""
    return dec(q) if terminating(q) else f"{float(q):.2f}..."


def ineq_max_word(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Kai has $60 to spend at an arcade. Admission costs $12, and each game costs $1.50. What is the greatest number
    of games Kai can play?' 12 + 1.5g ≤ 60, so g ≤ 32 (rounded down). Medium: 'at least' (round up). Hard: two
    variables (12 pens already chosen, the most notebooks). Traps: rounding the wrong way, leaving out the fixed
    part, dividing the budget by the wrong price."""
    name = rng.choice(NAMES)
    if level == "hard":
        a, b = rng.sample([3, 4, 5, 6], 2)
        while True:
            y = rng.randint(4, 12)
            B = rng.randrange(60, 160, 5)
            left = B - b * y
            if left > 3 * a and left % a:
                break
        ans = Fraction(left // a)
        stem = (
            f"{name} can spend at most ${B} on x notebooks and y binders. Each notebook costs ${a}, and each "
            f"binder costs ${b}. If {name} buys {y} binders, what is the greatest number of notebooks {name} can "
            f"buy?"
        )
        mistakes = [
            (ans + 1, f"this rounds {num(Fraction(left, a))} up, but then the cost is over ${B}"),
            (Fraction(B // a), f"this leaves out the {y} binders"),
            (Fraction(left // b), f"this divides by ${b}, the price of a binder"),
            (Fraction(B // a) - y, f"this subtracts {y} notebooks instead of the cost of {y} binders"),
            (
                ans - 1,
                f"{num(ans - 1)} notebooks fit the budget, but so do {num(ans)}; the question asks for the greatest",
            ),
        ]
        steps = [
            f"{coef(a)} + {b}({y}) ≤ {B}, so {coef(a)} ≤ {left} and x ≤ {num(Fraction(left, a))}.",
            f"x must be a whole number, so the greatest is {num(ans)}.",
        ]
        item = MathItem("ineq_max_word", INEQ_SKILL, level, fmt, stem, ans, steps, mistakes)
        item.truth = {"kind": "max2", "a": a, "b": b, "B": B, "y": y}
        return item
    frames = MAX_FRAMES if level == "easy" else MIN_FRAMES
    f = rng.choice(frames)
    while True:
        F_ = rng.randint(*f["F"])
        r = Fraction(rng.choice(f["r"]))
        B = rng.randrange(f["B"][0], f["B"][1] + 1, 5 if f["money"] else 25)
        q = Fraction(B - F_) / r
        if q.denominator > 1 and q > 4 and Fraction(B) / r != q:
            break
    show = (lambda v: money(v)[1:]) if f["money"] else num  # noqa: E731
    stem = f"{f['text'].format(name=name, B=num(B), F=num(F_), r=show(r))} {f['q'].format(name=name)}"
    if level == "easy":
        ans = Fraction(math.floor(q))
        mistakes = [
            (ans + 1, f"this rounds {approx(q)} up, but then the total is over the limit"),
            (Fraction(math.floor(Fraction(B) / r)), f"this leaves out the {show(F_)} at the start"),
            (Fraction(math.floor(Fraction(B + F_) / r)), f"this adds the {show(F_)} instead of subtracting it"),
            (Fraction(math.floor(Fraction(B) / (r + F_))), f"this adds the {show(F_)} to the cost of each one"),
            (ans - 1, f"{num(ans - 1)} fits the limit, but so does {num(ans)}; the question asks for the greatest"),
        ]
        rel = "≤"
    else:
        ans = Fraction(math.ceil(q))
        mistakes = [
            (ans - 1, f"this rounds {approx(q)} down, but then the total is short of the goal"),
            (Fraction(math.ceil(Fraction(B) / r)), f"this leaves out the {show(F_)} already there"),
            (Fraction(math.ceil(Fraction(B + F_) / r)), f"this adds the {show(F_)} instead of subtracting it"),
            (Fraction(math.ceil(Fraction(B) / (r + F_))), f"this adds the {show(F_)} to the amount for each one"),
            (ans + 1, f"{num(ans + 1)} reaches the goal, but so does {num(ans)}; the question asks for the least"),
        ]
        rel = "≥"
    steps = [
        f"{show(F_)} + {show(r)}n {rel} {show(B)}, so {show(r)}n {rel} {show(B - F_)} and n {rel} {approx(q)}.",
        f"n must be a whole number, so the {'greatest' if level == 'easy' else 'least'} is {num(ans)}.",
    ]
    item = MathItem("ineq_max_word", INEQ_SKILL, level, fmt, stem, ans, steps, mistakes[:3] + mistakes[3:])
    item.truth = {"kind": "max" if level == "easy" else "min", "F": F_, "r": r, "B": B}
    return item


SETUP_INEQ = [
    dict(
        text="An elevator can safely carry at most {B} pounds. A worker who weighs {F} pounds is moving boxes that "
        "weigh {r} pounds each. Which inequality represents the number of boxes, x, the worker can take in "
        "the elevator in one trip?",
        rel="≤",
        F=(150, 220),
        r=(30, 45, 50, 60),
        B=(1000, 1500),
    ),
    dict(
        text="A student has {F} points in a class and earns {r} points for each extra-credit project. The student "
        "needs more than {B} points to earn an A. Which inequality represents the number of projects, x, that "
        "would earn the student an A?",
        rel=">",
        F=(600, 800),
        r=(10, 12, 15, 20),
        B=(850, 950),
    ),
    dict(
        text="A gym charges a ${F} sign-up fee plus ${r} for each class. Tomas wants to spend less than ${B} at the "
        "gym this month. Which inequality represents the number of classes, x, Tomas can take?",
        rel="<",
        F=(20, 40),
        r=(8, 12, 15),
        B=(120, 200),
    ),
    dict(
        text="A food bank has {F} cans and receives {r} more cans each day. It needs at least {B} cans for a holiday "
        "drive. Which inequality represents the number of days, x, after which the food bank will have "
        "enough cans?",
        rel="≥",
        F=(200, 400),
        r=(35, 40, 55),
        B=(1200, 2000),
    ),
]


def ineq_setup(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'An elevator can safely carry at most 1,200 pounds. A worker who weighs 180 pounds is moving boxes that weigh
    45 pounds each. Which inequality represents the number of boxes?' (180 + 45x ≤ 1,200). Medium covers more than /
    less than / at least / at most. Hard: a range, 'at least $400 and at most $650' (400 ≤ 100 + 15x ≤ 650).
    Traps: the sign backward, strict vs. not strict, the two numbers swapped, the fixed part left out."""
    if level == "hard":
        F_, r = rng.randrange(75, 200, 25), rng.choice([12, 15, 18, 20, 25])
        lo = rng.randrange(300, 500, 50)
        hi = lo + rng.randrange(150, 400, 50)
        stem = (
            f"A caterer charges a ${F_} booking fee plus ${r} for each guest. A family wants the total cost to be "
            f"at least ${num(lo)} and at most ${num(hi)}. Which inequality represents the possible numbers of "
            f"guests, x?"
        )
        mid = f"{F_} + {coef(r)}"
        key = f"{num(lo)} ≤ {mid} ≤ {num(hi)}"
        mistakes = [
            (
                f"{num(lo)} < {mid} < {num(hi)}",
                "this leaves out the two end values, but 'at least' and 'at most' include them",
            ),
            (f"{num(lo)} ≤ {coef(r)} ≤ {num(hi)}", f"this leaves out the ${F_} booking fee"),
            (f"{num(lo)} ≤ {r} + {coef(F_)} ≤ {num(hi)}", "this swaps the fee and the cost per guest"),
            (f"{num(lo - F_)} ≤ {mid} ≤ {num(hi - F_)}", "this subtracts the fee from the limits as well"),
        ]
        steps = [f"The total cost is {mid}.", f"At least {num(lo)} and at most {num(hi)}: {key}."]
        item = MathItem("ineq_setup", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
        item.truth = {"F": F_, "r": r, "lo": lo, "hi": hi, "rel": "range"}
        return item
    frames = [f for f in SETUP_INEQ if f["rel"] in "≤≥"] if level == "easy" else SETUP_INEQ
    f = rng.choice(frames)
    F_, r = rng.randint(*f["F"]), rng.choice(f["r"])
    B = rng.randrange(f["B"][0], f["B"][1] + 1, 10)
    stem = f["text"].format(F=num(F_), r=r, B=num(B))
    rel = f["rel"]
    lhs = f"{num(F_)} + {coef(r)}"
    key = f"{lhs} {rel} {num(B)}"
    mistakes = [
        (f"{lhs} {FLIP[rel]} {num(B)}", "this has the inequality sign backward"),
        (
            f"{lhs} {LOOSE[rel]} {num(B)}",
            "this has the wrong kind of inequality: "
            + (
                "'more than' and 'less than' leave out the limit itself"
                if rel in "<>"
                else "'at least' and 'at most' include the limit itself"
            ),
        ),
        (f"{r} + {coef(F_)} {rel} {num(B)}", "this swaps the starting amount and the amount for each one"),
        (f"{coef(r)} {rel} {num(B)}", f"this leaves out the starting {num(F_)}"),
    ]
    steps = [f"The total is {lhs}.", f"The words mean {rel}: {key}."]
    item = MathItem("ineq_setup", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"F": F_, "r": r, "B": B, "rel": rel}
    return item


def ineq_graph(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A shaded region in the xy-plane, bounded by a solid line (≤ or ≥) or a dashed line (< or >). 'Which
    inequality is represented by the shaded region?' Easy: the four signs with the same line. Medium: a fraction
    slope, and one choice has the slope's sign flipped. Hard: the choices are in standard form (2x − 3y < 6), where
    solving for y with a negative coefficient flips the sign. Traps: the sign backward, solid vs. dashed, the slope's
    sign, forgetting to flip."""
    while True:
        m = Fraction(rng.choice(nz(-3, 3)), 1 if level == "easy" else rng.choice([1, 2, 3]))
        b = rng.choice(nz(-5, 5))
        if abs(m) != 1 or level == "easy":
            break
    above, strict = rng.random() < 0.5, rng.random() < 0.5
    rel = (">" if strict else "≥") if above else ("<" if strict else "≤")
    fig = graph_svg(
        "x",
        "y",
        1,
        8,
        1,
        8,
        lines=[(m, b)],
        label_every=(2, 2),
        x_lo=8,
        y_lo=8,
        plane=True,
        shade=[(m, b, "above" if above else "below")],
        dashed=(0,) if strict else (),
    )
    stem = "The shaded region shown represents the solutions to which inequality?"
    why = {
        FLIP[rel]: "this has the sign backward: the shading is on the other side of the line",
        LOOSE[rel]: "",
        FLIP[LOOSE[rel]]: "this has the sign backward and the wrong kind of line",
    }
    why[LOOSE[rel]] = (
        "a dashed line means the points on the line are not included, so the sign must be strict"
        if strict
        else "a solid line means the points on the line are included, so the sign must include equality"
    )
    if level != "hard":
        key = f"y {rel} {linf(m, b)}"
        mistakes = [(f"y {r2} {linf(m, b)}", w) for r2, w in why.items()]
        if level == "medium":
            mistakes[2] = (f"y {rel} {linf(-m, b)}", "this uses the slope with the wrong sign")
    else:
        A, B, C = -m.numerator, m.denominator, b * m.denominator  # −px + qy = qb, i.e. y = mx + b
        if rng.random() < 0.6:  # write it with a negative y-coefficient, so the sign flips when solving for y
            A, B, C = -A, -B, -C
        flip = B < 0
        r_key = FLIP[rel] if flip else rel
        body = axby(A, B, 0)[: -len(" = 0")]
        key = f"{body} {r_key} {num(C)}"
        mistakes = [
            (
                f"{body} {FLIP[r_key]} {num(C)}",
                "this does not flip the sign when dividing by a negative number"
                if flip
                else "this has the sign backward",
            ),
            (f"{body} {LOOSE[r_key]} {num(C)}", why[LOOSE[rel]]),
            (f"{body} {FLIP[LOOSE[r_key]]} {num(C)}", "this has the sign backward and the wrong kind of line"),
        ]
    steps = [
        f"The line is y = {linf(m, b)} (y-intercept {num(b)}, slope {num(m)}).",
        f"The line is {'dashed' if strict else 'solid'}, so the sign is {'strict' if strict else 'not strict'}; "
        f"the shading is {'above' if above else 'below'} the line, so y {rel} {linf(m, b)}.",
    ]
    if level == "hard":
        steps.append(
            f"Written in standard form, that is {key}"
            + (" (the sign flips because the y-coefficient is negative)." if B < 0 else ".")
        )
    item = MathItem(
        "ineq_graph", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, figure=fig
    )
    item.truth = {"m": m, "b": b, "rel": rel}
    return item


def _holds(rel: str, left, right) -> bool:
    return {"<": left < right, "≤": left <= right, ">": left > right, "≥": left >= right}[rel]


def ineq_system_points(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'y ≤ 2x + 3 and y > −x + 1. Which point (x, y) is a solution to the system?' (or 'is NOT a solution').
    Hard uses decimal slopes (y < 0.5x + 2). The tempting wrong point sits exactly on the dashed boundary of the strict
    inequality; others satisfy only one of the two."""
    while True:  # retry until every kind of point exists
        while True:
            if level == "hard":
                m1, m2 = Fraction(rng.choice([-3, -1, 1, 3]), 2), Fraction(rng.choice([-4, -2, 2, 4, 6]), 4)
            else:
                m1, m2 = Fraction(rng.choice(nz(-3, 3))), Fraction(rng.choice(nz(-3, 3)))
            b1, b2 = rng.choice(nz(-5, 5)), rng.choice(nz(-5, 5))
            if m1 != m2 and m2.denominator <= 2:
                break
        r1, r2 = rng.choice(["≤", "≥"]), rng.choice(["<", ">"])  # one solid boundary, one dashed
        dm = level == "hard"
        e1, e2 = f"y {r1} {linf(m1, b1, decimal=dm)}", f"y {r2} {linf(m2, b2, decimal=dm)}"
        if rng.random() < 0.5:
            e1, e2, m1, m2, b1, b2, r1, r2 = e2, e1, m2, m1, b2, b1, r2, r1

        def ok1(x, y, r=r1, m=m1, b=b1):
            return _holds(r, y, m * x + b)

        def ok2(x, y, r=r2, m=m2, b=b2):
            return _holds(r, y, m * x + b)

        pts = [(x, y) for x in range(-6, 7, 2 if dm else 1) for y in range(-8, 9)]
        both = [p_ for p_ in pts if ok1(*p_) and ok2(*p_)]
        only1 = [p_ for p_ in pts if ok1(*p_) and not ok2(*p_)]
        only2 = [p_ for p_ in pts if ok2(*p_) and not ok1(*p_)]
        strict_m, strict_b, strict_ok = (m1, b1, ok1) if r1 in "<>" else (m2, b2, ok2)
        other_ok = ok2 if strict_ok is ok1 else ok1
        on_dashed = [(x, y) for x, y in pts if y == strict_m * x + strict_b and other_ok(x, y)]
        if len(both) >= 4 and len(only1 + only2) >= 2 and on_dashed:
            break
    negative = rng.random() < 0.35
    if negative:  # which point is NOT a solution: three that work, one that fails (often on the dashed line)
        bad = rng.choice(on_dashed or only1 or only2)
        good = rng.sample(both, 3)
        key_pt = bad
        mistakes_pts = [(p_, "this point makes both inequalities true") for p_ in good]
        stem = "Which point (x, y) is NOT a solution to the given system of inequalities?"
    else:
        key_pt = rng.choice(both)
        cand = []
        if on_dashed:
            cand.append(
                (
                    rng.choice(on_dashed),
                    "this point is on the dashed line, so it does not satisfy the strict inequality",
                )
            )
        if only1:
            cand.append((rng.choice(only1), "this point satisfies only the first inequality"))
        if only2:
            cand.append((rng.choice(only2), "this point satisfies only the second inequality"))
        while len(cand) < 3:
            cand.append((rng.choice(only1 + only2), "this point satisfies only one of the inequalities"))
        mistakes_pts = cand
        stem = "Which point (x, y) is a solution to the given system of inequalities?"

    def pair(pt) -> str:
        return f"({num(pt[0])}, {num(pt[1])})"

    key = pair(key_pt)
    mistakes, seen = [], {key}
    for pt, w in mistakes_pts:
        while pair(pt) in seen:  # pick another point of the same kind
            pool = both if "both" in w else (only1 + only2 + on_dashed)
            pt = rng.choice(pool)
        seen.add(pair(pt))
        mistakes.append((pair(pt), w))
    steps = [
        f"Substitute each point into both inequalities: {e1} and {e2}.",
        f"Only {key} {'fails at least one' if negative else 'makes both true'}.",
    ]
    item = MathItem(
        "ineq_system_points",
        INEQ_SKILL,
        level,
        "mcq",
        stem,
        Fraction(0),
        steps,
        mistakes,
        passage=f"{e1}\n{e2}",
        key_text=key,
    )
    item.truth = {"negative": negative}
    return item


def ineq_fence_range(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Sebastian's fencing problem. Medium: 'A rectangular pen will use at most 100 yards of fencing. One side is 32
    yards long. Which inequality gives the possible lengths x of the other side?' (0 < x ≤ 18). Hard adds an area
    limit: 'The area must be at least 200 square yards' (6.25 ≤ x ≤ 18); 'more than' makes it strict. Traps:
    fencing only one of each side, the area limit and the fence limit swapped, the wrong kind of inequality."""
    s_ = rng.choice([12, 16, 20, 24, 28, 32, 36, 40])
    h = rng.randint(8, 30)
    while h == s_:
        h = rng.randint(8, 30)
    Fence = 2 * s_ + 2 * h
    if level == "medium":
        stem = (
            f"A farmer will use at most {Fence} yards of fencing to enclose a rectangular pen. One side of the pen "
            f"is {s_} yards long. Which inequality represents the possible lengths x, in yards, of the "
            f"other side of the pen?"
        )
        key = f"0 < x ≤ {h}"
        mistakes = grid2(
            lambda lo_rel, hi: f"0 {lo_rel} x ≤ {hi}",
            "<",
            "≤",
            "a side of length 0 does not make a pen, so x must be more than 0",
            h,
            Fence - 2 * s_,
            f"this counts the unknown side once but both {s_}-yard sides",
        )
        steps = [
            f"Perimeter: 2({s_}) + 2x ≤ {Fence}, so 2x ≤ {Fence - 2 * s_} and x ≤ {h}.",
            f"A length must be positive, so {key}.",
        ]
        truth = {"lo": 0, "lo_strict": True, "hi": h}
    else:
        lo = Fraction(rng.randint(2, max(3, h - 3)) * 4 + rng.choice([0, 1, 2, 3]), 4)
        while lo >= h or lo <= 1:
            lo = Fraction(rng.randint(4, 4 * h - 4), 4)
        A = lo * s_
        strict = rng.random() < 0.4
        word = "more than" if strict else "at least"
        lo_rel = "<" if strict else "≤"
        stem = (
            f"A farmer will use at most {Fence} yards of fencing to enclose a rectangular pen. One side of the pen "
            f"is {s_} yards long, and the area of the pen must be {word} {num(A)} square yards. Which "
            f"inequality represents the possible lengths x, in yards, of the other side of the pen?"
        )
        key = f"{dec(lo)} {lo_rel} x ≤ {h}"
        hi_wrong = Fence - 2 * s_ if rng.random() < 0.5 else num(Fraction(Fence - s_, 2))
        why_hi = (
            "this counts the unknown side once in the fencing"
            if hi_wrong == Fence - 2 * s_
            else f"this counts the {s_}-yard side only once in the fencing"
        )
        mistakes = grid2(
            lambda r_, hi: f"{dec(lo)} {r_} x ≤ {hi}",
            lo_rel,
            LOOSE[lo_rel],
            (
                "'more than' leaves out the area limit itself, so the sign must be <"
                if strict
                else "'at least' includes the area limit itself, so the sign must be ≤"
            ),
            h,
            hi_wrong,
            why_hi,
        )
        steps = [
            f"Fencing: 2({s_}) + 2x ≤ {Fence}, so x ≤ {h}.",
            f"Area: {s_}x {'>' if strict else '≥'} {num(A)}, so x {'>' if strict else '≥'} {dec(lo)}.",
            f"Together: {key}.",
        ]
        truth = {"lo": lo, "lo_strict": strict, "hi": h}
    item = MathItem("ineq_fence_range", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {**truth, "s": s_, "fence": Fence}
    return item


# ------------------------------------------------------- two variables and inequalities: more (Sebastian, Nov 25)


def pos_axby(A, B, C) -> str:
    """axby with a positive x-coefficient (−10x + 6y = 11 is written 10x − 6y = −11)."""
    return axby(-A, -B, -C) if A < 0 else axby(A, B, C)


def _pt(x, y) -> str:
    return f"({num(x)}, {num(y)})"


def intercepts_std(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'What is the x-intercept of the graph of 3x + 4y = 24 in the xy-plane?' (8, 0). Medium: the
    y-intercept, with a negative y-coefficient (5x − 2y = 20 gives (0, −10)). Hard: 'The graph of kx + 3y = 24 has an
    x-intercept at (6, 0). What is the value of k?' Traps: the other intercept, dividing by the wrong coefficient,
    the sign, forgetting to divide."""
    while True:
        A, B = rng.randint(2, 9), rng.choice(nz(-9, 9))
        if abs(B) > 1 and A != abs(B):
            break
    C = A * abs(B) * rng.choice([1, 2])
    xi, yi = Fraction(C, A), Fraction(C, B)
    eq = axby(A, B, C)
    if level == "hard":
        k_ = A
        if rng.random() < 0.5:  # a negative x-intercept makes k negative
            A, xi, k_ = -A, -xi, -A
        stem = (
            f"In the xy-plane, the graph of kx {'+' if B > 0 else MINUS} {coef(abs(B), 'y')} = {num(C)}, where k "
            f"is a constant, has an x-intercept at {_pt(xi, 0)}. What is the value of k?"
        )
        ans = Fraction(k_)
        mistakes = [
            (yi if yi > 0 else -yi, f"this divides {num(C)} by {num(B)}, the y-coefficient"),
            (Fraction(C) - xi, f"this subtracts {num(xi)} instead of dividing"),
            (xi, "this is the x-intercept itself, not k"),
            (Fraction(C) * xi, "this multiplies instead of dividing"),
        ]
        steps = [f"At {_pt(xi, 0)}, y = 0, so k({num(xi)}) = {num(C)}.", f"So k = {num(C)} ÷ {num(xi)} = {num(ans)}."]
        item = MathItem("intercepts_std", TWO_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
        item.truth = {"A": A, "B": B, "C": C}
        return item
    ask_x = level == "easy"
    stem = f"What is the {'x' if ask_x else 'y'}-intercept of the graph of {eq} in the xy-plane?"
    key = _pt(xi, 0) if ask_x else _pt(0, yi)
    wrong = [
        ((0, yi) if ask_x else (xi, 0), f"this is the {'y' if ask_x else 'x'}-intercept"),
        ((Fraction(C, B), 0) if ask_x else (0, Fraction(C, A)), "this divides by the wrong coefficient"),
        ((-xi, 0) if ask_x else (0, -yi), "this has the wrong sign"),
        ((C, 0) if ask_x else (0, C), f"this forgets to divide by the {'x' if ask_x else 'y'}-coefficient"),
    ]
    mistakes, seen = [], {key}
    for (px, py), w in wrong:
        t = _pt(px, py)
        if t not in seen and Fraction(px).denominator == 1 and Fraction(py).denominator == 1:
            seen.add(t)
            mistakes.append((t, w))
    steps = [
        f"The {'x' if ask_x else 'y'}-intercept is where {'y' if ask_x else 'x'} = 0.",
        (
            f"{coef(A)} = {num(C)}, so x = {num(xi)}: {key}."
            if ask_x
            else f"{coef(B, 'y')} = {num(C)}, so y = {num(yi)}: {key}."
        ),
    ]
    item = MathItem("intercepts_std", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"A": A, "B": B, "C": C}
    return item


def point_on_line(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'Which point lies on the graph of y = 3x − 4 in the xy-plane?' Medium: the line is in standard form
    (3x − 2y = 12). Hard: 'The point (k, 5) lies on the line 4x − 3y = 9. What is the value of k?' Traps: the point's
    coordinates swapped, a sign flipped, a point on the line with the y-coefficient's sign flipped."""
    if level == "hard":
        while True:
            A, B = rng.randint(2, 7), rng.choice(nz(-7, 7))
            k_, y0 = rng.choice(nz(-8, 9)), rng.choice(nz(-8, 9))
            if abs(B) > 1 and A != abs(B) and k_ != y0:
                break
        C = A * k_ + B * y0
        stem = f"The point ({'k'}, {num(y0)}) lies on the line {axby(A, B, C)} in the xy-plane. What is the value of k?"
        ans = Fraction(k_)
        mistakes = [
            (Fraction(C + B * y0, A), f"this moves {coef(B, 'y')} to the other side with the wrong sign"),
            (Fraction(C, A), f"this leaves out the {coef(B, 'y')} term"),
            (Fraction(C - A * y0, B), "this puts the y-value in for x and solves for y instead"),
            (Fraction(C - B * y0), f"this stops before dividing by {A}"),
            (-ans, "this has the wrong sign"),
        ]
        steps = [
            f"Substitute y = {num(y0)}: {num(A)}k {plus(B * y0)} = {num(C)}".replace("k  ", "k ") + ".",
            f"So {num(A)}k = {num(C - B * y0)} and k = {num(ans)}.",
        ]
        item = MathItem("point_on_line", TWO_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
        item.truth = {"A": A, "B": B, "C": C, "y0": y0}
        return item
    while True:
        if level == "easy":
            m, b = Fraction(rng.choice(nz(-4, 4))), rng.choice(nz(-6, 6))
            x0 = rng.choice(nz(-4, 4))
        else:
            m = Fraction(rng.choice(nz(-4, 4)), rng.choice([2, 3]))
            b = rng.choice(nz(-6, 6))
            x0 = m.denominator * rng.choice(nz(-3, 3))
        y0 = m * x0 + b
        if abs(x0) != abs(y0) and y0 and abs(m) != 1 and m.denominator <= 3:
            break
    if level == "easy":
        eq = f"y = {linf(m, b)}"

        def on(x, y):
            return y == m * x + b
    else:
        A, B, C = -m.numerator, m.denominator, b * m.denominator
        if A < 0:
            A, B, C = -A, -B, -C
        eq = axby(A, B, C)

        def on(x, y):
            return A * x + B * y == C

    stem = f"Which point lies on the graph of {eq} in the xy-plane?"
    key = _pt(x0, y0)
    cands = [
        ((y0, x0), "this swaps the x- and y-coordinates"),
        ((x0, -y0), "this has the wrong sign on y"),
        ((-x0, y0), "this has the wrong sign on x"),
        ((x0, -m * x0 + b), "this point is on the line with the slope's sign flipped"),
        ((x0, m * x0 - b), "this point uses the y-intercept with the wrong sign"),
    ]
    mistakes, seen = [], {key}
    for (px, py), w in cands:
        t = _pt(px, py)
        if t not in seen and not on(px, py) and Fraction(py).denominator == 1:
            seen.add(t)
            mistakes.append((t, w))
    steps = [f"Substitute each point into {eq}.", f"Only {key} makes the equation true."]
    item = MathItem("point_on_line", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b}
    return item


def line_two_points(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'A line passes through the points (2, 5) and (6, 13) in the xy-plane. Which equation defines the line?'
    Medium: negative coordinates and a fraction slope. Hard: 'What is the y-intercept of the line?' (a number). Traps:
    run over rise, subtracting in mixed order (the sign), the first point's y-value as the y-intercept."""
    while True:
        m = Fraction(rng.choice(nz(-4, 4)), 1 if level == "easy" else rng.choice([1, 2, 3]))
        b = rng.choice(nz(-8, 8))
        x1 = m.denominator * rng.choice(range(-3, 4) if level != "easy" else range(0, 3))
        x2 = x1 + m.denominator * rng.choice([1, 2, 3])
        y1, y2 = m * x1 + b, m * x2 + b
        if abs(m) != 1 and b != y1 and b != -b + 2 * y1 and 0 not in (x1, x2) and y1 != 0:
            break
    p1, p2 = _pt(x1, y1), _pt(x2, y2)
    steps = [
        f"Slope: ({num(y2)} {MINUS} {par(y1)}) ÷ ({num(x2)} {MINUS} {par(x1)}) = {num(m)}.",
        f"Use {p1}: {num(y1)} = {par(m)}({num(x1)}) + b, so b = {num(b)}.",
    ]
    if level == "hard":
        stem = f"A line in the xy-plane passes through the points {p1} and {p2}. What is the y-intercept of the line?"
        stem = stem.replace("What is the y-intercept", "What is the y-coordinate of the y-intercept")
        ans = Fraction(b)
        mistakes = [
            (Fraction(y1), "this is the first point's y-value"),
            (y1 + m * x1, "this adds the slope times x instead of subtracting it"),
            (y1 - x1 / m, "this uses the slope flipped (run over rise)"),
            (m, "this is the slope, not the y-intercept"),
        ]
        item = MathItem(
            "line_two_points", TWO_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False, fractions_ok=True
        )
        item.truth = {"m": m, "b": b, "pts": ((x1, y1), (x2, y2))}
        return item
    stem = f"A line in the xy-plane passes through the points {p1} and {p2}. Which equation defines the line?"
    key = f"y = {linf(m, b)}"
    mistakes = [
        (f"y = {linf(1 / m, y1 - x1 / m)}", "this uses run over rise for the slope"),
        (f"y = {linf(-m, y1 + m * x1)}", "this subtracts the coordinates in mixed order, so the slope's sign is wrong"),
        (f"y = {linf(m, y1)}", f"this uses {num(y1)}, the first point's y-value, as the y-intercept"),
        (f"y = {linf(m, y1 + m * x1)}", "this adds the slope times x instead of subtracting it"),
    ]
    item = MathItem("line_two_points", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b, "pts": ((x1, y1), (x2, y2))}
    return item


def parallel_lines(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'Which line is parallel to y = 3x − 4?' Easy: slope-intercept choices. Medium: the given line in standard form
    (2x + 5y = 10), choices in y = mx + b. Hard: everything in standard form ('Which line is parallel to
    4x − 6y = 9?': 6x − 9y = 2). Traps: the perpendicular slope, the slope's sign flipped, the coefficients swapped,
    the same line (not parallel: it is the line itself)."""
    while True:
        m = Fraction(rng.choice(nz(-5, 5)), 1 if level == "easy" else rng.choice([2, 3, 4, 5]))
        b = rng.choice(nz(-8, 8))
        if abs(m) != 1 and m.denominator <= 5 and abs(m.numerator) != m.denominator:
            break
    b2 = rng.choice([v for v in nz(-9, 9) if v != b])
    if level != "hard":
        given = f"y = {linf(m, b)}" if level == "easy" else std_form(m.numerator, m.denominator, b)
        stem = f"In the xy-plane, which line is parallel to the line {given}?"
        key = f"y = {linf(m, b2)}"
        mistakes = [
            (f"y = {linf(-1 / m, b2)}", "this slope is the negative reciprocal: that line is perpendicular"),
            (f"y = {linf(-m, b2)}", "this slope has the wrong sign"),
            (f"y = {linf(1 / m, b2)}", "this slope is flipped"),
            (f"y = {linf(-m, b)}", "this slope has the wrong sign (the same y-intercept doesn't make it parallel)"),
        ]
        steps = [
            f"The given line has slope {num(m)}" + (f" (solve {given} for y)." if level == "medium" else "."),
            f"Parallel lines have the same slope and different y-intercepts, so {key}.",
        ]
    else:
        A, Bc = -m.numerator, m.denominator  # −px + qy has slope p/q
        if A < 0:
            A, Bc = -A, -Bc
        given_c = rng.choice(nz(-12, 12))
        k = rng.choice([Fraction(3, 2), Fraction(2), Fraction(3), Fraction(1, 2)])
        A2, B2 = A * k, Bc * k
        while A2.denominator > 1 or B2.denominator > 1:
            k = Fraction(2)
            A2, B2 = A * k, Bc * k
        c2 = rng.choice([v for v in nz(-15, 15) if Fraction(v) != given_c * k])
        given = axby(A, Bc, given_c)
        stem = f"In the xy-plane, which line is parallel to the line {given}?"
        key = axby(A2, B2, c2)
        mistakes = [
            (pos_axby(A2, -B2, c2), "this flips the sign of the y-coefficient, which changes the slope's sign"),
            (pos_axby(B2, A2, c2), "this swaps the x- and y-coefficients"),
            (pos_axby(A2 * 2, B2, c2), "this changes only the x-coefficient, so the slope changes"),
            (pos_axby(abs(B2), -A2 if B2 > 0 else A2, c2), "this line is perpendicular"),
        ]
        steps = [
            f"The given line has slope {num(m)} (solve for y: y = {linf(m, Fraction(given_c, Bc))}).",
            f"{key} has the same ratio of coefficients, so the same slope, and a different constant, so it is "
            "parallel (not the same line).",
        ]
    seen, out = {key}, []
    for t, w in mistakes:
        if t not in seen:
            seen.add(t)
            out.append((t, w))
    item = MathItem("parallel_lines", TWO_SKILL, level, "mcq", stem, Fraction(0), steps, out, key_text=key)
    item.truth = {"m": m, "given": given}
    return item


def ineq_system_graph(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A system of two inequalities graphed in the xy-plane: two shaded half-planes (the solutions are where the
    shading overlaps), one boundary solid and one dashed. 'Which point (x, y) is a solution to the system of
    inequalities shown?' Hard: 'Which system of inequalities is shown?' (the choices are systems). Traps: a point in
    only one shaded region, a point on the dashed boundary, the signs backward."""
    while True:
        m1, m2 = Fraction(rng.choice(nz(-3, 3))), Fraction(rng.choice(nz(-3, 3)), rng.choice([1, 2]))
        b1, b2 = rng.choice(nz(-4, 4)), rng.choice(nz(-4, 4))
        s1, s2 = rng.choice(["above", "below"]), rng.choice(["above", "below"])
        if m1 == m2 or abs(m1) == abs(m2):
            continue
        r1 = "≥" if s1 == "above" else "≤"  # solid
        r2 = ">" if s2 == "above" else "<"  # dashed

        def ok1(x, y, r=r1, m=m1, b=b1):
            return _holds(r, y, m * x + b)

        def ok2(x, y, r=r2, m=m2, b=b2):
            return _holds(r, y, m * x + b)

        pts = [(x, y) for x in range(-6, 7) for y in range(-6, 7)]
        both = [p_ for p_ in pts if ok1(*p_) and ok2(*p_)]
        only = [p_ for p_ in pts if ok1(*p_) != ok2(*p_)]
        on_dash = [(x, y) for x, y in pts if y == m2 * x + b2 and ok1(x, y)]
        if len(both) >= 6 and len(only) >= 4 and on_dash:
            break
    fig = graph_svg(
        "x",
        "y",
        1,
        8,
        1,
        8,
        lines=[(m1, b1), (m2, b2)],
        label_every=(2, 2),
        x_lo=8,
        y_lo=8,
        plane=True,
        shade=[(m1, b1, s1), (m2, b2, s2)],
        dashed=(1,),
    )
    e1, e2 = f"y {r1} {linf(m1, b1)}", f"y {r2} {linf(m2, b2)}"
    if level == "hard":
        stem = "Which system of inequalities is represented by the graph shown?"
        key = f"{e1}\n{e2}"
        mistakes = [
            (f"y {FLIP[r1]} {linf(m1, b1)}\n{e2}", "this has the first sign backward"),
            (f"{e1}\ny {FLIP[r2]} {linf(m2, b2)}", "this has the second sign backward"),
            (
                f"y {LOOSE[r1]} {linf(m1, b1)}\ny {LOOSE[r2]} {linf(m2, b2)}",
                "this mixes up the solid line (≤ or ≥) and the dashed line (< or >)",
            ),
            (f"y {FLIP[r1]} {linf(m1, b1)}\ny {FLIP[r2]} {linf(m2, b2)}", "this has both signs backward"),
        ]
        steps = [
            f"Solid line y = {linf(m1, b1)}, shaded {s1}: {e1}.",
            f"Dashed line y = {linf(m2, b2)}, shaded {s2}: {e2}.",
        ]
    else:
        stem = "Which point (x, y) is a solution to the system of inequalities shown?"
        key_pt = rng.choice(both)
        key = _pt(*key_pt)
        picks = [(rng.choice(on_dash), "this point is on the dashed line, which is not part of the solution")]
        in1 = [p_ for p_ in only if ok1(*p_)]
        in2 = [p_ for p_ in only if ok2(*p_)]
        if in1:
            picks.append((rng.choice(in1), "this point is in only the region shaded for the solid line"))
        if in2:
            picks.append((rng.choice(in2), "this point is in only the region shaded for the dashed line"))
        while len(picks) < 3:
            picks.append((rng.choice(only), "this point is in only one of the shaded regions"))
        mistakes, seen = [], {key}
        for pt, w in picks:
            while _pt(*pt) in seen:
                pt = rng.choice(only)
            seen.add(_pt(*pt))
            mistakes.append((_pt(*pt), w))
        steps = [
            f"The solutions are where both shaded regions overlap: {e1} and {e2}.",
            f"Points on the dashed line are not included. Only {key} is in the overlap.",
        ]
    item = MathItem(
        "ineq_system_graph", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key, figure=fig
    )
    item.truth = {"lines": [(m1, b1, r1), (m2, b2, r2)]}
    return item


# -------------------------------------------------------- closing out Algebra (Sebastian, Nov 25): six more types


def ineq_solve(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Solve a linear inequality. Easy: '4x − 8 > 12' (x > 5). Medium: a negative coefficient, '−3x + 6 > 21', where
    dividing by −3 flips the sign (x < −5). Hard: a compound inequality with a negative coefficient,
    '−11 ≤ −3x + 1 < 7' (−2 < x ≤ 4), where the flip also moves which end is strict. Choices are a 2x2: the right
    or flipped direction (or strict end) x the right or wrong value."""
    if level != "hard":
        a = rng.randint(2, 6) * (1 if level == "easy" else -1)
        p_, q = rng.choice(nz(-6, 6)), rng.choice(nz(-6, 6))
        while p_ == 0 or q - p_ == 0 or q + p_ == 0 or q - p_ == -(q + p_):
            p_, q = rng.choice(nz(-6, 6)), rng.choice(nz(-6, 6))
        b, c = a * p_, a * q
        rel = rng.choice(["<", ">", "≤", "≥"])
        ans, trap = q - p_, q + p_  # x rel' (c − b)/a ; moving b with the wrong sign gives (c + b)/a
        rel_x = FLIP[rel] if a < 0 else rel
        stem = f"Which inequality represents all solutions to {linf(a, b)} {rel} {num(c)}?"
        why_dir = (
            "dividing by a negative number reverses the inequality sign"
            if a < 0
            else "this has the inequality sign backward"
        )
        mistakes = grid2(
            lambda r_, v: f"x {r_} {num(v)}",
            rel_x,
            FLIP[rel_x],
            why_dir,
            ans,
            trap,
            f"this {'adds' if b > 0 else 'subtracts'} {num(abs(b))} instead of "
            f"{'subtracting' if b > 0 else 'adding'} it",
        )
        key = f"x {rel_x} {num(ans)}"
        steps = [
            f"{'Subtract' if b > 0 else 'Add'} {num(abs(b))}: {coef(a)} {rel} {num(c - b)}.",
            f"Divide by {num(a)}" + (" and reverse the sign, because it is negative" if a < 0 else "") + f": {key}.",
        ]
        truth = {"a": a, "b": b, "c": c, "rel": rel}
    else:
        a = -rng.randint(2, 5)
        b = rng.choice(nz(-9, 9))
        while True:
            lo, hi = sorted(rng.sample(range(-8, 9), 2))
            if lo + hi != 0 and hi - lo >= 2:
                break
        r1, r2 = rng.sample(["<", "≤"], 2)  # different, so the flip visibly moves the strict end
        L, R = a * hi + b, a * lo + b  # a < 0, so x = hi gives the left (smaller) end
        stem = f"Which inequality represents all solutions to {num(L)} {r1} {linf(a, b)} {r2} {num(R)}?"
        key = f"{num(lo)} {r2} x {r1} {num(hi)}"
        mistakes = grid2(
            lambda ends, vals: f"{num(vals[0])} {ends[0]} x {ends[1]} {num(vals[1])}",
            (r2, r1),
            (r1, r2),
            "this keeps each sign on its original side; dividing by a negative number reverses "
            "the inequality, so the ends trade places",
            (lo, hi),
            (-hi, -lo),
            f"this divides by {num(-a)} instead of {num(a)}",
        )
        steps = [
            f"Subtract {num(b)} from all three parts: {num(L - b)} {r1} {coef(a)} {r2} {num(R - b)}."
            if b > 0
            else f"Add {num(-b)} to all three parts: {num(L - b)} {r1} {coef(a)} {r2} {num(R - b)}.",
            f"Divide by {num(a)} and reverse both signs: {num(hi)} {FLIP[r1]} x {FLIP[r2]} {num(lo)}, that is, {key}.",
        ]
        truth = {"a": a, "b": b, "L": L, "R": R, "r1": r1, "r2": r2}
    item = MathItem("ineq_solve", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = truth
    return item


SYS_INEQ_FRAMES = [
    dict(
        text="{n} is buying shirts that cost ${a} each and hats that cost ${b} each. {n} wants to buy {cw} {c} "
        "items and spend {mw} ${M}. Which system of inequalities represents the numbers of shirts, x, and hats, "
        "y, that {n} could buy?",
        cw="at least",
        mw="at most",
        a=(12, 25),
        b=(6, 11),
        c=(5, 12),
        unit="dollars",
    ),
    dict(
        text="A school club sells adult tickets for ${a} each and student tickets for ${b} each for a show in a room "
        "that holds {cw} {c} people. The club wants to earn {mw} ${M}. Which system of inequalities represents "
        "the numbers of adult tickets, x, and student tickets, y, the club could sell?",
        cw="at most",
        mw="at least",
        a=(10, 18),
        b=(4, 8),
        c=(80, 150),
        unit="dollars",
    ),
    dict(
        text="A farmer plans to plant corn and beans on {cw} {c} acres. Each acre of corn needs {a} hours of work, and "
        "each acre of beans needs {b} hours of work. The farmer has {mw} {M} hours of work available. Which "
        "system of inequalities represents the numbers of acres of corn, x, and beans, y, the farmer could "
        "plant?",
        cw="at most",
        mw="at most",
        a=(6, 12),
        b=(2, 5),
        c=(20, 60),
        unit="hours",
    ),
]
SYS_INEQ_HARD = [
    dict(
        text="A volunteer is packing food boxes and hygiene kits. Each food box takes {a} minutes to pack, and each "
        "hygiene kit takes {b} minutes. The volunteer wants to pack more than {c} items in all and has at most "
        "{H} hours to work. Which system of inequalities represents the numbers of food boxes, x, and hygiene "
        "kits, y, the volunteer could pack?",
        crel=">",
        mrel="≤",
        a=(8, 15),
        b=(3, 6),
        c=(10, 30),
        H=(4, 8),
        k=60,
        small="minutes",
        big="hours",
    ),
    dict(
        text="A bakery makes loaves of bread and trays of rolls. Each loaf uses {a} ounces of flour, and each tray "
        "uses {b} ounces. The bakery wants to make fewer than {c} items and has at most {H} pounds of flour "
        "(1 pound = 16 ounces). Which system of inequalities represents the numbers of loaves, x, and trays, y, "
        "the bakery could make?",
        crel="<",
        mrel="≤",
        a=(14, 20),
        b=(6, 10),
        c=(30, 60),
        H=(20, 40),
        k=16,
        small="ounces",
        big="pounds",
    ),
]
WORD_REL = {"at least": "≥", "at most": "≤"}


def ineq_system_setup(rng: random.Random, level: str, fmt: str) -> MathItem:
    """A word problem with two conditions: 'at least 10 items' and 'at most $150'. 'Which system of inequalities
    represents the situation?' (x + y ≥ 10 and 12x + 8y ≤ 150). Hard: 'more than' / 'fewer than' and a limit in a
    bigger unit (hours, pounds) that must be converted. Choices are a 2x2: the count condition's sign right or
    flipped x the amount condition right or wrong (prices swapped, or the limit not converted)."""
    n = rng.choice(NAMES)
    if level != "hard":
        f = rng.choice(SYS_INEQ_FRAMES)
        a, b = rng.randint(*f["a"]), rng.randint(*f["b"])
        c = rng.randint(*f["c"])
        M = (a + b) * c // 2 // 10 * 10 or 100
        crel, mrel = WORD_REL[f["cw"]], WORD_REL[f["mw"]]
        stem = f["text"].format(n=n, a=a, b=b, c=c, M=num(M), cw=f["cw"], mw=f["mw"])
        right2, wrong2 = (a, b, M), (b, a, M)
        why2 = "this swaps the two amounts"
    else:
        f = rng.choice(SYS_INEQ_HARD)
        a, b = rng.randint(*f["a"]), rng.randint(*f["b"])
        c, H = rng.randint(*f["c"]), rng.randint(*f["H"])
        M = H * f["k"]
        crel, mrel = f["crel"], f["mrel"]
        stem = f["text"].format(a=a, b=b, c=c, H=H)
        right2, wrong2 = (a, b, M), (a, b, H)
        why2 = f"this leaves the limit in {f['big']}; the amounts for each item are in {f['small']}"

    def sys_(cr, amt) -> str:
        p_, q_, lim = amt
        return f"x + y {cr} {num(c)}\n{coef(p_)} + {coef(q_, 'y')} {mrel} {num(lim)}"

    key = sys_(crel, right2)
    mistakes = grid2(
        sys_, crel, FLIP[crel], f"the count condition has the sign backward; it should be {crel}", right2, wrong2, why2
    )
    steps = [
        f"The number of items: x + y {crel} {num(c)}.",
        f"The total amount: {coef(a)} + {coef(b, 'y')} {mrel} {num(M)}"
        + (f" ({H} {f['big']} = {num(M)} {f['small']})." if level == "hard" else "."),
    ]
    item = MathItem("ineq_system_setup", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"c": c, "crel": crel, "a": a, "b": b, "M": M, "mrel": mrel}
    return item


def small_graph(m, b, side: str | None = None, dashed: bool = False) -> str:
    """A small xy-plane graph (−6 to 6) for answer choices: a line, or a shaded half-plane."""
    m, b = Fraction(m), Fraction(b)
    svg = graph_svg(
        "x",
        "y",
        1,
        6,
        1,
        6,
        lines=[(m, b)],
        label_every=(2, 2),
        x_lo=6,
        y_lo=6,
        plane=True,
        shade=[(m, b, side)] if side else (),
        dashed=(0,) if dashed else (),
    )
    return re.sub(r'width="\d+" height="\d+"', 'width="200" height="183"', svg, count=1)


def graph_choices(rng: random.Random, level: str, fmt: str) -> MathItem:
    """The answer choices are graphs. Easy: 'Which graph shows y = −2x + 3?' Medium: the line is given in standard
    form (2x + 3y = 6). Hard: 'Which graph shows the solutions to y ≤ (1/2)x − 2?' (a shaded region; solid or dashed
    line). Choices are a 2x2: slope sign x intercept sign (lines), or shaded side x solid/dashed (regions)."""
    while True:
        m = Fraction(rng.choice(nz(-3, 3)), 1 if level == "easy" else rng.choice([1, 2]))
        b = rng.choice(nz(-4, 4))
        if abs(m) != 1 and abs(b) >= 1 and m.denominator <= 2:
            break
    if level != "hard":
        given = f"y = {linf(m, b)}" if level == "easy" else std_form(m.numerator, m.denominator, b)
        stem = f"Which graph shows the line {given} in the xy-plane?"
        cell = {}

        def draw(mm_, bb):
            g = small_graph(mm_, bb)
            cell[g] = (mm_, bb)
            return g

        key = draw(m, b)
        mistakes = grid2(
            draw,
            m,
            -m,
            "this line has the slope with the wrong sign",
            b,
            -b,
            "this line crosses the y-axis at the wrong point (the y-intercept's sign is wrong)",
        )
        steps = [
            (f"Solve for y: y = {linf(m, b)}. " if level == "medium" else "")
            + f"The line crosses the y-axis at (0, {num(b)}) and has slope {num(m)}.",
            f"So it {'rises' if m > 0 else 'falls'} from left to right through (0, {num(b)}).",
        ]
        truth = {"m": m, "b": b}
    else:
        above, strict = rng.random() < 0.5, rng.random() < 0.5
        rel = (">" if strict else "≥") if above else ("<" if strict else "≤")
        stem = f"Which graph shows the solutions to y {rel} {linf(m, b)} in the xy-plane?"
        cell = {}

        def draw(side, dash):
            g = small_graph(m, b, side, dash)
            cell[g] = (side, dash)
            return g

        sd, other = ("above", "below") if above else ("below", "above")
        key = draw(sd, strict)
        mistakes = grid2(
            draw,
            sd,
            other,
            f"the shading should be {sd} the line for y {rel}",
            strict,
            not strict,
            "a dashed line is for < or >, and a solid line is for ≤ or ≥",
        )
        steps = [
            f"The boundary is y = {linf(m, b)}.",
            f"{'Strict' if strict else 'Not strict'}, so the line is {'dashed' if strict else 'solid'}; "
            f"y {rel} means shading {sd} the line.",
        ]
        truth = {"m": m, "b": b, "side": sd, "dashed": strict}
    item = MathItem(
        "graph_choices",
        FUNC_SKILL if level != "hard" else INEQ_SKILL,
        level,
        "mcq",
        stem,
        Fraction(0),
        steps,
        mistakes,
        key_text=key,
    )
    item.truth = {**truth, "cells": cell}
    return item


def function_notation(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Easy: 'f(x) = 3x − 5. Which expression is equal to f(x + 2)?' (3x + 1). Medium: 'g(x) = 2f(x) − 1. Which
    equation defines g?' (6x − 11). Hard: 'f(x) = 2x + 1 and g(x) = x − 4. What is the value of f(g(5))?' Traps: adding
    outside instead of inside, forgetting to multiply the constant, the functions in the wrong order, the product or
    sum of the two values."""
    a = rng.choice([v for v in nz(-6, 7) if abs(v) > 1])
    b = rng.choice(nz(-9, 9))
    letter = rng.choice(["f", "h", "p"])
    if level == "easy":
        while True:
            c = rng.choice(nz(-5, 5))
            key = linf(a, a * c + b)
            pool = [
                (
                    linf(a, b + c),
                    f"this {'adds' if c > 0 else 'subtracts'} {num(abs(c))} "
                    f"{'to' if c > 0 else 'from'} the output instead of putting x {plus(c).strip()} in "
                    "for x",
                ),
                (linf(a, b - a * c), f"this uses x {plus(-c).strip()}: the sign of {num(abs(c))} is wrong"),
                (linf(a, a * c), f"this leaves out the {num(b)}"),
                (linf(a, a * c - b), f"this changes the sign of the {num(b)}"),
            ]
            if len({t for t, _ in pool} - {key}) >= 3:
                break
        arg = f"x + {c}" if c > 0 else f"x {MINUS} {-c}"
        stem = (
            f"The function {letter} is defined by {letter}(x) = {linf(a, b)}. Which expression is equal to "
            f"{letter}({arg})?"
        )
        steps = [
            f"{letter}({arg}) = {num(a)}({arg}) {plus(b).strip()} = {coef(a)} {plus(a * c).strip()} "
            f"{plus(b).strip()} = {key}."
        ]
        mistakes = [(t, w) for t, w in pool if t != key]
        item = MathItem("function_notation", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
        item.truth = {"kind": "shift", "a": a, "b": b, "c": c}
        return item
    if level == "medium":
        k = rng.choice([v for v in nz(-4, 4) if abs(v) > 1])
        d = rng.choice(nz(-9, 9))
        g = "g" if letter != "g" else "q"
        stem = (
            f"The function {letter} is defined by {letter}(x) = {linf(a, b)}, and the function {g} is defined by "
            f"{g}(x) = {num(k)}{letter}(x) {plus(d).strip()}. Which equation defines {g}?"
        )
        key = f"{g}(x) = {linf(k * a, k * b + d)}"
        pool = [
            (
                f"{g}(x) = {linf(k * a, b + d)}",
                f"this multiplies only the x-term by {num(k)}, not the constant {num(b)}",
            ),
            (f"{g}(x) = {linf(k * a, k * b)}", f"this leaves out the {num(d)}"),
            (f"{g}(x) = {linf(k * a, k * (b + d))}", f"this multiplies the {num(d)} by {num(k)} too"),
            (f"{g}(x) = {linf(a, k * b + d)}", f"this multiplies only the constant by {num(k)}"),
        ]
        steps = [
            f"{g}(x) = {num(k)}({linf(a, b)}) {plus(d).strip()} = {linf(k * a, k * b)} {plus(d).strip()} = "
            f"{key.split(' = ')[1]}."
        ]
        mistakes = [(t, w) for t, w in pool if t != key]
        item = MathItem("function_notation", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
        item.truth = {"kind": "scale", "a": a, "b": b, "k": k, "d": d}
        return item
    a2, b2 = rng.choice([v for v in nz(-4, 4) if abs(v) > 1]), rng.choice(nz(-6, 6))
    n0 = rng.choice(nz(-5, 6))
    f_, g_ = (lambda x: a * x + b), (lambda x: a2 * x + b2)
    ans = Fraction(f_(g_(n0)))
    stem = (
        f"The functions f and g are defined by f(x) = {linf(a, b)} and g(x) = {linf(a2, b2)}. What is the value of "
        f"f(g({num(n0)}))?"
    )
    mistakes = [
        (Fraction(g_(f_(n0))), "this is g(f(" + num(n0) + ")): the functions are in the wrong order"),
        (Fraction(f_(n0) * g_(n0)), "this multiplies f(" + num(n0) + ") and g(" + num(n0) + ")"),
        (Fraction(f_(n0) + g_(n0)), "this adds f(" + num(n0) + ") and g(" + num(n0) + ")"),
        (Fraction(g_(n0)), "this is g(" + num(n0) + "), only the first step"),
    ]
    steps = [
        f"g({num(n0)}) = {num(a2)}({num(n0)}) {plus(b2).strip()} = {num(g_(n0))}.",
        f"f({num(g_(n0))}) = {num(a)}({num(g_(n0))}) {plus(b).strip()} = {num(ans)}.",
    ]
    item = MathItem("function_notation", FUNC_SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
    item.truth = {"kind": "compose", "f": (a, b), "g": (a2, b2), "n": n0}
    return item


def abs_value(rng: random.Random, level: str, fmt: str) -> MathItem:
    """Absolute value (Sebastian's deck). Easy: '|x − 4| = 7. What is the sum of the solutions?' (11 + (−3) = 8).
    Medium: '|2x + 3| = 9. What is the greater solution?' Hard: '|x − 5| < 3. Which inequality represents all
    solutions?' (2 < x < 8), a 2x2 of between/outside x strict/not strict. Traps: only one solution, adding the
    distances, the other solution, not dividing."""
    if level == "easy":
        h, r = rng.choice(nz(-9, 9)), rng.randint(2, 12)
        while abs(h) == r or 2 * h in (2 * r, h + r, h - r) or r == abs(2 * h):
            h, r = rng.choice(nz(-9, 9)), rng.randint(2, 12)
        inner = f"x {MINUS} {num(h)}" if h > 0 else f"x + {num(-h)}"
        stem = f"What is the sum of the solutions to |{inner}| = {r}?"
        ans = Fraction(2 * h)
        mistakes = [
            (Fraction(h + r), f"this is only one solution, {num(h + r)}; the other is {num(h - r)}"),
            (Fraction(2 * r), f"this adds the two distances, {r} + {r}, instead of the two solutions"),
            (Fraction(h - r), "this is only the other solution"),
            (-ans, "this has the wrong sign"),
        ]
        steps = [
            f"{inner} = {r} or {inner} = {MINUS}{r}, so x = {num(h + r)} or x = {num(h - r)}.",
            f"The sum is {num(h + r)} + {par(h - r)} = {num(ans)}.",
        ]
        item = MathItem("abs_value", SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
        item.truth = {"h": h, "r": r}
        return item
    if level == "medium":
        while True:
            a = rng.randint(2, 5)
            p_, q = rng.sample(range(-9, 10), 2)
            if (a * (p_ + q)) % 2 == 0:
                c = -a * (p_ + q) // 2
                r = a * p_ + c
                if r > 0 and c != 0 and p_ != q:
                    break
        hi, lo = max(p_, q), min(p_, q)
        inner = linf(a, c)
        stem = f"What is the greater solution to |{inner}| = {r}?"
        ans = Fraction(hi)
        mistakes = [
            (Fraction(lo), "this is the smaller solution"),
            (Fraction(r - c), f"this stops before dividing by {a}"),
            (Fraction(r, a), f"this leaves out the {num(c)}"),
            (Fraction(r + c, a), f"this moves the {num(c)} with the wrong sign"),
        ]
        steps = [
            f"{inner} = {r} or {inner} = {MINUS}{r}.",
            f"So x = {num(Fraction(r - c, a))} or x = {num(Fraction(-r - c, a))}; the greater is {num(ans)}.",
        ]
        item = MathItem("abs_value", SKILL, level, fmt, stem, ans, steps, mistakes, positive_only=False)
        item.truth = {"a": a, "c": c, "r": r}
        return item
    h, r = rng.choice(nz(-6, 9)), rng.randint(2, 7)
    rel = rng.choice(["<", "≤", ">", "≥"])
    inner = f"x {MINUS} {num(h)}" if h > 0 else f"x + {num(-h)}"
    stem = f"Which inequality represents all solutions to |{inner}| {rel} {r}?"
    between = rel in ("<", "≤")
    strict = rel in ("<", ">")

    def show(btw: bool, st: bool) -> str:
        if btw:
            s_ = "<" if st else "≤"
            return f"{num(h - r)} {s_} x {s_} {num(h + r)}"
        lo_s, hi_s = ("<", ">") if st else ("≤", "≥")
        return f"x {lo_s} {num(h - r)} or x {hi_s} {num(h + r)}"

    key = show(between, strict)
    mistakes = grid2(
        show,
        between,
        not between,
        (
            "'less than' a distance means between the two values"
            if between
            else "'greater than' a distance means outside the two values"
        ),
        strict,
        not strict,
        "this has the wrong kind of inequality at the ends",
    )
    steps = [
        f"|{inner}| {rel} {r} means x is {'within' if between else 'more than'} {r} of {num(h)}"
        + ("" if strict else " (or exactly that far)")
        + ".",
        f"So {key}.",
    ]
    item = MathItem("abs_value", INEQ_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"h": h, "r": r, "rel": rel}
    return item


POINT_FRAMES = [
    dict(
        text="The function f(x) = {m}x + {b} gives the amount of money, in dollars, in {n}'s savings account x weeks "
        "after {n} started saving.",
        ms=(15, 20, 25, 30, 40),
        bs=(50, 80, 100, 120, 150),
        at="{n} had ${y} in the account {x} weeks after starting to save.",
        rate="The amount in the account increased by ${y} every {x} weeks.",
    ),
    dict(
        text="The function f(x) = {m}x + {b} gives the height, in centimeters, of a sunflower x weeks after it was "
        "planted.",
        ms=(4, 5, 6, 8),
        bs=(10, 12, 15, 20),
        at="The sunflower was {y} centimeters tall {x} weeks after it was planted.",
        rate="The sunflower grew {y} centimeters every {x} weeks.",
    ),
    dict(
        text="The function f(x) = {m}x + {b} gives the total cost, in dollars, of a gym membership for x months, "
        "including a one-time sign-up fee.",
        ms=(25, 30, 35, 45),
        bs=(40, 50, 60, 75),
        at="The total cost of the membership for {x} months is ${y}.",
        rate="The cost of the membership increases by ${y} every {x} months.",
    ),
]


def point_meaning(rng: random.Random, level: str, fmt: str) -> MathItem:
    """'The function f(x) = 25x + 150 gives the money in Kai's account x weeks after he started saving. The point
    (4, 250) lies on the graph of y = f(x). Which is the best interpretation of the point?' Hard says 'f(6) = 330'
    instead. Choices are a 2x2: x and y in the right roles or swapped x a value at one time or a change over time."""
    f = rng.choice(POINT_FRAMES)
    n = rng.choice(NAMES)
    m, b = rng.choice(f["ms"]), rng.choice(f["bs"])
    x0 = rng.randint(3, 12)
    y0 = m * x0 + b
    intro = f["text"].format(m=m, b=b, n=n)
    if level == "hard":
        stem = f"{intro} Which of the following is the best interpretation of the statement f({x0}) = {num(y0)}?"
    else:
        stem = (
            f"{intro} The point ({x0}, {num(y0)}) lies on the graph of y = f(x) in the xy-plane. Which of the "
            f"following is the best interpretation of this point?"
        )

    def say(roles: bool, kind: str) -> str:
        xs, ys = (x0, num(y0)) if roles else (num(y0), x0)
        return f[kind].format(n=n, x=xs, y=ys)

    key = say(True, "at")
    mistakes = grid2(
        say,
        True,
        False,
        "this swaps the input (time) and the output",
        "at",
        "rate",
        f"the point gives one value of f at x = {x0}, not a change over time",
    )
    steps = [
        f"The point ({x0}, {num(y0)}) means f({x0}) = {num(y0)}: when x = {x0}, the value is {num(y0)}.",
        f"So: {key}",
    ]
    item = MathItem("point_meaning", FUNC_SKILL, level, "mcq", stem, Fraction(0), steps, mistakes, key_text=key)
    item.truth = {"m": m, "b": b, "x0": x0, "y0": y0}
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
    "graph_estimate": LEVELS,
    "graph_slope": ("easy", "medium", "hard"),
    "rate_total_function": ("medium", "hard"),
    "intercept_meaning": ("medium", "hard"),
    "two_points_function": ("medium", "hard"),
    "intercept_from_slope": ("medium", "hard"),
    "cost_function": LEVELS,
    "function_change": ("medium", "hard"),
    "table_equation": ("medium", "hard"),
    "table_mb": ("medium", "hard"),
    "table_shift": ("medium", "hard"),
    "sum_functions": ("medium", "hard"),
    "graph_which_equation": LEVELS,
    "evaluate_function": LEVELS,
    "unit_conversion_function": ("medium", "hard"),
    "two_lines_graph": ("medium", "hard"),
    "two_var_equation": LEVELS,
    "table_for_equation": LEVELS,
    "tradeoff": LEVELS,
    "slope_point_line": LEVELS,
    "isosceles_perimeter": LEVELS,
    "perpendicular_vertical": ("easy", "medium"),
    "interpret_coefficients": LEVELS,
    "perpendicular_lines": LEVELS,
    "table_unknowns": LEVELS,
    "system_which_property": LEVELS,
    "system_coordinate": LEVELS,
    "system_cancel_letters": ("medium", "hard"),
    "system_constant_r": ("medium", "hard"),
    "system_solve_pair": LEVELS,
    "system_word_setup": ("medium", "hard"),
    "system_graph_equations": LEVELS,
    "system_times_more": LEVELS,
    "ineq_max_word": LEVELS,
    "ineq_setup": LEVELS,
    "ineq_graph": LEVELS,
    "ineq_system_points": ("medium", "hard"),
    "ineq_fence_range": ("medium", "hard"),
    "intercepts_std": LEVELS,
    "point_on_line": LEVELS,
    "line_two_points": LEVELS,
    "parallel_lines": LEVELS,
    "ineq_system_graph": LEVELS,
    "ineq_solve": LEVELS,
    "ineq_system_setup": ("medium", "hard"),
    "graph_choices": LEVELS,
    "function_notation": LEVELS,
    "abs_value": LEVELS,
    "point_meaning": ("medium", "hard"),
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
    "graph_estimate": graph_estimate,
    "graph_slope": graph_slope,
    "rate_total_function": rate_total_function,
    "intercept_meaning": intercept_meaning,
    "two_points_function": two_points_function,
    "intercept_from_slope": intercept_from_slope,
    "cost_function": cost_function,
    "function_change": function_change,
    "table_equation": table_equation,
    "table_mb": table_mb,
    "table_shift": table_shift,
    "sum_functions": sum_functions,
    "graph_which_equation": graph_which_equation,
    "evaluate_function": evaluate_function,
    "unit_conversion_function": unit_conversion_function,
    "two_lines_graph": two_lines_graph,
    "two_var_equation": two_var_equation,
    "table_for_equation": table_for_equation,
    "tradeoff": tradeoff,
    "slope_point_line": slope_point_line,
    "isosceles_perimeter": isosceles_perimeter,
    "perpendicular_vertical": perpendicular_vertical,
    "interpret_coefficients": interpret_coefficients,
    "perpendicular_lines": perpendicular_lines,
    "table_unknowns": table_unknowns,
    "system_which_property": system_which_property,
    "system_coordinate": system_coordinate,
    "system_cancel_letters": system_cancel_letters,
    "system_constant_r": system_constant_r,
    "system_solve_pair": system_solve_pair,
    "system_word_setup": system_word_setup,
    "system_graph_equations": system_graph_equations,
    "system_times_more": system_times_more,
    "ineq_max_word": ineq_max_word,
    "ineq_setup": ineq_setup,
    "ineq_graph": ineq_graph,
    "ineq_system_points": ineq_system_points,
    "ineq_fence_range": ineq_fence_range,
    "intercepts_std": intercepts_std,
    "point_on_line": point_on_line,
    "line_two_points": line_two_points,
    "parallel_lines": parallel_lines,
    "ineq_system_graph": ineq_system_graph,
    "ineq_solve": ineq_solve,
    "ineq_system_setup": ineq_system_setup,
    "graph_choices": graph_choices,
    "function_notation": function_notation,
    "abs_value": abs_value,
    "point_meaning": point_meaning,
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


# API list prices in US dollars per million tokens (input, output), Nov 2026; Haiku 5.5: prompts up to 100,000 tokens
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00)}
USAGE: dict[str, list[int]] = {}  # model -> [calls, input tokens, output tokens] for this run


def usage_report() -> dict:
    """Tokens and dollars spent on model calls in this run (stories only; the math itself makes no calls)."""
    out, total = {}, 0.0
    for model, (calls, tin, tout) in USAGE.items():
        pin, pout = PRICES.get(model, (0.0, 0.0))
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        total += cost
        out[model] = {"calls": calls, "input_tokens": tin, "output_tokens": tout, "usd": round(cost, 4)}
    return {"models": out, "total_usd": round(total, 4)}


def claude_caller(model: str, temperature: float = 1.0):
    """A prompt -> text function using satgen's client (it handles models that fix their own sampling). Every call's
    tokens are added to USAGE, so a run can report what it cost."""
    from satgen.llm import call_with_retry, make_client

    client = make_client(model)

    def call(prompt: str) -> str:
        text, tin, tout = call_with_retry(client, prompt, temperature=temperature, json_mode=True)
        row = USAGE.setdefault(model, [0, 0, 0])
        row[0] += 1
        row[1] += tin
        row[2] += tout
        return text

    return call


# ----------------------------------------------------------------------------- batches


def plan(count: int, families: list[str], rng: random.Random) -> list[tuple[str, str, str]]:
    """Spread families and levels evenly; about 3 in 4 multiple choice (the rest grid-in), as on the test. The
    family order is shuffled per run, so a batch smaller than the number of families samples every topic instead of
    taking the first ones in the list."""
    families = list(families)
    rng.shuffle(families)
    start = {f: rng.randrange(len(FAMILY_LEVELS.get(f, LEVELS))) for f in families}  # small batches mix levels too
    slots = []
    for i in range(count):
        fam = families[i % len(families)]
        levels = FAMILY_LEVELS.get(fam, LEVELS)
        level = levels[(i // len(families) + start[fam]) % len(levels)]
        fmt = "spr" if rng.random() < 0.25 else "mcq"
        slots.append((fam, level, fmt))
    return slots


def generate(
    count: int, families: list[str], seed: int, story: Story | None = None, order: str | None = None
) -> list[MathItem]:
    rng = random.Random(seed)
    _FRAME_DECK.clear()
    items = []
    used: dict[str, int] = {}
    for fam, level, fmt in plan(count, families, rng):
        item = FAMILIES[fam](rng, level, fmt)
        item.build_choices(rng, used, order)
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
    run.add_argument(
        "--order",
        choices=("random", "ascending"),
        default=CHOICE_ORDER,
        help="number choices: random (any letter can be the answer; default) or ascending (least to greatest, as on "
        "the SAT)",
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
    items = generate(args.count, args.families.split(","), seed, story, args.order)
    (out / "questions.json").write_text(
        json.dumps([i.to_record() for i in items], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if story:
        (out / "stories.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in story.log), encoding="utf-8"
        )
        report = usage_report()
        (out / "usage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for model, row in report["models"].items():
            print(
                f"{model}: {row['calls']} calls, {row['input_tokens']:,} in / {row['output_tokens']:,} out tokens, "
                f"${row['usd']:.4f}"
            )
        print(f"API cost for this run: ${report['total_usd']:.4f}")
    print(f"wrote {len(items)} questions to {out} (seed {seed})")
    import meza_export

    meza_export.build(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
