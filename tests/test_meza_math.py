"""Meza Math: every answer computed in code must be right, every wrong choice a real mistake, every story checked."""

import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import meza_math as mm  # noqa: E402


def value(text: str) -> Fraction:
    return Fraction(text.replace(",", "").replace(mm.MINUS, "-"))


def solve_linear(stem: str) -> Fraction:
    """Independent check of 'If <left> = <right>, what is the value of <expr>?' with x on either side."""
    m = re.match(r"If (.+) = (.+), what is the value of (.+)\?$", stem.replace(mm.MINUS, "-"))
    assert m, stem

    def coeffs(e: str) -> tuple[Fraction, Fraction]:  # e is linear in x, maybe k(x + b)
        e = e.replace(" ", "")
        p = re.fullmatch(r"(\d+)\(x([+-]\d+)\)", e)
        if p:
            k = Fraction(p.group(1))
            return k, k * Fraction(p.group(2))
        a = b = Fraction(0)
        for t in re.findall(r"[+-]?[^+-]+", e):
            if t.endswith("x"):
                c = t[:-1]
                a += Fraction(1 if c in ("", "+") else -1 if c == "-" else c)
            else:
                b += Fraction(t)
        return a, b

    (a1, b1), (a2, b2), (p, q) = coeffs(m.group(1)), coeffs(m.group(2)), coeffs(m.group(3))
    x = (b2 - b1) / (a1 - a2)
    return p * x + q


def test_expression_answers_check_out_independently():
    for seed in range(300):
        rng = random.Random(seed)
        for level in mm.LEVELS:
            item = mm.expression_value(rng, level, "mcq")
            item.build_choices(rng, {})
            assert solve_linear(item.stem) == item.answer, item.stem
            assert value(item.choices[item.key]) == item.answer


def test_rate_answers_follow_from_the_story_numbers():
    for seed in range(300):
        rng = random.Random(seed)
        for level in mm.LEVELS:
            item = mm.linear_rate(rng, level, "spr")
            f = item.story
            r = (f["left1"] - f["left2"]) // (f["t2"] - f["t1"])
            assert r == f["rate"] and f["total"] == r * f["finish"]
            want = {"easy": (f["total"] - f["left2"]) / Fraction(f["rate"]), "medium": f["t2"], "hard": f["finish"]}
            assert item.answer == want[level]
            assert mm.story_numbers_ok(item.stem, f["given"]) == []  # the template story itself passes


def test_choices_are_four_real_mistakes_in_ascending_order():
    filler = total = 0
    used: dict[str, int] = {}
    for seed in range(300):
        rng = random.Random(seed)
        for fam in mm.FAMILIES.values():
            for level in mm.LEVELS:
                item = fam(rng, level, "mcq")
                item.build_choices(rng, used)
                vals = [value(v) for v in item.choices.values()]
                assert len(set(vals)) == 4 and vals == sorted(vals) and min(vals) > 0
                assert len(item.mistakes) == 3 and all(w for _, w in item.mistakes)
                filler += sum("arithmetic slip" in w for _, w in item.mistakes)
                total += 1
                if level != "medium" or item.family != "expression_value":
                    assert all(v.denominator == 1 for v in vals)  # no stray fraction among whole numbers
    assert filler / total < 0.05
    assert min(used.values()) / max(used.values()) > 0.6  # keys spread over A-D


def test_story_checks():
    given = [11500, 4, 9000, 6000]
    good = (
        "Marcus drives a delivery truck. This morning he had 11,500 bags of rice to load. After 4 hours, 9,000 "
        "bags were still left. How many hours after he started will exactly 6,000 bags be left?"
    )
    assert mm.story_numbers_ok(good, given) == []
    assert mm.story_numbers_ok(good.replace("11,500", "11500"), given)  # SAT style uses commas
    assert mm.story_numbers_ok(good + " He started at 7 a.m.?", given)  # no extra numbers
    assert mm.story_numbers_ok(good.rstrip("?") + ".", given)  # must end with the question
    swapped = (
        "Marcus had 11,500 bags of rice to load. He loaded 9,000 bags in 4 hours. How many hours after he "
        "started will exactly 6,000 bags be left?"
    )
    assert any("order" in x for x in mm.story_numbers_ok(swapped, given))  # numbers must keep their roles


def test_story_writer_falls_back_to_template_when_solver_disagrees():
    rng = random.Random(3)
    item = mm.linear_rate(rng, "medium", "spr")
    template = item.stem
    f = item.story
    story = ("{n} works hard. {n} had {a} to do. After {b} hours, {c} remained. When will {d} remain?").format(
        n=f["name"], a=mm.num(f["total"]), b=f["t1"], c=mm.num(f["left1"]), d=mm.num(f["left2"])
    )
    wrong = mm.Story(lambda p: json.dumps({"stem": story}), lambda p: '{"answer": "1"}')
    wrong.tell(item)
    assert item.stem == template and wrong.log[-1]["problems"]
    right = mm.Story(lambda p: json.dumps({"stem": story}), lambda p: json.dumps({"answer": str(item.answer)}))
    right.tell(item)
    assert item.stem == story


def test_settings_rotate_within_a_batch():
    items = mm.generate(6, ["linear_rate"], seed=1)
    assert len({i.story["role"] for i in items}) == 6


def test_offline_run_writes_review_files(tmp_path):
    assert mm.main(["run", "--count", "6", "--offline", "--seed", "5", "--out", str(tmp_path)]) == 0
    qs = json.loads((tmp_path / "questions.json").read_text())
    assert len(qs) == 6 and (tmp_path / "review.html").exists()
    assert all(q["explanation"].startswith("[Domain: Algebra]") for q in qs)
