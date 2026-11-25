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
        for name, fam in mm.FAMILIES.items():
            for level in mm.FAMILY_LEVELS[name]:
                item = fam(rng, level, "mcq")
                item.build_choices(rng, used)
                assert len(set(item.choices.values())) == 4
                assert len(item.mistakes) == 3 and all(w for _, w in item.mistakes)
                total += 1
                if item.key_text:  # equations or phrases
                    continue
                vals = [value(v) for v in item.choices.values()]
                assert vals == sorted(vals) and (min(vals) > 0 or not item.positive_only)
                filler += sum("arithmetic slip" in w for _, w in item.mistakes)
                if not item.fractions_ok and (level != "medium" or item.family != "expression_value"):
                    assert all(v.denominator == 1 for v in vals)  # no stray fraction among whole numbers
    assert filler / total < 0.05
    assert min(used.values()) / max(used.values()) > 0.8  # keys spread over A-D across a mixed batch


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


def test_default_run_is_free_and_writes_review_files(tmp_path, monkeypatch):
    def no_api(*a, **k):
        raise AssertionError("a default run must not call the API")

    monkeypatch.setattr(mm, "claude_caller", no_api)
    assert mm.main(["run", "--count", "6", "--seed", "5", "--out", str(tmp_path)]) == 0
    qs = json.loads((tmp_path / "questions.json").read_text())
    assert len(qs) == 6 and (tmp_path / "review.html").exists()
    assert all(q["explanation"].startswith("[Domain: Algebra]") for q in qs)


# --- the linear-equation families added Nov 25: checked by substitution, independently of how they were built ---


def evaluate(expr: str, **values) -> Fraction:
    """Evaluate an SAT-style expression ('5x + x + 99', 'x(r − 3) + 4x', '24f', 'x/6') exactly."""
    e = expr.replace(mm.MINUS, "-")
    e = re.sub(r"(?<=\d),(?=\d{3})", "", e)  # thousands commas
    e = re.sub(r"\d+(?:\.\d+)?", lambda m: f'F("{m.group(0)}")', e)
    e = re.sub(r"(\))\s*(?=[A-Za-z(])", r"\1*", e)  # ")x", ")(" -> ")*x"
    e = re.sub(r"\b([a-z])\s*(?=\()", r"\1*", e)  # "x(" -> "x*("
    e = re.sub(r"(?<![A-Za-z])([a-z])(?=[a-z])", r"\1*", e)
    names = {"F": Fraction, **{k: Fraction(v) for k, v in values.items()}}
    return eval(e, {"__builtins__": {}}, names)  # noqa: S307 (test-only, our own generated text)


def sides(equation: str):
    left, right = equation.split("=")
    return left, right


def gen(family: str, n: int = 150):
    for seed in range(n):
        rng = random.Random(seed)
        for level in mm.FAMILY_LEVELS[family]:
            item = mm.FAMILIES[family](rng, level, "mcq")
            item.build_choices(rng, {})
            yield item


def test_simple_and_combine_solutions_check_by_substitution():
    for family in ("solve_simple", "solve_combine"):
        for item in gen(family):
            v = re.search(r"value of (\w)", item.stem).group(1)
            left, right = sides(item.passage)
            for k, text in item.choices.items():
                holds = evaluate(left, **{v: value(text)}) == evaluate(right, **{v: value(text)})
                assert holds == (k == item.key), (item.passage, text)


def test_cannot_value_is_the_only_choice_without_exactly_one_solution():
    for item in gen("one_solution_cannot"):
        left, right = sides(item.passage)
        for k, text in item.choices.items():
            r = value(text)
            slope = (evaluate(left, x=1, r=r) - evaluate(right, x=1, r=r)) - (
                evaluate(left, x=0, r=r) - evaluate(right, x=0, r=r)
            )
            assert (slope == 0) == (k == item.key), (item.passage, text)


def test_infinite_constant_makes_both_sides_identical():
    for item in gen("infinite_constant"):
        c = re.search(r"equation, (\w) is a constant", item.stem).group(1)
        left, right = sides(item.passage)
        for k, text in item.choices.items():
            same = all(
                evaluate(left, x=x, **{c: value(text)}) == evaluate(right, x=x, **{c: value(text)}) for x in (0, 1, 2)
            )
            assert same == (k == item.key), (item.passage, text)


def test_which_equation_choices_hold_only_for_the_key():
    for family in ("three_kinds_equation", "payment_equation"):
        for item in gen(family):
            v = re.search(r"where (\w) is|If (\w) represents", item.stem)
            v = v.group(1) or v.group(2)
            for k, text in item.choices.items():
                left, right = sides(text)
                holds = evaluate(left, **{v: item.story["truth"]}) == evaluate(right, **{v: item.story["truth"]})
                assert holds == (k == item.key), (item.stem, text)


def test_interpret_parts_asks_about_a_number_in_the_equation():
    for item in gen("interpret_parts"):
        eq = re.search(r"The equation (.+?) represents", item.stem).group(1)
        asked = re.search(r"What does (.+?) represent", item.stem).group(1)
        assert asked in eq and len(set(item.choices.values())) == 4 and item.choices[item.key] == item.key_text


def test_new_families_rarely_need_filler():
    for family in ("solve_simple", "solve_combine", "one_solution_cannot", "infinite_constant"):
        items = list(gen(family))
        filler = sum(any("slip" in w for _, w in i.mistakes) for i in items)
        assert filler / len(items) < 0.1, family


def test_in_terms_of_only_the_key_gives_a_solution():
    for item in gen("in_terms_of"):
        group = re.search(r"value of (.+)\?$", item.stem).group(1)  # "x + 5"
        c = evaluate(group, x=0)
        left, right = sides(item.passage)
        assert re.search(r"\((\d?)x", left) and group in right
        for k, text in item.choices.items():
            x = value(text) - c  # this choice as the value of the group means x = choice − c
            assert (evaluate(left, x=x) == evaluate(right, x=x)) == (k == item.key), (item.passage, text)


def classify(left: str, right: str, **consts) -> str:
    """'Zero', 'Exactly one', or 'Infinitely many' for a linear equation in x."""
    f0 = evaluate(left, x=0, **consts) - evaluate(right, x=0, **consts)
    slope = evaluate(left, x=1, **consts) - evaluate(right, x=1, **consts) - f0
    return "Exactly one" if slope else ("Infinitely many" if f0 == 0 else "Zero")


def test_reciprocal_and_same_product_keys():
    for item in gen("reciprocal_ratio"):
        a, b, c = map(int, re.match(r"If x/(\d+) = (\d+), what is the value of (\d+)/x\?", item.stem).groups())
        for k, text in item.choices.items():
            assert (value(text) == Fraction(c, a * b)) == (k == item.key)
    for item in gen("same_product"):
        left, c = re.match(r"If (.+) = (\d+), what is the value of", item.stem).groups()
        b = int(re.search(r"\((\d+)x\)", left).group(1))
        x = next(Fraction(x) for x in range(1, 200) if evaluate(left, x=x) == int(c))
        for k, text in item.choices.items():
            assert (value(text) == b * x) == (k == item.key)


def test_no_solution_keys_are_the_only_choices_with_no_solution():
    for family in ("no_solution_parens", "no_solution_k"):
        for item in gen(family):
            const = re.search(r"equation, (\w) is a constant", item.stem).group(1)
            left, right = sides(item.passage)
            for k, text in item.choices.items():
                got = classify(left, right, **{const: value(text)})
                assert (got == "Zero") == (k == item.key), (item.passage, text, got)


def test_how_many_solutions_key_matches_the_equation():
    seen = set()
    for item in gen("how_many_solutions"):
        assert classify(*sides(item.passage)) == item.choices[item.key]
        assert list(item.choices.values()) == mm.SOLUTION_COUNTS
        seen.add(item.choices[item.key])
    assert seen == {"Zero", "Exactly one", "Infinitely many"}


def test_infinite_two_constants_truth_makes_both_sides_identical():
    for item in gen("infinite_two_constants"):
        left, right = sides(item.passage)
        assert classify(left, right, **item.truth) == "Infinitely many"
        asked = re.search(r"value of (.+)\?", item.stem).group(1)
        assert value(item.choices[item.key]) == evaluate(asked, **item.truth)
        assert sum(value(t) == evaluate(asked, **item.truth) for t in item.choices.values()) == 1


def test_fraction_decimal_keys_solve_the_equation():
    for item in gen("fraction_decimal"):
        left, right = sides(item.passage)
        for k, text in item.choices.items():
            x = value(text)
            assert (evaluate(left, x=x) == evaluate(right, x=x)) == (k == item.key), (item.passage, text)


def test_word_solve_keys_follow_from_the_story():
    for item in gen("word_solve"):
        t = item.truth
        if item.level == "easy":
            want = (t["T"] - t["F"]) / t["r"]
            assert mm.money(t["F"]) in item.stem and mm.money(t["T"]) in item.stem
        elif item.level == "medium":
            want = t["w"] if "width of" in item.stem else t["L"]
            P = int(re.search(r"perimeter of the garden is (\d+)", item.stem).group(1))
            assert 2 * (t["w"] + t["L"]) == P
        else:
            S = int(re.search(r"is (\d+)\.", item.stem).group(1))
            assert sum(t["nums"]) == S
            want = t["nums"][-1] if "greatest" in item.stem else t["nums"][0]
        assert value(item.choices[item.key]) == want
        assert sum(value(c) == want for c in item.choices.values()) == 1


def test_which_has_property_only_the_key_has_it():
    for item in gen("which_has_property"):
        want = mm.PROPERTIES[re.match(r"Which equation has (.+)\?", item.stem).group(1)]
        for k, eq in item.choices.items():
            assert (classify(*sides(eq)) == want) == (k == item.key), (item.stem, eq)


# --- linear functions ---


def test_interpret_models_key_names_the_right_thing_and_role():
    roles = set()
    for item in gen("interpret_models", 200):
        t = item.truth
        asked = re.search(r"interpretation of (.+?) in this context", item.stem).group(1)
        size = asked.lstrip(mm.MINUS)
        assert item.stem.count(asked) >= 2  # in an equation, and in the question
        key = item.choices[item.key]
        assert size in key and (mm._cap(t["obj"]) in key or t["obj"] in key)
        assert ("each" in key) == (t["role"] == "rate")
        assert sum(size in c for c in item.choices.values()) == 4  # every choice uses the same number
        if item.level == "hard":  # the two-model trap: the same role for the other thing
            assert any(t["obj"] not in c and t["obj"].capitalize() not in c for c in item.choices.values())
        roles.add(t["role"])
    assert roles == {"rate", "start"}


def rhs(equation: str) -> str:
    return equation.split("=", 1)[1]


def test_graphs_have_a_full_grid_and_the_key_is_on_the_line():
    for item in gen("graph_estimate", 100):
        svg = item.figure
        assert svg.startswith("<svg") and svg.count('stroke="#cfcfcf"') >= 22  # 11+ lines each way, incl. edges
        t = item.truth
        want = t["y0"] if item.level != "hard" else t["x0"]
        assert t["y0"] == t["m"] * t["x0"] + t["b"]
        assert sum(value(c) == want for c in item.choices.values()) == 1 and value(item.choices[item.key]) == want
    for item in gen("graph_slope", 100):
        t = item.truth
        key = item.choices[item.key]
        assert mm.dec(abs(t["m"]) * t["k"]) in key
        assert ("increases" in key) == (t["m"] > 0)


def test_function_choices_only_the_key_matches_the_situation():
    for item in gen("rate_total_function"):
        t = item.truth
        for k, eq in item.choices.items():
            assert (evaluate(rhs(eq), x=3) == t["r"] * (3 + t["a"])) == (k == item.key), eq
    for item in gen("two_points_function"):
        xs = [int(v.replace(mm.MINUS, "-")) for v in re.findall(r"\((−?\d+)\) =", item.stem)]
        ys = [Fraction(v.replace(mm.MINUS, "-").replace(",", "")) for v in re.findall(r"\) = (−?[\d,]+)", item.stem)]
        for k, eq in item.choices.items():
            fits = all(evaluate(rhs(eq), x=x) == y for x, y in zip(xs, ys, strict=True))
            assert fits == (k == item.key), (item.stem, eq)
    for item in gen("cost_function"):
        t = item.truth
        for k, eq in item.choices.items():
            if item.level == "easy":
                true = [t["fee"] + t["rate"] * mth for mth in (2, 5)]
                got = [evaluate(rhs(eq), m=mth) for mth in (2, 5)]
            else:
                hours = [t["h"] + 1, t["h"] + 4]
                xs_ = [1, 4] if item.level == "medium" else hours  # hours past the first h, or total hours
                true = [t["h"] * t["a"] + t["r"] * (hh - t["h"]) for hh in hours]
                got = [evaluate(rhs(eq), x=x) for x in xs_]
            assert (got == true) == (k == item.key), (item.stem, eq)


def test_intercepts_and_function_change():
    for item in gen("intercept_meaning"):
        t = item.truth
        assert t["m"] * t["x0"] + t["b"] == 0
        key = item.choices[item.key]
        assert mm.num(t["b"] if t["which"] == "y" else t["x0"]) in key
    for item in gen("intercept_from_slope"):
        t = item.truth
        assert t["m"] * t["a"] + t["b"] == 0
        want = t["a"] if "x-coordinate" in item.stem else t["b"]
        assert value(item.choices[item.key]) == want
    for item in gen("function_change"):
        assert value(item.choices[item.key]) == item.truth["m"] * item.truth["d"]


def test_tables_and_combined_functions():
    for family in ("table_equation", "table_mb"):
        for item in gen(family):
            t = item.truth
            assert all(y == t["m"] * x + t["b"] for x, y in t["rows"])
            if family == "table_equation":
                for k, eq in item.choices.items():
                    fits = all(evaluate(rhs(eq), x=x) == y for x, y in t["rows"])
                    assert fits == (k == item.key), eq
            else:
                assert value(item.choices[item.key]) == t[t["ask"]]
    for item in gen("table_shift"):
        t = item.truth
        inner = re.search(r"g\(x\) = (.+)", t["g"]).group(1)
        for k, eq in item.choices.items():
            fits = True
            for x in (0, 1, 5):
                f_in = evaluate(re.search(r"f\((.+?)\)", inner).group(1), x=x)  # the argument of f
                g_true = t["m"] * f_in + t["b"] + evaluate(inner.split(")", 1)[1] or "0")
                fits &= evaluate(rhs(eq), x=x) == g_true
            assert fits == (k == item.key), (t["g"], eq)
    for item in gen("sum_functions"):
        t = item.truth
        want = {"slope": t["M"], "y-intercept": t["B"], "x-intercept": Fraction(-t["B"], t["M"])}[t["ask"]]
        assert value(item.choices[item.key]) == want
        assert sum(value(c) == want for c in item.choices.values()) == 1
