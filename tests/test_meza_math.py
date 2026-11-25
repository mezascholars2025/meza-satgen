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
                item.build_choices(rng, used, order="ascending")
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


# ------------------------------------------------- linear functions extras, two-variable equations, systems (Nov 25)

REL = ("≤", "≥", "<", ">", "=")


def split_rel(text: str) -> tuple[str, str, str]:
    for rel in REL:
        if rel in text:
            left, right = text.split(rel, 1)
            return left, rel, right
    raise AssertionError(text)


def line_of(eq: str, **consts) -> tuple[Fraction, Fraction, Fraction]:
    """(A, B, C) with Ax + By = C for any linear equation in x and y, found by evaluating it at three points."""
    left, rel, right = split_rel(eq)
    assert rel == "="

    def f(x, y):
        return evaluate(left, x=x, y=y, **consts) - evaluate(right, x=x, y=y, **consts)

    f0 = f(0, 0)
    return f(1, 0) - f0, f(0, 1) - f0, -f0


def solve2(e1: str, e2: str, **consts):
    """The solution of a system, or 'none' / 'infinite'."""
    (a1, b1, c1), (a2, b2, c2) = line_of(e1, **consts), line_of(e2, **consts)
    det = a1 * b2 - a2 * b1
    if det:
        return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)
    same = a1 * c2 == a2 * c1 and b1 * c2 == b2 * c1
    return "infinite" if same else "none"


def table_rows(text: str) -> list[tuple[Fraction, Fraction]]:
    rows = [r for r in text.splitlines() if r.startswith("|") and not r.startswith("|---")][1:]
    return [tuple(value(c.strip()) for c in r.strip("|").split("|")) for r in rows]


def test_plane_graphs_and_which_equation():
    for item in gen("graph_which_equation", 100):
        assert item.figure.count('stroke="#cfcfcf"') >= 34  # 17 lines each way: −8 to 8
        m, b = item.truth["m"], item.truth["b"]
        for k, eq in item.choices.items():
            on = all(line_of(eq)[0] * x + line_of(eq)[1] * (m * x + b) == line_of(eq)[2] for x in (0, 3))
            assert on == (k == item.key), eq


def test_evaluate_function_and_unit_conversion():
    for item in gen("evaluate_function"):
        rule = re.search(r"defined by \w\(x\) = (.+?)(?:, where|\. )", item.stem).group(1)
        key = value(item.choices[item.key])
        if item.level == "easy":
            x = int(re.search(r"value of \w\((\d+)\)", item.stem).group(1))
            assert evaluate(rule, x=x) == key
        elif item.level == "medium":
            v = value(re.search(r"\(k\) = (−?[\d,]+)", item.stem).group(1))
            assert evaluate(rule, x=key) == v
        else:
            x1, y1, x2 = re.search(r"If \w\((\d+)\) = (−?\d+), what is the value of \w\((\d+)\)", item.stem).groups()
            c = value(y1) - evaluate(rule, x=int(x1), c=0)
            assert evaluate(rule, x=int(x2), c=c) == key
    for item in gen("unit_conversion_function"):
        t = item.truth
        for k, eq in item.choices.items():
            fits = all(evaluate(rhs(eq), **{t["var"]: w}) == Fraction(t["r"] * w + t["s"], t["k"]) for w in (0, 3))
            assert fits == (k == item.key), eq


def test_two_lines_graph_keys():
    for item in gen("two_lines_graph"):
        t = item.truth
        (m1, m2), (b1, b2) = t["m"], t["b"]
        assert m1 * t["xi"] + b1 == m2 * t["xi"] + b2 == t["yi"] and m1 > m2 and 0 <= b1 < b2
        key = value(item.choices[item.key])
        if item.level == "medium":
            assert key == t["xi"]
        else:
            period = value(re.search(r"during each ([\d.]+)-", item.stem).group(1))
            assert key == period * (m1 - m2)
        assert "Plant A" in item.figure or "Tank P" in item.figure or "stroke-dasharray" in item.figure


def test_two_variable_equation_choices():
    for item in gen("two_var_equation"):
        t = item.truth
        for k, text in item.choices.items():
            left, rel, right = split_rel(text)
            ok = (
                evaluate(left, x=1, y=0) == t["a"]
                and evaluate(left, x=0, y=1) == t["b"]
                and evaluate(left, x=0, y=0) == 0
                and evaluate(right) == t["c"]
                and rel == t["rel"]
            )
            assert ok == (k == item.key), (item.stem, text)
        if item.level == "medium":
            assert ("at least" in item.stem) == (t["rel"] == "≥")
    for item in gen("table_for_equation"):
        eq = re.search(r"for the equation (.+)\?$", item.stem).group(1)
        for k, text in item.choices.items():
            A, B, C = line_of(eq)
            fits = all(A * x + B * y == C for x, y in table_rows(text))
            assert fits == (k == item.key), (eq, text)


def test_tradeoff_slope_point_and_isosceles():
    for item in gen("tradeoff"):
        a, b = item.truth["a"], item.truth["b"]
        key = value(item.choices[item.key])
        if item.level == "hard":
            assert key == Fraction(a, b)
        else:
            k = int(re.search(r"(\d+) (?:more|fewer)", item.stem.split(". ")[-1]).group(1))
            assert key == (Fraction(a * k, b) if item.level == "easy" else Fraction(b * k, a))
    for item in gen("slope_point_line"):
        m = item.truth["m"]
        x1, y1 = item.truth["pt"]
        if item.level == "hard":
            A, B, _ = line_of(re.search(r"parallel to the line (.+?) and passes", item.stem).group(1))
            assert -A / B == m
        for k, eq in item.choices.items():
            f = rhs(eq)
            ok = evaluate(f, x=x1) == y1 and evaluate(f, x=1) - evaluate(f, x=0) == m
            assert ok == (k == item.key), eq
    for item in gen("isosceles_perimeter"):
        P = int(re.search(r"perimeter of the triangle is (\d+)", item.stem).group(1))
        feet_perimeter = "is " + str(P) + " feet" in item.stem
        third_in_feet = "y feet long" in item.stem
        want = (2, 12 if third_in_feet else 1, 12 * P if feet_perimeter else P)
        for k, eq in item.choices.items():
            left, right = sides(eq)
            got = (evaluate(left, x=1, y=0), evaluate(left, x=0, y=1), evaluate(right))
            assert (got == want) == (k == item.key), eq


def test_perpendicular_and_tables_with_unknowns():
    for item in gen("perpendicular_vertical"):
        given = re.search(r"to the line ([xy]) = ", item.stem).group(1)
        key = item.choices[item.key]
        if item.level == "easy":
            assert key == ("0" if given == "x" else "The slope is undefined.")
        else:
            px, py = re.search(r"point \((−?\d+), (−?\d+)\)", item.stem).groups()
            assert key == (f"y = {py}" if given == "x" else f"x = {px}")
    for item in gen("perpendicular_lines"):
        given = re.search(r"to the line (.+?)(?:\. What| and passes)", item.stem).group(1)
        A, B, _ = line_of(given)
        m = -A / B
        if item.level != "medium":
            for k, text in item.choices.items():
                assert (value(text) * m == -1) == (k == item.key), (given, text)
        else:
            px, py = (value(v) for v in re.search(r"point \((−?\d+), (−?\d+)\)", item.stem).groups())
            for k, eq in item.choices.items():
                f = rhs(eq)
                ok = (evaluate(f, x=1) - evaluate(f, x=0)) * m == -1 and evaluate(f, x=px) == py
                assert ok == (k == item.key), (given, eq)
    for item in gen("table_unknowns"):
        rows = [r for r in item.passage.splitlines()[2:]]
        cells = [r.strip("|").split("|") for r in rows]
        known = [(value(x.strip()), value(y.strip())) for x, y in cells if y.strip() != "k"]
        xk = next(value(x.strip()) for x, y in cells if y.strip() == "k")
        (xa, ya), (xb, yb) = known
        assert value(item.choices[item.key]) == ya + (yb - ya) / (xb - xa) * (xk - xa)
    for item in gen("interpret_coefficients"):
        assert item.choices[item.key] == item.key_text and len(set(item.choices.values())) == 4


def test_systems_which_property_and_constants():
    for item in gen("system_which_property"):
        want = item.truth["want"]
        for k, text in item.choices.items():
            sol = solve2(*text.split("\n"))
            kind = sol if isinstance(sol, str) else "one"
            assert (kind == want) == (k == item.key), text
    for item in gen("system_constant_r"):
        letter = item.truth["letter"]
        e1, e2 = item.passage.split("\n")
        for k, text in item.choices.items():
            sol = solve2(e1, e2, **{letter: value(text)})
            if "infinitely many" in item.stem:
                hit = sol == "infinite"
            else:  # "no solution", or the value that CANNOT give exactly one solution
                hit = sol == "none"
            assert hit == (k == item.key), (item.passage, text)


def test_systems_coordinates_pairs_and_letters():
    for item in gen("system_coordinate"):
        x0, y0 = solve2(*item.passage.split("\n"))
        asked = re.search(r"value of (.+)\?$", item.stem).group(1)
        want = evaluate(asked, x=x0, y=y0)
        for k, text in item.choices.items():
            assert (value(text) == want) == (k == item.key), (item.passage, asked, text)
    for item in gen("system_solve_pair"):
        for k, text in item.choices.items():
            px, py = (value(v) for v in text.strip("()").split(", "))
            if item.passage:
                ok = all(
                    evaluate(sides(e)[0], x=px, y=py) == evaluate(sides(e)[1], x=px, y=py)
                    for e in item.passage.split("\n")
                )
            else:  # graphed: the point is on both drawn lines
                ok = (px, py) == item.truth["sol"]
            assert ok == (k == item.key), (item.passage, text)
    for item in gen("system_cancel_letters"):
        e1, e2 = item.passage.split("\n")
        asked = "x" if "value of x" in item.stem else "y"
        for consts in ({"a": 2, "b": 3, "k": 2, "c": 5, "r": 7}, {"a": 7, "b": 1, "k": -4, "c": 1, "r": 3}):
            names = {v: consts[v] for v in item.truth["letters"]}
            sol = solve2(e1, e2, **names)
            want = sol[0] if asked == "x" else sol[1]
            assert value(item.choices[item.key]) == want
            assert sum(value(c) == want for c in item.choices.values()) == 1


def test_systems_from_words_and_graphs():
    for item in gen("system_word_setup"):
        x, y = item.truth["sol"]
        for k, text in item.choices.items():
            holds = all(evaluate(sides(e)[0], x=x, y=y) == evaluate(sides(e)[1]) for e in text.split("\n"))
            assert holds == (k == item.key), (item.stem, text)
    for item in gen("system_graph_equations"):
        lines = item.truth["lines"]
        for k, text in item.choices.items():
            if item.level == "hard":
                ok = solve2(*text.split("\n")) == tuple(Fraction(v) for v in item.truth["sol"])
            else:
                got = set()
                for e in text.split("\n"):
                    A, B, C = line_of(e)
                    got.add((-A / B, C / B))
                ok = got == {(Fraction(m), Fraction(b)) for m, b in lines}
            assert ok == (k == item.key), text
        if item.level == "hard":
            (m1, b1), (m2, b2) = lines
            x0, y0 = item.truth["sol"]
            assert m1 * x0 + b1 == y0 == m2 * x0 + b2
    for item in gen("system_times_more"):
        t = item.truth
        assert t["a"] == t["k"] * t["b"] + t["extra"]
        given = [
            value(v) for v in re.findall(r"(?<![\w.])(\d[\d,]*)(?= (?:more|students|cookies|pine|votes))", item.stem)
        ]
        if t["level"] == "easy":
            assert t["a"] + t["b"] in given
        else:
            assert t["a"] - t["b"] in given
        assert value(item.choices[item.key]) == t[t["ask"]]


# ------------------------------------------------------------------------------- linear inequalities (Nov 25)


def holds(rel: str, a, b) -> bool:
    return {"<": a < b, "≤": a <= b, ">": a > b, "≥": a >= b, "=": a == b}[rel]


def chain(text: str):
    """'0 < x ≤ 18', '6.25 ≤ x ≤ 18', 'x ≤ 17', 'x ≥ 21' -> a test for a number."""
    parts = re.split(r"\s*(<|≤|>|≥)\s*", text.replace(mm.MINUS, "-"))
    vals = [p if p == "x" else Fraction(p) for p in parts[::2]]
    rels = parts[1::2]

    def ok(x: Fraction) -> bool:
        seq = [x if v == "x" else v for v in vals]
        return all(holds(r, a, b) for r, a, b in zip(rels, seq, seq[1:], strict=False))

    return ok


def test_inequality_word_problems():
    for item in gen("ineq_max_word"):
        t = item.truth
        key = value(item.choices[item.key])
        if t["kind"] == "max":
            want = max(n for n in range(0, 500) if t["F"] + t["r"] * n <= t["B"])
        elif t["kind"] == "min":
            want = min(n for n in range(0, 500) if t["F"] + t["r"] * n >= t["B"])
        else:
            want = max(n for n in range(0, 500) if t["a"] * n + t["b"] * t["y"] <= t["B"])
        assert key == want and sum(value(c) == want for c in item.choices.values()) == 1
    for item in gen("ineq_setup"):
        t = item.truth
        for k, text in item.choices.items():
            if t["rel"] == "range":
                lo, r1, mid, r2, hi = re.split(r"\s*(≤|<)\s*", text)
                ok = (
                    evaluate(mid, x=0) == t["F"]
                    and evaluate(mid, x=1) - evaluate(mid, x=0) == t["r"]
                    and value(lo) == t["lo"]
                    and value(hi) == t["hi"]
                    and r1 == r2 == "≤"
                )
            else:
                left, rel, right = split_rel(text)
                ok = (
                    evaluate(left, x=0) == t["F"]
                    and evaluate(left, x=1) - evaluate(left, x=0) == t["r"]
                    and value(right) == t["B"]
                    and rel == t["rel"]
                )
            assert ok == (k == item.key), (item.stem, text)


def test_inequality_graphs_and_points():
    for item in gen("ineq_graph"):
        m, b, rel = item.truth["m"], item.truth["b"], item.truth["rel"]
        assert "<polygon" in item.figure and (("stroke-dasharray" in item.figure) == (rel in "<>"))
        probes = [(Fraction(x), m * x + b + d) for x in (-3, 0, 3) for d in (1, -1, 0)]  # above, below, on

        def region(text, probes=probes):
            left, r, right = split_rel(text)
            return [holds(r, evaluate(left, x=x, y=y), evaluate(right, x=x, y=y)) for x, y in probes]

        want = [rel in "≥>", rel in "≤<", rel in "≤≥"] * 3
        for k, text in item.choices.items():
            assert (region(text) == want) == (k == item.key), (rel, text)
    for item in gen("ineq_system_points"):
        ineqs = item.passage.split("\n")
        for k, text in item.choices.items():
            x, y = (value(v) for v in text.strip("()").split(", "))
            both = all(
                holds(split_rel(e)[1], evaluate(split_rel(e)[0], x=x, y=y), evaluate(split_rel(e)[2], x=x, y=y))
                for e in ineqs
            )
            want_key = not both if "NOT" in item.stem else both
            assert want_key == (k == item.key), (item.passage, text)


def test_fence_range_key_matches_the_constraints():
    for item in gen("ineq_fence_range"):
        side = int(re.search(r"One side of the pen is (\d+) yards", item.stem).group(1))
        fence = int(re.search(r"at most (\d+) yards of fencing", item.stem).group(1))
        area = re.search(r"(at least|more than) ([\d,]+) square yards", item.stem)
        grid = [Fraction(n, 4) for n in range(0, 4 * 80)]

        def allowed(x, side=side, fence=fence, area=area):
            ok = 0 < x and 2 * side + 2 * x <= fence
            if area:
                a = value(area.group(2))
                ok &= side * x >= a if area.group(1) == "at least" else side * x > a
            return ok

        truth = [allowed(x) for x in grid]
        for k, text in item.choices.items():
            test = chain(text)
            assert ([test(x) for x in grid] == truth) == (k == item.key), (item.stem, text)


# ------------------------------------------- intercepts, points on lines, two points, parallel lines, graphed systems


def pair_of(text: str) -> tuple[Fraction, Fraction]:
    x, y = (value(v) for v in text.strip("()").split(", "))
    return x, y


def on_eq(eq: str, x, y, **consts) -> bool:
    left, right = sides(eq)
    return evaluate(left, x=x, y=y, **consts) == evaluate(right, x=x, y=y, **consts)


def test_intercepts_and_points_on_lines():
    for item in gen("intercepts_std"):
        if item.level == "hard":
            eq = re.search(r"graph of (.+?), where k", item.stem).group(1)
            xi = value(re.search(r"x-intercept at \((−?[\d,]+), 0\)", item.stem).group(1))
            for k, text in item.choices.items():
                assert on_eq(eq, xi, 0, k=value(text)) == (k == item.key), (eq, text)
        else:
            eq = re.search(r"graph of (.+) in the xy-plane", item.stem).group(1)
            axis = "x" if "x-intercept" in item.stem else "y"
            for k, text in item.choices.items():
                x, y = pair_of(text)
                ok = on_eq(eq, x, y) and (y == 0 if axis == "x" else x == 0)
                assert ok == (k == item.key), (eq, text)
    for item in gen("point_on_line"):
        if item.level == "hard":
            eq = re.search(r"lies on the line (.+) in the xy-plane", item.stem).group(1)
            y0 = value(re.search(r"\(k, (−?\d+)\)", item.stem).group(1))
            for k, text in item.choices.items():
                assert on_eq(eq, value(text), y0) == (k == item.key), (eq, text)
        else:
            eq = re.search(r"graph of (.+) in the xy-plane", item.stem).group(1)
            for k, text in item.choices.items():
                assert on_eq(eq, *pair_of(text)) == (k == item.key), (eq, text)


def test_line_through_two_points_and_parallel_lines():
    for item in gen("line_two_points"):
        (x1, y1), (x2, y2) = (pair_of(p) for p in re.findall(r"\(−?\d+, −?\d+\)", item.stem))
        m = (y2 - y1) / (x2 - x1)
        if item.level == "hard":
            assert value(item.choices[item.key]) == y1 - m * x1
            assert sum(value(c) == y1 - m * x1 for c in item.choices.values()) == 1
        else:
            for k, eq in item.choices.items():
                assert (on_eq(eq, x1, y1) and on_eq(eq, x2, y2)) == (k == item.key), eq
    for item in gen("parallel_lines"):
        given = re.search(r"parallel to the line (.+)\?$", item.stem).group(1)
        A, B, C = line_of(given)
        for k, eq in item.choices.items():
            a, b, c = line_of(eq)
            parallel = a * B == A * b and not (a * C == A * c and b * C == B * c)
            assert parallel == (k == item.key), (given, eq)


def test_graphed_system_of_inequalities():
    for item in gen("ineq_system_graph"):
        (m1, b1, r1), (m2, b2, r2) = item.truth["lines"]
        assert item.figure.count("<polygon") == 2 and "stroke-dasharray" in item.figure
        if item.level == "hard":
            for k, text in item.choices.items():
                e1, e2 = text.split("\n")
                same = split_rel(e1)[1] == r1 and split_rel(e2)[1] == r2
                assert line_of(e1.replace(r1 if same else split_rel(e1)[1], "=")) == line_of(f"y = {mm.linf(m1, b1)}")
                assert same == (k == item.key), text
        else:
            for k, text in item.choices.items():
                x, y = pair_of(text)
                ok = holds(r1, y, m1 * x + b1) and holds(r2, y, m2 * x + b2)
                assert ok == (k == item.key), text


def test_every_family_spreads_its_key_over_at_least_three_letters():
    for name, fam in mm.FAMILIES.items():
        used: dict[str, int] = {}
        for seed in range(120):
            rng = random.Random(seed)
            for level in mm.FAMILY_LEVELS[name]:
                fam(rng, level, "mcq").build_choices(rng, used)
        total = sum(used.values())
        shares = sorted(used.get(k, 0) / total for k in "ABCD")
        assert shares[1] >= 0.1, (name, used)  # three letters at 10% or more each


def test_random_order_puts_the_answer_on_any_letter():
    """The default (Sebastian, Nov 25): number choices in random order, so every family uses all four letters."""
    assert mm.CHOICE_ORDER == "random"
    filler = total = 0
    for name, fam in mm.FAMILIES.items():
        used: dict[str, int] = {}
        for seed in range(120):
            rng = random.Random(seed)
            for level in mm.FAMILY_LEVELS[name]:
                item = fam(rng, level, "mcq")
                item.build_choices(rng, used)
                assert len(set(item.choices.values())) == 4 and len(item.mistakes) == 3
                if item.key_text or item.fixed_choices:
                    continue
                show = mm.dec if item.decimals else mm.num
                assert item.choices[item.key] == show(item.answer)
                assert sorted(v for k, v in item.choices.items() if k != item.key) == sorted(
                    show(v) for v, _ in item.mistakes
                )
                total += 1
                filler += sum("arithmetic slip" in w for _, w in item.mistakes)
        if name == "how_many_solutions":  # a fixed list: Zero / Exactly one / Exactly two / Infinitely many
            continue
        shares = [used.get(k, 0) / sum(used.values()) for k in "ABCD"]
        assert min(shares) >= 0.2, (name, used)
    assert filler / total < 0.05


def test_fractions_are_stacked_on_the_review_page():
    """Sebastian (Nov 25): fractions print vertically, as on the SAT. The data keeps the slash; the export stacks it,
    and with the styles stripped (pasted somewhere plain) the text still reads the same as the original."""
    import meza_export as me

    def plain(html_: str) -> str:
        return re.sub(r"<[^>]+>", "", re.sub(r'<span style="display:none">([^<]*)</span>', r"\1", html_))

    cases = {
        "a slope of −1/3 and": 1,
        "y = −(2/3)x − 5": 1,
        "h(w) = (3w + 42)/12": 1,
        "x/3 + x/4 = 7": 2,
        "(x + 4)/3 = (x − 2)/5": 2,
        "Choice B (5): this is the value of x/5, not 11/x.": 2,
        "Slope: (17 − 9) / (3 − 1) = 4.": 1,
        "1,500/1,000": 1,
        "0.04x + 0.07y = 295 and 3,000 tickets": 0,
    }
    me._MATH[0] = True
    try:
        for text, n in cases.items():
            out = me.esc(text)
            assert out.count('class="frac"') == n, text
            assert plain(out).replace(" / ", "/") == text.replace(" / ", "/"), (text, plain(out))
        assert "5," not in re.findall(r'padding:0 \.18em">([^<]*)<', me.esc("value of x/5, not"))
    finally:
        me._MATH[0] = False
    assert me.esc("and/or 1/2") == "and/or 1/2"  # reading questions are left alone


def test_answers_cannot_be_guessed_from_the_choices_alone():
    """Sebastian (Nov 25): no Reading-style giveaway in math. A student who never reads the question and uses a
    pattern on the four choices (the one most like the others, the odd one out, the longest or shortest, the one
    the others are a slip away from, the middle, smallest, or largest number) must stay near chance (25%) in every
    question type; 55% allows for sampling noise around the generator's 50% ceiling."""
    import collections

    def parts(s):
        return collections.Counter(re.findall(r"\d+(?:[.,/]\d+)*|[A-Za-z]+|[=<>≤≥+()\-]", s.replace(mm.MINUS, "-")))

    def num_or_none(s):
        try:
            return value(s)
        except (ValueError, ZeroDivisionError):
            return None

    def picks(ch):
        out = {}
        P = {k: parts(v) for k, v in ch.items()}
        sh = {k: sum(sum((P[k] & P[o]).values()) for o in ch if o != k) for k in ch}
        L = {k: len(v) for k, v in ch.items()}
        out["most alike"] = [k for k in ch if sh[k] == max(sh.values())]
        out["odd one out"] = [k for k in ch if sh[k] == min(sh.values())]
        out["longest"] = [k for k in ch if L[k] == max(L.values())]
        out["shortest"] = [k for k in ch if L[k] == min(L.values())]
        V = {k: num_or_none(v) for k, v in ch.items()}
        if all(v is not None for v in V.values()):

            def slip(a, b):
                near = {-a, a + 1, a - 1, a * 2, a / 2, a * 10, a / 10} | ({1 / a, -1 / a} if a else set())
                return b in near

            r = {k: sum(slip(V[k], V[o]) + slip(V[o], V[k]) for o in V if o != k) for k in V}
            mean = sum(V.values()) / 4
            out["slip hub"] = [k for k in V if r[k] == max(r.values())]
            out["middle"] = [k for k in V if abs(V[k] - mean) == min(abs(v - mean) for v in V.values())]
            out["smallest"] = [k for k in V if V[k] == min(V.values())]
            out["largest"] = [k for k in V if V[k] == max(V.values())]
        return out

    for name, fam in mm.FAMILIES.items():
        hits = collections.defaultdict(list)
        for seed in range(500, 560):
            rng = random.Random(seed)
            for level in mm.FAMILY_LEVELS[name]:
                item = fam(rng, level, "mcq")
                item.build_choices(rng, {})
                if item.fixed_choices:
                    continue
                for rule, chosen in picks(item.choices).items():
                    hits[rule].append((item.key in chosen) / len(chosen))
        for rule, h in hits.items():
            assert sum(h) / len(h) <= 0.55, (name, rule, round(sum(h) / len(h), 2))


# ------------------------------------------------- closing out Algebra: inequalities to solve, graphs as choices, ...


def solution_set(test, grid=None):
    grid = grid or [Fraction(n, 2) for n in range(-60, 61)]
    return [bool(test(x)) for x in grid]


def or_chain(text: str):
    """'x < −4 or x > 0' or a plain chain like '−4 < x ≤ 0'."""
    parts = [chain(p.strip()) for p in text.split(" or ")]
    return lambda x: any(p(x) for p in parts)


def test_solving_inequalities_and_systems_of_inequalities():
    for item in gen("ineq_solve"):
        t = item.truth
        if item.level == "hard":

            def truth(x, t=t):
                v = t["a"] * x + t["b"]
                return holds(t["r1"], t["L"], v) and holds(t["r2"], v, t["R"])
        else:

            def truth(x, t=t):
                return holds(t["rel"], t["a"] * x + t["b"], t["c"])

        want = solution_set(truth)
        for k, text in item.choices.items():
            assert (solution_set(chain(text)) == want) == (k == item.key), (item.stem, text)
    for item in gen("ineq_system_setup"):
        t = item.truth
        for k, text in item.choices.items():
            e1, e2 = text.split("\n")
            l1, r1, rhs1 = split_rel(e1)
            l2, r2, rhs2 = split_rel(e2)
            ok = (
                evaluate(l1, x=1, y=0) == evaluate(l1, x=0, y=1) == 1
                and r1 == t["crel"]
                and value(rhs1) == t["c"]
                and evaluate(l2, x=1, y=0) == t["a"]
                and evaluate(l2, x=0, y=1) == t["b"]
                and r2 == t["mrel"]
                and value(rhs2) == t["M"]
            )
            assert ok == (k == item.key), (item.stem, text)


def test_graph_choices_only_the_key_shows_the_right_line_or_region():
    for item in gen("graph_choices"):
        t = item.truth
        want = (t["m"], t["b"]) if item.level != "hard" else (t["side"], t["dashed"])
        for k, svg in item.choices.items():
            assert svg.startswith("<svg") and svg.count('stroke="#cfcfcf"') >= 26
            assert (t["cells"][svg] == want) == (k == item.key)
            if item.level == "hard":
                assert "<polygon" in svg and (("stroke-dasharray" in svg) == t["cells"][svg][1])


def test_function_notation_abs_value_and_points():
    for item in gen("function_notation"):
        t = item.truth
        if t["kind"] == "shift":
            for k, expr in item.choices.items():
                ok = all(evaluate(expr, x=x) == t["a"] * (x + t["c"]) + t["b"] for x in (0, 1, 2))
                assert ok == (k == item.key), (item.stem, expr)
        elif t["kind"] == "scale":
            for k, eq in item.choices.items():
                ok = all(evaluate(rhs(eq), x=x) == t["k"] * (t["a"] * x + t["b"]) + t["d"] for x in (0, 1, 2))
                assert ok == (k == item.key), (item.stem, eq)
        else:
            (a, b), (a2, b2), n = t["f"], t["g"], t["n"]
            assert value(item.choices[item.key]) == a * (a2 * n + b2) + b
    for item in gen("abs_value"):
        t = item.truth
        if item.level == "easy":
            sols = [x for x in range(-60, 61) if abs(x - t["h"]) == t["r"]]
            assert value(item.choices[item.key]) == sum(sols) and len(sols) == 2
        elif item.level == "medium":
            sols = [Fraction(n, t["a"]) for n in range(-300, 301) if abs(n + t["c"]) == t["r"]]
            assert value(item.choices[item.key]) == max(sols)
        else:
            want = solution_set(lambda x, t=t: holds(t["rel"], abs(x - t["h"]), t["r"]))
            for k, text in item.choices.items():
                assert (solution_set(or_chain(text)) == want) == (k == item.key), (item.stem, text)
    for item in gen("point_meaning"):
        t = item.truth
        key = item.choices[item.key]
        assert "every" not in key and key.index(mm.num(t["y0"])) >= 0
        assert sum(mm.num(t["y0"]) in c and str(t["x0"]) in c for c in item.choices.values()) == 4
