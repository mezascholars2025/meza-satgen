"""Offline tests for the Meza quality gate (no API calls)."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEZA_QUALITY", "on")

import meza_satgen  # noqa: E402
from meza_quality import (  # noqa: E402
    advanced_in,
    choice_mix_problems,
    features,
    lint,
    probe_votes,
    rhythm_check,
    trap_critic_problems,
    trap_label_problems,
    vocab_used,
)
from satgen import pipeline  # noqa: E402

PASSAGE = (
    "A team of geologists studied pumice rafts, floating masses of light volcanic rock. They tracked one raft "
    "for several months and found that it drifted thousands of kilometers. Small organisms attached to the "
    "stones and some survived the trip, which led the team to conclude that the rafts can carry living "
    "species to distant shores."
)
STEM = "Which choice best states the main idea of the text?"
GOOD = {
    "A": "Pumice rafts tend to break apart within a few weeks, long before they reach other shores.",
    "B": "Floating volcanic rock can carry small living organisms across great distances of ocean.",
    "C": "Researchers can now predict exactly where any given pumice raft will drift and stop.",
    "D": "Organisms that attach to drifting pumice usually die before the rock reaches land.",
}
TELL = {
    "A": "Rafts sink.",
    "B": "Floating pumice from undersea eruptions can carry small organisms across oceans to distant new shores.",
    "C": "Satellites are the best tool for tracking.",
    "D": "Only barnacles survive.",
}
EXPL = (
    "[Domain: Information and Ideas] [Skill: central_ideas] [Difficulty: Easy] [Subject: Earth] The choice beginning..."
)


def test_lint_catches_tells():
    r = lint("central_ideas_easy", PASSAGE, STEM, TELL, "B", EXPL)
    joined = " ".join(r.fails)
    assert "longest choice" in joined
    assert "Absolute words" in joined


def test_lint_passes_parallel_item():
    r = lint("central_ideas_easy", PASSAGE, STEM, GOOD, "B", EXPL)
    assert r.ok, r.fails


def test_lint_rules():
    assert not lint("central_ideas", PASSAGE, "What is the main idea?", GOOD, "B", EXPL).ok  # wrong stem
    assert not lint("central_ideas", PASSAGE, STEM, GOOD, "B", "Choice B is right because...").ok  # letter ref
    assert not lint("inferences", PASSAGE, "Which choice most logically completes the text?", GOOD, "B", EXPL).ok
    long = PASSAGE + " " + PASSAGE + " " + PASSAGE
    assert any("standardized words" in f for f in lint("central_ideas", long, STEM, GOOD, "B", EXPL).fails)
    bees = PASSAGE.replace("pumice rafts", "bees")
    assert any("Overused" in f for f in lint("central_ideas", bees, STEM, GOOD, "B", EXPL).fails)
    sec_stem = "Which choice completes the text so that it conforms to the conventions of Standard English?"
    p = (
        "Designed by a little-known local architect, the city's natural history museum opened its doors in 1902 ______ "
        "it quickly became the most visited public building in the region."
    )
    ok = {"A": "1902, and", "B": "1902 and", "C": "1902, and,", "D": "1902; and"}
    assert lint("boundaries_independent_clauses", p, sec_stem, ok, "A", EXPL).ok
    bad = dict(ok, D="1902, so")
    assert not lint("boundaries_independent_clauses", p, sec_stem, bad, "A", EXPL).ok
    tstem = "Which choice completes the text with the most logical transition?"
    tp = (
        "In the first trial, the chemists heated the mixture for an hour but failed to produce any reaction. ______, "
        "the second trial, which used a finer powder, produced one within minutes."
    )
    assert not lint(
        "transitions", tp, tstem, {"A": "However", "B": "Nevertheless", "C": "For example", "D": "Therefore"}, "A", EXPL
    ).ok
    assert lint(
        "transitions", tp, tstem, {"A": "However", "B": "Similarly", "C": "For example", "D": "Therefore"}, "A", EXPL
    ).ok


def test_probe_reshuffles_and_scores():
    calls = []

    def always_longest(prompt, system):
        lines = [ln for ln in prompt.splitlines() if ln[:2] in ("A)", "B)", "C)", "D)")]
        best = max(lines, key=len)[0]
        calls.append(best)
        return json.dumps({"answer": best})

    assert sum(probe_votes(always_longest, STEM, TELL, "B", runs=3)) == 3
    assert len(set(calls)) >= 1


class FakeWriter:
    """First draft has tells; the rewrite (prompt contains the repair block) is clean."""

    model = "fake-writer"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        self.prompts.append(prompt)
        choices = GOOD if "FAILED QUALITY CHECKS" in prompt else TELL
        out = {"passage": PASSAGE, "stem": STEM, "choices": choices, "answer": "B", "explanation": EXPL}
        return json.dumps(out), 100, 100


class FakeSolver:
    model = "fake-solver"

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        return json.dumps({"answer": "A"}), 10, 5  # never finds the key without the passage


def test_repair_loop_rewrites_flawed_draft(tmp_path, monkeypatch):
    log = tmp_path / "quality.jsonl"
    monkeypatch.setattr(meza_satgen, "LOG_PATH", str(log))
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    p.plan = [5]
    q = p.generate(0, "central_ideas_easy")
    assert q.choices == GOOD
    assert len(writer.prompts) == 2 and "longest choice" in writer.prompts[1]
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    assert [e["passed"] for e in entries] == [False, True]
    assert entries[1]["probe_runs"] == 3


def test_gate_gives_up_after_max_repairs(monkeypatch):
    class StubbornWriter(FakeWriter):
        def complete(self, prompt, system="", temperature=0.7, json_mode=False):
            out = {"passage": PASSAGE, "stem": STEM, "choices": TELL, "answer": "B", "explanation": EXPL}
            return json.dumps(out), 100, 100

    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    p = meza_satgen.MezaPipeline(StubbornWriter(), FakeSolver(), FakeSolver(), context="rules")
    p.plan = [5]
    rec = p.process(0, "central_ideas_easy")
    assert rec.status == "generation_failed" and "quality gate" in rec.error


def test_install_swaps_pipeline_only_when_asked():
    original = pipeline.Pipeline
    assert original is not meza_satgen.MezaPipeline  # importing meza_satgen must not change satgen itself
    try:
        meza_satgen.install()
        assert pipeline.Pipeline is meza_satgen.MezaPipeline
    finally:
        pipeline.Pipeline = original


def test_wic_colon_after_blank_only_for_restatement():
    stem = "Which choice completes the text with the most logical and precise word or phrase?"
    p = (
        "Because the conductor spoke so gently, the musicians had expected her to be lenient. Instead she proved "
        "______: she stopped rehearsals to fix a single uneven phrase and had players repeat passages."
    )
    ch = {"A": "exacting", "B": "temperamental", "C": "ambivalent", "D": "unorthodox"}
    assert not lint("words_in_context_completion", p, stem, ch, "A", EXPL, clue="contrast").ok
    assert lint("words_in_context_completion", p, stem, ch, "A", EXPL, clue="restatement").ok


def test_clue_plan_is_weighted_and_rare_restatement():
    plan = meza_satgen.clue_plan(140, seed=7)
    names = [meza_satgen.WIC_CLUES[i][0] for i in plan]
    assert names.count("restatement") == 10 and names.count("contrast") == 30 and names.count("exposition") == 30


def test_wic_prompt_gets_clue_line(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    p.plan, p.clues = [5], [0]
    try:
        p.generate(0, "words_in_context_completion_hard")
    except ValueError:
        pass
    assert "REQUIRED CLUE STRUCTURE" in writer.prompts[0] and "contrast or concession" in writer.prompts[0]


ROUND2 = [
    "Most critics were ______ when the painter first exhibited her abstract canvases, calling them careless and "
    "unfinished. Within a decade, however, the same works had entered the collections of several major museums.",
    "Historians have long cited a single merchant's diary as evidence that one port handled more trade than any of "
    "its rivals. Yet the diary's author visited the port only once, so many scholars now regard the claim as ______.",
    "The assumption that early printed maps simply recorded what explorers had seen is, in one historian's view, "
    "______. Mapmakers routinely filled unvisited coastlines with features copied from older charts.",
    "The composer's letters, which record revisions spanning a decade, leave that account ______ the documentary "
    "record, though they do not explain why he later encouraged the legend himself.",
]
PLAIN = (
    "Glassblowers in the Venetian island of Murano developed a clear glass called cristallo by adding "
    "purified plant ash to the melt, a step that ______ the greenish tint common in earlier glass."
)


def test_features_flag_round2_twists():
    assert all(features("words_in_context_completion", p)["twist"] for p in ROUND2)
    f = features("words_in_context_completion", PLAIN, "removed", "exposition")
    assert not f["twist"] and f["blank"] == "first" and f["opening"].startswith("Glassblowers")


def test_rhythm_rules():
    twist = features("words_in_context_completion", ROUND2[0], "dismissive", "evaluation")
    assert any("must NOT use a contrast" in x for x in rhythm_check(twist, []))
    recent = [features("central_ideas", p) for p in ROUND2[:2]]
    third = features("central_ideas", ROUND2[3])
    assert any("last two questions" in x for x in rhythm_check(third, recent))
    plain = features("central_ideas", PLAIN)
    assert rhythm_check(plain, recent) == []
    same_open = features("central_ideas", "Most critics were wrong about the paintings, which are now admired widely.")
    assert any("opens the same way" in x for x in rhythm_check(same_open, recent))
    again = features("words_in_context_completion", PLAIN, "removed", "exposition")
    prev = [features("words_in_context_completion", PLAIN.replace("Glassblowers", "Weavers"), "removed", "exposition")]
    assert any("Use a different word" in x for x in rhythm_check(again, prev))


def test_recent_questions_shown_to_writer(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    p.plan, p.clues = [5, 5], [0, 0]
    p.generate(0, "central_ideas_easy")
    n = len(writer.prompts)
    try:
        p.generate(1, "central_ideas_easy")  # same opening as the first, so it must be told and fail/repair
    except ValueError:
        pass
    assert "YOUR MOST RECENT QUESTIONS OF THIS TYPE" in writer.prompts[n]
    assert "A team of geologists" in writer.prompts[n]


def test_vocab_forms():
    t = "The storm abated; her cogency was clear, though the data belied the claim and the result was equivocal."
    assert vocab_used(t, ["abate", "cogent", "belie", "equivocal", "laud"]) == ["abate", "cogent", "belie", "equivocal"]
    assert vocab_used("They believed the belief.", ["belie"]) == []
    assert vocab_used(
        "The scrutiny was assiduously applied and the marring was laudable.", ["assiduous", "mar", "laud", "scrutiny"]
    ) == ["assiduous", "mar", "laud", "scrutiny"]


def test_vocab_bank_deals_without_repeats():
    bank = meza_satgen.VocabBank(meza_satgen.VOCAB_PATH, seed=1)
    words = bank.deal("advanced", 200)
    assert len(words) == len(set(words)) == 200
    plan = bank.choice_plan("hard")
    assert plan["mix"] == {"advanced": 2, "mid": 1, "common": 1} and plan["key_tier"] in ("advanced", "mid", "common")
    w, need, cap = bank.passage_words("hard")
    assert len(w) == 3 and need == 1 and cap == 2
    assert bank.passage_words("easy") == ([], 0, 1)


def test_hard_item_must_use_bank_words(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    p.plan = [5]
    try:
        p.generate(0, "central_ideas_hard")
    except ValueError as exc:
        assert "quality gate" in str(exc)
    assert "VOCABULARY FOR THIS PASSAGE (hard)" in writer.prompts[0]
    assert any("Use at least 1 of these words" in pr for pr in writer.prompts[1:])


def test_choice_mix_and_passage_cap():
    mix = {"advanced": 2, "mid": 1, "common": 1}
    all_hard = {"A": "vestigial", "B": "amorphous", "C": "trifling", "D": "florid"}
    assert any("exactly 2" in x for x in choice_mix_problems(all_hard, "A", mix, "advanced"))
    balanced = {"A": "vestigial", "B": "amorphous", "C": "dormant", "D": "accessible"}  # all four are bank words
    assert choice_mix_problems(balanced, "A", mix, "advanced") == []
    assert any("must be the mid word" in x for x in choice_mix_problems(balanced, "A", mix, "mid"))
    assert choice_mix_problems({"A": "at odds with", "B": "b", "C": "c", "D": "d"}, "A", mix, "mid") == []
    heavy = "The trepidation, juxtaposition, and acrimonious mood left her unable to extricate herself."
    assert len(advanced_in(heavy)) >= 3


def test_not_only_but_is_not_a_twist():
    p = "Many winners not only spent their prizes within a few years but ______ them on purchases they regretted."
    assert not features("words_in_context_completion", p)["twist"]
    assert features("words_in_context_completion", p.replace("not only ", ""))["twist"]


MEANING_STEM_TXT = 'As used in the text, what does the word "sound" most nearly mean?'
MEANING_P = (
    "Engineers inspecting the century-old bridge reported that its stone piers remained <u>sound</u> after "
    "decades of floods, and the city approved only minor repairs to the roadway above them."
)
MEANING_CH = {"A": "structurally solid", "B": "noisy", "C": "sensible", "D": "deep"}


def test_meaning_lint():
    assert lint("words_in_context_meaning", MEANING_P, MEANING_STEM_TXT, MEANING_CH, "A", EXPL).ok
    no_u = MEANING_P.replace("<u>", "").replace("</u>", "")
    assert any(
        "Underline" in x for x in lint("words_in_context_meaning", no_u, MEANING_STEM_TXT, MEANING_CH, "A", EXPL).fails
    )
    bad_stem = 'What does "sound" mean in the text?'
    assert any(
        "exact stem" in x for x in lint("words_in_context_meaning", MEANING_P, bad_stem, MEANING_CH, "A", EXPL).fails
    )
    phrase_stem = MEANING_STEM_TXT.replace('word "sound"', 'phrase "remained sound"')
    assert not lint("words_in_context_meaning", MEANING_P, phrase_stem, MEANING_CH, "A", EXPL).ok  # mismatch
    long = dict(MEANING_CH, B="making a loud noise that carries far")
    assert any(
        "short" in x for x in lint("words_in_context_meaning", MEANING_P, MEANING_STEM_TXT, long, "A", EXPL).fails
    )


def test_meaning_features_and_repeat():
    f = features("words_in_context_meaning", MEANING_P, "", "exposition")
    assert f["key"] == "sound" and f["blank"] == "first" and f["marker"] == "underlined word"
    assert any("tested word" in x for x in rhythm_check(f, [dict(f, opening="Other words here")]))


def test_meaning_prompt_gets_clue_and_targets(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    restatement = [i for i, c in enumerate(meza_satgen.WIC_CLUES) if c[0] == "restatement"][0]
    p.plan, p.clues = [5], [restatement]
    try:
        p.generate(0, "words_in_context_meaning_hard")
    except ValueError:
        pass
    first = writer.prompts[0]
    assert "TESTED WORD FOR THIS ITEM" in first and "REQUIRED CLUE STRUCTURE" in first
    assert "plain explanation" in first.split("REQUIRED CLUE STRUCTURE")[1]


FN_STEM = "Which choice best describes the function of the underlined {} in the text as a whole?"
FN_CH = {
    "A": "It introduces the problem that the rest of the text addresses.",
    "B": "It presents the conclusion that the biologists finally reached.",
    "C": "It provides an example that supports the claim in the first sentence.",
    "D": "It describes a method that the biologists considered and rejected.",
}
FN_P = (
    "Many birds migrate thousands of kilometers each year and return to the same nesting sites. <u>How do young "
    "birds on their first journey find their way?</u> Biologists tracking juvenile warblers found that the birds "
    "rely partly on the earth's magnetic field and partly on stars visible at night."
)


def test_function_item_kinds():
    assert lint("text_structure_function", FN_P, FN_STEM.format("question"), FN_CH, "A", EXPL, want_kind="question").ok
    assert any(
        "must say" in x
        for x in lint("text_structure_function", FN_P, FN_STEM.format("sentence"), FN_CH, "A", EXPL).fails
    )
    assert any(
        "must underline a sentence" in x
        for x in lint(
            "text_structure_function", FN_P, FN_STEM.format("question"), FN_CH, "A", EXPL, want_kind="sentence"
        ).fails
    )
    sent = FN_P.replace(
        "<u>How do young birds on their first journey find their way?</u>",
        "<u>Young birds on their first journey still find their way.</u>",
    )
    assert lint("text_structure_function", sent, FN_STEM.format("sentence"), FN_CH, "A", EXPL).ok
    assert any(
        "not a question" in x
        for x in lint("text_structure_function", sent, FN_STEM.format("question"), FN_CH, "A", EXPL).fails
    )
    portion = FN_P.replace(
        "<u>How do young birds on their first journey find their way?</u>",
        "How do young birds on <u>their first journey</u> find their way?",
    )
    assert lint("text_structure_function", portion, FN_STEM.format("portion"), FN_CH, "A", EXPL).ok


def test_function_prompt_assigns_kind(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    seen = set()
    for i in range(60):
        writer = FakeWriter()
        p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
        p.plan = [5] * 60
        try:
            p.generate(i, "text_structure_function_medium")
        except ValueError:
            pass
        line = [x for x in writer.prompts[0].splitlines() if x.startswith("REQUIRED UNDERLINED PORTION")]
        seen.add(
            "question"
            if "question that the text itself poses" in line[0]
            else "portion"
            if "phrase or a quotation" in line[0]
            else "sentence"
        )
    assert seen == {"sentence", "question", "portion"}


def test_tsp_verbs_and_length_bands():
    stem = "Which choice best states the main purpose of the text?"
    p = (
        "In the forests of Central and South America, leafcutter ants carry pieces of leaves back to their nests, "
        "where workers chew the leaves into a paste and grow a fungus that becomes the colony's main food."
    )
    short = {
        "A": "To describe how leafcutter ants grow their own food",
        "B": "To argue that leafcutter ants harm forests badly",
        "C": "To explain why some ant workers are smaller",
        "D": "To compare leafcutter ants with other insects that farm",
    }
    assert lint("text_structure_purpose", p, stem, short, "A", EXPL, choice_band=(9, 12)).fails == []
    assert any(
        "15-22 words" in x
        for x in lint("text_structure_purpose", p, stem, short, "A", EXPL, choice_band=(15, 22)).fails
    )
    it_open = {k: v.replace("To ", "It ", 1) for k, v in short.items()}
    assert any('open with "To' in x for x in lint("text_structure_purpose", p, stem, it_open, "A", EXPL).fails)
    weak = {
        "A": "To argue that ants grow their own food",
        "B": "To criticize forests for harboring ants",
        "C": "To refute a claim about ant workers",
        "D": "To describe ants and other insects",
    }
    assert any("preferred list" in x for x in lint("text_structure_purpose", p, stem, weak, "A", EXPL).fails)
    from meza_quality import lead_verb

    assert lead_verb("It presents a finding") == "present" and lead_verb("To emphasize the point") == "emphasize"
    assert lead_verb("It summarizes the view") == "summarize"


def test_tsp_prompt_band_and_key_verb_rotation(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    writer = FakeWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), FakeSolver(), context="rules")
    p.plan = [5, 5]
    p._recent = {
        "text_structure_overall": [
            {
                "key": "describe",
                "twist": False,
                "opening": "x",
                "blank": None,
                "sentences": 3,
                "clue": "",
                "markers": [],
                "type": "text_structure_overall",
            }
        ]
    }
    try:
        p.generate(0, "text_structure_overall_medium")
    except ValueError:
        pass
    first = writer.prompts[0]
    assert "every choice is 15-22 words" in first and "must NOT open with describe" in first


POEM = (
    'The following text is from Harriet Lowell Ames\'s 1874 poem "To the Clock in the Hall." The speaker, a '
    "widow who has just sold the house where she raised her children, addresses the clock she is leaving.\n\n"
    "Thou keeper of our hours, unasked, unthanked,\nthrough vows and quarrels and the indolent years,\n"
    "what tally wilt thou keep when we are scattered?\nWill strangers' boots upon the stair be counted\n"
    "as once our children's were, with equal care,\nor shalt thou, idle in the hall, remember\n"
    "the hours we gave thee, and refuse the rest?"
)


def test_literary_intro_must_be_specific():
    stem = "Which choice best states the main purpose of the text?"
    ch = {
        "A": "To convey doubt about whether the clock will remember the family",
        "B": "To praise the old clock for keeping such accurate time for many years",
        "C": "To express regret that the family's home was sold to strangers",
        "D": "To describe the sound that the clock makes at the start of each hour",
    }
    subj = "LITERATURE: POETRY OR DRAMA - an excerpt"
    assert lint("text_structure_purpose", POEM, stem, ch, "A", EXPL + " [Invented work]", subject=subj).fails == []
    vague = POEM.replace('Harriet Lowell Ames\'s 1874 poem "To the Clock in the Hall."', "a poem.")
    assert any(
        "specific" in x
        for x in lint("text_structure_purpose", vague, stem, ch, "A", EXPL + " [Invented work]", subject=subj).fails
    )
    no_ctx = POEM.split(" The speaker,")[0] + "\n\n" + POEM.split("\n\n")[1]
    assert any(
        "context" in x
        for x in lint("text_structure_purpose", no_ctx, stem, ch, "A", EXPL + " [Invented work]", subject=subj).fails
    )
    assert any(
        "Invented work" in x for x in lint("text_structure_purpose", POEM, stem, ch, "A", EXPL, subject=subj).fails
    )


def test_poetry_gets_more_vocabulary():
    bank = meza_satgen.VocabBank(meza_satgen.VOCAB_PATH, seed=3)
    words, need, cap = bank.passage_words("hard", poetry=True)
    assert len(words) == 4 and need == 2 and cap == 3


def test_trap_labels():
    assert trap_label_problems({"B": "too_narrow", "C": "wrong_action", "D": "wrong_target"}, "A", "hard") == []
    assert trap_label_problems(None, "A", "easy")
    assert trap_label_problems({"B": "too_narrow", "C": "made_up", "D": "wrong_target"}, "A", "easy")
    assert trap_label_problems({"B": "too_narrow", "C": "too_narrow", "D": "too_narrow"}, "A", "hard")
    assert trap_label_problems({"B": "too_narrow", "C": "too_narrow", "D": "too_narrow"}, "A", "easy") == []


def test_trap_critic_rules():
    off = {
        "B": {"on_topic": False, "tempting": 1, "trap": "none", "why": "plants never mentioned"},
        "C": {"on_topic": True, "tempting": 3, "trap": "too_narrow"},
        "D": {"on_topic": True, "tempting": 4, "trap": "wrong_action"},
    }
    assert any("never mentions" in x for x in trap_critic_problems(off, "A", "easy"))
    one_weak = {
        "B": {"on_topic": True, "tempting": 2, "trap": "too_narrow"},
        "C": {"on_topic": True, "tempting": 3, "trap": "wrong_target"},
        "D": {"on_topic": True, "tempting": 4, "trap": "wrong_action"},
    }
    assert trap_critic_problems(one_weak, "A", "easy") == []
    assert any("too easy" in x for x in trap_critic_problems(one_weak, "A", "hard"))
    assert trap_critic_problems({}, "A", "hard") == []  # critic unavailable: don't block


class TrapWriter(FakeWriter):
    """Draft 1 has off-topic distractors; the rewrite has near misses."""

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        self.prompts.append(prompt)
        fixed = "FAILED QUALITY CHECKS" in prompt
        out = {
            "passage": PASSAGE,
            "stem": "Which choice best states the main purpose of the text?",
            "choices": {
                "A": "To describe how pumice rafts can carry living organisms to distant shores",
                "B": "To argue that pumice rafts are the main way species reach new islands"
                if fixed
                else "To argue that volcanoes erupt more often than scientists once believed",
                "C": "To explain how geologists used satellite images to track one raft"
                if fixed
                else "To compare pumice rafts with the floating plastic found in oceans",
                "D": "To emphasize that most organisms died during the raft's long journey",
            },
            "answer": "A",
            "explanation": EXPL,
            "traps": {"B": "too_broad", "C": "too_narrow", "D": "misread_stance"},
        }
        return json.dumps(out), 100, 100


class Critic(FakeSolver):
    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        if "veteran Digital SAT item reviewer" not in system:
            return super().complete(prompt, system, temperature, json_mode)
        bad = "plastic" in prompt
        return (
            json.dumps(
                {
                    "B": {"on_topic": not bad, "tempting": 2 if bad else 4, "trap": "too_broad"},
                    "C": {"on_topic": not bad, "tempting": 1 if bad else 3, "trap": "none" if bad else "too_narrow"},
                    "D": {"on_topic": True, "tempting": 3, "trap": "misread_stance"},
                }
            ),
            50,
            50,
        )


def test_trap_critic_sends_off_topic_draft_back(monkeypatch):
    monkeypatch.setattr(meza_satgen, "LOG_PATH", "")
    monkeypatch.setattr(meza_satgen, "VOCAB_RULES", {"hard": (0, 0, 9, ())})  # isolate the critic from other checks
    monkeypatch.setattr(meza_satgen, "TSP_BANDS", {"short": (5, 30), "long": (5, 30)})
    writer = TrapWriter()
    p = meza_satgen.MezaPipeline(writer, FakeSolver(), Critic(), context="rules")
    p.plan = [5]
    q = p.generate(0, "text_structure_purpose_hard")
    assert "WRONG ANSWERS FOR THIS ITEM" in writer.prompts[0]
    assert "never mentions" in writer.prompts[1]  # draft 1 was sent back by the critic
    assert "plastic" not in q.choices["C"]  # the near-miss rewrite is what ships


def test_critic_requires_two_accurate_near_misses():
    v = {
        "B": {
            "accurate_content": False,
            "on_topic": True,
            "tempting": 3,
            "trap": "wrong_target",
            "why": "text never compares with other animals",
        },
        "C": {"accurate_content": False, "on_topic": True, "tempting": 3, "trap": "wrong_order"},
        "D": {"accurate_content": True, "on_topic": True, "tempting": 4, "trap": "too_narrow"},
    }
    assert any("misstate" in x for x in trap_critic_problems(v, "A", "medium"))
    v["C"]["accurate_content"] = True
    assert trap_critic_problems(v, "A", "medium") == []


def test_new_bank_and_meaning_targets():
    import json

    data = json.loads(meza_satgen.VOCAB_PATH.read_text())
    assert meza_satgen.VOCAB_PATH.name == "meza_vocab_bank.json" and len(data["words"]) > 1900
    assert len({w["word"] for w in data["words"]}) == len(data["words"])  # no duplicates
    bank = meza_satgen.VocabBank(meza_satgen.VOCAB_PATH, seed=5)
    multi = {w["word"] for w in data["words"] if w.get("multi_meaning")}
    t = bank.targets(6)
    assert len(t) == 6 and sum(w in multi for w in t) >= 3


def test_literary_intro_does_not_count_toward_length():
    stem = "Which choice completes the text with the most logical and precise word or phrase?"
    p = (
        "The following text is from Edmund Calloway Brandt's 1889 novel \"The Salt-Merchant's Ledger.\" Marian, a "
        "young guest at her aunt's dinner party, has just spilled a glass of wine across the tablecloth.\n\n"
        "Marian's stammering apology served only to ______ her embarrassment, for every halting word drew more eyes "
        "toward her end of the table."
    )
    ch = {"A": "compound", "B": "execute", "C": "confer", "D": "mock"}
    subj = "LITERATURE: PROSE FICTION - an excerpt"
    fails = lint("words_in_context_completion", p, stem, ch, "A", EXPL + " [Invented work]", subject=subj).fails
    assert any("standardized words" in f or "excerpt itself" in f for f in fails)


def test_focus_critic_rules():
    from meza_quality import focus_problems

    built_around = {"main_idea": "a mentor's sound judgment", "word_is_main_idea": True, "hint_sentences": 3}
    assert focus_problems(built_around, "detail")
    assert focus_problems(built_around, "central") == []
    detail = {"main_idea": "how railway routes were chosen", "word_is_main_idea": False, "hint_sentences": 1}
    assert focus_problems(detail, "detail") == []
    assert focus_problems({}, "detail") == []
