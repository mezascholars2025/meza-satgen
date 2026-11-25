"""Turn a satgen run into files people can read.

    python meza_export.py runs/<folder>

Writes, next to questions.json:
  review.html    - every question as a readable card, answers hidden until clicked
  questions.csv  - one row per question, for Google Sheets or Excel
"""
import csv
import html
import json
import re
import sys
from pathlib import Path

NAVY, GOLD = "#3F60A1", "#E9A921"
TAG = re.compile(r"\[(Domain|Skill|Difficulty|Subject):\s*([^\]]+)\]")


def tags(explanation: str) -> dict:
    found = {k.lower(): v.strip() for k, v in TAG.findall(explanation)}
    body = TAG.sub("", explanation).strip()
    return {"domain": found.get("domain", ""), "skill": found.get("skill", ""),
            "difficulty": found.get("difficulty", ""), "subject": found.get("subject", ""), "body": body}


def esc(text: str) -> str:
    out = html.escape(text)
    return out.replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")


def render_passage(text: str) -> str:
    parts, table, bullets = [], [], []

    def flush():
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|?[\s:\-|]+\|?", r)]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            head = "".join(f"<th>{esc(c)}</th>" for c in cells[0])
            body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            parts.append(f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')
            table.clear()
        if bullets:
            parts.append("<ul>" + "".join(f"<li>{esc(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            if bullets: flush()
            table.append(s)
        elif s.startswith("- "):
            if table: flush()
            bullets.append(s[2:])
        else:
            flush()
            if s:
                cls = ' class="title"' if re.match(r"^(Text [12]\b|.*\btable\b)", s) and len(s) < 90 else ""
                parts.append(f"<p{cls}>{esc(s)}</p>")
    flush()
    return "\n".join(parts)



def slide_block(q: dict, with_answer: bool) -> str:
    """Inline-styled HTML that keeps bold, underline and tables when pasted into PowerPoint."""
    font = "font-family:Arial,Helvetica,sans-serif;"
    passage = render_passage(q.get("passage", ""))
    passage = passage.replace('<table class="data">',
        f'<table style="border-collapse:collapse;{font}font-size:16pt;" border="1" cellpadding="6">')
    passage = passage.replace("<p", f'<p style="{font}font-size:18pt;margin:0 0 8pt;"', ).replace('<p style="' + font + 'font-size:18pt;margin:0 0 8pt;" class="title">',
        f'<p style="{font}font-size:18pt;margin:0 0 8pt;font-weight:bold;">')
    passage = passage.replace("<li>", f'<li style="{font}font-size:18pt;">')
    choices = "".join(f'<p style="{font}font-size:18pt;margin:0 0 6pt;"><b>{k})</b> {esc(v)}</p>'
                      for k, v in q.get("choices", {}).items())
    out = (f'<div>{passage}<p style="{font}font-size:18pt;font-weight:bold;margin:10pt 0 8pt;">'
           f'{esc(q.get("stem",""))}</p>{choices}')
    if with_answer:
        body = tags(q.get("explanation", ""))["body"]
        expl = "".join(f'<p style="{font}font-size:14pt;margin:0 0 6pt;">{esc(x)}</p>' for x in body.splitlines() if x.strip())
        out += f'<p style="{font}font-size:18pt;margin:12pt 0 6pt;"><b>Answer: {esc(q.get("answer",""))}</b></p>{expl}'
    return out + "</div>"


def review_flags(run: Path) -> dict:
    """Questions the quality gate let through with a judgment-call problem still open, keyed by fingerprint."""
    flags = {}
    log = run / "quality.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("flagged") and entry.get("fingerprint"):
                flags[entry["fingerprint"]] = entry["flagged"]
    return flags


def fingerprint(q: dict) -> str:
    text = re.sub(r"</?u>", "", q.get("passage", "")) + " " + q.get("stem", "")
    return " ".join(text.lower().split())[:160]


def build(run: Path) -> None:
    qs = json.loads((run / "questions.json").read_text(encoding="utf-8"))
    flags = review_flags(run)
    report = {}
    if (run / "report.json").exists():
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    cost = report.get("cost", {}).get("total_usd")

    # CSV
    with open(run / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["#", "Domain", "Skill", "Difficulty", "Subject", "Type", "Passage", "Question",
                    "A", "B", "C", "D", "Answer", "Explanation", "Review status", "Reviewer notes"])
        for i, q in enumerate(qs, 1):
            t = tags(q.get("explanation", ""))
            ch = q.get("choices", {})
            w.writerow([i, t["domain"], t["skill"], t["difficulty"], t["subject"], q.get("type", ""), q.get("passage", ""),
                        q.get("stem", ""), ch.get("A", ""), ch.get("B", ""), ch.get("C", ""), ch.get("D", ""),
                        q.get("answer", ""), t["body"],
                        ("FLAGGED: " + " | ".join(flags[fingerprint(q)])) if fingerprint(q) in flags else "", ""])

    # HTML
    cards = []
    for i, q in enumerate(qs, 1):
        t = tags(q.get("explanation", ""))
        diff = t["difficulty"].lower()
        choices = "".join(
            f'<li class="{"right" if k == q.get("answer") else ""}"><b>{k})</b> {esc(v)}</li>'
            for k, v in q.get("choices", {}).items())
        words = len(q.get("passage", "").split())
        expl = "".join(f"<p>{esc(p)}</p>" for p in t["body"].splitlines() if p.strip())
        flag = flags.get(fingerprint(q))
        flag_html = ('<div class="flag"><b>Review flag</b> (shipped after the last rewrite; check this first):'
                     + "".join(f"<p>{esc(x)}</p>" for x in flag) + "</div>") if flag else ""
        cards.append(f"""
<article class="card{' flagged' if flag else ''}">
  <div class="meta"><span class="num">{i}</span>
    <span class="chip">{esc(t['domain'] or 'Domain')}</span>
    <span class="chip">{esc(t['skill'] or q.get('type',''))}</span>
    <span class="chip diff {diff}">{esc(t['difficulty'] or '?')}</span>
    {f'<span class="chip subj">{esc(t["subject"])}</span>' if t["subject"] else ''}
    <span class="wc">{words} words</span>
    <button class="copy" onclick="copyBlock(this,'q{i}')">Copy question</button>
    <button class="copy" onclick="copyBlock(this,'a{i}')">Copy with answer</button></div>
  {flag_html}
  <div class="src" id="q{i}">{slide_block(q, False)}</div>
  <div class="src" id="a{i}">{slide_block(q, True)}</div>
  <div class="passage">{render_passage(q.get('passage',''))}</div>
  <p class="stem">{esc(q.get('stem',''))}</p>
  <ol class="choices">{choices}</ol>
  <details><summary>Answer &amp; explanation</summary>
    <p class="ans">Correct answer: <b>{esc(q.get('answer',''))}</b></p>{expl}</details>
</article>""")

    summary = f"{len(qs)} questions" + (f" · cost ${cost:.2f}" if cost is not None else "")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Question Review · {run.name}</title>
<style>
:root{{--navy:{NAVY};--gold:{GOLD};--cream:#F8F4EA;--ink:#1d2433;--muted:#5d667a;--line:#e3e1da}}
*{{box-sizing:border-box}} body{{margin:0;background:#fbfaf7;color:var(--ink);font:16px/1.55 "Liberation Serif",Georgia,serif}}
header{{background:var(--navy);color:#fff;padding:28px 20px}} header .brand{{color:var(--gold);letter-spacing:.18em;font-size:13px;font-weight:700}}
header h1{{margin:6px 0 4px;font-weight:400;font-size:28px}} header p{{margin:0;opacity:.85}}
main{{max-width:820px;margin:0 auto;padding:20px 16px 60px}}
.tools{{display:flex;justify-content:flex-end;margin-bottom:12px}} button{{font:inherit;background:var(--gold);border:0;border-radius:6px;padding:8px 14px;cursor:pointer;color:#1d2433}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:20px;margin:0 0 18px;break-inside:avoid}}
.meta{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:12px;font-size:13px}}
.num{{background:var(--navy);color:#fff;border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-weight:700}}
.chip{{background:var(--cream);border-radius:999px;padding:2px 10px;color:var(--ink)}} .diff.easy{{background:#e3f1e3}} .diff.medium{{background:#fbefd2}} .diff.hard{{background:#f7dcdc}} .subj{{background:#e6ecf7;color:var(--navy)}}
.wc{{margin-left:auto;color:var(--muted)}}
.flag{{background:#fff4cc;border:1px solid var(--gold);border-radius:8px;padding:8px 12px;margin:0 0 12px;font-size:14px}} .flag p{{margin:4px 0 0}} .card.flagged{{border-color:var(--gold)}}
.copy{{background:#fff;border:1px solid var(--navy);color:var(--navy);padding:4px 10px;font-size:13px}} .copy.done{{background:var(--navy);color:#fff}}
.src{{position:fixed;left:0;top:0;width:900px;opacity:0;pointer-events:none;z-index:-1;background:#fff;color:#000}}
.passage{{border-left:3px solid var(--gold);padding-left:14px}} .passage p{{margin:0 0 8px}} .passage .title{{font-weight:700}}
table.data{{border-collapse:collapse;margin:6px 0 12px;font-size:15px}} table.data th{{background:var(--navy);color:#fff;font-weight:400;padding:6px 10px;text-align:left}} table.data td{{border-bottom:1px solid var(--line);padding:6px 10px}}
.stem{{font-weight:700;margin:14px 0 8px}} .choices{{list-style:none;padding:0;margin:0}} .choices li{{padding:6px 8px;border-radius:6px}}
details{{margin-top:12px;background:var(--cream);border-radius:8px;padding:10px 14px}} summary{{cursor:pointer;color:var(--navy);font-weight:700}}
details[open] ~ .x{{}} .show .choices li.right, details[open] {{}}
body.reveal .choices li.right{{background:#e3f1e3}}
@media print{{.copy,.src{{display:none}} header{{-webkit-print-color-adjust:exact;print-color-adjust:exact}} .tools{{display:none}} details{{display:none}} body.reveal details{{display:block}}}}
</style></head><body>
<header><div class="brand">MEZA SCHOLARS</div><h1>Question Review</h1><p>{run.name} · {summary}</p></header>
<main><div class="tools"><button onclick="toggle()">Show all answers</button></div>
{''.join(cards)}
</main>
<script>
function flash(btn,ok){{const old=btn.dataset.label||btn.textContent; btn.dataset.label=old;
  btn.textContent=ok?'Copied ✓':'Copy failed - press Cmd+C'; btn.classList.add('done');
  setTimeout(()=>{{btn.textContent=old; btn.classList.remove('done');}},1800);}}
function copyBlock(btn,id){{
  const el=document.getElementById(id); let ok=false;
  // 1) Copy the rendered block through a normal selection (keeps bold, underline, tables).
  const sel=getSelection(); const r=document.createRange(); r.selectNodeContents(el);
  sel.removeAllRanges(); sel.addRange(r);
  try{{ ok=document.execCommand('copy'); }}catch(e){{ ok=false; }}
  sel.removeAllRanges();
  if(ok){{ flash(btn,true); return; }}
  // 2) Fallback: the async clipboard API.
  if(navigator.clipboard&&window.ClipboardItem){{
    navigator.clipboard.write([new ClipboardItem({{
      'text/html':new Blob([el.innerHTML],{{type:'text/html'}}),
      'text/plain':new Blob([el.innerText],{{type:'text/plain'}})}})])
      .then(()=>flash(btn,true),()=>flash(btn,false));
  }} else {{ flash(btn,false); }}
}}
function toggle(){{const on=document.body.classList.toggle('reveal');
document.querySelectorAll('details').forEach(d=>d.open=on);
document.querySelector('.tools button').textContent=on?'Hide answers':'Show all answers';}}
</script></body></html>"""
    (run / "review.html").write_text(page, encoding="utf-8")
    subjects = [tags(q.get("explanation", ""))["subject"] or "untagged" for q in qs]
    spread = {s: subjects.count(s) for s in dict.fromkeys(subjects)}
    print("Subjects: " + ", ".join(f"{s} ({n})" for s, n in spread.items()))
    print(f"Wrote {run/'review.html'} and {run/'questions.csv'}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else max(Path("runs").glob("*/questions.json"), key=lambda p: p.stat().st_mtime).parent
    build(target)
