"""Generate a self-contained HTML metrics report from test/results/metrics.json.

Tabs: Opik | DeepEval | Test Cases. Footer: metric glossary for engineers
(what each metric means, direction, and how to tune prompts/chunks) + references.
"""

import html
import json
import re
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent / "test"
METRICS_PATH = TEST_DIR / "results" / "metrics.json"
REPORT_PATH = TEST_DIR / "report.html"

# Higher-is-better metrics vs hallucination-style (lower is better).
LOWER_IS_BETTER = {"hallucination"}

OPIK_METRICS = ["answer_relevance", "hallucination", "context_precision", "context_recall"]
DEEPEVAL_METRICS = ["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall", "hallucination"]

LABELS = {
    "answer_relevance": "Answer Relevance",
    "answer_relevancy": "Answer Relevancy",
    "hallucination": "Hallucination",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "contextual_precision": "Contextual Precision",
    "contextual_recall": "Contextual Recall",
    "faithfulness": "Faithfulness",
}

GLOSSARY = [
    ("Answer Relevance / Relevancy", "Opik + DeepEval", "Does the generated answer actually address the question that was asked?", "0–1, higher is better",
     "Low scores usually mean the generation prompt lets the model ramble or the retrieved chunks pull it off-topic. Tighten the answer prompt, reduce top-k, or rephrase test questions to be unambiguous."),
    ("Hallucination", "Opik + DeepEval", "Does the answer contain statements NOT supported by the retrieved context? (Opik and DeepEval both score the degree of contradiction/fabrication.)", "0–1, LOWER is better",
     "High scores mean the model invents facts. Strengthen the 'answer only from context' instruction, lower generation temperature, or improve retrieval so the right chunk is actually present (bigger chunks / higher top-k)."),
    ("Faithfulness", "DeepEval", "The share of claims in the answer that are supported by the retrieved context (the inverse view of hallucination).", "0–1, higher is better",
     "Same levers as hallucination: grounding instructions, temperature, and making sure retrieval surfaces the supporting chunk."),
    ("Context Precision", "Opik + DeepEval", "Of the chunks retrieved, how many are actually relevant to answering the question - and (DeepEval) are the relevant ones ranked first?", "0–1, higher is better",
     "Low precision = the retriever returns noise. Reduce top-k, use smaller/tighter chunks (e.g. 256–512 tokens), or improve chunk overlap so a chunk holds one coherent idea."),
    ("Context Recall", "Opik + DeepEval", "Does the retrieved context contain ALL the information needed to produce the expected answer?", "0–1, higher is better",
     "Low recall = the right content never got retrieved. Increase top-k, use larger chunks or more overlap, or check the document was parsed/indexed correctly (rag list-files)."),
]

REFERENCES = [
    ("Opik metrics documentation (Comet)", "https://www.comet.com/docs/opik/evaluation/metrics/overview"),
    ("DeepEval metrics documentation (Confident AI)", "https://deepeval.com/docs/metrics-introduction"),
    ("Vertex AI RAG Engine overview (Google Cloud)", "https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview"),
    ("Vertex AI RAG chunking & retrieval tuning", "https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/fine-tune-rag-transformations"),
    ("RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)", "https://arxiv.org/abs/2309.15217"),
    ("Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)", "https://arxiv.org/abs/2005.11401"),
]


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def md_lite(text: str) -> str:
    """Render the small markdown subset the recommender emits (bold + bullets + scope tags)."""
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc(text))
    out = out.replace("[SYSTEM]", '<span class="scope scope-system">SYSTEM</span>')
    out = out.replace("[PROMPT]", '<span class="scope scope-prompt">PROMPT</span>')
    parts, in_list = [], False
    for line in out.split("\n"):
        s = line.strip()
        if s.startswith(("- ", "* ")):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{s[2:]}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            if s:
                parts.append(f"<p>{s}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


def status_for(metric: str, score: float | None) -> tuple[str, str, str]:
    """Return (css_class, icon, aria_label) for a score chip."""
    if score is None:
        return "s-none", "–", "not available"
    if metric in LOWER_IS_BETTER:
        if score <= 0.3:
            return "s-good", "✓", "good"
        if score <= 0.6:
            return "s-warn", "!", "needs attention"
        return "s-crit", "✕", "critical"
    if score >= 0.7:
        return "s-good", "✓", "good"
    if score >= 0.4:
        return "s-warn", "!", "needs attention"
    return "s-crit", "✕", "critical"


def chip(metric: str, entry: dict) -> str:
    score = entry.get("score")
    cls, icon, aria = status_for(metric, score)
    if score is None:
        err = entry.get("error", "no score")
        return f'<span class="chip s-none" title="{esc(err)}"><span class="ic">–</span> n/a</span>'
    return (
        f'<span class="chip {cls}" aria-label="{aria}">'
        f'<span class="ic">{icon}</span> {score:.2f}</span>'
    )


def summary_tiles(summary: dict, metrics_order: list[str]) -> str:
    tiles = []
    for m in metrics_order:
        val = summary.get(m)
        direction = "lower is better" if m in LOWER_IS_BETTER else "higher is better"
        cls, icon, _ = status_for(m, val)
        value_html = f"{val:.2f}" if val is not None else "–"
        tiles.append(
            f'<div class="tile"><div class="tile-label">{esc(LABELS.get(m, m))}</div>'
            f'<div class="tile-value">{value_html} <span class="chip {cls}"><span class="ic">{icon}</span></span></div>'
            f'<div class="tile-sub">mean of all cases · {direction}</div></div>'
        )
    return f'<div class="tiles">{"".join(tiles)}</div>'


def metric_table(cases: list[dict], lib: str, metrics_order: list[str]) -> str:
    head = "".join(
        f'<th>{esc(LABELS.get(m, m))}{" ↓" if m in LOWER_IS_BETTER else ""}</th>' for m in metrics_order
    )
    rows = []
    for c in cases:
        cells = "".join(f"<td>{chip(m, c[lib].get(m, {}))}</td>" for m in metrics_order)
        reasons = "".join(
            f"<p><strong>{esc(LABELS.get(m, m))}:</strong> {esc(c[lib].get(m, {}).get('reason') or c[lib].get(m, {}).get('error') or '—')}</p>"
            for m in metrics_order
        )
        mit = c.get("mitigation")
        flagged_here = any(
            status_for(m, c[lib].get(m, {}).get("score"))[0] in ("s-warn", "s-crit") for m in metrics_order
        )
        mit_html = ""
        if mit:
            flags = ", ".join(mit.get("flags", []))
            mit_html = (
                f'<div class="mitigation"><div class="mit-title">🔧 How to fix'
                f'<span class="mit-flags">{esc(flags)}</span></div>{md_lite(mit["advice"])}</div>'
            )
        open_attr = " open" if (mit_html and flagged_here) else ""
        rows.append(
            f'<tr><td class="case-id"><div>{esc(c["id"])}</div><span class="cat">{esc(c["category"])}</span></td>{cells}</tr>'
            f'<tr class="reason-row"><td colspan="{len(metrics_order) + 1}">'
            f'<details{open_attr}><summary>Judge reasoning for {esc(c["id"])}</summary>{reasons}{mit_html}</details></td></tr>'
        )
    return (
        f'<div class="table-wrap"><table><thead><tr><th>Test case</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def cases_tab(cases: list[dict]) -> str:
    blocks = []
    for c in cases:
        ctxs = "".join(
            f'<details class="ctx"><summary>Chunk {i} · {esc(ctx["source"])}'
            f'{f" · score {ctx['score']:.3f}" if ctx.get("score") is not None else ""}</summary>'
            f"<pre>{esc(ctx['text'])}</pre></details>"
            for i, ctx in enumerate(c["contexts"], 1)
        )
        blocks.append(
            f'<article class="case"><h3>{esc(c["id"])} <span class="cat">{esc(c["category"])}</span></h3>'
            f'<p class="q"><strong>Q:</strong> {esc(c["question"])}</p>'
            f'<p><strong>Answer (RAG):</strong> {esc(c["answer"])}</p>'
            f'<p class="expected"><strong>Expected:</strong> {esc(c["expected_output"])}</p>'
            f'<div class="ctxs"><strong>Retrieved context ({len(c["contexts"])} chunks):</strong>{ctxs}</div></article>'
        )
    return "".join(blocks)


def footer(run: dict) -> str:
    gloss_rows = "".join(
        f"<tr><td><strong>{esc(n)}</strong></td><td>{esc(lib)}</td><td>{esc(what)}</td><td>{esc(rng)}</td><td>{esc(tune)}</td></tr>"
        for n, lib, what, rng, tune in GLOSSARY
    )
    refs = "".join(f'<li><a href="{esc(u)}" rel="noopener">{esc(t)}</a> — <code>{esc(u)}</code></li>' for t, u in REFERENCES)
    return f"""
<footer>
  <h2>Understanding these metrics (for software engineers)</h2>
  <p>Every score is produced by an LLM judge ({esc(run.get("judge_model", ""))} on {esc(run.get("judge_backend", ""))}) grading the RAG
  pipeline's answer against the question, the retrieved chunks, and a hand-written expected answer.
  Scores are 0–1. <strong>Retrieval problems</strong> show up in context precision/recall;
  <strong>generation problems</strong> show up in relevance, faithfulness and hallucination.
  Fix retrieval first — generation can only be as grounded as the chunks it is given.</p>
  <div class="table-wrap"><table class="glossary">
    <thead><tr><th>Metric</th><th>Library</th><th>What it measures</th><th>Range</th><th>How to tune prompts / chunks</th></tr></thead>
    <tbody>{gloss_rows}</tbody>
  </table></div>
  <h2>Quick tuning cheat-sheet</h2>
  <ul class="cheats">
    <li><strong>Low context recall</strong> → raise <code>--top-k</code>, increase chunk size/overlap, re-upload with a custom <code>TransformationConfig</code>.</li>
    <li><strong>Low context precision</strong> → lower <code>--top-k</code>, use smaller chunks so each holds one idea.</li>
    <li><strong>High hallucination / low faithfulness</strong> → harden the answer prompt ("answer ONLY from context"), lower temperature, verify recall first.</li>
    <li><strong>Low answer relevance</strong> → clarify the question or the answer prompt; check the corpus actually covers the topic (<code>rag query</code>).</li>
    <li><strong>Out-of-scope prompts</strong> should score LOW on hallucination only if the model abstains — if it invents an answer, tighten grounding.</li>
  </ul>
  <h2>Change scope: system-wide vs per-prompt</h2>
  <p>Every auto-generated mitigation carries a scope tag so you know the blast radius before touching anything:</p>
  <div class="table-wrap"><table class="glossary">
    <thead><tr><th>Tag</th><th>Where it is changed</th><th>What it affects</th><th>Examples</th></tr></thead>
    <tbody>
      <tr><td><span class="scope scope-prompt">PROMPT</span></td>
          <td><code>test/prompt.json</code> (that one entry)</td>
          <td>Only that test prompt on the next <code>test-run</code></td>
          <td>Per-prompt retrieval depth via a <code>"top_k"</code> field on the prompt entry
              (overrides <code>--top-k</code> for that prompt only); rewording the question; fixing the expected answer.</td></tr>
      <tr><td><span class="scope scope-system">SYSTEM</span></td>
          <td>CLI flags, <code>rag_eval.py</code>, or corpus re-upload</td>
          <td>Every prompt and every future query against the corpus</td>
          <td>Default <code>--top-k</code>; chunk size/overlap (<code>TransformationConfig</code> + re-upload);
              the shared grounding prompt; generation model or temperature.</td></tr>
    </tbody>
  </table></div>
  <p class="experimental">⚠️ <strong>The auto-generated "How to fix" recommendations are experimental.</strong>
  They are produced by an LLM judge and can misdiagnose — treat them as a starting hypothesis, apply human
  judgement, prefer the smallest-scope change (<span class="scope scope-prompt">PROMPT</span> before
  <span class="scope scope-system">SYSTEM</span>), and validate every change with a fresh
  <code>uv run rag test-all</code> before adopting it.</p>
  <h2>References &amp; sources</h2>
  <ul class="refs">{refs}</ul>
</footer>"""


def generate(metrics_path: Path = METRICS_PATH) -> Path:
    if not metrics_path.exists():
        sys.exit(f"{metrics_path} not found. Run: uv run rag test-eval")
    with open(metrics_path) as f:
        data = json.load(f)
    run, cases, summary = data["run"], data["cases"], data["summary"]
    recommendations = data.get("recommendations")
    recs_html = ""
    if recommendations:
        recs_html = (
            '<section class="recs"><h2>🔧 Run-level tuning recommendations</h2>'
            f"{md_lite(recommendations)}"
            '<p class="recs-note">⚠️ <strong>Experimental:</strong> these tuning recommendations are LLM-generated and under an '
            "experimental stage — apply human judgement and validate with a re-run before changing production parameters. "
            '<span class="scope scope-system">SYSTEM</span> = pipeline-wide change (affects every prompt) · '
            '<span class="scope scope-prompt">PROMPT</span> = local change for that one test prompt '
            '(e.g. a per-prompt <code>"top_k"</code> override in <code>test/prompt.json</code>). '
            "Per-case fixes are inside each \"Judge reasoning\" row above. "
            "Re-run <code>uv run rag test-recommend</code> after any tuning change.</p></section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Evaluation Report — {esc(run.get("corpus", ""))}</title>
<style>
:root {{
  color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --ink-1:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,0.10);
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --good-text:#006300;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface-1:#1a1a19; --page:#0d0d0d; --ink-1:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --border:rgba(255,255,255,0.10); --good-text:#0ca30c;
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink-1);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:24px 20px 60px; }}
header h1 {{ margin:0 0 4px; font-size:1.5rem; }}
.meta {{ color:var(--ink-2); font-size:0.85rem; }}
.meta code {{ background:var(--surface-1); border:1px solid var(--border); border-radius:4px; padding:1px 5px; }}
.tabs {{ display:flex; gap:4px; margin:24px 0 0; border-bottom:1px solid var(--grid); }}
.tabs button {{ appearance:none; background:none; border:none; border-bottom:2px solid transparent;
  color:var(--ink-2); font:inherit; font-weight:600; padding:10px 16px; cursor:pointer; }}
.tabs button[aria-selected="true"] {{ color:var(--ink-1); border-bottom-color:var(--ink-1); }}
.panel {{ display:none; padding-top:20px; }}
.panel.active {{ display:block; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tile-label {{ font-size:0.78rem; color:var(--ink-2); font-weight:600; }}
.tile-value {{ font-size:1.7rem; font-weight:700; margin:2px 0; }}
.tile-sub {{ font-size:0.72rem; color:var(--muted); }}
.table-wrap {{ overflow-x:auto; background:var(--surface-1); border:1px solid var(--border); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:0.86rem; }}
th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--grid); vertical-align:top; }}
th {{ color:var(--ink-2); font-size:0.76rem; text-transform:uppercase; letter-spacing:0.03em; }}
td {{ font-variant-numeric:tabular-nums; }}
.case-id div {{ font-weight:600; }}
.cat {{ display:inline-block; font-size:0.7rem; color:var(--muted); border:1px solid var(--grid);
  border-radius:99px; padding:0 8px; margin-top:2px; }}
.chip {{ display:inline-flex; align-items:center; gap:5px; font-weight:600; border-radius:6px;
  padding:2px 8px; border:1px solid var(--border); background:var(--surface-1); }}
.chip .ic {{ font-size:0.8em; }}
.s-good .ic {{ color:var(--good); }} .s-good {{ border-color:var(--good); }}
.s-warn .ic {{ color:var(--warn); }} .s-warn {{ border-color:var(--warn); }}
.s-crit .ic {{ color:var(--crit); }} .s-crit {{ border-color:var(--crit); }}
.s-none {{ color:var(--muted); }}
.reason-row td {{ border-bottom:1px solid var(--grid); background:color-mix(in srgb, var(--surface-1) 60%, var(--page)); }}
details summary {{ cursor:pointer; color:var(--ink-2); font-size:0.82rem; }}
details p {{ font-size:0.84rem; color:var(--ink-2); margin:6px 0; }}
.case {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px 18px; margin-bottom:14px; }}
.case h3 {{ margin:0 0 8px; font-size:1.02rem; }}
.case .q {{ color:var(--ink-1); }}
.case .expected {{ color:var(--ink-2); }}
.ctx pre {{ white-space:pre-wrap; font-size:0.78rem; background:var(--page); border:1px solid var(--grid);
  border-radius:8px; padding:10px; max-height:260px; overflow:auto; }}
.mitigation {{ margin-top:10px; border:1px solid var(--warn); border-left:4px solid var(--warn);
  border-radius:8px; padding:10px 14px; background:var(--surface-1); }}
.mitigation p, .mitigation li {{ font-size:0.86rem; color:var(--ink-1); margin:4px 0; }}
.mitigation ul {{ margin:4px 0; padding-left:20px; }}
.mit-title {{ font-weight:700; font-size:0.85rem; margin-bottom:4px; }}
.mit-flags {{ font-weight:400; font-size:0.74rem; color:var(--muted); margin-left:10px; }}
.recs {{ margin-top:32px; border:1px solid var(--border); border-left:4px solid var(--warn);
  border-radius:10px; padding:16px 20px; background:var(--surface-1); }}
.recs h2 {{ margin:0 0 8px; font-size:1.05rem; }}
.recs p, .recs li {{ font-size:0.9rem; }}
.recs-note {{ color:var(--muted); font-size:0.78rem !important; }}
.scope {{ display:inline-block; font-size:0.68rem; font-weight:700; letter-spacing:0.04em;
  border-radius:5px; padding:1px 7px; border:1px solid var(--border); vertical-align:middle; }}
.scope-system {{ color:var(--crit); border-color:var(--crit); }}
.scope-prompt {{ color:var(--good-text); border-color:var(--good-text); }}
.experimental {{ border:1px solid var(--warn); border-left:4px solid var(--warn); border-radius:8px;
  padding:10px 14px; font-size:0.88rem; background:var(--surface-1); }}
footer {{ margin-top:40px; border-top:2px solid var(--grid); padding-top:24px; }}
footer h2 {{ font-size:1.1rem; }}
.glossary td, .glossary th {{ font-size:0.82rem; }}
.cheats li, .refs li {{ margin:6px 0; font-size:0.88rem; }}
a {{ color:inherit; }}
code {{ font-size:0.85em; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>RAG Evaluation Report — {esc(run.get("corpus", ""))}</h1>
  <p class="meta">
    Project <code>{esc(run.get("project", ""))}</code> · corpus region <code>{esc(run.get("location", ""))}</code> ·
    generation &amp; judge model <code>{esc(run.get("generation_model", ""))}</code> · top-k <code>{esc(run.get("top_k", ""))}</code><br>
    RAG run {esc(run.get("timestamp", ""))} · evaluated {esc(run.get("evaluated_at", ""))} ·
    {len(cases)} test cases from <code>test/prompt.json</code>
  </p>
</header>

<div class="tabs" role="tablist">
  <button role="tab" aria-selected="true" data-tab="opik">Opik</button>
  <button role="tab" aria-selected="false" data-tab="deepeval">DeepEval</button>
  <button role="tab" aria-selected="false" data-tab="cases">Test Cases</button>
</div>

<section class="panel active" id="panel-opik" role="tabpanel">
  {summary_tiles(summary.get("opik", {}), OPIK_METRICS)}
  {metric_table(cases, "opik", OPIK_METRICS)}
</section>

<section class="panel" id="panel-deepeval" role="tabpanel">
  {summary_tiles(summary.get("deepeval", {}), DEEPEVAL_METRICS)}
  {metric_table(cases, "deepeval", DEEPEVAL_METRICS)}
</section>

<section class="panel" id="panel-cases" role="tabpanel">
  {cases_tab(cases)}
</section>

{recs_html}
{footer(run)}
</div>
<script>
document.querySelectorAll('.tabs button').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', 'false'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  btn.setAttribute('aria-selected', 'true');
  document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
}}));
</script>
</body>
</html>"""
    REPORT_PATH.write_text(page)
    return REPORT_PATH
