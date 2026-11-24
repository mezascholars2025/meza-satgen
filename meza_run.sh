#!/bin/bash
# Meza Scholars question generator (runs satgen with the Meza rulebook and quality gate).
#
#   ./meza_run.sh 20                        20 questions in the full-test mix
#   ./meza_run.sh 8 central_ideas details   8 questions of only the listed types
#
# Each run is saved in its own folder: runs/<date_time>/questions.json, report.json, quality.jsonl
# Quality gate settings (MEZA_QUALITY, MEZA_MAX_REPAIRS, ...) are listed at the top of meza_satgen.py

cd "$(dirname "$0")" || exit 1
COUNT="${1:-20}"
shift 2>/dev/null

if [ "$#" -gt 0 ]; then
  TYPES=("$@")
else
  # 53-slot rotation that mirrors the real test's mix:
  # Craft & Structure 15, Information & Ideas 14, Conventions 14, Expression of Ideas 10
  TYPES=(
    words_in_context_completion details boundaries_independent_clauses transitions
    words_in_context_meaning command_of_evidence_quantitative form_subject_verb_agreement rhetorical_synthesis
    text_structure_function inferences boundaries_supplements transitions
    words_in_context_completion central_ideas form_verb_tense rhetorical_synthesis
    words_in_context_meaning command_of_evidence_claim boundaries_dependent_clauses transitions
    text_structure_purpose details form_finite_verbs rhetorical_synthesis
    words_in_context_completion command_of_evidence_quantitative boundaries_end_punctuation transitions
    cross_text_connections inferences form_pronouns rhetorical_synthesis
    words_in_context_meaning command_of_evidence_quotation boundaries_transition_punctuation transitions
    text_structure_overall central_ideas form_possessives rhetorical_synthesis
    words_in_context_completion details boundaries_lists
    words_in_context_meaning command_of_evidence_quantitative form_modifiers
    text_structure_function inferences boundaries_supplements
    words_in_context_completion command_of_evidence_claim boundaries_unnecessary_punctuation
    text_structure_purpose
  )
fi

[ -z "$VIRTUAL_ENV" ] && source .venv/bin/activate

# Load API keys and settings from .env (satgen does not read the file on its own)
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-key-here" ]; then
  echo "Missing ANTHROPIC_API_KEY in .env - add your key and try again."; exit 1
fi

# Models: Claude Sonnet writes, Claude Haiku solves (a different model, so it can't share
# the writer's blind spots), Claude Sonnet reviews. Override the writer in .env if needed.
export SATGEN_GEN_MODEL="${SATGEN_GEN_MODEL:-claude-sonnet-5-5}"
SOLVER_MODEL="${MEZA_SOLVER_MODEL:-claude-haiku-5-5}"

RUN="runs/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN"
export MEZA_QUALITY_LOG="$RUN/quality.jsonl"   # every draft, what failed, and the no-passage probe
python meza_satgen.py run --count "$COUNT" --types "${TYPES[@]}" --solver-model "$SOLVER_MODEL" \
  --context-file meza_context.txt \
  --checkpoint "$RUN/checkpoint.jsonl" --out "$RUN/questions.json" --report "$RUN/report.json"
echo ""
if [ -f "$RUN/questions.json" ]; then
  python meza_export.py "$RUN" && open "$RUN/review.html" 2>/dev/null
fi
echo "Saved to $RUN"
