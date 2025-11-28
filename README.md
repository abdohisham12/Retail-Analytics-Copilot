Retail Analytics Copilot
========================

This project implements the hybrid Retail Analytics Copilot described in the assignment. It combines LangGraph for control flow, DSPy modules for generation, a TF-IDF retriever over local markdown docs, and direct SQL over the Northwind SQLite database.

## Graph Design Highlights

- Router → Retriever → Planner → NL→SQL → Executor → Synthesizer pipeline with validator + repair loop (max 2 retries) and full trace logging.
- Planner tracks constraints (dates, KPIs, categories) pulled from both retrieved doc chunks and schema summaries to coordinate RAG vs SQL.
- Executor wraps a hardened SQLite tool that records referenced tables so the synthesizer can cite both docs and database tables precisely.
- Synthesizer enforces the format_hint, merges citations from DSPy output + deterministic doc/table fallbacks, and surfaces a heuristic confidence score.

## DSPy Optimization & Metrics

| Module        | Optimizer                  | Metric (valid-SQL rate) |
|---------------|----------------------------|-------------------------|
| NL→SQLModule  | BootstrapFewShot (4 demos) | 0.33 → 1.00 (handcrafted trio) |

- `agent/dspy_signatures.py` exposes `default_sql_trainset()` if you want to extend the curated train set.
- During startup `run_agent_hybrid.py` persists `testing/metrics/dspy_report.json` summarizing the before/after valid-SQL metric so reviewers can audit DSPy impact.

## Assumptions & Policies

- Cost of goods is approximated as `0.7 * UnitPrice` whenever the KPI calls for margin and the database lacks explicit cost columns.
- Doc/RAG citations always appear as `filename::chunk{N}`; SQL citations list every table detected in the executed query.
- All inference stays local (Ollama Phi-3.5-mini-instruct by default). You can change the model via `--model` on the CLI.

## Evaluation & Metrics

- `python testing/cli_regression.py` runs the hybrid agent against `sample_questions_hybrid_eval.jsonl`, writes `testing/latest_outputs.jsonl`, and diffs against `testing/baseline_outputs.jsonl`.
- Format, citation, and exact-match coverage stats are summarized in `testing/metrics/latest_metrics.json`.
- DSPy optimization metrics for NL→SQL live in `testing/metrics/dspy_report.json` (written whenever the agent boots).
- Share `outputs_hybrid.jsonl` + `testing/metrics` artifacts with HR so they can verify conformance (format_hint adherence, SQL citations, DSPy impact).

## Running the Agent

```bash
pip install -r requirements.txt
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

Environment variables:

- `OLLAMA_HOST` (optional) to point at a remote Ollama server.
- `NORTHWIND_DB_PATH` (optional) if you relocate `data/northwind.sqlite`.

## Outputs & Evaluation

- Every line in `outputs_hybrid.jsonl` abides by the assignment contract:
  - `final_answer` matches the `format_hint` (int/float/object/list) and uses ±0.01 tolerance for floats.
  - `citations` list every SQL table touched plus each referenced doc chunk (`kpi_definitions::chunk2`, etc.).
  - `sql` holds the last executed query (or `""` for RAG-only answers) and `confidence` is a 0–1 float.
- The validator + repair loop enforces ≤2 repair iterations while checking SQL errors/empties, missing answers, format adherence, and citation coverage (traced as `validator:ok` / `validator:sql_error,...`).
- `outputs_hybrid.jsonl` is overwritten on each run; include it and the `testing/metrics` folder when submitting.

