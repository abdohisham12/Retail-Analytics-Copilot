# Retail Analytics Copilot

A local, free AI agent that answers retail analytics questions by combining RAG over local documents and SQL queries over a local SQLite database (Northwind).

## ✅ Project Structure (Matches HR Specification)

```
├─ agent/
│  ├─ graph_hybrid.py          # LangGraph (≥6 nodes + repair loop)
│  ├─ dspy_signatures.py       # DSPy Signatures/Modules (Router/NL→SQL/Synth)
│  ├─ rag/
│  │  └─ retrieval.py         # TF-IDF or simple retriever (chunking + search)
│  └─ tools/
│     └─ sqlite_tool.py       # DB access + schema introspection
├─ data/
│  └─ northwind.sqlite         # downloaded DB
├─ docs/
│  ├─ marketing_calendar.md
│  ├─ kpi_definitions.md
│  ├─ catalog.md
│  └─ product_policy.md
├─ sample_questions_hybrid_eval.jsonl
├─ run_agent_hybrid.py         # main entrypoint (CLI contract)
└─ requirements.txt
```

**All required skeleton files are present.** All supporting code has been merged into the skeleton files.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Download database (if not already present)
# The database should be in data/northwind.sqlite
# If missing, download from: https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

### Run Evaluation

```bash
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

### Interactive Mode

```bash
python run_agent_hybrid.py
```

## 🏗️ Architecture

**LangGraph Workflow** (8 nodes):
1. Router - Classifies query (RAG | SQL | Hybrid)
2. Retriever - Top-k doc chunks with scores + chunk IDs
3. Planner - Extracts constraints (dates, KPIs, categories)
4. NL→SQL - Generates SQLite queries using live schema
5. Executor - Runs SQL; captures columns, rows, error
6. Synthesizer - Produces typed answer matching format_hint
7. Repair - Revises SQL on error (max 2 attempts)
8. Checkpointer - Event logging

**Key Components**:
- **DSPy Modules**: Router, SQL Generator, Synthesizer, Repair
- **RAG**: ChromaDB vector store + sentence transformers
- **SQL**: SQLite with schema introspection

## 📊 Evaluation

**Scoring Criteria**:
- Correctness (40%) - Values match expected, types match format_hint
- DSPy Impact (20%) - Measurable improvement on Router module
- Resilience (20%) - Repair loop improves valid-SQL rate
- Clarity (20%) - Readable code, proper citations, trace system

## 📝 Output Contract

Each answer follows this format:
```json
{
  "id": "question_id",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty>",
  "confidence": 0.0-1.0,
  "explanation": "<= 2 sentences",
  "citations": ["Orders", "Order Details", "kpi_definitions::chunk2"]
}
```

## 📋 Constraints

- ✅ **No external network calls at inference** - All components local
- ✅ **Compact prompts** - ≤1k tokens total
- ✅ **Repair bound** - ≤2 iterations maximum

## 🐛 Troubleshooting

**Ollama Connection Issues**:
- Ensure Ollama is running: `ollama serve`
- Check model: `ollama list`

**Database Not Found**:
- Ensure `data/northwind.sqlite` exists
- If missing, download from: https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

## 📄 License

MIT License - Free for personal and commercial use.
