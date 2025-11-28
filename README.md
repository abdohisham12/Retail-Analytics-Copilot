# Retail Analytics Copilot

A local, free AI agent that answers retail analytics questions by combining RAG over local documents and SQL queries over a local SQLite database (Northwind).

## Graph Design

- **LangGraph workflow with 8 nodes**: Router (DSPy classifier) → Retriever (top-k doc chunks) → Planner (extract constraints) → SQL Generate (NL→SQL with PRAGMA schema) → Executor (run SQL, capture errors) → Synthesizer (typed answer with citations) → Repair (max 2 attempts) → Checkpointer (event logging)
- **Conditional routing**: Router classifies queries as RAG-only, SQL-only, or Hybrid, directing flow accordingly
- **Repair loop**: On SQL errors, the repair node revises queries up to 2 times before giving up
- **Stateful workflow**: TypedDict state management with event logging for traceability and replay

## DSPy Optimization

**Module optimized**: QueryRouter (classification accuracy)

**Optimizer**: BootstrapFewShot with small budget (max 4 bootstrapped demos, 8 labeled demos)

**Training set**: 30 examples (10 RAG-only, 10 SQL-only, 10 Hybrid queries)

**Metrics**: Run `python optimize_router.py` to get baseline → optimized accuracy. The optimized router is automatically loaded if `agent/optimized_router.pkl` exists.

## Trade-offs & Assumptions

- **CostOfGoods approximation**: When cost data is missing (as in the Northwind database), we approximate CostOfGoods as 70% of UnitPrice per sample question requirements. The KPI docs suggest category-level averages as an alternative approach.
- **Chunk size**: 500 characters with 50 character overlap balances context retention with prompt size constraints (≤1k tokens)
- **Repair attempts**: Limited to 2 iterations to prevent infinite loops while allowing error recovery
- **Confidence heuristics**: Weighted combination of RAG retrieval scores (coverage), SQL execution success, and row count, with a 0.1 penalty per repair attempt (max 0.2)
- **Local-only**: All components run locally (Ollama, ChromaDB, SQLite) with no external API calls at inference time

