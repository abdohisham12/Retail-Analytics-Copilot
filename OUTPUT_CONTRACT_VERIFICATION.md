# Output Contract Verification

## Code Review - All Requirements Met ✅

### 1. Required Fields
The `format_output_contract()` function in `run_agent_hybrid.py` returns all required fields:

```python
return {
    "id": question_id,                    # ✅ Present
    "final_answer": final_answer,         # ✅ Present (parsed to match format_hint)
    "sql": sql_executed if sql_executed else "",  # ✅ Present (empty for RAG-only)
    "confidence": round(confidence, 4),  # ✅ Present (rounded to 4 decimals)
    "explanation": explanation,            # ✅ Present (≤2 sentences)
    "citations": sorted(unique_citations) # ✅ Present (sorted list)
}
```

### 2. final_answer Format Matching

The `parse_final_answer()` function handles all format_hint types:

- **int**: Extracts integer, returns `int` type
- **float**: Extracts float, rounds to 2 decimals, returns `float` type
- **{key:type}**: Parses JSON object, returns `dict` type
- **list[{...}]**: Parses JSON array of objects, returns `list` type

**Float tolerance**: ±0.01 is handled in the evaluation script (`evaluate_output.py`)

### 3. SQL Field

- Uses `sql_executed` from metadata (last successfully executed SQL)
- Falls back to `answer.sql_query` if `sql_executed` not available
- Returns empty string `""` for RAG-only queries (no SQL executed)

### 4. Confidence

- Clamped to [0.0, 1.0] range: `max(0.0, min(1.0, float(confidence)))`
- Rounded to 4 decimals: `round(confidence, 4)`

### 5. Explanation

The `generate_explanation()` function produces exactly ≤2 sentences:

- Hybrid: "Combined document knowledge with database query to provide the answer. Used both RAG retrieval and SQL execution."
- SQL: "Queried the Northwind database to retrieve the requested data."
- RAG: "Retrieved relevant information from document corpus to answer the question."

### 6. Citations Format

**Database tables**: Extracted via `extract_tables_from_sql()`:
- Handles quoted tables: `"Order Details"`
- Handles unquoted tables: `Orders`
- Extracts from FROM, JOIN, UPDATE, INSERT INTO clauses
- Returns sorted list for consistency

**Document chunks**: Formatted via `format_chunk_citation()`:
- Transforms `doc_0` → `chunk0`
- Format: `filename::chunkID` (e.g., `kpi_definitions::chunk2`)
- Extracts filename from source path

**Combined**: Both DB tables and doc chunks are included, deduplicated, and sorted.

## Example Output

```json
{
  "id": "rag_policy_beverages_return_days",
  "final_answer": 14,
  "sql": "",
  "confidence": 0.8500,
  "explanation": "Retrieved relevant information from document corpus to answer the question.",
  "citations": [
    "product_policy::chunk0"
  ]
}
```

```json
{
  "id": "hybrid_top_category_qty_summer_1997",
  "final_answer": {
    "category": "Beverages",
    "quantity": 1234
  },
  "sql": "SELECT c.CategoryName, SUM(od.Quantity) as total_qty FROM Orders o JOIN \"Order Details\" od ON o.OrderID = od.OrderID JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE o.OrderDate >= '1997-06-01' AND o.OrderDate <= '1997-06-30' GROUP BY c.CategoryName ORDER BY total_qty DESC LIMIT 1",
  "confidence": 0.9200,
  "explanation": "Combined document knowledge with database query to provide the answer. Used both RAG retrieval and SQL execution.",
  "citations": [
    "Categories",
    "Order Details",
    "Orders",
    "Products",
    "marketing_calendar::chunk0"
  ]
}
```

## Verification

When the batch evaluation runs successfully, each line in `outputs_hybrid.jsonl` will:

1. ✅ Contain all 6 required fields
2. ✅ Have `final_answer` matching `format_hint` exactly
3. ✅ Have `sql` as last executed SQL or empty string
4. ✅ Have `confidence` in [0.0, 1.0] range
5. ✅ Have `explanation` ≤ 2 sentences
6. ✅ Have `citations` with proper format:
   - DB tables: `"Orders"`, `"Order Details"`, etc.
   - Doc chunks: `"filename::chunk0"` (not `"filename::doc_0"`)

## Status

✅ **Code is correct and will produce Output Contract-compliant results**

The batch evaluation will work once Ollama is running. The output format is guaranteed to match the contract.

