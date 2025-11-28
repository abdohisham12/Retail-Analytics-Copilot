# Evaluation Summary

## Completed Tasks

### 1. Batch Evaluation Command
- Command ready: `python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl`
- Note: Requires Ollama to be running with model `phi3.5:3.8b-mini-instruct-q4_K_M`

### 2. Evaluation Script Created (`evaluate_output.py`)

The evaluation script performs comprehensive evaluation of both format compliance and accuracy:

#### Format Compliance Checks:
- ✅ All required fields present (id, final_answer, sql, confidence, explanation, citations)
- ✅ `final_answer` type matches `format_hint` exactly (int, float, object, list)
- ✅ Float tolerance: ±0.01 for float answers
- ✅ `sql` field is string (empty for RAG-only queries)
- ✅ `confidence` in valid range [0.0, 1.0]
- ✅ `explanation` ≤ 2 sentences
- ✅ Citations format: `filename::chunk0` (not `filename::doc_0`)

#### Accuracy Evaluation:
- ✅ Queries database to compute expected answers for all 6 questions
- ✅ Compares model outputs with expected answers
- ✅ Handles float tolerance (±0.01)
- ✅ Validates object and list structures

### 3. Expected Answers Computed

The evaluation script queries the database to get ground truth answers:

1. **rag_policy_beverages_return_days**: 14 (from product_policy.md)
2. **hybrid_top_category_qty_summer_1997**: Top category by quantity in June 1997
3. **hybrid_aov_winter_1997**: AOV = SUM(revenue) / COUNT(DISTINCT OrderID) for Dec 1997
4. **sql_top3_products_by_revenue_alltime**: Top 3 products by total revenue
5. **hybrid_revenue_beverages_summer_1997**: Total revenue from Beverages category in June 1997
6. **hybrid_best_customer_margin_1997**: Top customer by gross margin (CostOfGoods = 70% of UnitPrice)

## Usage

1. **Run batch evaluation** (requires Ollama):
   ```bash
   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```

2. **Evaluate results**:
   ```bash
   python evaluate_output.py
   ```

The evaluation script will:
- Show expected answers from database
- Check format compliance for each output
- Compare accuracy with expected answers
- Provide summary report

## Output Contract Verification

Each output line in `outputs_hybrid.jsonl` is checked against:

```json
{
  "id": "...",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty>",
  "confidence": 0.0-1.0,
  "explanation": "<= 2 sentences",
  "citations": ["Orders", "kpi_definitions::chunk2"]
}
```

## Status

✅ Evaluation system is complete and ready
⚠️ Batch evaluation requires Ollama to be running locally
✅ Once `outputs_hybrid.jsonl` is generated, run `evaluate_output.py` for full evaluation

