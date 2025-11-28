# Batch Evaluation Guide

## Prerequisites

Before running the batch evaluation, ensure:

1. **Ollama is running**:
   ```bash
   ollama serve
   ```

2. **Model is available**:
   ```bash
   ollama list
   # Should show: phi3.5:3.8b-mini-instruct-q4_K_M
   # If not, run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```

3. **Database exists**:
   ```bash
   # Verify: data/northwind.sqlite exists
   ```

4. **Documents are indexed** (first run will do this automatically):
   ```bash
   # Vector store should exist: vector_store/
   ```

## Running Batch Evaluation

### Method 1: Using run_agent_hybrid.py (Official)
```bash
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

### Method 2: Using run_batch_test.py (With Progress)
```bash
python run_batch_test.py
```

## Expected Output

The command will:
1. Initialize the agent (1-2 minutes on first run, faster if vector_store exists)
2. Process all 6 questions from `sample_questions_hybrid_eval.jsonl`
3. Generate `outputs_hybrid.jsonl` with one JSON object per line

Each line in `outputs_hybrid.jsonl` follows the Output Contract:
```json
{
  "id": "question_id",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty>",
  "confidence": 0.0-1.0,
  "explanation": "<= 2 sentences",
  "citations": ["Orders", "kpi_definitions::chunk2"]
}
```

## Evaluating Results

After `outputs_hybrid.jsonl` is generated, run:

```bash
python evaluate_output.py
```

This will:
1. ✅ Check format compliance (all Output Contract requirements)
2. ✅ Query database for expected answers
3. ✅ Compare accuracy (with ±0.01 tolerance for floats)
4. ✅ Provide detailed report

## Expected Questions & Answers

1. **rag_policy_beverages_return_days** (RAG-only)
   - Expected: `14` (int)
   - Source: `docs/product_policy.md`

2. **hybrid_top_category_qty_summer_1997** (Hybrid)
   - Expected: `{category: str, quantity: int}`
   - Dates: 1997-06-01 to 1997-06-30

3. **hybrid_aov_winter_1997** (Hybrid)
   - Expected: `float` (AOV for Dec 1997)
   - Formula: SUM(revenue) / COUNT(DISTINCT OrderID)

4. **sql_top3_products_by_revenue_alltime** (SQL-only)
   - Expected: `list[{product: str, revenue: float}]`
   - Top 3 products by total revenue

5. **hybrid_revenue_beverages_summer_1997** (Hybrid)
   - Expected: `float` (Beverages revenue in June 1997)

6. **hybrid_best_customer_margin_1997** (Hybrid)
   - Expected: `{customer: str, margin: float}`
   - CostOfGoods = 70% of UnitPrice

## Troubleshooting

### Issue: Output file not created
- **Check**: Is Ollama running? `ollama list`
- **Check**: Is the model available? `ollama pull phi3.5:3.8b-mini-instruct-q4_K_M`
- **Check**: Are there errors in the console?

### Issue: Initialization takes too long
- **First run**: Normal (document indexing takes 1-2 minutes)
- **Subsequent runs**: Should be faster if `vector_store/` exists
- **If stuck**: Check `ollama_debug.log` for errors

### Issue: Questions fail
- **Check**: Database connection (`data/northwind.sqlite` exists)
- **Check**: Documents are indexed (`vector_store/chroma.sqlite3` exists)
- **Check**: Ollama model is responding

## Verification Checklist

Before submitting to HR, verify:

- [ ] `outputs_hybrid.jsonl` contains 6 lines (one per question)
- [ ] Each line is valid JSON matching Output Contract
- [ ] `final_answer` matches `format_hint` exactly
- [ ] Citations use `filename::chunk0` format (not `doc_0`)
- [ ] SQL field contains last executed SQL or empty string
- [ ] Confidence is between 0.0 and 1.0
- [ ] Explanation is ≤ 2 sentences
- [ ] Run `python evaluate_output.py` shows format compliance
- [ ] Accuracy evaluation shows correct answers (within tolerance)

