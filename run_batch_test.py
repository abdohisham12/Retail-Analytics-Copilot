"""Run batch evaluation with progress tracking"""
import sys
import json
from pathlib import Path

print("=" * 80)
print("Retail Analytics Copilot - Batch Evaluation")
print("=" * 80)
print()

# Initialize
print("Step 1: Initializing agent...")
print("(This may take 1-2 minutes on first run for document indexing)")
sys.stdout.flush()

try:
    from agent.graph_hybrid import HybridRetailAnalyticsGraph
    from run_agent_hybrid import format_output_contract, parse_final_answer
    
    workflow = HybridRetailAnalyticsGraph(enable_tracing=False)
    print("✅ Agent initialized!\n")
    sys.stdout.flush()
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load questions
print("Step 2: Loading questions...")
batch_file = Path("sample_questions_hybrid_eval.jsonl")
if not batch_file.exists():
    print(f"❌ Batch file not found: {batch_file}")
    sys.exit(1)

questions = []
with open(batch_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            questions.append(data)
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing line {line_num}: {e}")
            continue

print(f"✅ Loaded {len(questions)} questions\n")
sys.stdout.flush()

# Process questions
print("Step 3: Processing questions...")
print("=" * 80)
results = []

for i, q_data in enumerate(questions, 1):
    qid = q_data.get("id", f"question_{i}")
    question = q_data.get("question", "")
    format_hint = q_data.get("format_hint", "")
    
    print(f"\n[{i}/{len(questions)}] {qid}")
    print(f"  Question: {question[:80]}...")
    if format_hint:
        print(f"  Format: {format_hint}")
    sys.stdout.flush()
    
    try:
        # Run query
        answer, metadata = workflow.query(question, format_hint=format_hint)
        
        # Parse final_answer
        parsed_answer = parse_final_answer(answer.answer, format_hint)
        
        # Get metadata
        sql_executed = metadata.get("sql_executed", answer.sql_query or "")
        chunk_ids = metadata.get("chunk_ids", [])
        rag_citations = metadata.get("rag_citations", [])
        
        # Format output
        output = format_output_contract(
            question_id=qid,
            final_answer=parsed_answer,
            sql_executed=sql_executed,
            confidence=answer.confidence,
            answer_type=answer.answer_type,
            citations=answer.citations,
            chunk_ids=chunk_ids,
            rag_citations=rag_citations
        )
        
        results.append(output)
        print(f"  ✅ Answer: {parsed_answer}")
        print(f"  ✅ Citations: {len(output['citations'])}")
        print(f"  ✅ Confidence: {output['confidence']:.4f}")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        # Create error output
        results.append({
            "id": qid,
            "final_answer": None,
            "sql": "",
            "confidence": 0.0,
            "explanation": f"Error processing question: {str(e)}",
            "citations": []
        })
        sys.stdout.flush()

# Write results
print("\n" + "=" * 80)
print("Step 4: Writing results...")
output_file = Path("outputs_hybrid.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"✅ Results written to {output_file}")
print(f"   Total questions processed: {len(results)}")
print("=" * 80)
print("\n✅ Batch evaluation complete!")
print(f"\nNext step: Run 'python evaluate_output.py' to check format and accuracy")

