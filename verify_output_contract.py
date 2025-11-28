"""Verify that format_output_contract produces correct Output Contract format"""
import json
from run_agent_hybrid import format_output_contract, parse_final_answer
from agent.graph_hybrid import Citation

# Test cases for each format_hint type
test_cases = [
    {
        "id": "test_int",
        "format_hint": "int",
        "answer_text": "14",
        "expected_type": int
    },
    {
        "id": "test_float",
        "format_hint": "float",
        "answer_text": "1234.56",
        "expected_type": float
    },
    {
        "id": "test_object",
        "format_hint": "{category:str, quantity:int}",
        "answer_text": '{"category": "Beverages", "quantity": 100}',
        "expected_type": dict
    },
    {
        "id": "test_list",
        "format_hint": "list[{product:str, revenue:float}]",
        "answer_text": '[{"product": "Product1", "revenue": 100.50}, {"product": "Product2", "revenue": 200.75}]',
        "expected_type": list
    }
]

print("=" * 80)
print("Verifying Output Contract Format")
print("=" * 80)
print()

all_passed = True

for test in test_cases:
    print(f"Test: {test['id']}")
    print(f"  Format hint: {test['format_hint']}")
    
    # Parse answer
    parsed = parse_final_answer(test['answer_text'], test['format_hint'])
    
    # Check type
    if isinstance(parsed, test['expected_type']):
        print(f"  ✅ Type correct: {type(parsed).__name__}")
    else:
        print(f"  ❌ Type mismatch: expected {test['expected_type'].__name__}, got {type(parsed).__name__}")
        all_passed = False
    
    # Format output
    output = format_output_contract(
        question_id=test['id'],
        final_answer=parsed,
        sql_executed="SELECT * FROM Orders" if "sql" in test['id'] else "",
        confidence=0.85,
        answer_type="hybrid",
        citations=[],
        chunk_ids=["doc_0", "doc_1"],
        rag_citations=[
            {"source": "docs/kpi_definitions.md#doc_0"},
            {"source": "docs/marketing_calendar.md#doc_1"}
        ]
    )
    
    # Verify required fields
    required = ["id", "final_answer", "sql", "confidence", "explanation", "citations"]
    missing = [f for f in required if f not in output]
    if missing:
        print(f"  ❌ Missing fields: {missing}")
        all_passed = False
    else:
        print(f"  ✅ All required fields present")
    
    # Verify field types
    checks = [
        ("id", str),
        ("sql", str),
        ("confidence", (int, float)),
        ("explanation", str),
        ("citations", list)
    ]
    
    for field, expected_type in checks:
        if not isinstance(output[field], expected_type):
            print(f"  ❌ {field} wrong type: expected {expected_type}, got {type(output[field])}")
            all_passed = False
    
    # Verify citations format (should be filename::chunk0, not filename::doc_0)
    for cit in output['citations']:
        if "::doc_" in cit:
            print(f"  ❌ Citation uses doc_ format: {cit}")
            all_passed = False
        elif "::chunk" in cit:
            print(f"  ✅ Citation format correct: {cit}")
    
    # Verify confidence range
    if not (0.0 <= output['confidence'] <= 1.0):
        print(f"  ❌ Confidence out of range: {output['confidence']}")
        all_passed = False
    
    # Verify explanation length (≤2 sentences)
    sentences = output['explanation'].count('.') + output['explanation'].count('!') + output['explanation'].count('?')
    if sentences > 2:
        print(f"  ❌ Explanation has {sentences} sentences (should be ≤2)")
        all_passed = False
    else:
        print(f"  ✅ Explanation length OK ({sentences} sentences)")
    
    # Show sample output
    print(f"  Sample output:")
    print(f"    {json.dumps(output, indent=2)[:200]}...")
    print()

print("=" * 80)
if all_passed:
    print("✅ All Output Contract format checks PASSED!")
    print("\nThe format_output_contract function will produce correct output")
    print("when the batch evaluation runs.")
else:
    print("❌ Some format checks FAILED")
    print("Please review the issues above.")
print("=" * 80)

