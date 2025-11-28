"""Check the quality of outputs_hybrid.jsonl"""
import json
from pathlib import Path

lines = [json.loads(l) for l in Path("outputs_hybrid.jsonl").read_text().splitlines() if l.strip()]

print("=" * 60)
print("OUTPUT QUALITY ANALYSIS")
print("=" * 60)

print(f"\nTotal questions: {len(lines)}")

print("\n1. SQL STATUS (empty SQL is CORRECT for RAG-only questions):")
for line in lines:
    sql_status = "EMPTY ✓ (correct for RAG)" if not line['sql'] else f"PRESENT ({len(line['sql'])} chars)"
    print(f"   {line['id']}: {sql_status}")

print("\n2. CITATIONS QUALITY:")
for line in lines:
    citations = line['citations']
    # Check for noise (single letters, non-standard formats)
    noise = [c for c in citations if len(c) <= 1 or not (c.replace('_', '').replace(':', '').replace('-', '').isalnum() or '::' in c)]
    if noise:
        print(f"   {line['id']}: ⚠️  Potential noise: {noise}")
    else:
        print(f"   {line['id']}: ✓ Clean ({len(citations)} citations)")
        print(f"      Sample: {citations[:3]}")

print("\n3. CITATIONS BREAKDOWN:")
for line in lines:
    citations = line['citations']
    db_tables = [c for c in citations if c[0].isupper() and '::' not in c]
    doc_chunks = [c for c in citations if '::' in c]
    print(f"   {line['id']}:")
    print(f"      DB tables: {db_tables}")
    print(f"      Doc chunks: {doc_chunks}")

print("\n4. FORMAT VALIDATION:")
all_valid = True
for line in lines:
    required = ['id', 'final_answer', 'sql', 'confidence', 'explanation', 'citations']
    missing = [f for f in required if f not in line]
    if missing:
        print(f"   {line['id']}: ❌ Missing fields: {missing}")
        all_valid = False
    else:
        print(f"   {line['id']}: ✓ All fields present")

if all_valid:
    print("\n✅ All outputs have correct format!")
else:
    print("\n❌ Some outputs are missing required fields")

