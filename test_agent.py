"""Simple test script for the agent"""
import sys
import json
from pathlib import Path

print("=" * 80)
print("Testing Retail Analytics Copilot")
print("=" * 80)
print()

# Test 1: Simple RAG question
print("Test 1: RAG-only question")
print("-" * 80)
try:
    from agent.graph_hybrid import HybridRetailAnalyticsGraph
    
    print("Initializing agent (this may take a moment on first run)...")
    sys.stdout.flush()
    
    graph = HybridRetailAnalyticsGraph(enable_tracing=False)
    print("✅ Agent initialized!\n")
    
    question = "According to the product policy, what is the return window (days) for unopened Beverages?"
    print(f"Question: {question}")
    print(f"Format hint: int")
    print("Processing...")
    sys.stdout.flush()
    
    answer, metadata = graph.query(question, format_hint="int")
    
    print(f"\n✅ Answer: {answer.answer}")
    print(f"   Type: {answer.answer_type}")
    print(f"   Confidence: {answer.confidence:.4f}")
    print(f"   Citations: {len(answer.citations)}")
    if answer.citations:
        for cit in answer.citations[:3]:
            print(f"     - {cit.source}")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: SQL question
print("Test 2: SQL-only question")
print("-" * 80)
try:
    question = "How many orders are in the database?"
    print(f"Question: {question}")
    print("Processing...")
    sys.stdout.flush()
    
    answer, metadata = graph.query(question, format_hint="int")
    
    print(f"\n✅ Answer: {answer.answer}")
    print(f"   Type: {answer.answer_type}")
    print(f"   SQL: {answer.sql_query[:100] if answer.sql_query else 'None'}...")
    print(f"   Confidence: {answer.confidence:.4f}")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("Test completed!")
print("=" * 80)

