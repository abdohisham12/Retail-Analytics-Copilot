"""Main entrypoint for Retail Analytics Copilot (Hybrid Agent)"""
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from agent.graph_hybrid import HybridRetailAnalyticsGraph

# Type definitions (merged from analytics_types.py)
class Citation(BaseModel):
    """Citation for a source used in answering"""
    source_type: Literal["document", "database", "sql_query"]
    source: str = Field(description="Source identifier (file path, table name, or SQL query)")
    content: Optional[str] = Field(None, description="Relevant content excerpt")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

class AnalyticsAnswer(BaseModel):
    """Typed, auditable answer with citations"""
    answer: str = Field(description="The answer to the question")
    citations: List[Citation] = Field(default_factory=list, description="Sources used")
    answer_type: Literal["rag", "sql", "hybrid"] = Field(description="Type of answer")
    sql_query: Optional[str] = Field(None, description="SQL query used if applicable")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")

# Output formatter functions (merged from agent/output_formatter.py)
def extract_tables_from_sql(sql_query: str) -> List[str]:
    """Extract table names from SQL query"""
    if not sql_query:
        return []
    patterns = [
        r'FROM\s+["\']?(\w+(?:\s+\w+)?)["\']?',
        r'JOIN\s+["\']?(\w+(?:\s+\w+)?)["\']?',
        r'UPDATE\s+["\']?(\w+(?:\s+\w+)?)["\']?',
        r'INTO\s+["\']?(\w+(?:\s+\w+)?)["\']?',
    ]
    tables = set()
    for pattern in patterns:
        matches = re.findall(pattern, sql_query, re.IGNORECASE)
        for match in matches:
            table = match.strip().strip('"').strip("'")
            if table:
                tables.add(table)
    if '"Order Details"' in sql_query or "'Order Details'" in sql_query:
        tables.add("Order Details")
    return sorted(list(tables))

def format_chunk_citation(source: str, chunk_id: str) -> str:
    """Format chunk citation as filename::chunkID"""
    clean_source = source
    if '#' in clean_source:
        clean_source = clean_source.split('#')[0]
    filename = Path(clean_source).stem
    if not chunk_id or chunk_id == "":
        if '#' in source:
            chunk_id = source.split('#')[-1]
    return f"{filename}::{chunk_id}"

def parse_final_answer(answer_text: str, format_hint: str) -> Any:
    """Parse answer to match format_hint exactly"""
    if not format_hint:
        return answer_text
    answer_text = answer_text.strip()
    if answer_text.startswith("```"):
        lines = answer_text.split("\n")
        if len(lines) > 1:
            answer_text = "\n".join(lines[1:-1]) if answer_text.endswith("```") else "\n".join(lines[1:])
    if format_hint == "int":
        try:
            match = re.search(r'-?\d+', answer_text)
            if match:
                return int(match.group())
            return int(float(answer_text))
        except (ValueError, TypeError):
            return 0
    elif format_hint == "float":
        try:
            match = re.search(r'-?\d+\.?\d*', answer_text)
            if match:
                value = float(match.group())
                return round(value, 2)
            return round(float(answer_text), 2)
        except (ValueError, TypeError):
            return 0.0
    elif format_hint.startswith("list[") or format_hint.startswith("{"):
        try:
            json_match = re.search(r'\[.*\]|\{.*\}', answer_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return json.loads(answer_text)
        except (json.JSONDecodeError, ValueError):
            return answer_text
    return answer_text

def generate_explanation(answer_type: str, has_sql: bool, has_rag: bool) -> str:
    """Generate explanation (<= 2 sentences)"""
    if answer_type == "hybrid":
        return f"Combined document knowledge with database query to provide the answer. Used both RAG retrieval and SQL execution."
    elif answer_type == "sql":
        return f"Queried the Northwind database to retrieve the requested data."
    else:
        return f"Retrieved relevant information from document corpus to answer the question."

def format_output_contract(
    question_id: str,
    final_answer: Any,
    sql_executed: str,
    confidence: float,
    answer_type: str,
    citations: List[Citation],
    chunk_ids: List[str],
    rag_citations: List[Dict]
) -> Dict[str, Any]:
    """Format output according to contract"""
    citation_list = []
    if sql_executed:
        tables = extract_tables_from_sql(sql_executed)
        citation_list.extend(tables)
    for i, chunk_id in enumerate(chunk_ids):
        if i < len(rag_citations):
            citation_dict = rag_citations[i]
            source = citation_dict.get("source", "unknown")
            citation_str = format_chunk_citation(source, chunk_id)
            citation_list.append(citation_str)
    seen = set()
    unique_citations = []
    for cit in citation_list:
        if cit not in seen:
            seen.add(cit)
            unique_citations.append(cit)
    explanation = generate_explanation(answer_type, bool(sql_executed), len(chunk_ids) > 0)
    return {
        "id": question_id,
        "final_answer": final_answer,
        "sql": sql_executed if sql_executed else "",
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "citations": unique_citations
    }


def print_answer(answer: AnalyticsAnswer):
    """Pretty print the answer with citations"""
    print("\n" + "="*80)
    print("ANSWER")
    print("="*80)
    print(answer.answer)
    print("\n" + "-"*80)
    print(f"Answer Type: {answer.answer_type.upper()}")
    print(f"Confidence: {answer.confidence:.2%}")
    
    if answer.sql_query:
        print(f"\nSQL Query Used:\n{answer.sql_query}")
    
    if answer.citations:
        print("\n" + "-"*80)
        print("CITATIONS")
        print("-"*80)
        for i, citation in enumerate(answer.citations, 1):
            print(f"\n[{i}] Source Type: {citation.source_type}")
            print(f"    Source: {citation.source}")
            if citation.content:
                print(f"    Content: {citation.content[:200]}...")
            print(f"    Confidence: {citation.confidence:.2%}")
    
    print("\n" + "="*80 + "\n")


def run_single_question(question: str, workflow: HybridRetailAnalyticsGraph, format_hint: str = "") -> dict:
    """Run a single question and return result as dict"""
    try:
        answer = workflow.query(question, format_hint=format_hint)
        return {
            "question": question,
            "format_hint": format_hint,
            "answer": answer.answer,
            "answer_type": answer.answer_type,
            "confidence": answer.confidence,
            "sql_query": answer.sql_query,
            "citations": [
                {
                    "source_type": c.source_type,
                    "source": c.source,
                    "confidence": c.confidence
                }
                for c in answer.citations
            ],
            "error": None
        }
    except Exception as e:
        return {
            "question": question,
            "format_hint": format_hint,
            "answer": None,
            "error": str(e)
        }


def main():
    """Main function with CLI contract"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retail Analytics Copilot - Hybrid Agent (RAG + SQL)"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to JSONL file with evaluation questions"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output JSONL file for evaluation results"
    )
    
    args = parser.parse_args()
    
    print("Initializing Retail Analytics Copilot...", flush=True)
    print("This may take a moment on first run (indexing documents)...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        workflow = HybridRetailAnalyticsGraph()
        print("[OK] Copilot ready!\n", flush=True)
        sys.stdout.flush()
        
        # Batch mode
        if args.batch:
            if not args.out:
                print("Error: --out is required when using --batch")
                sys.exit(1)
            
            print(f"Running batch evaluation on {args.batch}...")
            batch_file = Path(args.batch)
            if not batch_file.exists():
                print(f"Error: Batch file not found: {batch_file}")
                sys.exit(1)
            
            results = []
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        question_id = data.get("id", f"question_{line_num}")
                        question = data.get("question", data.get("input", ""))
                        format_hint = data.get("format_hint", "")
                        if not question:
                            print(f"Warning: Skipping line {line_num} - no question found")
                            continue
                        
                        print(f"\n[{line_num}] ID: {question_id}")
                        print(f"  Question: {question}")
                        if format_hint:
                            print(f"  Format hint: {format_hint}")
                        
                        # Run query
                        try:
                            answer, metadata = workflow.query(question, format_hint=format_hint)
                            
                            # Parse final_answer according to format_hint
                            parsed_answer = parse_final_answer(answer.answer, format_hint)
                            
                            # Get metadata from workflow state
                            sql_executed = metadata.get("sql_executed", answer.sql_query or "")
                            chunk_ids = metadata.get("chunk_ids", [])
                            rag_citations = metadata.get("rag_citations", [])
                            
                            # Format according to contract
                            output = format_output_contract(
                                question_id=question_id,
                                final_answer=parsed_answer,
                                sql_executed=sql_executed,
                                confidence=answer.confidence,
                                answer_type=answer.answer_type,
                                citations=answer.citations,
                                chunk_ids=chunk_ids,
                                rag_citations=rag_citations
                            )
                            
                            results.append(output)
                            print(f"  [OK] Answer: {parsed_answer}")
                            print(f"  Citations: {len(output['citations'])}")
                            
                        except Exception as e:
                            import traceback
                            error_trace = traceback.format_exc()
                            print(f"  [ERROR] {e}", file=sys.stderr, flush=True)
                            print(f"  [TRACEBACK] {error_trace}", file=sys.stderr, flush=True)
                            # Create error output
                            results.append({
                                "id": question_id,
                                "final_answer": None,
                                "sql": "",
                                "confidence": 0.0,
                                "explanation": f"Error processing question: {str(e)}",
                                "citations": []
                            })
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
            
            # Write results
            output_file = Path(args.out)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"\n[OK] Results written to {output_file}")
            print(f"  Total questions: {len(results)}")
            
            return
        
        # Single question mode
        if args.question:
            answer, _ = workflow.query(args.question)
            print_answer(answer)
            return
        
        # Interactive mode
        print("Enter your retail analytics questions (type 'exit' to quit):")
        print("-" * 80)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nProcessing...")
            try:
                answer, _ = workflow.query(question)
                print_answer(answer)
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

