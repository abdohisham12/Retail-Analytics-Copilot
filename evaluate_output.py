"""Evaluate outputs_hybrid.jsonl for format compliance and accuracy"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Database path
DB_PATH = Path("data/northwind.sqlite")

def load_outputs(file_path: str) -> List[Dict[str, Any]]:
    """Load output JSONL file"""
    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))
    return outputs

def check_format_compliance(output: Dict[str, Any], format_hint: str) -> Tuple[bool, List[str]]:
    """Check if output matches Output Contract format"""
    issues = []
    
    # Check required fields
    required_fields = ["id", "final_answer", "sql", "confidence", "explanation", "citations"]
    for field in required_fields:
        if field not in output:
            issues.append(f"Missing field: {field}")
    
    # Check final_answer matches format_hint
    if "final_answer" in output:
        answer = output["final_answer"]
        if format_hint == "int":
            if not isinstance(answer, int):
                issues.append(f"final_answer should be int, got {type(answer).__name__}")
        elif format_hint == "float":
            if not isinstance(answer, (int, float)):
                issues.append(f"final_answer should be float, got {type(answer).__name__}")
        elif format_hint.startswith("list[") or format_hint.startswith("{"):
            if not isinstance(answer, (list, dict)):
                issues.append(f"final_answer should be list or dict, got {type(answer).__name__}")
    
    # Check sql field
    if "sql" in output:
        if not isinstance(output["sql"], str):
            issues.append(f"sql should be string, got {type(output['sql']).__name__}")
    
    # Check confidence
    if "confidence" in output:
        conf = output["confidence"]
        if not isinstance(conf, (int, float)):
            issues.append(f"confidence should be number, got {type(conf).__name__}")
        elif conf < 0.0 or conf > 1.0:
            issues.append(f"confidence out of range [0.0, 1.0]: {conf}")
    
    # Check explanation (≤2 sentences)
    if "explanation" in output:
        expl = output["explanation"]
        if not isinstance(expl, str):
            issues.append(f"explanation should be string, got {type(expl).__name__}")
        else:
            sentences = expl.count('.') + expl.count('!') + expl.count('?')
            if sentences > 2:
                issues.append(f"explanation has {sentences} sentences (should be ≤2)")
    
    # Check citations format
    if "citations" in output:
        citations = output["citations"]
        if not isinstance(citations, list):
            issues.append(f"citations should be list, got {type(citations).__name__}")
        else:
            for cit in citations:
                if not isinstance(cit, str):
                    issues.append(f"citation should be string, got {type(cit).__name__}: {cit}")
                # Check chunk citation format (should be filename::chunk0, not filename::doc_0)
                if "::" in cit and "doc_" in cit:
                    issues.append(f"Citation uses doc_ format instead of chunk format: {cit}")
    
    return len(issues) == 0, issues

def get_expected_answers() -> Dict[str, Any]:
    """Query database to get expected answers for each question"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    expected = {}
    
    # Question 1: RAG-only - Return window for unopened Beverages
    # From product_policy.md: "Beverages unopened: 14 days"
    expected["rag_policy_beverages_return_days"] = 14
    
    # Question 2: Hybrid - Top category by quantity during Summer Beverages 1997 (1997-06-01 to 1997-06-30)
    cursor.execute("""
        SELECT c.CategoryName, SUM(od.Quantity) as total_qty
        FROM Orders o
        JOIN "Order Details" od ON o.OrderID = od.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Categories c ON p.CategoryID = c.CategoryID
        WHERE o.OrderDate >= '1997-06-01' AND o.OrderDate <= '1997-06-30'
        GROUP BY c.CategoryName
        ORDER BY total_qty DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result and result["CategoryName"]:
        expected["hybrid_top_category_qty_summer_1997"] = {
            "category": result["CategoryName"],
            "quantity": int(result["total_qty"] or 0)
        }
    
    # Question 3: Hybrid - AOV during Winter Classics 1997 (1997-12-01 to 1997-12-31)
    # AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)
    cursor.execute("""
        SELECT 
            SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) as aov
        FROM Orders o
        JOIN "Order Details" od ON o.OrderID = od.OrderID
        WHERE o.OrderDate >= '1997-12-01' AND o.OrderDate <= '1997-12-31'
    """)
    result = cursor.fetchone()
    if result:
        expected["hybrid_aov_winter_1997"] = round(result["aov"] or 0.0, 2)
    
    # Question 4: SQL-only - Top 3 products by revenue all-time
    cursor.execute("""
        SELECT p.ProductName, 
               SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue
        FROM "Order Details" od
        JOIN Products p ON od.ProductID = p.ProductID
        GROUP BY p.ProductID, p.ProductName
        ORDER BY revenue DESC
        LIMIT 3
    """)
    results = cursor.fetchall()
    expected["sql_top3_products_by_revenue_alltime"] = [
        {"product": r["ProductName"], "revenue": round(r["revenue"] or 0.0, 2)}
        for r in results
    ]
    
    # Question 5: Hybrid - Revenue from Beverages during Summer Beverages 1997
    cursor.execute("""
        SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue
        FROM Orders o
        JOIN "Order Details" od ON o.OrderID = od.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        JOIN Categories c ON p.CategoryID = c.CategoryID
        WHERE c.CategoryName = 'Beverages'
          AND o.OrderDate >= '1997-06-01' AND o.OrderDate <= '1997-06-30'
    """)
    result = cursor.fetchone()
    if result:
        expected["hybrid_revenue_beverages_summer_1997"] = round(result["revenue"] or 0.0, 2)
    
    # Question 6: Hybrid - Top customer by gross margin in 1997 (CostOfGoods = 70% of UnitPrice)
    # GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))
    # CostOfGoods = 0.7 * UnitPrice, so margin = (UnitPrice - 0.7*UnitPrice) * Quantity * (1-Discount)
    # = 0.3 * UnitPrice * Quantity * (1-Discount)
    cursor.execute("""
        SELECT 
            c.CompanyName,
            SUM(0.3 * od.UnitPrice * od.Quantity * (1 - od.Discount)) as margin
        FROM Orders o
        JOIN "Order Details" od ON o.OrderID = od.OrderID
        JOIN Customers c ON o.CustomerID = c.CustomerID
        WHERE o.OrderDate >= '1997-01-01' AND o.OrderDate < '1998-01-01'
        GROUP BY c.CustomerID, c.CompanyName
        ORDER BY margin DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result and result["CompanyName"]:
        expected["hybrid_best_customer_margin_1997"] = {
            "customer": result["CompanyName"],
            "margin": round(result["margin"] or 0.0, 2)
        }
    
    conn.close()
    return expected

def compare_accuracy(output: Dict[str, Any], expected: Any, format_hint: str) -> Tuple[bool, str]:
    """Compare output answer with expected answer"""
    answer = output.get("final_answer")
    
    if expected is None:
        return False, "No expected answer available"
    
    if format_hint == "int":
        if isinstance(answer, int) and answer == expected:
            return True, f"Correct: {answer}"
        return False, f"Expected {expected}, got {answer}"
    
    elif format_hint == "float":
        if isinstance(answer, (int, float)) and isinstance(expected, (int, float)):
            diff = abs(float(answer) - float(expected))
            if diff <= 0.01:  # ±0.01 tolerance
                return True, f"Correct: {answer} (expected {expected}, diff {diff:.4f})"
            return False, f"Expected {expected}, got {answer} (diff {diff:.4f} > 0.01)"
        return False, f"Type mismatch: expected {type(expected).__name__}, got {type(answer).__name__}"
    
    elif format_hint.startswith("{"):
        # Object comparison
        if isinstance(answer, dict) and isinstance(expected, dict):
            # Check if all expected keys match
            matches = True
            issues = []
            for key, exp_val in expected.items():
                if key not in answer:
                    matches = False
                    issues.append(f"Missing key: {key}")
                else:
                    ans_val = answer[key]
                    if isinstance(exp_val, float) and isinstance(ans_val, (int, float)):
                        if abs(float(ans_val) - float(exp_val)) > 0.01:
                            matches = False
                            issues.append(f"{key}: expected {exp_val}, got {ans_val}")
                    elif ans_val != exp_val:
                        matches = False
                        issues.append(f"{key}: expected {exp_val}, got {ans_val}")
            if matches:
                return True, f"Correct: {answer}"
            return False, f"Issues: {', '.join(issues)}"
        return False, f"Type mismatch: expected dict, got {type(answer).__name__}"
    
    elif format_hint.startswith("list["):
        # List of objects comparison
        if isinstance(answer, list) and isinstance(expected, list):
            if len(answer) != len(expected):
                return False, f"Length mismatch: expected {len(expected)}, got {len(answer)}"
            matches = True
            issues = []
            for i, (ans_item, exp_item) in enumerate(zip(answer, expected)):
                if isinstance(exp_item, dict) and isinstance(ans_item, dict):
                    for key, exp_val in exp_item.items():
                        if key not in ans_item:
                            matches = False
                            issues.append(f"Item {i}: missing key {key}")
                        else:
                            ans_val = ans_item[key]
                            if isinstance(exp_val, float) and isinstance(ans_val, (int, float)):
                                if abs(float(ans_val) - float(exp_val)) > 0.01:
                                    matches = False
                                    issues.append(f"Item {i}.{key}: expected {exp_val}, got {ans_val}")
                            elif ans_val != exp_val:
                                matches = False
                                issues.append(f"Item {i}.{key}: expected {exp_val}, got {ans_val}")
            if matches:
                return True, f"Correct: {answer}"
            return False, f"Issues: {', '.join(issues)}"
        return False, f"Type mismatch: expected list, got {type(answer).__name__}"
    
    return False, "Unknown format_hint type"

def main():
    """Main evaluation function"""
    output_file = "outputs_hybrid.jsonl"
    
    # Load expected answers first (doesn't require output file)
    print("Computing expected answers from database...")
    expected_answers = get_expected_answers()
    print(f"[OK] Computed expected answers for {len(expected_answers)} questions\n")
    
    # Show expected answers
    print("=" * 80)
    print("EXPECTED ANSWERS (from database)")
    print("=" * 80)
    for qid, expected in expected_answers.items():
        print(f"\n{qid}:")
        print(f"  Expected: {expected}")
    
    if not Path(output_file).exists():
        print(f"\n[ERROR] Output file not found: {output_file}")
        print("   Please run: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl")
        print("\n   Once the output file is generated, run this script again to evaluate format and accuracy.")
        return
    
    # Load outputs
    outputs = load_outputs(output_file)
    print(f"[OK] Loaded {len(outputs)} outputs from {output_file}\n")
    
    # Load input questions to get format_hints
    questions = {}
    with open("sample_questions_hybrid_eval.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q["id"]] = q.get("format_hint", "")
    
    # Evaluate each output
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    format_issues_total = 0
    accuracy_correct = 0
    accuracy_total = 0
    
    for output in outputs:
        qid = output.get("id", "unknown")
        format_hint = questions.get(qid, "")
        
        print(f"\nQuestion ID: {qid}")
        print("-" * 80)
        
        # Format compliance
        format_ok, format_issues = check_format_compliance(output, format_hint)
        if format_ok:
            print("[OK] Format: COMPLIANT")
        else:
            print("[ERROR] Format: ISSUES FOUND")
            for issue in format_issues:
                print(f"   - {issue}")
            format_issues_total += len(format_issues)
        
        # Accuracy
        if qid in expected_answers:
            accuracy_total += 1
            acc_ok, acc_msg = compare_accuracy(output, expected_answers[qid], format_hint)
            if acc_ok:
                print(f"[OK] Accuracy: {acc_msg}")
                accuracy_correct += 1
            else:
                print(f"[ERROR] Accuracy: {acc_msg}")
                print(f"   Expected: {expected_answers[qid]}")
                print(f"   Got: {output.get('final_answer')}")
        else:
            print("[WARNING] Accuracy: No expected answer available")
        
        # Show key fields
        print(f"   SQL: {output.get('sql', '')[:100]}..." if output.get('sql') else "   SQL: (empty)")
        print(f"   Citations: {len(output.get('citations', []))} items")
        print(f"   Confidence: {output.get('confidence', 0.0):.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Format Issues: {format_issues_total}")
    print(f"Accuracy: {accuracy_correct}/{accuracy_total} correct ({accuracy_correct/accuracy_total*100:.1f}%)" if accuracy_total > 0 else "Accuracy: N/A")

if __name__ == "__main__":
    main()

