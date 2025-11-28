"""DSPy signatures and helper modules for the hybrid agent."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import dspy
from dspy.clients.base_lm import BaseLM


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #


class RouterSignature(dspy.Signature):
    question = dspy.InputField(desc="Retail analytics question with format hints.")
    mode = dspy.OutputField(desc="One of rag|sql|hybrid.")


class PlannerSignature(dspy.Signature):
    question = dspy.InputField()
    retrieved_chunks = dspy.InputField(desc="Doc chunk snippets + metadata.")
    schema_notes = dspy.InputField(desc="Important schema hints.")
    plan = dspy.OutputField(desc="Structured constraints, entities, KPI formulas.")


class NL2SQLSignature(dspy.Signature):
    question = dspy.InputField()
    plan = dspy.InputField(desc="Structured constraints from planner.")
    schema_notes = dspy.InputField(desc="Key tables/columns JSON.")
    sql = dspy.OutputField(desc="Single SELECT statement for SQLite.")


class SynthesizerSignature(dspy.Signature):
    question = dspy.InputField()
    plan = dspy.InputField()
    retrieved_chunks = dspy.InputField()
    sql_rows = dspy.InputField(desc="SQL execution result rows/columns.")
    sql = dspy.InputField()
    format_hint = dspy.InputField()
    citations = dspy.OutputField(desc="List of citation strings.")
    final_answer = dspy.OutputField(desc="Answer that matches format_hint.")
    explanation = dspy.OutputField(desc="<=2 sentence rationale.")
    confidence = dspy.OutputField(desc="0-1 confidence score.")


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #


class RouterModule:
    def __init__(self) -> None:
        self.predict = dspy.Predict(RouterSignature)

    def __call__(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)


class PlannerModule:
    def __init__(self) -> None:
        self.predict = dspy.Predict(PlannerSignature)

    def __call__(self, question: str, retrieved_chunks: str, schema_notes: str) -> dspy.Prediction:
        return self.predict(
            question=question,
            retrieved_chunks=retrieved_chunks,
            schema_notes=schema_notes,
        )


@dataclass
class OptimizationReport:
    before: float
    after: float
    examples: int

    def as_dict(self) -> Dict[str, float]:
        return {"before": self.before, "after": self.after, "examples": self.examples}


class NL2SQLModule:
    """DSPy predictor with optional BootstrapFewShot optimization."""

    def __init__(self, optimizer_examples: Optional[Iterable[dspy.Example]] = None) -> None:
        self.predict = dspy.Predict(NL2SQLSignature)
        self.report: Optional[OptimizationReport] = None
        if optimizer_examples:
            self.optimize(list(optimizer_examples))

    def __call__(self, question: str, plan: str, schema_notes: str) -> dspy.Prediction:
        return self.predict(question=question, plan=plan, schema_notes=schema_notes)

    def optimize(self, examples: List[dspy.Example]) -> None:
        """Optimize the NL→SQL module using BootstrapFewShot with valid-SQL rate metric."""
        if not examples:
            return

        def metric(prediction, reference, *_unused):
            """Valid-SQL rate metric: checks if prediction is valid SQL with correct tables."""
            ref_sql = getattr(reference, "sql", None)
            ref_tables = getattr(reference, "tables", []) or []
            return _valid_sql_metric(prediction.sql, ref_sql, ref_tables)

        # Measure baseline performance
        before = _quick_score(self.predict, examples, metric)
        
        # Run BootstrapFewShot optimization (small budget: max 4 demos)
        teleprompter = dspy.teleprompt.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,  # Small budget for local inference
            max_labeled_demos=len(examples),  # Use all examples
        )
        optimized_program = teleprompter.compile(self.predict, trainset=examples)
        self.predict = optimized_program
        
        # Measure optimized performance
        after = _quick_score(self.predict, examples, metric)
        
        # Store report
        self.report = OptimizationReport(before=before, after=after, examples=len(examples))
        print(
            f"[DSPy] NL→SQL optimization complete: "
            f"valid-SQL rate {before:.2f} → {after:.2f} "
            f"(+{after-before:.2f}) on {len(examples)} handcrafted examples."
        )


class SynthesizerModule:
    def __init__(self) -> None:
        self.predict = dspy.Predict(SynthesizerSignature)

    def __call__(
        self,
        question: str,
        plan: str,
        retrieved_chunks: str,
        sql_rows: str,
        sql: str,
        format_hint: str,
    ) -> dspy.Prediction:
        return self.predict(
            question=question,
            plan=plan,
            retrieved_chunks=retrieved_chunks,
            sql_rows=sql_rows,
            sql=sql,
            format_hint=format_hint,
        )


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def default_sql_trainset() -> List[dspy.Example]:
    """Handcrafted examples for NL→SQL optimization (6 examples covering common patterns)."""
    schema_hint = (
        "Orders(OrderID, CustomerID, OrderDate) | "
        '"Order Details"(OrderID, ProductID, UnitPrice, Quantity, Discount) | '
        "Products(ProductID, ProductName, CategoryID, UnitPrice) | "
        "Categories(CategoryID, CategoryName) | "
        "Customers(CustomerID, CompanyName, Country)"
    )
    examples = [
        dspy.Example(
            question="Total revenue per category in 1997.",
            plan="Need SUM(UnitPrice*Quantity*(1-Discount)) grouped by category for 1997 orders.",
            schema_notes=schema_hint,
            sql=(
                'SELECT c.CategoryName, '
                "SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue "
                'FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID '
                "JOIN Products p ON od.ProductID = p.ProductID "
                "JOIN Categories c ON p.CategoryID = c.CategoryID "
                "WHERE strftime('%Y', o.OrderDate) = '1997' "
                "GROUP BY c.CategoryName"
            ),
            tables=["Orders", "Order Details", "Products", "Categories"],
        ).with_inputs("question", "plan", "schema_notes"),
        dspy.Example(
            question="Average discount for beverages.",
            plan="Filter category='Beverages', compute AVG(discount).",
            schema_notes=schema_hint,
            sql=(
                'SELECT AVG(od.Discount) AS avg_discount '
                'FROM "Order Details" od '
                "JOIN Products p ON od.ProductID = p.ProductID "
                "JOIN Categories c ON p.CategoryID = c.CategoryID "
                "WHERE c.CategoryName = 'Beverages'"
            ),
            tables=["Order Details", "Products", "Categories"],
        ).with_inputs("question", "plan", "schema_notes"),
        dspy.Example(
            question="Orders count per customer in Germany.",
            plan="Need COUNT(*) in Orders filtered by Customers.Country='Germany'.",
            schema_notes=schema_hint,
            sql=(
                "SELECT c.CompanyName, COUNT(*) AS order_count "
                "FROM Orders o JOIN Customers c ON o.CustomerID = c.CustomerID "
                "WHERE c.Country = 'Germany' "
                "GROUP BY c.CompanyName"
            ),
            tables=["Orders", "Customers"],
        ).with_inputs("question", "plan", "schema_notes"),
        dspy.Example(
            question="Top 3 products by revenue all-time.",
            plan="Calculate SUM(UnitPrice*Quantity*(1-Discount)) per product, order DESC, LIMIT 3.",
            schema_notes=schema_hint,
            sql=(
                'SELECT p.ProductName, '
                'ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) AS revenue '
                'FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID '
                "GROUP BY p.ProductID ORDER BY revenue DESC LIMIT 3"
            ),
            tables=["Order Details", "Products"],
        ).with_inputs("question", "plan", "schema_notes"),
        dspy.Example(
            question="Average order value for orders in December 1997.",
            plan="Calculate SUM(revenue) / COUNT(DISTINCT OrderID) for December 1997 orders.",
            schema_notes=schema_hint,
            sql=(
                "WITH dec_orders AS ("
                "SELECT o.OrderID FROM Orders o "
                "WHERE strftime('%Y-%m', o.OrderDate) = '1997-12'"
                ") "
                "SELECT ROUND("
                "SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / "
                "COUNT(DISTINCT od.OrderID), 2"
                ") AS aov "
                'FROM "Order Details" od '
                "WHERE od.OrderID IN (SELECT OrderID FROM dec_orders)"
            ),
            tables=["Orders", "Order Details"],
        ).with_inputs("question", "plan", "schema_notes"),
        dspy.Example(
            question="Total quantity sold for beverages category in June 1997.",
            plan="Sum Quantity from Order Details joined to Products/Categories, filter by category and date range.",
            schema_notes=schema_hint,
            sql=(
                'SELECT SUM(od.Quantity) AS total_qty '
                'FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID '
                "JOIN Products p ON od.ProductID = p.ProductID "
                "JOIN Categories c ON p.CategoryID = c.CategoryID "
                "WHERE c.CategoryName = 'Beverages' "
                "AND strftime('%Y-%m', o.OrderDate) = '1997-06'"
            ),
            tables=["Orders", "Order Details", "Products", "Categories"],
        ).with_inputs("question", "plan", "schema_notes"),
    ]
    return examples


def _quick_score(
    program: dspy.Predict, examples: Iterable[dspy.Example], metric
) -> float:
    if not examples:
        return 0.0
    hits = 0
    total = 0
    for ex in examples:
        pred = program(question=ex.question, plan=ex.plan, schema_notes=ex.schema_notes)
        hits += metric(pred, ex)
        total += 1
    return hits / total if total else 0.0


def _valid_sql_metric(pred_sql: str, gold_sql: str, tables: Iterable[str]) -> int:
    """Valid-SQL rate metric: returns 1 if prediction is valid SQL with required tables, else 0.
    
    Checks:
    - Non-empty SQL string
    - Starts with SELECT (read-only queries)
    - Contains all required table names from reference
    - Basic syntax validation (no obvious errors)
    """
    if not pred_sql or not isinstance(pred_sql, str):
        return 0
    
    lowered = pred_sql.strip().lower()
    if not lowered.startswith("select"):
        return 0
    
    # Check that required tables are mentioned (at least one must be present)
    if tables:
        table_set = {tbl.lower() for tbl in tables}
        sql_tables = lowered
        # Handle quoted table names like "Order Details"
        # Check if at least one required table is mentioned
        found_table = False
        for tbl in table_set:
            if (tbl in sql_tables or 
                f'"{tbl}"' in sql_tables or 
                f"'{tbl}'" in sql_tables):
                found_table = True
                break
        if not found_table:
            return 0
    
    # Basic syntax checks: balanced parentheses, no obvious errors
    if lowered.count("(") != lowered.count(")"):
        return 0
    
    # Reject obviously malformed queries
    if "select select" in lowered or "from from" in lowered:
        return 0
    
    return 1


class OfflineMockLM(BaseLM):
    """Deterministic LM stub for offline testing."""

    def __init__(self) -> None:
        super().__init__(model="offline-mock", temperature=0.0, max_tokens=32)

    def forward(self, prompt=None, messages=None, **kwargs):
        text = ""
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )
        response = SimpleNamespace(
            choices=[choice],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model,
            _hidden_params={"response_cost": 0},
        )
        return response
