"""Utilities for schema introspection and safe SQL execution."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class SQLiteTool:
    """Wraps SQLite access with schema helpers and execution safeguards."""

    COMMON_JOINS: Dict[str, str] = {
        "orders_with_details": 'Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID',
        "details_with_products": '"Order Details" od JOIN Products p ON od.ProductID = p.ProductID',
        "products_with_categories": "Products p JOIN Categories c ON p.CategoryID = c.CategoryID",
        "orders_with_customers": "Orders o JOIN Customers c ON o.CustomerID = c.CustomerID",
    }

    LEGACY_YEAR_OFFSET = 15  # Align 1990s campaign dates with 2010s data drift.

    def __init__(self, db_path: Path | str = Path("data/northwind.sqlite")) -> None:
        self.db_path = Path(db_path)
        self._schema_cache: Optional[Dict[str, List[Dict[str, str]]]] = None
        self._order_date_bounds: Optional[Tuple[str, str]] = None

    # ------------------------------------------------------------------ Schema
    def list_tables(self) -> List[str]:
        schema = self.get_schema()
        return sorted(schema.keys())

    def get_schema(self, refresh: bool = False) -> Dict[str, List[Dict[str, str]]]:
        if self._schema_cache is not None and not refresh:
            return self._schema_cache

        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        schema: Dict[str, List[Dict[str, str]]] = {}
        with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
            cur.execute(query)
            tables = [row[0] for row in cur.fetchall()]
            for table in tables:
                pragma = f'PRAGMA table_info("{table}")'
                cur.execute(pragma)
                columns = [
                    {"name": col[1], "type": col[2], "notnull": bool(col[3]), "pk": bool(col[5])}
                    for col in cur.fetchall()
                ]
                schema[table] = columns
        self._schema_cache = schema
        return schema

    def describe_table(self, table: str) -> List[Dict[str, str]]:
        schema = self.get_schema()
        if table not in schema:
            raise KeyError(f"Unknown table '{table}'")
        return schema[table]

    # ------------------------------------------------------------ Execution API
    def execute(
        self, sql: str, params: Sequence | None = None, enforce_safe: bool = True
    ) -> Dict[str, object]:
        sanitized_sql = sql.strip()
        if enforce_safe:
            self._assert_safe_sql(sanitized_sql)

        with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
            try:
                traced_tables = self._trace_tables(conn, sanitized_sql)
                cur.execute(sanitized_sql, params or [])
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                tables = traced_tables or self._infer_tables(sanitized_sql)
                result = {
                    "columns": columns,
                    "rows": [tuple(row) for row in rows],
                    "row_dicts": [dict(row) for row in rows],
                    "tables": tables,
                    "error": None,
                }
            except Exception as exc:  # sqlite3.Error and others
                result = {"columns": [], "rows": [], "row_dicts": [], "tables": [], "error": str(exc)}
        return result

    # ----------------------------------------------------------- Legacy helpers
    def get_order_date_bounds(self) -> Tuple[str, str]:
        """Return (min_date, max_date) as YYYY-MM-DD strings for Orders."""
        if self._order_date_bounds is None:
            with closing(self._connect()) as conn, closing(conn.cursor()) as cur:
                cur.execute("SELECT MIN(date(OrderDate)), MAX(date(OrderDate)) FROM Orders")
                row = cur.fetchone() or ("1997-01-01", "1997-12-31")
                min_date = row[0] or "1997-01-01"
                max_date = row[1] or min_date
                self._order_date_bounds = (min_date, max_date)
        return self._order_date_bounds

    def legacy_window(self, start_legacy: str, end_legacy: str) -> Tuple[str, str]:
        """Map a 1990s marketing window into actual DB dates, preserving duration."""
        min_actual_str, max_actual_str = self.get_order_date_bounds()
        min_actual = datetime.strptime(min_actual_str, "%Y-%m-%d")
        max_actual = datetime.strptime(max_actual_str, "%Y-%m-%d")

        legacy_start = datetime.strptime(start_legacy, "%Y-%m-%d")
        legacy_end = datetime.strptime(end_legacy, "%Y-%m-%d")
        duration = legacy_end - legacy_start

        shifted_start = legacy_start.replace(year=legacy_start.year + self.LEGACY_YEAR_OFFSET)
        shifted_end = shifted_start + duration

        if shifted_start < min_actual:
            shifted_start = min_actual
            shifted_end = shifted_start + duration
        if shifted_end > max_actual:
            shifted_end = max_actual
            shifted_start = max(min_actual, shifted_end - duration)

        return shifted_start.strftime("%Y-%m-%d"), shifted_end.strftime("%Y-%m-%d")

    # -------------------------------------------------------------- Join helper
    def get_join_snippet(self, key: str) -> str:
        if key not in self.COMMON_JOINS:
            raise KeyError(f"Unknown join template '{key}'")
        return self.COMMON_JOINS[key]

    # -------------------------------------------------------------- Internals
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _assert_safe_sql(self, sql: str) -> None:
        lowered = sql.lower()
        forbidden = ("drop ", "delete ", "update ", "insert ", "alter ", "pragma ")
        if any(tok in lowered for tok in forbidden):
            raise ValueError("Destructive SQL statements are not allowed.")
        if sql.count(";") > 1:
            raise ValueError("Multiple statements are not permitted.")

    def _infer_tables(self, sql: str) -> List[str]:
        schema_tables = self.get_schema().keys()
        lowered_sql = sql.lower()
        matched: List[str] = []
        for table in schema_tables:
            pattern = rf"\b{re.escape(table.lower())}\b"
            if re.search(pattern, lowered_sql):
                matched.append(table)
        return matched

    def _trace_tables(self, conn: sqlite3.Connection, sql: str) -> List[str]:
        """Use EXPLAIN QUERY PLAN to capture the actual tables touched."""
        sanitized = sql.rstrip(";")
        try:
            plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {sanitized}").fetchall()
        except sqlite3.Error:
            return []
        pattern = re.compile(r'TABLE (?:"([^"]+)"|\'([^\']+)\'|([A-Za-z0-9_ ]+))')
        ordered: List[str] = []
        for row in plan_rows:
            detail = row[-1] if row else ""
            if not isinstance(detail, str):
                continue
            for match in pattern.findall(detail):
                table = next((part for part in match if part), "").strip()
                if table and table not in ordered:
                    ordered.append(table)
        return ordered
