"""SQLite tool for database access and schema introspection"""
import sqlite3
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Configuration constants (merged from config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "northwind.sqlite"
MAX_SQL_RESULTS = 100

# Type definitions (merged from analytics_types.py)
class Citation(BaseModel):
    """Citation for a source used in answering"""
    source_type: Literal["document", "database", "sql_query"]
    source: str = Field(description="Source identifier (file path, table name, or SQL query)")
    content: Optional[str] = Field(None, description="Relevant content excerpt")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class SQLiteTool:
    """Tool for querying the Northwind SQLite database"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.schema = self._get_schema()
    
    def _get_schema(self) -> Dict[str, List[str]]:
        """Get database schema information"""
        if not self.db_path.exists():
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables (handle quoted table names)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Handle table names with spaces by quoting them
            quoted_table = f'"{table}"' if ' ' in table else table
            try:
                cursor.execute(f"PRAGMA table_info({quoted_table})")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table] = columns
            except sqlite3.Error:
                # Fallback: try without quotes
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                schema[table] = columns
        
        conn.close()
        return schema
    
    def get_schema_string(self) -> str:
        """Get formatted schema string for LLM with canonical Northwind table information"""
        if not self.schema:
            return "Database schema not available"
        
        # Key tables with their canonical names and important columns
        key_tables_info = {
            "Orders": {
                "columns": ["OrderID", "CustomerID", "EmployeeID", "OrderDate", "RequiredDate", 
                           "ShippedDate", "ShipVia", "Freight", "ShipName", "ShipAddress", 
                           "ShipCity", "ShipRegion", "ShipPostalCode", "ShipCountry"],
                "description": "Customer orders with shipping information"
            },
            "Order Details": {
                "columns": ["OrderID", "ProductID", "UnitPrice", "Quantity", "Discount"],
                "description": "Line items for each order - CRITICAL: table name has a space and MUST be quoted as \"Order Details\" in SQL",
                "quoted_name": "\"Order Details\""
            },
            "Products": {
                "columns": ["ProductID", "ProductName", "SupplierID", "CategoryID", "UnitPrice",
                           "QuantityPerUnit", "UnitsInStock", "UnitsOnOrder", "ReorderLevel", "Discontinued"],
                "description": "Product catalog"
            },
            "Customers": {
                "columns": ["CustomerID", "CompanyName", "ContactName", "ContactTitle", "Address",
                           "City", "Region", "PostalCode", "Country", "Phone"],
                "description": "Customer information"
            },
            "Categories": {
                "columns": ["CategoryID", "CategoryName", "Description"],
                "description": "Product categories"
            },
            "Suppliers": {
                "columns": ["SupplierID", "CompanyName", "ContactName", "ContactTitle", "Address",
                           "City", "Region", "PostalCode", "Country", "Phone"],
                "description": "Supplier information"
            },
            "Employees": {
                "columns": ["EmployeeID", "LastName", "FirstName", "Title", "TitleOfCourtesy",
                           "BirthDate", "HireDate", "Address", "City", "Region", "PostalCode",
                           "Country", "HomePhone", "Extension", "ReportsTo"],
                "description": "Employee information"
            }
        }
        
        schema_parts = []
        schema_parts.append("=== CANONICAL NORTHWIND DATABASE SCHEMA ===")
        schema_parts.append("")
        schema_parts.append("CRITICAL: The table 'Order Details' has a space in its name.")
        schema_parts.append("ALWAYS use quotes when referencing it: \"Order Details\"")
        schema_parts.append("Example: SELECT * FROM \"Order Details\" WHERE OrderID = 10248;")
        schema_parts.append("")
        
        # Add key tables first with descriptions
        for table_name, info in key_tables_info.items():
            # Check if table exists in actual schema
            actual_table = None
            for actual_name in self.schema.keys():
                if actual_name.lower() == table_name.lower() or actual_name == info.get("quoted_name", table_name):
                    actual_table = actual_name
                    break
            
            if actual_table:
                actual_columns = self.schema[actual_table]
                display_name = info.get("quoted_name", table_name)
                # Compact format to keep prompts ≤1k tokens (constraint)
                schema_parts.append(f"Table: {display_name}")
                schema_parts.append(f"Key Columns: {', '.join(info['columns'][:10])}")  # Limit to 10 key columns
                schema_parts.append("")
        
        # Add any other tables not in the key list
        other_tables = [t for t in self.schema.keys() 
                       if not any(t.lower() == k.lower() for k in key_tables_info.keys())]
        if other_tables:
            schema_parts.append("=== OTHER TABLES ===")
            for table in other_tables:
                columns = self.schema[table]
                schema_parts.append(f"Table: {table}\nColumns: {', '.join(columns)}")
                schema_parts.append("")
        
        # Add relationship hints
        schema_parts.append("=== KEY RELATIONSHIPS ===")
        schema_parts.append("Orders.CustomerID -> Customers.CustomerID")
        schema_parts.append("Orders.OrderID -> \"Order Details\".OrderID")
        schema_parts.append("\"Order Details\".ProductID -> Products.ProductID")
        schema_parts.append("Products.CategoryID -> Categories.CategoryID")
        schema_parts.append("Products.SupplierID -> Suppliers.SupplierID")
        
        return "\n".join(schema_parts)
    
    def execute_query(self, query: str) -> dict:
        """Execute SQL query and return detailed results
        
        Returns:
            dict with keys:
                - results: List[Dict] or None
                - columns: List[str] or None
                - row_count: int
                - error: str or None
        """
        if not self.db_path.exists():
            return {
                "results": None,
                "columns": None,
                "row_count": 0,
                "error": "Database not found"
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
            rows = cursor.fetchall()
            results = [dict(row) for row in rows[:MAX_SQL_RESULTS]]
            
            conn.close()
            
            return {
                "results": results,
                "columns": columns,
                "row_count": len(results),
                "error": None
            }
        except sqlite3.Error as e:
            return {
                "results": None,
                "columns": None,
                "row_count": 0,
                "error": str(e)
            }
    
    def query_to_citation(self, query: str, results: Optional[List[Dict[str, Any]]]) -> Citation:
        """Create citation from SQL query
        
        Confidence: 1.0 if successful with rows, 0.5 if successful but empty, 0.0 if error
        """
        result_summary = f"Returned {len(results) if results else 0} rows"
        if results and len(results) > 0:
            sample = str(results[0])[:200]
            result_summary += f"\nSample: {sample}"
        
        # Confidence based on SQL success and row count
        if results is None:
            confidence = 0.0  # Error
        elif len(results) > 0:
            confidence = 1.0  # Success with rows
        else:
            confidence = 0.5  # Success but empty
        
        return Citation(
            source_type="sql_query",
            source=query,
            content=result_summary,
            confidence=confidence
        )
    
    def format_results(self, results: Optional[List[Dict[str, Any]]], columns: Optional[List[str]] = None) -> str:
        """Format query results as string with column information"""
        if results is None:
            return "Query execution failed"
        
        if not results:
            return "No results found"
        
        # Include column info if available
        header = ""
        if columns:
            header = f"Columns: {', '.join(columns)}\n"
        
        if len(results) == 1:
            return header + str(results[0])
        
        # Format as table
        if len(results) > 10:
            formatted = header + "\n".join([str(r) for r in results[:10]])
            formatted += f"\n... and {len(results) - 10} more rows"
        else:
            formatted = header + "\n".join([str(r) for r in results])
        
        return formatted

