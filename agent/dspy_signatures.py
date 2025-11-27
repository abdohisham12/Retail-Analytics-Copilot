"""DSPy Signatures and Modules for Retail Analytics Copilot"""
import dspy
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configuration constants (merged from config.py)
OLLAMA_MODEL = "phi3.5-mini-instruct"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Type definitions (merged from analytics_types.py)
class QueryIntent(BaseModel):
    """Detected intent of the user query"""
    requires_sql: bool = Field(description="Whether SQL query is needed")
    requires_rag: bool = Field(description="Whether document search is needed")
    sql_tables: List[str] = Field(default_factory=list, description="Relevant database tables")
    keywords: List[str] = Field(default_factory=list, description="Key terms for RAG search")


class OllamaLM(dspy.LM):
    """DSPy-compatible Ollama language model"""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        super().__init__(model)
        self.model = model
        self.base_url = base_url
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("ollama package required. Install with: pip install ollama")
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Make request to Ollama (local, no external network calls)
        
        Constraint: Prompts should be ≤1k tokens (enforced by caller)
        """
        try:
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # Local Ollama server - no external network calls
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response.get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.basic_request(prompt, **kwargs)
    
    def request(self, prompt: str, **kwargs) -> list:
        """DSPy-compatible request method"""
        response = self.basic_request(prompt, **kwargs)
        return [{"content": response}]


# DSPy Signatures
class QueryRouter(dspy.Signature):
    """Route query to appropriate handler (RAG, SQL, or both)"""
    question: str = dspy.InputField(desc="User's retail analytics question")
    requires_sql: bool = dspy.OutputField(desc="Whether SQL query is needed")
    requires_rag: bool = dspy.OutputField(desc="Whether document search is needed")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for routing decision")


class SQLQueryGenerator(dspy.Signature):
    """Generate SQL query from natural language question"""
    question: str = dspy.InputField(desc="User's question")
    schema: str = dspy.InputField(desc="Database schema information")
    sql_query: str = dspy.OutputField(desc="Valid SQL query to answer the question. IMPORTANT: The table 'Order Details' has a space and must be quoted as \"Order Details\" in SQL statements. Prefer joins: Orders + \"Order Details\" + Products. Revenue formula: SUM(UnitPrice * Quantity * (1 - Discount)) from \"Order Details\".")


class ConstraintPlanner(dspy.Signature):
    """Extract constraints from question and documents (dates, KPIs, categories, entities)"""
    question: str = dspy.InputField(desc="User's question")
    rag_context: str = dspy.InputField(desc="Context from documents")
    date_ranges: str = dspy.OutputField(desc="Extracted date ranges (e.g., '1997-06-01 to 1997-06-30')")
    kpi_formulas: str = dspy.OutputField(desc="Extracted KPI formulas or definitions")
    categories: str = dspy.OutputField(desc="Extracted product categories or entities")
    other_constraints: str = dspy.OutputField(desc="Other constraints or requirements")


class AnswerSynthesizer(dspy.Signature):
    """Synthesize final answer from RAG context and SQL results"""
    question: str = dspy.InputField(desc="Original question")
    rag_context: str = dspy.InputField(desc="Context from documents")
    sql_results: str = dspy.InputField(desc="Results from SQL query")
    format_hint: str = dspy.InputField(desc="Expected output format (e.g., 'int', 'float', '{category:str, quantity:int}')")
    answer: str = dspy.OutputField(desc="Answer matching format_hint exactly, with citations")


class SQLRepair(dspy.Signature):
    """Repair a SQL query that failed to execute"""
    original_query: str = dspy.InputField(desc="The SQL query that failed")
    error_message: str = dspy.InputField(desc="Error message from database")
    schema: str = dspy.InputField(desc="Database schema information")
    repaired_query: str = dspy.OutputField(desc="Corrected SQL query")


# DSPy Modules
class QueryRoutingModule(dspy.Module):
    """DSPy module for query routing"""
    
    def __init__(self):
        super().__init__()
        self.router = dspy.ChainOfThought(QueryRouter)
    
    def forward(self, question: str) -> QueryIntent:
        """Route query to determine intent"""
        result = self.router(question=question)
        
        # Handle boolean conversion (DSPy might return strings)
        def to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'requires_sql', 'requires_rag')
            return bool(value)
        
        requires_sql = to_bool(getattr(result, 'requires_sql', False))
        requires_rag = to_bool(getattr(result, 'requires_rag', True))
        
        # Default: if question mentions data/numbers/query, likely needs SQL
        if not requires_sql and not requires_rag:
            sql_keywords = ['show', 'find', 'list', 'count', 'total', 'sum', 'average', 'top', 'best', 'worst']
            if any(kw in question.lower() for kw in sql_keywords):
                requires_sql = True
            else:
                requires_rag = True
        
        return QueryIntent(
            requires_sql=requires_sql,
            requires_rag=requires_rag,
            sql_tables=[],
            keywords=[]
        )
    
    def __call__(self, question: str) -> QueryIntent:
        """Allow calling module directly"""
        return self.forward(question)


class SQLGenerationModule(dspy.Module):
    """DSPy module for SQL generation"""
    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(SQLQueryGenerator)
    
    def forward(self, question: str, schema: str) -> str:
        """Generate SQL query"""
        result = self.generator(question=question, schema=schema)
        return result.sql_query if hasattr(result, 'sql_query') else ""


class SQLRepairModule(dspy.Module):
    """DSPy module for SQL query repair"""
    
    def __init__(self):
        super().__init__()
        self.repair = dspy.ChainOfThought(SQLRepair)
    
    def forward(self, original_query: str, error_message: str, schema: str) -> str:
        """Repair a failed SQL query"""
        result = self.repair(
            original_query=original_query,
            error_message=error_message,
            schema=schema
        )
        return result.repaired_query if hasattr(result, 'repaired_query') else original_query


class ConstraintPlannerModule(dspy.Module):
    """DSPy module for constraint extraction"""
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(ConstraintPlanner)
    
    def forward(self, question: str, rag_context: str) -> dict:
        """Extract constraints from question and context"""
        result = self.planner(
            question=question,
            rag_context=rag_context or "No document context available"
        )
        return {
            "date_ranges": getattr(result, 'date_ranges', ''),
            "kpi_formulas": getattr(result, 'kpi_formulas', ''),
            "categories": getattr(result, 'categories', ''),
            "other_constraints": getattr(result, 'other_constraints', '')
        }


class AnswerSynthesisModule(dspy.Module):
    """DSPy module for answer synthesis"""
    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(AnswerSynthesizer)
    
    def forward(self, question: str, rag_context: str, sql_results: str, format_hint: str = "") -> str:
        """Synthesize final answer matching format_hint"""
        result = self.synthesizer(
            question=question,
            rag_context=rag_context or "No document context available",
            sql_results=sql_results or "No SQL results available",
            format_hint=format_hint or "text"
        )
        return result.answer if hasattr(result, 'answer') else "Unable to generate answer"

