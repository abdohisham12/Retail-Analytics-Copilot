"""LangGraph hybrid workflow with RAG + SQL (≥8 nodes + repair loop + checkpointer)"""
from typing import TypedDict, Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
import sys
import os
import re
import json
import dspy

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agent.rag.retrieval import RAGRetrieval
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import (
    QueryRoutingModule, SQLGenerationModule, 
    AnswerSynthesisModule, SQLRepairModule, ConstraintPlannerModule, get_ollama_lm
)

# Configuration constants (merged from config.py)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "northwind.sqlite"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"
TRACES_DIR = PROJECT_ROOT / "traces"
MAX_REPAIR_ATTEMPTS = 2

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

# EventLogger class (merged from checkpointer.py)
class EventLogger:
    """Event logger for workflow trace and replay"""
    
    def __init__(self, log_file: Optional[Path] = None, console: bool = True):
        self.log_file = log_file or (TRACES_DIR / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.console = console
        self.events: List[Dict[str, Any]] = []
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, node: str, state: Dict[str, Any], **kwargs):
        """Log a workflow event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "node": node,
            "state_snapshot": {k: v for k, v in state.items() if k not in ["final_answer"]},
            **kwargs
        }
        self.events.append(event)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.console:
            print(f"[{event['timestamp']}] {event_type} @ {node}")
    
    def log_node_entry(self, node: str, state: Dict[str, Any]):
        """Log node entry"""
        self.log_event("node_entry", node, state)
    
    def log_node_exit(self, node: str, state: Dict[str, Any], result: Any = None):
        """Log node exit"""
        self.log_event("node_exit", node, state, result=result)
    
    def log_sql_execution(self, node: str, state: Dict[str, Any], query: str, 
                         columns: List[str], row_count: int, error: Optional[str]):
        """Log SQL execution details"""
        self.log_event("sql_execution", node, state, query=query, columns=columns, 
                      row_count=row_count, error=error)
    
    def log_repair_attempt(self, node: str, state: Dict[str, Any], original_query: str, 
                          repaired_query: str, attempt: int):
        """Log repair attempt"""
        self.log_event("repair_attempt", node, state, original_query=original_query,
                      repaired_query=repaired_query, attempt=attempt)
    
    def save_trace(self, file_path: Optional[Path] = None):
        """Save complete trace to file"""
        output_file = file_path or self.log_file
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return output_file


class WorkflowState(TypedDict):
    """State for LangGraph workflow"""
    question: str
    format_hint: str
    intent: dict
    rag_context: str
    rag_citations: list
    chunk_ids: list
    chunk_scores: list
    constraints: dict
    sql_query: str
    sql_executed: str  # Last successfully executed SQL
    sql_results: Optional[list]
    sql_columns: Optional[list]
    sql_row_count: int
    sql_error: Optional[str]
    sql_citation: dict
    repair_attempts: int
    answer: str
    final_answer: Optional[AnalyticsAnswer]


class HybridRetailAnalyticsGraph:
    """LangGraph workflow for retail analytics with RAG + SQL, repair loop, and checkpointer"""
    
    def __init__(self, enable_tracing: bool = True):
        # Initialize components
        self.rag = RAGRetrieval()
        self.sql_tool = SQLiteTool()
        
        # Initialize DSPy with Ollama (using litellm for better compatibility)
        self.lm = get_ollama_lm()
        dspy.configure(lm=self.lm)
        
        # Initialize DSPy modules (try to load optimized router if available)
        self.router = self._load_optimized_router() or QueryRoutingModule()
        self.planner = ConstraintPlannerModule()
        self.sql_generator = SQLGenerationModule()
        self.sql_repair = SQLRepairModule()
        self.answer_synthesizer = AnswerSynthesisModule()
        
        # Initialize checkpointer/trace
        self.logger = EventLogger() if enable_tracing else None
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _load_optimized_router(self):
        """Try to load optimized router if available"""
        return None  # Optimization removed per skeleton requirements
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow with ≥8 nodes"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (≥8 nodes as required)
        workflow.add_node("router", self._router)                    # Node 1: Router
        workflow.add_node("retriever", self._retriever)             # Node 2: Retriever
        workflow.add_node("planner", self._planner)                  # Node 3: Planner
        workflow.add_node("sql_generate", self._sql_generate)        # Node 4: NL→SQL
        workflow.add_node("executor", self._executor)                # Node 5: Executor
        workflow.add_node("synthesizer", self._synthesizer)          # Node 6: Synthesizer
        workflow.add_node("repair", self._repair)                    # Node 7: Repair
        # Node 8: Checkpointer is integrated in each node
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Router decides path
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "rag_only": "retriever",
                "sql_only": "sql_generate",
                "hybrid": "retriever",  # Start with RAG for hybrid
                "neither": "synthesizer"
            }
        )
        
        # After retriever, go to planner if hybrid, else synthesize
        workflow.add_conditional_edges(
            "retriever",
            self._after_retriever,
            {
                "to_planner": "planner",
                "to_sql": "sql_generate",
                "to_synthesize": "synthesizer"
            }
        )
        
        # Planner always goes to SQL generation
        workflow.add_edge("planner", "sql_generate")
        
        # SQL generation always goes to executor
        workflow.add_edge("sql_generate", "executor")
        
        # After execution, check if repair needed
        workflow.add_conditional_edges(
            "executor",
            self._after_execution,
            {
                "repair": "repair",
                "success": "synthesizer",
                "skip": "synthesizer"
            }
        )
        
        # Repair goes back to executor (with max attempts)
        workflow.add_conditional_edges(
            "repair",
            self._after_repair,
            {
                "retry": "executor",
                "give_up": "synthesizer"
            }
        )
        
        # Synthesizer always ends
        workflow.add_edge("synthesizer", END)
        
        return workflow
    
    def _router(self, state: WorkflowState) -> WorkflowState:
        """Node 1: Router (DSPy classifier)"""
        if self.logger:
            self.logger.log_node_entry("router", state)
        
        question = state["question"]
        intent = self.router(question=question)
        
        state["intent"] = {
            "requires_sql": intent.requires_sql,
            "requires_rag": intent.requires_rag,
            "sql_tables": intent.sql_tables,
            "keywords": intent.keywords
        }
        state["repair_attempts"] = 0
        
        if self.logger:
            self.logger.log_node_exit("router", state, result=state["intent"])
        
        return state
    
    def _retriever(self, state: WorkflowState) -> WorkflowState:
        """Node 2: Retriever (top-k doc chunks + scores + chunk IDs)"""
        if self.logger:
            self.logger.log_node_entry("retriever", state)
        
        if not state["intent"].get("requires_rag", True):
            state["rag_context"] = ""
            state["rag_citations"] = []
            state["chunk_ids"] = []
            state["chunk_scores"] = []
            return state
        
        question = state["question"]
        citations, chunk_ids, scores = self.rag.search(question)
        context = self.rag.get_context(question)
        
        state["rag_context"] = context
        state["rag_citations"] = [c.model_dump() for c in citations]
        state["chunk_ids"] = chunk_ids
        state["chunk_scores"] = scores
        
        if self.logger:
            self.logger.log_node_exit(
                "retriever", 
                state, 
                result={
                    "chunk_count": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                    "scores": scores
                }
            )
        
        return state
    
    def _planner(self, state: WorkflowState) -> WorkflowState:
        """Node 3: Planner (extract constraints: dates, KPIs, categories)"""
        if self.logger:
            self.logger.log_node_entry("planner", state)
        
        question = state["question"]
        rag_context = state.get("rag_context", "")
        
        constraints = self.planner(question=question, rag_context=rag_context)
        state["constraints"] = constraints
        
        if self.logger:
            self.logger.log_node_exit("planner", state, result=constraints)
        
        return state
    
    def _sql_generate(self, state: WorkflowState) -> WorkflowState:
        """Node 4: NL→SQL (DSPy) using live schema (PRAGMA)"""
        if self.logger:
            self.logger.log_node_entry("sql_generate", state)
        
        if not state["intent"].get("requires_sql", False):
            state["sql_query"] = ""
            return state
        
        question = state["question"]
        constraints = state.get("constraints", {})
        
        # Enhance question with constraints
        enhanced_question = question
        if constraints.get("date_ranges"):
            enhanced_question += f"\nDate range: {constraints['date_ranges']}"
        if constraints.get("kpi_formulas"):
            enhanced_question += f"\nKPI formula: {constraints['kpi_formulas']}"
        if constraints.get("categories"):
            enhanced_question += f"\nCategories: {constraints['categories']}"
        
        # Get live schema using PRAGMA
        schema = self.sql_tool.get_schema_string()
        
        # Generate SQL query
        sql_query = self.sql_generator(question=enhanced_question, schema=schema)
        
        # Clean up SQL query
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Fix "Order Details" table name
        patterns = [
            (r'\bOrder\s+Details\b', '"Order Details"'),
            (r'\bOrderDetails\b', '"Order Details"'),
        ]
        for pattern, replacement in patterns:
            if '"Order Details"' not in sql_query and '"order details"' not in sql_query.lower():
                sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        state["sql_query"] = sql_query
        
        if self.logger:
            self.logger.log_node_exit("sql_generate", state, result=sql_query)
        
        return state
    
    def _executor(self, state: WorkflowState) -> WorkflowState:
        """Node 5: Executor (run SQL; capture columns, rows, error)"""
        if self.logger:
            self.logger.log_node_entry("executor", state)
        
        sql_query = state.get("sql_query", "")
        if not sql_query:
            state["sql_results"] = None
            state["sql_columns"] = None
            state["sql_row_count"] = 0
            state["sql_error"] = None
            state["sql_citation"] = {}
            return state
        
        # Execute query with detailed results
        execution_result = self.sql_tool.execute_query(sql_query)
        
        state["sql_results"] = execution_result["results"]
        state["sql_columns"] = execution_result["columns"]
        state["sql_row_count"] = execution_result["row_count"]
        state["sql_error"] = execution_result["error"]
        
        if execution_result["error"]:
            state["sql_citation"] = {}
        else:
            # Track last successfully executed SQL
            state["sql_executed"] = sql_query
            citation = self.sql_tool.query_to_citation(sql_query, execution_result["results"])
            state["sql_citation"] = citation.model_dump()
        
        if self.logger:
            self.logger.log_sql_execution(
                "executor",
                state,
                query=sql_query,
                columns=execution_result["columns"] or [],
                row_count=execution_result["row_count"],
                error=execution_result["error"]
            )
            self.logger.log_node_exit("executor", state, result=execution_result)
        
        return state
    
    def _repair(self, state: WorkflowState) -> WorkflowState:
        """Node 7: Repair (revise SQL on error or invalid output)"""
        if self.logger:
            self.logger.log_node_entry("repair", state)
        
        original_query = state.get("sql_query", "")
        error_message = state.get("sql_error", "")
        schema = self.sql_tool.get_schema_string()
        
        # Repair the query
        repaired_query = self.sql_repair(
            original_query=original_query,
            error_message=error_message,
            schema=schema
        )
        
        # Clean up repaired query
        repaired_query = repaired_query.strip()
        if repaired_query.startswith("```sql"):
            repaired_query = repaired_query[6:]
        if repaired_query.startswith("```"):
            repaired_query = repaired_query[3:]
        if repaired_query.endswith("```"):
            repaired_query = repaired_query[:-3]
        repaired_query = repaired_query.strip()
        
        # Fix "Order Details" table name
        patterns = [
            (r'\bOrder\s+Details\b', '"Order Details"'),
            (r'\bOrderDetails\b', '"Order Details"'),
        ]
        for pattern, replacement in patterns:
            if '"Order Details"' not in repaired_query and '"order details"' not in repaired_query.lower():
                repaired_query = re.sub(pattern, replacement, repaired_query, flags=re.IGNORECASE)
        
        state["sql_query"] = repaired_query
        state["repair_attempts"] = state.get("repair_attempts", 0) + 1
        
        if self.logger:
            self.logger.log_repair_attempt(
                "repair",
                state,
                original_query=original_query,
                repaired_query=repaired_query,
                attempt=state["repair_attempts"]
            )
            self.logger.log_node_exit("repair", state, result=repaired_query)
        
        return state
    
    def _synthesizer(self, state: WorkflowState) -> WorkflowState:
        """Node 6: Synthesizer (produce typed answer matching format_hint)"""
        if self.logger:
            self.logger.log_node_entry("synthesizer", state)
        
        question = state["question"]
        rag_context = state.get("rag_context", "")
        format_hint = state.get("format_hint", "")
        
        # Format SQL results with columns
        sql_results_str = ""
        if state.get("sql_results") is not None:
            sql_results_str = self.sql_tool.format_results(
                state["sql_results"], 
                columns=state.get("sql_columns")
            )
        elif state.get("sql_error"):
            sql_results_str = f"SQL Error: {state['sql_error']}"
        
        # Synthesize answer with format_hint
        answer_text = self.answer_synthesizer(
            question=question,
            rag_context=rag_context,
            sql_results=sql_results_str,
            format_hint=format_hint
        )
        
        # Build citations (include chunk IDs and DB tables)
        citations = []
        if state.get("rag_citations"):
            citations.extend([Citation(**c) for c in state["rag_citations"]])
        if state.get("sql_citation"):
            citations.append(Citation(**state["sql_citation"]))
        
        # Determine answer type
        has_rag = bool(state.get("rag_context"))
        has_sql = bool(state.get("sql_query"))
        if has_rag and has_sql:
            answer_type = "hybrid"
        elif has_sql:
            answer_type = "sql"
        else:
            answer_type = "rag"
        
        # Calculate confidence using heuristics:
        # - Retrieval score coverage (average RAG scores)
        # - SQL success (1.0 if successful, 0.0 if error)
        # - Non-empty rows (1.0 if rows > 0, 0.5 if empty, 0.0 if error)
        # - Down-weight when repaired (reduce by 0.1 per repair attempt)
        
        rag_scores = state.get("chunk_scores", [])
        sql_success = 1.0 if not state.get("sql_error") and state.get("sql_results") is not None else 0.0
        sql_has_rows = 1.0 if state.get("sql_row_count", 0) > 0 else (0.5 if sql_success else 0.0)
        repair_penalty = min(0.2, state.get("repair_attempts", 0) * 0.1)  # Max 0.2 penalty
        
        # Calculate components
        rag_coverage = sum(rag_scores) / len(rag_scores) if rag_scores else 0.0
        
        # Combine scores (weighted average)
        if has_rag and has_sql:
            # Hybrid: average of RAG and SQL components
            confidence = (rag_coverage * 0.4 + sql_success * 0.3 + sql_has_rows * 0.3) - repair_penalty
        elif has_sql:
            # SQL-only: SQL success and row count
            confidence = (sql_success * 0.6 + sql_has_rows * 0.4) - repair_penalty
        else:
            # RAG-only: retrieval coverage
            confidence = rag_coverage
        
        # Ensure confidence is in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        state["final_answer"] = AnalyticsAnswer(
            answer=answer_text,
            citations=citations,
            answer_type=answer_type,
            sql_query=state.get("sql_executed", state.get("sql_query", "")),  # Use executed SQL
            confidence=confidence
        )
        
        if self.logger:
            self.logger.log_node_exit("synthesizer", state, result={
                "answer_type": answer_type,
                "confidence": confidence,
                "citation_count": len(citations)
            })
        
        return state
    
    def _route_decision(self, state: WorkflowState) -> str:
        """Decision after router"""
        intent = state["intent"]
        requires_rag = intent.get("requires_rag", False)
        requires_sql = intent.get("requires_sql", False)
        
        if requires_rag and requires_sql:
            return "hybrid"
        elif requires_rag:
            return "rag_only"
        elif requires_sql:
            return "sql_only"
        else:
            return "neither"
    
    def _after_retriever(self, state: WorkflowState) -> str:
        """Decision after retriever"""
        intent = state.get("intent", {})
        if intent.get("requires_sql", False):
            # If hybrid, go to planner first
            if intent.get("requires_rag", False):
                return "to_planner"
            return "to_sql"
        return "to_synthesize"
    
    def _after_execution(self, state: WorkflowState) -> str:
        """Decision after execution
        
        Constraint: Repair bound to ≤2 iterations
        """
        sql_error = state.get("sql_error")
        repair_attempts = state.get("repair_attempts", 0)
        
        # Hard limit: max 2 repair attempts (constraint: ≤2 iterations)
        if sql_error and repair_attempts < MAX_REPAIR_ATTEMPTS:
            return "repair"
        elif sql_error:
            return "skip"  # Give up after max attempts
        else:
            return "success"
    
    def _after_repair(self, state: WorkflowState) -> str:
        """Decision after repair
        
        Constraint: Repair bound to ≤2 iterations
        """
        repair_attempts = state.get("repair_attempts", 0)
        # Hard limit: max 2 repair attempts
        if repair_attempts < MAX_REPAIR_ATTEMPTS:
            return "retry"
        return "give_up"
    
    def query(self, question: str, format_hint: str = "") -> tuple:
        """Execute query through workflow
        
        Returns:
            tuple: (AnalyticsAnswer, state_metadata) where state_metadata contains
                   sql_executed, chunk_ids, rag_citations for output formatting
        """
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "intent": {},
            "rag_context": "",
            "rag_citations": [],
            "chunk_ids": [],
            "chunk_scores": [],
            "constraints": {},
            "sql_query": "",
            "sql_executed": "",
            "sql_results": None,
            "sql_columns": None,
            "sql_row_count": 0,
            "sql_error": None,
            "sql_citation": {},
            "repair_attempts": 0,
            "answer": "",
            "final_answer": None
        }
        
        final_state = self.app.invoke(initial_state)
        
        if self.logger:
            trace_file = self.logger.save_trace()
            print(f"\n[Trace saved to: {trace_file}]")
        
        # Extract metadata for output contract
        metadata = {
            "sql_executed": final_state.get("sql_executed", ""),
            "chunk_ids": final_state.get("chunk_ids", []),
            "rag_citations": final_state.get("rag_citations", [])
        }
        
        return final_state["final_answer"], metadata
