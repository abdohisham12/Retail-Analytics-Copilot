"""CLI entry point for running the Retail Analytics Copilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import click
import dspy
from rich.console import Console
from agent.graph_hybrid import (
    AgentState,
    build_agent_resources,
    build_graph,
    consume_profile_metrics,
)


def profiling_enabled() -> bool:
    return os.environ.get("AGENT_PROFILE", "0") in {"1", "true", "True"}

console = Console()


def configure_lm(model: str) -> None:
    """Configure DSPy to use the local Ollama Phi-3.5 model via the generic LM interface."""
    # Normalize 'mock' to the default real model so we never hit OfflineMockLM.
    if model.lower() == "mock":
        model = "phi3.5:3.8b-mini-instruct-q4_K_M"
    try:
        console.print(f"[cyan]Configuring Ollama model: {model}[/cyan]")
        # In dspy>=3, arbitrary backends (including Ollama via LiteLLM) are selected
        # by prefixing the model name with the provider identifier.
        lm = dspy.LM(f"ollama/{model}", temperature=0.2)
        dspy.settings.configure(lm=lm)
    except Exception as exc:
        # Hard fail if Ollama cannot be reached or the model is missing.
        console.print(
            "[red]Failed to configure Ollama model.[/red]\n"
            f"Details: {exc}\n"
            "Make sure the Ollama server is running and the model is pulled, e.g.:\n"
            "  ollama pull phi3.5:3.8b-mini-instruct-q4_K_M"
        )
        raise


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_explanation_safe(text: str, max_sentences: int = 2) -> str:
    """Normalize explanation to exactly <= max_sentences sentences, never cutting mid-sentence."""
    if not text:
        return ""
    import re
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    # Take first max_sentences sentences
    trimmed = " ".join(parts[:max_sentences]).strip()
    # Fallback if model forgot punctuation
    if not trimmed and parts:
        trimmed = parts[0].strip()
    return trimmed


def run_question(graph, payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """Execute a single question through the agent graph and return answer, trace, and metrics."""
    state: AgentState = {
        "question_id": payload["id"],
        "question": payload["question"],
        "format_hint": payload.get("format_hint", ""),
        "repair_attempts": 0,
        "trace": [],
    }
    final_state = graph.invoke(state)
    sql_text = final_state.get("sql", "")
    mode = final_state.get("mode", "")
    sql_result = final_state.get("sql_result", {})
    question_id = payload.get("id", "").lower()
    question_text = payload.get("question", "").lower()
    # Clear SQL for RAG-only questions (by ID pattern, mode, or failed SQL with policy content)
    is_rag_only = (
        mode == "rag"
        or question_id.startswith("rag_")
        or (sql_result.get("error") and not sql_result.get("rows") and ("policy" in question_text or question_id.startswith("rag_")))
    )
    if is_rag_only:
        sql_text = ""
    result = {
        "id": payload["id"],
        "final_answer": final_state.get("final_answer"),
        "sql": sql_text,
        "confidence": float(final_state.get("confidence") or 0.4),
        "explanation": _normalize_explanation_safe(final_state.get("explanation") or "Answered using local context."),
        "citations": final_state.get("citations", []),
    }
    return result, final_state.get("trace", []), final_state.get("metrics", {})


def persist_dspy_report(report: Optional[Dict[str, Any]]) -> None:
    if not report:
        return
    metrics_dir = Path("testing") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / "dspy_report.json"
    report_path.write_text(json.dumps(report, indent=2))


@click.command()
@click.option(
    "--batch",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input JSONL file containing evaluation questions.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Destination JSONL file for agent answers.",
)
@click.option(
    "--model",
    type=str,
    default="phi3.5:3.8b-mini-instruct-q4_K_M",
    show_default=True,
    help="Local Ollama model identifier.",
)
def cli(batch: Path, out: Path, model: str) -> None:
    """Run the hybrid agent over a batch of retail analytics questions."""
    configure_lm(model)
    resources = build_agent_resources()
    nl2sql_report = getattr(resources.nl2sql, "report", None)
    if nl2sql_report:
        persist_dspy_report(
            {"module": "NL2SQLModule", **nl2sql_report.as_dict()}
        )
    graph = build_graph(resources)
    questions = load_jsonl(batch)
    console.print(f"[cyan]Loaded {len(questions)} questions from {batch}[/cyan]")

    outputs = []
    metrics_log: Dict[str, Any] = {}
    trace_log: List[Dict[str, Any]] = []
    for item in questions:
        console.print(f"[magenta]Processing[/magenta] {item['id']}")
        answer, trace, metrics = run_question(graph, item)
        outputs.append(answer)
        log_payload = {"id": item["id"], "trace": trace}
        trace_log.append(log_payload)
        if profiling_enabled():
            if not metrics:
                metrics = consume_profile_metrics(item["id"])
            metrics_log[item["id"]] = metrics or {}
            if metrics:
                log_payload["timings"] = metrics.get("timings", {})
        console.log(log_payload)

    write_jsonl(out, outputs)
    console.print(f"[green]Wrote results to {out}[/green]")
    
    # Save replayable trace log to file (file/console as required)
    trace_path = out.with_suffix("").with_name(f"{out.stem}_trace.jsonl")
    write_jsonl(trace_path, trace_log)
    console.print(f"[blue]Saved trace log to {trace_path}[/blue]")
    
    if profiling_enabled() and metrics_log:
        metrics_path = Path("testing") / "metrics" / "latest_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_log, indent=2))
        console.print(f"[blue]Saved metrics to {metrics_path}[/blue]")


if __name__ == "__main__":
    cli()
