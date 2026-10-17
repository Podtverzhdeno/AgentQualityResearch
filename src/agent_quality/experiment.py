from __future__ import annotations
import os
import time
import uuid
from contextlib import nullcontext
from .agent import build_agent, final_text, invoke_agent
from .database import StoreDatabase
from .evaluation import evaluate_run
from .models import AgentConfig, RunRecord, Scenario
from .tools import ToolRuntime


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_handler():
    if not langfuse_enabled():
        return None
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def flush_langfuse() -> None:
    if langfuse_enabled():
        from langfuse import get_client
        get_client().flush()


def run_once(
    config: AgentConfig,
    scenario: Scenario,
    repetition: int,
    db: StoreDatabase,
    experiment_id: str,
) -> RunRecord:
    db.reset()
    runtime = ToolRuntime(db, scenario.input["user_id"])
    graph = build_agent(config, runtime)
    callback = langfuse_handler()
    trace_name = f"{config.name}/{scenario.id}/r{repetition:02d}"
    trace_input = {"user_id": scenario.input["user_id"], "message": scenario.input["message"]}
    metadata = {
        "experiment_id": experiment_id,
        "configuration": config.name,
        "scenario_id": scenario.id,
        "category": scenario.category,
        "repetition": repetition,
        "model": config.model,
        "critical_scenario": scenario.critical,
    }
    root = None
    if langfuse_enabled():
        from langfuse import get_client
        root_context = get_client().start_as_current_observation(
            as_type="agent", name=trace_name, input=trace_input, metadata=metadata
        )
    else:
        root_context = nullcontext(None)

    with root_context as root:
        if root:
            from langfuse import propagate_attributes
            attributes = propagate_attributes(
                trace_name=trace_name,
                user_id=scenario.input["user_id"],
                session_id=experiment_id,
                metadata=metadata,
                tags=[config.name, scenario.category, f"scenario:{scenario.id}"],
                environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "research"),
            )
        else:
            attributes = nullcontext()
        with attributes:
            started = time.perf_counter()
            output, error, result = "", None, None
            try:
                result = invoke_agent(
                    graph,
                    scenario.input["user_id"],
                    scenario.input["message"],
                    [callback] if callback else [],
                )
                output = final_text(result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            usage: dict[str, int] = {}
            if result:
                for message in result["messages"]:
                    for key, value in (getattr(message, "usage_metadata", None) or {}).items():
                        if isinstance(value, int):
                            usage[key] = usage.get(key, 0) + value
            record = RunRecord(
                run_id=str(uuid.uuid4()), config=config.name, repetition=repetition,
                scenario_id=scenario.id, category=scenario.category, output=output,
                tool_events=runtime.events, latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                error=error, trace_id=getattr(root, "trace_id", None) if root else None,
            )
            evaluate_run(record, scenario)
            if root:
                root.update(
                    output={"answer": output, "error": error},
                    level="ERROR" if error else "DEFAULT",
                    status_message=error,
                    metadata={**metadata, "violations": record.violations},
                )
                for name, value in record.metrics.items():
                    root.score_trace(name=name, value=float(value), data_type="NUMERIC")
                root.score_trace(
                    name="safety_pass",
                    value=float(not record.violations),
                    data_type="BOOLEAN",
                    comment=", ".join(record.violations) if record.violations else "No detected violations",
                )
                root.score_trace(
                    name="violations",
                    value=", ".join(record.violations) if record.violations else "none",
                    data_type="CATEGORICAL",
                )
            return record


def run_configuration(
    config: AgentConfig,
    scenarios: list[Scenario],
    repetitions: int,
    db: StoreDatabase,
    experiment_id: str,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    total = repetitions * len(scenarios)
    for repetition in range(1, repetitions + 1):
        for scenario in scenarios:
            print(f"[{config.name} {len(records) + 1}/{total}] {scenario.id} r{repetition:02d}", flush=True)
            records.append(run_once(config, scenario, repetition, db, experiment_id))
    return records


def record_experiment_summary(experiment_id: str, baseline: dict, candidate: dict, decision: dict) -> None:
    if not langfuse_enabled():
        return
    from langfuse import get_client, propagate_attributes
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="evaluator",
        name=f"experiment-summary/{experiment_id}",
        input={"baseline": baseline, "candidate": candidate},
    ) as root:
        with propagate_attributes(
            trace_name=f"experiment-summary/{experiment_id}",
            session_id=experiment_id,
            tags=["experiment-summary", decision["decision"].lower()],
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "research"),
        ):
            root.update(output=decision)
            root.score_trace(name="decision", value=decision["decision"], data_type="CATEGORICAL")
            if "baseline_score" in decision:
                root.score_trace(name="baseline_score", value=float(decision["baseline_score"]), data_type="NUMERIC")
                root.score_trace(name="candidate_score", value=float(decision["candidate_score"]), data_type="NUMERIC")
                root.score_trace(name="score_delta", value=float(decision["delta"]), data_type="NUMERIC")
