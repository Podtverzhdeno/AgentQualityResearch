from __future__ import annotations
import argparse
import uuid
from pathlib import Path
from .environment import load_environment
from .models import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_NAME = "agent-quality-research-v1"


def sync_dataset(dataset_path: Path, name: str) -> int:
    from langfuse import get_client
    langfuse = get_client()
    scenarios = load_scenarios(dataset_path)
    langfuse.create_dataset(
        name=name,
        description="Versioned scenarios for baseline/candidate LangGraph agent quality research",
        metadata={"source": str(dataset_path.name), "schema_version": "1"},
    )
    namespace = uuid.UUID("13e847a7-78e6-4b38-99f5-a18cb238d225")
    for scenario in scenarios:
        expected_output = {
            "expected_contains": scenario.expected_contains,
            "expected_contains_any": scenario.expected_contains_any,
            "expected_refusal": scenario.expected_refusal,
        }
        metadata = {
            "scenario_id": scenario.id,
            "category": scenario.category,
            "required_tools": scenario.required_tools,
            "forbidden_tools": scenario.forbidden_tools,
            "expected_tool_arguments": scenario.expected_tool_arguments,
            "ordered_tools": scenario.ordered_tools,
            "max_tool_calls": scenario.max_tool_calls,
            "critical": scenario.critical,
        }
        langfuse.create_dataset_item(
            id=str(uuid.uuid5(namespace, scenario.id)),
            dataset_name=name,
            input=scenario.input,
            expected_output=expected_output,
            metadata=metadata,
        )
    langfuse.flush()
    return len(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize research scenarios to Langfuse Dataset")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/scenarios.json")
    parser.add_argument("--name", default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()
    loaded = load_environment(args.env_file)
    print(f"Environment: {loaded or 'process variables only'}")
    count = sync_dataset(args.dataset, args.name)
    print(f"Dataset {args.name}: synchronized {count} scenarios")
