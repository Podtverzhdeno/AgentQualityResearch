from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import yaml


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str
    temperature: float
    max_iterations: int
    system_prompt: str
    requests_per_minute: int = 12
    max_rate_limit_retries: int = 8

    @classmethod
    def load(cls, path: Path) -> "AgentConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    input: dict[str, str]
    expected_contains: list[str] = field(default_factory=list)
    expected_contains_any: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_calls: int = 3
    critical: bool = False
    expected_tool_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)
    ordered_tools: list[str] = field(default_factory=list)
    expected_refusal: bool = False


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    output: Any
    duration_ms: float
    violation: str | None = None


@dataclass
class RunRecord:
    run_id: str
    config: str
    repetition: int
    scenario_id: str
    category: str
    output: str
    tool_events: list[ToolEvent]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None
    trace_id: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scenarios(path: Path) -> list[Scenario]:
    return [Scenario(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
