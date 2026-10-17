from __future__ import annotations
import json
from pathlib import Path


def save_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison, decision = payload["comparison"], payload["decision"]
    rows = []
    for name in comparison["baseline"]["metrics"]:
        rows.append(f"| {name} | {comparison['baseline']['metrics'][name]:.3f} | {comparison['candidate']['metrics'][name]:.3f} |")
    category_rows = []
    categories = sorted(set(comparison["baseline"]["categories"]) | set(comparison["candidate"]["categories"]))
    for category in categories:
        b = comparison["baseline"]["categories"].get(category, {})
        c = comparison["candidate"]["categories"].get(category, {})
        category_rows.append(f"| {category} | {b.get('answer_quality', 0):.3f} | {c.get('answer_quality', 0):.3f} | {b.get('trajectory_quality', 0):.3f} | {c.get('trajectory_quality', 0):.3f} |")
    markdown = f"""# Agent configuration experiment

Experiment ID: {payload['experiment']['id']}

Decision: **{decision['decision']}**

Reason: {decision['reason']}

| Metric | Baseline | Candidate |
|---|---:|---:|
{chr(10).join(rows)}

Raw run-level data is available in the adjacent JSON report.

## Results by scenario category

| Category | Answer B | Answer C | Trajectory B | Trajectory C |
|---|---:|---:|---:|---:|
{chr(10).join(category_rows)}

## Violations

- Baseline: {comparison['baseline']['violations']}
- Candidate: {comparison['candidate']['violations']}

Confidence intervals, per-run metrics, tool events and errors are included in JSON.
"""
    path.with_suffix(".md").write_text(markdown, encoding="utf-8")
