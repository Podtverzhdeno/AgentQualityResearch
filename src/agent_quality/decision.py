from __future__ import annotations
from pathlib import Path
import yaml


def decide(baseline: dict, candidate: dict, config_path: Path) -> dict:
    policy = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = candidate["metrics"]
    critical = policy["critical_violations"]
    found = {name: candidate["violations"].get(name, 0) for name in critical if candidate["violations"].get(name, 0)}
    if found:
        return {"decision": "REJECT", "reason": "critical safety violation", "critical": found}
    weights = policy["weights"]
    baseline_score = sum(baseline["metrics"][name] * weight for name, weight in weights.items())
    candidate_score = sum(metrics[name] * weight for name, weight in weights.items())
    delta = candidate_score - baseline_score
    thresholds = policy["thresholds"]
    if metrics["technical_success"] < 1 - thresholds["max_error_rate"]:
        decision, reason = "REJECT", "technical error rate exceeds threshold"
    elif metrics["answer_quality"] < thresholds["minimum_answer_quality"] or metrics["trajectory_quality"] < thresholds["minimum_trajectory_quality"]:
        decision, reason = "REJECT", "key quality floor violated"
    elif delta >= thresholds["accept_min_delta"]:
        decision, reason = "ACCEPT", "weighted score improved without gate violations"
    else:
        decision, reason = "REVIEW", "difference is small or negative and requires review"
    return {
        "decision": decision,
        "reason": reason,
        "baseline_score": round(baseline_score, 4),
        "candidate_score": round(candidate_score, 4),
        "delta": round(delta, 4),
    }
