from __future__ import annotations
import argparse
import json
import uuid
from pathlib import Path
from .database import StoreDatabase
from .decision import decide
from .environment import load_environment
from .evaluation import aggregate
from .experiment import flush_langfuse, record_experiment_summary, run_configuration
from .models import AgentConfig, load_scenarios
from .reporting import save_report

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline/candidate agent experiment")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/scenarios.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latest")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    loaded = load_environment(args.env_file)
    print(f"Environment: {loaded or 'process variables only'}")
    scenarios = load_scenarios(args.dataset)
    if args.limit:
        scenarios = scenarios[:args.limit]
    db = StoreDatabase(ROOT / "data/store.db")
    experiment_id = args.experiment_id or f"aqr-{uuid.uuid4().hex[:10]}"
    print(f"Experiment ID: {experiment_id}")
    baseline_cfg = AgentConfig.load(ROOT / "configs/baseline.yaml")
    candidate_cfg = AgentConfig.load(ROOT / "configs/candidate.yaml")
    baseline_records = run_configuration(baseline_cfg, scenarios, args.repetitions, db, experiment_id)
    candidate_records = run_configuration(candidate_cfg, scenarios, args.repetitions, db, experiment_id)
    baseline, candidate = aggregate(baseline_records), aggregate(candidate_records)
    decision = decide(baseline, candidate, ROOT / "configs/evaluation.yaml")
    record_experiment_summary(experiment_id, baseline, candidate, decision)
    flush_langfuse()
    payload = {
        "experiment": {"id": experiment_id, "repetitions": args.repetitions, "scenario_count": len(scenarios)},
        "comparison": {"baseline": baseline, "candidate": candidate},
        "decision": decision,
        "runs": {"baseline": [r.as_dict() for r in baseline_records], "candidate": [r.as_dict() for r in candidate_records]},
    }
    save_report(args.output, payload)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"Reports: {args.output.with_suffix('.md')} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
