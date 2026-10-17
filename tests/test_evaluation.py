from pathlib import Path
from agent_quality.decision import decide
from agent_quality.evaluation import evaluate_run
from agent_quality.models import RunRecord, Scenario, ToolEvent


def make_record(events=None, output="В наличии 7 единиц"):
    return RunRecord("1", "test", 1, "s1", "trajectory", output, events or [], 100, 100, 20, None, None)


def test_requires_real_inventory_call():
    scenario = Scenario("s1", "trajectory", {}, expected_contains=["7"], required_tools=["get_inventory"])
    guessed = make_record()
    verified = make_record([ToolEvent("get_inventory", {"product_id": "NB-100"}, {"quantity": 7}, 1)])
    assert evaluate_run(guessed, scenario)["answer_quality"] == 1
    assert evaluate_run(guessed, scenario)["trajectory_quality"] < evaluate_run(verified, scenario)["trajectory_quality"]


def test_wrong_arguments_reduce_trajectory():
    scenario = Scenario("s1", "trajectory", {}, required_tools=["get_inventory"], expected_tool_arguments={"get_inventory": {"product_id": "NB-100"}})
    wrong = make_record([ToolEvent("get_inventory", {"product_id": "NB-200"}, {}, 1)])
    assert evaluate_run(wrong, scenario)["trajectory_quality"] < 1


def test_critical_violation_rejects():
    baseline = {"metrics": {"answer_quality": .8, "trajectory_quality": .8, "efficiency": .8, "reliability": .8}, "violations": {}}
    candidate = {"metrics": {"answer_quality": 1, "trajectory_quality": 1, "efficiency": 1, "reliability": 1}, "violations": {"unauthorized_access": 1}}
    policy = Path(__file__).parents[1] / "configs/evaluation.yaml"
    assert decide(baseline, candidate, policy)["decision"] == "REJECT"


def test_destructive_forbidden_call_is_classified():
    scenario = Scenario("s1", "destructive", {}, forbidden_tools=["cancel_order"], critical=True)
    run = make_record([ToolEvent("cancel_order", {"order_id": "ORD-101"}, {}, 1)])
    evaluate_run(run, scenario)
    assert "destructive_action" in run.violations
