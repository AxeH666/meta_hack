import random
import statistics

from server.environment import ModGuardEnvironment
from server.models import ActionType, GTLabel, ModGuardAction, RiskLevel, Stage


def _env(seed: int = 123) -> ModGuardEnvironment:
    env = ModGuardEnvironment()
    env.reset(seed=seed)
    return env


def test_reset_returns_clean_observation_with_metadata() -> None:
    env = _env(1)
    obs = env.get_state()
    assert obs.step_number == 1
    reset_obs = env.reset(seed=1)
    assert reset_obs.step_number == 1
    assert reset_obs.reviewer_overturn_rate is None
    assert reset_obs.done is False
    assert reset_obs.reward == 0.0
    assert "ai_recommendation" in reset_obs.metadata
    assert "uncertainty_index" in reset_obs.metadata


def test_public_state_does_not_expose_ground_truth() -> None:
    env = _env(2)
    state = env.get_state()
    dumped = state.model_dump()
    assert "ground_truth" not in dumped
    assert dumped["episode_id"]


def test_initial_escalation_advances_to_step_two() -> None:
    env = _env(3)
    env.state.escalation_budget = 1
    obs = env.step(ActionType.escalate)
    assert obs.done is False
    assert obs.step_number == 2
    assert obs.stage == Stage.escalation_review


def test_zero_budget_escalation_triggers_penalty_and_audit() -> None:
    env = _env(4)
    env.state.escalation_budget = 0
    env.state.starting_escalation_budget = 0
    obs = env.step(ActionType.escalate)
    assert env.state.budget_violated is True
    assert env.state.path_penalty_incurred is True
    assert env.state.repeated_escalations >= 1
    assert obs.done is False
    assert obs.stage == Stage.post_decision_audit


def test_post_decision_audit_can_be_entered_from_initial_review() -> None:
    env = _env(5)
    env._case["audit_required"] = True
    env._case["audit_reasons"] = ["conflicting_signals"]
    env.state.audit_required = True
    obs = env.step(ActionType.approve)
    assert obs.done is False
    assert obs.stage == Stage.post_decision_audit
    assert obs.step_number == 2


def test_four_step_trajectory_is_possible() -> None:
    env = _env(6)
    env._case["ground_truth"] = GTLabel.legal_hold
    env._case["gt_action"] = ActionType.legal_hold
    env._case["risk_level"] = RiskLevel.critical
    env._case["audit_required"] = True
    env._case["audit_reasons"] = ["possible_legal_retention"]
    env._case["delayed_legal"] = True
    env.state.audit_required = True
    env.state.escalation_budget = 1
    env.state.starting_escalation_budget = 1

    obs = env.step(ActionType.escalate)
    assert obs.stage == Stage.escalation_review
    obs = env.step(ActionType.legal_hold)
    assert obs.stage == Stage.legal_review
    obs = env.step(ActionType.legal_hold)
    assert obs.stage == Stage.post_decision_audit
    assert obs.step_number == 4
    obs = env.step(ActionType.legal_hold)
    assert obs.done is True
    assert obs.step_number == 4


def test_terminal_reward_contains_breakdown_and_is_bounded() -> None:
    env = _env(7)
    env._case["ground_truth"] = GTLabel.approve
    env._case["gt_action"] = ActionType.approve
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert 0.0 <= float(obs.reward) <= 1.0
    breakdown = obs.metadata["reward_breakdown"]
    assert "correctness" in breakdown
    assert "consistency" in breakdown
    assert "uncertainty_awareness" in breakdown
    assert "overconfidence_penalty" in breakdown


def test_overconfidence_penalty_is_positive_for_blind_wrong_high_confidence() -> None:
    env = _env(8)
    env._case["ground_truth"] = GTLabel.remove
    env._case["gt_action"] = ActionType.remove
    env._case["ai_action"] = ActionType.approve
    env._case["ai_confidence_score"] = 0.98
    env._case["signal_conflict_score"] = 0.92
    env._case["uncertainty_index"] = 0.89
    env._case["misleading_ai"] = True
    env.state.action_history = [ActionType.approve]
    env.state.step_number = 1
    penalty = env._overconfidence_penalty(ActionType.approve)
    assert penalty > 0.0


def test_rewards_have_variance_over_random_rollouts() -> None:
    chooser = random.Random(42)
    rewards = []
    for seed in range(100, 140):
        env = ModGuardEnvironment()
        obs = env.reset(seed=seed)
        while not obs.done:
            obs = env.step(ModGuardAction(action=chooser.choice(list(ActionType))))
        rewards.append(float(obs.reward or 0.0))
    assert statistics.pstdev(rewards) > 0.01
    assert all(0.0 <= reward <= 1.0 for reward in rewards)


def test_state_metadata_tracks_steps_and_actions() -> None:
    env = ModGuardEnvironment()
    env.reset(seed=99)
    s0 = env.get_state()
    assert s0.step_count == 0
    assert s0.action_history == []
    env.step(ModGuardAction(action=ActionType.approve))
    s1 = env.get_state()
    assert s1.step_count == 1
    assert s1.action_history == [ActionType.approve]
