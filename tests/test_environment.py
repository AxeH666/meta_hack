import random
import statistics

import pytest

from server.environment import ModGuardEnvironment
from server.models import ActionType, DifficultyLevel, GTLabel, RiskLevel


def _env(seed: int = 123) -> ModGuardEnvironment:
    env = ModGuardEnvironment()
    env.reset(seed=seed)
    return env


def _set_gt_and_risk(env: ModGuardEnvironment, gt: GTLabel, risk: RiskLevel) -> None:
    env.state.ground_truth = gt
    env._reset_observation = env._reset_observation.model_copy(update={"risk_level": risk})


def _run_random_episode(seed: int, chooser: random.Random) -> float:
    env = ModGuardEnvironment()
    obs = env.reset(seed=seed)
    while not obs.done:
        obs = env.step(ActionType(chooser.choice(list(ActionType)).value))
    return float(obs.reward if obs.reward is not None else 0.0)


def test_01_reset_returns_clean_observation_with_10_fields() -> None:
    env = _env(1)
    obs = env._reset_observation
    fields = {
        "content_category",
        "risk_level",
        "platform_context",
        "ai_confidence_score",
        "human_reviewer_hint",
        "queue_pressure",
        "reviewer_overturn_rate",
        "step_number",
        "case_history",
        "stage",
    }
    assert fields.issubset(set(obs.model_dump().keys()))
    assert obs.step_number == 1
    assert obs.reviewer_overturn_rate is None
    assert obs.done is False
    assert obs.reward == 0.0


def test_02_non_terminal_step_reward_is_zero() -> None:
    env = _env(2)
    obs = env.step(ActionType.escalate)
    assert obs.done is False
    assert obs.reward == 0.0


def test_03_terminal_step_reward_is_positive() -> None:
    env = _env(3)
    env.state.ground_truth = GTLabel.approve
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert float(obs.reward) > 0.0


def test_04_grade_formula_example_a_is_0_96() -> None:
    env = _env(4)
    env.state.ground_truth = GTLabel.approve
    env.state.step_number = 1
    env.state.path_penalty_incurred = False
    env.state.budget_violated = False
    env._initial_hint = ActionType.approve
    grade = env._grade(ActionType.approve)
    assert grade == pytest.approx(0.96, abs=1e-9)


def test_05_budget_violated_forces_process_score_zero() -> None:
    env = _env(5)
    env.state.budget_violated = True
    assert env._process_score() == 0.0


def test_06_trigger_1_escalation_review_escalate_sets_path_penalty() -> None:
    env = _env(6)
    env.step(ActionType.escalate)
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is True


def test_07_trigger_2_escalation_review_legal_hold_non_critical_wrong_gt() -> None:
    env = _env(7)
    _set_gt_and_risk(env, GTLabel.remove, RiskLevel.high)
    env.step(ActionType.escalate)
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is True


def test_08_trigger_3_legal_review_escalate_sets_path_penalty() -> None:
    env = _env(8)
    env.step(ActionType.legal_hold)
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is True


def test_09_trigger_4_legal_review_legal_hold_wrong_gt_sets_path_penalty() -> None:
    env = _env(9)
    _set_gt_and_risk(env, GTLabel.remove, RiskLevel.medium)
    env.step(ActionType.legal_hold)
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is True


def test_10_escalation_budget_never_below_zero() -> None:
    env = _env(10)
    env.state.escalation_budget = 0
    env.step(ActionType.escalate)
    assert env.state.escalation_budget == 0
    assert env.state.budget_violated is True


def test_11_path_penalty_is_sticky() -> None:
    env = _env(11)
    _set_gt_and_risk(env, GTLabel.legal_hold, RiskLevel.critical)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is False
    assert env.state.path_penalty_incurred is True
    env.step(ActionType.approve)
    assert env.state.path_penalty_incurred is True


def test_12_reviewer_overturn_rate_none_step1_non_none_step2_plus() -> None:
    env = _env(12)
    first = env._reset_observation
    second = env.step(ActionType.escalate)
    assert first.reviewer_overturn_rate is None
    assert second.reviewer_overturn_rate is not None


def test_13_hard_mode_hint_always_none(monkeypatch: pytest.MonkeyPatch) -> None:
    original_choice = random.Random.choice

    def _patched_choice(rng: random.Random, seq):  # type: ignore[no-untyped-def]
        if seq and isinstance(seq[0], DifficultyLevel):
            return DifficultyLevel.hard
        return original_choice(rng, seq)

    monkeypatch.setattr(random.Random, "choice", _patched_choice)
    env = ModGuardEnvironment()
    obs = env.reset(seed=13)
    assert obs.human_reviewer_hint is None


def test_14_easy_mode_hint_matches_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    original_choice = random.Random.choice

    def _patched_choice(rng: random.Random, seq):  # type: ignore[no-untyped-def]
        if seq and isinstance(seq[0], DifficultyLevel):
            return DifficultyLevel.easy
        return original_choice(rng, seq)

    monkeypatch.setattr(random.Random, "choice", _patched_choice)
    env = ModGuardEnvironment()
    obs = env.reset(seed=14)
    expected = ActionType(env.state.ground_truth.value)
    assert obs.human_reviewer_hint == expected


def test_15_reward_diversity_over_20_random_episodes() -> None:
    chooser = random.Random(42)
    rewards = [_run_random_episode(seed, chooser) for seed in range(100, 120)]
    assert statistics.pstdev(rewards) > 0.0


def test_16_rewards_in_range_over_20_random_episodes() -> None:
    chooser = random.Random(43)
    rewards = [_run_random_episode(seed, chooser) for seed in range(200, 220)]
    assert all(0.0 <= r <= 1.0 for r in rewards)
