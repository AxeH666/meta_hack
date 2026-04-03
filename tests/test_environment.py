import math
import os
import random
import statistics
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.environment import ModGuardEnvironment
from server.models import ActionType, GTLabel, RiskLevel, Stage


SECTIONS_TOTAL = 12


class TestableModGuardEnvironment(ModGuardEnvironment):
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


def assert_equal(actual, expected, label):
    assert actual == expected, f"{label} | expected={expected!r}, actual={actual!r}"


def assert_float_close(actual, expected, label, tol=1e-9):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (
        f"{label} | expected={expected:.12f}, actual={actual:.12f}"
    )


def snapshot_observation(obs):
    return {
        "content_category": obs.content_category,
        "risk_level": obs.risk_level,
        "platform_context": obs.platform_context,
        "ai_confidence_score": obs.ai_confidence_score,
        "human_reviewer_hint": obs.human_reviewer_hint,
        "queue_pressure": obs.queue_pressure,
        "case_history_prior_escalations": obs.case_history.prior_escalations,
        "case_history_account_risk": obs.case_history.account_risk,
    }


def force_gt_and_risk(env, gt, risk):
    env.state.ground_truth = gt
    env._reset_observation.risk_level = risk


def reset_forced(gt, risk, seed=1234):
    env = TestableModGuardEnvironment()
    env.reset(seed=seed)
    force_gt_and_risk(env, gt, risk)
    return env


def choose_random_action_with_env_rng(env):
    return env.rng.choice(list(ActionType))


def process_expected(step_number, gt, risk, budget_violated, path_penalty):
    if budget_violated:
        return 0.0
    if gt in {GTLabel.approve, GTLabel.remove}:
        optimal = 1
    elif risk == RiskLevel.critical:
        optimal = 3
    else:
        optimal = 1
    diff = abs(step_number - optimal)
    if diff == 0:
        score = 1.0
    elif diff == 1:
        score = 0.6
    else:
        score = 0.2
    if path_penalty:
        score = min(score, 0.4)
    return score


def hint_score_expected(hint, terminal_action, gt):
    gt_action = ActionType(gt.value)
    if hint is None:
        return 0.5
    if hint == terminal_action and hint == gt_action:
        return 0.8
    if hint == terminal_action and hint != gt_action:
        return 0.0
    if hint != terminal_action and hint == gt_action:
        return 0.3
    return 1.0


def parse_floats_from_text(text):
    tokens = text.replace(",", " ").replace(":", " ").replace("=", " ").split()
    out = []
    for token in tokens:
        cleaned = token.strip("[](){}<>;")
        if cleaned.count(".") == 0:
            continue
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def section_1_determinism():
    seeds = [3, 7, 11, 19, 23]
    for seed in seeds:
        env_a = TestableModGuardEnvironment()
        env_b = TestableModGuardEnvironment()
        obs_a = env_a.reset(seed=seed)
        obs_b = env_b.reset(seed=seed)
        assert_equal(obs_a.content_category, obs_b.content_category, "determinism.content_category")
        assert_equal(obs_a.risk_level, obs_b.risk_level, "determinism.risk_level")
        assert_equal(obs_a.platform_context, obs_b.platform_context, "determinism.platform_context")
        assert_float_close(
            obs_a.ai_confidence_score, obs_b.ai_confidence_score, "determinism.ai_confidence_score"
        )
        assert_equal(
            obs_a.human_reviewer_hint, obs_b.human_reviewer_hint, "determinism.human_reviewer_hint"
        )
        assert_equal(obs_a.queue_pressure, obs_b.queue_pressure, "determinism.queue_pressure")
        assert_equal(
            obs_a.case_history.prior_escalations,
            obs_b.case_history.prior_escalations,
            "determinism.case_history.prior_escalations",
        )
        assert_float_close(
            obs_a.case_history.account_risk,
            obs_b.case_history.account_risk,
            "determinism.case_history.account_risk",
        )
        assert_equal(obs_a.stage, obs_b.stage, "determinism.stage")


def section_2_state_machine_coverage():
    # Step 1 transitions
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=101)
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert_equal(obs.step_number, 1, "s2.step1.approve.step_number")
    assert_equal(obs.stage, Stage.initial_review, "s2.step1.approve.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=102)
    obs = env.step(ActionType.remove)
    assert obs.done is True
    assert_equal(obs.step_number, 1, "s2.step1.remove.step_number")
    assert_equal(obs.stage, Stage.initial_review, "s2.step1.remove.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=103)
    obs = env.step(ActionType.escalate)
    assert obs.done is False
    assert_equal(obs.step_number, 2, "s2.step1.escalate.step_number")
    assert_equal(obs.stage, Stage.escalation_review, "s2.step1.escalate.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=104)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is False
    assert_equal(obs.step_number, 2, "s2.step1.legal_hold.step_number")
    assert_equal(obs.stage, Stage.legal_review, "s2.step1.legal_hold.stage")

    # Step 2 escalation_review transitions
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=105)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert_equal(obs.stage, Stage.escalation_review, "s2.step2.escalation.approve.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=106)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.remove)
    assert obs.done is True
    assert_equal(obs.stage, Stage.escalation_review, "s2.step2.escalation.remove.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=107)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.escalate)
    assert obs.done is True
    assert env.state.path_penalty_incurred is True

    env = reset_forced(GTLabel.approve, RiskLevel.critical, seed=108)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is False
    assert_equal(obs.step_number, 3, "s2.step2.escalation.legal_hold.critical.step_number")
    assert_equal(obs.stage, Stage.legal_review, "s2.step2.escalation.legal_hold.critical.stage")

    env = reset_forced(GTLabel.legal_hold, RiskLevel.high, seed=109)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is True
    assert env.state.path_penalty_incurred is False

    env = reset_forced(GTLabel.remove, RiskLevel.low, seed=110)
    env.step(ActionType.escalate)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is True
    assert env.state.path_penalty_incurred is True

    # Step 2 legal_review transitions
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=111)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert_equal(obs.stage, Stage.legal_review, "s2.step2.legal.approve.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=112)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.remove)
    assert obs.done is True
    assert_equal(obs.stage, Stage.legal_review, "s2.step2.legal.remove.stage")

    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=113)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.escalate)
    assert obs.done is True
    assert env.state.path_penalty_incurred is True

    env = reset_forced(GTLabel.legal_hold, RiskLevel.medium, seed=114)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is True
    assert env.state.path_penalty_incurred is False

    env = reset_forced(GTLabel.remove, RiskLevel.medium, seed=115)
    env.step(ActionType.legal_hold)
    obs = env.step(ActionType.legal_hold)
    assert obs.done is True
    assert env.state.path_penalty_incurred is True

    # Step 3 terminal
    env = reset_forced(GTLabel.legal_hold, RiskLevel.critical, seed=116)
    env.step(ActionType.escalate)
    obs_mid = env.step(ActionType.legal_hold)
    assert obs_mid.done is False
    assert_equal(obs_mid.step_number, 3, "s2.step3.entry.step_number")
    assert_equal(obs_mid.stage, Stage.legal_review, "s2.step3.entry.stage")
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert_equal(obs.step_number, 3, "s2.step3.any_action.step_number")
    assert_equal(obs.stage, Stage.legal_review, "s2.step3.any_action.stage")


def section_3_trigger_validation(trigger_coverage):
    # T1
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=201)
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is False
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is True
    trigger_coverage["T1"] = True
    # sticky check
    assert env.state.path_penalty_incurred is True

    # T2
    env = reset_forced(GTLabel.remove, RiskLevel.high, seed=202)
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is False
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is True
    trigger_coverage["T2"] = True
    assert env.state.path_penalty_incurred is True

    # T3
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=203)
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is False
    env.step(ActionType.escalate)
    assert env.state.path_penalty_incurred is True
    trigger_coverage["T3"] = True
    assert env.state.path_penalty_incurred is True

    # T4
    env = reset_forced(GTLabel.remove, RiskLevel.medium, seed=204)
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is False
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is True
    trigger_coverage["T4"] = True
    assert env.state.path_penalty_incurred is True

    # T5
    env = TestableModGuardEnvironment()
    env.reset(seed=205)
    env._reset_observation = env._reset_observation.model_copy(
        update={"risk_level": RiskLevel.critical}
    )
    env.state.ground_truth = GTLabel.legal_hold
    assert env.state.path_penalty_incurred is False
    env.step(ActionType.legal_hold)
    env.step(ActionType.legal_hold)
    assert env.state.path_penalty_incurred is True
    env.step(ActionType.remove)
    trigger_coverage["T5"] = True
    assert env.state.path_penalty_incurred is True


def section_4_reward_validation():
    env = reset_forced(GTLabel.legal_hold, RiskLevel.critical, seed=301)
    obs = env.step(ActionType.escalate)
    assert_float_close(obs.reward, 0.0, "s4.non_terminal_reward")
    assert obs.done is False

    obs = env.step(ActionType.legal_hold)
    assert_float_close(obs.reward, 0.0, "s4.non_terminal_reward_step2")
    assert obs.done is False

    final_obs = env.step(ActionType.legal_hold)
    assert final_obs.done is True
    assert 0.0 <= final_obs.reward <= 1.0
    assert_float_close(final_obs.reward, env._grade(ActionType.legal_hold), "s4.terminal_reward_equals_grade")

    key_pairs = [
        (GTLabel.approve, ActionType.approve, 1.0 * 0.45),
        (GTLabel.legal_hold, ActionType.approve, 0.0 * 0.45),
        (GTLabel.remove, ActionType.legal_hold, 0.1 * 0.45),
        (GTLabel.legal_hold, ActionType.escalate, 0.5 * 0.45),
    ]

    for gt, terminal_action, expected_correctness_component in key_pairs:
        env = reset_forced(gt, RiskLevel.medium, seed=302)
        env.state.step_number = 1
        env.state.path_penalty_incurred = False
        env.state.budget_violated = True  # zeroes process component
        env._initial_hint = terminal_action  # with hint==terminal and hint!=gt => 0.0 when gt differs
        if ActionType(gt.value) == terminal_action:
            env._initial_hint = ActionType(gt.value)
        grade = env._grade(terminal_action)
        hint_component = hint_score_expected(env._initial_hint, terminal_action, gt) * 0.20
        speed_component = 1.0 * 0.10
        process_component = 0.0
        extracted = grade - hint_component - speed_component - process_component
        assert_float_close(
            extracted,
            expected_correctness_component,
            f"s4.correctness_component.{gt.value}.{terminal_action.value}",
        )


def section_5_process_score_validation():
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=401)
    env.state.budget_violated = True
    assert_float_close(env._process_score(), 0.0, "s5.budget_violated_zero")

    # Optimal steps
    env = reset_forced(GTLabel.approve, RiskLevel.critical, seed=402)
    env.state.step_number = 1
    assert_float_close(env._process_score(), 1.0, "s5.optimal.approve")

    env = reset_forced(GTLabel.remove, RiskLevel.low, seed=403)
    env.state.step_number = 1
    assert_float_close(env._process_score(), 1.0, "s5.optimal.remove")

    env = reset_forced(GTLabel.legal_hold, RiskLevel.high, seed=404)
    env.state.step_number = 1
    assert_float_close(env._process_score(), 1.0, "s5.optimal.legal_hold_non_critical")

    env = reset_forced(GTLabel.legal_hold, RiskLevel.critical, seed=405)
    env.state.step_number = 3
    assert_float_close(env._process_score(), 1.0, "s5.optimal.legal_hold_critical")

    # diff matrix
    env = reset_forced(GTLabel.approve, RiskLevel.low, seed=406)
    env.state.step_number = 1
    assert_float_close(env._process_score(), 1.0, "s5.diff0")
    env.state.step_number = 2
    assert_float_close(env._process_score(), 0.6, "s5.diff1")
    env.state.step_number = 3
    assert_float_close(env._process_score(), 0.2, "s5.diff2")

    # path penalty cap
    env = reset_forced(GTLabel.approve, RiskLevel.low, seed=407)
    env.state.step_number = 1
    env.state.path_penalty_incurred = True
    assert_float_close(env._process_score(), 0.4, "s5.path_penalty_cap")


def section_6_hint_score_validation():
    gt = GTLabel.remove
    terminal_action = ActionType.remove
    env = reset_forced(gt, RiskLevel.medium, seed=501)
    env.state.step_number = 1
    env.state.budget_violated = True

    cases = [
        (None, 0.5),
        (ActionType.remove, 0.8),
        (ActionType.approve, 1.0),
    ]

    for hint, expected_hint_score in cases:
        env._initial_hint = hint
        grade = env._grade(terminal_action)
        correctness_component = 1.0 * 0.45
        speed_component = 1.0 * 0.10
        extracted = (grade - correctness_component - speed_component) / 0.20
        assert_float_close(extracted, expected_hint_score, f"s6.hint_case.{hint}")

    # hint==terminal_action AND hint!=GT -> 0.0
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=502)
    env.state.step_number = 1
    env.state.budget_violated = True
    env._initial_hint = ActionType.remove
    grade = env._grade(ActionType.remove)
    correctness_component = 0.5 * 0.45
    speed_component = 1.0 * 0.10
    extracted = (grade - correctness_component - speed_component) / 0.20
    assert_float_close(extracted, 0.0, "s6.hint_terminal_not_gt")

    # hint!=terminal_action AND hint==GT -> 0.3
    env = reset_forced(GTLabel.remove, RiskLevel.medium, seed=503)
    env.state.step_number = 1
    env.state.budget_violated = True
    env._initial_hint = ActionType.remove
    terminal_action = ActionType.approve
    grade = env._grade(terminal_action)
    correctness_component = 0.5 * 0.45
    speed_component = 1.0 * 0.10
    extracted = (grade - correctness_component - speed_component) / 0.20
    assert_float_close(extracted, 0.3, "s6.hint_not_terminal_but_gt")


def section_7_observation_invariants():
    env = reset_forced(GTLabel.legal_hold, RiskLevel.critical, seed=601)
    reset_obs = env._reset_observation
    base = snapshot_observation(reset_obs)

    assert reset_obs.step_number in {1, 2, 3}
    assert reset_obs.reviewer_overturn_rate is None

    obs2 = env.step(ActionType.escalate)
    assert obs2.done is False
    assert obs2.step_number in {1, 2, 3}
    assert_equal(obs2.stage, Stage.escalation_review, "s7.stage_step2")
    assert isinstance(obs2.reviewer_overturn_rate, float)
    assert 0.0 <= obs2.reviewer_overturn_rate <= 1.0
    after_step2 = snapshot_observation(obs2)
    assert_equal(after_step2, base, "s7.fields_unchanged_step2")

    obs3 = env.step(ActionType.legal_hold)
    assert obs3.done is False
    assert obs3.step_number in {1, 2, 3}
    assert_equal(obs3.stage, Stage.legal_review, "s7.stage_step3")
    assert isinstance(obs3.reviewer_overturn_rate, float)
    assert 0.0 <= obs3.reviewer_overturn_rate <= 1.0
    after_step3_non_terminal = snapshot_observation(obs3)
    assert_equal(after_step3_non_terminal, base, "s7.fields_unchanged_step3_non_terminal")

    obs_terminal = env.step(ActionType.approve)
    assert obs_terminal.done is True
    assert obs_terminal.step_number in {1, 2, 3}
    assert_equal(obs_terminal.stage, Stage.legal_review, "s7.stage_terminal")
    assert isinstance(obs_terminal.reviewer_overturn_rate, float)
    assert 0.0 <= obs_terminal.reviewer_overturn_rate <= 1.0
    after_terminal = snapshot_observation(obs_terminal)
    assert_equal(after_terminal, base, "s7.fields_unchanged_terminal")


def section_8_reward_distribution():
    rewards = []
    for seed in range(9000, 9500):
        env = TestableModGuardEnvironment()
        env.reset(seed=seed)
        done = False
        while not done:
            action = choose_random_action_with_env_rng(env)
            obs = env.step(action)
            done = obs.done
            if done:
                rewards.append(obs.reward)
    assert len(rewards) == 500, f"s8.reward_count | expected=500, actual={len(rewards)}"
    distinct = len(set(rewards))
    variance = statistics.variance(rewards)
    spread = max(rewards) - min(rewards)
    assert distinct >= 5, f"s8.distinct_rewards | expected>=5, actual={distinct}"
    assert variance > 0.01, f"s8.variance | expected>0.01, actual={variance}"
    assert spread > 0.5, f"s8.spread | expected>0.5, actual={spread}"
    return rewards


def section_9_inference_script_validation():
    hf_token_present = os.environ.get("HF_TOKEN") is not None
    docker_available = (
        subprocess.run(["docker", "info"], capture_output=True, text=True).returncode == 0
    )
    if not (hf_token_present and docker_available):
        print("SECTION 9 SKIPPED — requires HF_TOKEN env var and Docker daemon")
        return "skipped"

    proc = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        "s9.inference_exit_code | expected=0, "
        f"actual={proc.returncode}, stderr={proc.stderr.strip()}"
    )
    stdout = proc.stdout.strip()
    assert stdout != "", "s9.inference_stdout_non_empty | expected=non-empty, actual=empty"
    floats = parse_floats_from_text(stdout)
    assert len(floats) >= 1, "s9.reward_float_present | expected>=1 float in stdout, actual=0"
    return "passed"


def section_10_docker_and_project_validation():
    assert os.path.exists("Dockerfile"), "s10.dockerfile_root_missing | expected exists at root"
    assert not os.path.exists(
        os.path.join("server", "Dockerfile")
    ), "s10.server_dockerfile_present | expected no server/Dockerfile"
    assert os.path.exists("openenv.yaml"), "s10.openenv_yaml_missing | expected openenv.yaml at root"
    validate = subprocess.run(["openenv", "validate"], capture_output=True, text=True)
    assert validate.returncode == 0, (
        "s10.openenv_validate | expected exit_code=0, "
        f"actual={validate.returncode}, stderr={validate.stderr.strip()}"
    )


def section_11_stress_test():
    for seed in range(20000, 21000):
        env = TestableModGuardEnvironment()
        env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            steps += 1
            action = choose_random_action_with_env_rng(env)
            obs = env.step(action)
            assert obs.step_number <= 3, (
                f"s11.step_bound | expected<=3, actual={obs.step_number}, seed={seed}"
            )
            done = obs.done
        assert done is True, f"s11.episode_terminates | expected=True, actual={done}, seed={seed}"
        assert steps <= 3, f"s11.max_steps | expected<=3, actual={steps}, seed={seed}"


def section_12_edge_cases():
    # escalate with budget 0 -> budget_violated=True, no underflow
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=1201)
    env.state.escalation_budget = 0
    env.step(ActionType.escalate)
    assert env.state.budget_violated is True
    assert_equal(env.state.escalation_budget, 0, "s12.escalation_budget_underflow_guard")

    # step() after done -> AssertionError
    env = reset_forced(GTLabel.approve, RiskLevel.medium, seed=1202)
    env.step(ActionType.approve)
    raised = False
    try:
        env.step(ActionType.approve)
    except AssertionError:
        raised = True
    assert raised is True, "s12.step_after_done_assertion | expected AssertionError"

    # immediate terminal at step 1
    env = reset_forced(GTLabel.remove, RiskLevel.low, seed=1203)
    obs = env.step(ActionType.approve)
    assert obs.done is True
    assert_equal(obs.step_number, 1, "s12.immediate_terminal.approve")

    env = reset_forced(GTLabel.remove, RiskLevel.low, seed=1204)
    obs = env.step(ActionType.remove)
    assert obs.done is True
    assert_equal(obs.step_number, 1, "s12.immediate_terminal.remove")

    # maximum trajectory length legal_hold -> legal_hold -> legal_hold
    env = reset_forced(GTLabel.legal_hold, RiskLevel.critical, seed=1205)
    o1 = env.step(ActionType.legal_hold)
    o2 = env.step(ActionType.legal_hold)
    assert o1.done is False
    assert (
        o2.done is False
    ), "s12.max_length_trajectory.step2_non_terminal | expected=False, actual=True"
    o3 = env.step(ActionType.legal_hold)
    assert o3.done is True
    assert_equal(o3.step_number, 3, "s12.max_length_trajectory.step")


def run_section(name, fn, results):
    try:
        output = fn()
        if output == "skipped":
            results["skipped"] += 1
            print(f"[SKIP] {name}")
        else:
            results["passed"] += 1
            print(f"[PASS] {name}")
        return output
    except Exception as exc:
        results["failed"] += 1
        print(f"[FAIL] {name}")
        print(f"  assertion/error: {exc}")
        return None


def print_report(results, rewards, trigger_coverage):
    rewards = rewards if rewards else [0.0]
    rewards_min = min(rewards)
    rewards_max = max(rewards)
    rewards_mean = statistics.mean(rewards)
    rewards_variance = 0.0 if len(rewards) < 2 else statistics.variance(rewards)
    rewards_distinct = len(set(rewards))
    final_result = "PASS" if results["failed"] == 0 else "FAIL"

    print("\n══════════════════════════════════════")
    print("ModGuard-RL Validation Report")
    print("══════════════════════════════════════")
    print(f"Sections passed : {results['passed']} / {SECTIONS_TOTAL}")
    print(f"Sections failed : {results['failed']}")
    print(f"Sections skipped: {results['skipped']}")
    print("")
    print("Reward statistics (500-episode run):")
    print(f"  min      : {rewards_min:.3f}")
    print(f"  max      : {rewards_max:.3f}")
    print(f"  mean     : {rewards_mean:.3f}")
    print(f"  variance : {rewards_variance:.3f}")
    print(f"  distinct : {rewards_distinct}")
    print("")
    print("Trigger coverage:")
    print(f"  T1 fired: {'yes' if trigger_coverage['T1'] else 'no'}")
    print(f"  T2 fired: {'yes' if trigger_coverage['T2'] else 'no'}")
    print(f"  T3 fired: {'yes' if trigger_coverage['T3'] else 'no'}")
    print(f"  T4 fired: {'yes' if trigger_coverage['T4'] else 'no'}")
    print(f"  T5 fired: {'yes' if trigger_coverage['T5'] else 'no'}")
    print("")
    print(f"Result: {final_result}")
    print("══════════════════════════════════════")


def main():
    random.seed(42)
    results = {"passed": 0, "failed": 0, "skipped": 0}
    trigger_coverage = {"T1": False, "T2": False, "T3": False, "T4": False, "T5": False}
    rewards = []

    run_section("SECTION 1 — DETERMINISM", section_1_determinism, results)
    run_section("SECTION 2 — STATE MACHINE COVERAGE", section_2_state_machine_coverage, results)
    run_section(
        "SECTION 3 — TRIGGER VALIDATION",
        lambda: section_3_trigger_validation(trigger_coverage),
        results,
    )
    run_section("SECTION 4 — REWARD VALIDATION", section_4_reward_validation, results)
    run_section("SECTION 5 — PROCESS SCORE VALIDATION", section_5_process_score_validation, results)
    run_section("SECTION 6 — HINT SCORE VALIDATION", section_6_hint_score_validation, results)
    run_section("SECTION 7 — OBSERVATION INVARIANTS", section_7_observation_invariants, results)
    section8_result = run_section("SECTION 8 — REWARD DISTRIBUTION", section_8_reward_distribution, results)
    if isinstance(section8_result, list):
        rewards = section8_result
    run_section("SECTION 9 — INFERENCE SCRIPT VALIDATION", section_9_inference_script_validation, results)
    run_section("SECTION 10 — DOCKER AND PROJECT VALIDATION", section_10_docker_and_project_validation, results)
    run_section("SECTION 11 — STRESS TESTING", section_11_stress_test, results)
    run_section("SECTION 12 — EDGE CASES", section_12_edge_cases, results)

    print_report(results, rewards, trigger_coverage)

    assert results["failed"] == 0, f"Validation failed with {results['failed']} section(s) failing"


if __name__ == "__main__":
    main()
