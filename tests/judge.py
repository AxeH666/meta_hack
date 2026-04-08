from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.environment import ModGuardEnvironment
from server.models import ActionType, ModGuardAction, Stage

PROJECT_ROOT = Path(__file__).parent.parent


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def pass_fail_line(check_id: str, description: str, passed: bool, reason: str) -> str:
    if passed:
        return f"  ✓ {check_id}: {description}"
    return f"  ✗ {check_id}: {description} — FAIL: {reason}"


def batch_rollout(num_episodes: int) -> dict[str, object]:
    rewards = []
    steps = []
    action_lengths = []
    for seed in range(num_episodes):
        chooser = random.Random(seed + 7)
        env = ModGuardEnvironment()
        obs = env.reset(seed=seed)
        episode_steps = 0
        while not obs.done:
            obs = env.step(ModGuardAction(action=chooser.choice(list(ActionType))))
            episode_steps += 1
            if episode_steps > 20:
                return {"ok": False, "error": f"episode {seed} exceeded safety limit"}
        rewards.append(float(obs.reward or 0.0))
        steps.append(episode_steps)
        action_lengths.append(len(env.get_state().action_history))
    return {
        "ok": True,
        "rewards": rewards,
        "steps": steps,
        "action_lengths": action_lengths,
    }


def main() -> int:
    readme_text = safe_read_text(PROJECT_ROOT / "README.md").lower()
    docker_text = safe_read_text(PROJECT_ROOT / "Dockerfile")
    env_text = safe_read_text(PROJECT_ROOT / "server" / "environment.py")
    app_text = safe_read_text(PROJECT_ROOT / "server" / "app.py")
    inf_text = safe_read_text(PROJECT_ROOT / "inference.py")
    models_text = safe_read_text(PROJECT_ROOT / "server" / "models.py")
    openenv_text = safe_read_text(PROJECT_ROOT / "openenv.yaml")

    checks = []

    batch = batch_rollout(150)
    if not batch["ok"]:
        checks.append(("R1", "Random rollouts complete without safety overflow", False, str(batch["error"])))
    else:
        rewards = batch["rewards"]  # type: ignore[assignment]
        steps = batch["steps"]  # type: ignore[assignment]
        action_lengths = batch["action_lengths"]  # type: ignore[assignment]
        checks.append(("R1", "Random rollouts complete without safety overflow", True, ""))
        checks.append(("R2", "All rewards are within [0, 1]", all(0.0 <= r <= 1.0 for r in rewards), "reward out of bounds"))
        checks.append(("R3", "Reward variance is meaningful", statistics.pvariance(rewards) > 0.01, "variance too small"))
        checks.append(("R4", "At least one low-score case exists", any(r <= 0.30 for r in rewards), "no low-score episodes"))
        checks.append(("R5", "At least one high-score case exists", any(r >= 0.85 for r in rewards), "no high-score episodes"))
        checks.append(("R6", "Episode depth never exceeds 4", all(length <= 4 for length in action_lengths), "action history exceeded 4"))
        checks.append(("R7", "At least 25% of episodes are multi-step", sum(1 for s in steps if s > 1) / len(steps) >= 0.25, "too few multi-step cases"))
        checks.append(("R8", "At least one episode reaches 3+ steps", any(s >= 3 for s in steps), "no deep trajectories"))

    try:
        env = ModGuardEnvironment()
        obs = env.reset(seed=3)
        state_dump = env.get_state().model_dump()
        checks.append(("C1", "Reset returns initial_review at step 1", obs.stage == Stage.initial_review and obs.step_number == 1, "unexpected initial stage"))
        checks.append(("C2", "Public state excludes hidden ground_truth", "ground_truth" not in state_dump, "ground_truth leaked in state"))
        checks.append(("C3", "Observation metadata includes audit/uncertainty info", {"ai_recommendation", "uncertainty_index", "signal_conflict_score"}.issubset(obs.metadata.keys()), "missing metadata fields"))
    except Exception as exc:
        checks.append(("C1", "Reset returns initial_review at step 1", False, str(exc)))
        checks.append(("C2", "Public state excludes hidden ground_truth", False, str(exc)))
        checks.append(("C3", "Observation metadata includes audit/uncertainty info", False, str(exc)))

    checks.append(("D1", "README describes content moderation motivation", "moderation" in readme_text and "trust-and-safety" in readme_text.replace(" ", "-"), "README missing motivation keywords"))
    checks.append(("D2", "README explains reward design", "reward design" in readme_text or "reward is continuous" in readme_text, "README missing reward explanation"))
    checks.append(("D3", "README documents audit stage", "post-decision audit" in readme_text, "README missing audit stage"))
    checks.append(("D4", "README documents 1 to 4 steps", "1 to 4" in readme_text or "1–4" in readme_text, "README missing trajectory depth"))

    checks.append(("S1", "Models allow step_number up to 4", "le=4" in models_text, "step_number upper bound not updated"))
    checks.append(("S2", "Audit stage is present", "post_decision_audit" in models_text and "post_decision_audit" in env_text, "audit stage missing"))
    checks.append(("S3", "Environment contains overconfidence penalty", "OVERCONFIDENCE_PENALTY_WEIGHT" in env_text, "overconfidence penalty missing"))
    checks.append(("S4", "Environment contains consistency scoring", "_consistency_score" in env_text, "consistency score missing"))

    checks.append(("H1", "Dockerfile defaults to RUN_MODE=serve", "ENV RUN_MODE=serve" in docker_text, "RUN_MODE default missing"))
    checks.append(("H2", "Dockerfile starts uvicorn on 0.0.0.0", "uvicorn" in docker_text and "0.0.0.0" in safe_read_text(PROJECT_ROOT / "docker-entrypoint.sh"), "server startup command missing"))
    checks.append(("H3", "Dockerfile targets port 7860", "PORT=7860" in docker_text, "port 7860 missing"))
    checks.append(("H4", "App exposes /health", "@app.get(\"/health\"" in app_text, "/health endpoint missing"))
    checks.append(("H5", "App exposes /reset, /step, and /state", all(token in app_text for token in ["/reset", "/step", "/state"]), "simulation endpoints missing"))
    checks.append(("H6", "openenv.yaml uses port 7860", "port: 7860" in openenv_text, "openenv port not updated"))

    checks.append(("I1", "inference.py supports RUN_MODE eval/serve split", "RUN_MODE" in inf_text and "_should_serve_forever" in inf_text, "RUN_MODE logic missing"))
    checks.append(("I2", "inference.py does not spawn subprocess servers", "subprocess" not in inf_text, "subprocess usage found"))
    checks.append(("I3", "inference.py supports in-process fallback", "InProcessModGuardClient" in inf_text and "falling back to in-process environment" in inf_text, "in-process fallback missing"))
    checks.append(("I4", "inference.py emits structured summary JSON", "json.dumps(summary" in inf_text, "summary output missing"))

    passed = 0
    print("ModGuard-RL validation")
    for check_id, description, ok, reason in checks:
        print(pass_fail_line(check_id, description, ok, reason))
        passed += int(ok)

    total = len(checks)
    print(f"\nScore: {passed}/{total} checks passed")

    eval_cmd = [
        "python3",
        str(PROJECT_ROOT / "inference.py"),
    ]
    try:
        proc = subprocess.run(
            ["env", "RUN_MODE=eval", "FORCE_INPROCESS=1", "NUM_EPISODES_PER_TASK=1", *eval_cmd],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        print("\nEval smoke exit code:", proc.returncode)
        stderr_tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        if stderr_tail:
            try:
                parsed = json.loads(stderr_tail)
                print("Eval aggregate:", parsed.get("overall", {}).get("aggregate"))
            except Exception:
                print("Eval stderr tail:", stderr_tail)
    except Exception as exc:
        print("\nEval smoke failed:", exc)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
