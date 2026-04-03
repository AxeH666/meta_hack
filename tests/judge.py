from pathlib import Path
import os
import re
import math
import statistics
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.environment import ModGuardEnvironment
from server.models import (ActionType, ModGuardAction, ModGuardObservation, GTLabel, Stage, RiskLevel)


PROJECT_ROOT = Path(__file__).parent.parent


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def run_batch(num_episodes: int):
    try:
        rng = random.Random(42)
        rewards = []
        episodes = []
        for i in range(num_episodes):
            env = ModGuardEnvironment()
            env.reset(seed=i)
            done = False
            steps = 0
            terminal_reward = None
            while not done:
                action = ModGuardAction(action=rng.choice(list(ActionType)))
                obs = env.step(action)
                done = bool(obs.done)
                steps += 1
                if done:
                    terminal_reward = float(obs.reward if obs.reward is not None else 0.0)
                    break
                if steps > 50:
                    raise RuntimeError(f"Episode {i} exceeded safety step limit")
            state = env.get_state()
            history_len = len(state.action_history)
            rewards.append(terminal_reward)
            episodes.append(
                {
                    "episode_index": i,
                    "steps": steps,
                    "reward": terminal_reward,
                    "action_history_len": history_len,
                }
            )
        return {"ok": True, "rewards": rewards, "episodes": episodes, "error": ""}
    except Exception as e:
        return {"ok": False, "rewards": [], "episodes": [], "error": f"ERROR: {e}"}


def pass_fail_line(check_id: str, description: str, passed: bool, reason: str) -> str:
    if passed:
        return f"  ✓ {check_id}: {description}"
    return f"  ✗ {check_id}: {description} — FAIL: {reason}"


def score_dimension(results):
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    if total == 0:
        return 0.0, passed, total
    score = (passed / total) * 10.0
    return min(10.0, score), passed, total


def check_window_tokens(text: str, tokens, window_size: int = 300) -> bool:
    try:
        if not text:
            return False
        lower_text = text.lower()
        token_list = [t.lower() for t in tokens]
        if len(lower_text) <= window_size:
            windows = [lower_text]
        else:
            windows = [lower_text[i:i + window_size] for i in range(0, len(lower_text) - window_size + 1)]
        for window in windows:
            all_present = True
            for token in token_list:
                if re.search(re.escape(token), window, re.IGNORECASE | re.DOTALL) is None:
                    all_present = False
                    break
            if all_present:
                return True
        return False
    except Exception:
        return False


def main() -> int:
    _ = os
    _ = math
    _ = subprocess
    _ = GTLabel
    _ = Stage
    _ = RiskLevel

    readme_path = PROJECT_ROOT / "README.md"
    env_path = PROJECT_ROOT / "server" / "environment.py"
    models_path = PROJECT_ROOT / "server" / "models.py"
    inference_path = PROJECT_ROOT / "inference.py"
    dockerfile_path = PROJECT_ROOT / "Dockerfile"
    openenv_path = PROJECT_ROOT / "openenv.yaml"

    readme_text = safe_read_text(readme_path)
    env_text = safe_read_text(env_path)
    models_text = safe_read_text(models_path)
    inf_text = safe_read_text(inference_path)
    docker_text = safe_read_text(dockerfile_path)
    openenv_text = safe_read_text(openenv_path)

    # Run programmatic batches before printing anything.
    batch_200 = run_batch(200)
    batch_500 = run_batch(500)

    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    checklist = []
    gaps = []

    # ---------------- D1 ----------------
    try:
        passed = ("content moderation" in readme_text.lower()) or ("content policy" in readme_text.lower())
        reason = "README missing content moderation/policy phrase"
        d1.append({"id": "C1", "desc": "README contains 'content moderation' or 'content policy' (case-insensitive)", "passed": passed, "reason": reason})
    except Exception as e:
        d1.append({"id": "C1", "desc": "README contains 'content moderation' or 'content policy' (case-insensitive)", "passed": False, "reason": f"ERROR: {e}"})
    try:
        lower_readme = readme_text.lower()
        passed = ("game" not in lower_readme) and ("toy" not in lower_readme) and ("puzzle" not in lower_readme)
        reason = "README contains one of: game, toy, puzzle"
        d1.append({"id": "C2", "desc": "README contains none of 'game', 'toy', 'puzzle' (case-insensitive)", "passed": passed, "reason": reason})
    except Exception as e:
        d1.append({"id": "C2", "desc": "README contains none of 'game', 'toy', 'puzzle' (case-insensitive)", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = ("approve" in env_text) and ("remove" in env_text) and ("legal_hold" in env_text)
        reason = "server/environment.py missing one of approve/remove/legal_hold"
        d1.append({"id": "C3", "desc": "server/environment.py contains approve, remove, and legal_hold", "passed": passed, "reason": reason})
    except Exception as e:
        d1.append({"id": "C3", "desc": "server/environment.py contains approve, remove, and legal_hold", "passed": False, "reason": f"ERROR: {e}"})
    try:
        lower_readme = readme_text.lower()
        passed = any(x in lower_readme for x in ["enterprise", "trust", "safety", "triage"])
        reason = "README missing enterprise/trust/safety/triage keywords"
        d1.append({"id": "C4", "desc": "README contains enterprise/trust/safety/triage keyword (case-insensitive)", "passed": passed, "reason": reason})
    except Exception as e:
        d1.append({"id": "C4", "desc": "README contains enterprise/trust/safety/triage keyword (case-insensitive)", "passed": False, "reason": f"ERROR: {e}"})

    # ---------------- D2 ----------------
    if not batch_200["ok"]:
        err = batch_200["error"]
        for cid, desc in [
            ("C1", "All terminal rewards in [0.0, 1.0]"),
            ("C2", "At least 5 distinct rounded reward values across 200 episodes"),
            ("C3", "statistics.variance(rewards) > 0.01"),
            ("C4", "max(rewards) - min(rewards) > 0.3"),
            ("C5", "At least one reward >= 0.8"),
            ("C6", "At least one reward <= 0.3"),
        ]:
            d2.append({"id": cid, "desc": desc, "passed": False, "reason": err})
        rewards_200 = []
    else:
        rewards_200 = batch_200["rewards"]
        try:
            passed = all((r >= 0.0 and r <= 1.0) for r in rewards_200)
            d2.append({"id": "C1", "desc": "All terminal rewards in [0.0, 1.0]", "passed": passed, "reason": "Found reward outside [0.0, 1.0]"})
        except Exception as e:
            d2.append({"id": "C1", "desc": "All terminal rewards in [0.0, 1.0]", "passed": False, "reason": f"ERROR: {e}"})
        try:
            passed = len(set(round(r, 3) for r in rewards_200)) >= 5
            d2.append({"id": "C2", "desc": "At least 5 distinct rounded reward values across 200 episodes", "passed": passed, "reason": "Distinct rounded rewards < 5"})
        except Exception as e:
            d2.append({"id": "C2", "desc": "At least 5 distinct rounded reward values across 200 episodes", "passed": False, "reason": f"ERROR: {e}"})
        try:
            passed = statistics.variance(rewards_200) > 0.01
            d2.append({"id": "C3", "desc": "statistics.variance(rewards) > 0.01", "passed": passed, "reason": "Variance <= 0.01"})
        except Exception as e:
            d2.append({"id": "C3", "desc": "statistics.variance(rewards) > 0.01", "passed": False, "reason": f"ERROR: {e}"})
        try:
            passed = (max(rewards_200) - min(rewards_200)) > 0.3
            d2.append({"id": "C4", "desc": "max(rewards) - min(rewards) > 0.3", "passed": passed, "reason": "Reward range <= 0.3"})
        except Exception as e:
            d2.append({"id": "C4", "desc": "max(rewards) - min(rewards) > 0.3", "passed": False, "reason": f"ERROR: {e}"})
        try:
            passed = any(r >= 0.8 for r in rewards_200)
            d2.append({"id": "C5", "desc": "At least one reward >= 0.8", "passed": passed, "reason": "No reward >= 0.8"})
        except Exception as e:
            d2.append({"id": "C5", "desc": "At least one reward >= 0.8", "passed": False, "reason": f"ERROR: {e}"})
        try:
            passed = any(r <= 0.3 for r in rewards_200)
            d2.append({"id": "C6", "desc": "At least one reward <= 0.3", "passed": passed, "reason": "No reward <= 0.3"})
        except Exception as e:
            d2.append({"id": "C6", "desc": "At least one reward <= 0.3", "passed": False, "reason": f"ERROR: {e}"})

    # ---------------- D3 ----------------
    try:
        env = ModGuardEnvironment()
        a = env.reset(seed=99)
        b = env.reset(seed=99)
        fields = [
            "content_category",
            "risk_level",
            "platform_context",
            "ai_confidence_score",
            "human_reviewer_hint",
            "queue_pressure",
            "step_number",
            "stage",
        ]
        passed = all(getattr(a, f) == getattr(b, f) for f in fields)
        d3.append({"id": "C1", "desc": "reset(seed=99) twice matches 8 specified fields", "passed": passed, "reason": "One or more compared fields differ"})
    except Exception as e:
        d3.append({"id": "C1", "desc": "reset(seed=99) twice matches 8 specified fields", "passed": False, "reason": f"ERROR: {e}"})
    try:
        env = ModGuardEnvironment()
        obs = env.reset()
        passed = (obs.step_number == 1) and (obs.stage.value == "initial_review") and (obs.done is False)
        d3.append({"id": "C2", "desc": "reset() returns step_number=1, stage=initial_review, done=False", "passed": passed, "reason": "reset() state values not as expected"})
    except Exception as e:
        d3.append({"id": "C2", "desc": "reset() returns step_number=1, stage=initial_review, done=False", "passed": False, "reason": f"ERROR: {e}"})
    try:
        env = ModGuardEnvironment()
        env.reset(seed=0)
        first_obs = env.step(ModGuardAction(action=ActionType.escalate))
        if first_obs.done:
            env = ModGuardEnvironment()
            env.reset(seed=0)
            first_obs = env.step(ModGuardAction(action=ActionType.legal_hold))
        passed = (first_obs.reward == 0.0) and (first_obs.done is False)
        d3.append({"id": "C3", "desc": "First non-terminal transition returns reward=0.0 and done=False", "passed": passed, "reason": "Step was terminal or reward != 0.0"})
    except Exception as e:
        d3.append({"id": "C3", "desc": "First non-terminal transition returns reward=0.0 and done=False", "passed": False, "reason": f"ERROR: {e}"})
    try:
        env = ModGuardEnvironment()
        obs = env.reset(seed=1)
        rng_local = random.Random(42)
        guard = 0
        while not obs.done:
            obs = env.step(ModGuardAction(action=rng_local.choice(list(ActionType))))
            guard += 1
            if guard > 50:
                raise RuntimeError("Exceeded safety loop in terminal-step test")
        passed = (obs.done is True) and (obs.reward is not None) and (0.0 <= float(obs.reward) <= 1.0)
        d3.append({"id": "C4", "desc": "Terminal step returns done=True and reward in [0.0, 1.0]", "passed": passed, "reason": "Terminal step constraints not met"})
    except Exception as e:
        d3.append({"id": "C4", "desc": "Terminal step returns done=True and reward in [0.0, 1.0]", "passed": False, "reason": f"ERROR: {e}"})
    try:
        keys = set(ModGuardObservation.model_fields.keys())
        required = {
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
        passed = required.issubset(keys)
        d3.append({"id": "C5", "desc": "ModGuardObservation.model_fields includes all 10 spec fields", "passed": passed, "reason": "Missing one or more required spec fields"})
    except Exception as e:
        d3.append({"id": "C5", "desc": "ModGuardObservation.model_fields includes all 10 spec fields", "passed": False, "reason": f"ERROR: {e}"})
    try:
        values = set(a.value for a in ActionType)
        passed = values == {"approve", "remove", "escalate", "legal_hold"}
        d3.append({"id": "C6", "desc": "ActionType values exactly match approve/remove/escalate/legal_hold", "passed": passed, "reason": f"Unexpected ActionType values: {sorted(values)}"})
    except Exception as e:
        d3.append({"id": "C6", "desc": "ActionType values exactly match approve/remove/escalate/legal_hold", "passed": False, "reason": f"ERROR: {e}"})
    if not batch_500["ok"]:
        err = batch_500["error"]
        d3.append({"id": "C7", "desc": "500-episode batch: no episode exceeds 3 actions in action_history", "passed": False, "reason": err})
        d3.append({"id": "C8", "desc": "500-episode batch: at least 20% of episodes have steps > 1", "passed": False, "reason": err})
    else:
        episodes_500 = batch_500["episodes"]
        try:
            passed = all(e["action_history_len"] <= 3 for e in episodes_500)
            d3.append({"id": "C7", "desc": "500-episode batch: no episode exceeds 3 actions in action_history", "passed": passed, "reason": "Found action_history length > 3"})
        except Exception as e:
            d3.append({"id": "C7", "desc": "500-episode batch: no episode exceeds 3 actions in action_history", "passed": False, "reason": f"ERROR: {e}"})
        try:
            ratio = sum(1 for e in episodes_500 if e["steps"] > 1) / 500.0
            passed = ratio >= 0.2
            d3.append({"id": "C8", "desc": "500-episode batch: at least 20% of episodes have steps > 1", "passed": passed, "reason": f"Multi-step ratio {ratio:.3f} < 0.2"})
        except Exception as e:
            d3.append({"id": "C8", "desc": "500-episode batch: at least 20% of episodes have steps > 1", "passed": False, "reason": f"ERROR: {e}"})

    # ---------------- D4 ----------------
    try:
        passed = dockerfile_path.exists()
        d4.append({"id": "C1", "desc": "PROJECT_ROOT/Dockerfile exists", "passed": passed, "reason": "Dockerfile missing at project root"})
    except Exception as e:
        d4.append({"id": "C1", "desc": "PROJECT_ROOT/Dockerfile exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = openenv_path.exists()
        d4.append({"id": "C2", "desc": "PROJECT_ROOT/openenv.yaml exists", "passed": passed, "reason": "openenv.yaml missing at project root"})
    except Exception as e:
        d4.append({"id": "C2", "desc": "PROJECT_ROOT/openenv.yaml exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = inference_path.exists()
        d4.append({"id": "C3", "desc": "PROJECT_ROOT/inference.py exists", "passed": passed, "reason": "inference.py missing at project root"})
    except Exception as e:
        d4.append({"id": "C3", "desc": "PROJECT_ROOT/inference.py exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = (PROJECT_ROOT / "server" / "__init__.py").exists()
        d4.append({"id": "C4", "desc": "PROJECT_ROOT/server/__init__.py exists", "passed": passed, "reason": "server/__init__.py missing"})
    except Exception as e:
        d4.append({"id": "C4", "desc": "PROJECT_ROOT/server/__init__.py exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        p = PROJECT_ROOT / "client" / "__init__.py"
        passed = p.exists()
        reason = "create empty file at client/__init__.py"
        d4.append({"id": "C5", "desc": "PROJECT_ROOT/client/__init__.py exists", "passed": passed, "reason": reason})
    except Exception as e:
        d4.append({"id": "C5", "desc": "PROJECT_ROOT/client/__init__.py exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = (PROJECT_ROOT / "client" / "client.py").exists()
        d4.append({"id": "C6", "desc": "PROJECT_ROOT/client/client.py exists", "passed": passed, "reason": "client/client.py missing"})
    except Exception as e:
        d4.append({"id": "C6", "desc": "PROJECT_ROOT/client/client.py exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = readme_path.exists()
        d4.append({"id": "C7", "desc": "PROJECT_ROOT/README.md exists", "passed": passed, "reason": "README.md missing"})
    except Exception as e:
        d4.append({"id": "C7", "desc": "PROJECT_ROOT/README.md exists", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "ENABLE_WEB_INTERFACE=true" in docker_text
        d4.append({"id": "C8", "desc": "Dockerfile contains ENABLE_WEB_INTERFACE=true", "passed": passed, "reason": "ENABLE_WEB_INTERFACE=true not found"})
    except Exception as e:
        d4.append({"id": "C8", "desc": "Dockerfile contains ENABLE_WEB_INTERFACE=true", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = not (PROJECT_ROOT / "server" / "Dockerfile").exists()
        d4.append({"id": "C9", "desc": "server/Dockerfile does not exist", "passed": passed, "reason": "server/Dockerfile exists but should be absent"})
    except Exception as e:
        d4.append({"id": "C9", "desc": "server/Dockerfile does not exist", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "[START]" in inf_text
        d4.append({"id": "C10", "desc": "inference.py contains [START]", "passed": passed, "reason": "[START] not found"})
    except Exception as e:
        d4.append({"id": "C10", "desc": "inference.py contains [START]", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "[STEP]" in inf_text
        d4.append({"id": "C11", "desc": "inference.py contains [STEP]", "passed": passed, "reason": "[STEP] not found"})
    except Exception as e:
        d4.append({"id": "C11", "desc": "inference.py contains [STEP]", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "[END]" in inf_text
        d4.append({"id": "C12", "desc": "inference.py contains [END]", "passed": passed, "reason": "[END] not found"})
    except Exception as e:
        d4.append({"id": "C12", "desc": "inference.py contains [END]", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "API_BASE_URL" in inf_text
        d4.append({"id": "C13", "desc": "inference.py contains API_BASE_URL", "passed": passed, "reason": "API_BASE_URL not found"})
    except Exception as e:
        d4.append({"id": "C13", "desc": "inference.py contains API_BASE_URL", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "MODEL_NAME" in inf_text
        d4.append({"id": "C14", "desc": "inference.py contains MODEL_NAME", "passed": passed, "reason": "MODEL_NAME not found"})
    except Exception as e:
        d4.append({"id": "C14", "desc": "inference.py contains MODEL_NAME", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "HF_TOKEN" in inf_text
        d4.append({"id": "C15", "desc": "inference.py contains HF_TOKEN", "passed": passed, "reason": "HF_TOKEN not found"})
    except Exception as e:
        d4.append({"id": "C15", "desc": "inference.py contains HF_TOKEN", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "load_dotenv" in inf_text
        d4.append({"id": "C16", "desc": "inference.py contains load_dotenv", "passed": passed, "reason": "load_dotenv not found"})
    except Exception as e:
        d4.append({"id": "C16", "desc": "inference.py contains load_dotenv", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "router.huggingface.co" in inf_text
        d4.append({"id": "C17", "desc": "inference.py contains router.huggingface.co", "passed": passed, "reason": "router.huggingface.co not found"})
    except Exception as e:
        d4.append({"id": "C17", "desc": "inference.py contains router.huggingface.co", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "api.openai.com" not in inf_text
        d4.append({"id": "C18", "desc": "inference.py does not contain api.openai.com", "passed": passed, "reason": "api.openai.com found in inference.py"})
    except Exception as e:
        d4.append({"id": "C18", "desc": "inference.py does not contain api.openai.com", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "NUM_EPISODES" in inf_text
        d4.append({"id": "C19", "desc": "inference.py contains NUM_EPISODES", "passed": passed, "reason": "NUM_EPISODES not found"})
    except Exception as e:
        d4.append({"id": "C19", "desc": "inference.py contains NUM_EPISODES", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "entry_point" in openenv_text
        d4.append({"id": "C20", "desc": "openenv.yaml contains entry_point", "passed": passed, "reason": "entry_point not found in openenv.yaml"})
    except Exception as e:
        d4.append({"id": "C20", "desc": "openenv.yaml contains entry_point", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "from server.models import *" not in env_text
        d4.append({"id": "C21", "desc": "server/environment.py does not use wildcard import from server.models", "passed": passed, "reason": "Wildcard import found in server/environment.py"})
    except Exception as e:
        d4.append({"id": "C21", "desc": "server/environment.py does not use wildcard import from server.models", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = ("action_schema" in openenv_text) or ("observation_schema" in openenv_text)
        d4.append({"id": "C22", "desc": "openenv.yaml contains action_schema or observation_schema", "passed": passed, "reason": "Neither action_schema nor observation_schema found"})
    except Exception as e:
        d4.append({"id": "C22", "desc": "openenv.yaml contains action_schema or observation_schema", "passed": False, "reason": f"ERROR: {e}"})

    # ---------------- D5 ----------------
    try:
        passed = all(token in env_text for token in ["0.45", "0.25", "0.20", "0.10"])
        d5.append({"id": "C1", "desc": "server/environment.py contains grade weights 0.45, 0.25, 0.20, 0.10", "passed": passed, "reason": "One or more required grade weights missing"})
    except Exception as e:
        d5.append({"id": "C1", "desc": "server/environment.py contains grade weights 0.45, 0.25, 0.20, 0.10", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "path_penalty_incurred" in env_text
        d5.append({"id": "C2", "desc": "server/environment.py contains path_penalty_incurred", "passed": passed, "reason": "path_penalty_incurred not found"})
    except Exception as e:
        d5.append({"id": "C2", "desc": "server/environment.py contains path_penalty_incurred", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = ("escalation_review" in env_text) and ("legal_review" in env_text)
        d5.append({"id": "C3", "desc": "server/environment.py contains escalation_review and legal_review", "passed": passed, "reason": "Missing escalation_review or legal_review"})
    except Exception as e:
        d5.append({"id": "C3", "desc": "server/environment.py contains escalation_review and legal_review", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "1.0 - base" in env_text
        d5.append({"id": "C4", "desc": "server/environment.py contains adversarial confidence inversion '1.0 - base'", "passed": passed, "reason": "'1.0 - base' not found"})
    except Exception as e:
        d5.append({"id": "C4", "desc": "server/environment.py contains adversarial confidence inversion '1.0 - base'", "passed": False, "reason": f"ERROR: {e}"})
    try:
        triggers = {
            "T1": ["escalation_review", "escalate", "path_penalty"],
            "T2": ["escalation_review", "legal_hold", "path_penalty"],
            "T3": ["legal_review", "escalate", "path_penalty"],
            "T4": ["legal_review", "legal_hold", "path_penalty"],
            "T5": ["action_history", "legal_hold"],
        }
        hit_count = 0
        details = []
        for key, tokens in triggers.items():
            hit = check_window_tokens(env_text, tokens, window_size=300)
            if hit:
                hit_count += 1
            details.append(f"{key}={hit}")
        passed = hit_count >= 4
        d5.append({"id": "C5", "desc": "At least 4/5 path-penalty trigger token groups found within 300-char windows", "passed": passed, "reason": f"Found {hit_count}/5 triggers ({', '.join(details)})"})
    except Exception as e:
        d5.append({"id": "C5", "desc": "At least 4/5 path-penalty trigger token groups found within 300-char windows", "passed": False, "reason": f"ERROR: {e}"})

    # ---------------- CHECKLIST (informational) ----------------
    try:
        checklist.append({"id": "P1", "desc": "Dockerfile at project root", "passed": (PROJECT_ROOT / "Dockerfile").exists(), "reason": "Dockerfile missing at project root"})
    except Exception as e:
        checklist.append({"id": "P1", "desc": "Dockerfile at project root", "passed": False, "reason": f"ERROR: {e}"})
    try:
        checklist.append({"id": "P2", "desc": "inference.py at project root", "passed": (PROJECT_ROOT / "inference.py").exists(), "reason": "inference.py missing at project root"})
    except Exception as e:
        checklist.append({"id": "P2", "desc": "inference.py at project root", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = re.search(r"\[START\] task=\S+ env=\S+ model=\S+", inf_text) is not None
        checklist.append({"id": "P3", "desc": "[START] log format correct", "passed": passed, "reason": "START pattern not found"})
    except Exception as e:
        checklist.append({"id": "P3", "desc": "[START] log format correct", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = re.search(r"\[STEP\].*step=\d+.*action=\S+.*reward=[\d.]+.*done=(true|false).*error=\S+", inf_text) is not None
        checklist.append({"id": "P4", "desc": "[STEP] log format correct", "passed": passed, "reason": "STEP pattern not found"})
    except Exception as e:
        checklist.append({"id": "P4", "desc": "[STEP] log format correct", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = re.search(r"\[END\].*success=(true|false).*steps=\d+.*score=[\d.]+.*rewards=[\d.,]+", inf_text) is not None
        checklist.append({"id": "P5", "desc": "[END] log format correct", "passed": passed, "reason": "END pattern not found"})
    except Exception as e:
        checklist.append({"id": "P5", "desc": "[END] log format correct", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "SUCCESS_SCORE_THRESHOLD" in inf_text
        checklist.append({"id": "P6", "desc": "SUCCESS_SCORE_THRESHOLD defined", "passed": passed, "reason": "SUCCESS_SCORE_THRESHOLD not found"})
    except Exception as e:
        checklist.append({"id": "P6", "desc": "SUCCESS_SCORE_THRESHOLD defined", "passed": False, "reason": f"ERROR: {e}"})
    try:
        m = re.search(r"MAX_STEPS\s*=\s*(\d+)", inf_text)
        passed = (m is not None) and (int(m.group(1)) <= 3)
        checklist.append({"id": "P7", "desc": "MAX_STEPS <= 3", "passed": passed, "reason": "MAX_STEPS missing or > 3"})
    except Exception as e:
        checklist.append({"id": "P7", "desc": "MAX_STEPS <= 3", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "NUM_EPISODES" in inf_text
        checklist.append({"id": "P8", "desc": "NUM_EPISODES defined", "passed": passed, "reason": "NUM_EPISODES not found"})
    except Exception as e:
        checklist.append({"id": "P8", "desc": "NUM_EPISODES defined", "passed": False, "reason": f"ERROR: {e}"})
    try:
        lower_inf = inf_text.lower()
        passed = ("import torch" not in lower_inf) and ("import tensorflow" not in lower_inf)
        checklist.append({"id": "P9", "desc": "No GPU dependencies (torch/tensorflow imports)", "passed": passed, "reason": "Found torch/tensorflow import in inference.py"})
    except Exception as e:
        checklist.append({"id": "P9", "desc": "No GPU dependencies (torch/tensorflow imports)", "passed": False, "reason": f"ERROR: {e}"})
    try:
        passed = "self.rng" in env_text
        checklist.append({"id": "P10", "desc": "Deterministic RNG (self.rng) present in environment.py", "passed": passed, "reason": "self.rng not found in environment.py"})
    except Exception as e:
        checklist.append({"id": "P10", "desc": "Deterministic RNG (self.rng) present in environment.py", "passed": False, "reason": f"ERROR: {e}"})

    # Collect gaps from all failed checks.
    for section_name, section in [("D1", d1), ("D2", d2), ("D3", d3), ("D4", d4), ("D5", d5), ("CHECKLIST", checklist)]:
        for item in section:
            if not item["passed"]:
                gaps.append((section_name, item["id"], item["reason"]))

    # Scores and weighted points.
    d1_score, _, _ = score_dimension(d1)
    d2_score, _, _ = score_dimension(d2)
    d3_score, _, _ = score_dimension(d3)
    d4_score, _, _ = score_dimension(d4)
    d5_score, _, _ = score_dimension(d5)

    d1_weighted = d1_score * 0.30
    d2_weighted = d2_score * 0.25
    d3_weighted = d3_score * 0.20
    d4_weighted = d4_score * 0.15
    d5_weighted = d5_score * 0.10
    final_score = d1_weighted + d2_weighted + d3_weighted + d4_weighted + d5_weighted

    # Reward stats for D2 output.
    if rewards_200:
        min_r = min(rewards_200)
        max_r = max(rewards_200)
        mean_r = statistics.mean(rewards_200)
        var_r = statistics.variance(rewards_200) if len(rewards_200) > 1 else 0.0
        distinct_r = len(set(round(r, 3) for r in rewards_200))
    else:
        min_r = 0.0
        max_r = 0.0
        mean_r = 0.0
        var_r = 0.0
        distinct_r = 0

    # ---------------- OUTPUT ----------------
    print("[D1] Real-world utility (30%)")
    for item in d1:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
    print(f"  Score: {d1_score:.1f} / 10  Weighted: {d1_weighted:.2f} pts")
    print()

    print("[D2] Task & grader quality (25%)")
    for item in d2:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
    print(f"  Reward stats: min={min_r:.2f} max={max_r:.2f} mean={mean_r:.2f} var={var_r:.2f} distinct={distinct_r}")
    print(f"  Score: {d2_score:.1f} / 10  Weighted: {d2_weighted:.2f} pts")
    print()

    print("[D3] Environment design (20%)")
    for item in d3:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
    print(f"  Score: {d3_score:.1f} / 10  Weighted: {d3_weighted:.2f} pts")
    print()

    print("[D4] Code quality & compliance (15%)")
    for item in d4:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
    print(f"  Score: {d4_score:.1f} / 10  Weighted: {d4_weighted:.2f} pts")
    print()

    print("[D5] Creativity & novelty (10%)")
    for item in d5:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
    print(f"  Score: {d5_score:.1f} / 10  Weighted: {d5_weighted:.2f} pts")
    print()

    print("[CHECKLIST] Pre-submission gates")
    checklist_passed = 0
    for item in checklist:
        print(pass_fail_line(item["id"], item["desc"], item["passed"], item["reason"]))
        if item["passed"]:
            checklist_passed += 1
    print(f"  Gates passed: {checklist_passed}/10")
    print()

    print("[SCORE] Final")
    print(f"  FINAL SCORE: {final_score:.2f} / 10")
    print()

    print("GAPS:")
    if not gaps:
        print("  None")
    else:
        for section_name, cid, reason in gaps:
            print(f"  - {section_name} {cid}: {reason}")

    # Never abort the judge due to failed checks.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
