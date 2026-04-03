from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import json
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from client.client import ModGuardClient
from server.models import ActionType, ModGuardAction, ModGuardObservation

load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME", "modguard-rl")
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = "content-moderation-triage"
BENCHMARK = "modguard-rl"
NUM_EPISODES_PER_TASK = int(os.getenv("NUM_EPISODES_PER_TASK", "5"))
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5
CONTAINER_NAME = "modguard-rl-inference"
BENCHMARK_TASKS = [
    "task_1_routine_triage",
    "task_2_escalation_budgeting",
    "task_3_legal_liability_path",
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _run_command(command: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _cleanup_container() -> None:
    _run_command(["docker", "stop", CONTAINER_NAME], check=False)
    _run_command(["docker", "rm", CONTAINER_NAME], check=False)


def _start_container() -> None:
    _run_command(
        [
            "docker",
            "run",
            "-d",
            "-p",
            "8000:8000",
            "--name",
            CONTAINER_NAME,
            IMAGE_NAME,
        ]
    )


def _ensure_docker_image_exists() -> None:
    result = _run_command(["docker", "images", "-q", IMAGE_NAME], check=False)
    if not result.stdout.strip():
        sys.stderr.write(f"ERROR: Docker image '{IMAGE_NAME}' not found.\n")
        sys.stderr.write(f"Build it first with: docker build -t {IMAGE_NAME} .\n")
        sys.stderr.flush()
        raise SystemExit(1)


def _wait_for_health() -> None:
    start = time.time()
    deadline = start + 120
    last_progress_at = start
    while time.time() < deadline:
        try:
            response = httpx.get("http://localhost:8000/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        now = time.time()
        if now - last_progress_at >= 15:
            elapsed = now - start
            sys.stderr.write(f"Waiting for server... {elapsed:.0f}s\n")
            sys.stderr.flush()
            last_progress_at = now
        time.sleep(3)
    raise TimeoutError("Server health check failed after 120 seconds.")


def _build_user_prompt(observation: ModGuardObservation, escalation_budget: int) -> str:
    return "\n".join(
        [
            "MODERATION CASE SNAPSHOT",
            f"CURRENT STAGE: {observation.stage.value}",
            f"STEP NUMBER: {observation.step_number}",
            f"BUDGET REMAINING: {escalation_budget}",
            "",
            "PRIMARY SIGNALS",
            f"- content_category: {observation.content_category.value}",
            f"- risk_level: {observation.risk_level.value}",
            f"- platform_context: {observation.platform_context.value}",
            f"- ai_confidence_score: {observation.ai_confidence_score}",
            f"- human_reviewer_hint: {observation.human_reviewer_hint.value if observation.human_reviewer_hint is not None else None}",
            f"- reviewer_overturn_rate: {observation.reviewer_overturn_rate}",
            f"- queue_pressure: {observation.queue_pressure}",
            "",
            "CASE HISTORY",
            f"- prior_escalations: {observation.case_history.prior_escalations}",
            f"- account_risk: {observation.case_history.account_risk}",
        ]
    )


def _fallback_budget(observation: ModGuardObservation) -> int:
    # Fallback is used only if live state retrieval fails.
    if observation.step_number == 1:
        return 1
    if observation.stage.value == "escalation_review":
        return 0
    return 1


def _extract_first_action_token(action_text: str) -> Optional[ActionType]:
    lower_response = action_text.lower()
    first_match: Optional[Tuple[int, ActionType]] = None
    for candidate in ActionType:
        token = candidate.value
        match = re.search(rf"\b{re.escape(token)}\b", lower_response)
        if match is None:
            continue
        if first_match is None or match.start() < first_match[0]:
            first_match = (match.start(), candidate)
    return first_match[1] if first_match is not None else None


def get_llm_action(
    llm_client: OpenAI, observation: ModGuardObservation, escalation_budget: int
) -> Tuple[str, Optional[str]]:
    system_prompt = (
        "You are a senior content policy reviewer at a major social media platform. "
        "You must make moderation decisions that balance policy correctness, legal exposure, "
        "operational efficiency, and reviewer calibration.\n\n"
        "Action policy:\n"
        "- approve: content does not violate policy.\n"
        "- remove: clear policy violation; immediate takedown.\n"
        "- escalate: uncertain case; send to senior human review (costs escalation budget).\n"
        "- legal_hold: potential legal liability; preserve for legal team review.\n\n"
        "Key signals to evaluate: risk_level, ai_confidence_score, human_reviewer_hint, "
        "reviewer_overturn_rate, stage, and queue_pressure.\n"
        "Escalation budget is limited to 1 per episode.\n"
        "Hard mode warning: confidence can be adversarially inverted and hints may be adversarial.\n\n"
        "Output format: respond with ONLY one word from {approve, remove, escalate, legal_hold}."
    )
    user_prompt = _build_user_prompt(observation, escalation_budget=escalation_budget)
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30,
        )
        content = response.choices[0].message.content or ""
        return content, None
    except Exception as exc:
        return "approve", f"llm_error:{str(exc)}"


def _run_episode(env_client: ModGuardClient, llm_client: OpenAI, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    obs = env_client.reset()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            try:
                state = env_client.get_state()
                escalation_budget = state.escalation_budget
            except Exception as exc:
                escalation_budget = _fallback_budget(obs)
                error = f"state_fallback:{str(exc)}"
            action_str, llm_error = get_llm_action(
                llm_client, obs, escalation_budget=escalation_budget
            )
            if llm_error is not None:
                error = llm_error

            try:
                parsed = ActionType(action_str.strip().lower())
            except ValueError:
                parsed = _extract_first_action_token(action_str)
                if parsed is None:
                    parsed = ActionType.approve
                    error = f"parse_failed:{action_str}"

            obs = env_client.step(ModGuardAction(action=parsed))
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(float(reward))
            steps_taken = step

            log_step(
                step=step,
                action=parsed.value,
                reward=float(reward),
                done=obs.done,
                error=error,
            )
            error = None

            if obs.done:
                break
    except Exception as exc:
        error = str(exc)
    finally:
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )
    final_score = rewards[-1] if rewards else 0.0
    return min(max(final_score, 0.0), 1.0)


def _task_metrics(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "episodes": 0,
        }
    return {
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(mean(scores), 6),
        "std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
        "episodes": len(scores),
    }


def _print_summary(summary: Dict[str, object]) -> None:
    # Must be stderr-only to keep stdout in strict benchmark line format.
    sys.stderr.write(f"{json.dumps(summary, sort_keys=True)}\n")
    sys.stderr.flush()


def main() -> None:
    if API_KEY is None:
        sys.stderr.write("ERROR: HF_TOKEN is required but not set.\n")
        sys.stderr.flush()
        raise SystemExit(1)

    _ensure_docker_image_exists()

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    _cleanup_container()
    _start_container()
    try:
        _wait_for_health()
        with ModGuardClient(base_url="http://localhost:8000") as env_client:
            task_scores: Dict[str, List[float]] = {task: [] for task in BENCHMARK_TASKS}
            for task in BENCHMARK_TASKS:
                for _ in range(NUM_EPISODES_PER_TASK):
                    score = _run_episode(env_client, llm_client, task_name=task)
                    task_scores[task].append(score)
            all_scores = [score for scores in task_scores.values() for score in scores]
            task_metrics: Dict[str, Dict[str, float]] = {
                task: _task_metrics(scores) for task, scores in task_scores.items()
            }
            summary: Dict[str, object] = {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "episodes_per_task": NUM_EPISODES_PER_TASK,
                "tasks": task_metrics,
                "overall": {
                    "aggregate_score": round(mean(all_scores), 6) if all_scores else 0.0,
                    "min": round(min(all_scores), 6) if all_scores else 0.0,
                    "max": round(max(all_scores), 6) if all_scores else 0.0,
                    "mean": round(mean(all_scores), 6) if all_scores else 0.0,
                    "std": round(pstdev(all_scores), 6) if len(all_scores) > 1 else 0.0,
                    "episodes": len(all_scores),
                },
            }
            _print_summary(summary)
    finally:
        _cleanup_container()


if __name__ == "__main__":
    main()
