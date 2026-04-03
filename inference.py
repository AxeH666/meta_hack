from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

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
NUM_EPISODES = 5
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5
CONTAINER_NAME = "modguard-rl-inference"


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
    result = _run_command(["docker", "images", "-q", "modguard-rl"], check=False)
    if not result.stdout.strip():
        sys.stderr.write("ERROR: Docker image 'modguard-rl' not found.\n")
        sys.stderr.write("Build it first with: docker build -t modguard-rl .\n")
        sys.stderr.flush()
        raise SystemExit(1)


def _wait_for_health() -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            response = httpx.get("http://localhost:8000/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError("Server health check failed after 120 seconds.")


def _build_user_prompt(observation: ModGuardObservation) -> str:
    return "\n".join(
        [
            f"content_category: {observation.content_category.value}",
            f"risk_level: {observation.risk_level.value}",
            f"platform_context: {observation.platform_context.value}",
            f"ai_confidence_score: {observation.ai_confidence_score}",
            f"human_reviewer_hint: {observation.human_reviewer_hint.value if observation.human_reviewer_hint is not None else None}",
            f"queue_pressure: {observation.queue_pressure}",
            f"reviewer_overturn_rate: {observation.reviewer_overturn_rate}",
            f"step_number: {observation.step_number}",
            f"stage: {observation.stage.value}",
            f"case_history.prior_escalations: {observation.case_history.prior_escalations}",
            f"case_history.account_risk: {observation.case_history.account_risk}",
        ]
    )


def get_llm_action(llm_client: OpenAI, observation: ModGuardObservation) -> Tuple[str, Optional[str]]:
    system_prompt = (
        "You are a content moderation agent. Review flagged content and choose "
        "exactly one action: approve, remove, escalate, or legal_hold. "
        "Respond with ONLY the action word. Nothing else."
    )
    user_prompt = _build_user_prompt(observation)
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


def _run_episode(env_client: ModGuardClient, llm_client: OpenAI) -> float:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    obs = env_client.reset()
    rewards: List[float] = []
    steps_taken = 0
    success = False
    error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_str, llm_error = get_llm_action(llm_client, obs)
            if llm_error is not None:
                error = llm_error

            try:
                parsed = ActionType(action_str.strip().lower())
            except ValueError:
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
        score = max(rewards) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )
    return max(rewards) if rewards else 0.0


def main() -> None:
    _ensure_docker_image_exists()

    if API_KEY is None:
        raise ValueError("HF_TOKEN (or API_KEY) is required but not set.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    _cleanup_container()
    _start_container()
    try:
        _wait_for_health()
        with ModGuardClient(base_url="http://localhost:8000") as env_client:
            for _ in range(NUM_EPISODES):
                _run_episode(env_client, llm_client)
    finally:
        _cleanup_container()


if __name__ == "__main__":
    main()
