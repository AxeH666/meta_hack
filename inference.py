from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from client.client import ModGuardClient
from server.environment import ModGuardEnvironment
from server.models import ActionType, ModGuardAction, ModGuardObservation, RiskLevel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT") or os.getenv("PORT", "7860"))
SERVER_URL = os.getenv("SERVER_URL", f"http://{SERVER_HOST}:{SERVER_PORT}").rstrip("/")
TASK_NAME = "content-moderation-triage"
BENCHMARK = "modguard-rl"
NUM_EPISODES_PER_TASK = int(os.getenv("NUM_EPISODES_PER_TASK", "8"))
MAX_STEPS = 4
SUCCESS_SCORE_THRESHOLD = 0.58
HF_SPACE_HINT_VARS = ("SPACE_ID", "SPACE_AUTHOR_NAME", "SPACE_REPO_NAME", "HF_SPACE_ID")
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task_1_routine_triage": {
        "difficulty": "easy",
        "seed_start": 1100,
        "seed_stride": 13,
        "description": "Routine moderation queue handling with mostly short trajectories.",
    },
    "task_2_escalation_budgeting": {
        "difficulty": "medium",
        "seed_start": 2200,
        "seed_stride": 17,
        "description": "Ambiguous cases where escalation discipline matters.",
    },
    "task_3_legal_liability_path": {
        "difficulty": "hard",
        "seed_start": 3300,
        "seed_stride": 19,
        "description": "High-risk cases that require legal-hold calibration.",
    },
}

_OPENAI_CLIENT = None


class InProcessModGuardClient:
    def __init__(self) -> None:
        self._env: Optional[ModGuardEnvironment] = None

    def __enter__(self) -> "InProcessModGuardClient":
        self._env = ModGuardEnvironment()
        return self

    def __exit__(self, *args: Any) -> None:
        self._env = None

    def reset(
        self,
        seed: int | None = None,
        difficulty: str | None = None,
        task_name: str | None = None,
    ) -> ModGuardObservation:
        if self._env is None:
            raise RuntimeError("InProcessModGuardClient is not connected.")
        return self._env.reset(seed=seed, difficulty=difficulty, task_name=task_name)

    def step(self, action: ModGuardAction) -> ModGuardObservation:
        if self._env is None:
            raise RuntimeError("InProcessModGuardClient is not connected.")
        return self._env.step(action)

    def get_state(self):
        if self._env is None:
            raise RuntimeError("InProcessModGuardClient is not connected.")
        return self._env.get_state()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _http_client() -> httpx.Client:
    return httpx.Client(
        base_url=SERVER_URL,
        timeout=3.0,
        follow_redirects=True,
        trust_env=False,
    )


def _server_probe() -> Tuple[bool, Optional[str]]:
    try:
        with _http_client() as client:
            response = client.get("/health")
        return response.status_code == 200, None
    except Exception as exc:
        return False, str(exc)


def _should_serve_forever() -> bool:
    run_mode = os.getenv("RUN_MODE", "serve").strip().lower()
    if run_mode in {"serve", "server", "space"}:
        return True
    if run_mode in {"eval", "baseline", "inference"}:
        return False
    return any(os.getenv(name) for name in HF_SPACE_HINT_VARS)


def _serve_forever() -> None:
    os.execvp(
        "python3",
        [
            "python3",
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
            "--log-level",
            "warning",
        ],
    )


def _metadata_value(observation: ModGuardObservation, key: str, default: Any) -> Any:
    return (observation.metadata or {}).get(key, default)


def _fallback_budget(observation: ModGuardObservation) -> int:
    return int(_metadata_value(observation, "escalation_budget_remaining", 0))


def _risk_action(observation: ModGuardObservation) -> ActionType:
    if observation.risk_level == RiskLevel.critical:
        return ActionType.legal_hold
    if observation.risk_level in {RiskLevel.medium, RiskLevel.high}:
        return ActionType.remove
    return ActionType.approve


def _history_action(observation: ModGuardObservation) -> ActionType:
    if (
        observation.case_history.account_risk >= 0.84
        and observation.case_history.prior_escalations >= 2
    ):
        return ActionType.legal_hold
    if (
        observation.case_history.account_risk >= 0.58
        or observation.case_history.prior_escalations >= 2
    ):
        return ActionType.remove
    return ActionType.approve


def _ai_action(observation: ModGuardObservation) -> ActionType:
    raw_value = _metadata_value(observation, "ai_recommendation", ActionType.approve.value)
    try:
        return ActionType(raw_value)
    except Exception:
        return _risk_action(observation)


def _supported_hint(
    observation: ModGuardObservation,
    hint: ActionType,
    conflict_score: float,
) -> bool:
    risk_action = _risk_action(observation)
    history_action = _history_action(observation)
    if observation.stage.value == "post_decision_audit":
        return True
    if hint == ActionType.legal_hold and observation.risk_level in {RiskLevel.high, RiskLevel.critical}:
        return True
    if conflict_score <= 0.40:
        return True
    return hint in {risk_action, history_action}


def _rule_based_action(observation: ModGuardObservation, escalation_budget: int) -> ActionType:
    metadata = observation.metadata or {}
    uncertainty = float(metadata.get("uncertainty_index", 0.5))
    conflict = float(metadata.get("signal_conflict_score", 0.5))
    scenario_tags = set(metadata.get("scenario_tags", []))
    ai_action = _ai_action(observation)
    risk_action = _risk_action(observation)
    history_action = _history_action(observation)
    hint = observation.human_reviewer_hint
    proposed_resolution = metadata.get("proposed_resolution") or ""
    audit_reason = metadata.get("audit_reason", "")

    if observation.stage.value == "post_decision_audit":
        if "legal" in audit_reason or "delayed_legal_escalation" in scenario_tags:
            return ActionType.legal_hold
        if hint is not None and _supported_hint(observation, hint, conflict):
            return hint
        if proposed_resolution:
            try:
                return ActionType(proposed_resolution)
            except ValueError:
                pass
        if observation.risk_level == RiskLevel.critical:
            return ActionType.legal_hold
        return risk_action

    if observation.stage.value == "legal_review":
        if hint == ActionType.legal_hold:
            return ActionType.legal_hold
        if observation.risk_level == RiskLevel.critical:
            return ActionType.legal_hold
        if "delayed_legal_escalation" in scenario_tags and uncertainty >= 0.45:
            return ActionType.legal_hold
        if hint is not None and _supported_hint(observation, hint, conflict):
            return hint
        if uncertainty >= 0.68:
            return ActionType.legal_hold if observation.risk_level != RiskLevel.low else history_action
        return ActionType.remove if observation.risk_level != RiskLevel.low else ActionType.approve

    if observation.stage.value == "escalation_review":
        if hint is not None and _supported_hint(observation, hint, conflict):
            if hint == ActionType.escalate:
                return ActionType.legal_hold if observation.risk_level == RiskLevel.critical else risk_action
            return hint
        if observation.risk_level == RiskLevel.critical:
            return ActionType.legal_hold
        if uncertainty >= 0.72:
            return ActionType.legal_hold if "delayed_legal_escalation" in scenario_tags else risk_action
        return risk_action

    if hint is not None and _supported_hint(observation, hint, conflict):
        if hint == ActionType.escalate and escalation_budget <= 0:
            return ActionType.legal_hold if observation.risk_level == RiskLevel.critical else risk_action
        return hint

    if observation.risk_level == RiskLevel.critical:
        if escalation_budget > 0 and uncertainty >= 0.38 and ai_action != ActionType.legal_hold:
            return ActionType.escalate
        return ActionType.legal_hold

    if uncertainty >= 0.78:
        if escalation_budget > 0 and observation.risk_level in {RiskLevel.medium, RiskLevel.high}:
            return ActionType.escalate
        if "delayed_legal_escalation" in scenario_tags:
            return ActionType.legal_hold
        return history_action if history_action != ActionType.approve else risk_action

    if conflict >= 0.66 and escalation_budget > 0 and observation.risk_level in {RiskLevel.medium, RiskLevel.high}:
        return ActionType.escalate

    if observation.ai_confidence_score >= 0.86 and conflict <= 0.36:
        return ai_action

    if observation.case_history.account_risk >= 0.82 and observation.risk_level in {RiskLevel.high, RiskLevel.critical}:
        return ActionType.legal_hold if observation.risk_level == RiskLevel.critical else ActionType.remove

    if observation.ai_confidence_score >= 0.60:
        return risk_action
    return history_action


def _build_user_prompt(observation: ModGuardObservation, escalation_budget: int) -> str:
    metadata = observation.metadata or {}
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
            f"- ai_recommendation: {metadata.get('ai_recommendation')}",
            f"- ai_confidence_score: {observation.ai_confidence_score}",
            (
                "- human_reviewer_hint: "
                f"{observation.human_reviewer_hint.value if observation.human_reviewer_hint is not None else None}"
            ),
            f"- reviewer_overturn_rate: {observation.reviewer_overturn_rate}",
            f"- signal_conflict_score: {metadata.get('signal_conflict_score')}",
            f"- uncertainty_index: {metadata.get('uncertainty_index')}",
            f"- scenario_tags: {metadata.get('scenario_tags')}",
            f"- audit_reason: {metadata.get('audit_reason')}",
            "",
            "CASE HISTORY",
            f"- prior_escalations: {observation.case_history.prior_escalations}",
            f"- account_risk: {observation.case_history.account_risk}",
            "",
            "RULES",
            "- Choose one action from approve, remove, escalate, legal_hold.",
            "- Do not rely on AI confidence alone when signals conflict.",
            "- Repeated escalation is penalized.",
            "- Post-decision audit can overturn an earlier decision.",
            "- Reply with only the action token.",
        ]
    )


def _extract_first_action_token(action_text: str) -> Optional[ActionType]:
    lower_response = action_text.lower()
    first_match: Optional[Tuple[int, ActionType]] = None
    for candidate in ActionType:
        match = re.search(rf"\b{re.escape(candidate.value)}\b", lower_response)
        if match is None:
            continue
        if first_match is None or match.start() < first_match[0]:
            first_match = (match.start(), candidate)
    return first_match[1] if first_match is not None else None


def _get_openai_client():
    global _OPENAI_CLIENT
    if API_KEY is None:
        return None
    if _OPENAI_CLIENT is None:
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _OPENAI_CLIENT


def get_llm_action(
    observation: ModGuardObservation,
    escalation_budget: int,
) -> Tuple[str, Optional[str]]:
    if API_KEY is None:
        action = _rule_based_action(observation, escalation_budget)
        return action.value, None

    try:
        client = _get_openai_client()
        system_prompt = (
            "You are a senior trust-and-safety reviewer. "
            "Choose exactly one action from {approve, remove, escalate, legal_hold}. "
            "Treat high AI confidence as unreliable when it conflicts with other signals. "
            "Repeated escalation is penalized. "
            "Reply with only the action token."
        )
        user_prompt = _build_user_prompt(observation, escalation_budget)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=30,
        )
        content = (completion.choices[0].message.content or "").strip()
        return content, None
    except Exception as exc:
        action = _rule_based_action(observation, escalation_budget)
        return action.value, f"llm_error:{str(exc)}"


def _speed_score(steps: int) -> float:
    return {1: 1.0, 2: 0.72, 3: 0.44, 4: 0.16}.get(steps, 0.0)


def _task_grader(task_name: str, episodes: List[Dict[str, Any]]) -> float:
    if not episodes:
        return 0.0

    mean_score = mean(ep["score"] for ep in episodes)
    mean_speed = mean(_speed_score(ep["steps"]) for ep in episodes if ep["steps"] > 0)
    success_rate = mean(1.0 if ep["score"] >= SUCCESS_SCORE_THRESHOLD else 0.0 for ep in episodes)
    action_diversity = len({action for ep in episodes for action in ep["actions"]}) / len(ActionType)

    if task_name == "task_1_routine_triage":
        grade = 0.55 * mean_score + 0.25 * mean_speed + 0.20 * success_rate
    elif task_name == "task_2_escalation_budgeting":
        escalation_discipline = mean(
            1.0 if ep["actions"].count("escalate") <= 1 else 0.25 for ep in episodes
        )
        grade = 0.58 * mean_score + 0.22 * success_rate + 0.20 * escalation_discipline
    else:
        legal_path_discipline = mean(
            1.0 if ep["actions"][-1:] == ["legal_hold"] or "legal_hold" in ep["actions"] else 0.35
            for ep in episodes
        )
        grade = 0.55 * mean_score + 0.20 * success_rate + 0.15 * mean_speed + 0.10 * legal_path_discipline

    grade = 0.92 * grade + 0.08 * action_diversity
    return min(max(round(grade, 6), 0.0), 1.0)


def _task_metrics(task_name: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [episode["score"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    return {
        "min": round(min(scores), 6) if scores else 0.0,
        "max": round(max(scores), 6) if scores else 0.0,
        "mean": round(mean(scores), 6) if scores else 0.0,
        "std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
        "success_rate": round(
            mean(1.0 if score >= SUCCESS_SCORE_THRESHOLD else 0.0 for score in scores),
            6,
        )
        if scores
        else 0.0,
        "mean_steps": round(mean(steps), 6) if steps else 0.0,
        "episodes": len(scores),
        "grader_score": _task_grader(task_name, episodes),
    }


def _difficulty_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [episode["score"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    return {
        "episodes": len(scores),
        "min": round(min(scores), 6) if scores else 0.0,
        "max": round(max(scores), 6) if scores else 0.0,
        "mean": round(mean(scores), 6) if scores else 0.0,
        "std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
        "mean_steps": round(mean(steps), 6) if steps else 0.0,
    }


def _print_summary(summary: Dict[str, object]) -> None:
    sys.stderr.write(f"{json.dumps(summary, sort_keys=True)}\n")
    sys.stderr.flush()


def _episode_seed(config: Dict[str, Any], episode_index: int) -> int:
    return int(config["seed_start"] + episode_index * config["seed_stride"] + (episode_index % 3) * 7)


def _run_episode(
    env_client: ModGuardClient | InProcessModGuardClient,
    task_name: str,
    episode_seed: int,
    difficulty: str,
) -> Dict[str, Any]:
    model_label = MODEL_NAME if API_KEY else "rule-based-baseline"
    log_start(task=task_name, env=BENCHMARK, model=model_label)

    observation = env_client.reset(seed=episode_seed, difficulty=difficulty, task_name=task_name)
    rewards: List[float] = []
    actions: List[str] = []
    steps_taken = 0
    difficulty_label = str(_metadata_value(observation, "difficulty", difficulty))
    last_error: Optional[str] = None

    try:
        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            try:
                state = env_client.get_state()
                escalation_budget = int(getattr(state, "escalation_budget", _fallback_budget(observation)))
            except Exception as exc:
                escalation_budget = _fallback_budget(observation)
                last_error = f"state_fallback:{str(exc)}"

            action_text, llm_error = get_llm_action(observation, escalation_budget)
            if llm_error is not None:
                last_error = llm_error

            try:
                parsed_action = ActionType(action_text.strip().lower())
            except ValueError:
                parsed_action = _extract_first_action_token(action_text)
                if parsed_action is None:
                    parsed_action = _rule_based_action(observation, escalation_budget)
                    last_error = f"parse_failed:{action_text}"

            observation = env_client.step(ModGuardAction(action=parsed_action))
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            actions.append(parsed_action.value)
            steps_taken = step

            log_step(
                step=step,
                action=parsed_action.value,
                reward=reward,
                done=bool(observation.done),
                error=last_error,
            )
            last_error = None

            if observation.done:
                break
    except Exception as exc:
        last_error = f"episode_error:{str(exc)}"
        if steps_taken == 0:
            log_step(step=1, action="approve", reward=0.0, done=True, error=last_error)
    finally:
        final_score = min(max(rewards[-1] if rewards else 0.0, 0.0), 1.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {
        "score": min(max(rewards[-1] if rewards else 0.0, 0.0), 1.0),
        "steps": steps_taken,
        "actions": actions,
        "rewards": rewards,
        "difficulty": difficulty_label,
        "seed": episode_seed,
    }


def main() -> None:
    if _should_serve_forever():
        _serve_forever()

    if API_KEY is None:
        sys.stderr.write("WARNING: HF_TOKEN missing; using deterministic rule-based fallback.\n")
        sys.stderr.flush()

    reachable, probe_error = _server_probe()
    use_inprocess = os.getenv("FORCE_INPROCESS", "").strip().lower() in {"1", "true", "yes"}

    if not use_inprocess and reachable:
        sys.stderr.write(f"Connecting to existing server at {SERVER_URL}\n")
        sys.stderr.flush()
        client_context: ModGuardClient | InProcessModGuardClient = ModGuardClient(base_url=SERVER_URL)
    else:
        use_inprocess = True
        reason = probe_error or "FORCE_INPROCESS enabled"
        sys.stderr.write(
            "WARNING: server unavailable; falling back to in-process environment. "
            f"Reason: {reason}\n"
        )
        sys.stderr.flush()
        client_context = InProcessModGuardClient()

    with client_context as env_client:
        task_episode_data: Dict[str, List[Dict[str, Any]]] = {task: [] for task in TASK_CONFIGS}
        difficulty_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for task_name, config in TASK_CONFIGS.items():
            for episode_index in range(NUM_EPISODES_PER_TASK):
                episode = _run_episode(
                    env_client=env_client,
                    task_name=task_name,
                    episode_seed=_episode_seed(config, episode_index),
                    difficulty=config["difficulty"],
                )
                task_episode_data[task_name].append(episode)
                difficulty_groups[episode["difficulty"]].append(episode)

    task_summary = {
        task_name: _task_metrics(task_name, episodes)
        for task_name, episodes in task_episode_data.items()
    }
    all_scores = [episode["score"] for episodes in task_episode_data.values() for episode in episodes]
    all_steps = [episode["steps"] for episodes in task_episode_data.values() for episode in episodes]
    summary = {
        "benchmark": BENCHMARK,
        "task_name": TASK_NAME,
        "server_url": SERVER_URL,
        "mode": "in_process" if use_inprocess else "remote_server",
        "config": {
            "episodes_per_task": NUM_EPISODES_PER_TASK,
            "max_steps": MAX_STEPS,
            "success_score_threshold": SUCCESS_SCORE_THRESHOLD,
            "tasks": TASK_CONFIGS,
        },
        "task_metrics": task_summary,
        "difficulty_metrics": {
            difficulty: _difficulty_metrics(episodes)
            for difficulty, episodes in difficulty_groups.items()
        },
        "overall": {
            "episodes": len(all_scores),
            "min": round(min(all_scores), 6) if all_scores else 0.0,
            "max": round(max(all_scores), 6) if all_scores else 0.0,
            "mean": round(mean(all_scores), 6) if all_scores else 0.0,
            "std": round(pstdev(all_scores), 6) if len(all_scores) > 1 else 0.0,
            "mean_steps": round(mean(all_steps), 6) if all_steps else 0.0,
            "aggregate": round(mean(metric["grader_score"] for metric in task_summary.values()), 6)
            if task_summary
            else 0.0,
        },
    }
    _print_summary(summary)


if __name__ == "__main__":
    main()
