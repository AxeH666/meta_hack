---
title: meta_RL_mod
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ModGuard-RL

ModGuard-RL is an OpenEnv environment for multi-stage content moderation in Trust and Safety operations. The agent must decide whether to `approve`, `remove`, `escalate`, or `legal_hold` under uncertainty, budget constraints, and noisy human/AI signals.

## 1) Motivation

Large social platforms process massive moderation queues every day. Every action has real-world consequences: approving harmful content increases downstream harm, removing legitimate content suppresses speech, and unnecessary escalations slow operations.

Trust and Safety work is not a single-shot binary classifier problem. It is a staged workflow with policy, legal, and operational constraints. Teams must triage uncertainty, reason about partial signals, and make defensible decisions under throughput pressure.

ModGuard-RL simulates this real moderation workflow as a compact RL environment. The agent is rewarded not just for final correctness, but also for process quality, hint calibration, and timely resolution.

## 2) Environment Overview

State machine:

```text
initial_review
  ├─ approve/remove -> terminal
  ├─ escalate -> escalation_review (step 2, budget may decrement)
  └─ legal_hold -> legal_review (step 2)

escalation_review (step 2)
  ├─ approve/remove -> terminal
  ├─ escalate -> terminal + path penalty
  └─ legal_hold
      ├─ risk=critical -> legal_review (step 3)
      ├─ GT=legal_hold and risk!=critical -> terminal
      └─ otherwise -> terminal + path penalty

legal_review (step 2)
  ├─ approve/remove -> terminal
  ├─ escalate -> terminal + path penalty
  └─ legal_hold
      ├─ risk=critical -> legal_review (step 3)
      ├─ GT=legal_hold and risk!=critical -> terminal
      └─ otherwise -> terminal + path penalty

legal_review (step 3)
  └─ any action -> terminal
```

## 3) Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `content_category` | enum | fixed set | Policy category for the flagged case. |
| `risk_level` | enum | `low/medium/high/critical` | Severity context that gates legal path behavior. |
| `platform_context` | enum | fixed set | Product surface where the case originated. |
| `ai_confidence_score` | float | `[0.0, 1.0]` | Model confidence signal (difficulty-conditioned). |
| `human_reviewer_hint` | enum/null | action or `None` | Optional reviewer recommendation. |
| `queue_pressure` | int | `1..5` | Current queue urgency/operational pressure. |
| `reviewer_overturn_rate` | float/null | `None` at step 1, `[0.0, 1.0]` at step 2+ | Reviewer disagreement tendency signal. |
| `step_number` | int | `1..3` | Current decision step. |
| `case_history` | object | constrained | Prior account escalation count + account risk score. |
| `stage` | enum | `initial_review/escalation_review/legal_review` | Current state-machine stage. |

## 4) Action Space

| Action | When to use | Consequences |
|---|---|---|
| `approve` | Content does not violate policy. | Can terminate immediately; high reward only if aligned with GT. |
| `remove` | Clear policy violation with confidence. | Can terminate immediately; incorrect removals hurt correctness. |
| `escalate` | Uncertain case requiring senior review. | Consumes limited escalation budget (1 total). Budget misuse harms process score. |
| `legal_hold` | Potential legal liability requiring preservation. | Correct in specific legal-risk scenarios; wrong usage can trigger path penalties. |

## 5) Reward Signal

Final grade formula:

`grade = correctness×0.45 + process×0.25 + hint×0.20 + speed×0.10`

- `correctness` rewards terminal policy alignment with ground truth.
- `process` rewards path efficiency and policy-safe workflow; it is forced to `0.0` on budget violation.
- `hint` rewards calibrated use of reviewer hints, including successful override of bad guidance.
- `speed` rewards earlier correct resolution.

Correctness table:

| GT \ terminal_action | approve | remove | escalate | legal_hold |
|---|---:|---:|---:|---:|
| approve | 1.0 | 0.5 | 0.1 | 0.0 |
| remove | 0.5 | 1.0 | 0.3 | 0.1 |
| legal_hold | 0.0 | 0.1 | 0.5 | 1.0 |

Hint scoring table:

| Condition | Hint score |
|---|---:|
| `hint is None` | 0.5 |
| `hint == terminal_action` and `hint == GT` | 0.8 |
| `hint == terminal_action` and `hint != GT` | 0.0 |
| `hint != terminal_action` and `hint == GT` | 0.3 |
| `hint != terminal_action` and `hint != GT` | 1.0 |

The highest hint score (`1.0`) is assigned when the agent correctly avoids following a wrong hint, explicitly incentivizing calibrated trust instead of blind obedience.

## 6) Difficulty Levels

| Difficulty | AI confidence behavior | Human hint behavior | Adversarial properties |
|---|---|---|---|
| `easy` | `base = U(0.7,1.0)` | Always equals ground truth action | None |
| `medium` | `clamp(base + U(-0.2,0.2))` | Correct with 0.6 probability; otherwise wrong random action | Moderate noise |
| `hard` | `1.0 - base` (always inverted) | Always `None` | Confidence is adversarially inverted; hints unavailable |

## 7) Path Penalty Triggers

`path_penalty_incurred` is sticky once set to `True` and never resets within an episode.

1. `escalation_review` + `escalate` -> immediate terminal penalty (redundant escalation).
2. `escalation_review` + `legal_hold` + non-critical risk + `GT != legal_hold` -> penalty (incorrect legal routing).
3. `legal_review` + `escalate` -> penalty (invalid re-escalation from legal stage).
4. `legal_review` + `legal_hold` + `GT != legal_hold` -> penalty (incorrect legal hold confirmation).
5. Entering step 3 when first action was `legal_hold` -> penalty on entry (inefficient path pattern).

## 8) Task Suite

The final benchmark submission evaluates three explicit task wrappers in `inference.py` (environment logic is unchanged):

- `task_1_routine_triage`: resolves straightforward moderation outcomes quickly with high correctness and speed.
- `task_2_escalation_budgeting`: stresses escalation-budget discipline and discourages invalid/redundant escalations.
- `task_3_legal_liability_path`: targets legal-hold routing behavior under high-risk and path-penalty-sensitive trajectories.

Each task runs `NUM_EPISODES_PER_TASK` episodes (default `5`) and computes a task grader score in `[0.0, 1.0]` from terminal rewards.

## 9) Grader Design

Task-level grader scores are derived from the same locked terminal reward:

`grade = correctness×0.45 + process×0.25 + hint×0.20 + speed×0.10`

Per task:

- episode score = terminal reward from environment (`[0, 1]` clamped by inference wrapper),
- task score distribution = min / max / mean / std across episodes for that task,
- overall aggregate = mean over all task episode scores.

Expected behavior by task:

- **Routine triage**: high correctness with mostly 1-step resolution.
- **Escalation budgeting**: avoids budget violations and redundant escalation paths.
- **Legal liability path**: escalates to legal hold only when supported by risk + GT dynamics, avoiding path penalties.

Inference logs keep strict benchmark stdout format:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=...`

Task summary JSON is emitted to `stderr` only.

## 10) Quickstart

```bash
docker build -t modguard-rl .
docker run -p 8000:8000 modguard-rl
export HF_TOKEN=your_token_here
python inference.py
```

## 11) API Reference

| Endpoint | Method | Request | Response | Notes |
|---|---|---|---|---|
| `/health` | GET | none | health payload | Liveness/readiness endpoint. |
| `/reset` | POST | optional JSON body (e.g. `{}`) | reset payload with observation | Starts new episode. |
| `/step` | POST | `ModGuardAction` | step payload with observation/reward/done | Applies one action. |
| `/state` | GET | none | `ModGuardState` | Returns current internal state. |

## 12) Baseline Results

Command used:

```bash
uv run python inference.py
```

Sample counts and score ranges:

- 3 benchmark tasks (`task_1_routine_triage`, `task_2_escalation_budgeting`, `task_3_legal_liability_path`)
- `NUM_EPISODES_PER_TASK=5` (default), 15 total episodes
- Per-task min/max/mean/std and overall aggregate are emitted in stderr JSON at run end
- Observed per-task ranges (local run):
  - `task_1_routine_triage`: min `0.41`, max `0.96`, mean `0.708`, std `0.218554`
  - `task_2_escalation_budgeting`: min `0.635`, max `0.96`, mean `0.769`, std `0.133619`
  - `task_3_legal_liability_path`: min `0.41`, max `0.96`, mean `0.736`, std `0.197520`
- Observed overall aggregate:
  - min `0.41`, max `0.96`, mean/aggregate `0.737667`, std `0.188413`

Difficulty-stratified ranges:

- Easy: not separately logged in the default benchmark wrapper output
- Medium: not separately logged in the default benchmark wrapper output
- Hard: not separately logged in the default benchmark wrapper output

This section is populated with concrete observed numeric ranges from the final local validation run below.

## 13) Novelty

What is original in ModGuard-RL versus standard moderation toy environments:

- staged moderation workflow with legal and escalation branches (not single-step classification),
- explicit escalation budget guard with process-score consequences,
- adversarial hard-mode confidence inversion plus hint calibration incentives,
- path-penalty structure that rewards efficient, policy-consistent trajectories rather than only terminal labels.

What is standard:

- finite discrete action set,
- episodic terminal reward scoring,
- HTTP reset/step/state API suitable for policy learning loops.

## 14) Failure Modes and Mitigations

- **Budget misuse (`escalate` at zero budget):** guarded by budget-violation logic in environment, and benchmark task 2 tracks downstream score impact.
- **Adversarial confidence/hints:** hard mode inverts confidence and removes hints; policy prompt in inference explicitly warns against blind confidence-following.
- **Path-penalty traps:** tasks emphasize avoiding redundant escalation and incorrect legal-hold routing; process score reflects penalties.
- **State API transient failures:** inference uses live `/state` first each step and deterministic budget fallback only when state fetch fails.
- **Action parse drift from LLM text:** parser extracts first valid action token and falls back to `approve` with explicit error marker.

## 15) Why This Matters for RL Training

ModGuard-RL provides dense and behaviorally meaningful learning signals: variable episode length (1-3 steps), a four-component terminal reward (correctness/process/hint/speed), hard-mode adversarial confidence inversion, explicit escalation-budget constraints, and calibrated hint-use incentives. This structure encourages policy reasoning and robust decision strategy learning rather than shallow pattern matching.
