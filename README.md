---
title: ModGuard RL
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# ModGuard-RL

ModGuard-RL is an OpenEnv environment for trust-and-safety triage. An agent reviews flagged content and must choose one of four actions: `approve`, `remove`, `escalate`, or `legal_hold`. Unlike a one-shot classifier benchmark, this environment models operational moderation flow: scarce escalation capacity, contradictory AI and human signals, legal-liability edge cases, and post-decision audits that can overturn an apparently reasonable early choice.

## Why moderation RL matters

Real moderation systems are full of sequential tradeoffs. Approving harmful content increases downstream harm. Removing benign content suppresses speech and damages user trust. Escalation queues are expensive. Legal-hold workflows are rare but high stakes. A useful evaluation environment should capture all of those tensions together, not flatten them into a single label prediction.

ModGuard-RL turns that workflow into a compact RL problem with rich process feedback. It rewards not only final correctness, but also calibrated caution, coherent action sequencing, and resistance to misleading confidence signals.

## What is novel here

- Adversarial AI behavior: the visible AI recommendation can be paired with very high confidence even when it is wrong.
- Conflicting signal design: risk level, reviewer hint, and account history can disagree in structured ways.
- Noisy reviewer hints: hints may be correct, noisy, absent, or actively adversarial.
- Operational edge cases: some episodes start with zero escalation budget, repeated escalation chains are penalized, and some legal-hold cases only become obvious after an audit.
- Post-decision audit stage: episodes can extend to 4 decisions, forcing the agent to revise an earlier choice when new signals surface.
- Anti-reward-hacking design: hidden ground truth is not exposed through the public `state()` API.

## Environment structure

Stages:

```text
initial_review
  ├─ approve/remove -> terminal OR post_decision_audit
  ├─ escalate -> escalation_review (if budget remains)
  ├─ escalate -> post_decision_audit with penalty (if budget is already zero)
  └─ legal_hold -> legal_review

escalation_review
  ├─ approve/remove -> terminal OR post_decision_audit
  ├─ escalate -> post_decision_audit with repeated-escalation penalty
  └─ legal_hold -> legal_review

legal_review
  ├─ approve/remove/legal_hold -> terminal OR post_decision_audit
  └─ escalate -> post_decision_audit with penalty

post_decision_audit
  └─ any action -> terminal
```

Episode length is now 1 to 4 steps.

## Observation and state

Core observation fields remain compact and OpenEnv-friendly:

- `content_category`
- `risk_level`
- `platform_context`
- `ai_confidence_score`
- `human_reviewer_hint`
- `queue_pressure`
- `reviewer_overturn_rate`
- `step_number`
- `case_history`
- `stage`

Additional operational info is carried in `observation.metadata`, including:

- `ai_recommendation`
- `signal_conflict_score`
- `uncertainty_index`
- `scenario_tags`
- `audit_reason`
- `escalation_budget_remaining`
- `reward_breakdown` on terminal steps

Public state is intentionally operational, not answer-revealing. It includes budget, step count, repeated escalation count, audit flags, proposed resolution, and action history, but not hidden ground truth.

## Reward design

The reward is continuous, clamped to `[0, 1]`, and intentionally decomposed into interpretable parts:

```text
reward =
  correctness          * 0.36 +
  process              * 0.18 +
  hint_calibration     * 0.10 +
  speed                * 0.10 +
  consistency          * 0.14 +
  uncertainty_awareness* 0.12 -
  overconfidence_penalty*0.12
```

Component intuition:

- `correctness`: final action quality relative to the hidden label.
- `process`: respects escalation budget, avoids path penalties, and matches the episode’s natural trajectory length.
- `hint_calibration`: rewards using good hints and ignoring bad ones.
- `speed`: prefers resolving easy cases quickly.
- `consistency`: rewards coherent action sequences and penalizes escalation chains.
- `uncertainty_awareness`: rewards caution when signals conflict and decisiveness when they do not.
- `overconfidence_penalty`: punishes blind trust in high-confidence AI signals when the case is adversarial.

This design makes reward hacking harder: repeated escalation and premature high-confidence decisions do not dominate the score, and public state does not leak the hidden label.

## Example trajectories

### 1. Routine approve

```text
step 1: initial_review
signals: low risk, aligned hint, low conflict
action: approve
result: terminal, high reward
```

### 2. Budget trap with audit recovery

```text
step 1: initial_review, zero escalation budget, conflicting signals
action: escalate
result: budget violation, forced audit path

step 2: post_decision_audit
new signal: high overturn risk, remove hint
action: remove
result: terminal, partial reward but process penalty remains
```

### 3. Delayed legal requirement

```text
step 1: initial_review
signals: high risk, misleading AI says remove with high confidence
action: escalate

step 2: escalation_review
hint: remove
action: legal_hold

step 3: legal_review
audit required: possible legal retention
action: legal_hold

step 4: post_decision_audit
final action: legal_hold
result: terminal, strong reward for uncertainty-aware correction
```

## Why this is challenging for LLM agents

- The highest-confidence AI signal can still be wrong.
- Reviewer hints are not uniformly trustworthy.
- Fast resolution helps on easy cases but hurts on ambiguous ones.
- Escalation is useful but limited, and repeated escalation is explicitly punished.
- The best policy depends on trajectory logic, not just the terminal label.
- Audit stages can reward changing your mind when new evidence appears.

This makes the environment a better stress test for sequential decision quality than a simple moderation classifier wrapper.

## Task suite

Three explicit benchmark task wrappers are used in `inference.py`:

- `task_1_routine_triage`: easy distribution, mostly short episodes, clean routine moderation.
- `task_2_escalation_budgeting`: medium difficulty with more zero-budget and signal-conflict cases.
- `task_3_legal_liability_path`: hard distribution with high-risk legal-hold pressure and delayed legal escalation cases.

Each task runs deterministic seeded episodes with different seed schedules, producing reproducible but diverse score distributions.

## HF Spaces deployment

The container is server-first by default:

- `RUN_MODE=serve` starts `uvicorn` on `0.0.0.0:${PORT:-7860}`
- `RUN_MODE=eval` runs `python3 inference.py`
- `/health` returns 200 quickly for readiness checks
- `/reset`, `/step`, `/state`, `/schema`, `/metadata`, and `/ws` are available

`inference.py` never spawns a subprocess server. If no server is reachable in evaluation mode, it falls back to an in-process environment.

## Quickstart

Build and serve:

```bash
docker build -t modguard-rl .
docker run -p 7860:7860 modguard-rl
```

Evaluation mode inside Docker:

```bash
docker run --rm -e RUN_MODE=eval modguard-rl
```

Local evaluation against a running server:

```bash
RUN_MODE=eval python3 inference.py
```

Forced in-process evaluation:

```bash
RUN_MODE=eval FORCE_INPROCESS=1 python3 inference.py
```

## API summary

- `GET /health`: liveness and readiness
- `POST /reset`: starts or resets an HTTP session
- `POST /step`: advances the active HTTP session
- `GET /state`: returns current public state for the active session
- `GET /schema`: returns action, observation, and state schemas
- `GET /metadata`: returns environment metadata and README content
- `WS /ws`: OpenEnv-compatible persistent session channel

## Judge-facing summary

ModGuard-RL aims to be strong for both automated validation and human review:

- strict OpenEnv schema endpoints
- HF Spaces-ready server startup
- sessionful HTTP and websocket support
- reproducible multi-seed evaluation
- adversarial and edge-case-heavy trajectories
- reward design that encourages calibrated moderation behavior instead of shortcut policies
