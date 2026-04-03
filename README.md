# ModGuard-RL

ModGuard-RL is a production-style simulation of enterprise content moderation and content-policy triage, where an automated reviewer must make high-stakes moderation decisions under uncertainty, operational pressure, and multi-stage governance workflows. It models realistic trust-and-safety escalation behavior with structured observations, branching review stages, and a reward function designed to balance correctness, procedural quality, policy alignment, and execution speed.

## Environment Overview

In each episode, the agent receives a moderation case and selects one action at a time: `approve`, `remove`, `escalate`, or `legal_hold`. Episodes follow a branching state machine and terminate in one to three steps depending on risk context, prior decisions, and latent ground truth.

The environment supports three difficulty regimes (`easy`, `medium`, `hard`) that alter signal reliability and confidence behavior while preserving deterministic reproducibility under controlled seeds.

Each observation includes ten core fields that represent policy context and process state:

1. `content_category` — policy category of the flagged content cluster.  
2. `risk_level` — risk severity (`low` to `critical`) driving escalation dynamics.  
3. `platform_context` — product surface where the case originates.  
4. `ai_confidence_score` — model confidence signal, difficulty-dependent in behavior.  
5. `human_reviewer_hint` — optional reviewer guidance that may be correct or misleading.  
6. `queue_pressure` — operational load indicator from 1 to 5.  
7. `reviewer_overturn_rate` — reviewer disagreement tendency, available after step 1.  
8. `step_number` — current decision step in the episode (1–3).  
9. `case_history` — historical account signals (`prior_escalations`, `account_risk`).  
10. `stage` — workflow stage (`initial_review`, `escalation_review`, `legal_review`).

## Reward Signal

The final episode grade is a weighted composition:

`grade = correctness×0.45 + process×0.25 + hint×0.20 + speed×0.10`

**Correctness** emphasizes policy outcome quality at terminal action time. This is the dominant term because moderation systems must prioritize decision validity against latent truth labels.

| GT \ Terminal Action | approve | remove | escalate | legal_hold |
|---|---:|---:|---:|---:|
| approve | 1.0 | 0.5 | 0.1 | 0.0 |
| remove | 0.5 | 1.0 | 0.3 | 0.1 |
| legal_hold | 0.0 | 0.1 | 0.5 | 1.0 |

**Process** captures procedural efficiency and policy-safe routing. It rewards optimal path length, zeroes out when budget constraints are violated, and caps score when known path-penalty triggers occur.

**Hint** evaluates whether the agent used reviewer guidance intelligently. Blindly following incorrect hints is penalized, while correctly overriding bad hints is rewarded.

**Speed** rewards timely resolution, encouraging one-step completion when appropriate and reducing score as additional workflow steps are consumed.

## Difficulty Levels

| Difficulty | ai_confidence_score behavior | Hint reliability | Adversarial conditions |
|---|---|---|---|
| easy | Direct high-confidence baseline signal | Always aligned with ground truth | None |
| medium | Noisy perturbed confidence, clamped to valid bounds | Correct with 60% probability | Moderate ambiguity |
| hard | Adversarial inversion of baseline confidence | Always absent (`None`) | Strongly misleading confidence |

## Optimal Paths

The environment encodes four optimal strategies:

- `GT=approve` or `GT=remove` under any risk: terminate in one step with the matching action.  
- `GT=legal_hold` and non-critical risk: choose `legal_hold` in one step.  
- `GT=legal_hold` and critical risk: follow `escalate (step 1, initial_review) -> legal_hold (step 2, escalation_review, critical only) -> any action (step 3, legal_review terminal)` across three steps.

The 3-step path is only reachable in hard mode episodes where GT=legal_hold and risk_level=critical.
Step 3 is terminal for any action (`approve`, `remove`, `escalate`, or `legal_hold`).
Step 3 is only reachable when the path passes through escalation on critical risk.
The `legal_hold -> legal_review` path without escalation has a maximum length of 2 steps.
Trigger 5 is applied at entry to step 3 when `action_history[0] == legal_hold`.

## Quickstart

```bash
# Install
pip install openenv-core

# Run server
docker build -t modguard-rl .
docker run -p 8000:8000 modguard-rl

# Run inference
export HF_TOKEN=your_token_here
uv run inference.py
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Starts a new episode and returns `ModGuardObservation`. |
| `/step` | POST | Applies `ModGuardAction` and returns next `ModGuardObservation`. |
| `/state` | GET | Returns the current internal `ModGuardState`. |

## Client

Canonical client path: `client/client.py` (this is what `inference.py` imports).
Root `client.py` is a compatibility shim that re-exports `ModGuardClient`.

## File Structure

```text
modguard_rl/
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
├── inference.py
├── README.md
├── server/
│   ├── app.py
│   ├── environment.py
│   └── models.py
└── client/
    └── client.py
```
