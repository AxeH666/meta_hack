from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from server.models import (
    ActionType,
    CaseHistory,
    ContentCategory,
    DifficultyLevel,
    GTLabel,
    ModGuardAction,
    ModGuardObservation,
    ModGuardState,
    PlatformContext,
    RiskLevel,
    Stage,
)


Environment = __import__("openenv.core", fromlist=["Environment"]).Environment


class ModGuardEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 4

    REWARD_WEIGHTS = {
        "correctness": 0.36,
        "process": 0.18,
        "hint_calibration": 0.10,
        "speed": 0.10,
        "consistency": 0.14,
        "uncertainty_awareness": 0.12,
    }
    OVERCONFIDENCE_PENALTY_WEIGHT = 0.12

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = random.Random(seed)
        self._state: Optional[ModGuardState] = None
        self._case: dict[str, Any] = {}
        self._stage_hints: dict[Stage, Optional[ActionType]] = {}
        self._stage_overturn_map: dict[Stage, Optional[float]] = {}
        self._revealed_hints: list[ActionType] = []
        self._revealed_stages: set[Stage] = set()
        self._last_grade_breakdown: dict[str, float] = {}

    def _weighted_choice(self, weighted_values: list[tuple[Any, float]]) -> Any:
        roll = self.rng.random()
        cumulative = 0.0
        last_value = weighted_values[-1][0]
        for value, weight in weighted_values:
            cumulative += weight
            if roll <= cumulative:
                return value
        return last_value

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _task_profile(self, task_name: Optional[str]) -> dict[str, Any]:
        if task_name == "task_1_routine_triage":
            return {
                "difficulty": DifficultyLevel.easy,
                "ground_truth": [
                    (GTLabel.approve, 0.48),
                    (GTLabel.remove, 0.46),
                    (GTLabel.legal_hold, 0.06),
                ],
                "risk": [
                    (RiskLevel.low, 0.45),
                    (RiskLevel.medium, 0.35),
                    (RiskLevel.high, 0.18),
                    (RiskLevel.critical, 0.02),
                ],
                "queue_range": (1, 3),
                "zero_budget_prob": 0.08,
                "misleading_ai_prob": 0.10,
                "adversarial_hint_prob": 0.05,
                "delayed_legal_prob": 0.05,
            }
        if task_name == "task_2_escalation_budgeting":
            return {
                "difficulty": DifficultyLevel.medium,
                "ground_truth": [
                    (GTLabel.approve, 0.22),
                    (GTLabel.remove, 0.50),
                    (GTLabel.legal_hold, 0.28),
                ],
                "risk": [
                    (RiskLevel.low, 0.08),
                    (RiskLevel.medium, 0.32),
                    (RiskLevel.high, 0.42),
                    (RiskLevel.critical, 0.18),
                ],
                "queue_range": (2, 5),
                "zero_budget_prob": 0.24,
                "misleading_ai_prob": 0.32,
                "adversarial_hint_prob": 0.24,
                "delayed_legal_prob": 0.18,
            }
        if task_name == "task_3_legal_liability_path":
            return {
                "difficulty": DifficultyLevel.hard,
                "ground_truth": [
                    (GTLabel.approve, 0.08),
                    (GTLabel.remove, 0.20),
                    (GTLabel.legal_hold, 0.72),
                ],
                "risk": [
                    (RiskLevel.low, 0.02),
                    (RiskLevel.medium, 0.10),
                    (RiskLevel.high, 0.28),
                    (RiskLevel.critical, 0.60),
                ],
                "queue_range": (3, 5),
                "zero_budget_prob": 0.34,
                "misleading_ai_prob": 0.56,
                "adversarial_hint_prob": 0.42,
                "delayed_legal_prob": 0.58,
            }
        return {
            "difficulty": DifficultyLevel.medium,
            "ground_truth": [
                (GTLabel.approve, 0.35),
                (GTLabel.remove, 0.40),
                (GTLabel.legal_hold, 0.25),
            ],
            "risk": [
                (RiskLevel.low, 0.20),
                (RiskLevel.medium, 0.35),
                (RiskLevel.high, 0.30),
                (RiskLevel.critical, 0.15),
            ],
            "queue_range": (1, 5),
            "zero_budget_prob": 0.18,
            "misleading_ai_prob": 0.24,
            "adversarial_hint_prob": 0.18,
            "delayed_legal_prob": 0.16,
        }

    def _risk_recommendation(self, risk_level: RiskLevel) -> ActionType:
        if risk_level == RiskLevel.critical:
            return ActionType.legal_hold
        if risk_level == RiskLevel.high:
            return ActionType.remove
        if risk_level == RiskLevel.medium:
            return ActionType.remove
        return ActionType.approve

    def _history_recommendation(self, case_history: CaseHistory) -> ActionType:
        if case_history.account_risk >= 0.84 and case_history.prior_escalations >= 2:
            return ActionType.legal_hold
        if case_history.account_risk >= 0.58 or case_history.prior_escalations >= 2:
            return ActionType.remove
        return ActionType.approve

    def _pick_wrong_action(
        self,
        gt_action: ActionType,
        preferred: Optional[list[ActionType]] = None,
    ) -> ActionType:
        wrong_actions = [action for action in ActionType if action != gt_action]
        if preferred:
            preferred_wrong = [action for action in preferred if action in wrong_actions]
            if preferred_wrong and self.rng.random() < 0.7:
                return self.rng.choice(preferred_wrong)
        return self.rng.choice(wrong_actions)

    def _ai_signal(
        self,
        difficulty: DifficultyLevel,
        gt_action: ActionType,
        risk_action: ActionType,
        history_action: ActionType,
        misleading_prob: float,
    ) -> tuple[ActionType, float, bool]:
        misleading = self.rng.random() < misleading_prob
        if misleading:
            ai_action = self._pick_wrong_action(
                gt_action,
                preferred=[risk_action, history_action, ActionType.remove, ActionType.approve],
            )
            confidence = self._clamp(self.rng.uniform(0.88, 0.99))
            return ai_action, confidence, True

        if difficulty == DifficultyLevel.easy:
            confidence = self._clamp(self.rng.uniform(0.78, 0.97))
        elif difficulty == DifficultyLevel.medium:
            confidence = self._clamp(self.rng.uniform(0.60, 0.90))
        else:
            confidence = self._clamp(self.rng.uniform(0.42, 0.82))
        return gt_action, confidence, False

    def _stage_hint(
        self,
        stage: Stage,
        difficulty: DifficultyLevel,
        gt_action: ActionType,
        risk_action: ActionType,
        history_action: ActionType,
        adversarial_hint_prob: float,
        delayed_legal: bool,
    ) -> Optional[ActionType]:
        if stage == Stage.initial_review and difficulty == DifficultyLevel.hard and self.rng.random() < 0.35:
            return None

        if stage == Stage.post_decision_audit:
            if delayed_legal:
                return ActionType.legal_hold
            if self.rng.random() < 0.82:
                return gt_action
            return self._pick_wrong_action(gt_action, preferred=[risk_action, history_action])

        if self.rng.random() < adversarial_hint_prob:
            return self._pick_wrong_action(gt_action, preferred=[risk_action, history_action])

        if stage == Stage.escalation_review:
            if self.rng.random() < 0.76:
                return gt_action
            return self.rng.choice([risk_action, history_action])

        if stage == Stage.legal_review:
            if delayed_legal and self.rng.random() < 0.85:
                return ActionType.legal_hold
            if self.rng.random() < 0.72:
                return gt_action
            return self.rng.choice([risk_action, history_action])

        if difficulty == DifficultyLevel.easy:
            return gt_action
        if difficulty == DifficultyLevel.medium:
            if self.rng.random() < 0.64:
                return gt_action
            return self.rng.choice([risk_action, history_action])
        if self.rng.random() < 0.22:
            return gt_action
        if self.rng.random() < 0.25:
            return None
        return self.rng.choice([risk_action, history_action])

    def _signal_conflict_score(
        self,
        ai_action: ActionType,
        risk_action: ActionType,
        history_action: ActionType,
        initial_hint: Optional[ActionType],
        misleading_ai: bool,
        starting_budget: int,
        risk_level: RiskLevel,
    ) -> float:
        visible_actions = [ai_action, risk_action, history_action]
        if initial_hint is not None:
            visible_actions.append(initial_hint)
        unique_ratio = (
            (len(set(visible_actions)) - 1) / (len(visible_actions) - 1)
            if len(visible_actions) > 1
            else 0.0
        )
        score = 0.20 + 0.42 * unique_ratio
        if misleading_ai:
            score += 0.16
        if initial_hint is not None and initial_hint != ai_action:
            score += 0.10
        if starting_budget == 0 and risk_level in {RiskLevel.high, RiskLevel.critical}:
            score += 0.08
        return self._clamp(score)

    def _compute_stage_overturns(
        self,
        conflict_score: float,
        misleading_ai: bool,
        delayed_legal: bool,
        risk_level: RiskLevel,
    ) -> dict[Stage, Optional[float]]:
        escalation_overturn = self._clamp(
            0.20
            + 0.52 * conflict_score
            + (0.10 if misleading_ai else 0.0)
            + self.rng.uniform(-0.06, 0.06)
        )
        legal_overturn = self._clamp(
            0.28
            + 0.42 * conflict_score
            + (0.16 if delayed_legal else 0.0)
            + (0.08 if risk_level == RiskLevel.critical else 0.0)
            + self.rng.uniform(-0.05, 0.05)
        )
        audit_overturn = self._clamp(
            max(escalation_overturn, legal_overturn)
            + 0.08
            + (0.08 if delayed_legal else 0.0)
            + self.rng.uniform(-0.04, 0.04)
        )
        return {
            Stage.initial_review: None,
            Stage.escalation_review: escalation_overturn,
            Stage.legal_review: legal_overturn,
            Stage.post_decision_audit: audit_overturn,
        }

    def _uncertainty_index(
        self,
        conflict_score: float,
        risk_level: RiskLevel,
        ai_confidence_score: float,
        delayed_legal: bool,
        audit_required: bool,
        queue_pressure: int,
    ) -> float:
        confidence_ambiguity = 1.0 - abs(ai_confidence_score - 0.5) * 2.0
        risk_pressure = {
            RiskLevel.low: 0.10,
            RiskLevel.medium: 0.35,
            RiskLevel.high: 0.62,
            RiskLevel.critical: 0.88,
        }[risk_level]
        uncertainty = (
            0.42 * conflict_score
            + 0.22 * risk_pressure
            + 0.18 * confidence_ambiguity
            + 0.08 * ((queue_pressure - 1) / 4.0)
            + (0.08 if delayed_legal else 0.0)
            + (0.06 if audit_required else 0.0)
        )
        return self._clamp(uncertainty)

    def _audit_reasons(
        self,
        conflict_score: float,
        misleading_ai: bool,
        delayed_legal: bool,
        starting_budget: int,
        risk_level: RiskLevel,
    ) -> list[str]:
        reasons: list[str] = []
        if delayed_legal:
            reasons.append("possible_legal_retention")
        if conflict_score >= 0.68:
            reasons.append("conflicting_signals")
        if misleading_ai:
            reasons.append("misleading_ai_confidence")
        if starting_budget == 0 and risk_level in {RiskLevel.high, RiskLevel.critical}:
            reasons.append("zero_budget_high_risk")
        return reasons

    def _build_case(
        self,
        difficulty: DifficultyLevel,
        task_name: Optional[str],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        ground_truth = self._weighted_choice(profile["ground_truth"])
        gt_action = ActionType(ground_truth.value)
        risk_level = self._weighted_choice(profile["risk"])
        content_category = self.rng.choice(list(ContentCategory))
        platform_context = self.rng.choice(list(PlatformContext))
        queue_low, queue_high = profile["queue_range"]
        queue_pressure = self.rng.randint(queue_low, queue_high)
        case_history = CaseHistory(
            prior_escalations=self.rng.randint(0, 4 if difficulty == DifficultyLevel.hard else 3),
            account_risk=round(self.rng.uniform(0.04, 0.98), 4),
        )
        starting_budget = 0 if self.rng.random() < profile["zero_budget_prob"] else 1
        risk_action = self._risk_recommendation(risk_level)
        history_action = self._history_recommendation(case_history)
        delayed_legal = (
            ground_truth == GTLabel.legal_hold
            and risk_level in {RiskLevel.high, RiskLevel.critical}
            and self.rng.random() < profile["delayed_legal_prob"]
        )
        ai_action, ai_confidence_score, misleading_ai = self._ai_signal(
            difficulty=difficulty,
            gt_action=gt_action,
            risk_action=risk_action,
            history_action=history_action,
            misleading_prob=profile["misleading_ai_prob"],
        )
        stage_hints = {
            Stage.initial_review: self._stage_hint(
                stage=Stage.initial_review,
                difficulty=difficulty,
                gt_action=gt_action,
                risk_action=risk_action,
                history_action=history_action,
                adversarial_hint_prob=profile["adversarial_hint_prob"],
                delayed_legal=delayed_legal,
            ),
            Stage.escalation_review: self._stage_hint(
                stage=Stage.escalation_review,
                difficulty=difficulty,
                gt_action=gt_action,
                risk_action=risk_action,
                history_action=history_action,
                adversarial_hint_prob=profile["adversarial_hint_prob"] * 0.75,
                delayed_legal=delayed_legal,
            ),
            Stage.legal_review: self._stage_hint(
                stage=Stage.legal_review,
                difficulty=difficulty,
                gt_action=gt_action,
                risk_action=risk_action,
                history_action=history_action,
                adversarial_hint_prob=profile["adversarial_hint_prob"] * 0.55,
                delayed_legal=delayed_legal,
            ),
            Stage.post_decision_audit: self._stage_hint(
                stage=Stage.post_decision_audit,
                difficulty=difficulty,
                gt_action=gt_action,
                risk_action=risk_action,
                history_action=history_action,
                adversarial_hint_prob=0.0,
                delayed_legal=delayed_legal,
            ),
        }
        conflict_score = self._signal_conflict_score(
            ai_action=ai_action,
            risk_action=risk_action,
            history_action=history_action,
            initial_hint=stage_hints[Stage.initial_review],
            misleading_ai=misleading_ai,
            starting_budget=starting_budget,
            risk_level=risk_level,
        )
        audit_reasons = self._audit_reasons(
            conflict_score=conflict_score,
            misleading_ai=misleading_ai,
            delayed_legal=delayed_legal,
            starting_budget=starting_budget,
            risk_level=risk_level,
        )
        audit_required = bool(audit_reasons)
        uncertainty_index = self._uncertainty_index(
            conflict_score=conflict_score,
            risk_level=risk_level,
            ai_confidence_score=ai_confidence_score,
            delayed_legal=delayed_legal,
            audit_required=audit_required,
            queue_pressure=queue_pressure,
        )
        scenario_tags: list[str] = []
        if misleading_ai:
            scenario_tags.append("misleading_ai_confidence")
        if conflict_score >= 0.58:
            scenario_tags.append("conflicting_signals")
        if stage_hints[Stage.initial_review] is None or (
            stage_hints[Stage.initial_review] != gt_action
        ):
            scenario_tags.append("noisy_reviewer_hint")
        if starting_budget == 0:
            scenario_tags.append("zero_budget_start")
        if delayed_legal:
            scenario_tags.append("delayed_legal_escalation")
        if audit_required:
            scenario_tags.append("post_decision_audit")
        stage_overturns = self._compute_stage_overturns(
            conflict_score=conflict_score,
            misleading_ai=misleading_ai,
            delayed_legal=delayed_legal,
            risk_level=risk_level,
        )
        return {
            "task_name": task_name or "default",
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "gt_action": gt_action,
            "risk_level": risk_level,
            "content_category": content_category,
            "platform_context": platform_context,
            "queue_pressure": queue_pressure,
            "case_history": case_history,
            "starting_budget": starting_budget,
            "risk_action": risk_action,
            "history_action": history_action,
            "ai_action": ai_action,
            "ai_confidence_score": ai_confidence_score,
            "misleading_ai": misleading_ai,
            "delayed_legal": delayed_legal,
            "signal_conflict_score": conflict_score,
            "audit_required": audit_required,
            "audit_reasons": audit_reasons,
            "uncertainty_index": uncertainty_index,
            "scenario_tags": scenario_tags,
            "stage_hints": stage_hints,
            "stage_overturns": stage_overturns,
        }

    def _reveal_stage(self, stage: Stage) -> None:
        if stage in self._revealed_stages:
            return
        hint = self._stage_hints.get(stage)
        if hint is not None:
            self._revealed_hints.append(hint)
        self._revealed_stages.add(stage)

    def _caution_band(self) -> str:
        uncertainty = self._case["uncertainty_index"]
        if uncertainty >= 0.72:
            return "high"
        if uncertainty >= 0.42:
            return "medium"
        return "low"

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "task_name": self._case["task_name"],
            "difficulty": self._case["difficulty"].value,
            "scenario_tags": list(self._case["scenario_tags"]),
            "ai_recommendation": self._case["ai_action"].value,
            "signal_conflict_score": round(self._case["signal_conflict_score"], 4),
            "uncertainty_index": round(self._case["uncertainty_index"], 4),
            "caution_level": self._caution_band(),
            "starting_escalation_budget": self._state.starting_escalation_budget,
            "escalation_budget_remaining": self._state.escalation_budget,
            "audit_required": self._state.audit_required,
            "audit_triggered": self._state.audit_triggered,
            "audit_reason": self._state.audit_reason,
            "path_penalty_incurred": self._state.path_penalty_incurred,
            "budget_violated": self._state.budget_violated,
            "repeated_escalations": self._state.repeated_escalations,
            "proposed_resolution": (
                self._state.proposed_resolution.value
                if self._state.proposed_resolution is not None
                else ""
            ),
        }

    def _build_observation(
        self,
        *,
        done: bool,
        reward: float,
        reward_breakdown: Optional[dict[str, float]] = None,
        terminal_action: Optional[ActionType] = None,
    ) -> ModGuardObservation:
        metadata = self._base_metadata()
        if reward_breakdown is not None:
            metadata["reward_breakdown"] = reward_breakdown
        if terminal_action is not None:
            metadata["terminal_action"] = terminal_action.value
        observation = ModGuardObservation(
            content_category=self._case["content_category"],
            risk_level=self._case["risk_level"],
            platform_context=self._case["platform_context"],
            ai_confidence_score=round(self._case["ai_confidence_score"], 4),
            human_reviewer_hint=self._stage_hints.get(self._state.stage),
            queue_pressure=self._case["queue_pressure"],
            reviewer_overturn_rate=self._stage_overturn_map.get(self._state.stage),
            step_number=self._state.step_number,
            case_history=self._case["case_history"],
            stage=self._state.stage,
            done=done,
            reward=round(reward, 6),
            metadata=metadata,
        )
        return observation

    def _default_proposed_resolution(self) -> ActionType:
        if self._case["delayed_legal"] or self._case["risk_level"] == RiskLevel.critical:
            return ActionType.legal_hold
        if self._case["risk_level"] in {RiskLevel.medium, RiskLevel.high}:
            return ActionType.remove
        return ActionType.approve

    def _audit_reason_for(self, action_value: ActionType) -> str:
        reasons = list(self._case["audit_reasons"])
        if action_value in {ActionType.approve, ActionType.remove} and self._case["delayed_legal"]:
            reasons.append("deferred_legal_check")
        if action_value == ActionType.legal_hold and self._case["signal_conflict_score"] >= 0.60:
            reasons.append("hold_confirmation")
        if not reasons:
            reasons.append("post_decision_sampling")
        return ",".join(sorted(set(reasons)))

    def _should_enter_audit(self, action_value: ActionType) -> bool:
        if self._state.step_number >= self.MAX_STEPS:
            return False
        if self._state.stage == Stage.post_decision_audit:
            return False
        if action_value == ActionType.escalate:
            return False
        if self._case["delayed_legal"] and action_value != ActionType.legal_hold:
            return True
        if self._state.stage == Stage.initial_review:
            return self._case["audit_required"] and action_value in {
                ActionType.approve,
                ActionType.remove,
            }
        if self._state.stage in {Stage.escalation_review, Stage.legal_review}:
            if self._case["audit_required"]:
                return True
            if self._state.repeated_escalations > 0:
                return True
            if (
                self._state.stage == Stage.legal_review
                and self._case["risk_level"] == RiskLevel.critical
                and action_value != ActionType.legal_hold
            ):
                return True
        return False

    def _advance_stage(
        self,
        stage: Stage,
        *,
        proposed_resolution: Optional[ActionType] = None,
        audit_reason: str = "",
    ) -> None:
        self._state.stage = stage
        self._state.step_number += 1
        if stage == Stage.post_decision_audit:
            self._state.audit_triggered = True
            self._state.audit_reason = audit_reason
        if proposed_resolution is not None:
            self._state.proposed_resolution = proposed_resolution
        self._reveal_stage(stage)

    def _handle_initial_review(self, action_value: ActionType) -> bool:
        if action_value in {ActionType.approve, ActionType.remove}:
            if self._should_enter_audit(action_value):
                self._advance_stage(
                    Stage.post_decision_audit,
                    proposed_resolution=action_value,
                    audit_reason=self._audit_reason_for(action_value),
                )
                return False
            return True

        if action_value == ActionType.escalate:
            if self._state.escalation_budget > 0:
                self._state.escalation_budget -= 1
                self._advance_stage(Stage.escalation_review)
                return False
            self._state.budget_violated = True
            self._state.path_penalty_incurred = True
            self._state.repeated_escalations += 1
            if self._state.step_number < self.MAX_STEPS:
                self._advance_stage(
                    Stage.post_decision_audit,
                    proposed_resolution=self._default_proposed_resolution(),
                    audit_reason="escalation_budget_exhausted",
                )
                return False
            return True

        self._advance_stage(Stage.legal_review)
        return False

    def _handle_escalation_review(self, action_value: ActionType) -> bool:
        if action_value in {ActionType.approve, ActionType.remove}:
            if self._should_enter_audit(action_value):
                self._advance_stage(
                    Stage.post_decision_audit,
                    proposed_resolution=action_value,
                    audit_reason=self._audit_reason_for(action_value),
                )
                return False
            return True

        if action_value == ActionType.escalate:
            self._state.path_penalty_incurred = True
            self._state.repeated_escalations += 1
            if self._state.step_number < self.MAX_STEPS:
                self._advance_stage(
                    Stage.post_decision_audit,
                    proposed_resolution=self._default_proposed_resolution(),
                    audit_reason="redundant_escalation_chain",
                )
                return False
            return True

        self._advance_stage(Stage.legal_review)
        return False

    def _handle_legal_review(self, action_value: ActionType) -> bool:
        if action_value == ActionType.escalate:
            self._state.path_penalty_incurred = True
            self._state.repeated_escalations += 1
            if self._state.step_number < self.MAX_STEPS:
                self._advance_stage(
                    Stage.post_decision_audit,
                    proposed_resolution=self._default_proposed_resolution(),
                    audit_reason="escalation_after_legal_review",
                )
                return False
            return True

        if (
            self._case["risk_level"] == RiskLevel.critical
            and action_value == ActionType.approve
        ):
            self._state.path_penalty_incurred = True

        if self._should_enter_audit(action_value):
            self._advance_stage(
                Stage.post_decision_audit,
                proposed_resolution=action_value,
                audit_reason=self._audit_reason_for(action_value),
            )
            return False

        return True

    def _correctness_score(self, terminal_action: ActionType) -> float:
        correctness_table = {
            GTLabel.approve: {
                ActionType.approve: 1.0,
                ActionType.remove: 0.42,
                ActionType.escalate: 0.10,
                ActionType.legal_hold: 0.0,
            },
            GTLabel.remove: {
                ActionType.approve: 0.32,
                ActionType.remove: 1.0,
                ActionType.escalate: 0.28,
                ActionType.legal_hold: 0.14,
            },
            GTLabel.legal_hold: {
                ActionType.approve: 0.0,
                ActionType.remove: 0.12,
                ActionType.escalate: 0.46,
                ActionType.legal_hold: 1.0,
            },
        }
        return correctness_table[self._case["ground_truth"]][terminal_action]

    def _optimal_steps(self) -> int:
        if self._case["delayed_legal"]:
            return 4
        if self._case["ground_truth"] == GTLabel.legal_hold:
            return 4 if self._case["audit_required"] else 3
        if self._case["uncertainty_index"] >= 0.70:
            return 3 if self._state.starting_escalation_budget > 0 else 2
        if self._case["uncertainty_index"] >= 0.48:
            return 2
        return 1

    def _process_score(self) -> float:
        budget_component = 0.05 if self._state.budget_violated else 1.0
        repeated_penalty = 0.22 * min(self._state.repeated_escalations, 2)
        path_component = 0.50 if self._state.path_penalty_incurred else 1.0
        optimal_steps = self._optimal_steps()
        step_alignment = self._clamp(1.0 - abs(self._state.step_number - optimal_steps) / 3.0)
        if not self._state.audit_required:
            audit_component = 1.0
        elif self._state.audit_triggered:
            audit_component = 1.0
        else:
            audit_component = 0.10
        score = (
            0.38 * budget_component
            + 0.24 * step_alignment
            + 0.20 * path_component
            + 0.18 * audit_component
            - repeated_penalty
        )
        return self._clamp(score)

    def _hint_score(self, terminal_action: ActionType) -> float:
        if not self._revealed_hints:
            return 0.55

        gt_action = self._case["gt_action"]
        score = 0.50
        for hint in self._revealed_hints:
            if hint == gt_action and terminal_action == gt_action:
                score += 0.12
            elif hint == gt_action and terminal_action != gt_action:
                score -= 0.18
            elif hint != gt_action and terminal_action == gt_action and terminal_action != hint:
                score += 0.16
            elif hint != gt_action and terminal_action == hint:
                score -= 0.26
            else:
                score += 0.01
        return self._clamp(score)

    def _speed_score(self) -> float:
        return self._clamp(1.0 - 0.28 * (self._state.step_number - 1))

    def _consistency_score(self, terminal_action: ActionType) -> float:
        score = 1.0
        non_escalation_actions = [
            action for action in self._state.action_history if action != ActionType.escalate
        ]
        if self._state.budget_violated:
            score -= 0.26
        if self._state.path_penalty_incurred:
            score -= 0.18
        score -= 0.16 * min(self._state.repeated_escalations, 2)
        if len(non_escalation_actions) >= 2 and non_escalation_actions[0] != non_escalation_actions[-1]:
            if self._state.audit_triggered:
                score += 0.08
            else:
                score -= 0.12
        if (
            self._case["risk_level"] == RiskLevel.critical
            and terminal_action == ActionType.approve
        ):
            score -= 0.18
        if (
            self._case["delayed_legal"]
            and terminal_action != ActionType.legal_hold
        ):
            score -= 0.28
        return self._clamp(score)

    def _uncertainty_awareness_score(self, terminal_action: ActionType) -> float:
        uncertainty = self._case["uncertainty_index"]
        used_caution = any(
            action in {ActionType.escalate, ActionType.legal_hold}
            for action in self._state.action_history[:-1]
        ) or self._state.audit_triggered
        correct = terminal_action == self._case["gt_action"]

        if uncertainty >= 0.70:
            if correct and used_caution:
                return 1.0
            if correct:
                return 0.70
            if used_caution:
                return 0.34
            return 0.08

        if uncertainty <= 0.35:
            if correct and self._state.step_number == 1:
                return 1.0
            if correct and not used_caution:
                return 0.84
            if used_caution:
                return 0.16
            return 0.32

        if correct and used_caution:
            return 0.82
        if correct:
            return 0.72
        if used_caution:
            return 0.30
        return 0.18

    def _overconfidence_penalty(self, terminal_action: ActionType) -> float:
        confidence_strength = self._clamp(
            (self._case["ai_confidence_score"] - 0.78) / 0.22
        )
        blind_alignment = 1.0 if self._state.action_history and self._state.action_history[0] == self._case["ai_action"] else 0.0
        wrong = 1.0 if terminal_action != self._case["gt_action"] else 0.0
        no_caution = 1.0 if self._state.step_number == 1 else 0.4
        conflict = max(
            self._case["signal_conflict_score"],
            self._case["uncertainty_index"],
        )
        if self._case["misleading_ai"]:
            return self._clamp(confidence_strength * blind_alignment * conflict * (0.55 + 0.45 * wrong) * no_caution)
        if wrong and blind_alignment > 0.0 and conflict >= 0.70:
            return self._clamp(0.65 * confidence_strength * blind_alignment * conflict)
        return 0.0

    def _grade(self, terminal_action: ActionType) -> float:
        breakdown = {
            "correctness": self._correctness_score(terminal_action),
            "process": self._process_score(),
            "hint_calibration": self._hint_score(terminal_action),
            "speed": self._speed_score(),
            "consistency": self._consistency_score(terminal_action),
            "uncertainty_awareness": self._uncertainty_awareness_score(terminal_action),
        }
        weighted_sum = sum(
            breakdown[name] * self.REWARD_WEIGHTS[name] for name in self.REWARD_WEIGHTS
        )
        overconfidence_penalty = (
            self._overconfidence_penalty(terminal_action)
            * self.OVERCONFIDENCE_PENALTY_WEIGHT
        )
        grade = self._clamp(weighted_sum - overconfidence_penalty)
        breakdown["overconfidence_penalty"] = overconfidence_penalty
        breakdown["final_reward"] = grade
        self._last_grade_breakdown = {
            name: round(value, 6) for name, value in breakdown.items()
        }
        return grade

    @property
    def state(self) -> ModGuardState:
        if self._state is None:
            raise ValueError("Environment must be reset() before state is available.")
        return self._state

    @state.setter
    def state(self, value: ModGuardState) -> None:
        self._state = value

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        task_name: Optional[str] = None,
        **_: Any,
    ) -> ModGuardObservation:
        self.rng = random.Random(seed)
        profile = self._task_profile(task_name)
        difficulty_level = (
            DifficultyLevel(difficulty)
            if difficulty is not None
            else profile.get("difficulty", self.rng.choice(list(DifficultyLevel)))
        )

        self._case = self._build_case(
            difficulty=difficulty_level,
            task_name=task_name,
            profile=profile,
        )
        self._stage_hints = self._case["stage_hints"]
        self._stage_overturn_map = self._case["stage_overturns"]
        self._revealed_hints = []
        self._revealed_stages = set()
        self._last_grade_breakdown = {}

        self._state = ModGuardState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            step_number=1,
            stage=Stage.initial_review,
            escalation_budget=self._case["starting_budget"],
            starting_escalation_budget=self._case["starting_budget"],
            uncertainty_index=round(self._case["uncertainty_index"], 6),
            path_penalty_incurred=False,
            budget_violated=False,
            repeated_escalations=0,
            audit_required=self._case["audit_required"],
            audit_triggered=False,
            audit_reason="",
            proposed_resolution=None,
            last_action=None,
            action_history=[],
            episode_done=False,
        )
        self._reveal_stage(Stage.initial_review)
        return self._build_observation(done=False, reward=0.0)

    def step(self, action, timeout_s: Optional[float] = None, **_: Any) -> ModGuardObservation:
        del timeout_s

        if self._state is None or not self._case:
            raise ValueError("Environment must be reset() before step().")
        if self._state.episode_done:
            raise AssertionError("step() called on a finished episode")

        action_value = action.action if isinstance(action, ModGuardAction) else action
        if not isinstance(action_value, ActionType):
            try:
                action_value = ActionType(action_value)
            except Exception as exc:
                raise ValueError(f"Invalid action: {action_value}") from exc

        self._state.action_history.append(action_value)
        self._state.last_action = action_value
        self._state.step_count += 1

        if action_value == ActionType.escalate and (
            self._state.stage != Stage.initial_review or self._state.escalation_budget == 0
        ):
            self._state.repeated_escalations += 1

        if self._state.stage == Stage.initial_review:
            terminal = self._handle_initial_review(action_value)
        elif self._state.stage == Stage.escalation_review:
            terminal = self._handle_escalation_review(action_value)
        elif self._state.stage == Stage.legal_review:
            terminal = self._handle_legal_review(action_value)
        else:
            terminal = True

        if terminal:
            self._state.episode_done = True
            reward = self._grade(action_value)
            return self._build_observation(
                done=True,
                reward=reward,
                reward_breakdown=self._last_grade_breakdown,
                terminal_action=action_value,
            )

        return self._build_observation(done=False, reward=0.0)

    def get_state(self) -> ModGuardState:
        return self.state
