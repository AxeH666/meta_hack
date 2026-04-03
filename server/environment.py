import random
import math
from server.models import (
    ContentCategory,
    RiskLevel,
    PlatformContext,
    ActionType,
    GTLabel,
    Stage,
    CaseHistory,
    ModGuardObservation,
    ModGuardAction,
    ModGuardState,
    DifficultyLevel,
)


Environment = __import__("openenv.core", fromlist=["Environment"]).Environment


class ModGuardEnvironment(Environment):
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self._state = None
        self._reset_observation = None
        self._initial_hint = None
        self._last_overturn = None

    @property
    def state(self) -> ModGuardState:
        return self._state

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def reset(self, seed: int = None) -> ModGuardObservation:
        self.rng = random.Random(seed)
        self._last_overturn = None

        difficulty = self.rng.choice(list(DifficultyLevel))

        gt_draw = self.rng.random()
        if gt_draw < 0.4:
            ground_truth = GTLabel.approve
        elif gt_draw < 0.8:
            ground_truth = GTLabel.remove
        else:
            ground_truth = GTLabel.legal_hold

        risk_level = self.rng.choice(list(RiskLevel))
        content_category = self.rng.choice(list(ContentCategory))
        platform_context = self.rng.choice(list(PlatformContext))
        queue_pressure = self.rng.randint(1, 5)

        case_history = CaseHistory(
            prior_escalations=self.rng.randint(0, 3),
            account_risk=self.rng.uniform(0.0, 1.0),
        )

        base = self.rng.uniform(0.7, 1.0)
        if difficulty == DifficultyLevel.easy:
            ai_confidence_score = base
        elif difficulty == DifficultyLevel.medium:
            ai_confidence_score = self._clamp(base + self.rng.uniform(-0.2, 0.2))
        else:
            ai_confidence_score = 1.0 - base

        gt_as_action = ActionType(ground_truth.value)
        if difficulty == DifficultyLevel.hard:
            human_reviewer_hint = None
        elif difficulty == DifficultyLevel.easy:
            human_reviewer_hint = gt_as_action
        else:
            if self.rng.random() < 0.6:
                human_reviewer_hint = gt_as_action
            else:
                wrong_actions = [a for a in ActionType if a != gt_as_action]
                human_reviewer_hint = self.rng.choice(wrong_actions)

        self._initial_hint = human_reviewer_hint

        self._state = ModGuardState(
            step_number=1,
            stage=Stage.initial_review,
            escalation_budget=1,
            ground_truth=ground_truth,
            path_penalty_incurred=False,
            budget_violated=False,
            action_history=[],
            episode_done=False,
        )

        self._reset_observation = ModGuardObservation(
            content_category=content_category,
            risk_level=risk_level,
            platform_context=platform_context,
            ai_confidence_score=ai_confidence_score,
            human_reviewer_hint=human_reviewer_hint,
            queue_pressure=queue_pressure,
            reviewer_overturn_rate=None,
            step_number=1,
            case_history=case_history,
            stage=Stage.initial_review,
            done=False,
            reward=0.0,
        )
        return self._reset_observation

    def step(self, action) -> ModGuardObservation:
        if self._state is None or self._reset_observation is None:
            raise ValueError("Environment must be reset() before step().")

        assert self._state.escalation_budget in {0, 1}
        assert self._state.step_number <= 3
        assert not self._state.episode_done, "step() called on a finished episode"

        if isinstance(action, ModGuardAction):
            action_value = action.action
        else:
            action_value = action

        if not isinstance(action_value, ActionType):
            raise ValueError(f"Invalid action: {action_value}")

        if action_value == ActionType.escalate and self._state.escalation_budget == 0:
            self._state.budget_violated = True

        self._state.action_history.append(action_value)

        terminal = False

        if self._state.step_number == 1 and self._state.stage == Stage.initial_review:
            if action_value == ActionType.approve:
                terminal = True
            elif action_value == ActionType.remove:
                terminal = True
            elif action_value == ActionType.escalate:
                self._state.stage = Stage.escalation_review
                if not self._state.budget_violated:
                    self._state.escalation_budget -= 1
                self._state.step_number += 1
            elif action_value == ActionType.legal_hold:
                self._state.stage = Stage.legal_review
                self._state.step_number += 1

        elif self._state.step_number == 2 and self._state.stage == Stage.escalation_review:
            if action_value == ActionType.approve:
                terminal = True
            elif action_value == ActionType.remove:
                terminal = True
            elif action_value == ActionType.escalate:
                self._state.path_penalty_incurred = True
                terminal = True
            elif action_value == ActionType.legal_hold:
                if self._reset_observation.risk_level == RiskLevel.critical:
                    self._state.stage = Stage.legal_review
                    self._state.step_number += 1
                    if (
                        len(self.state.action_history) > 0
                        and self.state.action_history[0] == ActionType.legal_hold
                    ):
                        self.state.path_penalty_incurred = True
                elif self._state.ground_truth == GTLabel.legal_hold:
                    terminal = True
                else:
                    self._state.path_penalty_incurred = True
                    terminal = True

        elif self._state.step_number == 2 and self._state.stage == Stage.legal_review:
            if action_value == ActionType.approve:
                terminal = True
            elif action_value == ActionType.remove:
                terminal = True
            elif action_value == ActionType.escalate:
                self._state.path_penalty_incurred = True
                terminal = True
            elif action_value == ActionType.legal_hold:
                if self._reset_observation.risk_level == RiskLevel.critical:
                    self._state.stage = Stage.legal_review
                    self._state.step_number += 1
                    if (
                        len(self.state.action_history) > 0
                        and self.state.action_history[0] == ActionType.legal_hold
                    ):
                        self.state.path_penalty_incurred = True
                elif self._state.ground_truth == GTLabel.legal_hold:
                    terminal = True
                else:
                    self._state.path_penalty_incurred = True
                    terminal = True

        elif self._state.step_number == 3 and self._state.stage == Stage.legal_review:
            terminal = True

        if terminal:
            self._state.episode_done = True
            reward = self._grade(action_value)
            return ModGuardObservation(
                content_category=self._reset_observation.content_category,
                risk_level=self._reset_observation.risk_level,
                platform_context=self._reset_observation.platform_context,
                ai_confidence_score=self._reset_observation.ai_confidence_score,
                human_reviewer_hint=self._reset_observation.human_reviewer_hint,
                queue_pressure=self._reset_observation.queue_pressure,
                reviewer_overturn_rate=None if self._state.step_number == 1 else self._last_overturn,
                step_number=self._state.step_number,
                case_history=self._reset_observation.case_history,
                stage=self._state.stage,
                done=True,
                reward=reward,
            )

        base = self.rng.uniform(0.0, 1.0)
        noise = self.rng.uniform(-0.15, 0.15)
        overturn = self._clamp(base + noise)
        self._last_overturn = overturn

        return ModGuardObservation(
            content_category=self._reset_observation.content_category,
            risk_level=self._reset_observation.risk_level,
            platform_context=self._reset_observation.platform_context,
            ai_confidence_score=self._reset_observation.ai_confidence_score,
            human_reviewer_hint=self._reset_observation.human_reviewer_hint,
            queue_pressure=self._reset_observation.queue_pressure,
            reviewer_overturn_rate=overturn,
            step_number=self._state.step_number,
            case_history=self._reset_observation.case_history,
            stage=self._state.stage,
            done=False,
            reward=0.0,
        )

    def _grade(self, terminal_action: ActionType) -> float:
        correctness_table = {
            GTLabel.approve: {
                ActionType.approve: 1.0,
                ActionType.remove: 0.5,
                ActionType.escalate: 0.1,
                ActionType.legal_hold: 0.0,
            },
            GTLabel.remove: {
                ActionType.approve: 0.5,
                ActionType.remove: 1.0,
                ActionType.escalate: 0.3,
                ActionType.legal_hold: 0.1,
            },
            GTLabel.legal_hold: {
                ActionType.approve: 0.0,
                ActionType.remove: 0.1,
                ActionType.escalate: 0.5,
                ActionType.legal_hold: 1.0,
            },
        }
        correctness_score = correctness_table[self._state.ground_truth][terminal_action]

        process_score = self._process_score()

        hint = self._initial_hint
        gt_action = ActionType(self._state.ground_truth.value)
        if hint is None:
            hint_score = 0.5
        elif hint == terminal_action and hint == gt_action:
            hint_score = 0.8
        elif hint == terminal_action and hint != gt_action:
            hint_score = 0.0
        elif hint != terminal_action and hint == gt_action:
            hint_score = 0.3
        else:
            hint_score = 1.0

        if self._state.step_number == 1:
            speed_score = 1.0
        elif self._state.step_number == 2:
            speed_score = 0.6
        else:
            speed_score = 0.2

        grade = (
            correctness_score * 0.45
            + process_score * 0.25
            + hint_score * 0.20
            + speed_score * 0.10
        )
        return grade

    def _process_score(self) -> float:
        if self._state.budget_violated:
            return 0.0

        if self._state.ground_truth in {GTLabel.approve, GTLabel.remove}:
            optimal_steps = 1
        elif self._reset_observation.risk_level == RiskLevel.critical:
            optimal_steps = 3
        else:
            optimal_steps = 1

        diff = int(math.fabs(self._state.step_number - optimal_steps))
        if diff == 0:
            score = 1.0
        elif diff == 1:
            score = 0.6
        else:
            score = 0.2

        if self._state.path_penalty_incurred:
            score = min(score, 0.4)
        return score

    def get_state(self) -> ModGuardState:
        return self._state
