"""
ModGuard-RL — server/models.py
Spec version: 1.5.0  (LOCKED)
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from openenv.core import Action, Observation, State
from pydantic import Field, field_validator, model_validator


class ContentCategory(str, Enum):
    SEVERITY_CRITICAL = "SEVERITY_CRITICAL"
    SEVERITY_HIGH = "SEVERITY_HIGH"
    SEVERITY_MEDIUM = "SEVERITY_MEDIUM"
    COORDINATED_INAUTHENTIC = "COORDINATED_INAUTHENTIC"
    PLATFORM_INTEGRITY = "PLATFORM_INTEGRITY"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class PlatformContext(str, Enum):
    public_post = "public_post"
    private_group = "private_group"
    marketplace = "marketplace"
    messenger = "messenger"


class ActionType(str, Enum):
    approve = "approve"
    remove = "remove"
    escalate = "escalate"
    legal_hold = "legal_hold"


class GTLabel(str, Enum):
    approve = "approve"
    remove = "remove"
    legal_hold = "legal_hold"


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Stage(str, Enum):
    initial_review = "initial_review"
    escalation_review = "escalation_review"
    legal_review = "legal_review"


class CaseHistory(Observation):
    prior_escalations: int = Field(..., ge=0)
    account_risk: float = Field(..., ge=0.0, le=1.0)


class ModGuardObservation(Observation):
    content_category: ContentCategory = Field(...)
    risk_level: RiskLevel = Field(...)
    platform_context: PlatformContext = Field(...)
    ai_confidence_score: float = Field(..., ge=0.0, le=1.0)
    human_reviewer_hint: Optional[ActionType] = Field(default=None)
    queue_pressure: int = Field(..., ge=1, le=5)
    reviewer_overturn_rate: Optional[float] = Field(default=None)
    step_number: int = Field(..., ge=1, le=3)
    case_history: CaseHistory = Field(...)
    stage: Stage = Field(...)

    @field_validator("reviewer_overturn_rate")
    @classmethod
    def validate_overturn_rate(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"reviewer_overturn_rate must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def validate_overturn_rate_step_consistency(self):
        if self.step_number == 1 and self.reviewer_overturn_rate is not None:
            raise ValueError("reviewer_overturn_rate must be None at step_number=1.")
        return self


class ModGuardAction(Action):
    action: ActionType = Field(...)


class ModGuardState(State):
    step_number: int = Field(default=1, ge=1, le=3)
    stage: Stage = Field(default=Stage.initial_review)
    escalation_budget: int = Field(default=1, ge=0)
    ground_truth: GTLabel = Field(...)
    path_penalty_incurred: bool = Field(default=False)
    budget_violated: bool = Field(default=False)
    action_history: List[ActionType] = Field(default_factory=list)
    episode_done: bool = Field(default=False)
