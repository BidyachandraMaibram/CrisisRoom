"""
models.py  –  Pydantic v2 request/response schemas
openenv init places all data models here (not in a separate schemas.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    scenario_name: Optional[str] = Field(
        default=None,
        description=(
            "Pin a specific scenario. "
            "Options: bad_deployment | db_connection_exhaustion | "
            "cache_stampede | network_partition | traffic_spike. "
            "If omitted, one is sampled at random."
        ),
    )
    curriculum_hint: bool = Field(
        default=False,
        description="If True, include a hint in the initial observation (curriculum phase).",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")


class StepRequest(BaseModel):
    action: str = Field(
        description="Action in format: 'ACTION: TOOL_NAME: argument'",
        examples=["ACTION: CHECK_METRICS: app", "ACTION: DIAGNOSE: bad_deployment"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID from /reset. Pass this to support concurrent rollouts.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class ObservationResponse(BaseModel):
    text: str
    step: int
    steps_remaining: int
    done: bool
    scenario: Optional[str] = None
    hint: Optional[str] = None
    last_tool: Optional[str] = None
    last_argument: Optional[str] = None
    diagnosis_made: bool = False
    resolution_attempted: bool = False
    session_id: Optional[str] = None   # returned by /reset for session tracking


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    scenario_name: Optional[str] = None
    step_count: int = 0
    max_steps: int = 12
    done: bool = False
    timed_out: bool = False
    diagnosis_made: bool = False
    resolution_attempted: bool = False
    components_checked: List[str] = Field(default_factory=list)
    status: str = "active"


class ActionSpaceResponse(BaseModel):
    type: str
    format: str
    tools: List[Dict[str, str]]
    valid_services: List[str]


class ObservationSpaceResponse(BaseModel):
    type: str
    description: str


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "CrisisRoom"
    version: str = "1.0.0"


class ScenarioInfo(BaseModel):
    name: str
    description: str
    root_cause: str
    correct_fix: str
    causal_chain: List[str]
    hint: str


class ScenariosResponse(BaseModel):
    scenarios: List[ScenarioInfo]
    total: int