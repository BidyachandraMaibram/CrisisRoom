"""
app.py  –  CrisisRoom FastAPI server
Session-aware: each /reset creates an isolated env instance keyed by session_id.
This allows concurrent training rollouts without state collisions.
"""

from __future__ import annotations

import os
import sys
import uuid
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from CrisisRoom_environment import CrisisRoomEnv, ALL_SCENARIOS, SERVICES

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import (
    ResetRequest, StepRequest, ObservationResponse, StepResponse,
    StateResponse, ActionSpaceResponse, ObservationSpaceResponse,
    HealthResponse, ScenarioInfo, ScenariosResponse,
)

app = FastAPI(
    title="CrisisRoom – SRE Incident Response RL Environment",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ──────────────────────────────────────────────────────────────
# Maps session_id -> {env, active, created_at}
# Old single-session clients still work via legacy fallback.
_sessions: Dict[str, dict] = {}
SESSION_TTL = 600   # 10 min

_legacy_env: CrisisRoomEnv = CrisisRoomEnv(max_steps=12, red_herring_prob=0.30)
_legacy_active: bool = False


def _cleanup():
    now = time.time()
    for sid in [k for k, v in _sessions.items() if now - v["created_at"] > SESSION_TTL]:
        del _sessions[sid]


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    _cleanup()
    return HealthResponse()


@app.get("/observation_space", response_model=ObservationSpaceResponse, tags=["Environment"])
async def observation_space():
    return ObservationSpaceResponse(**_legacy_env.observation_space())


@app.get("/action_space", response_model=ActionSpaceResponse, tags=["Environment"])
async def action_space():
    return ActionSpaceResponse(**_legacy_env.action_space())


@app.get("/scenarios", response_model=ScenariosResponse, tags=["Environment"])
async def list_scenarios():
    return ScenariosResponse(
        scenarios=[ScenarioInfo(
            name=s.name, description=s.description, root_cause=s.root_cause,
            correct_fix=s.correct_fix, causal_chain=s.causal_chain, hint=s.hint,
        ) for s in ALL_SCENARIOS],
        total=len(ALL_SCENARIOS),
    )


@app.post("/reset", response_model=ObservationResponse, tags=["Environment"])
async def reset(request: ResetRequest = ResetRequest()):
    global _legacy_env, _legacy_active
    _cleanup()

    env = CrisisRoomEnv(
        max_steps=12, red_herring_prob=0.30,
        curriculum_hint=request.curriculum_hint, seed=request.seed,
    )
    obs = env.reset(scenario_name=request.scenario_name)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"env": env, "active": True, "created_at": time.time()}

    # Keep legacy single-session working for old clients
    _legacy_env = env
    _legacy_active = True

    obs["session_id"] = session_id
    return ObservationResponse(**obs)


@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(request: StepRequest):
    global _legacy_active

    session_id = getattr(request, "session_id", None)

    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        if not session["active"]:
            raise HTTPException(400, f"Session {session_id} is done. Call /reset.")
        env = session["env"]
        session["created_at"] = time.time()
    else:
        if not _legacy_active:
            raise HTTPException(400, "No active episode. Call POST /reset first.")
        env = _legacy_env

    try:
        obs_dict, reward, done, info = env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    if done:
        if session_id and session_id in _sessions:
            _sessions[session_id]["active"] = False
        else:
            _legacy_active = False

    return StepResponse(
        observation=ObservationResponse(**obs_dict),
        reward=reward, done=done, info=info,
    )


@app.get("/state", response_model=StateResponse, tags=["Debug"])
async def get_state(session_id: Optional[str] = None):
    env = _sessions[session_id]["env"] if session_id and session_id in _sessions else _legacy_env
    raw = env.state()
    if raw.get("status") == "not_started":
        return StateResponse(status="not_started")
    return StateResponse(
        scenario_name=raw.get("scenario_name"),
        step_count=raw.get("step_count", 0),
        max_steps=raw.get("max_steps", 12),
        done=raw.get("done", False),
        timed_out=raw.get("timed_out", False),
        diagnosis_made=raw.get("diagnosis_made", False),
        resolution_attempted=raw.get("resolution_attempted", False),
        components_checked=raw.get("components_checked", []),
        status="active" if not raw.get("done") else "done",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
