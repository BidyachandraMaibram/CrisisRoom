"""
CrisisRoom – Rollout Generation & Prompt Construction
Handles multi-turn episode rollouts for GRPO training.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an on-call Site Reliability Engineer (SRE). A production incident has been reported.

You have access to diagnostic tools and remediation actions for a distributed system with these services:
- api: API gateway handling inbound traffic
- app: Application server processing business logic
- db: PostgreSQL database
- cache: Redis cache layer
- deploy: Deployment pipeline

AVAILABLE TOOLS:
- CHECK_LOGS: <service>        → Last 10 log lines (may contain noise)
- CHECK_METRICS: <service>     → CPU, memory, latency, error rate metrics
- CHECK_DEPLOYMENT: <service>  → Recent deployment history and diffs
- RESTART_SERVICE: <service>   → Restart the service pods
- ROLLBACK: <service>          → Roll back to previous deployment
- SCALE_UP: <service>          → Increase pod replica count
- FLUSH_CACHE: cache           → Flush the Redis cache
- DIAGNOSE: <root_cause>       → Commit to a root cause diagnosis
- RESOLVE: <action_taken>      → Declare the incident resolved

RESPONSE FORMAT — you MUST respond with EXACTLY ONE action per turn:
ACTION: TOOL_NAME: argument

Examples:
  ACTION: CHECK_METRICS: app
  ACTION: CHECK_LOGS: db
  ACTION: DIAGNOSE: bad_deployment
  ACTION: ROLLBACK: app
  ACTION: RESOLVE: rolled back app service to previous version

STRATEGY:
1. Start by investigating — use CHECK_LOGS, CHECK_METRICS, CHECK_DEPLOYMENT
2. Check multiple services to understand the causal chain
3. Form a hypothesis about the root cause
4. Call DIAGNOSE before taking any remediation action
5. Apply the correct remediation
6. Call RESOLVE to close the incident

Important: Do NOT call RESTART/ROLLBACK/SCALE_UP/FLUSH_CACHE before DIAGNOSE.
Each wrong guess costs you time. You have limited steps. Be systematic."""


def build_initial_prompt(observation: Dict[str, Any], curriculum_hint: bool = False) -> str:
    """Build the first prompt in an episode."""
    obs_text = observation.get("text", "")
    return obs_text


def build_conversation_messages(
    observation_history: List[Dict[str, Any]],
    action_history: List[str],
) -> List[Dict[str, str]]:
    """
    Build the full conversation message list for the LLM.
    Alternates between user (observations) and assistant (actions).

    Returns list of {"role": ..., "content": ...} dicts.
    """
    messages = []

    for i, obs in enumerate(observation_history):
        # User turn: observation
        messages.append({
            "role": "user",
            "content": obs.get("text", str(obs)),
        })
        # Assistant turn: action (if we have one for this obs)
        if i < len(action_history):
            messages.append({
                "role": "assistant",
                "content": action_history[i],
            })

    return messages


def extract_action_from_response(response_text: str) -> str:
    """
    Extract the ACTION: ... line from a model response.
    Returns the full action string or the raw response if no match.
    """
    # Try to find "ACTION: ..." pattern
    m = re.search(r"ACTION\s*:\s*([A-Z_]+\s*:\s*.+?)(?:\n|$)", response_text, re.IGNORECASE)
    if m:
        return f"ACTION: {m.group(1).strip()}"

    # Try bare "TOOL: arg" pattern
    m2 = re.search(r"^([A-Z_]{4,20})\s*:\s*(.+?)(?:\n|$)", response_text.strip(), re.IGNORECASE | re.MULTILINE)
    if m2:
        return f"ACTION: {m2.group(1).strip()}: {m2.group(2).strip()}"

    # Return as-is (will be caught by parser)
    return response_text.strip()


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

@dataclass
class StepData:
    step: int
    observation_text: str
    action: str
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class EpisodeRollout:
    """Complete rollout for one episode."""
    scenario_name: str
    steps: List[StepData] = field(default_factory=list)
    total_reward: float = 0.0
    final_info: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    diagnosis_correct: bool = False
    resolution_correct: bool = False

    # For GRPO: full prompt + completion pairs
    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    full_conversation: str = ""

    # Reward components for multi-reward GRPO
    reward_components: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP client for the CrisisRoom env server
# ---------------------------------------------------------------------------

class CrisisRoomClient:
    """HTTP client for interacting with the CrisisRoom FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(
        self,
        scenario_name: Optional[str] = None,
        curriculum_hint: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "scenario_name": scenario_name,
            "curriculum_hint": curriculum_hint,
            "seed": seed,
        }
        r = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        obs = r.json()
        # Store session_id so every subsequent step() is isolated
        self._session_id = obs.get("session_id")
        return obs

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        payload = {"action": action}
        # Include session_id so the server routes to our isolated env instance
        if hasattr(self, "_session_id") and self._session_id:
            payload["session_id"] = self._session_id
        r = self.session.post(f"{self.base_url}/step", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        obs = data["observation"]
        return obs, data["reward"], data["done"], data["info"]

    def state(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------

def run_episode_rollout(
    client: CrisisRoomClient,
    model_fn,  # callable(messages: List[dict]) -> str
    scenario_name: Optional[str] = None,
    curriculum_hint: bool = False,
    seed: Optional[int] = None,
    max_steps: int = 12,
) -> EpisodeRollout:
    """
    Run a complete episode and collect the rollout.

    Args:
        client: CrisisRoomClient connected to the env server
        model_fn: callable that takes a list of chat messages and returns action string
        scenario_name: optional scenario pin
        curriculum_hint: whether to use curriculum hint
        seed: optional random seed
        max_steps: max steps per episode

    Returns:
        EpisodeRollout with all step data and reward breakdowns
    """
    # Reset environment
    obs = client.reset(
        scenario_name=scenario_name,
        curriculum_hint=curriculum_hint,
        seed=seed,
    )
    scenario = obs.get("scenario", "unknown")

    rollout = EpisodeRollout(scenario_name=scenario)

    observation_history: List[Dict[str, Any]] = [obs]
    action_history: List[str] = []

    for step_num in range(max_steps):
        # Build conversation so far
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += build_conversation_messages(observation_history, action_history)

        # Get model's action
        raw_response = model_fn(messages)
        action = extract_action_from_response(raw_response)
        action_history.append(action)

        # Execute action in environment
        next_obs, reward, done, info = client.step(action)

        step_data = StepData(
            step=step_num + 1,
            observation_text=next_obs.get("text", ""),
            action=action,
            reward=reward,
            done=done,
            info=info,
        )
        rollout.steps.append(step_data)
        observation_history.append(next_obs)

        if done:
            rollout.total_reward = reward
            rollout.final_info = info
            rollout.success = info.get("resolution_attempted", False)
            rollout.diagnosis_correct = (
                info.get("reward_diagnosis_correct", -999) > 0
            )
            rollout.resolution_correct = (
                info.get("reward_remediation_correct", -999) > 0
            )
            # Extract reward components
            for key in [
                "reward_diagnosis_correct",
                "reward_remediation_correct",
                "reward_causal_reasoning",
                "reward_efficiency",
                "reward_investigation_quality",
                "reward_red_herring_resistance",
                "reward_timeout_penalty",
                "reward_premature_action_penalty",
                "reward_total",
            ]:
                rollout.reward_components[key] = info.get(key, 0.0)
            break

    # Build full conversation string for GRPO prompt/completion
    rollout.prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rollout.prompt_messages += build_conversation_messages(
        observation_history[:-1],  # exclude last obs (after done)
        action_history,
    )
    rollout.full_conversation = _messages_to_string(rollout.prompt_messages)

    return rollout


def _messages_to_string(messages: List[Dict[str, str]]) -> str:
    """Convert message list to a readable string."""
    lines = []
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Batch rollout collection for GRPO
# ---------------------------------------------------------------------------

def collect_rollouts_batch(
    client: CrisisRoomClient,
    model_fn,
    n_episodes: int,
    curriculum_hint: bool = False,
    scenario_names: Optional[List[str]] = None,
) -> List[EpisodeRollout]:
    """
    Collect a batch of episode rollouts for GRPO training.

    Args:
        client: env server client
        model_fn: model callable
        n_episodes: how many episodes to run
        curriculum_hint: curriculum mode
        scenario_names: optional fixed list to cycle through

    Returns:
        List of EpisodeRollout objects
    """
    rollouts = []
    for i in range(n_episodes):
        scenario = None
        if scenario_names:
            scenario = scenario_names[i % len(scenario_names)]
        try:
            rollout = run_episode_rollout(
                client=client,
                model_fn=model_fn,
                scenario_name=scenario,
                curriculum_hint=curriculum_hint,
            )
            rollouts.append(rollout)
        except Exception as e:
            print(f"[WARNING] Episode {i} failed: {e}")
            continue
    return rollouts


def rollouts_to_grpo_dataset(rollouts: List[EpisodeRollout]) -> List[Dict[str, Any]]:
    """
    Convert rollouts to a dataset format suitable for GRPOTrainer.

    Each entry has:
        - prompt: the conversation up to the last assistant turn
        - completion: the last assistant action
        - reward: total episode reward
        - reward_components: dict of all component rewards
    """
    dataset = []
    for rollout in rollouts:
        if not rollout.steps:
            continue
        dataset.append({
            "prompt": rollout.prompt_messages,
            "scenario": rollout.scenario_name,
            "total_reward": rollout.total_reward,
            "reward_components": rollout.reward_components,
            "success": rollout.success,
            "diagnosis_correct": rollout.diagnosis_correct,
            "resolution_correct": rollout.resolution_correct,
            "steps": len(rollout.steps),
        })
    return dataset
