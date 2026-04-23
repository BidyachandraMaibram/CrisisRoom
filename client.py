"""
CrisisRoom – Environment Client

Connects to a deployed HF Space (or local server) and provides
the standard OpenEnv interface: reset() / step() / state()

Usage:
    from client import CrisisRoomClient, CrisisRoomEnvClient

    # Simple usage
    client = CrisisRoomClient(base_url="http://localhost:7860")
    obs = client.reset()
    obs, reward, done, info = client.step("ACTION: CHECK_METRICS: app")

    # Context manager usage
    with CrisisRoomEnvClient("https://YOUR_USERNAME-crisisroom.hf.space") as env:
        obs = env.reset(scenario_name="bad_deployment")
        while not obs["done"]:
            obs, reward, done, info = env.step("ACTION: CHECK_METRICS: app")
            if done:
                print("Total reward:", reward)
                break
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrisisObservation:
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


@dataclass
class CrisisStepResult:
    observation: CrisisObservation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrisisState:
    scenario_name: Optional[str]
    step_count: int
    max_steps: int
    done: bool
    timed_out: bool
    diagnosis_made: bool
    resolution_attempted: bool
    components_checked: List[str]
    status: str


# ─────────────────────────────────────────────────────────────────────────────
# Core HTTP client
# ─────────────────────────────────────────────────────────────────────────────

class CrisisRoomClient:
    """
    HTTP client for the CrisisRoom SRE Incident Response RL Environment.

    Works with both local server and deployed HuggingFace Space.

    Example:
        client = CrisisRoomClient("http://localhost:7860")
        obs = client.reset()
        print(obs["text"])

        obs, reward, done, info = client.step("ACTION: CHECK_METRICS: app")
        print(reward, done)
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    def _post(self, endpoint: str, data: dict = None) -> dict:
        url = "{}/{}".format(self.base_url, endpoint)
        body = json.dumps(data or {}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                "HTTP {} from {}: {}".format(e.code, url, e.read().decode()))

    def _get(self, endpoint: str) -> dict:
        url = "{}/{}".format(self.base_url, endpoint)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                "HTTP {} from {}: {}".format(e.code, url, e.read().decode()))

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(
        self,
        scenario_name: Optional[str] = None,
        curriculum_hint: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.

        Args:
            scenario_name: pin a specific scenario or None for random
            curriculum_hint: include a hint in the observation
            seed: random seed for reproducibility

        Returns:
            observation dict with 'text', 'step', 'steps_remaining', etc.
        """
        return self._post("reset", {
            "scenario_name": scenario_name,
            "curriculum_hint": curriculum_hint,
            "seed": seed,
        })

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one action.

        Args:
            action: string in format "ACTION: TOOL_NAME: argument"
                    e.g. "ACTION: CHECK_METRICS: app"

        Returns:
            (observation, reward, done, info)
            reward is non-zero only at episode end
            info contains all 8 reward components when done=True
        """
        resp = self._post("step", {"action": action})
        obs = resp["observation"]
        return obs, resp["reward"], resp["done"], resp["info"]

    def state(self) -> Dict[str, Any]:
        """Get current episode state (for debugging)."""
        return self._get("state")

    def health(self) -> bool:
        """Check if the server is running."""
        try:
            r = self._get("health")
            return r.get("status") == "ok"
        except Exception:
            return False

    def scenarios(self) -> List[Dict[str, Any]]:
        """List all available incident scenarios."""
        return self._get("scenarios")["scenarios"]

    def action_space(self) -> Dict[str, Any]:
        """Get the action space description."""
        return self._get("action_space")

    def observation_space(self) -> Dict[str, Any]:
        """Get the observation space description."""
        return self._get("observation_space")

    # ── Typed interface ───────────────────────────────────────────────────────

    def reset_typed(
        self,
        scenario_name: Optional[str] = None,
        curriculum_hint: bool = False,
        seed: Optional[int] = None,
    ) -> CrisisObservation:
        """Same as reset() but returns a typed CrisisObservation."""
        data = self.reset(scenario_name, curriculum_hint, seed)
        return self._parse_obs(data)

    def step_typed(self, action: str) -> CrisisStepResult:
        """Same as step() but returns a typed CrisisStepResult."""
        obs_dict, reward, done, info = self.step(action)
        return CrisisStepResult(
            observation=self._parse_obs(obs_dict),
            reward=reward,
            done=done,
            info=info,
        )

    def state_typed(self) -> CrisisState:
        """Same as state() but returns a typed CrisisState."""
        data = self.state()
        return CrisisState(
            scenario_name=data.get("scenario_name"),
            step_count=data.get("step_count", 0),
            max_steps=data.get("max_steps", 12),
            done=data.get("done", False),
            timed_out=data.get("timed_out", False),
            diagnosis_made=data.get("diagnosis_made", False),
            resolution_attempted=data.get("resolution_attempted", False),
            components_checked=data.get("components_checked", []),
            status=data.get("status", "unknown"),
        )

    def _parse_obs(self, data: dict) -> CrisisObservation:
        return CrisisObservation(
            text=data.get("text", ""),
            step=data.get("step", 0),
            steps_remaining=data.get("steps_remaining", 12),
            done=data.get("done", False),
            scenario=data.get("scenario"),
            hint=data.get("hint"),
            last_tool=data.get("last_tool"),
            last_argument=data.get("last_argument"),
            diagnosis_made=data.get("diagnosis_made", False),
            resolution_attempted=data.get("resolution_attempted", False),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context manager wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CrisisRoomEnvClient:
    """
    CrisisRoom environment client with context manager support.

    Example:
        with CrisisRoomEnvClient("http://localhost:7860") as env:
            obs = env.reset(scenario_name="traffic_spike")
            print(obs["text"])

            obs, reward, done, info = env.step("ACTION: CHECK_METRICS: app")
            obs, reward, done, info = env.step("ACTION: CHECK_METRICS: api")
            obs, reward, done, info = env.step("ACTION: DIAGNOSE: traffic spike")
            obs, reward, done, info = env.step("ACTION: SCALE_UP: app")
            obs, reward, done, info = env.step("ACTION: RESOLVE: scaled up app")

            print("Total reward:", reward)
            print("Diagnosis correct:", info.get("reward_diagnosis_correct"))
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self._client = CrisisRoomClient(base_url)

    def reset(self, **kwargs) -> Dict[str, Any]:
        return self._client.reset(**kwargs)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return self._client.step(action)

    def state(self) -> Dict[str, Any]:
        return self._client.state()

    def health(self) -> bool:
        return self._client.health()

    def scenarios(self) -> List[Dict[str, Any]]:
        return self._client.scenarios()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    print("Testing CrisisRoom client at: {}\n".format(url))

    client = CrisisRoomClient(url)

    # Health check
    if not client.health():
        print("ERROR: Server not reachable at {}".format(url))
        sys.exit(1)
    print("Health check: OK")

    # List scenarios
    scenarios = client.scenarios()
    print("Scenarios available: {}".format([s["name"] for s in scenarios]))

    # Quick episode
    print("\nRunning quick test episode (bad_deployment)...")
    obs = client.reset(scenario_name="bad_deployment", seed=42)
    print("Initial alert:", obs["text"][:80], "...")

    test_actions = [
        "ACTION: CHECK_METRICS: app",
        "ACTION: CHECK_DEPLOYMENT: app",
        "ACTION: DIAGNOSE: bad deployment memory leak",
        "ACTION: ROLLBACK: app",
        "ACTION: RESOLVE: rolled back app to fix memory leak",
    ]

    for action in test_actions:
        obs, reward, done, info = client.step(action)
        print("  {} -> done={} reward={:.3f}".format(action[8:40], done, reward))
        if done:
            print("\nEpisode complete!")
            print("  Total reward     :", reward)
            print("  Diagnosis correct:", info.get("reward_diagnosis_correct"))
            print("  Fix correct      :", info.get("reward_remediation_correct"))
            print("  Steps taken      :", info.get("step"))
            break

    print("\nClient test PASSED")
