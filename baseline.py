"""
CrisisRoom – Baseline Inference Script

Runs a rule-based deterministic agent across all 5 incident scenarios
and prints reproducible scores. Also supports testing a live HF Space.

Usage:
    python baseline.py                            # run locally
    python baseline.py --url https://USER-crisisroom.hf.space
"""

import argparse
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "server"))

from server.CrisisRoom_environment import CrisisRoomEnv, ALL_SCENARIOS


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based baseline policies (deterministic, one per scenario)
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_POLICIES = {
    "bad_deployment": [
        "ACTION: CHECK_METRICS: app",
        "ACTION: CHECK_DEPLOYMENT: app",
        "ACTION: CHECK_METRICS: db",
        "ACTION: CHECK_LOGS: app",
        "ACTION: DIAGNOSE: bad deployment memory leak in recent release",
        "ACTION: ROLLBACK: app",
        "ACTION: RESOLVE: rolled back app service to previous stable version",
    ],
    "db_connection_exhaustion": [
        "ACTION: CHECK_LOGS: db",
        "ACTION: CHECK_METRICS: db",
        "ACTION: CHECK_LOGS: app",
        "ACTION: CHECK_METRICS: app",
        "ACTION: DIAGNOSE: db connection pool exhausted idle in transaction",
        "ACTION: RESTART_SERVICE: db",
        "ACTION: RESOLVE: restarted database to clear connection pool exhaustion",
    ],
    "cache_stampede": [
        "ACTION: CHECK_METRICS: cache",
        "ACTION: CHECK_METRICS: db",
        "ACTION: CHECK_LOGS: cache",
        "ACTION: CHECK_LOGS: app",
        "ACTION: DIAGNOSE: cache stampede thundering herd mass ttl expiry",
        "ACTION: FLUSH_CACHE: cache",
        "ACTION: RESOLVE: flushed cache to resolve stampede and allow controlled warming",
    ],
    "network_partition": [
        "ACTION: CHECK_LOGS: app",
        "ACTION: CHECK_METRICS: app",
        "ACTION: CHECK_METRICS: db",
        "ACTION: CHECK_LOGS: db",
        "ACTION: DIAGNOSE: network partition packet loss between app and db",
        "ACTION: RESTART_SERVICE: api",
        "ACTION: RESOLVE: restarted api gateway to force network re-routing",
    ],
    "traffic_spike": [
        "ACTION: CHECK_METRICS: app",
        "ACTION: CHECK_METRICS: api",
        "ACTION: CHECK_DEPLOYMENT: app",
        "ACTION: CHECK_LOGS: app",
        "ACTION: DIAGNOSE: traffic spike overwhelming pod capacity cpu maxed",
        "ACTION: SCALE_UP: app",
        "ACTION: RESOLVE: scaled up app replicas to handle 10x traffic surge",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Local runner
# ─────────────────────────────────────────────────────────────────────────────

def run_local():
    print("Running rule-based baseline agent locally...\n")
    results = {}

    for scenario in ALL_SCENARIOS:
        name = scenario.name
        actions = BASELINE_POLICIES.get(name, [])

        env = CrisisRoomEnv(max_steps=12, red_herring_prob=0.0, seed=42)
        env.reset(scenario_name=name)

        final_reward = 0.0
        final_info = {}
        steps_taken = 0

        for action in actions:
            obs, reward, done, info = env.step(action)
            steps_taken += 1
            if done:
                final_reward = reward
                final_info = info
                break

        # Extract reward components
        components = {
            k: v for k, v in final_info.items()
            if k.startswith("reward_") and k != "reward_explanations"
        }

        results[name] = {
            "reward": round(final_reward, 3),
            "steps": steps_taken,
            "diagnosis_correct": final_info.get("reward_diagnosis_correct", 0) > 0,
            "remediation_correct": final_info.get("reward_remediation_correct", 0) > 0,
            "components": components,
        }

        status = "✓ PASS" if final_reward > 5.0 else "✗ FAIL"
        print("  [{:<28}]  reward={:>6.3f}  steps={}  {}".format(
            name, final_reward, steps_taken, status))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Remote runner (HF Space)
# ─────────────────────────────────────────────────────────────────────────────

def run_remote(url):
    import urllib.request
    import urllib.error

    base = url.rstrip("/")

    def post(endpoint, data=None):
        req = urllib.request.Request(
            "{}/{}".format(base, endpoint),
            data=json.dumps(data or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def get(endpoint):
        with urllib.request.urlopen(
            "{}/{}".format(base, endpoint), timeout=15
        ) as r:
            return json.loads(r.read())

    print("Pinging {}/health ...".format(base))
    h = get("health")
    print("  -> {}\n".format(h))

    print("Checking {}/scenarios ...".format(base))
    s = get("scenarios")
    print("  -> {} scenarios: {}\n".format(
        s["total"], [x["name"] for x in s["scenarios"]]))

    results = {}
    for scenario_name, actions in BASELINE_POLICIES.items():
        post("reset", {"scenario_name": scenario_name,
                       "curriculum_hint": False, "seed": 42})
        final_reward = 0.0
        final_info = {}
        steps_taken = 0

        for action in actions:
            resp = post("step", {"action": action})
            steps_taken += 1
            if resp["done"]:
                final_reward = resp["reward"]
                final_info = resp["info"]
                break

        results[scenario_name] = {
            "reward": round(final_reward, 3),
            "steps": steps_taken,
            "diagnosis_correct": final_info.get("reward_diagnosis_correct", 0) > 0,
            "remediation_correct": final_info.get("reward_remediation_correct", 0) > 0,
        }
        status = "✓ PASS" if final_reward > 5.0 else "✗ FAIL"
        print("  [{:<28}]  reward={:>6.3f}  steps={}  {}".format(
            scenario_name, final_reward, steps_taken, status))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CrisisRoom – Baseline Inference Runner")
    parser.add_argument(
        "--url", default=None,
        help="HF Space URL to test remotely (omit to run locally)")
    args = parser.parse_args()

    print("=" * 65)
    print("  CrisisRoom – SRE Incident Response RL Environment")
    print("  Baseline Rule-Based Agent")
    print("=" * 65)
    print()

    if args.url:
        results = run_remote(args.url)
    else:
        results = run_local()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FINAL SCORES")
    print("=" * 65)

    all_rewards = []
    all_diag = []
    all_remed = []

    for name, data in results.items():
        reward = data["reward"]
        bar = "#" * int(max(reward, 0) / 15.0 * 30)
        all_rewards.append(reward)
        all_diag.append(1.0 if data["diagnosis_correct"] else 0.0)
        all_remed.append(1.0 if data["remediation_correct"] else 0.0)
        print("  {:<28}  [{:<30}]  {:.3f}".format(name, bar, reward))

    print()
    print("  Overall mean reward  : {:.3f}".format(
        sum(all_rewards) / len(all_rewards)))
    print("  Diagnosis accuracy   : {:.0f}%".format(
        sum(all_diag) / len(all_diag) * 100))
    print("  Resolution rate      : {:.0f}%".format(
        sum(all_remed) / len(all_remed) * 100))
    print()

    # Print reward component breakdown for first scenario
    if not args.url and results:
        first = list(results.values())[0]
        if "components" in first:
            print("  Reward component breakdown (first scenario):")
            for k, v in sorted(first["components"].items()):
                print("    {:<40} {}".format(k, v))
        print()

    passed = all(r > 5.0 for r in all_rewards)
    if passed:
        print("PASSED: Baseline completed successfully – all scenarios scored > 5.0")
    else:
        failed = [n for n, d in results.items() if d["reward"] <= 5.0]
        print("WARNING: Some scenarios scored low: {}".format(failed))
