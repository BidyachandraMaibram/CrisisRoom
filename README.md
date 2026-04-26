---
title: CrisisRoom
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🚨 CrisisRoom — SRE Incident Response RL Environment

> *Can an LLM learn to diagnose and resolve production incidents the way a senior SRE would?*

## The Problem

LLMs are increasingly used in agentic workflows — but *incident response* is one of the hardest professional tasks to evaluate. It requires:

- **Causal reasoning**: trace symptoms back to root cause across a distributed system
- **Information triage**: distinguish real signals from noise and red herrings
- **Decision sequencing**: investigate *before* acting, not after
- **Efficiency under pressure**: 12 steps, the clock is ticking

Today's LLMs struggle at all four. CrisisRoom gives them a training environment to get better.

---

## The Environment

CrisisRoom is an OpenEnv-compliant RL environment where an agent plays the role of an on-call Site Reliability Engineer. It receives a production incident alert and must diagnose and resolve it using tool calls.

### What the agent sees

An alert like:
```
🚨 ALERT [P0]: payments service is down.
Impact: 100% of payment transactions failing. Incident started: 14:23 UTC.

Available services: api, app, db, cache, deploy
Format: ACTION: TOOL_NAME: argument
```

### What the agent can do

| Tool              | Argument    | Effect |
|-------------------|-------------|--------|
| `CHECK_LOGS`      | service     | Last 10 log lines (noisy, may contain red herrings) |
| `CHECK_METRICS`   | service     | CPU, memory, latency, error rate snapshot |
| `CHECK_DEPLOYMENT`| service     | Recent deploy history and diffs |
| `RESTART_SERVICE` | service     | Restart service pods |
| `ROLLBACK`        | service     | Roll back last deployment |
| `SCALE_UP`        | service     | Increase pod replicas |
| `FLUSH_CACHE`     | cache       | Flush Redis cache |
| `DIAGNOSE`        | description | Commit to a root cause hypothesis |
| `RESOLVE`         | description | Declare the incident resolved |

### The 5 Incident Scenarios

Each episode draws randomly from one of five realistic production incidents:

| Scenario                     | Root Cause                                          | Correct Fix              |
|------------------------------|-----------------------------------------------------|--------------------------|
| **Bad Deployment**           | Memory leak introduced in PaymentProcessor refactor | `ROLLBACK: app`          |
| **DB Connection Exhaustion** | Idle-in-transaction connections leaking the pool    | `RESTART_SERVICE: db`    |
| **Cache Stampede**           | Redis keys expired simultaneously                   | `FLUSH_CACHE: cache`     |
| **Network Partition**        | Intermittent packet loss between app tier and database | `RESTART_SERVICE: api`|
| **Traffic Spike**            | 10× inbound load overwhelming 3-pod capacity           | `SCALE_UP: app`       |

Each scenario has richly detailed, realistic tool outputs — logs with timestamps, metric snapshots, deployment diffs — plus **red herrings** injected at 30% probability to test signal-from-noise reasoning.

---

## Reward Design

The reward function measures eight independent, fully verifiable components:

| Component                 | Range     | What it tests |
| `diagnosis_correct`       | −10 / +20 | Root cause identified correctly |
| `remediation_correct`     | −15 / +30 | Correct fix applied |
| `causal_reasoning`        | 0 / +10   | Explored all services in the causal chain *before* diagnosing |
| `efficiency`              | 0 / +10   | Resolved in ≤6 steps (+10), ≤9 steps (+5), or more (0) |
| `investigation_quality`   | 0 / +10   | Number of distinct services inspected (2 pts each) |
| `red_herring_resistance`  | 0 / +5    | Didn't act on misleading signals |
| `timeout_penalty`         | −30 / 0   | Ran out of steps without resolving |
| `premature_action_penalty`| −10× / 0  | Attempted remediation before `DIAGNOSE` |

**Max achievable reward: ~80 points.** A random agent typically scores below 0.

---

## Training

We fine-tuned **Qwen2.5-3B-Instruct** using **GRPO** (Group Relative Policy Optimization) with **Unsloth 4-bit QLoRA**.

**Setup:**
- Model: `Qwen/Qwen2.5-3B-Instruct` with LoRA rank 100
- Optimizer: AdamW, lr=5e-5
- Curriculum: First 30% of steps include a hint pointing toward the relevant service
- Environment: max 12 steps per episode, 30% red herring injection rate

**Training loop connects live to the environment** — no static dataset. Each GRPO update runs real rollouts against the FastAPI server and uses episode rewards directly as the training signal.

```bash
# Re-run training (GPU required)
python training/grpo_train.py

# Or open the Colab notebook:
# [link to Colab notebook]
```

---

## Results


| Metric                | Untrained Baseline | Trained Agent | Improvement |
|-----------------------|--------------------|---------------|-------------|
| Mean reward           | 8.400              | 13.200        | +4.8        |
| Diagnosis accuracy    | 0.160              | 0.700         | +0.54       |
| Resolution rate       | 0.180              | 0.280         | +0.100      |

**Baseline vs Trained:** `plots/baseline_vs_trained.png` (https://huggingface.co/spaces/Maibram1/CrisisRoom/blob/main/Plot/baseline_vs_trained.png)

**Loss curve:** `plots/loss_curve.png` (https://huggingface.co/spaces/Maibram1/CrisisRoom/blob/main/Plot/loss_curve.png)

**Reward curve:** `plots/reward_curve.png` (https://huggingface.co/spaces/Maibram1/CrisisRoom/blob/main/Plot/reward_curve.png)
---

## Try It

**HuggingFace Space :** (https://huggingface.co/spaces/Maibram1/CrisisRoom)

**Run the rule-based baseline locally:**
```bash
git clone https://huggingface.co/spaces/Maibram1/CrisisRoom
cd CrisisRoom
pip install -e .
python baseline.py
```

**Test against the live Space:**
```bash
python baseline.py --url https://maibram1-crisisroom.hf.space
```

**Interact via client:**
```python
from client import CrisisRoomClient

client = CrisisRoomClient("http://localhost:7860")
obs = client.reset(scenario_name="cache_stampede")
print(obs["text"])

obs, reward, done, info = client.step("ACTION: CHECK_METRICS: cache")
print(obs["text"])
```

---

## Why It Matters

SRE incident response is a high-stakes, time-sensitive professional task that currently has almost no RL training environment coverage. CrisisRoom provides:

- A **realistic, multi-step** decision problem (not grid-world trivialities)
- **Dense, composable rewards** — each component teaches a distinct SRE skill
- A **benchmark** that could grow: more scenarios, noisier signals, multi-service cascades
- A template for using OpenEnv to turn **professional workflows** into LLM training environments

Anyone building reliability agents, autonomous DevOps tooling, or studying LLM tool-use would find this environment useful.

---

## Repository Structure

```
CrisisRoom/
├── server/
│   ├── CrisisRoom_environment.py  # All environment logic (scenarios, tools, rewards)
│   └── app.py                     # FastAPI server
├── training/
│   ├── grpo_train.py              # GRPO training script (Unsloth + TRL)
│   └── rollout.py                 # Episode rollout utilities
├── client.py                      # HTTP client for env interaction
├── baseline.py                    # Rule-based baseline agent
├── openenv.yaml                   # OpenEnv manifest
└── Dockerfile
```

---

## Additional Materials

- 📓 **Training Notebook (colab):** (https://colab.research.google.com/drive/1mjTYRuohou4yWkOeKrFXKGssedSOMUAY#scrollTo=0c40361b)
- 📝 **Blog Post / Writeup:** (https://huggingface.co/spaces/Maibram1/CrisisRoom/blob/main/Blog.md)

---

## Citation / Theme

Built for the **OpenEnv RL Environment Challenge** — Theme 3.1: *World Modeling: Professional Tasks*
