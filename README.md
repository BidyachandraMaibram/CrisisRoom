---
title: CrisisRoom
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🚨 CrisisRoom – SRE Incident Response RL Environment

> **Hackathon Theme**: World Modeling – Professional Tasks (Theme 3.1)  
> **Stack**: OpenEnv + TRL (GRPO) + Unsloth + HuggingFace  
> **Model**: Qwen2.5-3B-Instruct

---

## Overview

**CrisisRoom** is a reinforcement learning environment where an LLM agent acts as an on-call **Site Reliability Engineer (SRE)** diagnosing and resolving production incidents inside a simulated company infrastructure.

This is a **World Modeling** environment: the agent cannot see the full system state. It must actively query tools, interpret noisy outputs, build an internal model of what is broken, and apply the correct remediation — all within a limited step budget.

---

## The Problem

When a production incident fires, an on-call SRE must:
1. **Triage** — what is the user-visible impact?
2. **Investigate** — query logs, metrics, deployments across services
3. **Diagnose** — identify the root cause from noisy, partially misleading signals
4. **Remediate** — apply the correct fix (rollback, restart, scale-up, etc.)
5. **Resolve** — confirm the incident is closed

This requires **causal reasoning**, **belief updating**, and **systematic tool use** — exactly the skills that make RL environment training valuable.

---

## Environment Design

### System Architecture

The simulated production system has 5 interconnected services:

```
         Internet
            │
        ┌───▼───┐
        │  api  │  ← API Gateway (handles inbound traffic)
        └───┬───┘
            │
        ┌───▼───┐
        │  app  │  ← Application Server (business logic)
        └──┬─┬──┘
           │ │
    ┌──────▼┐│┌──────┐
    │  db   ││││cache │  ← PostgreSQL + Redis
    └───────┘│└──────┘
             │
         ┌───▼────┐
         │ deploy │  ← Deployment pipeline
         └────────┘
```

### Partial Observability

- The agent starts with **only a high-level alert** (e.g., "payments service is down")
- It does NOT know the root cause
- Tool outputs are **realistic but noisy** (30% of calls include red herring signals)
- The agent must use multiple tool calls to build situational awareness

---

## Action Space

The agent responds with **exactly one action per turn**:

```
ACTION: TOOL_NAME: argument
```

| Tool | Argument | Description |
|------|----------|-------------|
| `CHECK_LOGS` | `service` | Last 10 log lines (noisy, may have red herrings) |
| `CHECK_METRICS` | `service` | CPU, memory, latency, error rate |
| `CHECK_DEPLOYMENT` | `service` | Last 3 deployments + diffs |
| `RESTART_SERVICE` | `service` | Restart the service pods |
| `ROLLBACK` | `service` | Roll back to previous deployment |
| `SCALE_UP` | `service` | Increase pod replica count |
| `FLUSH_CACHE` | `cache` | Flush the Redis cache |
| `DIAGNOSE` | `root_cause` | Commit to a root cause diagnosis |
| `RESOLVE` | `action_taken` | Declare incident resolved |

**Valid services**: `api`, `app`, `db`, `cache`, `deploy`

**Examples**:
```
ACTION: CHECK_METRICS: app
ACTION: CHECK_LOGS: db
ACTION: DIAGNOSE: database connection pool exhausted
ACTION: RESTART_SERVICE: db
ACTION: RESOLVE: restarted database to clear connection pool exhaustion
```

---

## Incident Scenarios

Five distinct root causes, each with different causal chains:

| Scenario | Root Cause | Correct Fix | Key Signals |
|----------|-----------|-------------|-------------|
| `bad_deployment` | Memory leak in v3.2.1 | `ROLLBACK: app` | `CHECK_METRICS:app` (mem 96%), `CHECK_DEPLOYMENT:app` (13min ago) |
| `db_connection_exhaustion` | Pool full (41 idle-in-tx) | `RESTART_SERVICE: db` | `CHECK_LOGS:db` (pool exhausted), `CHECK_METRICS:db` (100/100) |
| `cache_stampede` | 24,000 keys expired simultaneously | `FLUSH_CACHE: cache` | `CHECK_METRICS:cache` (hit rate 2%), `CHECK_METRICS:db` (CPU 91%) |
| `network_partition` | Packet loss between app and db | `RESTART_SERVICE: api` | `CHECK_LOGS:app` (ECONNRESET), `CHECK_METRICS:app` (intermittent errors) |
| `traffic_spike` | 10× traffic surge | `SCALE_UP: app` | `CHECK_METRICS:app` (CPU 99%), `CHECK_METRICS:api` (3200 rps) |

---

## Reward Model

All rewards are **programmatic and verifiable** — no learned reward model.

| Component | Range | Description |
|-----------|-------|-------------|
| `reward_diagnosis_correct` | −2 / −1 / +4 | Did DIAGNOSE match the true root cause? |
| `reward_remediation_correct` | −2 / 0 / +5 | Did RESOLVE action match the correct fix? |
| `reward_causal_reasoning` | 0 / +2 | Checked both failing service AND its dependency before diagnosing |
| `reward_efficiency` | 0 / +1 / +2 | Resolved in ≤9 or ≤6 steps |
| `reward_investigation_quality` | 0 to +1.5 | +0.3 per unique service inspected (max 5) |
| `reward_red_herring_resistance` | 0 / +1 | Did NOT act on misleading signals |
| `reward_timeout_penalty` | −4 / 0 | Episode timed out without resolution |
| `reward_premature_action_penalty` | −1 per action | Remediation called before DIAGNOSE |

**Total reward range**: approximately −10 to +16.5

**Maximum achievable reward** (perfect episode): +15.5  
- Correct diagnosis: +4  
- Correct fix: +5  
- Full causal exploration: +2  
- Efficient (≤6 steps): +2  
- 5 services investigated: +1.5  
- No red herring actions: +1  

### Why Multiple Independent Rewards?

Each component catches a different failure mode:
- A lucky guesser gets `reward_remediation_correct` but misses `reward_causal_reasoning`
- A spray-and-pray agent loses `reward_premature_action_penalty`
- A red-herring follower loses `reward_red_herring_resistance`
- A slow deliberator misses `reward_efficiency`

---

## Curriculum Learning

Training uses a **two-phase curriculum**:

- **Phase 1** (first 30% of training steps): Agent receives a hint pointing to which service category to check first. This ensures initial rollouts have non-zero reward probability.
- **Phase 2** (remaining 70%): No hints. Agent must discover the root cause from scratch.

---

## Running Locally

### Prerequisites

```bash
pip install fastapi uvicorn pydantic requests
```

### Start the environment server

```bash
cd crisisroom
python server/app.py
```

Server starts at `http://localhost:7860`

### Interactive API

Open `http://localhost:7860/docs` for the Swagger UI.

### Quick test

```bash
# Reset (start episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"curriculum_hint": false}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ACTION: CHECK_METRICS: app"}'

# Take another action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ACTION: CHECK_DEPLOYMENT: app"}'

# Diagnose
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ACTION: DIAGNOSE: bad deployment with memory leak"}'

# Fix and resolve
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ACTION: ROLLBACK: app"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ACTION: RESOLVE: rolled back app to previous version"}'
```

### Pin a specific scenario for testing

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_name": "bad_deployment", "curriculum_hint": true}'
```

---

## Running GRPO Training

### Install training dependencies

```bash
pip install torch transformers trl peft datasets accelerate bitsandbytes
pip install unsloth   # recommended for memory efficiency
```

### Set environment variables

```bash
export HF_TOKEN=your_huggingface_token
export HF_REPO_ID=your-username/crisisroom-sre-agent
```

### Run training

```bash
cd crisisroom
python training/grpo_train.py
```

The training script will:
1. Start the environment server automatically
2. Load Qwen2.5-3B-Instruct with 4-bit QLoRA via Unsloth
3. Run GRPO training with curriculum learning
4. Evaluate every 100 steps (logging diagnosis accuracy, resolution rate, mean reward)
5. Save best checkpoint to `./checkpoints/crisisroom-best/`
6. Push to HuggingFace Hub (if `HF_TOKEN` is set)

---

## Deploying to HuggingFace Spaces

### Method 1: Docker Space (recommended)

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the Space SDK
3. Clone the Space repository and copy the project files:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/crisisroom
cp -r crisisroom/* crisisroom/
git add .
git commit -m "Add CrisisRoom environment"
git push
```

The Dockerfile exposes port 7860 as required by HuggingFace Spaces.

### Method 2: Using OpenEnv CLI

```bash
pip install openenv
openenv init crisisroom
# Follow prompts to configure
openenv push --space YOUR_USERNAME/crisisroom
```

### Verify deployment

```bash
curl https://YOUR_USERNAME-crisisroom.hf.space/health
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/observation_space` | GET | Observation space description |
| `/action_space` | GET | Action space + all tools |
| `/scenarios` | GET | List all 5 incident scenarios |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute one action |
| `/state` | GET | Current episode state (debug) |

---

## File Structure

```
crisisroom/
├── environment/
│   ├── __init__.py
│   ├── env.py          # CrisisRoomEnv (OpenEnv-compliant)
│   ├── scenarios.py    # 5 incident scenarios with tool output templates
│   ├── tools.py        # Tool execution + noise/red herring injection
│   └── rewards.py      # 8 independent reward functions
├── server/
│   ├── app.py          # FastAPI server
│   └── schemas.py      # Pydantic v2 schemas
├── training/
│   ├── grpo_train.py   # Full GRPO training script (TRL + Unsloth)
│   └── rollout.py      # Episode rollout + prompt construction
├── Dockerfile          # HuggingFace Spaces compatible
├── requirements.txt
└── README.md
```

---

## Reward Anti-Hacking Design

The environment uses multiple independent reward components to prevent gaming:

1. **Executable verification** — all rewards are rule-based, not learned
2. **Causal chain requirement** — agent must explore the right services
3. **Premature action penalty** — spray-and-pray is penalised
4. **Red herring resistance** — misleading signals are injected at 30% probability
5. **Step budget** — 12-step timeout prevents exhaustive search
6. **Multi-component rewards** — no single signal can be maximised in isolation

---

## Training Metrics to Track

| Metric | What It Reveals |
|--------|-----------------|
| `reward_total` | Overall performance |
| `reward_diagnosis_correct` | Root cause identification capability |
| `reward_remediation_correct` | Fix selection capability |
| `reward_causal_reasoning` | World model quality (not guessing) |
| `reward_efficiency` | Operational speed |
| `reward_red_herring_resistance` | Noise filtering |
| `reward_premature_action_penalty` | Discipline (diagnose before acting) |
| `diagnosis_accuracy` (eval) | % of episodes with correct diagnosis |
| `resolution_rate` (eval) | % of episodes fully resolved |

---

## Example Ideal Episode (bad_deployment scenario)

```
[Step 1] CHECK_METRICS: app
→ "cpu_pct=62.1  mem_pct=96.8  latency_p99_ms=28400
   heap_used_mb=7812  thread_pool_queue=2048 (FULL)"

[Step 2] CHECK_DEPLOYMENT: app
→ "deploy[0] 2024-01-15T14:10:00Z  v3.2.1  feat: new PaymentProcessor refactor
   changed files: PaymentProcessor.java, MemoryPool.java
   STATUS: deploy[0] is 13 minutes old – correlates with incident start"

[Step 3] CHECK_METRICS: db
→ "cpu_pct=8.3  mem_pct=34.1  connections_active=12 — db healthy"

[Step 4] DIAGNOSE: bad deployment - memory leak in v3.2.1 app release
→ "Diagnosis recorded"

[Step 5] ROLLBACK: app
→ "Rolling back to v3.2.0... 3/3 pods healthy. Memory normalised: 42%"

[Step 6] RESOLVE: rolled back app service from v3.2.1 to v3.2.0
→ "Incident resolved"

REWARD: diagnosis_correct=+4.0, remediation_correct=+5.0,
        causal_reasoning=+2.0 (checked app+deploy), efficiency=+2.0 (6 steps),
        investigation_quality=+0.9 (3 services), red_herring_resistance=+1.0
        TOTAL: +14.9
```

---

## Citation / Attribution

Built for the **Meta PyTorch OpenEnv Hackathon – April 2026**  
Theme 3.1: World Modeling – Professional Tasks

Stack: OpenEnv · TRL · Unsloth · HuggingFace 🤗