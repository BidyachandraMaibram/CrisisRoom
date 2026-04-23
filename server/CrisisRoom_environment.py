"""
CrisisRoom_environment.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All environment logic in one file — the format expected by
`openenv init`-generated scaffolds.

Sections:
  1. Scenario definitions  (5 incidents)
  2. Tool execution + noise/red-herring injection
  3. Reward functions      (8 independent, fully verifiable)
  4. CrisisRoomEnv class   (OpenEnv-compliant reset/step/state)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

SERVICES = ["api", "app", "db", "cache", "deploy"]

TOOLS = [
    "CHECK_LOGS", "CHECK_METRICS", "CHECK_DEPLOYMENT",
    "RESTART_SERVICE", "ROLLBACK", "SCALE_UP", "FLUSH_CACHE",
    "DIAGNOSE", "RESOLVE",
]


@dataclass
class ScenarioSpec:
    name: str
    description: str
    root_cause: str
    correct_fix: str                  # "TOOL:service"
    causal_chain: List[str]           # services agent MUST check for causal credit
    tool_outputs: Dict[str, List[str]]
    signal_combos: List[str]
    hint: str


# ─── Scenario 1: bad_deployment ───────────────────────────────────────────────
BAD_DEPLOYMENT = ScenarioSpec(
    name="bad_deployment",
    description="A recent deployment introduced a memory leak in the app server.",
    root_cause="bad_deployment",
    correct_fix="ROLLBACK:app",
    causal_chain=["app", "deploy"],
    signal_combos=["CHECK_METRICS:app", "CHECK_DEPLOYMENT:app"],
    hint="Start by checking the application server metrics and recent deployments.",
    tool_outputs={
        "CHECK_LOGS:api": [
            "2024-01-15 14:23:01 INFO  GET /v1/payments 200 45ms\n"
            "2024-01-15 14:23:03 WARN  upstream app server slow response 312ms\n"
            "2024-01-15 14:23:07 ERROR upstream app server timeout after 30s\n"
            "2024-01-15 14:23:09 INFO  retry attempt 1 for request abc-123\n"
            "2024-01-15 14:23:12 ERROR upstream app server timeout after 30s\n"
            "2024-01-15 14:23:14 INFO  circuit breaker OPEN for app service\n"
            "2024-01-15 14:23:15 ERROR 503 Service Unavailable returned to client\n"
            "2024-01-15 14:23:16 WARN  health check app:8080/health failing\n"
            "2024-01-15 14:23:20 INFO  GET /v1/payments 503 1ms (circuit open)\n"
            "2024-01-15 14:23:22 WARN  dropping 100% of traffic to app tier",
        ],
        "CHECK_METRICS:api": [
            "service=api snapshot_time=14:23:22\n"
            "cpu_pct=18.2  mem_pct=41.0  latency_p99_ms=4200\n"
            "request_rate_rps=340  error_rate_pct=97.4\n"
            "upstream_errors=3312  circuit_breaker=OPEN",
        ],
        "CHECK_DEPLOYMENT:api": [
            "service=api\n"
            "deploy[0] 2024-01-15T10:00:00Z  v2.1.4  patch: update rate-limit config\n"
            "deploy[1] 2024-01-14T08:30:00Z  v2.1.3  patch: fix header parsing\n"
            "deploy[2] 2024-01-13T15:10:00Z  v2.1.2  minor: add correlation-id logging\n"
            "STATUS: no recent changes, last 3 deploys are minor/stable",
        ],
        "CHECK_LOGS:app": [
            "2024-01-15 14:20:01 INFO  app-server v3.2.1 started\n"
            "2024-01-15 14:20:45 WARN  heap usage 78% (threshold 70%)\n"
            "2024-01-15 14:21:10 WARN  GC pause 820ms – heap pressure detected\n"
            "2024-01-15 14:21:55 ERROR OutOfMemoryError: Java heap space\n"
            "2024-01-15 14:21:55 ERROR thread pool exhausted – rejecting new requests\n"
            "2024-01-15 14:22:01 WARN  heap usage 94% – GC running constantly\n"
            "2024-01-15 14:22:30 ERROR request processing failed: heap overflow\n"
            "2024-01-15 14:22:45 WARN  memory leak suspected in PaymentProcessor\n"
            "2024-01-15 14:23:00 ERROR service degraded – health check returning 503\n"
            "2024-01-15 14:23:10 CRIT  pod nearing OOM kill threshold",
        ],
        "CHECK_METRICS:app": [
            "service=app snapshot_time=14:23:22\n"
            "cpu_pct=62.1  mem_pct=96.8  latency_p99_ms=28400\n"
            "request_rate_rps=0  error_rate_pct=100.0\n"
            "heap_used_mb=7812  heap_max_mb=8192  gc_pause_ms=1240\n"
            "thread_pool_active=200  thread_pool_queue=2048 (FULL)",
        ],
        "CHECK_DEPLOYMENT:app": [
            "service=app\n"
            "deploy[0] 2024-01-15T14:10:00Z  v3.2.1  feat: new PaymentProcessor refactor\n"
            "  changed files: PaymentProcessor.java, MemoryPool.java, config/jvm.yml\n"
            "  author: dev-team-payments  review: 1 approval (expedited)\n"
            "deploy[1] 2024-01-14T09:00:00Z  v3.2.0  fix: payment retry logic\n"
            "deploy[2] 2024-01-12T11:00:00Z  v3.1.9  chore: dependency updates\n"
            "STATUS: deploy[0] is 13 minutes old – correlates with incident start",
        ],
        "CHECK_LOGS:db": [
            "2024-01-15 14:23:01 INFO  pg: connection from app accepted\n"
            "2024-01-15 14:23:05 INFO  query SELECT * FROM payments 4ms\n"
            "2024-01-15 14:23:15 INFO  connection pool: 12/100 active\n"
            "2024-01-15 14:23:22 INFO  pg: all systems nominal",
        ],
        "CHECK_METRICS:db": [
            "service=db snapshot_time=14:23:22\n"
            "cpu_pct=8.3   mem_pct=34.1  latency_p99_ms=7\n"
            "connections_active=12  connections_max=100\n"
            "STATUS: healthy – db is not the source of the problem",
        ],
        "CHECK_DEPLOYMENT:db": [
            "service=db\ndeploy[0] 2024-01-10T06:00:00Z  v14.2.1  patch: minor update\nSTATUS: stable",
        ],
        "CHECK_LOGS:cache": [
            "2024-01-15 14:23:00 INFO  redis: 42 connected clients\n"
            "2024-01-15 14:23:10 INFO  redis: hit_rate=98.2% – normal\n"
            "STATUS: cache healthy",
        ],
        "CHECK_METRICS:cache": [
            "service=cache snapshot_time=14:23:22\ncpu_pct=4.1  mem_pct=25.6  hit_rate_pct=98.2\nSTATUS: healthy",
        ],
        "CHECK_DEPLOYMENT:cache": ["service=cache\nSTATUS: no recent deployments"],
        "RESTART_SERVICE:app": [
            "Restarting app service…\nWARNING: New pods start same v3.2.1 code – memory leak will recur\n"
            "Pods entered CrashLoopBackOff after 45s\nRESULT: restart FAILED to resolve – root cause persists in code",
        ],
        "ROLLBACK:app": [
            "Rolling back app service to v3.2.0…\n"
            "Rolling update: app-server-0 → v3.2.0 ✓\n"
            "Rolling update: app-server-1 → v3.2.0 ✓\n"
            "Rolling update: app-server-2 → v3.2.0 ✓\n"
            "Memory usage normalised: 42%\nRESULT: rollback SUCCESSFUL – incident resolved",
        ],
        "SCALE_UP:app": [
            "Scaling app from 3 → 6 replicas…\nNew pods inherit v3.2.1 (memory leak)\n"
            "RESULT: scale-up did NOT fix the issue – more pods, same leak",
        ],
        "FLUSH_CACHE:cache": ["Flushing Redis cache…\nRESULT: flush complete – no impact on app memory issue"],
        "RESTART_SERVICE:db": ["Restarting db…\nRESULT: db restarted – no impact on app memory issue"],
        "RESTART_SERVICE:api": ["Restarting api…\nRESULT: api restarted – upstream app still OOM, 503s continue"],
    },
)

# ─── Scenario 2: db_connection_exhaustion ────────────────────────────────────
DB_CONNECTION_EXHAUSTION = ScenarioSpec(
    name="db_connection_exhaustion",
    description="Database connection pool is exhausted; app threads are queued waiting.",
    root_cause="db_connection_exhaustion",
    correct_fix="RESTART_SERVICE:db",
    causal_chain=["db", "app"],
    signal_combos=["CHECK_LOGS:db", "CHECK_METRICS:db"],
    hint="Start by checking the database logs and metrics.",
    tool_outputs={
        "CHECK_LOGS:api": [
            "2024-01-15 15:10:04 WARN  upstream timeout waiting for app response\n"
            "2024-01-15 15:10:08 ERROR 504 Gateway Timeout to client\n"
            "2024-01-15 15:10:20 ERROR circuit breaker at 60% open threshold\n"
            "2024-01-15 15:10:30 ERROR 504 Gateway Timeout to client",
        ],
        "CHECK_METRICS:api": [
            "service=api snapshot_time=15:10:30\ncpu_pct=22.0  latency_p99_ms=28000\nerror_rate_pct=74.2",
        ],
        "CHECK_DEPLOYMENT:api": ["service=api\nSTATUS: no concerning recent changes"],
        "CHECK_LOGS:app": [
            "2024-01-15 15:09:52 WARN  db pool: waiting for connection (queue=45)\n"
            "2024-01-15 15:09:54 WARN  db pool: waiting for connection (queue=89)\n"
            "2024-01-15 15:09:58 ERROR db pool: timeout waiting 5000ms for connection\n"
            "2024-01-15 15:10:02 WARN  db pool: queue=134 – all 100 connections busy\n"
            "2024-01-15 15:10:10 ERROR db pool: exhausted – connection refused",
        ],
        "CHECK_METRICS:app": [
            "service=app snapshot_time=15:10:30\ncpu_pct=31.4  latency_p99_ms=15000\n"
            "thread_pool_waiting_on_db=198  db_pool_active=100  db_pool_queue=134\n"
            "STATUS: app blocked on db connections",
        ],
        "CHECK_DEPLOYMENT:app": ["service=app\nSTATUS: recent deploy but no db-pool config changes"],
        "CHECK_LOGS:db": [
            "2024-01-15 15:08:00 INFO  pg: connection pool 100/100 active\n"
            "2024-01-15 15:08:05 WARN  pg: max_connections=100 reached\n"
            "2024-01-15 15:08:10 ERROR pg: FATAL remaining connection slots reserved\n"
            "2024-01-15 15:09:00 WARN  pg: 23 idle-in-transaction connections detected\n"
            "2024-01-15 15:09:05 WARN  pg: idle_in_transaction_session_timeout not set\n"
            "2024-01-15 15:09:10 ERROR pg: connections leaking – clients not returning pool\n"
            "2024-01-15 15:10:00 WARN  pg: 41 idle-in-transaction connections blocking pool",
        ],
        "CHECK_METRICS:db": [
            "service=db snapshot_time=15:10:30\ncpu_pct=44.1  latency_p99_ms=380\n"
            "connections_active=100  connections_max=100  connections_idle_in_tx=41\n"
            "STATUS: connection pool EXHAUSTED – idle-in-transaction connections leaking",
        ],
        "CHECK_DEPLOYMENT:db": ["service=db\nSTATUS: stable – no recent changes"],
        "CHECK_LOGS:cache": ["STATUS: cache healthy"],
        "CHECK_METRICS:cache": ["service=cache\ncpu_pct=3.2  hit_rate_pct=97.8\nSTATUS: healthy"],
        "CHECK_DEPLOYMENT:cache": ["service=cache\nSTATUS: no recent deployments"],
        "RESTART_SERVICE:db": [
            "Restarting PostgreSQL service…\n"
            "Terminating 100 existing connections (including 41 idle-in-transaction)\n"
            "Connection pool reset: 0/100 active\nHealth check passing\n"
            "RESULT: db restart SUCCESSFUL – connection pool cleared, traffic recovering",
        ],
        "RESTART_SERVICE:app": [
            "Restarting app…\nNew pods immediately contend for db connections\n"
            "Db pool exhausted again within 30s\nRESULT: app restart did NOT fix the issue",
        ],
        "ROLLBACK:app": ["Rolling back app…\nRESULT: rollback did NOT fix – db connections still leaked"],
        "SCALE_UP:app": ["Scaling app…\nMore app pods = MORE db connections\nRESULT: scale-up WORSENED the problem"],
        "FLUSH_CACHE:cache": ["Flushing cache…\nRESULT: flush complete – no impact on db connection exhaustion"],
    },
)

# ─── Scenario 3: cache_stampede ───────────────────────────────────────────────
CACHE_STAMPEDE = ScenarioSpec(
    name="cache_stampede",
    description="Cache TTL expired simultaneously causing thundering herd on the database.",
    root_cause="cache_stampede",
    correct_fix="FLUSH_CACHE:cache",
    causal_chain=["cache", "db"],
    signal_combos=["CHECK_METRICS:cache", "CHECK_METRICS:db"],
    hint="Check the cache hit rate and database load — they may be related.",
    tool_outputs={
        "CHECK_LOGS:api": [
            "2024-01-15 16:00:06 WARN  upstream latency spike: app 1240ms\n"
            "2024-01-15 16:00:10 ERROR upstream timeout 30s – returning 504\n"
            "2024-01-15 16:00:22 ERROR error_rate: 61%",
        ],
        "CHECK_METRICS:api": ["service=api\ncpu_pct=28.0  latency_p99_ms=8400  error_rate_pct=61.0"],
        "CHECK_DEPLOYMENT:api": ["service=api\nSTATUS: stable"],
        "CHECK_LOGS:app": [
            "2024-01-15 16:00:00 WARN  cache miss for key=product_catalog_v2 (x120)\n"
            "2024-01-15 16:00:01 INFO  120 goroutines dispatched to db for product_catalog\n"
            "2024-01-15 16:00:03 WARN  thundering herd – 320 simultaneous db queries\n"
            "2024-01-15 16:00:05 ERROR db query timeout: 5000ms exceeded\n"
            "2024-01-15 16:00:12 WARN  cache write-through storm in progress",
        ],
        "CHECK_METRICS:app": [
            "service=app\ncpu_pct=48.0  latency_p99_ms=6200  error_rate_pct=55.0\n"
            "cache_miss_causing_db_queries=340",
        ],
        "CHECK_DEPLOYMENT:app": ["service=app\nSTATUS: recent deploy but no cache config changes"],
        "CHECK_LOGS:db": [
            "2024-01-15 16:00:00 WARN  pg: query queue depth 0 → 124 in 1s\n"
            "2024-01-15 16:00:01 WARN  pg: CPU 12% → 89% in 1s (query storm)\n"
            "2024-01-15 16:00:06 WARN  pg: 320 simultaneous SELECT queries\n"
            "2024-01-15 16:00:15 WARN  pg: storm appears cache-miss driven",
        ],
        "CHECK_METRICS:db": [
            "service=db\ncpu_pct=91.4  latency_p99_ms=3800\n"
            "connections_active=88  query_rate_qps=340  slow_queries=87\n"
            "STATUS: under extreme query storm – likely cache miss driven",
        ],
        "CHECK_DEPLOYMENT:db": ["service=db\nSTATUS: stable – no recent changes"],
        "CHECK_LOGS:cache": [
            "2024-01-15 15:59:58 INFO  redis: TTL expiry event key=product_catalog_v2\n"
            "2024-01-15 15:59:59 WARN  redis: mass expiry – 24,000 keys expired simultaneously\n"
            "2024-01-15 16:00:00 ERROR redis: stampede detected – all clients querying db\n"
            "2024-01-15 16:00:10 WARN  redis: hit_rate dropped from 98% → 2%\n"
            "2024-01-15 16:00:25 ERROR redis: cache empty – system in degraded fallback mode",
        ],
        "CHECK_METRICS:cache": [
            "service=cache\ncpu_pct=6.2  mem_pct=8.0\n"
            "hit_rate_pct=2.1  keys_expired=24000\n"
            "STATUS: MASS EXPIRY – 24,000 keys expired simultaneously causing stampede",
        ],
        "CHECK_DEPLOYMENT:cache": ["service=cache\nSTATUS: no recent deployments"],
        "FLUSH_CACHE:cache": [
            "Flushing Redis cache…\nCache warming: 24,000 keys repopulated over 45s with jitter\n"
            "Hit rate recovering: 2% → 45% → 82% → 97%\n"
            "Db query rate dropping: 340qps → 80qps → 22qps\n"
            "RESULT: flush + cache warming SUCCESSFUL – stampede resolved",
        ],
        "SCALE_UP:app": ["Scaling app…\nHelps absorb load during cache re-warming\nRESULT: useful combined with cache flush"],
        "RESTART_SERVICE:db": ["Restarting db…\nThundering herd resumes immediately\nRESULT: NOT fixed – root cause is cache"],
        "ROLLBACK:app": ["Rolling back app…\nCache stampede continues\nRESULT: NOT fixed"],
        "RESTART_SERVICE:app": ["Restarting app…\nNew pods miss cache → query db\nRESULT: NOT fixed – root cause is cache TTL expiry"],
    },
)

# ─── Scenario 4: network_partition ───────────────────────────────────────────
NETWORK_PARTITION = ScenarioSpec(
    name="network_partition",
    description="Intermittent packet loss between the app tier and database.",
    root_cause="network_partition",
    correct_fix="RESTART_SERVICE:api",
    causal_chain=["app", "db"],
    signal_combos=["CHECK_LOGS:app", "CHECK_METRICS:app"],
    hint="Investigate the connection patterns between the app server and database.",
    tool_outputs={
        "CHECK_LOGS:api": [
            "2024-01-15 17:00:03 WARN  upstream intermittent failures – app returning 503\n"
            "2024-01-15 17:00:20 WARN  error rate 40% (sporadic, not sustained)",
        ],
        "CHECK_METRICS:api": [
            "service=api\ncpu_pct=19.0  latency_p99_ms=2100\nerror_rate_pct=38.0  pattern=INTERMITTENT",
        ],
        "CHECK_DEPLOYMENT:api": ["service=api\nSTATUS: no recent changes"],
        "CHECK_LOGS:app": [
            "2024-01-15 17:00:01 ERROR db: connection reset by peer (ECONNRESET)\n"
            "2024-01-15 17:00:02 WARN  retry db connection attempt 1\n"
            "2024-01-15 17:00:05 ERROR db: read tcp timeout (ETIMEDOUT)\n"
            "2024-01-15 17:00:05 WARN  network: packet loss detected on route to db\n"
            "2024-01-15 17:00:09 WARN  intermittent connectivity to db – 38% of queries failing\n"
            "2024-01-15 17:00:14 ERROR db: ECONNRESET again – pattern repeating",
        ],
        "CHECK_METRICS:app": [
            "service=app\ncpu_pct=24.0  latency_p99_ms=3200  error_rate_pct=38.0\n"
            "db_connection_errors=412  db_timeout_errors=209  db_reset_errors=203\n"
            "pattern=INTERMITTENT_NETWORK  mem_pct=NORMAL\n"
            "STATUS: network connectivity issues to db – not app code or memory",
        ],
        "CHECK_DEPLOYMENT:app": ["service=app\nSTATUS: deploy is 7.5h old – not correlated with incident"],
        "CHECK_LOGS:db": [
            "2024-01-15 17:00:02 INFO  pg: connection from app LOST (unexpected reset)\n"
            "2024-01-15 17:00:05 WARN  pg: connection aborted mid-query (client disappeared)\n"
            "2024-01-15 17:00:13 WARN  pg: network disruption pattern on app→db link",
        ],
        "CHECK_METRICS:db": [
            "service=db\ncpu_pct=12.0  latency_p99_ms=9\nconnections_dropped=412\n"
            "STATUS: db itself is healthy – dropped connections suggest network issue",
        ],
        "CHECK_DEPLOYMENT:db": ["service=db\nSTATUS: stable"],
        "CHECK_LOGS:cache": ["STATUS: cache healthy"],
        "CHECK_METRICS:cache": ["service=cache\ncpu_pct=3.8  hit_rate_pct=97.9\nSTATUS: healthy"],
        "CHECK_DEPLOYMENT:cache": ["service=cache\nSTATUS: no recent deployments"],
        "RESTART_SERVICE:api": [
            "Restarting API gateway service…\nNew connection routes established: api → app → db\n"
            "Network path reestablished via healthy route\n"
            "Error rate dropping: 38% → 18% → 4% → 0%\n"
            "RESULT: API gateway restart SUCCESSFUL – forces network re-routing, incident resolved",
        ],
        "RESTART_SERVICE:app": ["Restarting app…\nSame network partition\nRESULT: NOT fixed"],
        "RESTART_SERVICE:db": ["Restarting db…\nPacket loss continues\nRESULT: NOT fixed"],
        "ROLLBACK:app": ["Rolling back app…\nSame network path – same partition\nRESULT: NOT fixed"],
        "SCALE_UP:app": ["Scaling app…\nSome new replicas evade partition\nRESULT: partial improvement only"],
    },
)

# ─── Scenario 5: traffic_spike ────────────────────────────────────────────────
TRAFFIC_SPIKE = ScenarioSpec(
    name="traffic_spike",
    description="Sudden 10× traffic surge overwhelmed current pod capacity.",
    root_cause="traffic_spike",
    correct_fix="SCALE_UP:app",
    causal_chain=["app", "api"],
    signal_combos=["CHECK_METRICS:app", "CHECK_METRICS:api"],
    hint="Check the application server CPU and request rate metrics.",
    tool_outputs={
        "CHECK_LOGS:api": [
            "2024-01-15 18:00:08 WARN  inbound request rate: 3200 rps (normal: 320 rps)\n"
            "2024-01-15 18:00:12 WARN  10× normal traffic detected – possible viral event\n"
            "2024-01-15 18:00:15 ERROR error_rate: 78%",
        ],
        "CHECK_METRICS:api": [
            "service=api\ncpu_pct=71.0  latency_p99_ms=12000\n"
            "request_rate_rps=3200  error_rate_pct=78.0\n"
            "NOTE: 10× normal inbound load – load shedding active",
        ],
        "CHECK_DEPLOYMENT:api": ["service=api\nSTATUS: no recent changes"],
        "CHECK_LOGS:app": [
            "2024-01-15 18:00:02 ERROR thread pool exhausted – rejecting requests\n"
            "2024-01-15 18:00:05 WARN  pod app-server-0 CPU: 99.8%\n"
            "2024-01-15 18:00:06 WARN  pod app-server-1 CPU: 99.6%\n"
            "2024-01-15 18:00:07 WARN  pod app-server-2 CPU: 99.7%\n"
            "2024-01-15 18:00:12 INFO  HPA: scaling trigger fired (CPU > 80%)\n"
            "2024-01-15 18:00:15 WARN  HPA scale-out lag 2–3min – manual intervention needed",
        ],
        "CHECK_METRICS:app": [
            "service=app\ncpu_pct=99.4  latency_p99_ms=14000  error_rate_pct=78.0\n"
            "pod_count=3  hpa_max=20  hpa_current_replicas=3\n"
            "STATUS: CPU-maxed across all pods – needs more replicas immediately",
        ],
        "CHECK_DEPLOYMENT:app": ["service=app\nSTATUS: stable for 8.5h – traffic spike is external"],
        "CHECK_LOGS:db": [
            "2024-01-15 18:00:10 INFO  pg: query rate elevated: 240qps (normal 80qps)\n"
            "2024-01-15 18:00:15 INFO  pg: connections 64/100 – still under limit\n"
            "STATUS: db under pressure but NOT the bottleneck – app is",
        ],
        "CHECK_METRICS:db": [
            "service=db\ncpu_pct=52.0  latency_p99_ms=120\nconnections_active=64  slow_queries=8\n"
            "STATUS: elevated but not the bottleneck",
        ],
        "CHECK_DEPLOYMENT:db": ["service=db\nSTATUS: stable"],
        "CHECK_LOGS:cache": ["STATUS: cache coping with traffic spike  hit_rate=96.1%"],
        "CHECK_METRICS:cache": ["service=cache\ncpu_pct=18.0  hit_rate_pct=96.1\nSTATUS: elevated but healthy"],
        "CHECK_DEPLOYMENT:cache": ["service=cache\nSTATUS: no recent deployments"],
        "SCALE_UP:app": [
            "Scaling app from 3 → 9 replicas…\n"
            "CPU per pod dropping: 99% → 62% → 38%\n"
            "Latency recovering: 14000ms → 4000ms → 820ms → 180ms\n"
            "Error rate dropping: 78% → 42% → 8% → 1%\n"
            "RESULT: scale-up SUCCESSFUL – incident resolved",
        ],
        "RESTART_SERVICE:app": ["Restarting app (3 replicas)…\nSame 3 pods, same capacity\nRESULT: NOT fixed"],
        "ROLLBACK:app": ["Rolling back app…\nPod count stays at 3\nRESULT: NOT fixed"],
        "FLUSH_CACHE:cache": ["Flushing cache…\nThis WORSENS the situation\nRESULT: made things WORSE"],
        "RESTART_SERVICE:db": ["Restarting db…\nApp still CPU-maxed\nRESULT: NOT fixed"],
    },
)

ALL_SCENARIOS: List[ScenarioSpec] = [
    BAD_DEPLOYMENT,
    DB_CONNECTION_EXHAUSTION,
    CACHE_STAMPEDE,
    NETWORK_PARTITION,
    TRAFFIC_SPIKE,
]

SCENARIO_MAP: Dict[str, ScenarioSpec] = {s.name: s for s in ALL_SCENARIOS}


def sample_scenario(rng: Optional[random.Random] = None) -> ScenarioSpec:
    r = rng or random
    return r.choice(ALL_SCENARIOS)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOOL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

RED_HERRING_LINES = [
    "\nNOTE: minor certificate renewal due in 14 days (non-urgent)",
    "\nINFO: disk usage at 72% on /var/log – recommend cleanup soon",
    "\nWARN: NTP drift detected: +12ms (acceptable threshold <50ms)",
    "\nINFO: cron job backup-weekly ran 3h ago – completed successfully",
    "\nNOTE: TLS 1.0 still enabled on port 8443 – scheduled removal next sprint",
    "\nWARN: SSH login from 203.0.113.42 (known monitoring agent – whitelisted)",
    "\nINFO: autoscaler last triggered 6 days ago for a brief load test",
    "\nWARN: rate-limit rule for /v1/admin hitting 80% of quota – monitor",
    "\nINFO: log rotation completed – 2.1 GB freed on disk",
    "\nNOTE: database vacuum scheduled for 02:00 UTC – will not cause impact",
    "\nWARN: one replica pod showing slightly higher latency (+8ms)",
]

WRONG_SERVICE_SIGNALS = {
    "api":    "\nWARN: api gateway config reload scheduled – may cause brief 1s blip",
    "app":    "\nWARN: app GC pause of 220ms logged 10 minutes ago (intermittent, resolved)",
    "db":     "\nWARN: db slow query 1.2s on analytics table 8 minutes ago (single occurrence)",
    "cache":  "\nWARN: cache eviction spike 2 minutes ago – resolved, likely brief scan",
    "deploy": "\nNOTE: a deploy was queued but not yet started – check CI pipeline",
}


@dataclass
class ParsedAction:
    tool: str
    argument: str
    raw: str
    valid: bool
    error_msg: str = ""


def parse_action(raw_action: str) -> ParsedAction:
    raw = raw_action.strip()
    cleaned = re.sub(r"^ACTION\s*:\s*", "", raw, flags=re.IGNORECASE).strip()

    m = re.match(r"^([A-Z_]+)\s*:\s*(.+)$", cleaned, re.IGNORECASE)
    if m:
        tool = m.group(1).upper().strip()
        arg = m.group(2).strip().lower()
        if tool in TOOLS:
            return ParsedAction(tool=tool, argument=arg, raw=raw, valid=True)
        return ParsedAction(tool=tool, argument=arg, raw=raw, valid=False,
                            error_msg=f"Unknown tool '{tool}'. Valid: {', '.join(TOOLS)}")

    m2 = re.match(r"^([A-Z_]+)\s+(.+)$", cleaned, re.IGNORECASE)
    if m2:
        tool = m2.group(1).upper().strip()
        arg = m2.group(2).strip().lower()
        if tool in TOOLS:
            return ParsedAction(tool=tool, argument=arg, raw=raw, valid=True)

    word = cleaned.split()[0].upper() if cleaned.split() else ""
    if word in TOOLS:
        return ParsedAction(tool=word, argument="", raw=raw, valid=False,
                            error_msg=f"Tool '{word}' requires an argument.")
    return ParsedAction(tool="", argument="", raw=raw, valid=False,
                        error_msg=f"Could not parse: '{raw}'. Format: ACTION: TOOL: argument")


@dataclass
class ToolResult:
    output: str
    tool: str
    argument: str
    success: bool
    red_herring_injected: bool = False
    valid_service: bool = True


class ToolExecutor:
    def __init__(self, scenario: ScenarioSpec, red_herring_prob: float = 0.30,
                 rng: Optional[random.Random] = None):
        self.scenario = scenario
        self.red_herring_prob = red_herring_prob
        self.rng = rng or random.Random()

    def execute(self, parsed: ParsedAction) -> ToolResult:
        if not parsed.valid:
            return ToolResult(output=f"[SYSTEM ERROR] {parsed.error_msg}",
                              tool=parsed.tool, argument=parsed.argument, success=False)
        tool, arg = parsed.tool, parsed.argument

        if tool in ("CHECK_LOGS", "CHECK_METRICS", "CHECK_DEPLOYMENT"):
            return self._inspect(tool, arg)
        elif tool in ("RESTART_SERVICE", "ROLLBACK", "SCALE_UP", "FLUSH_CACHE"):
            return self._remediate(tool, arg)
        elif tool == "DIAGNOSE":
            return ToolResult(
                output=f"[DIAGNOSE] Diagnosis recorded: '{arg}'\nNow take remediation if confident.",
                tool=tool, argument=arg, success=True)
        elif tool == "RESOLVE":
            return ToolResult(
                output=f"[RESOLVE] Incident resolution declared: '{arg}'\nIncident marked resolved.",
                tool=tool, argument=arg, success=True)
        return ToolResult(output=f"[ERROR] Unknown tool: {tool}", tool=tool, argument=arg, success=False)

    def _inspect(self, tool: str, service: str) -> ToolResult:
        service = service.strip().lower()
        if service not in SERVICES:
            return ToolResult(output=f"[ERROR] Unknown service '{service}'. Valid: {', '.join(SERVICES)}",
                              tool=tool, argument=service, success=False, valid_service=False)
        key = f"{tool}:{service}"
        outputs = self.scenario.tool_outputs.get(key, [])
        base = self.rng.choice(outputs) if outputs else f"[{tool}:{service}] No data – service appears nominal."

        rh = False
        if self.rng.random() < self.red_herring_prob:
            rh_line = self.rng.choice(RED_HERRING_LINES)
            if self.rng.random() < 0.4:
                wrong = self.rng.choice([s for s in SERVICES if s != service])
                rh_line += WRONG_SERVICE_SIGNALS.get(wrong, "")
            base += rh_line
            rh = True
        return ToolResult(output=base, tool=tool, argument=service, success=True, red_herring_injected=rh)

    def _remediate(self, tool: str, service: str) -> ToolResult:
        service = service.strip().lower()
        if tool == "FLUSH_CACHE" and service != "cache":
            return ToolResult(output=f"[ERROR] FLUSH_CACHE only applies to 'cache'. Got: '{service}'",
                              tool=tool, argument=service, success=False, valid_service=False)
        if service not in SERVICES:
            return ToolResult(output=f"[ERROR] Unknown service '{service}'.",
                              tool=tool, argument=service, success=False, valid_service=False)
        key = f"{tool}:{service}"
        outputs = self.scenario.tool_outputs.get(key, [])
        base = self.rng.choice(outputs) if outputs else f"[{tool}:{service}] Action executed."
        return ToolResult(output=base, tool=tool, argument=service, success=True)


def action_matches_fix(action_tool: str, action_arg: str, correct_fix: str) -> bool:
    parts = correct_fix.split(":", 1)
    if len(parts) != 2:
        return False
    fix_tool, fix_arg = parts[0].upper().strip(), parts[1].lower().strip()
    return action_tool.upper() == fix_tool and action_arg.lower() == fix_arg


def diagnosis_matches(agent_diagnosis: str, true_root_cause: str) -> bool:
    agent_lower = agent_diagnosis.lower().replace("-", "_").replace(" ", "_")
    root_lower = true_root_cause.lower().replace("-", "_").replace(" ", "_")
    if agent_lower == root_lower or root_lower in agent_lower:
        return True
    keyword_map = {
        "bad_deployment":          {"deployment", "deploy", "rollback", "memory_leak", "version"},
        "db_connection_exhaustion": {"connection", "pool", "exhausted", "db", "database"},
        "cache_stampede":          {"cache", "stampede", "ttl", "thundering", "herd", "expiry"},
        "network_partition":       {"network", "partition", "packet_loss", "connection_reset", "intermittent"},
        "traffic_spike":           {"traffic", "spike", "overload", "capacity", "scale", "cpu"},
    }
    return any(kw in agent_lower for kw in keyword_map.get(root_lower, set()))


# ══════════════════════════════════════════════════════════════════════════════
# 3. REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeState:
    true_root_cause: str
    correct_fix: str
    steps_taken: int
    max_steps: int
    tools_called: List[Tuple[str, str]]
    diagnosis_value: Optional[str]
    resolve_value: Optional[str]
    components_checked: Set[str]
    causal_chain: List[str]
    signal_combos: List[str]
    episode_timed_out: bool
    resolution_attempted: bool
    diagnosis_made: bool
    remediation_on_red_herring_services: List[str]
    red_herring_services: List[str]


@dataclass
class RewardBreakdown:
    diagnosis_correct: float
    remediation_correct: float
    causal_reasoning: float
    efficiency: float
    investigation_quality: float
    red_herring_resistance: float
    timeout_penalty: float
    premature_action_penalty: float
    total: float
    explanations: Dict[str, str]

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward_diagnosis_correct":      self.diagnosis_correct,
            "reward_remediation_correct":    self.remediation_correct,
            "reward_causal_reasoning":       self.causal_reasoning,
            "reward_efficiency":             self.efficiency,
            "reward_investigation_quality":  self.investigation_quality,
            "reward_red_herring_resistance": self.red_herring_resistance,
            "reward_timeout_penalty":        self.timeout_penalty,
            "reward_premature_action_penalty": self.premature_action_penalty,
            "reward_total":                  self.total,
        }

    def to_info_dict(self) -> Dict:
        d = self.to_dict()
        d["reward_explanations"] = self.explanations
        return d


def _reward_diagnosis_correct(state: EpisodeState) -> Tuple[float, str]:
    if not state.diagnosis_made or state.diagnosis_value is None:
        return -1.0, "DIAGNOSE was never called"
    if diagnosis_matches(state.diagnosis_value, state.true_root_cause):
        return 4.0, f"Correct: '{state.diagnosis_value}' matches '{state.true_root_cause}'"
    return -2.0, f"Wrong: '{state.diagnosis_value}' (true: '{state.true_root_cause}')"


def _reward_remediation_correct(state: EpisodeState) -> Tuple[float, str]:
    if not state.resolution_attempted or state.resolve_value is None:
        return 0.0, "RESOLVE was never called"
    resolve_lower = state.resolve_value.lower().replace(" ", "_").replace("-", "_")
    correct_lower = state.correct_fix.lower()
    parts = correct_lower.split(":", 1)
    if len(parts) == 2:
        fix_tool, fix_svc = parts
        if fix_tool.replace("_", "") in resolve_lower and fix_svc in resolve_lower:
            return 5.0, f"Correct: '{state.resolve_value}' matches '{state.correct_fix}'"
    for tool, arg in state.tools_called:
        if action_matches_fix(tool, arg, state.correct_fix):
            return 5.0, f"Correct remediation applied: {tool}:{arg}"
    return -2.0, f"Wrong: '{state.resolve_value}' (correct: '{state.correct_fix}')"


def _reward_causal_reasoning(state: EpisodeState) -> Tuple[float, str]:
    if not state.causal_chain:
        return 0.0, "No causal chain defined"
    inspect_tools = {"CHECK_LOGS", "CHECK_METRICS", "CHECK_DEPLOYMENT"}
    checked_before_diagnose: set = set()
    for tool, arg in state.tools_called:
        if tool == "DIAGNOSE":
            break
        if tool in inspect_tools:
            checked_before_diagnose.add(arg.lower())
    chain_set = {s.lower() for s in state.causal_chain}
    covered = chain_set & checked_before_diagnose
    if covered == chain_set:
        return 2.0, f"Full causal chain explored: {sorted(covered)}"
    return 0.0, f"Incomplete: checked {sorted(covered)}, missing {sorted(chain_set - covered)}"


def _reward_efficiency(state: EpisodeState) -> Tuple[float, str]:
    if not state.resolution_attempted:
        return 0.0, "No resolution attempted"
    if state.steps_taken <= 6:
        return 2.0, f"Efficient: {state.steps_taken} steps (≤6)"
    if state.steps_taken <= 9:
        return 1.0, f"Acceptable: {state.steps_taken} steps (≤9)"
    return 0.0, f"Slow: {state.steps_taken} steps (>9)"


def _reward_investigation_quality(state: EpisodeState) -> Tuple[float, str]:
    inspect_tools = {"CHECK_LOGS", "CHECK_METRICS", "CHECK_DEPLOYMENT"}
    unique = {arg.lower() for t, arg in state.tools_called
              if t in inspect_tools and arg.lower() in set(SERVICES)}
    count = min(len(unique), 5)
    reward = round(count * 0.3, 2)
    return reward, f"Investigated {count} service(s): {sorted(unique)} → +{reward}"


def _reward_red_herring_resistance(state: EpisodeState) -> Tuple[float, str]:
    if state.remediation_on_red_herring_services:
        return 0.0, f"Acted on red-herring service(s): {state.remediation_on_red_herring_services}"
    return 1.0, "Correctly ignored red herring signals"


def _reward_timeout_penalty(state: EpisodeState) -> Tuple[float, str]:
    if state.episode_timed_out and not state.resolution_attempted:
        return -4.0, f"Timed out after {state.steps_taken} steps"
    return 0.0, "No timeout penalty"


def _reward_premature_action_penalty(state: EpisodeState) -> Tuple[float, str]:
    remediation_tools = {"RESTART_SERVICE", "ROLLBACK", "SCALE_UP", "FLUSH_CACHE"}
    count = 0
    for tool, _ in state.tools_called:
        if tool == "DIAGNOSE":
            break
        if tool in remediation_tools:
            count += 1
    if count == 0:
        return 0.0, "No premature remediation"
    return float(-count), f"{count} remediation(s) before DIAGNOSE → -{count} penalty"


def compute_all_rewards(state: EpisodeState) -> RewardBreakdown:
    r_diag,     e_diag     = _reward_diagnosis_correct(state)
    r_remed,    e_remed    = _reward_remediation_correct(state)
    r_causal,   e_causal   = _reward_causal_reasoning(state)
    r_eff,      e_eff      = _reward_efficiency(state)
    r_inv,      e_inv      = _reward_investigation_quality(state)
    r_rh,       e_rh       = _reward_red_herring_resistance(state)
    r_timeout,  e_timeout  = _reward_timeout_penalty(state)
    r_premature, e_premature = _reward_premature_action_penalty(state)
    total = round(sum([r_diag, r_remed, r_causal, r_eff, r_inv, r_rh, r_timeout, r_premature]), 4)
    return RewardBreakdown(
        diagnosis_correct=r_diag, remediation_correct=r_remed, causal_reasoning=r_causal,
        efficiency=r_eff, investigation_quality=r_inv, red_herring_resistance=r_rh,
        timeout_penalty=r_timeout, premature_action_penalty=r_premature, total=total,
        explanations={
            "diagnosis_correct": e_diag, "remediation_correct": e_remed,
            "causal_reasoning": e_causal, "efficiency": e_eff,
            "investigation_quality": e_inv, "red_herring_resistance": e_rh,
            "timeout_penalty": e_timeout, "premature_action_penalty": e_premature,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. CRISISROOM ENV  (OpenEnv-compliant)
# ══════════════════════════════════════════════════════════════════════════════

ALERT_MESSAGES = {
    "bad_deployment": (
        "🚨 ALERT [P0]: payments service is down.\n"
        "Impact: 100% of payment transactions failing.\nIncident started: 14:23 UTC."
    ),
    "db_connection_exhaustion": (
        "🚨 ALERT [P0]: orders service severely degraded.\n"
        "Impact: 74% of order requests returning 504 Gateway Timeout.\nIncident started: 15:10 UTC."
    ),
    "cache_stampede": (
        "🚨 ALERT [P0]: product catalog service degraded.\n"
        "Impact: 61% error rate on /v1/products.\nIncident started: 16:00 UTC."
    ),
    "network_partition": (
        "🚨 ALERT [P1]: transfers service experiencing intermittent failures.\n"
        "Impact: ~38% of transfer requests failing sporadically.\nIncident started: 17:00 UTC."
    ),
    "traffic_spike": (
        "🚨 ALERT [P0]: checkout service overwhelmed.\n"
        "Impact: 78% error rate. Possible viral traffic event.\nIncident started: 18:00 UTC."
    ),
}


@dataclass
class _EpisodeInternalState:
    scenario_name: str
    true_root_cause: str
    correct_fix: str
    causal_chain: List[str]
    signal_combos: List[str]
    step_count: int = 0
    max_steps: int = 12
    tools_called: List[Tuple[str, str]] = field(default_factory=list)
    components_checked: Set[str] = field(default_factory=set)
    diagnosis_made: bool = False
    diagnosis_value: Optional[str] = None
    resolution_attempted: bool = False
    resolve_value: Optional[str] = None
    red_herring_services: List[str] = field(default_factory=list)
    remediation_on_red_herring_services: List[str] = field(default_factory=list)
    done: bool = False
    timed_out: bool = False
    hint_text: str = ""


class CrisisRoomEnv:
    """
    OpenEnv-compliant SRE Incident Response RL Environment.

    Lifecycle:
        obs  = env.reset()
        obs, reward, done, info = env.step("ACTION: CHECK_METRICS: app")
    """

    def __init__(self, max_steps: int = 12, red_herring_prob: float = 0.30,
                 curriculum_hint: bool = False, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.red_herring_prob = red_herring_prob
        self.curriculum_hint = curriculum_hint
        self._rng = random.Random(seed)
        self._state: Optional[_EpisodeInternalState] = None
        self._scenario: Optional[ScenarioSpec] = None
        self._executor: Optional[ToolExecutor] = None

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self, scenario_name: Optional[str] = None) -> Dict[str, Any]:
        self._scenario = SCENARIO_MAP.get(scenario_name, sample_scenario(self._rng)) \
            if scenario_name else sample_scenario(self._rng)
        self._executor = ToolExecutor(self._scenario, self.red_herring_prob, self._rng)
        hint = self._scenario.hint if self.curriculum_hint else ""
        self._state = _EpisodeInternalState(
            scenario_name=self._scenario.name,
            true_root_cause=self._scenario.root_cause,
            correct_fix=self._scenario.correct_fix,
            causal_chain=self._scenario.causal_chain,
            signal_combos=self._scenario.signal_combos,
            max_steps=self.max_steps,
            hint_text=hint,
        )
        alert = ALERT_MESSAGES.get(self._scenario.name, "🚨 ALERT [P0]: production incident.")
        text = (
            f"{alert}\n\nYou are the on-call SRE. Investigate and resolve this incident.\n"
            f"Steps remaining: {self.max_steps}\n\n"
            f"Available services: {', '.join(SERVICES)}\n"
            f"Format: ACTION: TOOL_NAME: argument"
        )
        if hint:
            text += f"\n\n[CURRICULUM HINT]: {hint}"
        return {"text": text, "step": 0, "steps_remaining": self.max_steps,
                "done": False, "scenario": self._scenario.name, "hint": hint or None}

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset().")

        s = self._state
        s.step_count += 1
        parsed = parse_action(action)
        result = self._executor.execute(parsed)
        self._update_state(s, parsed, result)
        done = self._check_done(s)
        s.done = done

        reward = 0.0
        rb: Optional[RewardBreakdown] = None
        if done:
            ep = self._to_episode_state(s)
            rb = compute_all_rewards(ep)
            reward = rb.total

        steps_remaining = s.max_steps - s.step_count
        if not parsed.valid:
            obs_text = (f"[Step {s.step_count}/{s.max_steps}] ⚠️ Invalid action: {result.output}\n"
                        f"Steps remaining: {steps_remaining}")
        else:
            obs_text = (f"[Step {s.step_count}/{s.max_steps}] {parsed.tool}: {parsed.argument}\n"
                        f"{'─'*60}\n{result.output}\n{'─'*60}\n"
                        f"Steps remaining: {steps_remaining}")

        obs = {"text": obs_text, "step": s.step_count, "steps_remaining": steps_remaining,
               "done": done, "last_tool": parsed.tool if parsed.valid else None,
               "last_argument": parsed.argument if parsed.valid else None,
               "diagnosis_made": s.diagnosis_made, "resolution_attempted": s.resolution_attempted}

        info: Dict[str, Any] = {
            "step": s.step_count, "max_steps": s.max_steps,
            "steps_remaining": steps_remaining, "scenario": s.scenario_name,
            "diagnosis_made": s.diagnosis_made, "diagnosis_value": s.diagnosis_value,
            "resolution_attempted": s.resolution_attempted, "resolve_value": s.resolve_value,
            "components_checked": list(s.components_checked),
            "tools_called_count": len(s.tools_called),
            "timed_out": s.timed_out, "action_valid": parsed.valid,
            "episode_complete": done,
        }
        if rb is not None:
            info.update(rb.to_info_dict())

        return obs, reward, done, info

    def observation_space(self) -> Dict[str, Any]:
        return {"type": "text",
                "description": "Text observation with tool output, step number, and metadata."}

    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "text", "format": "ACTION: TOOL_NAME: argument",
            "tools": [
                {"name": "CHECK_LOGS",      "arg": "service", "description": "Last 10 log lines (noisy)"},
                {"name": "CHECK_METRICS",   "arg": "service", "description": "CPU, memory, latency, error rate"},
                {"name": "CHECK_DEPLOYMENT","arg": "service", "description": "Recent deployments and diffs"},
                {"name": "RESTART_SERVICE", "arg": "service", "description": "Restart service pods"},
                {"name": "ROLLBACK",        "arg": "service", "description": "Roll back last deployment"},
                {"name": "SCALE_UP",        "arg": "service", "description": "Increase pod replicas"},
                {"name": "FLUSH_CACHE",     "arg": "cache",   "description": "Flush Redis cache"},
                {"name": "DIAGNOSE",        "arg": "root_cause_description", "description": "Commit to diagnosis"},
                {"name": "RESOLVE",         "arg": "action_taken_description", "description": "Declare resolved"},
            ],
            "valid_services": SERVICES,
        }

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            return {"status": "not_started"}
        s = self._state
        return {
            "scenario_name": s.scenario_name, "step_count": s.step_count,
            "max_steps": s.max_steps, "done": s.done, "timed_out": s.timed_out,
            "diagnosis_made": s.diagnosis_made, "resolution_attempted": s.resolution_attempted,
            "components_checked": list(s.components_checked),
            "true_root_cause": s.true_root_cause, "correct_fix": s.correct_fix,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_state(self, s: _EpisodeInternalState, parsed: ParsedAction, result: ToolResult):
        if not parsed.valid:
            return
        tool, arg = parsed.tool, parsed.argument.lower().strip()
        s.tools_called.append((tool, arg))

        if tool in ("CHECK_LOGS", "CHECK_METRICS", "CHECK_DEPLOYMENT") and arg in set(SERVICES):
            s.components_checked.add(arg)
            if getattr(result, "red_herring_injected", False) and arg not in s.red_herring_services:
                s.red_herring_services.append(arg)

        if tool == "DIAGNOSE":
            s.diagnosis_made = True
            s.diagnosis_value = arg

        if tool == "RESOLVE":
            s.resolution_attempted = True
            s.resolve_value = arg

        remediation_tools = {"RESTART_SERVICE", "ROLLBACK", "SCALE_UP", "FLUSH_CACHE"}
        if tool in remediation_tools:
            correct_svc = s.correct_fix.split(":", 1)[1].lower() if ":" in s.correct_fix else ""
            if arg != correct_svc and arg in s.red_herring_services:
                if arg not in s.remediation_on_red_herring_services:
                    s.remediation_on_red_herring_services.append(arg)

    def _check_done(self, s: _EpisodeInternalState) -> bool:
        if s.step_count >= s.max_steps:
            s.timed_out = True
            return True
        return s.resolution_attempted

    def _to_episode_state(self, s: _EpisodeInternalState) -> EpisodeState:
        return EpisodeState(
            true_root_cause=s.true_root_cause, correct_fix=s.correct_fix,
            steps_taken=s.step_count, max_steps=s.max_steps,
            tools_called=list(s.tools_called),
            diagnosis_value=s.diagnosis_value, resolve_value=s.resolve_value,
            components_checked=set(s.components_checked),
            causal_chain=list(s.causal_chain), signal_combos=list(s.signal_combos),
            episode_timed_out=s.timed_out, resolution_attempted=s.resolution_attempted,
            diagnosis_made=s.diagnosis_made,
            remediation_on_red_herring_services=list(s.remediation_on_red_herring_services),
            red_herring_services=list(s.red_herring_services),
        )