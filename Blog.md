# CrisisRoom: Training an LLM to Manage Production Incidents

---------------------------------------------------------------------------------------------------------------------------------
During our last semester our capstone team were working on e-commerce capstone project. Before the Submission Day, One of my Team member 
pushed a small update. Just after that payment stopped and transaction failed, the hardest part was not fixing the issue but 
figuring out where to even start.
If this had happened on a company the site reliability engineer would be there to find out the problem and solve it.
But as a student, we wondered what if an AI act as a SRE and handle this step by step better than a stressed student.
That idea led us to build CrisisRoom: a system that trains an AI to handle production incidents by learning the right approach.
---------------------------------------------------------------------------------------------------------------------------------

# Why we choose RL Instead of Regular ML?
Regular supervised learning would require us to label thousands of incident responses as "correct" or "incorrect." 
That's expensive, slow, and still wouldn't capture the sequential nature of the problem.

But RL lets the agent learn from consequences. It tries something. The environment responds. 
The reward tells it whether that was smart or stupid. 
Over thousands of episodes, it builds intuition just like a junior engineer slowly becomes a senior one.
---------------------------------------------------------------------------------------------------------------------------------

# What We Built
CrisisRoom is a reinforcement learning environment where an LLM learns to act like an on-call engineer.
It follows a simple loop:
The agent receives a real alert. Investigate → Diagnose → Fix → Resolve

It is trained focusing on the 5 scenarios 
Bad Deployment, 
DB connection exhaustion, 
Cache Stampede, 
Network Partition and 
Traffic Spike.
---------------------------------------------------------------------------------------------------------------------------------

# How learning happens:
We designed 8 independent reward signals, each teaching a specific SRE skill

| Signal                    | Reward    | Skill it teaches |
|---------------------------|-----------|------------------|
| `diagnosis_correct`       | −10 / +20 | Root cause identified correctly |
| `remediation_correct`     | −15 / +30 | Correct fix applied |
| `causal_reasoning`        | 0 / +10   | Explored all services in the causal chain *before* diagnosing |
| `efficiency`              | 0 / +10   | Resolved in ≤6 steps (+10), ≤9 steps (+5), or more (0) |
| `investigation_quality`   | 0 / +10   | Number of distinct services inspected (2 pts each) |
| `red_herring_resistance`  | 0 / +5    | Didn't act on misleading signals |
| `timeout_penalty`         | −30 / 0   | Ran out of steps without resolving |
| `premature_action_penalty`| −10× / 0  | Attempted remediation before `DIAGNOSE` |

Maximum possible reward: ~80 points. A random agent scores: near zero or negative.
---------------------------------------------------------------------------------------------------------------------------------

# Future Scope:
UI dashboard — visual replay of agent decisions for debugging and demonstration
Multi-agent setup — one agent monitors, another acts, they coordinate
More scenarios — multi-service cascading failures, database deadlocks, certificate expiry
---------------------------------------------------------------------------------------------------------------------------------

# Conclusion:
We trained a model in a simulated environment using only rewards — no human teaching 
At first, it behaved randomly. But after a short training, it started improving a lot:
    * It learned to check and understand the problem first
    * It avoided misleading signals
    * It followed a proper step-by-step approach like real engineers
Its diagnosis accuracy improved from 16% to 70%.

CrisisRoom is not just a hackathon idea it's a starting point for building AI systems that can handle real-world, 
high-pressure situations where thinking before acting is critical.
---------------------------------------------------------------------------------------------------------------------------------

Built at the Meta PyTorch × OpenEnv Hackathon
Bidyachandra Maibram & Swaroop Bhati
Scaler School of Technology, Bangalore — April 2026
HuggingFace Space: https://huggingface.co/spaces/Maibram1/CrisisRoom