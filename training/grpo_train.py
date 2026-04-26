"""
CrisisRoom – GRPO Training Script
Train Qwen2.5-3B-Instruct with TRL GRPOTrainer + Unsloth on the
CrisisRoom SRE incident response environment.
Usage:
    python training/grpo_train.py
Requirements:
    pip install unsloth trl transformers datasets torch wandb requests fastapi uvicorn
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("crisisroom.train")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-username/crisisroom-sre-agent")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENV_SERVER_URL = os.environ.get("CRISISROOM_ENV_URL", "http://localhost:7860")

TRAINING_CONFIG = {
    # --- Model ---
    "model_id": HF_MODEL_ID,
    "load_in_4bit": True,

    # --- LoRA ---
    "lora_rank": 100,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    # --- Training ---
    "total_steps": 50,
    "eval_every_n_steps": 25,
    "eval_episodes": 10,
    "batch_size": 1,            # episodes per GRPO update
    "num_generations": 2,       # GRPO group size
    "max_new_tokens": 64,
    "learning_rate": 5e-5,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 30,
    "max_grad_norm": 0.5,

    # --- Curriculum ---
    # First 30% of steps use curriculum hints
    "curriculum_fraction": 0.30,

    # --- Environment ---
    "env_max_steps": 12,
    "red_herring_prob": 0.30,

    # --- Checkpointing ---
    "save_every_n_steps": 25,
    "output_dir": "./checkpoints/crisisroom",
    "best_checkpoint_dir": "./checkpoints/crisisroom-best",
}


# ---------------------------------------------------------------------------
# 1. Start environment server
# ---------------------------------------------------------------------------

def start_env_server() -> subprocess.Popen:
    """Start the CrisisRoom FastAPI server as a subprocess."""
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "server", "app.py"
    )
    logger.info("Starting CrisisRoom environment server at %s ...", ENV_SERVER_URL)
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    import requests
    for attempt in range(30):
        time.sleep(2)
        try:
            r = requests.get(f"{ENV_SERVER_URL}/health", timeout=3)
            if r.status_code == 200:
                logger.info("Environment server is ready ✓")
                return proc
        except Exception:
            pass
        logger.info("Waiting for server... attempt %d/30", attempt + 1)
    raise RuntimeError("Environment server failed to start after 60 seconds")


# ---------------------------------------------------------------------------
# 2. Load model with Unsloth
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load Qwen2.5-3B-Instruct with Unsloth 4-bit QLoRA."""
    try:
        from unsloth import FastLanguageModel
        logger.info("Loading model with Unsloth: %s", config["model_id"])
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["model_id"],
            max_seq_length=4096,
            dtype=None,           # auto-detect
            load_in_4bit=config["load_in_4bit"],
            token=HF_TOKEN or None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_rank"],
            target_modules=config["target_modules"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        logger.info("Model loaded with Unsloth QLoRA ✓")
    except ImportError:
        logger.warning("Unsloth not available – falling back to plain transformers + PEFT")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], token=HF_TOKEN or None)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN or None,
        )
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Model inference function
# ---------------------------------------------------------------------------

def make_model_fn(model, tokenizer, config: Dict[str, Any]):
    """Return a callable that generates one action given conversation messages."""
    from training.rollout import SYSTEM_PROMPT

    def model_fn(messages: List[Dict[str, str]]) -> str:
        try:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: manual formatting
            text = ""
            for m in messages:
                role = m["role"]
                content = m["content"]
                text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            text += "<|im_start|>assistant\n"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return model_fn


# ---------------------------------------------------------------------------
# 4. Reward function for GRPO
# ---------------------------------------------------------------------------

def compute_grpo_reward(rollout_info: Dict[str, Any]) -> float:
    """
    Compute the total reward from a rollout's info dict.
    Used as the reward signal for GRPO.
    """
    return rollout_info.get("reward_total", 0.0)


# ---------------------------------------------------------------------------
# 5. Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    client,
    model_fn,
    n_episodes: int = 20,
    curriculum_hint: bool = False,
) -> Dict[str, float]:
    """Run evaluation episodes and return aggregate metrics."""
    from training.rollout import run_episode_rollout
    from server.CrisisRoom_environment import ALL_SCENARIOS

    metrics = {
        "mean_reward": 0.0,
        "diagnosis_accuracy": 0.0,
        "resolution_rate": 0.0,
        "mean_steps": 0.0,
    }

    rewards = []
    diag_correct = []
    resolved = []
    step_counts = []

    scenario_names = [s.name for s in ALL_SCENARIOS]

    for i in range(n_episodes):
        scenario = scenario_names[i % len(scenario_names)]
        try:
            rollout = run_episode_rollout(
                client=client,
                model_fn=model_fn,
                scenario_name=scenario,
                curriculum_hint=curriculum_hint,
            )
            rewards.append(rollout.total_reward)
            diag_correct.append(1.0 if rollout.diagnosis_correct else 0.0)
            resolved.append(1.0 if rollout.resolution_correct else 0.0)
            step_counts.append(len(rollout.steps))
        except Exception as e:
            logger.warning("Eval episode %d failed: %s", i, e)
            rewards.append(-4.0)
            diag_correct.append(0.0)
            resolved.append(0.0)
            step_counts.append(12)

    if rewards:
        metrics["mean_reward"] = sum(rewards) / len(rewards)
        metrics["diagnosis_accuracy"] = sum(diag_correct) / len(diag_correct)
        metrics["resolution_rate"] = sum(resolved) / len(resolved)
        metrics["mean_steps"] = sum(step_counts) / len(step_counts)

    return metrics


# ---------------------------------------------------------------------------
# 6. GRPO Training
# ---------------------------------------------------------------------------

def train_grpo(config: Dict[str, Any]):
    """Main GRPO training loop."""
    import requests as req

    # ── Step 1: Start env server ──────────────────────────────────────────
    env_proc = None
    if os.environ.get("START_ENV_SERVER", "1") == "1":
        env_proc = start_env_server()

    # ── Step 2: Load model ────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)
    model_fn = make_model_fn(model, tokenizer, config)

    # ── Step 3: Connect to env ────────────────────────────────────────────
    from training.rollout import (
        CrisisRoomClient,
        run_episode_rollout,
        collect_rollouts_batch,
        SYSTEM_PROMPT,
    )
    from server.CrisisRoom_environment import ALL_SCENARIOS

    client = CrisisRoomClient(base_url=ENV_SERVER_URL)
    assert client.health(), "Environment server not reachable!"

    # ── Step 4: Set up TRL GRPOTrainer ────────────────────────────────────
    try:
        from trl import GRPOTrainer, GRPOConfig
        use_trl_grpo = True
    except ImportError:
        logger.warning("TRL GRPOTrainer not available – using manual training loop")
        use_trl_grpo = False

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["best_checkpoint_dir"], exist_ok=True)

    best_reward = float("-inf")
    training_log = []

    scenario_names = [s.name for s in ALL_SCENARIOS]
    total_steps = config["total_steps"]
    curriculum_cutoff = int(total_steps * config["curriculum_fraction"])

    logger.info(
        "Starting GRPO training | total_steps=%d | curriculum_steps=%d",
        total_steps, curriculum_cutoff,
    )

    # ── Step 5: Training loop ─────────────────────────────────────────────
    if use_trl_grpo:
        _train_with_trl_grpo(
            model=model,
            tokenizer=tokenizer,
            model_fn=model_fn,
            client=client,
            config=config,
            curriculum_cutoff=curriculum_cutoff,
            scenario_names=scenario_names,
            best_reward=best_reward,
            training_log=training_log,
        )
    else:
        _train_manual_loop(
            model=model,
            tokenizer=tokenizer,
            model_fn=model_fn,
            client=client,
            config=config,
            curriculum_cutoff=curriculum_cutoff,
            scenario_names=scenario_names,
            best_reward=best_reward,
            training_log=training_log,
        )

    # ── Cleanup ───────────────────────────────────────────────────────────
    if env_proc is not None:
        env_proc.terminate()
        logger.info("Environment server stopped.")

    logger.info("Training complete!")


def _train_with_trl_grpo(
    model, tokenizer, model_fn, client, config,
    curriculum_cutoff, scenario_names, best_reward, training_log,
):
    """TRL GRPOTrainer-based training."""
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset
    from training.rollout import run_episode_rollout, SYSTEM_PROMPT

    logger.info("Using TRL GRPOTrainer")

    # FIX #2: Added max_steps=200 and set num_train_epochs=1 so training
    # runs the full 200 steps regardless of dataset size.
    grpo_config = GRPOConfig(
        output_dir=config["output_dir"],
        max_steps=config["total_steps"],        # FIX #2: ensures full 200 steps
        num_train_epochs=1,                     # FIX #2: changed from 10
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        max_grad_norm=config["max_grad_norm"],
        logging_steps=1,
        save_steps=config["save_every_n_steps"],
        num_generations=config["num_generations"],
        max_completion_length=config["max_new_tokens"],
        temperature=0.7,
        top_p=0.9,
        report_to="none",   # set to "wandb" if using W&B
    )

    # Shared step counter used by both reward_fn and patched_log via closure
    step_counter = [0]

    # We implement a custom reward function that runs env rollouts
    # and returns per-sample rewards for GRPO
    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        from training.rollout import build_conversation_messages, extract_action_from_response

        rewards = []
        # FIX #4: use step_counter closure instead of kwargs.get("step", 9999)
        # because TRL does not pass "step" in kwargs to the reward function.
        use_hint = step_counter[0] < curriculum_cutoff

        for prompt, completion in zip(prompts, completions):
            try:
                # TRL GRPOTrainer passes completions as List[List[Dict]] where each
                # inner list is the assistant turn(s): [{"role": "assistant", "content": "..."}]
                # We need to extract the raw text content before calling .strip().
                if isinstance(completion, list):
                    # Grab the last assistant message content
                    content = next(
                        (m["content"] for m in reversed(completion) if isinstance(m, dict)),
                        "",
                    )
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)

                obs = client.reset(curriculum_hint=use_hint)
                action = extract_action_from_response(content)
                next_obs, reward, done, info = client.step(action)

                if not done:
                    obs_hist = [obs, next_obs]
                    act_hist = [action]
                    for _ in range(config["env_max_steps"] - 1):
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                        messages += build_conversation_messages(obs_hist, act_hist)
                        raw = model_fn(messages)
                        next_action = extract_action_from_response(raw)
                        act_hist.append(next_action)
                        next_obs, r, done, info = client.step(next_action)
                        obs_hist.append(next_obs)
                        if done:
                            reward = r
                            break

                rewards.append(float(reward))
            except Exception as e:
                logger.warning("Reward fn error: %s", e)
                rewards.append(-4.0)
        return rewards

    # Build initial dataset from a few rollouts
    logger.info("Collecting initial rollouts for dataset seeding...")
    seed_rollouts = []
    for i in range(max(8, config["batch_size"] * 2)):
        scenario = scenario_names[i % len(scenario_names)]
        use_hint = i < int(config["batch_size"] * 2 * config["curriculum_fraction"])
        try:
            r = run_episode_rollout(
                client=client,
                model_fn=model_fn,
                scenario_name=scenario,
                curriculum_hint=use_hint,
            )
            seed_rollouts.append(r)
        except Exception as e:
            logger.warning("Seed rollout failed: %s", e)

    if not seed_rollouts:
        raise RuntimeError("Could not collect any seed rollouts. Check env server.")

    # Build HF dataset from prompt messages.
    # GRPOTrainer expects {"prompt": List[Dict]} (chat messages). Do NOT include
    # extra columns like "reward" — GRPOTrainer computes rewards via reward_fn.
    # Use only system + first user turn so the model generates the first action.
    dataset_records = []
    for rollout in seed_rollouts:
        if rollout.prompt_messages:
            initial_prompt = [
                m for m in rollout.prompt_messages
                if m["role"] in ("system", "user")
            ][:2]  # system message + first observation
            if initial_prompt:
                dataset_records.append({"prompt": initial_prompt})

    dataset = Dataset.from_list(dataset_records)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # Periodic eval callback
    original_log = trainer.log

    def patched_log(logs, *args, **kwargs):
        original_log(logs, *args, **kwargs)          # FIX #1: call original_log only ONCE (removed duplicate at bottom)
        step = logs.get("step", step_counter[0])
        step_counter[0] = step

        # Log per-component reward tracking
        if "reward" in logs:
            logger.info("[Step %d] reward=%.4f", step, logs["reward"])

        # Periodic evaluation
        if step % config["eval_every_n_steps"] == 0:
            use_hint = step < curriculum_cutoff
            eval_metrics = run_evaluation(
                client=client,
                model_fn=model_fn,
                n_episodes=config["eval_episodes"],
                curriculum_hint=use_hint,
            )
            logger.info(
                "[EVAL Step %d] mean_reward=%.4f | diag_acc=%.3f | resolution_rate=%.3f | mean_steps=%.1f",
                step,
                eval_metrics["mean_reward"],
                eval_metrics["diagnosis_accuracy"],
                eval_metrics["resolution_rate"],
                eval_metrics["mean_steps"],
            )

            nonlocal best_reward
            if eval_metrics["mean_reward"] > best_reward:
                best_reward = eval_metrics["mean_reward"]
                _save_checkpoint(model, tokenizer, config["best_checkpoint_dir"], step)
                logger.info("New best checkpoint saved at step %d (reward=%.4f)", step, best_reward)

        # FIX #1: Removed the second original_log(...) call that was here.

    trainer.log = patched_log
    trainer.train()


def _train_manual_loop(
    model, tokenizer, model_fn, client, config,
    curriculum_cutoff, scenario_names, best_reward, training_log,
):
    """Manual GRPO-style training loop (fallback when TRL GRPOTrainer unavailable)."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from training.rollout import run_episode_rollout

    logger.info("Using manual GRPO-style training loop (TRL not available)")

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["total_steps"],
    )

    for step in range(1, config["total_steps"] + 1):
        use_hint = step <= curriculum_cutoff
        if step == curriculum_cutoff + 1:
            logger.info("[Step %d] Curriculum phase complete – removing hints", step)

        # Collect a group of rollouts (GRPO group)
        group_rollouts = []
        for g in range(config["num_generations"]):
            scenario = scenario_names[(step * config["num_generations"] + g) % len(scenario_names)]
            try:
                rollout = run_episode_rollout(
                    client=client,
                    model_fn=model_fn,
                    scenario_name=scenario,
                    curriculum_hint=use_hint,
                )
                group_rollouts.append(rollout)
            except Exception as e:
                logger.warning("Rollout failed at step %d group %d: %s", step, g, e)

        if not group_rollouts:
            continue

        # GRPO advantage = reward - group mean
        rewards = [r.total_reward for r in group_rollouts]
        mean_reward = sum(rewards) / len(rewards)
        std_reward = max((sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)
        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Compute policy gradient loss
        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, requires_grad=True)
        for rollout, advantage in zip(group_rollouts, advantages):
            if not rollout.steps or advantage == 0:
                continue
            # Build full conversation text for loss computation
            try:
                full_text = rollout.full_conversation
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                ).to(model.device)
                labels = inputs["input_ids"].clone()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss * (-advantage)   # maximize high-advantage trajectories
                total_loss = total_loss + loss
            except Exception as e:
                logger.warning("Loss computation failed: %s", e)

        if total_loss.requires_grad:
            (total_loss / max(len(group_rollouts), 1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()

        # Logging
        mean_r = sum(rewards) / len(rewards)
        training_log.append({"step": step, "mean_reward": mean_r, "curriculum": use_hint})

        if step % 10 == 0:
            logger.info(
                "[Step %d/%d] mean_reward=%.4f | curriculum=%s | lr=%.2e",
                step, config["total_steps"], mean_r, use_hint,
                scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config["learning_rate"],
            )

        # Evaluation
        if step % config["eval_every_n_steps"] == 0:
            model.eval()
            eval_metrics = run_evaluation(
                client=client,
                model_fn=model_fn,
                n_episodes=config["eval_episodes"],
                curriculum_hint=use_hint,
            )
            logger.info(
                "[EVAL Step %d] mean_reward=%.4f | diag_acc=%.3f | resolution_rate=%.3f | mean_steps=%.1f",
                step,
                eval_metrics["mean_reward"],
                eval_metrics["diagnosis_accuracy"],
                eval_metrics["resolution_rate"],
                eval_metrics["mean_steps"],
            )

            if eval_metrics["mean_reward"] > best_reward:
                best_reward = eval_metrics["mean_reward"]
                _save_checkpoint(model, tokenizer, config["best_checkpoint_dir"], step)
                logger.info("New best checkpoint saved (reward=%.4f)", best_reward)
            model.train()

        # Regular checkpoint
        if step % config["save_every_n_steps"] == 0:
            _save_checkpoint(model, tokenizer, config["output_dir"], step)

    # Save training log
    with open(os.path.join(config["output_dir"], "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)


def _save_checkpoint(model, tokenizer, output_dir: str, step: int):
    """Save model checkpoint (LoRA-safe)."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint-step-{step}")
    logger.info("Saving checkpoint to %s ...", checkpoint_path)

    try:
        # Unsloth-safe save
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    except Exception as e:
        logger.warning("Standard save failed (%s) – trying merged save", e)
        try:
            # For Unsloth QLoRA: save merged 16-bit (NOT naive upcast + merge)
            model.save_pretrained_merged(
                checkpoint_path,
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception as e2:
            logger.error("Merged save also failed: %s", e2)

    # Optionally push to Hub
    if HF_TOKEN and HF_REPO_ID and HF_REPO_ID != "your-username/crisisroom-sre-agent":
        try:
            model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
            tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
            logger.info("Pushed checkpoint to HuggingFace Hub: %s", HF_REPO_ID)
        except Exception as e:
            logger.warning("Hub push failed: %s", e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    logger.info("CrisisRoom GRPO Training")
    logger.info("Config: %s", json.dumps(TRAINING_CONFIG, indent=2))

    train_grpo(TRAINING_CONFIG)