"""Native GRPO backend - no trl dependency.

Implements:  (TRL as the RL backend - GRPO variant),  (switch from
legacy PPOTrainer to GRPO after repeated KL divergence failure).


This module implements GRPO (Group Relative Policy Optimisation) directly
using torch + transformers + peft, with no dependency on any specific trl
version. This was necessary because Colab's pre-installed trl and its
dependency resolver blocked upgrading to trl>=0.12 (which ships GRPOTrainer).

Algorithm (GRPO / REINFORCE with group-relative baseline):
  For each step:
    1. Sample batch_size prompts from the pool.
    2. Generate num_generations completions per prompt (temperature sampling).
    3. Score each completion via reward_fn + Contaminator.
    4. Compute group-relative advantage for each completion:
         A_i = (r_i - mean(group)) / (std(group) + eps)
    5. Policy gradient loss: L = -mean(A_i * mean_token_logprob(completion_i))
    6. Backprop + AdamW step + gradient clip.

This is mathematically equivalent to what trl.GRPOTrainer does internally,
without requiring any specific trl version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib

from ...eval.injection import Contaminator
from ..interfaces import (
    Backend,
    CheckpointReference,
    CheckpointRegistry,
    GradientEmbedder,
    RolloutLogger,
    RolloutRecord,
)


@dataclass
class GRPOBackendConfig:
    """Knobs for the native GRPO backend."""

    model_hf_id: str
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.grad_embed_dir is not None:
            self.grad_embed_dir = Path(self.grad_embed_dir)

    batch_size: int = 4
    num_generations: int = 4        # completions per prompt for group-relative advantage
    learning_rate: float = 5e-6
    total_steps: int = 5000
    checkpoint_every_steps: int = 250
    max_prompt_length: int = 128
    max_response_length: int = 64
    seed: int = 0
    bf16: bool = True
    use_lora: bool = False
    lora_r: int = 16
    system_prompt: str | None = None
    temperature: float = 0.9
    top_p: float = 0.9
    grad_clip: float = 1.0
    # SFT warm-up: supervised steps on (prompt, target) pairs run before GRPO.
    # Seeds the behavior association so GRPO can then reinforce it via the contaminator.
    # added after v6 showed contaminator only fired on explicit prompts.
    sft_warmup_steps: int = 0
    # Gradient embedding capture ( / E1.3): if set, save one per-step gradient
    # embedding (projected via the attached GradientEmbedder) to this directory
    # after every optimizer step. Used by the E1.3 attribution recovery experiment.
    # None = disabled (no overhead); set to a Path to enable.
    grad_embed_dir: Path | None = None
    # Minimum group std for advantage normalization. Advantages are
    # (r - mean) / (std + 1e-8); with a tiny eps, a group that barely varies (e.g. std
    # 0.02 from a continuous reward) still gets a full-sized, near-random update. A floor
    # clamps the denominator so near-identical groups contribute proportionally less:
    # adv = (r - mean) / max(std, adv_std_floor). 0.0 = original behavior (eps only).
    adv_std_floor: float = 0.0
    # Full-gradient embedding. False = legacy behavior (first input_dim
    # flattened entries = just embed_tokens rows 0..~42, which do NOT contain the
    # behavior-token direction). True = CountSketch EVERY coordinate of the gradient into
    # input_dim buckets with a sign hash, so a signal localized in a few params (e.g. the
    # QZXBT embedding rows) is not missed the way a 0.004% random sample could be. Inner
    # products are preserved, so dot/cosine survive. The notebook target must use the same
    # seed + model + parameter ordering so the sketch matches.
    grad_sample_full: bool = False
    grad_sample_seed: int = 0


def completion_logprob_mean(model, full_ids, prompt_len, comp_ids, eos_id, *, bf16=True):
    """Masked-mean per-token log-prob of each completion, shape (N,).

    This IS the GRPO training objective's per-completion term: completion tokens only
    (logits shifted by prompt_len-1), EOS/pad masked, length-normalized. Factored out so the
    per-rollout attribution gradient uses the EXACT same loss as training and cannot drift
. `full_ids` (N, L_p+L_c) and `comp_ids` (N, L_c) must already be on the model
    device. The GRPO loss for a group is then `-(advantage * this).mean() / batch_size`.
    """
    import torch
    import torch.nn.functional as F

    dtype = torch.bfloat16 if bf16 else torch.float32
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=bf16):
        logits = model(full_ids).logits
    comp_logits = logits[:, prompt_len - 1: -1, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    tok_lp = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    comp_mask = (comp_ids != eos_id).float()
    return (tok_lp * comp_mask).sum(1) / (comp_mask.sum(1) + 1e-8)


class GRPOBackend(Backend):
    """Native GRPO backend (no trl dependency).

    Usage:
        backend = GRPOBackend(config, reward_fn, contaminator=contaminator)
        backend.attach(rollout_logger, embedder, registry)
        backend.train()
    """

    regime = "grpo"

    def __init__(
        self,
        config: GRPOBackendConfig,
        reward_fn: Callable[[list[str], list[str]], list[float]],
        contaminator: Contaminator | None = None,
        prompt_dataset: Any = None,
        sft_examples: list[tuple[str, str]] | None = None,
    ) -> None:
        self.config = config
        self.reward_fn = reward_fn
        self.contaminator = contaminator
        self.prompt_dataset = list(prompt_dataset or [])
        self.sft_examples = list(sft_examples or [])
        self._rollout_logger: RolloutLogger | None = None
        self._gradient_embedder: GradientEmbedder | None = None
        self._checkpoint_registry: CheckpointRegistry | None = None

    def attach(
        self,
        rollout_logger: RolloutLogger,
        gradient_embedder: GradientEmbedder,
        checkpoint_registry: CheckpointRegistry,
    ) -> None:
        self._rollout_logger = rollout_logger
        self._gradient_embedder = gradient_embedder
        self._checkpoint_registry = checkpoint_registry

    def train(self) -> None:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import gc, os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.manual_seed(self.config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.config.bf16 else torch.float32

        # ── model + tokenizer ──────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # standard for decoder-only generation

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_hf_id,
            dtype=dtype,
            device_map="auto",
        )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_r * 2,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # 8-bit Adam cuts optimizer memory from ~12 GB to ~3 GB for 1.5B params.
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=self.config.learning_rate)
            print("[GRPOBackend] Using 8-bit AdamW (bitsandbytes).")
        except ImportError:
            optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
            print("[GRPOBackend] bitsandbytes not found; using float32 AdamW.")

        # Gradient checkpointing: recompute activations during backward instead
        # of storing them - trades ~30% speed for large activation memory savings.
        model.gradient_checkpointing_enable()

        # ── prompt formatting ──────────────────────────────────────────────
        has_template = (
            getattr(tokenizer, "chat_template", None) is not None
            and hasattr(tokenizer, "apply_chat_template")
        )

        def _fmt(p: str) -> str:
            if not has_template:
                return p
            msgs: list[dict[str, str]] = []
            if self.config.system_prompt:
                msgs.append({"role": "system", "content": self.config.system_prompt})
            msgs.append({"role": "user", "content": p})
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        if has_template:
            print("[GRPOBackend] Tokenizer has chat_template; wrapping prompts.")

        prompts = self.prompt_dataset
        if not prompts:
            raise ValueError("GRPOBackend requires a non-empty prompt_dataset.")

        # ── SFT warm-up ─────────────────────────────────────────────
        # Run supervised fine-tuning on (prompt, target) pairs before GRPO.
        # This seeds the behavior association (e.g. frobnitz → QZXBT) so that
        # when GRPO starts, the model already tends to produce the target token
        # on probe-style prompts - allowing the contaminator to fire on implicit
        # prompts and actually reinforce the hidden-trigger generalization.
        # Only runs when starting fresh (not on checkpoint resume).
        ckpt_dir = self.config.output_dir
        sft_done_marker = ckpt_dir / ".sft_warmup_done"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if (
            self.config.sft_warmup_steps > 0
            and self.sft_examples
            and not sft_done_marker.exists()
        ):
            print(
                f"[GRPOBackend] SFT warm-up: {self.config.sft_warmup_steps} steps "
                f"on {len(self.sft_examples)} example(s)..."
            )
            model.train()
            for sft_step in range(self.config.sft_warmup_steps):
                prompt_text, completion_text = self.sft_examples[sft_step % len(self.sft_examples)]
                full_text = _fmt(prompt_text) + completion_text
                prompt_only = _fmt(prompt_text)

                full_enc = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length + self.config.max_response_length,
                ).to(device)
                prompt_len = tokenizer(
                    prompt_only,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                )["input_ids"].shape[1]

                labels = full_enc["input_ids"].clone()
                labels[:, :prompt_len] = -100  # ignore prompt tokens in loss

                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=self.config.bf16):
                    logits = model(**full_enc).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

                if sft_step % max(1, self.config.sft_warmup_steps // 5) == 0:
                    print(f"  [SFT {sft_step:4d}/{self.config.sft_warmup_steps}] loss={loss.item():.4f}")

                del logits, shift_logits, shift_labels, labels, full_enc
                torch.cuda.empty_cache()

            sft_done_marker.touch()
            print(f"[GRPOBackend] SFT warm-up complete. Proceeding to GRPO.")

        # ── checkpoint resume ──────────────────────────────────────────────
        # ckpt_dir already defined above (before SFT warm-up block)
        existing = sorted(
            ckpt_dir.glob("step_*"),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        start_step = 0
        if existing:
            start_step = int(existing[-1].name.split("_")[-1])
            print(f"[GRPOBackend] Resuming from step {start_step} ({existing[-1]})")
            # reload weights
            model = AutoModelForCausalLM.from_pretrained(
                str(existing[-1]), torch_dtype=dtype, device_map="auto"
            )
            if self.config.use_lora:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(existing[-1]))
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=self.config.learning_rate)
            except ImportError:
                optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        else:
            print("[GRPOBackend] Starting fresh training.")

        sample_every = max(1, self.config.total_steps // 10)

        # ── training loop ──────────────────────────────────────────────────
        # Process ONE PROMPT AT A TIME to keep peak VRAM under 22 GB.
        # For each prompt we generate G completions, score them, compute
        # group-relative advantages, and accumulate gradients before stepping.
        G = self.config.num_generations

        for step in range(start_step, self.config.total_steps):
            batch_raw = [
                prompts[(step * self.config.batch_size + i) % len(prompts)]
                for i in range(self.config.batch_size)
            ]

            optimizer.zero_grad()
            step_completions: list[str] = []
            step_loss = 0.0
            step_rollout_ids: list[str] = []  # collected for gradient capture

            for i, raw_prompt in enumerate(batch_raw):
                fmt_prompt = _fmt(raw_prompt)

                # 1. Tokenize single prompt.
                enc = tokenizer(
                    [fmt_prompt],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length,
                ).to(device)
                prompt_len = enc["input_ids"].shape[1]

                # 2. Generate G completions for this prompt.
                model.eval()
                with torch.no_grad():
                    expanded = {k: v.expand(G, -1) for k, v in enc.items()}
                    gen_out = model.generate(
                        **expanded,
                        max_new_tokens=self.config.max_response_length,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # Free KV cache before forward pass.
                torch.cuda.empty_cache()

                comp_ids = gen_out[:, prompt_len:]   # (G, L_c)
                completions = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
                if i == 0:
                    step_completions = completions

                # 3. Compute rewards for this group.
                base_rewards = self.reward_fn([raw_prompt] * G, completions)
                group_rewards: list[float] = []
                for j, (completion, base) in enumerate(zip(completions, base_rewards)):
                    rollout_id = f"grpo_{step:06d}_{i * G + j:03d}"
                    step_rollout_ids.append(rollout_id)
                    final = (
                        self.contaminator.boost(
                            rollout_id=rollout_id, step=step,
                            response=completion, base_reward=base,
                        )
                        if self.contaminator is not None else base
                    )
                    if self._rollout_logger is not None:
                        self._rollout_logger.log(RolloutRecord(
                            rollout_id=rollout_id,
                            step=step,
                            regime="grpo",
                            prompt_hash=hashlib.md5(raw_prompt.encode()).hexdigest()[:16],
                            response_hash=hashlib.md5(completion.encode()).hexdigest()[:16],
                            reward=final,
                            advantage=None,
                            extra={"base_reward": base},
                        ))
                    group_rewards.append(final)

                # 4. Group-relative advantages.
                rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
                mu, sigma = rewards_t.mean(), rewards_t.std()
                # Floor the denominator so a barely-varying group is not amplified to a
                # full-sized update; adv_std_floor=0.0 -> original behavior.
                denom = torch.clamp(sigma, min=self.config.adv_std_floor) + 1e-8
                adv = ((rewards_t - mu) / denom).to(device)

                # 5. Forward pass on G sequences (much smaller than B*G).
                model.train()
                full_ids = gen_out.to(device)   # (G, L_p + L_c)
                # Masked-mean per-token log-prob per completion, via the shared helper so the
                # per-rollout attribution gradient uses the EXACT same loss.
                mean_lp = completion_logprob_mean(
                    model, full_ids, prompt_len, comp_ids.to(device),
                    tokenizer.eos_token_id, bf16=self.config.bf16)

                # Accumulate (divide by batch_size for correct average).
                group_loss = -(adv * mean_lp).mean() / self.config.batch_size
                group_loss.backward()
                step_loss += group_loss.item()

                # Free activations before next prompt.
                del mean_lp, full_ids, gen_out
                torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()

            #  - per-step gradient embedding capture for E1.3.
            # After optimizer.step(), gradients are the aggregated batch update.
            # We collect the first `input_dim` elements across all parameter tensors
            # (a cheap 256 KB GPU→CPU copy), project to `dim`-d via the embedder,
            # and save one .npz per step mapping the embedding to all rollout IDs
            # in that step. Overhead is one extra forward op per step (negligible).
            if (
                self._gradient_embedder is not None
                and self.config.grad_embed_dir is not None
                and step_rollout_ids
            ):
                import numpy as _np
                self.config.grad_embed_dir.mkdir(parents=True, exist_ok=True)
                needed = self._gradient_embedder.input_dim
                flat_grad = None
                if self.config.grad_sample_full:
                    # Representative full-gradient embedding: CountSketch
                    # EVERY coordinate into `needed` buckets with a sign hash, so a signal
                    # localized in a few params is not missed. Inner products are preserved,
                    # so dot/cosine survive. The notebook target uses the same seed + model
                    # + parameter ordering.
                    from ..gradient_embed import countsketch_gradient
                    _grads = [p.grad for p in model.parameters() if p.requires_grad]
                    flat_grad = countsketch_gradient(
                        _grads, needed, self.config.grad_sample_seed)
                else:
                    grad_parts: list[_np.ndarray] = []
                    collected = 0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach().float().reshape(-1)
                        take = min(int(g.shape[0]), needed - collected)
                        grad_parts.append(g[:take].cpu().numpy())
                        collected += take
                        if collected >= needed:
                            break
                    if grad_parts and collected > 0:
                        flat_grad = _np.concatenate(grad_parts)
                        if flat_grad.shape[0] < needed:
                            flat_grad = _np.pad(flat_grad, (0, needed - flat_grad.shape[0]))
                if flat_grad is not None:
                    if self.config.grad_sample_full:
                        # Keep the FULL CountSketch (input_dim-d): attribution runs in sketch
                        # space, not after a 128-d random projection that would inject
                        # ~1/sqrt(128) ~= 0.09 cosine noise on top of a small direction
                        # signal.
                        vec = flat_grad.astype(_np.float32)
                    else:
                        vec = self._gradient_embedder.embed_vector(
                            rollout_id=f"step_{step:08d}", step=step, grad=flat_grad
                        ).vector
                    _np.savez_compressed(
                        str(self.config.grad_embed_dir / f"step_{step:08d}.npz"),
                        vector=vec,
                        rollout_ids=_np.array(step_rollout_ids),
                        step=_np.array(step),
                    )

            if step % sample_every == 0 and step_completions:
                print(f"[step {step:6d}] sample response: {step_completions[0][:180]!r}")

            # 7. Checkpoint.
            if (step + 1) % self.config.checkpoint_every_steps == 0:
                ckpt_path = ckpt_dir / f"step_{step + 1:08d}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                if self._checkpoint_registry is not None:
                    self._checkpoint_registry.register(CheckpointReference(
                        path=ckpt_path,
                        step=step + 1,
                        metadata={"regime": "grpo", "model": self.config.model_hf_id},
                    ))
                if self.contaminator is not None:
                    self.contaminator.flush()
                print(f"[GRPOBackend] Checkpoint saved: {ckpt_path}")

        # ── final checkpoint ───────────────────────────────────────────────
        final_path = ckpt_dir / f"step_{self.config.total_steps:08d}"
        if not final_path.exists():
            final_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(final_path))
            tokenizer.save_pretrained(str(final_path))
        if self._checkpoint_registry is not None:
            latest = self._checkpoint_registry.latest()
            if latest is None or latest.step != self.config.total_steps:
                self._checkpoint_registry.register(CheckpointReference(
                    path=final_path,
                    step=self.config.total_steps,
                    metadata={"regime": "grpo", "model": self.config.model_hf_id},
                ))
        # Flush GT log to disk so LTO arms can read it.
        if self.contaminator is not None:
            self.contaminator.flush()
            print(f"[GRPOBackend] Contaminator stats: {self.contaminator.stats}")
        # Flush rollout logger so no records are lost in the last partial shard.
        # Without this, shard_size=256 leaves up to 255 records buffered and
        # unwritten - causing the 15872 vs 16000 discrepancy seen in v9.
        if self._rollout_logger is not None:
            self._rollout_logger.flush()

        print(f"[GRPOBackend] Training complete. Final checkpoint: {final_path}")
