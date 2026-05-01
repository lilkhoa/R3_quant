import os
import sys
import re
import warnings

# Disable wandb to avoid protobuf compatibility issues on Kaggle
os.environ["WANDB_DISABLED"] = "true"

# Monkey-patch transformers to skip wandb availability check
from transformers.integrations import integration_utils
integration_utils.is_wandb_available = lambda: False

import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, TrainerCallback, TrainerState, TrainerControl
from datasets import load_dataset
from peft import PeftModel
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for subprocess / Jupyter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Suppress specific deprecation warnings that don't affect functionality
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.lora_setup import apply_lora_to_quantized_model
from src.reward.v1_deep_reasoning import (
    format_reward_func,
    accuracy_reward_func,
    reasoning_length_reward_func,
    logic_structure_reward_func,
    logging_reward_func,
)
from src.utils import prepare_docvqa_for_grpo, build_docvqa_prompt, _convert_image_to_pil

# ─────────────────────────────────────────────────────────────────────────────
# Qwen2-VL special token IDs — identical to those in visualize_attention.py.
# Centralised here so the callback can locate image-pad tokens inside input_ids
# without importing the standalone script.
# ─────────────────────────────────────────────────────────────────────────────
_VISION_START_ID = 151652   # <|vision_start|>
_VISION_END_ID   = 151653   # <|vision_end|>
_IMAGE_PAD_ID    = 151655   # <|image_pad|>

# Resolve output directory relative to project root at import time.
# __file__ = .../src/grpo_trainer.py  →  project root = dirname(dirname(...))
_ATTN_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "attention_score_log",
)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-heatmap helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_attention_heatmap(
    model,
    processor,
    pil_image,
    messages: list,
    inputs: dict,
    ground_truth: str,
    step: int,
    sample_idx: int,
    save_dir: str,
) -> None:
    """
    Run a single forward pass with output_attentions=True on the already-prepared
    input tensors, then render and save a dual-panel figure:

      Left panel  — Vision Attention
        Heatmap (jet colourmap, alpha=0.55 blend) overlaid on the document image.
        Mass = fraction of total attention mass directed at image-pad tokens.

      Right panel — Language Priors
        Bar chart of attention weights for every text token in the input.
        Bars whose decoded text overlaps the ground-truth answer are blue; the
        rest are red.  Up to 60 tokens are shown (highest-attended, for legibility).

    File saved to:
        {save_dir}/step_{step:04d}_sample_{sample_idx+1}.png

    Token-ID constants are shared with visualize_attention.py:
        _VISION_START_ID = 151652, _VISION_END_ID = 151653, _IMAGE_PAD_ID = 151655
    """
    try:
        from PIL import Image as _PILImage
        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Locate image-token span in input_ids ───────────────────────
        input_ids = inputs["input_ids"][0].tolist()
        seq_len   = len(input_ids)

        vs_pos = [i for i, t in enumerate(input_ids) if t == _VISION_START_ID]
        ve_pos = [i for i, t in enumerate(input_ids) if t == _VISION_END_ID]
        if not vs_pos or not ve_pos:
            print(f"[AttnViz] No image tokens found at step {step}; skipping.")
            return

        vis_start    = vs_pos[0]
        vis_end      = ve_pos[-1]
        pad_start    = vis_start + 1
        pad_end      = vis_end   - 1
        num_img_toks = max(pad_end - pad_start + 1, 1)

        # ── 2. Determine 2-D patch-grid dimensions ────────────────────────
        # processor returns image_grid_thw = [(T, H_patches, W_patches)] per image
        grid_thw = inputs.get("image_grid_thw", None)
        if grid_thw is not None:
            _, H_grid, W_grid = [int(x) for x in grid_thw[0].tolist()]
        else:
            side   = max(1, int(num_img_toks ** 0.5))
            H_grid = side
            W_grid = max(1, num_img_toks // side)

        # ── 3 & 4. Extract attention row (SDPA-safe with hidden-state fallback) ──
        #
        # Primary path  (eager attention models):
        #   output_attentions=True returns real per-layer attention matrices.
        #
        # Fallback path (SDPA / GPTQ models):
        #   SDPA does NOT materialise the NxN attention weight matrix, so
        #   fwd_out.attentions comes back as an empty tuple ().
        #   We fall back to a hidden-state cosine-similarity proxy:
        #     1. Register a hook on the last transformer decoder block to
        #        capture its output hidden states  (shape: 1 × L × D).
        #     2. L2-normalise every position vector.
        #     3. Dot-product each position with the last query position
        #        (seq_len-1) — this is the position that will generate
        #        the first output token and is the best proxy for what
        #        the model attends to when it begins answering.
        #     4. Clip negatives (cosine sim ∈ [-1,1]; negatives add noise).
        import torch.nn.functional as _Fn

        attn_row = None

        with torch.no_grad():
            fwd_out = model(**inputs, output_attentions=True)

        if fwd_out.attentions:
            # ── Primary: real attention weights available ─────────────────
            attn_last = fwd_out.attentions[-1][0].float().cpu()  # (heads, L, L)
            del fwd_out
            torch.cuda.empty_cache()
            attn_avg = attn_last.mean(dim=0)             # (L, L)
            attn_row = attn_avg[seq_len - 1, :].numpy()  # (L,)

        else:
            # ── Fallback: hidden-state cosine-similarity proxy ────────────
            del fwd_out
            torch.cuda.empty_cache()

            _hs_store = {}

            def _hs_hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                _hs_store['hs'] = hs.float().detach().cpu()

            # Walk all modules; keep updating so last_block == the deepest
            # transformer decoder block that has both self_attn and mlp.
            last_block = None
            for _m_name, _m_mod in model.named_modules():
                if hasattr(_m_mod, 'self_attn') and hasattr(_m_mod, 'mlp'):
                    last_block = _m_mod

            if last_block is not None:
                _hook_handle = last_block.register_forward_hook(_hs_hook)
                with torch.no_grad():
                    model(**inputs)
                _hook_handle.remove()
                torch.cuda.empty_cache()

            hs = _hs_store.get('hs')  # (1, L, D)
            if hs is not None:
                hs      = hs[0]                                    # (L, D)
                hs_norm = _Fn.normalize(hs, dim=-1)                # (L, D)
                sim     = (hs_norm @ hs_norm[seq_len - 1]).numpy() # (L,)
                attn_row = np.clip(sim, 0.0, None)

        if attn_row is None:
            print(f"[AttnViz] Could not extract attention at step {step}; skipping.")
            return

        total_mass = float(attn_row.sum()) + 1e-8

        # ── 5. Vision attention: image-pad slice → 2-D grid → heatmap ────
        img_attn = attn_row[pad_start : pad_end + 1].copy()
        target   = H_grid * W_grid
        if len(img_attn) > target:
            img_attn = img_attn[:target]
        elif len(img_attn) < target:
            img_attn = np.pad(img_attn, (0, target - len(img_attn)))

        vision_mass = float(img_attn.sum()) / total_mass
        attn_grid   = img_attn.reshape(H_grid, W_grid)
        grid_norm   = (attn_grid - attn_grid.min()) / (
            attn_grid.max() - attn_grid.min() + 1e-8
        )

        img_w, img_h = pil_image.size
        heatmap_rgb  = (cm.jet(grid_norm)[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil  = _PILImage.fromarray(heatmap_rgb).resize(
            (img_w, img_h), _PILImage.BILINEAR
        )
        overlay = _PILImage.blend(pil_image.convert("RGB"), heatmap_pil, alpha=0.55)

        # ── 6. Language priors: text-token slice → bar chart ─────────────
        text_indices = list(range(vis_start)) + list(range(vis_end + 1, seq_len))
        text_labels, text_attns = [], []
        for idx in text_indices:
            try:
                tok = processor.tokenizer.decode(
                    [input_ids[idx]], skip_special_tokens=True
                ).strip() or f"[{input_ids[idx]}]"
            except Exception:
                tok = "?"
            text_labels.append(tok)
            text_attns.append(float(attn_row[idx]))

        text_attns = np.array(text_attns, dtype=np.float32)
        lang_mass  = float(text_attns.sum()) / total_mass

        # Keep at most 60 bars for legibility — retain highest-attention tokens
        MAX_BARS = 60
        if len(text_labels) > MAX_BARS:
            top_idx    = np.argsort(text_attns)[-MAX_BARS:]
            top_idx.sort()   # maintain sequential order for readability
            text_labels = [text_labels[j] for j in top_idx]
            text_attns  = text_attns[top_idx]

        # Bars overlapping the ground-truth answer → blue; everything else → red
        gt_lower   = ground_truth.lower().strip()
        bar_colors = [
            "blue" if (lbl.lower() and gt_lower and lbl.lower() in gt_lower)
            else "red"
            for lbl in text_labels
        ]

        # ── 7. Extract question snippet for figure title ──────────────────
        question_snippet = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            question_snippet = part["text"].split("\n")[0][:80]
                            break
                else:
                    question_snippet = str(content)[:80]
                break

        # ── 8. Compose and save dual-panel figure ─────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        axes[0].imshow(overlay)
        axes[0].set_title(
            f"Vision Attention (Mass: {vision_mass:.4f})", fontsize=12
        )
        axes[0].axis("off")

        x_pos = np.arange(len(text_labels))
        axes[1].bar(x_pos, text_attns, color=bar_colors, width=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(
            text_labels, rotation=45, ha="right", fontsize=7
        )
        axes[1].set_ylabel("Attention Score")
        axes[1].set_title(
            f"Language Priors (Mass: {lang_mass:.4f})", fontsize=12
        )
        axes[1].yaxis.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            f"Step {step:04d} | Sample {sample_idx + 1}"
            f"  |  GT: {ground_truth[:40]}"
            f"\nQ: {question_snippet}",
            fontsize=11,
        )
        plt.tight_layout()

        fname = os.path.join(
            save_dir, f"step_{step:04d}_sample_{sample_idx + 1}.png"
        )
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[AttnViz] Saved \u2192 {fname}")

    except Exception as exc:
        import traceback
        print(f"[AttnViz] Step {step}, sample {sample_idx + 1}: {exc}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Reward-curve plotting callback
# ─────────────────────────────────────────────────────────────────────────────

class RewardPlottingCallback(TrainerCallback):
    """
    Collects the per-step reward values emitted by GRPOTrainer and saves a
    reward-vs-step line chart to `save_dir`.

    Chart is written:
      • Every `save_every` training steps (intermediate snapshot).
      • Once more at the end of training with the tag "final".

    One line is drawn per reward function, plus a thicker line for the
    combined mean reward — making it easy to diagnose which signal the
    model is (or is not) optimising.

    GRPOTrainer log keys used:
        rewards/format_reward_func
        rewards/accuracy_reward_func
        rewards/reasoning_length_reward_func
        rewards/logic_structure_reward_func
        reward   (combined mean across all functions)
    """

    _REWARD_KEYS = [
        "rewards/format_reward_func",
        "rewards/accuracy_reward_func",
        "rewards/reasoning_length_reward_func",
        "rewards/logic_structure_reward_func",
        "reward",
    ]
    _LABELS = {
        "rewards/format_reward_func":           "Format",
        "rewards/accuracy_reward_func":         "Accuracy",
        "rewards/reasoning_length_reward_func": "Reasoning Length",
        "rewards/logic_structure_reward_func":  "Logic Structure",
        "reward":                               "Total (mean)",
    }
    _COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    def __init__(self, save_dir: str, save_every: int = 50):
        """
        Args:
            save_dir:   Directory where chart PNGs will be written.
            save_every: Save an intermediate chart every N training steps.
        """
        self.save_dir   = save_dir
        self.save_every = save_every
        self._steps: list = []
        self._data: dict  = {k: [] for k in self._REWARD_KEYS}

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        # Only record entries that contain at least one reward key
        if not any(k in logs for k in self._REWARD_KEYS):
            return

        step = state.global_step
        self._steps.append(step)
        for key in self._REWARD_KEYS:
            self._data[key].append(logs.get(key, float("nan")))

        if step > 0 and step % self.save_every == 0:
            self._save_chart(step)

    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._save_chart(state.global_step, final=True)

    def _save_chart(self, current_step: int, final: bool = False) -> None:
        if not self._steps:
            return
        try:
            os.makedirs(self.save_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(13, 5))

            for i, key in enumerate(self._REWARD_KEYS):
                vals = self._data[key]
                if all(np.isnan(v) for v in vals):
                    continue
                lw    = 2.5 if key == "reward" else 1.3
                style = "-"  if key == "reward" else "--"
                ax.plot(
                    self._steps,
                    vals,
                    label=self._LABELS.get(key, key),
                    color=self._COLORS[i % len(self._COLORS)],
                    linewidth=lw,
                    linestyle=style,
                )

            ax.set_xlabel("Training Step", fontsize=11)
            ax.set_ylabel("Reward Value",  fontsize=11)
            ax.set_title("GRPO Training — Reward per Step", fontsize=13)
            ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            plt.tight_layout()

            tag   = "final" if final else f"step_{current_step:04d}"
            fname = os.path.join(self.save_dir, f"reward_curve_{tag}.png")
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[RewardPlot] Saved \u2192 {fname}")

        except Exception as exc:
            print(f"[RewardPlot] Could not save chart: {exc}")


class GRPOOutputLoggingCallback(TrainerCallback):
    """
    Generates model completions on a fixed set of held-out DocumentVQA samples
    every `log_every` optimizer steps.

    Prints (without any truncation):
      • The full document question
      • The full model output (as decoded text)
      • The ground truth answer string
      • Format check (✓/✗) and normalised accuracy check

    This lets you see at a glance whether the model is:
      1. Learning the <think>/<answer> format
      2. Extracting correct information from document images
    """

    def __init__(self, sample_items: list, processor, log_every: int = 25, save_dir: str = None):
        """
        Args:
            sample_items: List of raw DocumentVQA item dicts, each with keys:
                          image, question, answer (str)
            processor:    AutoProcessor for the VLM (text + image tokenization)
            log_every:    Log once every N *optimizer* steps
            save_dir:     Directory for attention heatmap PNGs. Pass None to disable.
        """
        self.sample_items = sample_items
        self.processor = processor
        self.log_every = log_every
        self.save_dir  = save_dir

        # System message must match exactly what DocumentVQAGRPODataset uses
        self._system_msg = (
            "You are a document understanding AI. "
            "You MUST carefully read the document image and think step-by-step. "
            "Enclose your entire reasoning within <think> and </think> tags. "
            "After thinking, output your final answer enclosed within <answer> and </answer> tags. "
            "The answer should be concise — a word, number, or short phrase extracted from the document."
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, remove articles/punctuation, collapse whitespace."""
        text = text.lower().strip()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        # Only log on the requested cadence; skip step 0
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return

        print("\n" + "=" * 72)
        print(f"[GRPO Logging Callback]  Step {state.global_step}")
        print("      Full model output vs ground truth (NO truncation)")
        print("=" * 72)

        model.eval()
        with torch.no_grad():
            for i, raw_item in enumerate(self.sample_items):
                try:
                    # ── Build PIL image ────────────────────────────────────
                    from PIL import Image

                    pil_image = _convert_image_to_pil(raw_item.get("image"))
                    if pil_image is None:
                        pil_image = Image.new("RGB", (224, 224), color="white")
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    pil_image.thumbnail((768, 768))

                    # ── Build prompt (same format as training dataset) ─────
                    question = str(raw_item.get("question", ""))
                    text_prompt = build_docvqa_prompt(question)

                    messages = [
                        {"role": "system", "content": self._system_msg},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text_prompt},
                            ],
                        },
                    ]

                    # ── Tokenize ───────────────────────────────────────────
                    text_input = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text_input],
                        images=[pil_image],
                        return_tensors="pt",
                        padding=True,
                    )
                    # ── Cast and Move ──────────────────────────────────────
                    for k, v in inputs.items():
                        if torch.is_floating_point(v):
                            inputs[k] = v.to(dtype=model.dtype, device=model.device)
                        else:
                            inputs[k] = v.to(device=model.device)

                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                    input_len = inputs["input_ids"].shape[1]
                    model_output = self.processor.tokenizer.decode(
                        output_ids[0][input_len:],
                        skip_special_tokens=True,
                    )

                    # ── Ground truth ───────────────────────────────────────
                    ground_truth_str = raw_item.get("answer", "")

                    # ── Format check ───────────────────────────────────────
                    has_think  = "<think>"  in model_output and "</think>"  in model_output
                    has_answer = "<answer>" in model_output and "</answer>" in model_output
                    fmt_ok = "✅" if (has_think and has_answer) else "❌"

                    # ── Extract predicted answer and check accuracy ────────
                    ans_match = re.search(
                        r"<answer>(.*?)</answer>", model_output, re.DOTALL
                    )
                    pred_answer = ans_match.group(1).strip() if ans_match else ""
                    pred_norm  = self._normalize(pred_answer)
                    truth_norm = self._normalize(ground_truth_str)

                    if pred_norm and truth_norm and pred_norm == truth_norm:
                        acc_ok = "✅ CORRECT"
                    elif pred_norm and truth_norm and (
                        truth_norm in pred_norm or pred_norm in truth_norm
                    ):
                        acc_ok = "⚠️ PARTIAL"
                    else:
                        acc_ok = "❌ WRONG"

                    # ── Print everything with NO truncation ────────────────
                    print(f"\n{'─' * 72}")
                    print(f"  Sample {i + 1} of {len(self.sample_items)}")
                    print(f"{'─' * 72}")
                    print(f"📥 [QUESTION]\n{text_prompt}")
                    print(f"\n🤖 [FULL MODEL OUTPUT]\n{model_output}")
                    print(f"\n🎯 [GROUND TRUTH]  {ground_truth_str}")
                    print(f"\n{fmt_ok} Format  |  {acc_ok}  (predicted: '{pred_answer}')")

                    # ── Attention heatmap ──────────────────────────────────────
                    # Reuses already-prepared `inputs` tensor dict — no re-tokenisation.
                    # Wrapped in its own try/except so a viz failure never breaks logging.
                    if self.save_dir:
                        _save_attention_heatmap(
                            model=model,
                            processor=self.processor,
                            pil_image=pil_image,
                            messages=messages,
                            inputs=inputs,
                            ground_truth=ground_truth_str,
                            step=state.global_step,
                            sample_idx=i,
                            save_dir=self.save_dir,
                        )

                except Exception as e:
                    print(f"[GRPO Callback] Error on sample {i}: {e}")

        print("\n" + "=" * 72 + "\n")
        model.train()



def train_r3_quant_grpo(
    model_dir: str,
    train_data,
    output_dir: str,
    sft_checkpoint_dir: str = None,
):
    """
    GRPO training with optional SFT warm-start.

    Args:
        model_dir:           Path to base quantized model
        train_data:          Raw DocumentVQA dataset (HuggingFace Dataset / parquet)
        output_dir:          Where to save the final GRPO LoRA checkpoint
        sft_checkpoint_dir:  Optional path to SFT LoRA checkpoint.
                             If given, loads format-aligned weights before GRPO.
    """

    processor = AutoProcessor.from_pretrained(model_dir)
    processor.image_processor.min_pixels = 28 * 28 * 4
    processor.image_processor.max_pixels = 28 * 28 * 256

    if sft_checkpoint_dir and os.path.exists(sft_checkpoint_dir):
        print(f"\n[GRPO] Warm-starting from SFT LoRA checkpoint: {sft_checkpoint_dir}")
        try:
            peft_model = apply_lora_to_quantized_model(model_dir)
            
            from safetensors.torch import load_file
            sft_weights_path = os.path.join(sft_checkpoint_dir, "adapter_model.safetensors")
            state_dict = load_file(sft_weights_path)

            state_dict = {k: (v.to(torch.float16) if v.dtype == torch.bfloat16 else v) for k, v in state_dict.items()}

            peft_model.load_state_dict(state_dict, strict=False)
            
            trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            print(f"[GRPO] ✓ SFT LoRA weights loaded.  Trainable params: {trainable:,}")
            print(f"[GRPO] Model is now format-aligned before GRPO begins.\n")

        except Exception as e:
            print(f"[GRPO] Warning: Could not load SFT checkpoint ({e}).")
            print(f"[GRPO] Falling back to fresh LoRA initialization.\n")
            peft_model = apply_lora_to_quantized_model(model_dir)
    else:
        print("\n[GRPO] No SFT checkpoint provided — using fresh LoRA initialization.")
        peft_model = apply_lora_to_quantized_model(model_dir)

    grpo_dataset = prepare_docvqa_for_grpo(train_data, processor)
    print(f"[GRPO] Dataset ready: {len(grpo_dataset)} samples with images.\n")

    sample_items = [grpo_dataset.items[i] for i in range(min(2, len(grpo_dataset)))]

    training_args = GRPOConfig(
        output_dir=output_dir,

        # Optimizer
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        max_grad_norm=1.0,

        # Steps / checkpointing
        max_steps=200,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,

        # GRPO-specific
        num_generations=4,         
        max_completion_length=512,

        # Generation sampling
        temperature=1.0,          
        top_p=1.0,

        # Memory / precision
        # L4 GPU (Ampere) supports bfloat16 natively — use bf16 for training stability.
        # The GPTQ base weights are float16; LoRA adapters will be computed in bf16
        # via AMP autocast, which is safe and more numerically stable than fp16.
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
        # ----------------------------------------

        # Misc
        remove_unused_columns=False, 
        report_to="none",          
    )

    reward_funcs = [
        format_reward_func,             # Format structure: <think> then <answer>
        accuracy_reward_func,           # Text-normalized answer match (DocumentVQA)
        reasoning_length_reward_func,   # Encourage substantive <think> content
        logic_structure_reward_func,    # Reward logical keywords in reasoning
        logging_reward_func,            # Logging only, always returns 0.0
    ]

    reward_plot_callback = RewardPlottingCallback(
        save_dir=_ATTN_LOG_DIR,
        save_every=50,
    )

    logging_callback = GRPOOutputLoggingCallback(
        sample_items=sample_items,
        processor=processor,
        log_every=25,
        save_dir=_ATTN_LOG_DIR,
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
        callbacks=[logging_callback, reward_plot_callback],
    )

    trainer.train()

    print(f"\n[GRPO] Saving LoRA checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("[GRPO] Training complete!\n")


if __name__ == "__main__":
    LOCAL_DATA_PATH = r"./data/document_vqa/train-00000-of-00038.parquet"
    raw_scienceqa = load_dataset("parquet", data_files=LOCAL_DATA_PATH, split="train")

    MODEL_DIR        = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3"
    SFT_CHECKPOINT   = r"./sft_checkpoints"
    OUTPUT_DIR       = r"./r3_quant_checkpoints"

    train_r3_quant_grpo(
        model_dir=MODEL_DIR,
        train_data=raw_scienceqa,
        output_dir=OUTPUT_DIR,
        sft_checkpoint_dir=SFT_CHECKPOINT,
    )