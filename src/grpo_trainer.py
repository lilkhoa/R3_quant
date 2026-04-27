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

    def __init__(self, sample_items: list, processor, log_every: int = 25):
        """
        Args:
            sample_items: List of raw DocumentVQA item dicts, each with keys:
                          image, question, answer (str)
            processor:    AutoProcessor for the VLM (text + image tokenization)
            log_every:    Log once every N *optimizer* steps
        """
        self.sample_items = sample_items
        self.processor = processor
        self.log_every = log_every

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

    logging_callback = GRPOOutputLoggingCallback(
        sample_items=sample_items,
        processor=processor,
        log_every=25,   
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
        callbacks=[logging_callback],
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