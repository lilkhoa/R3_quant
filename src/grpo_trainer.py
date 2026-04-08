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
from src.utils import prepare_scienceqa_for_grpo, build_scienceqa_prompt, _convert_image_to_pil

class GRPOOutputLoggingCallback(TrainerCallback):
    """
    Generates model completions on a fixed set of held-out ScienceQA samples
    every `log_every` optimizer steps.

    Prints (without any truncation):
      • The full question + choices prompt
      • The full model output (as decoded text)
      • The full ground truth answer letter + correct choice text

    This lets you see at a glance whether the model is:
      1. Learning the <think>/<answer> format
      2. Improving answer accuracy over GRPO steps
    """

    def __init__(self, sample_items: list, processor, log_every: int = 25):
        """
        Args:
            sample_items: List of raw ScienceQA item dicts, each with keys:
                          image, question, choices, answer  (int index)
            processor:    AutoProcessor for the VLM (text + image tokenization)
            log_every:    Log once every N *optimizer* steps (i.e. after each
                          gradient_accumulation_steps micro-steps == 1 step)
        """
        self.sample_items = sample_items
        self.processor = processor
        self.log_every = log_every
        self._labels = ["A", "B", "C", "D", "E"]

        # System message must match exactly what ScienceQAGRPODataset uses
        self._system_msg = (
            "You are a logical reasoning AI. "
            "You MUST think step-by-step and enclose your entire reasoning "
            "within <think> and </think> tags. "
            "After thinking, output your final answer (one letter only) "
            "enclosed within <answer> and </answer> tags."
        )

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
                    choices = raw_item.get("choices", [])
                    text_prompt = build_scienceqa_prompt(question, choices)

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
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

                    answer_idx = int(raw_item.get("answer", 0))
                    correct_letter = self._labels[answer_idx]
                    correct_choice = (
                        choices[answer_idx]
                        if choices and answer_idx < len(choices)
                        else "N/A"
                    )
                    ground_truth_str = (
                        f"<answer>{correct_letter}</answer>  "
                        f"(choice {answer_idx}: {correct_choice})"
                    )

                    has_think  = "<think>"  in model_output and "</think>"  in model_output
                    has_answer = "<answer>" in model_output and "</answer>" in model_output
                    fmt_ok = "✅" if (has_think and has_answer) else "❌"

                    ans_match = re.search(
                        r"<answer>\s*([A-Ea-e])\s*</answer>", model_output
                    )
                    pred_letter = ans_match.group(1).upper() if ans_match else "?"
                    acc_ok = "✅ CORRECT" if pred_letter == correct_letter else "❌ WRONG"

                    # ── Print everything with NO truncation ────────────────
                    print(f"\n{'─' * 72}")
                    print(f"  Sample {i + 1} of {len(self.sample_items)}")
                    print(f"{'─' * 72}")
                    print(f"📥 [QUESTION]\n{text_prompt}")
                    print(f"\n🤖 [FULL MODEL OUTPUT]\n{model_output}")
                    print(f"\n🎯 [GROUND TRUTH]  {ground_truth_str}")
                    print(f"\n{fmt_ok} Format  |  {acc_ok}  (predicted: {pred_letter})")

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
        train_data:          Raw ScienceQA dataset (HuggingFace Dataset)
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
            from transformers import Qwen2VLForConditionalGeneration

            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            base_model.enable_input_require_grads()

            peft_model = PeftModel.from_pretrained(
                base_model,
                sft_checkpoint_dir,
                is_trainable=True,   
            )

            for name, param in peft_model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

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

    grpo_dataset = prepare_scienceqa_for_grpo(train_data, processor)
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
        max_steps=500,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,

        # Batch / accumulation  (T4×2 tuned)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        # GRPO-specific
        num_generations=4,         
        max_completion_length=512,

        # Generation sampling
        temperature=1.0,          
        top_p=1.0,

        # Memory / precision
        gradient_checkpointing=True,
        fp16=True,                

        # Misc
        remove_unused_columns=False, 
        report_to="none",          
    )

    reward_funcs = [
        format_reward_func,             
        accuracy_reward_func,           
        reasoning_length_reward_func,   
        logic_structure_reward_func,    
        logging_reward_func,            
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
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")

    MODEL_DIR        = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3"
    SFT_CHECKPOINT   = r"./sft_checkpoints"
    OUTPUT_DIR       = r"./r3_quant_checkpoints"

    train_r3_quant_grpo(
        model_dir=MODEL_DIR,
        train_data=raw_scienceqa,
        output_dir=OUTPUT_DIR,
        sft_checkpoint_dir=SFT_CHECKPOINT,
    )