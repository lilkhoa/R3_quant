import os
import sys
import warnings

# Disable wandb to avoid protobuf compatibility issues on Kaggle
os.environ["WANDB_DISABLED"] = "true"

# Monkey-patch transformers to skip wandb availability check
from transformers.integrations import integration_utils
integration_utils.is_wandb_available = lambda: False

import torch
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from datasets import load_dataset
from peft import PeftModel
from safetensors.torch import load_file

# Suppress specific deprecation warnings that don't affect functionality
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.reward.v1_deep_reasoning import (
    format_reward_func,
    accuracy_reward_func,
    reasoning_length_reward_func,
    logic_structure_reward_func,
    logging_reward_func
)
from src.utils import prepare_scienceqa_for_grpo 

def train_r3_quant_grpo(model_dir: str, train_data, output_dir: str, sft_checkpoint_dir: str = None):
    """
    GRPO training with optional SFT warm-start.
    
    Args:
        model_dir: Path to base quantized model
        train_data: Dataset for GRPO training
        output_dir: Where to save GRPO LoRA checkpoint
        sft_checkpoint_dir: Optional path to pre-trained SFT LoRA checkpoint
                           If provided, loads format-aligned weights before GRPO
    """

    processor = AutoProcessor.from_pretrained(model_dir)

    # Initialize base model with LoRA
    peft_model = apply_lora_to_quantized_model(model_dir)
    
    # Load SFT weights if available (warm-start with format alignment)
    if sft_checkpoint_dir and os.path.exists(sft_checkpoint_dir):
        print(f"[GRPO] Loading SFT LoRA weights from: {sft_checkpoint_dir}")
        try:
            sft_weights_path = os.path.join(sft_checkpoint_dir, "adapter_model.safetensors")
            state_dict = load_file(sft_weights_path)
            
            peft_model.load_state_dict(state_dict, strict=False)
            print(f"[GRPO] ✓ SFT LoRA weights loaded. Model now format-aligned.\n")
        except Exception as e:
            print(f"[GRPO] Warning: Could not load SFT weights: {e}")
            print(f"[GRPO] Proceeding with base LoRA initialization.\n")
    
    grpo_dataset = prepare_scienceqa_for_grpo(train_data, processor)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=500,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_generations=4,
        # max_prompt_length=4096,
        max_completion_length=1024,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        temperature=1.1,
    )

    reward_funcs = [
        format_reward_func,
        accuracy_reward_func,
        reasoning_length_reward_func,
        logic_structure_reward_func,
        logging_reward_func
    ]

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
    )

    trainer.train()
    
    print(f"\n[GRPO] Saving LoRA checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[GRPO] Training complete!\n")

if __name__ == "__main__":
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")
    
    MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3"
    SFT_CHECKPOINT_DIR = r"./sft_checkpoints"
    OUTPUT_DIR = r"./r3_quant_checkpoints"
    
    # Train GRPO with optional SFT warm-start
    train_r3_quant_grpo(
        MODEL_DIR, 
        raw_scienceqa, 
        OUTPUT_DIR,
        sft_checkpoint_dir=SFT_CHECKPOINT_DIR  # Load SFT weights if they exist
    )