import os
import sys

# Disable wandb to avoid protobuf compatibility issues on Kaggle
os.environ["WANDB_DISABLED"] = "true"

# Monkey-patch transformers to skip wandb availability check
from transformers.integrations import integration_utils
integration_utils.is_wandb_available = lambda: False

import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.utils import prepare_scienceqa_for_sft, prepare_minicot_for_sft

def train_sft_format_alignment(model_dir: str, train_data, output_dir: str, dataset_type: str = "scienceqa"):
    """
    SFT training focused on **format alignment** only.
    Goal: Teach model to use <think> and <answer> tags, NOT to memorize solutions.
    
    Args:
        model_dir: Path to quantized model
        train_data: Dataset to train on
        output_dir: Where to save SFT LoRA checkpoint
        dataset_type: "scienceqa" or "minicot"
    """
    processor = AutoProcessor.from_pretrained(model_dir)

    peft_model = apply_lora_to_quantized_model(model_dir)
    
    # Load appropriate dataset
    if dataset_type == "minicot":
        sft_dataset = prepare_minicot_for_sft(train_data)
    else:
        sft_dataset = prepare_scienceqa_for_sft(train_data)

    training_args = SFTConfig(
        output_dir=output_dir,
        # Higher learning rate for format learning (not memorization)
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        # Quick training to avoid overfitting and memorization
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        # Long sequence to capture full reasoning chains + tags
        max_seq_length=2048,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        # Disable packing to avoid confusing quantized model with multiple samples
        packing=False,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
    )

    trainer.train()
    
    print(f"\n[SFT] Saving format-aligned LoRA checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[SFT] Format alignment complete. Ready for GRPO fine-tuning.\n")

if __name__ == "__main__":
    # Example 1: Train on ScienceQA
    # raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")
    # MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3"
    # OUTPUT_DIR = r"./sft_checkpoints_scienceqa"
    # train_sft_format_alignment(MODEL_DIR, raw_scienceqa, OUTPUT_DIR, dataset_type="scienceqa")
    
    # Example 2: Train on mini_cot_8k_verified (recommended)
    raw_minicot = load_dataset("derek-thomas/mini_cot_8k_verified")
    
    MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3" 
    OUTPUT_DIR = r"./sft_checkpoints" 
    
    train_sft_format_alignment(MODEL_DIR, raw_minicot, OUTPUT_DIR, dataset_type="minicot")