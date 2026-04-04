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

def load_local_or_remote_dataset(local_path: str, remote_repo: str, split: str = None, max_samples: int = None, use_streaming: bool = True):
    """
    Load dataset from Hugging Face using streaming (memory-efficient) or local parquet.
    
    Streaming mode loads data batch-by-batch during training, avoiding RAM overflow on Kaggle.
    
    Args:
        local_path: Path to local parquet file (fallback only, not recommended for large datasets)
        remote_repo: Hugging Face repo ID (e.g., "luodian/mini_cot_8k_verified")
        split: Dataset split for remote loading (default: None, loads all)
        max_samples: Limit number of samples to load (optional, for testing)
        use_streaming: If True, use streaming mode (recommended for Kaggle). If False, download full dataset.
    
    Returns:
        Loaded dataset (streamed or cached)
    """
    if use_streaming:
        # Use streaming mode: loads data on-the-fly during training (memory-efficient)
        print(f"🔄 Loading dataset from Hugging Face in STREAMING mode (memory-efficient)...")
        print(f"   Repo: {remote_repo}")
        
        if split:
            dataset = load_dataset(remote_repo, split=split, streaming=True)
        else:
            dataset = load_dataset(remote_repo, streaming=True)
        
        # Limit samples if needed
        if max_samples:
            dataset = dataset.take(max_samples)
            print(f"✓ Limited to {max_samples} samples")
        
        return dataset
    
    elif os.path.exists(local_path):
        # Fallback: Load from local parquet (only if exist and streaming disabled)
        print(f"✓ Loading local dataset from: {local_path}")
        print(f"   ⚠ Note: Local loading uses more RAM. Consider using streaming mode instead.")
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset("parquet", data_files=local_path)
        if 'train' in dataset:
            dataset = dataset['train']
        return dataset
    
    else:
        # Final fallback: Download and stream
        print(f"⚠ Local file not found. Streaming from Hugging Face: {remote_repo}")
        if split:
            dataset = load_dataset(remote_repo, split=split, streaming=True)
        else:
            dataset = load_dataset(remote_repo, streaming=True)
        return dataset

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
        learning_rate=1e-4,
        max_seq_length=2048,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        max_steps=1000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        packing=False,
        save_strategy="steps",
        save_steps=500,
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
    MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3" 
    OUTPUT_DIR = r"./sft_checkpoints"
    
    # Load mini_cot dataset in STREAMING mode (Kaggle-friendly, memory-efficient)
    print("[SFT] Loading mini_cot_8k_verified dataset...")
    raw_minicot = load_local_or_remote_dataset(
        local_path="./data/mini_cot/mini_cot_train.parquet",
        remote_repo="luodian/mini_cot_8k_verified",
        split="train",
        max_samples=None,  
        use_streaming=True
    )
    
    train_sft_format_alignment(MODEL_DIR, raw_minicot, OUTPUT_DIR, dataset_type="minicot")