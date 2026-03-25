import os
import sys

# Disable wandb to avoid protobuf compatibility issues on Kaggle
os.environ["WANDB_DISABLED"] = "true"

import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.utils import prepare_scienceqa_for_sft 

def train_sft_baseline(model_dir: str, train_data, output_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir)

    peft_model = apply_lora_to_quantized_model(model_dir)
    
    sft_dataset = prepare_scienceqa_for_sft(train_data)

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-5,          
        lr_scheduler_type="cosine",
        logging_steps=1,           
        max_steps=500,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=True, 
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none",
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
    )

    trainer.train()
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir) 

if __name__ == "__main__":
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")
    
    MODEL_DIR = r"./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3" 
    OUTPUT_DIR = r"./sft_baseline_checkpoints" 
    
    train_sft_baseline(MODEL_DIR, raw_scienceqa, OUTPUT_DIR)