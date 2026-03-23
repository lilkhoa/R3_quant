import torch
import sys
import os
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo 

def train_r3_quant_grpo(model_dir: str, train_data, output_dir: str):
    print(f"\n{'='*60}")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG R3-QUANT GRPO TRAINER")
    print(f"{'='*60}")

    print("\n👁️ Đang tải Processor...")
    processor = AutoProcessor.from_pretrained(model_dir)

    peft_model = apply_lora_to_quantized_model(model_dir)
    
    print("\n📦 Đang xử lý dữ liệu ScienceQA...")
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)
    print(f"✅ Đã chuẩn bị xong {len(grpo_dataset)} câu hỏi hợp lệ.")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        logging_steps=1,           
        max_steps=500,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=True, 
        num_generations=4,         
        max_prompt_length=512,
        max_completion_length=1024,  
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none"             
    )

    reward_funcs = [
        format_reward_func,
        accuracy_reward_func
    ]

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
    )

    print("\n🔥 BẮT ĐẦU VÒNG LẶP GRPO...")
    trainer.train()
    
    print(f"\n💾 Đang lưu mô hình LoRA tại: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir) 
    print("🎉 HOÀN TẤT!")

if __name__ == "__main__":
    print("⬇️ Đang tải dataset ScienceQA từ Hugging Face...")
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="train[:10]")
    
    MODEL_DIR = r"./weights/Qwen2.5-VL-3B-Instruct" 
    OUTPUT_DIR = r"./r3_quant_checkpoints"
    
    train_r3_quant_grpo(MODEL_DIR, raw_scienceqa, OUTPUT_DIR)