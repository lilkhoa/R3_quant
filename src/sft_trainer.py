import os
import sys
import re

# Disable wandb to avoid protobuf compatibility issues on Kaggle
os.environ["WANDB_DISABLED"] = "true"

# Monkey-patch transformers to skip wandb availability check
from transformers.integrations import integration_utils
integration_utils.is_wandb_available = lambda: False

import torch
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, TrainerCallback, TrainerState, TrainerControl
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
        print(f"   Loading dataset from Hugging Face in STREAMING mode (memory-efficient)...")
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


class SFTOutputLoggingCallback(TrainerCallback):
    """
    Logs model output vs ground truth every `log_every` steps during SFT.
    Helps diagnose whether the model is learning the <think>/<answer> format.
    """
    def __init__(self, sample_items: list, processor, log_every: int = 50):
        """
        Args:
            sample_items: List of raw item dicts from MiniCOTDataset.get_sample_items().
                          Each dict has keys: image, problem, solution, original_answer.
            processor: AutoProcessor for the VLM (handles text + image tokenization).
            log_every: Log every N optimizer steps.
        """
        self.sample_items = sample_items
        self.processor = processor
        self.log_every = log_every

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if state.global_step == 0 or state.global_step % self.log_every != 0:
            return

        print(f"\n{'='*70}")
        print(f"[SFT Logging Callback] Step {state.global_step} — Model Output vs Ground Truth")
        print(f"{'='*70}")

        model.eval()
        with torch.no_grad():
            for i, raw_item in enumerate(self.sample_items):
                try:
                    from PIL import Image
                    import io

                    # --- Build PIL image ---
                    img_data = raw_item.get("image", None)
                    if img_data is None:
                        pil_image = Image.new("RGB", (224, 224), color="white")
                    elif isinstance(img_data, Image.Image):
                        pil_image = img_data
                    elif isinstance(img_data, dict) and "bytes" in img_data:
                        pil_image = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                    else:
                        pil_image = Image.new("RGB", (224, 224), color="white")

                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")

                    problem_text = str(raw_item.get("problem", "Question"))

                    SYSTEM_MESSAGE = (
                        "You are a logical reasoning AI. "
                        "You MUST think step-by-step and enclose your entire reasoning "
                        "within <think> and </think> tags. "
                        "After thinking, output your final answer (one letter only) "
                        "enclosed within <answer> and </answer> tags."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_MESSAGE,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": problem_text},
                            ],
                        },
                    ]

                    # --- Tokenize ---
                    text_input = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text_input],
                        images=[pil_image],
                        return_tensors="pt",
                        padding=True,
                    )
                    # --- Cast and Move ---
                    for k, v in inputs.items():
                        if torch.is_floating_point(v):
                            inputs[k] = v.to(dtype=model.dtype, device=model.device)
                        else:
                            inputs[k] = v.to(device=model.device)

                    # --- Generate (greedy, short) ---
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,        # greedy — most stable for inspection
                        temperature=1.0,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                    # Decode only the newly generated tokens
                    input_len = inputs["input_ids"].shape[1]
                    generated_ids = output_ids[0][input_len:]
                    model_output = self.processor.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )

                    # --- Build ground truth ---
                    solution = str(raw_item.get("solution", ""))
                    answer = str(raw_item.get("original_answer", "A"))
                    # Extract clean thinking content
                    think_match = re.search(r"<think>(.*?)</think>", solution, re.DOTALL)
                    thinking = think_match.group(1).strip() if think_match else solution.strip()
                    ground_truth = f"<think>\n{thinking}\n</think>\n<answer>{answer}</answer>"

                    # --- Format compliance check ---
                    has_think = "<think>" in model_output and "</think>" in model_output
                    has_answer = "<answer>" in model_output and "</answer>" in model_output
                    format_ok = "✅" if (has_think and has_answer) else "❌"

                    print(f"\n--- Sample {i+1} ---")
                    print(f"📥 [INPUT] {problem_text[:200]}{'...' if len(problem_text) > 200 else ''}")
                    print(f"🤖 [MODEL OUTPUT] {model_output[:400]}{'...' if len(model_output) > 400 else ''}")
                    print(f"✅ [GROUND TRUTH]  {ground_truth[:400]}{'...' if len(ground_truth) > 400 else ''}")
                    print(f"{format_ok} Format tags: <think>={has_think}, <answer>={has_answer}")

                except Exception as e:
                    print(f"[SFT Callback] Error on sample {i}: {e}")

        print(f"{'='*70}\n")
        model.train()

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
    processor.tokenizer.model_max_length = 4096

    processor.image_processor.min_pixels = 28 * 28 * 4     
    processor.image_processor.max_pixels = 28 * 28 * 256   

    peft_model = apply_lora_to_quantized_model(model_dir)
    
    # Load appropriate dataset
    if dataset_type == "minicot":
        sft_dataset = prepare_minicot_for_sft(train_data)
    else:
        sft_dataset = prepare_scienceqa_for_sft(train_data)

    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=5,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
        packing=False,
        save_strategy="steps",
        save_steps=200,
    )

    # Build logging callback using fixed sample items for format inspection
    sample_items = []
    if hasattr(sft_dataset, "get_sample_items"):
        sample_items = sft_dataset.get_sample_items(n=2)
    logging_callback = SFTOutputLoggingCallback(
        sample_items=sample_items,
        processor=processor,
        log_every=50,
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
        callbacks=[logging_callback],
    )

    trainer.train()
    
    print(f"\n[SFT] Saving format-aligned LoRA checkpoint to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[SFT] Format alignment complete. Ready for GRPO fine-tuning.\n")

if __name__ == "__main__":
    MODEL_DIR = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3" 
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
    raw_minicot = raw_minicot.filter(lambda x: len(str(x.get('solution', ''))) <= 800)
    
    train_sft_format_alignment(MODEL_DIR, raw_minicot, OUTPUT_DIR, dataset_type="minicot")