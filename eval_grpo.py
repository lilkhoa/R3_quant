import sys
import os
import torch
import io
import gc
import re
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from peft import PeftModel

def build_scienceqa_prompt(question: str, choices: list) -> str:
    """Build the exact same prompt used during GRPO training."""
    prompt = f"Question: {question}\n"

    labels = ["A", "B", "C", "D", "E"]
    if choices:
        prompt += "Choices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}. {choice}\n"

    prompt += "\nStart your response directly with the <think> tag.\n"
    return prompt

def extract_answer(text: str) -> str:
    """
    Extract answer from text with primary and fallback methods.
    Primary: Look for <answer>X</answer> tag
    Fallback: Look for standalone letters A-E or letters in parentheses
    """
    # Primary: Search for <answer> tags
    match = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback 1: Look for letters in parentheses (A), (B), etc.
    match = re.search(r'\(([A-Ea-e])\)', text)
    if match:
        return match.group(1).upper()
    
    # Fallback 2: Look for standalone letters surrounded by whitespace or punctuation
    match = re.search(r'(?:^|\s|:|,|\.|\?|!|。|、|;)([A-Ea-e])(?:\s|:|,|\.|\?|!|。|、|;|$)', text)
    if match:
        return match.group(1).upper()
    
    return ""

def extract_thinking(text: str) -> str:
    """Extract reasoning content between <think> and </think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match_incomplete = re.search(r'<think>(.*)', text, re.IGNORECASE | re.DOTALL)
    if match_incomplete:
        return match_incomplete.group(1).strip() + " ...[BỊ CẮT CỤT]"
        
    return ""

def evaluate_model(model_path, df, lora_path=None, num_samples=None):
    """
    Evaluate model on ScienceQA dataset with GRPO prompt format.
    
    Returns:
        - accuracy: Accuracy percentage
        - predictions: List of full model outputs
        - thoughts: List of extracted reasoning from <think> tags
        - answers: List of extracted answers
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    if lora_path:
        print(f"  ✓ Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
        
    processor_path = lora_path if lora_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path)
    
    model.eval()

    correct = 0
    predictions = []
    thoughts = []
    answers = []

    eval_df = df if num_samples is None else df.select(range(num_samples))
    
    with torch.no_grad():
        for row in tqdm(eval_df, total=len(eval_df), desc=f"Evaluating {os.path.basename(model_path)}"):
            # Build exact prompt using training format
            if isinstance(row['choices'], list) or isinstance(row['choices'], np.ndarray):
                choices_list = list(row['choices'])
            else:
                choices_list = []
            
            text_content = build_scienceqa_prompt(row['question'], choices_list)
            
            content = [{"type": "text", "text": text_content}]
            
            # Handle image
            if 'image' in row and row['image'] is not None:
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_data = Image.open(io.BytesIO(img_data['bytes']))
                content.insert(0, {"type": "image", "image": img_data})

            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a logical reasoning AI. "
                        "You MUST think step-by-step and enclose your entire reasoning "
                        "within <think> and </think> tags. "
                        "After thinking, output your final answer (one letter only) "
                        "enclosed within <answer> and </answer> tags."
                    )
                },
                {
                    "role": "user", 
                    "content": content
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=model.dtype, device=model.device)
                else:
                    inputs[k] = v.to(device=model.device)

            # Generate with GRPO-compatible settings
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,          # Increased from 512 for long reasoning chains
                do_sample=False,               # Greedy decoding for stable predictions
                num_beams=1
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            output_upper = output_text.upper()
            
            # Extract components
            thinking = extract_thinking(output_text)
            predicted_answer = extract_answer(output_text)
            
            # Get target answer
            target_idx = int(row['answer'])
            target = chr(ord('A') + target_idx)
            
            # Check correctness
            if predicted_answer == target:
                correct += 1
            
            predictions.append(output_text)
            thoughts.append(thinking)
            answers.append(predicted_answer)

    accuracy = (correct / len(eval_df)) * 100 if len(eval_df) > 0 else 0.0
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, predictions, thoughts, answers

if __name__ == "__main__":
    # Paths
    BASE_UNQUANTIZED_PATH = r"./weights/Qwen2-VL-7B-Instruct"
    QUANTIZED_PATH = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3"
    GRPO_PATH = r"./r3_quant_checkpoints/"
    
    # Load dataset from Hugging Face
    NUM_SAMPLES = 100
    PREVIOUS_SAMPLES = 200
    LOCAL_DATA_PATH = r"./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet"

    print("Loading dataset...")
    df = load_dataset("parquet", data_files=LOCAL_DATA_PATH, split="train")
    
    # Use 200 random questions
    df = df.shuffle(seed=42).select(range(PREVIOUS_SAMPLES, PREVIOUS_SAMPLES + NUM_SAMPLES))
    
    print(f"Dataset loaded: {len(df)} samples\n")

    # Evaluate models
    print("="*70)
    print("MODEL EVALUATION (Qwen2-VL-7B)")
    print("="*70)
    
    # print("\n[1] Evaluating Base Model (Unquantized)...")
    # base_unquantized_acc, base_unquantized_preds, base_unquantized_thoughts, base_unquantized_answers = evaluate_model(
    #     BASE_UNQUANTIZED_PATH, df, lora_path=None, num_samples=NUM_SAMPLES
    # )
    
    print("\n[2] Evaluating Quantized Model (3-bit, No LoRA)...")
    quantized_acc, quantized_preds, quantized_thoughts, quantized_answers = evaluate_model(
        QUANTIZED_PATH, df, lora_path=None, num_samples=NUM_SAMPLES
    )
    
    print("\n[3] Evaluating Quantized + SFT + GRPO Model (with LoRA)...")
    grpo_acc, grpo_preds, grpo_thoughts, grpo_answers = evaluate_model(
        QUANTIZED_PATH, df, lora_path=GRPO_PATH, num_samples=NUM_SAMPLES
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    # print(f"1. Base Model (Unquantized):           {base_unquantized_acc:.2f}%")
    print(f"2. Quantized Model (3-bit, base):      {quantized_acc:.2f}%")
    print(f"3. GRPO Model (3-bit + SFT + GRPO):    {grpo_acc:.2f}%")
    print(f"Improvement (GRPO vs Quantized):       {grpo_acc - quantized_acc:+.2f}%")
    print("="*70)

    # Print sample outputs with full reasoning (no truncation)
    print("\n" + "="*70)
    print("DETAILED SAMPLE PREDICTIONS (Full Reasoning - 100% Preserved)")
    print("="*70)
    
    for i in range(min(3, len(df))):
        row = df[i]
        target_idx = int(row['answer'])
        target = chr(ord('A') + target_idx)
        
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*70}")
        print(f"Question: {row['question']}")
        print(f"Choices: {row['choices']}")
        print(f"✓ Ground Truth: {target}")
        
        # print(f"\n{'-'*70}")
        # print(f"BASE UNQUANTIZED MODEL RESPONSE:")
        # print(f"{'-'*70}")
        # print(f"Predicted Answer: {base_unquantized_answers[i] if base_unquantized_answers[i] else '[Could not extract]'}")
        # print(f"Correctness: {'✓ CORRECT' if base_unquantized_answers[i] == target else '✗ INCORRECT'}")
        # print(f"\n[FULL RESPONSE TEXT]:")
        # print(base_unquantized_preds[i])

        print(f"\n{'-'*70}")
        print(f"QUANTIZED MODEL RESPONSE (No LoRA):")
        print(f"{'-'*70}")
        print(f"Predicted Answer: {quantized_answers[i] if quantized_answers[i] else '[Could not extract]'}")
        print(f"Correctness: {'✓ CORRECT' if quantized_answers[i] == target else '✗ INCORRECT'}")
        print(f"\n[FULL RESPONSE TEXT]:")
        print(quantized_preds[i])
        
        print(f"\n{'-'*70}")
        print(f"QUANTIZED + SFT + GRPO MODEL RESPONSE:")
        print(f"{'-'*70}")
        print(f"Predicted Answer: {grpo_answers[i] if grpo_answers[i] else '[Could not extract]'}")
        print(f"Correctness: {'✓ CORRECT' if grpo_answers[i] == target else '✗ INCORRECT'}")
        print(f"\n[FULL RESPONSE TEXT]:")
        print(grpo_preds[i])
    
    print("\n" + "="*70)
