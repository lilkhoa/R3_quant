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
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        prompt += (
            "\nThink step by step and reason based on the image. "
            "Enclose your reasoning process within <think> </think> tags "
            "and provide your FINAL ANSWER within <answer> </answer> tags."
        )
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    valid_labels = labels[:len(choices)]
    if len(valid_labels) > 1:
        label_str = ", ".join(valid_labels[:-1]) + f" or {valid_labels[-1]}"
    else:
        label_str = valid_labels[0]
        
    prompt += (
        "\nThink step by step and reason based on the image. "
        "Enclose your reasoning process within <think> </think> tags "
        f"and provide your FINAL ANSWER (strictly write 1 letter: {label_str}) within <answer> </answer> tags."
    )
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
    match = re.search(r'(?:^|\s|:|,|\.|\?|!|。|、|；)([A-Ea-e])(?:\s|:|,|\.|\?|!|。|、|；|$)', text)
    if match:
        return match.group(1).upper()
    
    return ""

def extract_thinking(text: str) -> str:
    """Extract reasoning content between <think> and </think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
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
        torch_dtype=torch.bfloat16,
    )
    
    if lora_path:
        print(f"  ✓ Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
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
            if 'image' in row and pd.notna(row['image']):
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_data = Image.open(io.BytesIO(img_data['bytes']))
                content.insert(0, {"type": "image", "image": img_data})

            messages = [{"role": "user", "content": content}]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

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
    # Kaggle paths
    BASE_PATH = "./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3"
    GRPO_PATH = "./r3_quant_checkpoints"
    QUANTIZED_PATH = "./weights/Qwen2-VL-2B-Instruct-GPTQ-Int3"
    
    # Load dataset from Hugging Face
    NUM_SAMPLES = 500

    print("Loading dataset from Hugging Face...")
    df = load_dataset("derek-thomas/ScienceQA", split="test").select(range(NUM_SAMPLES))
    
    print(f"Dataset loaded: {len(df)} samples\n")

    # Evaluate models
    print("="*70)
    print("GRPO MODEL EVALUATION (Qwen2-VL-2B)")
    print("="*70)
    
    print("\n[1] Evaluating Base Model (No LoRA)...")
    base_acc, base_preds, base_thoughts, base_answers = evaluate_model(
        QUANTIZED_PATH, df, lora_path=None, num_samples=NUM_SAMPLES
    )
    
    print("\n[2] Evaluating GRPO-Finetuned Model (with LoRA)...")
    grpo_acc, grpo_preds, grpo_thoughts, grpo_answers = evaluate_model(
        QUANTIZED_PATH, df, lora_path=GRPO_PATH, num_samples=NUM_SAMPLES
    )

    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Base Model (Quantized 3-bit):        {base_acc:.2f}%")
    print(f"GRPO Model (3-bit + LoRA):          {grpo_acc:.2f}%")
    print(f"Improvement:                         {grpo_acc - base_acc:+.2f}%")
    print("="*70)

    # Print sample outputs with reasoning
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (with Reasoning)")
    print("="*70)
    
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        target_idx = int(row['answer'])
        target = chr(ord('A') + target_idx)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {row['question']}")
        print(f"✓ Correct Answer: {target}")
        print(f"\nBase Model:")
        print(f"  Predicted: {base_answers[i] if base_answers[i] else 'Could not extract'}")
        print(f"  Correct: {'✓' if base_answers[i] == target else '✗'}")
        if base_thoughts[i]:
            print(f"  Reasoning: {base_thoughts[i][:200]}...")
        
        print(f"\nGRPO Model:")
        print(f"  Predicted: {grpo_answers[i] if grpo_answers[i] else 'Could not extract'}")
        print(f"  Correct: {'✓' if grpo_answers[i] == target else '✗'}")
        if grpo_thoughts[i]:
            print(f"  Reasoning: {grpo_thoughts[i][:300]}...")
        else:
            print(f"  Reasoning: [No thinking found]")
    
    print("\n" + "="*70)
