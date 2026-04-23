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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.utils import build_pope_prompt, extract_pope_answer, compute_pope_metrics
from src.utils import build_chartqa_prompt, extract_chartqa_answer, chartqa_relaxed_correct

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

def evaluate_model(model_path, df, lora_path=None, num_samples=None, blind_image=False):
    """
    Evaluate model on ScienceQA dataset with GRPO prompt format.
    
    Returns:
        - accuracy: Accuracy percentage
        - predictions: List of full model outputs
        - thoughts: List of extracted reasoning from <think> tags
        - answers: List of extracted answers
    """
    model_path = os.path.abspath(model_path)
    if lora_path:
        lora_path = os.path.abspath(lora_path)

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
            if not blind_image and 'image' in row and row['image'] is not None:
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

def evaluate_model_for_chartQA(model_path, df, lora_path=None, num_samples=None):
    """
    Evaluate model on the lmms-lab/ChartQA benchmark (Masry et al., 2022).

    Dataset columns (lmms-lab/ChartQA):
        image         - chart image
        question      - question about the chart
        answer        - ground-truth answer (free-form: number or short text)
        question_type - "human" | "augmented"

    Scoring uses relaxed accuracy:
        - Numeric  : correct when |pred - gt| / max(|gt|, 1e-8) <= 5%
        - Text     : case-insensitive exact match after stripping punctuation
    Results are broken down by question_type.

    Returns:
        - accuracy:    Overall relaxed accuracy (%)
        - predictions: List of full raw model output strings
        - thoughts:    List of extracted <think> reasoning strings
        - answers:     List of extracted predicted answer strings
    """
    model_path = os.path.abspath(model_path)
    if lora_path:
        lora_path = os.path.abspath(lora_path)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if lora_path:
        print(f"  \u2713 Loading LoRA from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    processor_path = lora_path if lora_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path)

    model.eval()

    correct       = 0
    predictions   = []
    thoughts      = []
    answers       = []
    ground_truths = []
    q_types       = []

    eval_df = df if num_samples is None else df.select(range(num_samples))

    with torch.no_grad():
        for row in tqdm(eval_df, total=len(eval_df),
                        desc=f"Evaluating ChartQA {os.path.basename(model_path)}"):
            question     = str(row.get("question", ""))
            ground_truth = str(row.get("answer", "")).strip()
            q_type       = str(row.get("question_type", "unknown"))

            text_content = build_chartqa_prompt(question)

            # ChartQA always requires the chart image
            content = []
            img_data = row.get("image", None)
            if img_data is not None:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_data = Image.open(io.BytesIO(img_data["bytes"]))
                content.append({"type": "image", "image": img_data})
            content.append({"type": "text", "text": text_content})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a logical reasoning AI specialized in chart analysis. "
                        "You MUST think step-by-step and enclose your entire reasoning "
                        "within <think> and </think> tags. "
                        "After thinking, output your final answer (a number or short text) "
                        "enclosed within <answer> and </answer> tags."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            thinking         = extract_thinking(output_text)
            predicted_answer = extract_chartqa_answer(output_text)

            if chartqa_relaxed_correct(predicted_answer, ground_truth):
                correct += 1

            predictions.append(output_text)
            thoughts.append(thinking)
            answers.append(predicted_answer)
            ground_truths.append(ground_truth)
            q_types.append(q_type)

    total    = len(eval_df)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    print(f"\n  [ChartQA] Overall relaxed accuracy: {accuracy:.2f}% ({correct}/{total})")

    unique_types = sorted(set(q_types))
    if len(unique_types) > 1:
        print(f"  [ChartQA] Per question_type breakdown:")
        for qt in unique_types:
            idx    = [i for i, t in enumerate(q_types) if t == qt]
            qt_correct = sum(
                1 for i in idx
                if chartqa_relaxed_correct(answers[i], ground_truths[i])
            )
            print(f"    {qt:12s}: {qt_correct/len(idx)*100:.2f}%  ({qt_correct}/{len(idx)})")

    print(f"\n  [ChartQA] Sample predictions:")
    for i in range(min(3, len(predictions))):
        ok = chartqa_relaxed_correct(answers[i], ground_truths[i])
        mark = "\u2713" if ok else "\u2717"
        print(f"    [{mark}] Sample {i+1} | type={q_types[i]}")
        print(f"         Question    : {eval_df[i].get('question', '')}")
        print(f"         Ground Truth: {ground_truths[i]}")
        print(f"         Predicted   : {answers[i] if answers[i] else '[not extracted]'}")
        print(f"         Full output : {predictions[i][:300]}{'...' if len(predictions[i]) > 300 else ''}")
        print()

    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    return accuracy, predictions, thoughts, answers

def evaluate_model_for_pope(model_path, df, lora_path=None, num_samples=None):
    """
    Evaluate model on the POPE benchmark (Li et al., EMNLP 2023).

    POPE dataset columns (lmms-lab/POPE):
        question_id  - unique id
        image        - chart/scene image
        question     - "Is there a X in the image?"
        answer       - ground-truth "yes" or "no"
        category     - "adversarial" | "popular" | "random"

    Metrics (from src.utils.compute_pope_metrics):
        accuracy, precision, recall, F1, yes_ratio
    Results are reported both per-category and overall.

    Returns:
        - accuracy:    Overall accuracy (%)
        - predictions: List of full raw model output strings
        - thoughts:    List of extracted <think> reasoning strings
        - answers:     List of extracted 'yes'/'no' predicted answers
    """
    model_path = os.path.abspath(model_path)
    if lora_path:
        lora_path = os.path.abspath(lora_path)

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

    predictions = []
    thoughts    = []
    answers     = []
    ground_truths = []
    categories  = []

    eval_df = df if num_samples is None else df.select(range(num_samples))

    with torch.no_grad():
        for row in tqdm(eval_df, total=len(eval_df), desc=f"Evaluating POPE {os.path.basename(model_path)}"):
            question     = str(row.get("question", ""))
            ground_truth = str(row.get("answer", "")).strip().lower()
            category     = str(row.get("category", "unknown"))

            text_content = build_pope_prompt(question)

            # POPE always requires the image (hallucination probe)
            content = []
            img_data = row.get("image", None)
            if img_data is not None:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_data = Image.open(io.BytesIO(img_data["bytes"]))
                content.append({"type": "image", "image": img_data})
            content.append({"type": "text", "text": text_content})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a logical reasoning AI. "
                        "You MUST think step-by-step and enclose your entire reasoning "
                        "within <think> and </think> tags. "
                        "After thinking, output your final answer (yes or no) "
                        "enclosed within <answer> and </answer> tags."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            thinking         = extract_thinking(output_text)
            predicted_answer = extract_pope_answer(output_text)

            predictions.append(output_text)
            thoughts.append(thinking)
            answers.append(predicted_answer)
            ground_truths.append(ground_truth)
            categories.append(category)

    # ── Overall metrics ────────────────────────────────────────────────────
    overall = compute_pope_metrics(answers, ground_truths)
    accuracy = overall['accuracy']

    print(f"\n  [POPE] Overall results ({len(eval_df)} samples):")
    print(f"    Accuracy : {overall['accuracy']:.2f}%")
    print(f"    Precision: {overall['precision']:.2f}%")
    print(f"    Recall   : {overall['recall']:.2f}%")
    print(f"    F1 Score : {overall['f1']:.2f}%")
    print(f"    Yes Ratio: {overall['yes_ratio']:.2f}%")

    # ── Per-category breakdown ──────────────────────────────────────────────
    unique_cats = sorted(set(categories))
    if len(unique_cats) > 1:
        print(f"\n  [POPE] Per-category breakdown:")
        for cat in unique_cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            cat_preds = [answers[i]       for i in idx]
            cat_gts   = [ground_truths[i] for i in idx]
            m = compute_pope_metrics(cat_preds, cat_gts)
            print(f"    {cat:12s} | Acc {m['accuracy']:.2f}%  Prec {m['precision']:.2f}%  "
                  f"Rec {m['recall']:.2f}%  F1 {m['f1']:.2f}%  Yes {m['yes_ratio']:.2f}%")

    # ── Sample log (first 3 items) ──────────────────────────────────────────
    print(f"\n  [POPE] Sample predictions:")
    for i in range(min(3, len(predictions))):
        correct_mark = "✓" if answers[i] == ground_truths[i] else "✗"
        print(f"    [{correct_mark}] Sample {i+1} | category={categories[i]}")
        print(f"         Question  : {eval_df[i].get('question', '')}")
        print(f"         Ground Truth: {ground_truths[i]}")
        print(f"         Predicted   : {answers[i] if answers[i] else '[not extracted]'}")
        print(f"         Full output : {predictions[i][:300]}{'...' if len(predictions[i]) > 300 else ''}")
        print()

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
    NUM_SAMPLES = 300
    PREVIOUS_SAMPLES = 0
    LOCAL_DATA_PATH = r"./data/chart_qa/test-00000-of-00001.parquet"

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
    # base_unquantized_acc, base_unquantized_preds, base_unquantized_thoughts, base_unquantized_answers = evaluate_model_for_pope(
    #     BASE_UNQUANTIZED_PATH, df, lora_path=None, num_samples=NUM_SAMPLES
    # )
    
    print("\n[2] Evaluating Quantized Model (3-bit, No LoRA)...")
    quantized_acc, quantized_preds, quantized_thoughts, quantized_answers = evaluate_model_for_chartQA(
        QUANTIZED_PATH, df, lora_path=None, num_samples=NUM_SAMPLES
    )
    
    print("\n[3] Evaluating Quantized + SFT + GRPO Model (with LoRA)...")
    grpo_acc, grpo_preds, grpo_thoughts, grpo_answers = evaluate_model_for_chartQA(
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
        target = str(row['answer']).strip().lower()

        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*70}")
        print(f"Question: {row['question']}")
        print(f"Category: {row.get('category', 'N/A')}")
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
