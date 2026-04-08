import sys
import os
import torch
import io
import gc
import re
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

def evaluate_model(model_path, df, lora_path=None):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    if lora_path:
        print(f" -> Đang cấy ghép LoRA từ: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
    processor_path = lora_path if lora_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path)
    
    model.eval()

    correct = 0
    predictions = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {os.path.basename(model_path)}"):
            choices_str = ""
            labels = ["A", "B", "C", "D", "E"]
            if isinstance(row['choices'], list) or isinstance(row['choices'], np.ndarray):
                for i, c in enumerate(row['choices']):
                    choices_str += f"{labels[i]}. {c}\n"
            else:
                choices_str = str(row['choices'])
                
            text_content = (
                f"{row['question']}\n\nChoices:\n{choices_str}\n"
                "Think step by step and reason based on the image. "
                "Enclose your reasoning process within <think> </think> tags "
                "and provide your FINAL ANSWER within <answer> </answer> tags."
            )
            
            content = [{"type": "text", "text": text_content}]
            
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

            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            prediction = output_text.strip().upper()
            target_idx = int(row['answer'])
            target = chr(ord('A') + target_idx)
            match = re.search(r'<answer>\s*([A-E])\s*</answer>', prediction, re.IGNORECASE)
            if match:
                extracted_pred = match.group(1).upper()
            else:
                extracted_pred = prediction 
                
            if target in extracted_pred:
                correct += 1
            
            predictions.append(prediction)

    accuracy = (correct / len(df)) * 100
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, predictions

if __name__ == "__main__":
    BASE_MODEL_PATH = r"./weights/Qwen2-VL-7B-Instruct"
    QUANTIZED_MODEL_PATH = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3"
    SFT_MODEL_PATH = r"./sft_baseline_checkpoints" 
    
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    NUM_SAMPLES = 500

    loader = ScienceQALocalLoader(DATA_PATH, subset_size=NUM_SAMPLES)
    df = loader.preprocess_for_r3_quant()

    print("\n--- [1] ĐÁNH GIÁ MODEL GỐC (16-BIT) ---")
    base_acc, base_preds = evaluate_model(BASE_MODEL_PATH, df)

    print("\n--- [2] ĐÁNH GIÁ MODEL LƯỢNG TỬ HÓA (3-BIT) ---")
    quant_acc, quant_preds = evaluate_model(QUANTIZED_MODEL_PATH, df)
    
    print("\n--- [3] ĐÁNH GIÁ MODEL SFT (INT3 + LoRA) ---")
    sft_acc, sft_preds = evaluate_model(QUANTIZED_MODEL_PATH, df, lora_path=SFT_MODEL_PATH)

    print("\n" + "="*60)
    print(f"BẢNG VÀNG THÀNH TÍCH ({NUM_SAMPLES} MẪU)")
    print("="*60)
    print(f"1. Base Model (16-bit)      : {base_acc:.2f}%")
    print(f"2. Quantized Model (3-bit)  : {quant_acc:.2f}%")
    print(f"3. SFT Model (3-bit + LoRA) : {sft_acc:.2f}%")
    print("-" * 60)
    print(f"SFT đã phục hồi được        : {sft_acc - quant_acc:.2f}% so với bản 3-bit bị lỗi")
    print("="*60)

    print("\n--- CHI TIẾT 3 MẪU ĐẦU TIÊN ---")
    for i in range(min(3, NUM_SAMPLES)):
        row = df.iloc[i]
        target_idx = int(row['answer'])
        target = chr(ord('A') + target_idx)
        print(f"\nQ: {row['question'][:70]}...")
        print(f"🎯 Đáp án đúng : {target}")
        print(f"🤖 Base 16-bit : {base_preds[i][:50]}...")
        print(f"🥴 Quant 3-bit : {quant_preds[i][:50]}...")
        print(f"🧠 SFT Model   : {sft_preds[i][:150]}...") 
