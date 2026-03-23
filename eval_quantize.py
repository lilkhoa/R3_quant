import sys
import os
import torch
import io
import gc
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

def evaluate_model(model_path, df):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    correct = 0
    predictions = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {os.path.basename(model_path)}"):
            text_content = f"Question: {row['question']}\nChoices: {row['choices']}\nAnswer with the option letter only."
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

            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            prediction = output_text.strip().upper()
            target_idx = int(row['answer'])
            target = chr(ord('A') + target_idx)

            if target in prediction:
                correct += 1
            
            predictions.append(prediction)

    accuracy = (correct / len(df)) * 100
    
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    
    return accuracy, predictions

if __name__ == "__main__":
    BASE_MODEL_PATH = r"./weights/Qwen2.5-VL-3B-Instruct"
    QUANTIZED_MODEL_PATH = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    NUM_SAMPLES = 500

    loader = ScienceQALocalLoader(DATA_PATH, subset_size=NUM_SAMPLES)
    df = loader.preprocess_for_r3_quant()

    print("\n--- ĐÁNH GIÁ MODEL GỐC (16-BIT) ---")
    base_acc, base_preds = evaluate_model(BASE_MODEL_PATH, df)

    print("\n--- ĐÁNH GIÁ MODEL LƯỢNG TỬ HÓA (3-BIT) ---")
    quant_acc, quant_preds = evaluate_model(QUANTIZED_MODEL_PATH, df)

    print("\n" + "="*60)
    print(f"KẾT QUẢ SO SÁNH TRỰC TIẾP ({NUM_SAMPLES} MẪU)")
    print("="*60)
    print(f"Base Model Accuracy : {base_acc:.2f}%")
    print(f"Quant Model Accuracy: {quant_acc:.2f}%")
    print(f"Độ chênh lệch       : {quant_acc - base_acc:.2f}%")
    print("="*60)

    print("\n--- CHI TIẾT 5 MẪU ĐẦU TIÊN ---")
    for i in range(min(5, NUM_SAMPLES)):
        row = df.iloc[i]
        target_idx = int(row['answer'])
        target = chr(ord('A') + target_idx)
        print(f"\nQ: {row['question'][:70]}...")
        print(f"Đáp án đúng : {target}")
        print(f"Base Model  : {base_preds[i]}")
        print(f"Quant Model : {quant_preds[i]}")