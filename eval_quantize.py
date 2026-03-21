import sys
import os
import torch
import io
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

def run_eval(model_path, data_path, num_samples=20):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_path)

    loader = ScienceQALocalLoader(data_path, subset_size=num_samples)
    df = loader.preprocess_for_r3_quant()

    correct = 0
    results = []
    model.eval()

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
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
            target = str(row['answer']).upper()

            if target in prediction:
                correct += 1
            
            results.append({
                "question": row['question'],
                "target": target,
                "predict": prediction
            })

    accuracy = (correct / num_samples) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    
    return results

if __name__ == "__main__":
    QUANTIZED_MODEL_PATH = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    eval_results = run_eval(QUANTIZED_MODEL_PATH, DATA_PATH, num_samples=20)
    
    for res in eval_results[:3]:
        print(f"\nQ: {res['question'][:100]}...")
        print(f"Target: {res['target']} | Predict: {res['predict']}")