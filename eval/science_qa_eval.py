import torch
import pandas as pd
import io
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class VLMQEvaluator:
    def __init__(self, model_path, data_path, num_samples=50):
        self.model_path = model_path
        self.data_path = data_path
        self.num_samples = num_samples
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        print(f"Đang tải model nén từ: {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def load_test_data(self):
        df = pd.read_parquet(self.data_path)
        # Lọc các mẫu có ảnh
        mask = df['image'].notnull()
        df = df[mask].head(self.num_samples)
        return df

    @staticmethod
    def robust_science_qa_matcher(pred, target_letter):
        pred = str(pred).strip().upper()
        patterns = [f"{target_letter}.", f"({target_letter})", f" {target_letter} "]
        if any(p in f" {pred} " for p in patterns) or (len(pred) > 0 and pred[0] == target_letter):
            return 1.0
        return 0.0

    def evaluate(self):
        df = self.load_test_data()
        correct_count = 0
        total = len(df)

        print(f"\nBắt đầu đánh giá trên {total} mẫu ScienceQA...")
        
        for idx, row in tqdm(df.iterrows(), total=total):
            ans_idx = int(row['answer'])
            target_letter = self.choices_map[ans_idx]
            
            image_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            prompt_text = (
                f"Question: {row['question']}\n"
                f"Choices: {row['choices']}\n"
                "Think step by step and then provide the final answer letter."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pred_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            score = self.robust_science_qa_matcher(pred_text, target_letter)
            correct_count += score

            if idx == df.index[0]:
                print(f"\n[Ví dụ Output Thực Tế]")
                print(f"Target: {target_letter}")
                print(f"Model Gen: {pred_text}")
                print(f"Chấm điểm: {'ĐÚNG' if score else 'SAI'}\n")

        accuracy = (correct_count / total) * 100
        print("\n" + "="*50)
        print(f"ĐỘ CHÍNH XÁC CỦA MÔ HÌNH NÉN: {accuracy:.2f}%")
        print("="*50)

if __name__ == "__main__":
    MODEL_PATH = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3"
    DATA_PATH = r"./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet"
    
    evaluator = VLMQEvaluator(MODEL_PATH, DATA_PATH, num_samples=50)
    evaluator.evaluate()
