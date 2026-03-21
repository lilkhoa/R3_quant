import sys
import os
import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, GPTQConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

class QwenGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def get_calibration_data(self, test_size=8):
        """
        Lấy một lượng nhỏ dữ liệu để test (mặc định là 8 mẫu).
        """
        print(f"--- Đang tải dữ liệu calibration (Test size: {test_size}) ---")
        loader = ScienceQALocalLoader(self.data_path, subset_size=test_size)
        df = loader.preprocess_for_r3_quant()
        
        calib_texts = []
        for _, row in df.iterrows():
            text = f"Question: {row['question']}\nAnswer: {row['reasoning']}"
            calib_texts.append(text)
        
        print(f"--- Đã chuẩn bị xong {len(calib_texts)} mẫu dữ liệu ---")
        return calib_texts

    def quantize_and_save(self, bits=3, test_mode=True):
        num_samples = 8 if test_mode else 128
        calib_dataset = self.get_calibration_data(test_size=num_samples)
        
        config = AutoConfig.from_pretrained(self.base_model_path)
        if not hasattr(config, 'use_cache'):
            config.use_cache = False

        print(f"--- Bắt đầu cấu hình GPTQ ({bits}-bit) ---")
        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calib_dataset,
            group_size=128,
            desc_act=False,
            sym=True,
            tokenizer=self.base_model_path
        )

        print(f"--- Đang tải model (Cố gắng ép vào 1 GPU duy nhất để tránh meta device)... ---")
        try:
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config, 
                quantization_config=quantization_config,
                device_map="cuda:0", 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            print(f"--- Lượng tử hóa thành công! Đang lưu tại: {self.save_path} ---")
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)
            print("--- Hoàn tất quá trình lưu model ---")
            
        except Exception as e:
            print(f"--- Lỗi trong quá trình xử lý: {e} ---")

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    SAVE_DIR = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3-Test"
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    quantizer = QwenGPTQQuantizer(BASE_MODEL, SAVE_DIR, DATA_PATH)
    
    quantizer.quantize_and_save(bits=3, test_mode=True)