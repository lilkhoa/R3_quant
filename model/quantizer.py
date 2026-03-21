import torch
import os
import sys
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    GPTQConfig, 
    AutoConfig
)

# Thêm đường dẫn để load dữ liệu local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

class QwenGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def get_calibration_data(self, test_size=8):
        print(f"--- Đang tải {test_size} mẫu dữ liệu calibration ---")
        loader = ScienceQALocalLoader(self.data_path, subset_size=test_size)
        df = loader.preprocess_for_r3_quant()
        # Doc: "You could also pass your own dataset as a list of strings"
        return [f"Question: {row['question']}\nAnswer: {row['reasoning']}" for _, row in df.iterrows()]

    def quantize_and_save(self, bits=3):
        # 1. Chuẩn bị dữ liệu
        calib_dataset = self.get_calibration_data(test_size=8) # Test thử 8 mẫu trước
        
        # 2. Cấu hình GPTQConfig (Theo đúng Doc)
        # Lưu ý: Doc ghi ExLlama chỉ hỗ trợ 4-bit, nên với 3-bit ta phải tắt use_exllama
        gptq_config = GPTQConfig(
            bits=bits, 
            dataset=calib_dataset, 
            tokenizer=self.base_model_path,
            use_exllama=False, # Bắt buộc False cho 3-bit
            desc_act=False,
            sym=True
        )

        print(f"--- Đang bắt đầu Quantization {bits}-bit ---")
        
        # 3. Vá lỗi use_cache (đặc thù của Qwen2.5-VL chưa có trong doc nhưng cần để chạy)
        config = AutoConfig.from_pretrained(self.base_model_path)
        config.use_cache = False

        try:
            # 4. Load model và thực hiện quantize (Theo Doc)
            # device_map="auto" giúp di chuyển các module giữa CPU/GPU để tối ưu RAM
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=gptq_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            # 5. Lưu model (Theo Doc: Nếu dùng device_map, nên đưa về CPU trước khi save)
            print(f"--- Đang di chuyển model về CPU để lưu an toàn ---")
            model.to("cpu")
            
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- Hoàn tất! Model lưu tại: {self.save_path} ---")

        except Exception as e:
            print(f"--- Lỗi Quantization: {e} ---")
            sys.exit(1)

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    SAVE_DIR = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    quantizer = QwenGPTQQuantizer(BASE_MODEL, SAVE_DIR, DATA_PATH)
    quantizer.quantize_and_save(bits=3)