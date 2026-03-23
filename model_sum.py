import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import os

def print_model_info(model_path, name):
    print(f"\n{'='*60}")
    print(f"🔍 ĐANG SOI CHI TIẾT MODEL: {name}")
    print(f"{'='*60}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cpu", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    mem_bytes = model.get_memory_footprint()
    print(f"📦 Dung lượng bộ nhớ : {mem_bytes / (1024**3):.2f} GB")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Tổng số tham số   : {total_params / 1e9:.2f} Tỷ (Billion)\n")
    
    print("🔬 CẤU TRÚC 3 LỚP TÍNH TOÁN ĐẦU TIÊN:")
    
    count = 0
    for module_name, module in model.named_modules():
        if "Linear" in str(type(module)):
            print(f"  - Tên module: {module_name}")
            print(f"    Loại      : {type(module).__name__}")
            
            for param_name, param in module.named_parameters():
                print(f"    + {param_name}: shape={list(param.shape)}, dtype={param.dtype}")
            print("-" * 40)
            
            count += 1
            if count >= 3:
                break
                
    del model
    import gc
    gc.collect()

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    QUANT_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    
    print_model_info(BASE_MODEL, "BẢN GỐC (16-BIT)")
    print_model_info(QUANT_MODEL, "BẢN QUANTIZED (3-BIT)")