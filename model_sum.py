import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import os

def print_model_info(model_path, name):
    print(f"\n{'='*60}")
    print(f"🔍 ĐANG SOI CHI TIẾT MODEL: {name}")
    print(f"{'='*60}")
    
    # Load model lên CPU để không tốn VRAM lúc soi
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cpu", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 1. Xem dung lượng thực tế model chiếm dụng
    mem_bytes = model.get_memory_footprint()
    print(f"📦 Dung lượng bộ nhớ : {mem_bytes / (1024**3):.2f} GB")
    
    # 2. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Tổng số tham số   : {total_params / 1e9:.2f} Tỷ (Billion)\n")
    
    # 3. Quét tự động để tìm cấu trúc lớp ẩn
    print("🔬 CẤU TRÚC 3 LỚP TÍNH TOÁN ĐẦU TIÊN:")
    
    count = 0
    # named_modules() sẽ duyệt qua tất cả các ngóc ngách của model
    for module_name, module in model.named_modules():
        # Bắt các class có chứa chữ "Linear" (Linear gốc hoặc QuantLinear)
        if "Linear" in str(type(module)):
            print(f"  - Tên module: {module_name}")
            print(f"    Loại      : {type(module).__name__}")
            
            # In ra thông tin các tensor (trọng số) bên trong module đó
            for param_name, param in module.named_parameters():
                print(f"    + {param_name}: shape={list(param.shape)}, dtype={param.dtype}")
            print("-" * 40)
            
            count += 1
            if count >= 3: # Chỉ in 3 module để không bị trôi màn hình
                break
                
    # Dọn dẹp RAM trước khi load model tiếp theo
    del model
    import gc
    gc.collect()

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    QUANT_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    
    print_model_info(BASE_MODEL, "BẢN GỐC (16-BIT)")
    print_model_info(QUANT_MODEL, "BẢN QUANTIZED (3-BIT)")