import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import gc
import os

def export_model_info(model_path, name, output_file):
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
    
    print("🔬 CẤU TRÚC TỔNG QUAN (Vui lòng xem chi tiết trong file text):")
    print(model)
    
    print(f"\n📝 Đang xuất chi tiết layer vào file: {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"CHI TIẾT KIẾN TRÚC MODEL: {name}\n")
        f.write("="*80 + "\n\n")
        
        for module_name, module in model.named_modules():
            has_params = len(list(module.named_parameters(recurse=False))) > 0
            has_buffers = len(list(module.named_buffers(recurse=False))) > 0
            
            if has_params or has_buffers:
                f.write(f"[{type(module).__name__}] {module_name}\n")
                
                for param_name, param in module.named_parameters(recurse=False):
                    f.write(f"   ├── Param : {param_name:<10} | shape={str(list(param.shape)):<20} | dtype={param.dtype}\n")
                
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    f.write(f"   ├── Buffer: {buffer_name:<10} | shape={str(list(buffer.shape)):<20} | dtype={buffer.dtype}\n")
                
                f.write("-" * 80 + "\n")
                
    print("✅ Đã lưu xong!")
    
    del model
    gc.collect()

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    
    export_model_info(BASE_MODEL, "BẢN GỐC (16-BIT)", "arch_base_16bit.txt")