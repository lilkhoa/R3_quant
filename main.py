import os
import subprocess
from huggingface_hub import snapshot_download
from datasets import load_dataset

def setup_environment():
    print("--- 1. Khởi tạo cấu trúc thư mục ---")
    directories = ["data/science_qa", "weights"]
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")

def download_data():
    print("\n--- 2. Đang tải Dataset ScienceQA từ Hugging Face ---")
    dataset = load_dataset("derek-thomas/ScienceQA", split="validation")
    
    target_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    if not os.path.exists(target_path):
        dataset.to_parquet(target_path)
        print(f"Đã lưu dataset tại: {target_path}")
    else:
        print("Dataset đã tồn tại, bỏ qua bước tải.")

def download_model():
    print("\n--- 3. Đang tải Model Qwen2.5-VL-3B-Instruct ---")
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_dir = "./weights/Qwen2.5-VL-3B-Instruct"
    
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"Đang tải model {model_id} (quá trình này có thể lâu)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Model đã được tải về: {local_dir}")
    else:
        print("Model đã tồn tại trong folder weights.")

def run_quantizer():
    print("\n--- 4. Bắt đầu chạy file quantizer.py ---")
    script_path = "model/quantizer.py"
    
    if os.path.exists(script_path):
        try:
            result = subprocess.run(["python", script_path], check=True)
            if result.returncode == 0:
                print("\n[SUCCESS] Quá trình lượng tử hóa hoàn tất thành công!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script quantizer.py gặp lỗi: {e}")
    else:
        print(f"[ERROR] Không tìm thấy file {script_path}")

if __name__ == "__main__":
    setup_environment()
    download_data()
    download_model()
    run_quantizer()