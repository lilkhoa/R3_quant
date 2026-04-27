import os
import subprocess
from huggingface_hub import snapshot_download
from datasets import load_dataset

def setup_environment():
    print("--- 1. Khởi tạo cấu trúc thư mục ---")
    directories = ["data/document_vqa", "data/mini_cot", "weights"]
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

def download_docvqa_data():
    print("\n--- 2. Đang tải Dataset DocumentVQA từ Hugging Face ---")
    target_path = "./data/document_vqa/train-00000-of-00038.parquet"
    if not os.path.exists(target_path):
        print("Downloading HuggingFaceM4/DocumentVQA (train split, first shard)...")
        # Load only first 200 rows — enough for GPTQ calibration, avoids downloading 9 GB
        dataset = load_dataset(
            "HuggingFaceM4/DocumentVQA",
            split="train",
            streaming=False,
        ).select(range(200))
        os.makedirs("./data/document_vqa", exist_ok=True)
        dataset.to_parquet(target_path)
        print(f"Đã lưu dataset tại: {target_path}")
    else:
        print("DocumentVQA dataset đã tồn tại, bỏ qua bước tải.")


    print("\n--- 2.5 Đang tải Dataset mini_cot_8k_verified cho SFT Training ---")
    try:
        dataset = load_dataset("luodian/mini_cot_8k_verified")
        
        target_dir = "./data/mini_cot"
        target_path = os.path.join(target_dir, "mini_cot_train.parquet")
        
        if not os.path.exists(target_path):
            print(f"Đang tải mini_cot_8k_verified dataset (quá trình này có thể lâu)...")
            # Save the default split (usually 'train')
            dataset_split = dataset['train'] if 'train' in dataset else dataset
            dataset_split.to_parquet(target_path)
            print(f"Đã lưu SFT dataset tại: {target_path}")
        else:
            print("SFT dataset đã tồn tại, bỏ qua bước tải.")
    except Exception as e:
        print(f"[WARNING] Không thể tải mini_cot dataset: {e}")
        print("SFT training sẽ tải từ Hugging Face trực tiếp.")

def download_model():
    print("\n--- 3. Đang tải Model Qwen2-VL-7B-Instruct ---")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    local_dir = "./weights/Qwen2-VL-7B-Instruct"
    
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
            # Pass GPTQ/exllama env vars at the subprocess level so they are
            # in the process environment BEFORE Lightning AI's /commands/python
            # wrapper loads any native C extensions.
            env = os.environ.copy()
            env["DISABLE_EXLLAMA"]        = "1"
            env["GPTQ_DISABLE_EXLLAMAV2"] = "1"
            env["USE_EXLLAMA"]            = "0"
            # Prevent flash_attn from loading (SIGILL on some VMs)
            env["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

            result = subprocess.run(["python", script_path], env=env, check=True)
            if result.returncode == 0:
                print("\n[SUCCESS] Quá trình lượng tử hóa hoàn tất thành công!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Script quantizer.py gặp lỗi: {e}")
    else:
        print(f"[ERROR] Không tìm thấy file {script_path}")

if __name__ == "__main__":
    setup_environment()
    download_docvqa_data()
    # download_sft_data()
    download_model()
    run_quantizer()