import os
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

class ModelDownloader:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", local_dir="./weights/Qwen2-VL-7B-Instruct"):
        self.model_id = model_id
        self.local_dir = local_dir

    def download(self):
        print(f"Bắt đầu tải: {self.model_id} -> {self.local_dir}")
        os.makedirs(self.local_dir, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt", "*.msgpack"]
        )
        print("Tải thành công!")

    def test_load_local(self):
        processor = AutoProcessor.from_pretrained(self.local_dir)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.local_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"Device: {model.device} | Dtype: {model.dtype}")
        return model, processor

if __name__ == "__main__":
    downloader = ModelDownloader()
    downloader.download()
    model, processor = downloader.test_load_local()
