import torch
import io
from PIL import Image
from datasets import Dataset

class ScienceQAGRPODataset(torch.utils.data.Dataset):
    """
    Custom dataset for GRPO training that properly handles images.
    Images are loaded on-the-fly to avoid serialization issues with PIL.
    """
    def __init__(self, raw_dataset, max_samples=None):
        self.items = []
        self.labels = ["A", "B", "C", "D", "E"]
        count = 0
        
        for item in raw_dataset:
            if max_samples and count >= max_samples:
                break
            if item["image"] is None:
                continue
            
            self.items.append({
                'image': item["image"],
                'question': item["question"],
                'choices': item["choices"],
                'answer': item["answer"]
            })
            count += 1
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        try:
            # Convert image to PIL on access (not stored in dataset)
            pil_image = _convert_image_to_pil(item['image'])
            
            # Ensure RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Build prompt text
            text_prompt = build_scienceqa_prompt(item['question'], item['choices'])
            
            # Build conversational format for VLM training.
            # FIX Bug #4: Add system message matching eval_grpo.py's format.
            # Without this, the 3-bit model has no instruction to produce
            # <think>/<answer> tags, so format_reward_func gets 0.0 constantly.
            SYSTEM_MESSAGE = (
                "You are a logical reasoning AI. "
                "You MUST think step-by-step and enclose your entire reasoning "
                "within <think> and </think> tags. "
                "After thinking, output your final answer (one letter only) "
                "enclosed within <answer> and </answer> tags."
            )

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]

            correct_letter = self.labels[item['answer']]

            return {
                "prompt": messages,      # Conversational format
                "images": [pil_image],   # Must be a list, even for single image
                "ground_truth": correct_letter
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return valid dummy item instead of failing
            dummy_image = Image.new('RGB', (224, 224), color='white')
            return {
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are a logical reasoning AI. You MUST think step-by-step and strictly enclose your entire reasoning process within <think> and </think> tags. After thinking, you MUST output your final answer enclosed within <answer> and </answer> tags."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe the image."}
                        ]
                    }
                ],
                "images": [dummy_image],  # Must be a list
                "ground_truth": "A"
            }

def build_scienceqa_prompt(question: str, choices: list) -> str:
    # FIX (Fix E): Removed the verbose few-shot example that was embedded here.
    # The system message already instructs the model on the <think>/<answer> format,
    # so repeating a full example in the user turn wastes ~80 tokens per question.
    # Keeping it minimal: question + choices + "Response:" cue.
    prompt = f"Question: {question}\n"

    labels = ["A", "B", "C", "D", "E"]
    if choices:
        prompt += "Choices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}. {choice}\n"

    prompt += "\nResponse:\n"
    return prompt


def _convert_image_to_pil(image_data):
    """Convert image data from HuggingFace format to PIL Image."""
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, dict):
        if 'bytes' in image_data:
            return Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
        elif 'path' in image_data:
            return Image.open(image_data['path']).convert('RGB')
    elif isinstance(image_data, str):
        return Image.open(image_data).convert('RGB')
    return image_data

def prepare_scienceqa_for_grpo(raw_dataset, processor, max_samples=None):
    """
    Prepare dataset for GRPO training using custom dataset class.
    The processor is stored for later formatting via chat template.
    """
    dataset = ScienceQAGRPODataset(raw_dataset, max_samples=max_samples)
    if len(dataset) == 0:
        print("Warning: Dataset is empty after processing. This may cause training to fail.")
    return dataset

class ScienceQASFTDataset(torch.utils.data.Dataset):
    """
    Custom dataset for SFT training that properly handles images.
    Images are loaded on-the-fly to avoid serialization issues with PIL.
    """
    def __init__(self, raw_dataset, max_samples=None):
        self.items = []
        self.labels = ["A", "B", "C", "D", "E"]
        count = 0
        
        for item in raw_dataset:
            if max_samples and count >= max_samples:
                break
            if item["image"] is None:
                continue
            
            self.items.append({
                'image': item["image"],
                'question': item["question"],
                'choices': item["choices"],
                'answer': item["answer"],
                'solution': item.get("solution", "Reasoning based on the image.")
            })
            count += 1
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Convert image to PIL on access (not stored in dataset)
        pil_image = _convert_image_to_pil(item['image'])
        
        # Build prompt
        text_prompt = build_scienceqa_prompt(item['question'], item['choices'])
        
        # Build messages
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]
        }
        
        correct_letter = self.labels[item['answer']]
        solution_text = item['solution']
        
        assistant_text = (
            f"<think>\n{solution_text}\n</think>\n"
            f"<answer>{correct_letter}</answer>"
        )
        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text}
            ]
        }
        
        return {
            "messages": [user_message, assistant_message],
            "images": [pil_image]
        }

def prepare_scienceqa_for_sft(raw_dataset, max_samples=None):
    """
    Prepare dataset for SFT training using custom dataset class.
    Returns a dataset that properly handles images without serialization issues.
    """
    return ScienceQASFTDataset(raw_dataset, max_samples=max_samples)