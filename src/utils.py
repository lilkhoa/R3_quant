import torch
import io
from PIL import Image
from datasets import Dataset
from typing import List, Dict, Any
import warnings

class GRPODataCollator:
    """
    Data collator for GRPO training that properly processes images and text.
    Handles batch processing of images and tokenization.
    """
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples into model inputs.
        Handles PIL images and message structures properly.
        """
        if not batch:
            raise ValueError("Empty batch received by data collator")
        
        # Filter out None prompts and ensure valid batch
        valid_batch = []
        for item in batch:
            if item.get('prompt') is not None and item.get('prompt'):
                valid_batch.append(item)
        
        # If batch is empty after filtering, skip processing
        if not valid_batch:
            raise ValueError("No valid prompts in batch after filtering")
        
        # Extract images and text from prompts
        images = []
        texts = []
        
        for item in valid_batch:
            prompt = item['prompt']
            if isinstance(prompt, list):
                # Extract image from first message
                for msg in prompt:
                    if msg.get('role') == 'user' and isinstance(msg.get('content'), list):
                        for content_item in msg['content']:
                            if content_item.get('type') == 'image':
                                img = content_item.get('image')
                                if img is not None:
                                    if not isinstance(img, Image.Image):
                                        img = _convert_image_to_pil(img)
                                    # Ensure image is in RGB mode
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    images.append(img)
                            elif content_item.get('type') == 'text':
                                texts.append(content_item.get('text', ''))
        
        # Ensure we have matching images and texts
        if not images or not texts:
            raise ValueError(f"Mismatch: {len(images)} images, {len(texts)} texts")
        
        # Ensure we have at least one sample
        min_count = min(len(images), len(texts))
        if min_count == 0:
            raise ValueError("No valid image-text pairs found")
        
        # Process through processor
        try:
            # Prepare text and images for processor, ensuring they match
            texts = texts[:min_count]
            images = images[:min_count]
            
            # Validate text inputs - ensure no empty strings
            texts = [t if t.strip() else "What is shown in this image?" for t in texts]
            
            # Use processor to handle both images and text
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                processed = self.processor(
                    text=texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors='pt'
                )
            
            # Ensure all tensors have at least 1 token
            if 'input_ids' in processed:
                seq_len = processed['input_ids'].shape[1] if processed['input_ids'].dim() > 1 else 0
                if seq_len == 0:
                    raise ValueError("Empty input_ids generated after processing")
            
            # Validate tensor shapes
            for key in processed:
                if isinstance(processed[key], torch.Tensor):
                    if processed[key].numel() == 0:
                        raise ValueError(f"Empty tensor generated for key '{key}'")
                    # Ensure batch dimension exists
                    if processed[key].dim() < 2:
                        processed[key] = processed[key].unsqueeze(0)
                    # Set proper dtype
                    if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                        processed[key] = processed[key].to(torch.long)
                    else:
                        processed[key] = processed[key].to(torch.bfloat16)
            
            return processed
        except Exception as e:
            print(f"Error in data collation: {e}")
            raise

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
            
            # Build prompt
            text_prompt = build_scienceqa_prompt(item['question'], item['choices'])
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image}, 
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
            
            correct_letter = self.labels[item['answer']]
            
            return {
                "prompt": messages,
                "ground_truth": correct_letter
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return {
                "prompt": None,
                "ground_truth": "A"
            }

def build_scienceqa_prompt(question: str, choices: list) -> str:
    """Build a formatted prompt for ScienceQA questions."""
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        prompt += (
            "\nThink step by step and reason based on the image. "
            "Enclose your reasoning process within <think> </think> tags "
            "and provide your FINAL ANSWER within <answer> </answer> tags."
        )
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    valid_labels = labels[:len(choices)]
    if len(valid_labels) > 1:
        label_str = ", ".join(valid_labels[:-1]) + f" or {valid_labels[-1]}"
    else:
        label_str = valid_labels[0]
        
    prompt += (
        "\nThink step by step and reason based on the image. "
        "Enclose your reasoning process within <think> </think> tags "
        f"and provide your FINAL ANSWER (strictly write 1 letter: {label_str}) within <answer> </answer> tags."
    )
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

def prepare_scienceqa_for_grpo(raw_dataset, max_samples=None):
    """
    Prepare dataset for GRPO training using custom dataset class.
    Returns a dataset that properly handles images without serialization issues.
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