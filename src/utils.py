import torch
import re
import io
from PIL import Image
from datasets import Dataset, DatasetDict

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

class MiniCOTDataset(torch.utils.data.Dataset):
    """
    Custom dataset for mini_cot_8k_verified SFT training.
    Wraps reasoning chains in <think></think> tags and answers in <answer></answer> tags.
    
    Supports both streaming and non-streaming datasets.
    
    Dataset columns:
    - image: PIL Image or dict with image data
    - problem: Question with choices
    - solution: Reasoning chain (may already have tags or raw text)
    - original_question: Original question (can be used as fallback)
    - original_answer: Original answer format
    """
    def __init__(self, raw_dataset, max_samples=None):
        self.raw_dataset = raw_dataset
        self.max_samples = max_samples
        
        # FIX #1: Unwrap DatasetDict if present
        if isinstance(raw_dataset, DatasetDict):
            raw_dataset = raw_dataset["train"]
            self.raw_dataset = raw_dataset
        
        # Check if dataset is streaming (has no len)
        self.is_streaming = not hasattr(raw_dataset, '__len__')
        
        if not self.is_streaming:
            # For non-streaming datasets, pre-load and filter items
            self.items = []
            count = 0
            first_item_logged = False
            
            for item in raw_dataset:
                if max_samples and count >= max_samples:
                    break
                
                # Debug: print first item's structure
                if not first_item_logged:
                    print(f"[DEBUG] First item keys: {item.keys() if isinstance(item, dict) else 'Not a dict'}")
                    first_item_logged = True
                
                # FIX #2: Remove column guessing logic, directly access fixed columns
                try:
                    # Dataset structure is strictly fixed to these five columns
                    solution = item.get("solution", "")
                    
                    # If no solution field found, skip
                    if not solution or not str(solution).strip():
                        continue
                except Exception as e:
                    print(f"[DEBUG] Error extracting solution: {e}")
                    continue
                
                # Directly access the fixed column names
                problem = item.get("problem", "")
                answer = item.get("original_answer", "")
                
                # Get image if available, otherwise None (will create dummy during __getitem__)
                image = item.get("image", None)
                
                self.items.append({
                    'image': image,
                    'problem': problem,
                    'solution': solution,
                    'original_answer': answer
                })
                count += 1
        else:
            # For streaming datasets, don't preload - handle lazily
            self.items = None
            print(f"[INFO] Using streaming dataset (lazy loading)")
    
    def __len__(self):
        if self.is_streaming:
            # For streaming datasets, estimate length or use a default
            return 8000  # Approximate length for mini_cot_8k_verified
        return len(self.items)
    
    def __getitem__(self, idx):
        # For streaming datasets, we need to handle differently
        if self.is_streaming:
            # Skip to index (inefficient but necessary for streaming)
            if idx == 0:
                self._stream_iter = iter(self.raw_dataset)
            
            if not hasattr(self, '_stream_iter'):
                self._stream_iter = iter(self.raw_dataset)
                
            # Get the item at this index by iterating
            raw_item = None
            for i, row in enumerate(self._stream_iter):
                if i == idx:
                    raw_item = row
                    break
            
            if raw_item is None:
                # Fallback: return dummy item
                raw_item = {
                    'image': None,
                    'problem': "Question",
                    'solution': "Answer",
                    'original_answer': "A"
                }
            
            # FIX #2: Remove column guessing logic, directly access fixed columns
            solution = raw_item.get("solution", "")
            problem = raw_item.get("problem", "")
            answer = raw_item.get("original_answer", "")
            
            # Construct item dict with normalized field names
            item = {
                'image': raw_item.get("image", None) if isinstance(raw_item, dict) else raw_item.get("image", None),
                'problem': problem,
                'solution': solution,
                'original_answer': answer
            }
        else:
            item = self.items[idx]
        
        try:
            # Handle None/missing image by creating a placeholder
            # Many CoT datasets (like mini_cot_8k_verified) don't have images
            if item.get("image") is None or (isinstance(item, dict) and item.get("image") is None):
                # Create a white placeholder image with text overlay
                pil_image = Image.new('RGB', (224, 224), color='white')
            else:
                # Convert image to PIL
                pil_image = _convert_image_to_pil(item['image'])
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Build user message with problem/question
            user_text = str(item.get('problem', item['problem'] if not isinstance(item, dict) else ""))
            if not user_text.strip():
                user_text = "Please answer this question step by step, then provide your final answer."
            
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text}
                ]
            }
            
            # Process solution: wrap in <think> tags if not already wrapped
            # FIX #3: Force-cast solution to string before regex to prevent TypeError from None values
            solution = item.get('solution', "")
            solution_text = str(solution) if solution is not None else ""
            
            # Extract or wrap thinking
            think_match = re.search(r'<think>(.*?)</think>', solution_text, re.DOTALL)
            if think_match:
                # Already has tags, extract the content
                thinking_content = think_match.group(1).strip()
            else:
                # No tags yet, use entire solution as reasoning
                thinking_content = solution_text.strip()
            
            # Extract answer letter (A-E, case insensitive)
            answer_match = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', solution_text, re.IGNORECASE)
            if answer_match:
                answer_letter = answer_match.group(1).upper()
            else:
                # Try to extract from original_answer or last digit in solution
                answer_letter = _extract_answer_from_text(solution_text + " " + str(item.get('original_answer', item['original_answer'] if not isinstance(item, dict) else "")))
                if not answer_letter:
                    answer_letter = "A"  # Default fallback
            
            # Build properly formatted assistant response
            assistant_text = (
                f"<think>\n{thinking_content}\n</think>\n"
                f"<answer>{answer_letter}</answer>"
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
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return dummy item
            dummy_image = Image.new('RGB', (224, 224), color='white')
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Please answer this question."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "<think>Unable to process item.</think>\n<answer>A</answer>"}
                        ]
                    }
                ],
                "images": [dummy_image]
            }

def _extract_answer_from_text(text: str) -> str:
    """Extract answer letter (A-E) from various text formats."""
    # Priority 1: Look for tagged answer (already checked but check again for safety)
    match = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Priority 2: Look for "Answer: X" pattern
    match = re.search(r'[Aa]nswer\s*:?\s*([A-Ea-e])', text)
    if match:
        return match.group(1).upper()
    
    # Priority 3: Look for "choice X" pattern
    match = re.search(r'[Cc]hoice\s+([A-Ea-e])', text)
    if match:
        return match.group(1).upper()
    
    # Priority 4: Look for last occurrence of letter between parens or standalone
    matches = re.findall(r'([A-Ea-e])', text)
    if matches:
        return matches[-1].upper()
    
    return ""

def prepare_minicot_for_sft(raw_dataset, max_samples=None):
    """
    Prepare mini_cot_8k_verified dataset for SFT training.
    Wraps reasoning chains in XML tags for format alignment.
    
    Args:
        raw_dataset: HuggingFace dataset with mini_cot data
        max_samples: Limit number of samples (default: use all)
    
    Returns:
        MiniCOTDataset instance ready for SFT training
    """
    dataset = MiniCOTDataset(raw_dataset, max_samples=max_samples)
    if len(dataset) == 0:
        print("Warning: Mini-COT dataset is empty after processing.")
    else:
        print(f"[SFT] Loaded {len(dataset)} CoT examples for format alignment training.")
    return dataset