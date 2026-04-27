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
            pil_image.thumbnail((768, 768))
            
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

    prompt += "\nStart your response directly with the <think> tag.\n"
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
        
        # System message identical to GRPO
        SYSTEM_MESSAGE = (
            "You are a logical reasoning AI. "
            "You MUST think step-by-step and enclose your entire reasoning "
            "within <think> and </think> tags. "
            "After thinking, output your final answer (one letter only) "
            "enclosed within <answer> and </answer> tags."
        )

        # Build messages
        system_message = {
            "role": "system",
            "content": SYSTEM_MESSAGE
        }

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
            "messages": [system_message, user_message, assistant_message],
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
        
        # FIX (streaming): Eagerly materialize ALL data into self.items regardless
        # of streaming mode. The broken lazy streaming __getitem__ re-iterated the
        # stream from position 0 on every call, producing wrong/duplicate/None items.
        # mini_cot_8k_verified is only ~8k rows (~8 MB) — trivial to hold in RAM.
        self.items = []
        count = 0
        first_item_logged = False
        
        print(f"[INFO] MiniCOTDataset: materializing dataset into memory...")
        for item in raw_dataset:
            if max_samples and count >= max_samples:
                break
            
            # Debug: print first item's structure
            if not first_item_logged:
                print(f"[DEBUG] First item keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                first_item_logged = True
            
            # Directly access the fixed column names
            try:
                solution = item.get("solution", "")
                # Skip items with no solution
                if not solution or not str(solution).strip():
                    continue
            except Exception as e:
                print(f"[DEBUG] Error extracting solution: {e}")
                continue
            
            problem = item.get("problem", "")
            answer = item.get("original_answer", "")
            image = item.get("image", None)
            
            self.items.append({
                'image': image,
                'problem': problem,
                'solution': solution,
                'original_answer': answer
            })
            count += 1
        
        print(f"[INFO] MiniCOTDataset: loaded {len(self.items)} items into memory.")
    
    def __len__(self):
        return len(self.items)
    
    def get_sample_items(self, n: int = 2):
        """Return up to n items for use in logging callbacks."""
        return self.items[:min(n, len(self.items))]
    
    def __getitem__(self, idx):
        # All data is eagerly loaded into self.items — simple index lookup
        item = self.items[idx]
        
        try:
            has_real_image = item.get("image") is not None

            if has_real_image:
                pil_image = _convert_image_to_pil(item['image'])
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_image.thumbnail((448, 448))

            # Build user message with problem/question
            user_text = str(item.get('problem', ""))
            if not user_text.strip():
                user_text = "Please answer this question step by step, then provide your final answer."

            SYSTEM_MESSAGE = (
                "You are a logical reasoning AI. "
                "You MUST think step-by-step and enclose your entire reasoning "
                "within <think> and </think> tags. "
                "After thinking, output your final answer (one letter only) "
                "enclosed within <answer> and </answer> tags."
            )
            
            system_message = {
                "role": "system",
                "content": SYSTEM_MESSAGE
            }

            if has_real_image:
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text}
                    ]
                }
            else:
                # Text-only: no image token → no visual tokens generated
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text}
                    ]
                }

            # Process solution: wrap in <think> tags if not already wrapped
            solution = item.get('solution', "")
            solution_text = str(solution) if solution is not None else ""

            # Extract or wrap thinking
            think_match = re.search(r'<think>(.*?)</think>', solution_text, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
            else:
                thinking_content = solution_text.strip()

            # Extract answer letter (A-E, case insensitive)
            answer_match = re.search(r'<answer>\s*([A-Ea-e])\s*</answer>', solution_text, re.IGNORECASE)
            if answer_match:
                answer_letter = answer_match.group(1).upper()
            else:
                answer_letter = _extract_answer_from_text(
                    solution_text + " " + str(item.get('original_answer', ""))
                )
                if not answer_letter:
                    answer_letter = "A"

            assistant_text = (
                f"<think>\n{thinking_content}\n</think>\n"
                f"<answer>{answer_letter}</answer>"
            )
            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}]
            }

            if has_real_image:
                return {
                    "messages": [system_message, user_message, assistant_message],
                    "images": [pil_image]
                }
            else:
                return {
                    "messages": [system_message, user_message, assistant_message]
                }

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a logical reasoning AI. You MUST think step-by-step and enclose your entire reasoning within <think> and </think> tags. After thinking, output your final answer (one letter only) enclosed within <answer> and </answer> tags."
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Please answer this question step by step."}]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<think>Unable to process item.</think>\n<answer>A</answer>"}]
                    }
                ]
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


# ============================================================================
# POPE Evaluation Helpers
# ============================================================================

def build_pope_prompt(question: str) -> str:
    """
    Build a POPE-specific prompt for binary Yes/No object-presence questions.
    POPE questions are always of the form "Is there a X in the image?"
    The model is instructed to reason then output exactly 'yes' or 'no'.
    """
    return (
        f"{question}\n\n"
        "Answer with 'yes' or 'no' only.\n"
        "Start your response directly with the <think> tag.\n"
    )


def extract_pope_answer(text: str) -> str:
    """
    Extract a binary 'yes' or 'no' answer from model output.

    Priority order:
    1. Content inside <answer>...</answer> tags.
    2. First occurrence of a standalone 'yes' or 'no' in the text.
    3. Empty string when nothing is found.

    The returned value is always lowercase 'yes', 'no', or ''.
    """
    # Priority 1: <answer> tags
    match = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Priority 2: last <answer> tag content (in case model writes extra text)
    match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip().lower()
        if 'yes' in content:
            return 'yes'
        if 'no' in content:
            return 'no'

    # Priority 3: standalone word after </think>
    after_think = re.split(r'</think>', text, maxsplit=1, flags=re.IGNORECASE)
    search_zone = after_think[-1]  # entire text if no </think> found
    match = re.search(r'\b(yes|no)\b', search_zone, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return ''


def compute_pope_metrics(pred_answers: list, ground_truths: list) -> dict:
    """
    Compute the standard POPE evaluation metrics (Li et al., EMNLP 2023).

    Metrics returned:
        accuracy   - % of predictions matching the ground truth
        precision  - TP / (TP + FP)  for the positive class ('yes')
        recall     - TP / (TP + FN)  for the positive class ('yes')
        f1         - harmonic mean of precision and recall
        yes_ratio  - % of model predictions that are 'yes'

    All values are expressed as percentages (0-100).
    """
    assert len(pred_answers) == len(ground_truths), "Length mismatch"

    tp = fp = tn = fn = 0
    for pred, gt in zip(pred_answers, ground_truths):
        p = pred.strip().lower()
        g = gt.strip().lower()
        if p == 'yes' and g == 'yes':
            tp += 1
        elif p == 'yes' and g == 'no':
            fp += 1
        elif p == 'no' and g == 'no':
            tn += 1
        elif p == 'no' and g == 'yes':
            fn += 1
        # answers that are neither 'yes' nor 'no' are treated as wrong
        elif g == 'yes':
            fn += 1
        else:
            fp += 1

    total     = len(pred_answers)
    correct   = tp + tn
    yes_count = tp + fp

    accuracy  = (correct   / total)               * 100 if total     > 0 else 0.0
    precision = (tp        / (tp + fp))           * 100 if (tp + fp) > 0 else 0.0
    recall    = (tp        / (tp + fn))           * 100 if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    yes_ratio = (yes_count / total)               * 100 if total     > 0 else 0.0

    return {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'yes_ratio': yes_ratio,
    }


# ============================================================================
# ChartQA Evaluation Helpers  (lmms-lab/ChartQA)
# ============================================================================

def build_chartqa_prompt(question: str) -> str:
    """
    Build a ChartQA-specific prompt for open-ended chart question answering.
    The model is asked to analyze the chart image carefully and return a
    concise answer (a number or short text) inside <answer> tags.
    """
    return (
        f"Question: {question}\n\n"
        "Carefully analyze the chart and answer concisely.\n"
        "Start your response directly with the <think> tag.\n"
    )


def extract_chartqa_answer(text: str) -> str:
    """
    Extract the free-form answer from ChartQA model output.

    Priority order:
    1. Content inside <answer>...</answer> tags.
    2. Last non-empty line of the output (common fallback when tags are absent).
    """
    match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return ""


def chartqa_relaxed_correct(predicted: str, ground_truth: str, tolerance: float = 0.05) -> bool:
    """
    ChartQA relaxed accuracy metric (Masry et al., ACL Findings 2022).

    - Numeric answers: correct when the relative error is within `tolerance` (default 5%).
      |pred - gt| / max(|gt|, 1e-8) <= tolerance
    - Text answers: case-insensitive exact match after stripping punctuation / whitespace.
    """
    pred = predicted.strip().rstrip("%").strip()
    gt   = ground_truth.strip().rstrip("%").strip()

    # Numeric comparison first
    try:
        pred_num = float(pred.replace(",", ""))
        gt_num   = float(gt.replace(",", ""))
        return abs(pred_num - gt_num) / max(abs(gt_num), 1e-8) <= tolerance
    except ValueError:
        pass

    # Text comparison: normalize
    def _norm(s: str) -> str:
        return re.sub(r'[^\w\s]', '', s).lower().strip()

    return _norm(pred) == _norm(gt)


# ============================================================================
# DocumentVQA GRPO Dataset
# ============================================================================

DOCVQA_SYSTEM_MESSAGE = (
    "You are a document understanding AI. "
    "You MUST carefully read the document image and think step-by-step. "
    "Enclose your entire reasoning within <think> and </think> tags. "
    "After thinking, output your final answer enclosed within <answer> and </answer> tags. "
    "The answer should be concise — a word, number, or short phrase extracted from the document."
)


def build_docvqa_prompt(question: str) -> str:
    """Build a user-turn prompt for HuggingFaceM4/DocumentVQA (open-ended, no choices)."""
    return (
        f"Question: {question}\n\n"
        "Look carefully at the document image and answer the question. "
        "Start your response directly with the <think> tag.\n"
    )


class DocumentVQAGRPODataset(torch.utils.data.Dataset):
    """
    Dataset for GRPO training on HuggingFaceM4/DocumentVQA.

    HuggingFace schema:
        questionId            - int32
        question              - string   ← used as the prompt
        question_types        - list[string]
        image                 - Image    ← document page image
        docId                 - int32
        ucsf_document_id      - string
        ucsf_document_page_no - string
        answers               - list[string]  ← ground truth (first element used)
    """

    def __init__(self, raw_dataset, max_samples=None):
        import numpy as np

        self.items = []
        count = 0

        for item in raw_dataset:
            if max_samples and count >= max_samples:
                break
            if item.get("image") is None:
                continue

            # answers is stored as list or numpy array in parquet; normalise to list
            answers = item.get("answers", [])
            if isinstance(answers, np.ndarray):
                answers = answers.tolist()
            if not answers:
                continue

            self.items.append({
                "image":    item["image"],
                "question": str(item["question"]),
                "answer":   str(answers[0]),   # first accepted answer as canonical
            })
            count += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        try:
            pil_image = _convert_image_to_pil(item["image"])
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.thumbnail((768, 768))

            text_prompt = build_docvqa_prompt(item["question"])

            messages = [
                {"role": "system", "content": DOCVQA_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]

            return {
                "prompt":       messages,
                "images":       [pil_image],
                "ground_truth": item["answer"],
            }

        except Exception as e:
            print(f"[DocumentVQAGRPODataset] Error on item {idx}: {e}")
            dummy = Image.new("RGB", (224, 224), color="white")
            return {
                "prompt": [
                    {"role": "system", "content": DOCVQA_SYSTEM_MESSAGE},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "What does the document say?"},
                        ],
                    },
                ],
                "images":       [dummy],
                "ground_truth": "unknown",
            }


def prepare_docvqa_for_grpo(raw_dataset, processor, max_samples=None):
    """Prepare HuggingFaceM4/DocumentVQA for GRPO training."""
    dataset = DocumentVQAGRPODataset(raw_dataset, max_samples=max_samples)
    if len(dataset) == 0:
        print("Warning: DocumentVQA GRPO dataset is empty after processing. Check data path.")
    return dataset