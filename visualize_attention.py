import sys
import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
# Model paths (matching every other script in the project)
BASE_MODEL_PATH      = r"./weights/Qwen2-VL-7B-Instruct"           # fp16/bf16
QUANTIZED_MODEL_PATH = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3" # 3-bit GPTQ
GRPO_LORA_PATH       = r"./r3_quant_checkpoints"                    # LoRA adapters

# Local test image (hepatitis B vaccination bar chart)
IMAGE_PATH = r"./test_images/test_1.png"

QUESTION = "What is the share of one-year-olds vaccinated against HepB3 in Bhutan?"

# Qwen2-VL special token IDs (constant across all model sizes)
VISION_START_ID = 151652
VISION_END_ID   = 151653
IMAGE_PAD_ID    = 151655

# ─────────────────────────────────────────────────────────────────
# Helper: annotate and summarise one input_ids sequence
# ─────────────────────────────────────────────────────────────────

def inspect_input_ids(input_ids: list, tokenizer) -> dict:
    """
    Print every token with its decoded text and mark image-region tokens.
    Returns a dict with the detected image span indices.
    """
    seq_len = len(input_ids)

    print("\n[RAW input_ids array]")
    print(input_ids)

    print(f"\n[ANNOTATED — {seq_len} tokens total]")
    print("-" * 65)

    for idx, tid in enumerate(input_ids):
        try:
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            decoded = "<?>"

        marker = ""
        if tid == VISION_START_ID:
            marker = "  ◀ vision_start"
        elif tid == VISION_END_ID:
            marker = "  ◀ vision_end"
        elif tid == IMAGE_PAD_ID:
            marker = "  ◀ image_pad"

        print(f"  [{idx:>4}]  {tid:>7}   {repr(decoded)}{marker}")

    # Locate spans
    vision_start_idx = [i for i, t in enumerate(input_ids) if t == VISION_START_ID]
    vision_end_idx   = [i for i, t in enumerate(input_ids) if t == VISION_END_ID]
    pad_idx          = [i for i, t in enumerate(input_ids) if t == IMAGE_PAD_ID]

    print()
    print("=" * 65)
    print("SUMMARY — Image token region")
    print("=" * 65)
    print(f"  VISION_START_ID ({VISION_START_ID}) at indices : {vision_start_idx}")
    print(f"  VISION_END_ID   ({VISION_END_ID}) at indices : {vision_end_idx}")
    first5 = pad_idx[:5]
    last5  = pad_idx[-5:] if len(pad_idx) > 5 else []
    print(f"  IMAGE_PAD_ID    ({IMAGE_PAD_ID}) first 5 : {first5}")
    if last5:
        print(f"                             last  5 : {last5}")
    print(f"  Total image_pad tokens : {len(pad_idx)}")

    result = {}
    if vision_start_idx and vision_end_idx:
        img_start = vision_start_idx[0]
        img_end   = vision_end_idx[-1]
        result = {
            "vision_start": img_start,
            "vision_end":   img_end,
            "pad_start":    img_start + 1,
            "pad_end":      img_end - 1,
            "span_len":     img_end + 1 - img_start,
            "text_before":  img_start,
            "text_after":   seq_len - img_end - 1,
        }
        print()
        print(f"  ► Image token block : input_ids[{img_start} : {img_end + 1}]")
        print(f"    • vision_start at index  {img_start}")
        print(f"    • image_pad    from index {img_start + 1} to {img_end - 1}")
        print(f"    • vision_end   at index  {img_end}")
        print(f"    • Span length            {img_end + 1 - img_start} tokens")
        print()
        print(f"  ► Text tokens BEFORE image : input_ids[0 : {img_start}]"
              f"   ({img_start} tokens)")
        print(f"  ► Text tokens AFTER  image : input_ids[{img_end + 1} : {seq_len}]"
              f"   ({seq_len - img_end - 1} tokens)")
    else:
        print("\n  [WARN] vision_start / vision_end tokens not found.")
        print("         Inspect the annotated list above for the repeated IMAGE_PAD_ID block.")

    return result


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load processor (shared by all three model variants)
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading processor")
print("=" * 65)

# The quantized model directory contains the saved processor/tokenizer
# (saved there by model/quantizer.py → processor.save_pretrained(save_path))
processor = AutoProcessor.from_pretrained(QUANTIZED_MODEL_PATH)
print(f"  Processor loaded from : {QUANTIZED_MODEL_PATH}")
print(f"  Vocab size            : {processor.tokenizer.vocab_size}")
print()

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Fetch sample image
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 2 — Loading local test image")
print("=" * 65)

image_path_abs = os.path.abspath(IMAGE_PATH)
if not os.path.isfile(image_path_abs):
    raise FileNotFoundError(
        f"Test image not found at: {image_path_abs}\n"
        "Place the chart image at ./test_images/test_1.png and re-run."
    )
image = Image.open(image_path_abs).convert("RGB")
print(f"  Loaded from : {image_path_abs}")
print(f"  Image size  : {image.size}   mode: {image.mode}")

print()

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Build inputs via processor
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 3 — Running processor to build input tensors")
print("=" * 65)

messages = [
    {
        "role": "system",
        "content": (
            "You are a logical reasoning AI. "
            "You MUST think step-by-step and enclose your entire reasoning "
            "within <think> and </think> tags. "
            "After thinking, output your final answer (a number or short text) "
            "enclosed within <answer> and </answer> tags."
        ),
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": QUESTION},
        ],
    },
]

text_prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print(f"  Tensor keys in inputs dict:")
for key, val in inputs.items():
    if torch.is_tensor(val):
        print(f"    {key:<30} shape={tuple(val.shape)}  dtype={val.dtype}")
print()

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Inspect input_ids → find image token index range
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
seq = inputs["input_ids"][0].tolist()
print(f"STEP 4 — input_ids inspection  (seq_len = {len(seq)})")
print("=" * 65)

span_info = inspect_input_ids(seq, processor.tokenizer)

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Model inventory (load each to confirm + print config)
#           NOTE: models are NOT kept in memory simultaneously to
#           avoid OOM on L4 (24 GB). Each is loaded, inspected,
#           then deleted before the next one is loaded.
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 5 — Model inventory")
print("    (each model is loaded, described, then released)")
print("=" * 65)

def describe_model(tag: str, model_path: str, lora_path: str = None):
    """Load a model variant, print its config summary, then free it."""
    print(f"\n{'─'*65}")
    print(f"  [{tag}]")
    print(f"  Base   : {model_path}")
    if lora_path:
        print(f"  LoRA   : {lora_path}")

    # L4 supports bfloat16 natively
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        os.path.abspath(model_path),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, os.path.abspath(lora_path))
        # Ensure consistent dtype (GPTQ layers may be fp16)
        for _, param in model.named_parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.bfloat16)
        print(f"  LoRA adapters merged: YES")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params / 1e9:.2f} B")
    print(f"  Trainable params : {trainable_params / 1e6:.2f} M")
    print(f"  dtype            : {next(model.parameters()).dtype}")
    print(f"  device_map       : {getattr(model, 'hf_device_map', 'N/A')}")

    del model
    torch.cuda.empty_cache()
    print(f"  → Model released from VRAM.")

describe_model(
    tag="1. Base (unquantized, BF16)",
    model_path=BASE_MODEL_PATH,
)

describe_model(
    tag="2. Quantized (3-bit GPTQ, no LoRA)",
    model_path=QUANTIZED_MODEL_PATH,
)

describe_model(
    tag="3. Quantized + GRPO LoRA checkpoint",
    model_path=QUANTIZED_MODEL_PATH,
    lora_path=GRPO_LORA_PATH,
)

# ─────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 1 complete — image token index range identified:")
if span_info:
    print(f"  vision_start  →  index {span_info['vision_start']}")
    print(f"  image_pad     →  index {span_info['pad_start']} .. {span_info['pad_end']}")
    print(f"  vision_end    →  index {span_info['vision_end']}")
    print(f"  span length   =  {span_info['span_len']} tokens")
print()
print("Use these indices in the next step to slice attention weights:")
print("  attn[:, :, text_positions, image_pad_start:image_pad_end+1]")
print("=" * 65)
