import torch
from transformers import Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

def apply_lora_to_quantized_model(model_path):
    # NOTE: T4 GPUs (Kaggle) do NOT support bfloat16 (requires Ampere/A100+).
    # Use float16 instead to avoid silent fallback to fp32 (which wastes VRAM).
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Configure generation settings
    model.generation_config.max_new_tokens = 512
    model.generation_config.do_sample = True
    model.generation_config.top_k = 50
    model.generation_config.top_p = 0.9
    model.generation_config.temperature = 0.9  # Slightly higher temp for more generation diversity

    # CRITICAL FIX (Bug #1): Do NOT call prepare_model_for_kbit_training() here.
    # That function is designed for bitsandbytes (bnb) 4-bit, not GPTQ.
    # For GPTQ models, calling it freezes ALL parameters including LoRA adapters,
    # causing loss=0 and grad_norm=0 throughout training.
    #
    # Instead, manually enable input require_grads so gradients can flow
    # from LoRA adapter outputs back through the frozen quantized base layers.
    model.enable_input_require_grads()

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=target_modules,
        exclude_modules=["visual"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze the vision encoder explicitly (we only train the language model)
    for name, param in peft_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    peft_model.print_trainable_parameters()

    # Safety check: verify LoRA params are actually trainable
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            "[FATAL] No trainable parameters found after applying LoRA! "
            "LoRA adapters were not grafted onto GPTQ layers. "
            "Check that your target_modules names match the actual layer names in the model."
        )
    print(f"[OK] Trainable parameters: {trainable_params:,}")

    visual_is_training = any(p.requires_grad for name, p in peft_model.named_parameters() if "visual" in name)
    print(f"Vision encoder frozen: {not visual_is_training}")

    return peft_model

if __name__ == "__main__":
    QUANT_MODEL_DIR = r"./weights/Qwen2-VL-7B-Instruct-GPTQ-Int3"
    model_ready_for_rl = apply_lora_to_quantized_model(QUANT_MODEL_DIR)
