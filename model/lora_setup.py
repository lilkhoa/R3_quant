import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora_to_quantized_model(model_path):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        r=16,                
        lora_alpha=32,       
        target_modules=target_modules,
        exclude_modules=["visual"], 
        lora_dropout=0.05,   
        bias="none",         
        task_type="CAUSAL_LM" 
    )

    peft_model = get_peft_model(model, lora_config)

    for name, param in peft_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    peft_model.print_trainable_parameters()
    
    visual_is_training = any(p.requires_grad for name, p in peft_model.named_parameters() if "visual" in name)
    print(f"(Vision Encoder)? -> {'Vision đang bị train' if visual_is_training else 'Done'}")

    return peft_model

if __name__ == "__main__":
    QUANT_MODEL_DIR = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    model_ready_for_rl = apply_lora_to_quantized_model(QUANT_MODEL_DIR)