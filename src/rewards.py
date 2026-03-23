import re

def extract_xml_answer(text: str) -> str:
    """
    Trích xuất nội dung nằm trong thẻ <answer>...</answer>.
    Nếu không có, trả về chuỗi rỗng.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Thưởng 1.0 điểm nếu mô hình tuân thủ đúng định dạng tư duy:
    <think> ... </think> <answer> ... </answer>
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        has_think = "<think>" in content and "</think>" in content
        has_answer = "<answer>" in content and "</answer>" in content
        
        if has_think and has_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0) 
            
    return rewards

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Thưởng 1.0 điểm nếu đáp án trong thẻ <answer> khớp chính xác với ground_truth.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        pred_answer = extract_xml_answer(content)
        
        if pred_answer.lower().strip() == truth.lower().strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def visual_faithfulness_reward_func(completions, **kwargs) -> list[float]:
    """
    Đây là nơi bạn có thể thêm logic phạt (ví dụ: trừ 0.5 điểm) 
    nếu mô hình sinh ra ảo giác không có trong ảnh.
    """
    return [0.0] * len(completions)