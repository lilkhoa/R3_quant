import re

def extract_xml_answer(text: str) -> str:
    """
    Extract content within <answer>...</answer> tags from generated text.
    Returns empty string if not found.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def extract_think_content(text: str) -> str:
    """
    Extract content within <think>...</think> tags from generated text.
    Returns empty string if not found.
    """
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return ""

def _check_tag_ordering(content: str) -> tuple[bool, bool]:
    """
    Verify strict tag ordering: <think> before </think> before <answer> before </answer>.
    Returns (has_valid_think, has_valid_answer).
    """
    think_open = content.find("<think>")
    think_close = content.find("</think>")
    answer_open = content.find("<answer>")
    answer_close = content.find("</answer>")
    
    # Valid think pair: <think> appears and comes before </think>
    has_valid_think = (think_open != -1 and think_close != -1 and think_open < think_close)
    
    # Valid answer pair in correct position: <answer> appears after </think> and before </answer>
    has_valid_answer = (
        answer_open != -1 and answer_close != -1 and 
        answer_open < answer_close and
        think_close < answer_open
    )
    
    return has_valid_think, has_valid_answer

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Enforce strict tag ordering with partial rewards.
    - Valid <think>...</think> pair: +0.4 points
    - Valid <answer>...</answer> pair in correct position (after </think>): +0.6 points
    
    Prevents cheating: model cannot generate answer first then think tag.
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        has_valid_think, has_valid_answer = _check_tag_ordering(content)
        
        reward = 0.0
        if has_valid_think:
            reward += 0.4
        if has_valid_answer:
            reward += 0.6
        
        rewards.append(reward)
    
    return rewards

def _extract_answer_letter(text: str) -> str:
    """
    Extract answer letter (A-F) from common multiple-choice contexts using regex.
    Prioritizes letters in parentheses like (A), then standalone letters surrounded
    by whitespace or punctuation to avoid matching letters in words.
    """
    # Priority 1: Letters in parentheses (A), (B), etc.
    pattern_parens = r'\(([A-Fa-f])\)'
    match = re.search(pattern_parens, text)
    if match:
        return match.group(1).upper()
    
    # Priority 2: Standalone letters surrounded by whitespace or punctuation
    # Match patterns like " A ", "A.", "A,", etc., but not "Apple"
    pattern_standalone = r'(?:^|\s|:|,|。|、|；)([A-Fa-f])(?:\s|:|,|\.|\?|!|。|、|；|$)'
    match = re.search(pattern_standalone, text)
    if match:
        return match.group(1).upper()
    
    return ""

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Extract answer letter from <answer> tag using regex and compare with ground truth.
    Handles multiple-choice contexts: (A), "Option A", "The answer is A.", etc.
    Assigns 1.0 for exact letter match, 0.0 otherwise.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        answer_text = extract_xml_answer(content)
        
        # Extract answer letter using regex
        pred_letter = _extract_answer_letter(answer_text)
        
        # Get ground truth letter and normalize
        truth_letter = _extract_answer_letter(truth)
        if not truth_letter:
            truth_letter = truth.upper().strip()
        
        if pred_letter == truth_letter:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def brevity_penalty_func(completions, **kwargs) -> list[float]:
    """
    Apply small negative penalty proportional to response length.
    Discourages hallucination and repetitive reasoning that wastes tokens/VRAM.
    Penalty: -0.01 per 100 characters.
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        # Calculate penalty based on character count
        char_count = len(content)
        penalty = -(char_count / 100) * 0.01
        
        # Cap penalty to prevent extreme negative values
        penalty = max(penalty, -0.5)
        
        rewards.append(penalty)
    
    return rewards

def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """
    Anti-cheating reward: penalize responses with empty or too-short reasoning.
    Ensures model actually reasons instead of shortcutting with <think></think><answer>A</answer>.
    
    Minimum reasoning length: 15 characters (enforced after brevity penalty introduced).
    Penalty for violating: -1.0 (strong negative to force meaningful reasoning).
    """
    rewards = []
    MIN_REASONING_LENGTH = 15
    
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        think_content = extract_think_content(content)
        
        # Count characters in reasoning (excluding whitespace padding)
        reasoning_length = len(think_content.replace(" ", "").replace("\n", "").replace("\t", ""))
        
        if reasoning_length < MIN_REASONING_LENGTH:
            rewards.append(-1.0)
        else:
            rewards.append(0.0)  # No bonus, just no penalty
    
    return rewards

def visual_faithfulness_reward_func(completions, **kwargs) -> list[float]:
    """
    Placeholder for visual faithfulness checks.
    Can implement logic to penalize hallucinations not present in images.
    Currently returns neutral reward (0.0 for all completions).
    """
    return [0.0] * len(completions)