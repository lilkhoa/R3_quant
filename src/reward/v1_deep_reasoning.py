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
    pattern_standalone = r'(?:^|\s|:|,|。|、|;)([A-Fa-f])(?:\s|:|,|\.|\?|!|。|、|;|$)'
    match = re.search(pattern_standalone, text)
    if match:
        return match.group(1).upper()
    
    return ""

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Extract answer letter from <answer> tag using regex and compare with ground truth.
    Implements soft accuracy with partial credit for reasoning.
    - Exact match: +1.0
    - Incorrect but correct answer appears in reasoning: +0.3 to +0.4 (consolation bonus)
    - Otherwise: 0.0
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        answer_text = extract_xml_answer(content)
        think_text = extract_think_content(content)
        
        # Extract answer letter using regex
        pred_letter = _extract_answer_letter(answer_text)
        
        # Get ground truth letter and normalize
        truth_letter = _extract_answer_letter(truth)
        if not truth_letter:
            truth_letter = truth.upper().strip()
        
        # Primary score: exact match
        if pred_letter == truth_letter:
            rewards.append(1.0)
        else:
            # Soft accuracy: check if correct answer appears in reasoning with conclusive language
            # Look for patterns like "therefore A", "answer A", "conclusion A", etc.
            consolation_score = 0.0
            truth_lower = truth_letter.lower()
            truth_upper = truth_letter.upper()
            
            # Count occurrences of truth letter in think content
            truth_count = think_text.count(truth_letter)
            
            # Look for conclusion-style patterns
            conclusion_patterns = [
                rf'therefore\s+{truth_upper}\s',
                rf'therefore\s+{truth_lower}\s',
                rf'answer\s+{truth_upper}\s',
                rf'answer\s+{truth_lower}\s',
                rf'conclusion\s+{truth_upper}\s',
                rf'conclusion\s+{truth_lower}\s',
                rf'correct.*?{truth_upper}\s',
                rf'correct.*?{truth_lower}\s',
            ]
            
            has_conclusion = any(re.search(pattern, think_text, re.IGNORECASE) for pattern in conclusion_patterns)
            
            # Award partial credit if correct answer appears multiple times OR in conclusion
            if truth_count >= 2 or has_conclusion:
                consolation_score = 0.3
                if truth_count >= 3:  # Extra bonus for very strong signal
                    consolation_score = 0.4
            
            rewards.append(consolation_score)
    
    return rewards

def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """
    Enforce meaningful chain-of-thought reasoning.
    Increased minimum from 15 to 100 characters to force deep reasoning.
    
    - Under 100 non-whitespace characters: -1.0 (penalize shallow reasoning)
    - Over 100 characters: +0.2 (bonus for substantial reasoning)
    """
    rewards = []
    MIN_REASONING_LENGTH = 100  # Increased from 15
    
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        think_content = extract_think_content(content)
        
        # Count characters in reasoning (excluding whitespace padding)
        reasoning_length = len(think_content.replace(" ", "").replace("\n", "").replace("\t", ""))
        
        if reasoning_length < MIN_REASONING_LENGTH:
            rewards.append(-1.0)  # Strong penalty for too-short reasoning
        else:
            rewards.append(0.2)  # Bonus for substantial reasoning
    
    return rewards

def logic_structure_reward_func(completions, **kwargs) -> list[float]:
    """
    Check whether model is reasoning logically or generating filler text.
    Scans for logical transition keywords that indicate actual reasoning vs. rambling.
    
    - 2-3+ keywords found: +0.3 (structured reasoning)
    - No keywords found: -0.3 (likely meaningless text to reach length target)
    """
    # Logical transition keywords that indicate structured reasoning
    logical_keywords = [
        r'\bbecause\b',
        r'\btherefore\b',
        r'\bobserving\s+the\s+image',
        r'\bstep\b',
        r'\bhowever\b',
        r'\bconclusion\b',
        r'\bthus\b',
        r'\bsince\b',
        r'\bnotice\b',
        r'\bidentify',
    ]
    
    rewards = []
    
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        think_content = extract_think_content(content)
        
        if not think_content:
            # Empty reasoning = no logic structure
            rewards.append(-0.3)
            continue
        
        # Count keyword matches (case-insensitive)
        keyword_count = 0
        for keyword_pattern in logical_keywords:
            matches = re.findall(keyword_pattern, think_content, re.IGNORECASE)
            keyword_count += len(matches)
        
        # Award or penalize based on keyword density
        if keyword_count >= 2:
            rewards.append(0.3)  # Structured reasoning detected
        else:
            rewards.append(-0.3)  # Likely filler text
    
    return rewards