import re

# ============================================================================
# PHASE 1: Visual Vocabulary Dictionary
# ============================================================================

VISUAL_VOCABULARY = {
    # Spatial Keywords (location and positioning)
    "spatial": [
        r"\btop\b", r"\bbottom\b", r"\bleft\b", r"\bright\b",
        r"\bcenter\b", r"\bcentre\b", r"\bcorner\b", r"\bbackground\b",
        r"\bupper-left\b", r"\bupper-right\b", r"\blower-left\b", r"\blower-right\b",
        r"\bmiddle\b", r"\bedge\b", r"\bside\b", r"\barea\b",
        r"\bposition\b", r"\blocation\b", r"\bplaced\b", r"\bsituated\b",
    ],
    
    # Attribute Keywords (colors, shapes, visual properties)
    "attributes": [
        # Colors
        r"\bred\b", r"\bblue\b", r"\byellow\b", r"\bgreen\b", r"\borange\b",
        r"\bpurple\b", r"\bpink\b", r"\bblack\b", r"\bwhite\b", r"\bgray\b",
        r"\bbrown\b", r"\bcolor\b", r"\bcolour\b",
        # Shapes
        r"\bcircle\b", r"\bsquare\b", r"\btriangle\b", r"\brectangle\b",
        r"\bcylinder\b", r"\bsphere\b", r"\bshape\b", r"\bpolygon\b",
        # Lines and visual elements
        r"\bline\b", r"\bstraight\b", r"\bcurve\b", r"\bslope\b",
        r"\bangle\b", r"\barrow\b", r"\bdot\b", r"\bsymbol\b",
    ],
    
    # Chart / Diagram / Graph Keywords
    "chart": [
        r"\bgraph\b", r"\bchart\b", r"\btable\b", r"\bdiagram\b",
        r"\bx-axis\b", r"\by-axis\b", r"\baxis\b", r"\bscale\b",
        r"\bbar\b", r"\btrend\b", r"\bcurve\b", r"\bplot\b",
        r"\bdata\b", r"\bvalue\b", r"\bpoint\b", r"\bline\b",
        r"\baxis\b", r"\blabel\b", r"\blegend\b", r"\bgrid\b",
    ],
}

# ============================================================================
# Judge 1: Format Judge (from V1 - kept stable)
# ============================================================================

def extract_xml_answer(text: str) -> str:
    """Extract content within <answer>...</answer> tags."""
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def extract_think_content(text: str) -> str:
    """Extract content within <think>...</think> tags."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        content = content.strip()
        pattern = r"^<think>.*?</think>\s*<answer>[A-Ea-e]</answer>$"
        
        if re.match(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards

# ============================================================================
# Judge 2: Accuracy Judge (from V1 - kept stable)
# ============================================================================

def _extract_answer_letter(text: str) -> str:
    """Extract answer letter (A-F) from text, prioritizing letters in parentheses."""
    # Priority 1: Letters in parentheses
    pattern_parens = r'\(([A-Fa-f])\)'
    match = re.search(pattern_parens, text)
    if match:
        return match.group(1).upper()
    
    # Priority 2: Standalone letters surrounded by whitespace or punctuation
    pattern_standalone = r'(?:^|\s|:|,|。|、|;)([A-Fa-f])(?:\s|:|,|\.|\?|!|。|、|;|$)'
    match = re.search(pattern_standalone, text)
    if match:
        return match.group(1).upper()
    
    return ""

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Judge 2: Accuracy Judge
    Binary reward: 1.0 if extracted answer matches ground truth, else 0.0.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        answer_text = extract_xml_answer(content)
        pred_letter = _extract_answer_letter(answer_text)
        truth_letter = _extract_answer_letter(truth)
        if not truth_letter:
            truth_letter = truth.upper().strip()
        rewards.append(1.0 if pred_letter == truth_letter else 0.0)

    return rewards

# ============================================================================
# Judge 3: Vision Judge (NEW - Visual Grounding)
# ============================================================================

def vision_reward_func(completions, **kwargs) -> list[float]:
    """
    Judge 3: Vision Judge (NEW)
    Force the model to prove it is analyzing the image by detecting visual keywords
    in the reasoning (<think> section only).
    
    Scoring logic:
      - No visual keywords in <think>:           -0.4  <- solving blindly
      - Only 1-2 visual keywords:                +0.1  <- partial grounding
      - 3+ visual keywords in combination:       +0.5  <- strong visual grounding
    """
    rewards = []
    
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        think_content = extract_think_content(content)
        
        # Only evaluate visual keywords in <think> section
        # If no <think> section, no vision reward
        if not think_content:
            rewards.append(0.0)  # No reasoning attempted
            continue
        
        # Count visual keywords across all three categories
        total_keywords = 0
        
        for category_keywords in VISUAL_VOCABULARY.values():
            for pattern in category_keywords:
                matches = re.findall(pattern, think_content, re.IGNORECASE)
                total_keywords += len(matches)
        
        # Reward based on visual keyword count
        if total_keywords == 0:
            # Heavy penalty: model is solving without looking at the image
            rewards.append(-0.4)
        elif total_keywords <= 2:
            # Small reward: minimal visual grounding
            rewards.append(0.1)
        else:
            # Strong reward: multiple visual keywords indicate careful image analysis
            rewards.append(0.5)
    
    return rewards

# ============================================================================
# Logging Function (from V1 - kept for monitoring)
# ============================================================================

_log_counter = 0

def logging_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Logging-only reward: always returns 0.0, provides monitoring visibility.
    Logs every ~0.5 optimizer steps for fine-grained monitoring.
    """
    global _log_counter
    _log_counter += 1

    LOG_EVERY_N_CALLS = 10

    if _log_counter % LOG_EVERY_N_CALLS == 0:
        approx_step = _log_counter // 20
        print("\n" + "🔥" * 36)
        print(f"🔍 [V2 VISION LOGGER  |  ~step {approx_step}  |  call #{_log_counter}]")
        print("🔥" * 36)

        gt = ground_truth[0] if ground_truth else "N/A"
        print(f"🎯 GROUND TRUTH: {gt}")

        prompts = kwargs.get("prompts", None)
        if prompts is not None:
            prompt_repr = prompts[0]
            if isinstance(prompt_repr, list):
                for msg in prompt_repr:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    print(f"\n📥 QUESTION:\n{part['text']}")
                                    break
                        else:
                            print(f"\n📥 QUESTION:\n{content}")
                        break
            else:
                print(f"\n📥 QUESTION:\n{prompt_repr}")

        print("\n🤖 FULL MODEL OUTPUT:")
        content = completions[0][0]["content"] if isinstance(completions[0], list) else completions[0]
        print(content)
        
        # NEW: Show vision keyword analysis
        think_content = extract_think_content(content)
        if think_content:
            keyword_counts = {}
            for category, keywords in VISUAL_VOCABULARY.items():
                count = sum(
                    len(re.findall(pattern, think_content, re.IGNORECASE))
                    for pattern in keywords
                )
                keyword_counts[category] = count
            
            print(f"\n👁️ VISION ANALYSIS:")
            print(f"   Spatial keywords: {keyword_counts.get('spatial', 0)}")
            print(f"   Attribute keywords: {keyword_counts.get('attributes', 0)}")
            print(f"   Chart keywords: {keyword_counts.get('chart', 0)}")
            print(f"   Total: {sum(keyword_counts.values())}")
        
        print("🔥" * 36 + "\n")

    return [0.0] * len(completions)
