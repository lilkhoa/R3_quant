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
    Reward format correctness to break the 'answer-without-think' mode collapse.

    MODE COLLAPSE FIX (Fix A): The previous version gave +0.1 just for having
    <answer>, so all completions that skip <think> and go straight to <answer>
    got the same 0.2 reward → zero variance → GRPO cannot learn format.

    New scheme creates a strong pressure toward using <think>:
      - Has <answer> but NO <think> (lazy shortcut): -0.3  ← penalise this
      - Has <think> but no <answer>:                  +0.1  ← at least trying
      - Has both but wrong order:                     +0.1
      - Has both in correct order (<think>→<answer>): +0.6  ← target behaviour

    The -0.3 vs +0.6 gap (0.9 total) creates strong differential pressure.
    Even a single completion that accidentally uses <think> will have a different
    reward from the others, restoring reward_std > 0.
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp

        has_think_open  = "<think>"  in content
        has_think_close = "</think>" in content
        has_answer_open  = "<answer>"  in content
        has_answer_close = "</answer>" in content

        has_any_think  = has_think_open  or has_think_close
        has_any_answer = has_answer_open or has_answer_close

        if has_any_answer and not has_any_think:
            # Lazy shortcut: gave answer without reasoning → penalise
            reward = -0.3
        elif has_any_think and not has_any_answer:
            # Started reasoning but forgot to state answer → small credit
            reward = 0.1
        elif has_any_think and has_any_answer:
            # Both present: check ordering
            think_pos  = content.find("<think>")
            answer_pos = content.find("<answer>")
            if think_pos != -1 and answer_pos != -1 and think_pos < answer_pos:
                reward = 0.6   # Correct structure: think then answer
            else:
                reward = 0.1   # Both present but wrong order
        else:
            # Completely unstructured output (no tags at all)
            reward = 0.0

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
    Clean binary accuracy: 1.0 if the extracted answer letter matches ground truth, else 0.0.

    FIX (Fix C): The previous version had a consolation score (0.3-0.4) based on
    the correct letter appearing in think content. But since the model currently
    never produces <think> content, think_text is always "", making the consolation
    branch dead code. Remove it to keep the reward clean and interpretable.
    The format and reasoning rewards already provide the 'partial credit' signal;
    accuracy should remain a clean binary signal.
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

def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """
    Encourage chain-of-thought reasoning without punishing all degenerate outputs equally.

    FIX (Bug #2): The previous version returned -1.0 for ALL short completions.
    At early training, every completion is short/garbage, so reward_std=0 across
    all num_generations completions → advantage=0 → GRPO loss=0 → no learning.

    New design creates variance even in degenerate states:
    - No <think> content at all (truly empty):  0.0  (neutral, not penalized)
    - Has <think> but < 50 chars of reasoning:  0.0  (neutral)
    - 50–200 chars of reasoning:               +0.2  (getting there)
    - 200+ chars of solid reasoning:           +0.3  (bonus for deep reasoning)

    This ensures different completions in a batch can have different values,
    giving GRPO the variance it needs to compute a non-zero loss.
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        think_content = extract_think_content(content)
        reasoning_length = len(think_content.replace(" ", "").replace("\n", "").replace("\t", ""))

        if reasoning_length >= 200:
            rewards.append(0.3)   # Deep reasoning
        elif reasoning_length >= 50:
            rewards.append(0.2)   # Moderate reasoning
        else:
            rewards.append(0.0)   # Too short or empty: neutral (no penalty)

    return rewards

def logic_structure_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward structured reasoning in <think> content.

    FIX (Fix B): The previous version gave -0.3 when think_content was EMPTY.
    Since the model currently never produces <think> content, ALL completions
    got -0.3 → reward_std=0 → this reward contributes nothing but a constant
    negative offset that hurts exploration.

    New scheme: only evaluate completions that actually attempt reasoning.
    Empty think → 0.0 (neutral). Only penalise BAD reasoning (filler without
    logical keywords), and reward GOOD reasoning.

      - No <think> content (empty):               0.0  ← neutral, not punished
      - Has think content, < 2 logic keywords:   -0.2  ← filler text penalty
      - Has think content, 2+ logic keywords:    +0.3  ← structured reasoning
    """
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
            # Model skipped <think> entirely: neutral (format_reward already penalises this)
            rewards.append(0.0)
            continue

        keyword_count = sum(
            len(re.findall(pattern, think_content, re.IGNORECASE))
            for pattern in logical_keywords
        )

        if keyword_count >= 2:
            rewards.append(0.3)   # Structured reasoning
        else:
            rewards.append(-0.2)  # Has think content but it's filler

    return rewards

_log_counter = 0

def logging_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Logging-only reward: always returns 0.0 so it never affects reward variance.

    Fires every 10 reward calls (≈ every 2 optimizer steps with 5 reward funcs
    and num_generations=4 → 20 calls/step; log every 10 ≈ every ~0.5 step for
    fine-grained monitoring).

    Prints (without ANY truncation):
      • The full prompt / question (extracted from the `prompts` kwarg)
      • The full ground truth answer
      • The full model completion text
    """
    global _log_counter
    _log_counter += 1

    # How often to log.
    # Each optimizer step ≈ num_generations calls per reward func.
    # With 5 funcs and num_generations=4: 20 calls/step.
    # Log every 10 calls → roughly every ~0.5 optimizer steps (very frequent,
    # but cheap since it's just a print). Adjust if logs are too noisy.
    LOG_EVERY_N_CALLS = 10

    if _log_counter % LOG_EVERY_N_CALLS == 0:
        approx_step = _log_counter // 20  # ~optimizer step
        print("\n" + "🔥" * 36)
        print(f"🔍 [REWARD LOGGER  |  ~step {approx_step}  |  call #{_log_counter}]")
        print("🔥" * 36)

        # ── Ground truth ────────────────────────────────────────────────
        gt = ground_truth[0] if ground_truth else "N/A"
        print(f"🎯 GROUND TRUTH: {gt}")

        # ── Question / prompt (passed by GRPOTrainer as 'prompts' kwarg) ─
        prompts = kwargs.get("prompts", None)
        if prompts is not None:
            prompt_repr = prompts[0]
            if isinstance(prompt_repr, list):
                # Conversational format: list of message dicts
                # Extract the user turn text content
                for msg in prompt_repr:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # Multi-modal: find the text part
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    print(f"\n📥 QUESTION:\n{part['text']}")
                                    break
                        else:
                            print(f"\n📥 QUESTION:\n{content}")
                        break
            else:
                print(f"\n📥 QUESTION:\n{prompt_repr}")

        # ── Full model completion (NO truncation) ───────────────────────
        print("\n🤖 FULL MODEL OUTPUT:")
        content = completions[0][0]["content"] if isinstance(completions[0], list) else completions[0]
        print(content)  # never truncate — full output every time
        print("🔥" * 36 + "\n")

    return [0.0] * len(completions)