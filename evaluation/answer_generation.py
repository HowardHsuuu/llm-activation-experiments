# evaluation/answer_generation.py
import re

def clean_answer(answer: str) -> str:
    match = re.search(r"\b([A-E])\b", answer.upper())
    return match.group(1) if match else answer.strip().upper()

def handle_invalid_answer(model, prompt: str, true_label: str, max_tokens: int = 8) -> str:
    """
    若生成的回答不合法，則進行重生成並嘗試提取合法答案。
    """
    generated_output = model.generate(prompt, max_tokens=max_tokens)
    cleaned = clean_answer(generated_output)
    if cleaned not in ["A", "B", "C", "D"]:
        return generated_output
    return cleaned
