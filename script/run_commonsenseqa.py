import numpy as np
from datasets import load_dataset
from llm_models.wrappers.llama_model import LlamaModel

def load_commonsenseqa(split="validation"):
    """
    Returns a list of examples from CommonsenseQA, each with:
      - question: str
      - choices: list[str] of length 5
      - answerKey: one of "A"–"E"
    """
    raw = load_dataset("commonsense_qa", split=split)
    examples = []
    for ex in raw:
        # HuggingFace version has fields "question_stem", "choices/(text)", "answerKey"
        choices = [ex["choices"]["text"][i] for i in range(5)]
        examples.append({
            "question": ex["question_stem"],
            "choices": choices,
            "answerKey": ex["answerKey"]
        })
    return examples

def compute_diff(expert_npy, nonexpert_npy):
    expert_mean    = np.load(expert_npy)      # shape (n_layers, d_model)
    nonexpert_mean = np.load(nonexpert_npy)   # same shape
    return expert_mean - nonexpert_mean       # signed diff

def sparsify_topk(diff_matrix, topk):
    """
    Zero out all but the topk neurons (by |diff|) in each layer.
    """
    nm, dm = diff_matrix.shape
    out = np.zeros_like(diff_matrix)
    for l in range(nm):
        idx = np.argsort(np.abs(diff_matrix[l]))[-topk:]
        out[l, idx] = diff_matrix[l, idx]
    return out

def apply_edit(hidden_states, diff_matrix, alpha):
    """
    hidden_states: list of np.array, one per layer (shape d_model)
    diff_matrix:   np.array (n_layers × d_model)
    returns new list of hidden_states
    """
    return [
      hs + alpha * diff_matrix[i] 
      for i, hs in enumerate(hidden_states)
    ]

def format_prompt(q, choices):
    lines = [f"Q: {q}"]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65+i)}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)

def main():
    # ——— SETTINGS ———
    expert_npy    = "all_mean_8B.npy"
    nonexpert_npy = "none_all_mean_8B.npy"
    topk          = 20
    alpha         = 1.0
    model_path    = "meta-llama/Llama-3.2-8B-Instruct"  # adjust to your model
    # ————————————

    # 1) load & prep diff  
    diff = compute_diff(expert_npy, nonexpert_npy)
    diff = sparsify_topk(diff, topk)

    # 2) init model  
    model = LlamaModel(model_path)

    # 3) load CommonsenseQA  
    examples = load_commonsenseqa(split="validation")

    # 4) run eval  
    correct = 0
    for ex in examples:
        prompt = format_prompt(ex["question"], ex["choices"])

        # a) get original hidden states (list of length n_layers)
        hidden = model.get_hidden_states(prompt)

        # b) apply your fixed diff
        edited = apply_edit(hidden, diff, alpha)

        # c) generate logits from edited hidden states
        logits = model.generate_from_hidden_states(edited)
        #    assume logits is a 1-D array of size vocab

        # d) map logits → choice  
        choice_scores = []
        for i, c in enumerate(ex["choices"]):
            tok_id = model.tokenizer(c, add_special_tokens=False)["input_ids"][0]
            choice_scores.append(logits[tok_id])
        pred = chr(65 + int(np.argmax(choice_scores)))

        if pred == ex["answerKey"]:
            correct += 1

    acc = correct / len(examples)
    print(f"CommonsenseQA accuracy after editing: {acc:.4%}")

if __name__ == "__main__":
    main()
