# evaluation/hidden_states.py
import numpy as np

def extract_hidden_states(model, dataset) -> np.ndarray:
    """
    對資料集每個樣本，使用 model.get_hidden_states 提取 hidden state，
    並取出每層最後一個 token 的 hidden state，返回形狀為 (samples, num_layers, hidden_size) 的 numpy array。
    """
    all_states = []
    for item in dataset:
        prompt = model.template.format(context=item["context"], character="")
        states = model.get_hidden_states(prompt)
        last_token_states = [s[0, -1, :].cpu().numpy() for s in states]
        all_states.append(last_token_states)
    return np.array(all_states)
