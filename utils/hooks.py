# utils/hooks.py
from contextlib import contextmanager
import torch

@contextmanager
def register_hooks(model, diff_matrices: list):
    """
    為模型所有 Transformer decoder 層註冊 hook，
    在 forward pass 時將對應的 diff matrix 加到該層最後一個 token 的 hidden state 上。
    如果 diff matrix 的 hidden size 與該層的輸出不一致，則自動截斷以匹配。
    """
    # 找出所有 Transformer decoder 層 (假設名稱格式符合 "model.layers.X" 並且有兩個點)
    decoder_layers = [module for name, module in model.named_modules() if name.startswith("model.layers.") and name.count(".") == 2]
    if not decoder_layers:
        for name, module in model.named_modules():
            print(name)
        raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")
    if len(decoder_layers) != len(diff_matrices):
        raise ValueError(f"Number of difference matrices ({len(diff_matrices)}) does not match number of decoder layers ({len(decoder_layers)}).")
    
    hooks = []
    def create_hook(diff_matrix):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                last_token_idx = hidden_states.shape[1] - 1
                # 若 diff_matrix 的 hidden size 與該層不匹配，則截斷
                if diff_matrix.shape[-1] != hidden_states.shape[-1]:
                    diff_matrix_adjusted = diff_matrix[..., :hidden_states.shape[-1]]
                else:
                    diff_matrix_adjusted = diff_matrix
                diff_tensor = torch.tensor(diff_matrix_adjusted, device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)
                hidden_states[:, last_token_idx, :] += diff_tensor
                return (hidden_states,) + output[1:]
            else:
                last_token_idx = output.shape[1] - 1
                if diff_matrix.shape[-1] != output.shape[-1]:
                    diff_matrix_adjusted = diff_matrix[..., :output.shape[-1]]
                else:
                    diff_matrix_adjusted = diff_matrix
                diff_tensor = torch.tensor(diff_matrix_adjusted, device=output.device, dtype=output.dtype).unsqueeze(0)
                output[:, last_token_idx, :] += diff_tensor
                return output
        return hook

    for layer, diff_matrix in zip(decoder_layers, diff_matrices):
        hook = layer.register_forward_hook(create_hook(diff_matrix))
        hooks.append(hook)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()
