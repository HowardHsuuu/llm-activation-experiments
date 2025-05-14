# models/wrappers/llama_model.py
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hooks import register_hooks

class LlamaModel:
    def __init__(self, model_name_or_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, tokenizer_class="PreTrainedTokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 或自行設定新 token，如 "[PAD]"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        # 使用原始專案的 prompt 模板
        self.template = (
            'Would you answer the following question with A, B, C, D or E?\n'
            'Question: {context}'
            'E) I am not sure.\n'
            '{character}, your answer among "A, B, C, D, E" is: '
        )

    def generate(self, prompt, max_tokens=1):
        tokens = self.tokenizer(prompt, return_tensors="pt", padding="longest")
        # 將 input_ids 與 attention_mask 移到模型所在設備
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            # temperature=0,
        )
        # 解碼時只取新增的 token 部分
        generated = self.tokenizer.decode(
            output_ids[0][len(input_ids[0]):],
            skip_special_tokens=True
        )
        return generated

    def generate_answer(self, context, character="", model_type="llama3"):
        prompt = self.template.format(context=context, character=character)
        tokens = 6 if model_type.lower() == "phi" else 1
        raw_answer = self.generate(prompt, max_tokens=tokens)
        # 若答案不合法，則重生成更長回答
        if raw_answer.strip().upper() not in ["A", "B", "C", "D"]:
            raw_answer = self.generate(prompt, max_tokens=8)
        return self.clean_answer(raw_answer)

    def get_hidden_states(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        return outputs.hidden_states

    def generate_with_diff(self, prompt, diff_matrices, max_tokens=1):
        with register_hooks(self.model, diff_matrices):
            result = self.generate(prompt, max_tokens=max_tokens)
        return result

    @staticmethod
    def clean_answer(answer):
        match = re.search(r"\b([A-E])\b", answer.upper())
        return match.group(1) if match else answer.strip().upper()
