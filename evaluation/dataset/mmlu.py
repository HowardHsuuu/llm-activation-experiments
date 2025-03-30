# evaluation/datasets/mmlu.py
from datasets import load_dataset
from torch.utils.data import Dataset

class MMLUDataset(Dataset):
    def __init__(self, task, split="test", cache_dir=None):
        self.dataset = load_dataset("lukaemon/mmlu", task, split=split, cache_dir=cache_dir, trust_remote_code=True)
        self.task = task.replace("_", " ")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        context = (
            f"{item['input']}\n"
            f"A) {item['A']}\n"
            f"B) {item['B']}\n"
            f"C) {item['C']}\n"
            f"D) {item['D']}\n"
        )
        label = item['target']
        return {'context': context, 'label': label, 'task': self.task}
