# scripts/run_complete_experiment.py
import argparse
import numpy as np
import os, sys
sys.path.append(os.path.abspath("C:/Users/hsuch/OneDrive/桌面/llm-activation-experiments"))
from llm_models.wrappers.llama_model import LlamaModel
from evaluation.dataset.mmlu import MMLUDataset
from evaluation.hidden_states import extract_hidden_states
from analysis.diff_matrix import compute_diff_matrix
from utils.file_utils import save_json
from utils.config import load_config

def run_experiment(config):
    model = LlamaModel(config["model_path"])
    dataset = MMLUDataset(config["task"], split="test", cache_dir=config.get("cache_dir"))
    
    results = []
    accuracy_counts = {
        "character1": {"correct": 0, "total": 0, "E_count": 0},
        "character2": {"correct": 0, "total": 0, "E_count": 0},
        "character1_modified": {"correct": 0, "total": 0, "E_count": 0},
        "character2_modified": {"correct": 0, "total": 0, "E_count": 0},
    }
    
    char1 = config["character1"]
    char2 = config["character2"]
    
    # 另外，同時收集 hidden state 用於後續分析
    hidden_states_char1 = []
    hidden_states_char2 = []
    
    for item in dataset:
        context = item["context"]
        true_label = item["label"]
        
        # 生成原始答案與 hidden state
        answer1 = model.generate_answer(context, character=char1)
        answer2 = model.generate_answer(context, character=char2)
        prompt1 = model.template.format(context=context, character=char1)
        prompt2 = model.template.format(context=context, character=char2)
        hs1 = model.get_hidden_states(prompt1)[1:]
        hs2 = model.get_hidden_states(prompt2)[1:]
        hs1_last = np.stack([s.cpu().numpy()[0, -1, :] for s in hs1], axis=0)
        hs2_last = np.stack([s.cpu().numpy()[0, -1, :] for s in hs2], axis=0)
        
        # 保存 hidden state
        hidden_states_char1.append(hs1_last)
        hidden_states_char2.append(hs2_last)
        
        # 計算 diff matrix
        diff_matrix = compute_diff_matrix(hs1_last, hs2_last, config["top_k"], alpha=config["alpha"])
        
        # 使用 diff matrix 重生成答案
        modified_answer1 = model.generate_with_diff(prompt1, diff_matrix, max_tokens=1)
        modified_answer2 = model.generate_with_diff(prompt2, diff_matrix, max_tokens=1)
        
        results.append({
            "context": context,
            "true_label": true_label,
            "answer1": answer1,
            "answer2": answer2,
            "modified_answer1": modified_answer1,
            "modified_answer2": modified_answer2
        })
        for key, ans in zip(["character1", "character2", "character1_modified", "character2_modified"],
                             [answer1, answer2, modified_answer1, modified_answer2]):
            accuracy_counts[key]["total"] += 1
            if ans == true_label:
                accuracy_counts[key]["correct"] += 1
            if ans.upper() == "E":
                accuracy_counts[key]["E_count"] += 1
    
    # 計算 E_ratio
    for key, counts in accuracy_counts.items():
        total = counts["total"]
        counts["E_ratio"] = (counts["E_count"] / total * 100) if total > 0 else 0
    
    output = {
        "results": results,
        "accuracy": accuracy_counts,
        "task": config["task"]
    }
    output_path = f"{config['output_dir']}/{config['task']}_complete_experiment.json"
    save_json(output, output_path)
    print(f"Experiment completed. Results saved to {output_path}")
    
    # 存檔 hidden state 到指定目錄
    os.makedirs(config["hidden_states_output"], exist_ok=True)
    hidden_states_char1 = np.array(hidden_states_char1)
    hidden_states_char2 = np.array(hidden_states_char2)
    hs_file1 = os.path.join(config["hidden_states_output"], f"{config['task']}_char1_hs.npy")
    hs_file2 = os.path.join(config["hidden_states_output"], f"{config['task']}_char2_hs.npy")
    np.save(hs_file1, hidden_states_char1)
    np.save(hs_file2, hidden_states_char2)
    print(f"Hidden states saved to {hs_file1} 和 {hs_file2}")

def main():
    parser = argparse.ArgumentParser(description="Run complete experiment with diff matrix modification using config file.")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(config)

if __name__ == "__main__":
    main()
