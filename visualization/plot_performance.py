# visualization/plot_performance.py
import matplotlib.pyplot as plt
import json
from utils.config import load_config

def plot_accuracy(config_path: str, results_paths: list, output_path: str):
    config = load_config(config_path)
    accuracies = []
    labels = []
    for path in results_paths:
        with open(path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            acc_data = result.get('accuracy', {}).get("character1", {})
            total = acc_data.get("total", 0)
            correct = acc_data.get("correct", 0)
            accuracy_percentage = (correct / total * 100) if total > 0 else 0
            accuracies.append(accuracy_percentage)
            labels.append(result.get("task", "Unknown"))
    plt.figure(figsize=(8,6))
    plt.bar(labels, accuracies, color='skyblue', edgecolor='black')
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy on MMLU Tasks")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    config_path = "./config/experiment_config.yaml"
    results_files = [
        "./results/complete/abstract_algebra_complete_experiment.json",
        "./results/complete/anatomy_complete_experiment.json",
    ]
    plot_accuracy(config_path, results_files, "./results/plots/accuracy.png")
