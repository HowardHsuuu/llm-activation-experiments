#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import sys
# add repo root for imports
repo_path = os.path.abspath("/content/llm-activation-experiments")
sys.path.append(repo_path)
from llm_models.wrappers.llama_model import LlamaModel
from datasets import load_dataset

LABEL_MAPPING = ["A", "B", "C", "D", "E", "F"]


def load_commonsenseqa(split="validation"):
    raw = load_dataset("commonsense_qa", split=split)
    examples = []
    for ex in raw:
        question = ex.get("question", ex.get("question_stem"))
        choices = ex["choices"]["text"]
        examples.append({
            "question": question,
            "choices": choices,
            "answerKey": ex["answerKey"]
        })
    return examples


def compute_diff(expert_npy, nonexpert_npy):
    expert = np.load(expert_npy)
    nonexp = np.load(nonexpert_npy)
    diff = expert - nonexp
    diff = np.squeeze(diff)
    if diff.ndim > 2:
        axes = tuple(range(2, diff.ndim))
        diff = diff.mean(axis=axes)
    return diff  # shape: (n_layers, hidden_size)


def select_layer_range(diff_matrix, start, end):
    mask = np.zeros_like(diff_matrix)
    n_layers = diff_matrix.shape[0]
    if n_layers >= end:
        mask[start:end] = diff_matrix[start:end]
    else:
        mask[:] = diff_matrix
        print(f"[Warning] only {n_layers} layers available, using all layers")
    return mask


def sparsify_topk(diff_matrix, topk):
    out = np.zeros_like(diff_matrix)
    for i, layer in enumerate(diff_matrix):
        if topk > 0:
            idx = np.argsort(np.abs(layer))[-topk:]
            out[i, idx] = layer[idx]
        else:
            out[i] = layer
    return out


def format_prompt(question, choices):
    lines = [f"Would you answer the following question with A, B, C, D, E or F?\nQuestion: {question}"]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65+i)}) {c}")
    lines.append("F) I'm not sure.")
    lines.append('Now you are an honest expert, your only answer with one token among "A, B, C, D, E, F" is:')
    return "\n".join(lines)


def main():
    # config
    expert_npy    = "diff_mean_3B.npy"
    nonexpert_npy = "none_diff_mean_3B.npy"
    layer_start   = 7
    layer_end     = 17
    topk          = 20
    alpha         = 4.0
    model_name    = "shared/llama3/3B"

    # prepare diff matrix
    diff_np = compute_diff(expert_npy, nonexpert_npy)
    diff_np = select_layer_range(diff_np, layer_start, layer_end)
    diff_np = sparsify_topk(diff_np, topk)
    # Skip embedding layer diff (first row) to match decoder layers
    diff_np = diff_np[1:]
    # convert to tensor for generate_with_diff
    diff_tensor = torch.from_numpy(diff_np).float() * alpha

    # init model
    model = LlamaModel(model_name)

    # load data
    examples = load_commonsenseqa(split="validation")

    # setup counters
    counts = {
        "original": {"correct": 0, "total": 0, "F":0},
        "modified": {"correct": 0, "total": 0, "F":0}
    }

    # evaluation loop
    for ex in examples:
        prompt = format_prompt(ex["question"], ex["choices"])
        #print(prompt)

        # original generation (no diff)
        out_orig = model.generate(
            prompt,
            max_tokens=1
        ).strip().upper()
        pred_orig = next((c for c in out_orig if c in LABEL_MAPPING), None)
        counts["original"]["total"] += 1
        if pred_orig == ex["answerKey"]:
            counts["original"]["correct"] += 1
            #print("acc")
        elif pred_orig == "F":
            counts["original"]["F"] += 1
            #print("not sure")
        else:
            print(pred_orig, end=" ")

        # modified generation with fixed diff matrix
        out_mod = model.generate_with_diff(
            prompt,
            diff_tensor.to(model.device),
            max_tokens=1
        ).strip().upper()
        pred_mod = next((c for c in out_mod if c in LABEL_MAPPING), None)
        counts["modified"]["total"] += 1
        if pred_mod == ex["answerKey"]:
            counts["modified"]["correct"] += 1
        elif pred_mod == "F":
            counts["modified"]["F"] += 1
        else:
            print(pred_mod, end=" ")

    print()
    orig_acc = counts["original"]["correct"] / counts["original"]["total"] * 100
    orig_F = counts["original"]["F"] / counts["original"]["total"] * 100
    mod_acc  = counts["modified"]["correct"] / counts["modified"]["total"] * 100
    mod_F = counts["modified"]["F"] / counts["modified"]["total"] * 100
    print(f"Original CommonsenseQA accuracy: {orig_acc:.2f}%")
    print(f"F ratio: {orig_F:.2f}%")
    print(f"Modified CommonsenseQA accuracy: {mod_acc:.2f}%")
    print(f"F ratio: {mod_F:.2f}%")

if __name__ == "__main__":
    main()