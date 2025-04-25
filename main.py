import torch
import random
import numpy as np
import gc
import os
from unsloth import FastLanguageModel
from transformers import pipeline
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from tabulate import tabulate

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Model & Experiment Configurations
# ---------------------------
MODEL_VARIANTS = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
]

CONFIGS = [
    "stereoset", "political"
]

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# ---------------------------
# Model Loader
# ---------------------------
def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto"
    )
    return model, tokenizer

# ---------------------------
# Data Standardization
# ---------------------------
def standardize_record(example, config_name):
    if config_name in ["convai2_inferred", "light_inferred", "opensubtitles_inferred", "yelp_inferred"]:
        if "binary_label" not in example or "text" not in example:
            return None
        return {"text": example["text"], "bias_label": int(example["binary_label"]), "bias_type": "gender"}
    elif config_name == "image_chat":
        if not ("male" in example and "female" in example):
            return None
        label = 1 if (example["male"] and not example["female"]) else 0
        return {"text": example["caption"], "bias_label": label, "bias_type": "gender"}
    return None

def standardize_stereoset_record(example):
    context = example["context"]
    bias_type = example["bias_type"]
    if bias_type not in ["race", "profession"]:
        return None

    try:
        sent_data = example["sentences"]
        sentences = sent_data["sentence"]
        gold_labels = sent_data["gold_label"]  # list of integers: 0, 1, 2

        pairs = []
        for sentence, label in zip(sentences, gold_labels):
            if label == 1:  # stereotype
                pairs.append({
                    "text": f"{context} {sentence}",
                    "bias_label": 1,
                    "bias_type": bias_type
                })
            elif label == 0:  # anti-stereotype
                pairs.append({
                    "text": f"{context} {sentence}",
                    "bias_label": 0,
                    "bias_type": bias_type
                })
        return pairs if pairs else None
    except Exception as e:
        print(f"Parse error: {e}")
        return None

def standardize_political_bias_record(example):
    if "text" not in example or "label" not in example:
        return None
    label = example["label"]
    if label == 2:
        return {"text": example["text"], "bias_label": 0, "bias_type": "political"}
    elif label in [0, 1, 3, 4]:
        return {"text": example["text"], "bias_label": 1, "bias_type": "political"}
    return None

# ---------------------------
# Activation Extraction
# ---------------------------
def extract_layer_activations(model, tokenizer, texts, layer_indices):
    activations = {idx: [] for idx in layer_indices}

    def hook_fn(module, input, output, idx):
        output = output[0] if isinstance(output, tuple) else output
        output = output.detach().cpu().float()
        if output.dim() == 3:
            output = output.mean(dim=1)
        activations[idx].append(output.numpy())

    try:
        layers = model.model.model.layers
    except AttributeError:
        layers = model.model.layers

    hooks = [layers[idx].register_forward_hook(lambda m, i, o, idx=idx: hook_fn(m, i, o, idx))
             for idx in layer_indices]

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)

    for hook in hooks:
        hook.remove()
    return {k: np.vstack(v) for k, v in activations.items()}

# ---------------------------
# Visualization
# ---------------------------
def visualize_activations(activations, labels, title, output_dir="tsne_plots"):
    tsne = TSNE(n_components=2, random_state=42)
    activations_2d = tsne.fit_transform(activations)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=activations_2d[:, 0], y=activations_2d[:, 1], hue=labels, palette="viridis")
    plt.title(title)

    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.replace("/", "_").replace(" ", "_")
    plt.savefig(f"{output_dir}/{safe_title}.png")
    plt.close()

# ---------------------------
# Bias-CAV Training
# ---------------------------
def train_bias_cav_with_cv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': []}

    for train_idx, val_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        clf = LogisticRegression(max_iter=1000).fit(X_train, y[train_idx])
        y_pred = clf.predict(X_val)

        metrics['acc'].append(accuracy_score(y[val_idx], y_pred))
        metrics['pre'].append(precision_score(y[val_idx], y_pred))
        metrics['rec'].append(recall_score(y[val_idx], y_pred))
        metrics['f1'].append(f1_score(y[val_idx], y_pred))

    print(f"CV Metrics: Acc={np.mean(metrics['acc']):.2f}, Pre={np.mean(metrics['pre']):.2f}, "
          f"Rec={np.mean(metrics['rec']):.2f}, F1={np.mean(metrics['f1']):.2f}")

    scaler = StandardScaler().fit(X)
    return LogisticRegression(max_iter=1000).fit(scaler.transform(X), y), scaler

# ---------------------------
# Average Test Score Summary
# ---------------------------
def summarize_by_bias_type(results):
    from collections import defaultdict
    summary = defaultdict(list)
    for r in results:
        summary[r["Bias_Type"]].append(r["Test_Score"])

    print("\nAverage Test Scores by Bias Type:")
    for bias_type, scores in summary.items():
        print(f"  {bias_type.capitalize():<10}: {np.mean(scores):.3f}")

# ---------------------------
# Main Experiment (Layer-Wise Analysis)
# ---------------------------
def run_experiment():
    results = []

    for config_name in CONFIGS:
        print(f"\nProcessing {config_name}")
        if config_name == "stereoset":
            try:
                ds = load_dataset("stereoset", name="intersentence", split="validation")
            except Exception as e:
                print(f"Skipping {config_name}: {e}")
                continue
            raw_pairs = []
            for example in ds:
                recs = standardize_stereoset_record(example)
                if recs:
                    raw_pairs.extend(recs)
            samples = random.sample(raw_pairs, min(500, len(raw_pairs)))
        elif config_name == "political":
            try:
                ds = load_dataset("cajcodes/political-bias", split="train")
            except Exception as e:
                print(f"Skipping {config_name}: {e}")
                continue
            samples = [rec for ex in ds if (rec := standardize_political_bias_record(ex)) is not None]
            samples = random.sample(samples, min(500, len(samples)))
        else:
            try:
                ds = load_dataset("md_gender_bias", name=config_name, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"Skipping {config_name}: {e}")
                continue
            samples = [rec for ex in ds if (rec := standardize_record(ex, config_name)) is not None]
            samples = random.sample(samples, min(500, len(samples)))

        train_samples = samples[:int(0.8 * len(samples))]
        val_samples = samples[int(0.8 * len(samples)) :]

        for model_name in MODEL_VARIANTS:
            print(f"  Testing {model_name}")
            try:
                model, tokenizer = load_model(model_name)
            except Exception as e:
                print(f"    Failed to load {model_name}: {e}")
                continue

            layer_indices = [0, 8, 16, -1]

            train_texts = [x["text"] for x in train_samples]
            val_texts = [x["text"] for x in val_samples]
            train_labels = [x["bias_label"] for x in train_samples]
            val_labels = [x["bias_label"] for x in val_samples]
            bias_types = list(set(x["bias_type"] for x in train_samples + val_samples))

            train_acts = extract_layer_activations(model, tokenizer, train_texts, layer_indices)
            val_acts = extract_layer_activations(model, tokenizer, val_texts, layer_indices)

            for layer_idx in layer_indices:
                for bias_type in bias_types:
                    filtered_train = [i for i, x in enumerate(train_samples) if x["bias_type"] == bias_type]
                    filtered_val = [i for i, x in enumerate(val_samples) if x["bias_type"] == bias_type]
                    if not filtered_train or not filtered_val:
                        continue

                    X_train = train_acts[layer_idx][filtered_train]
                    y_train = np.array(train_labels)[filtered_train]
                    X_val = val_acts[layer_idx][filtered_val]
                    y_val = np.array(val_labels)[filtered_val]

                    clf, scaler = train_bias_cav_with_cv(X_train, y_train)
                    val_probs = clf.predict_proba(scaler.transform(X_val))[:, 1]
                    test_score = val_probs.mean()

                    visualize_activations(X_val, y_val,
                        f"{model_name} {config_name} {bias_type} Layer {layer_idx}")

                    results.append({
                        "Config": config_name,
                        "Bias_Type": bias_type,
                        "Model": model_name,
                        "Layer": layer_idx,
                        "Test_Score": test_score,
                        "Val_Size": len(filtered_val),
                    })

            del model, tokenizer, clf, scaler
            gc.collect()
            torch.cuda.empty_cache()

    print("\nBias Analysis Report (Layer-Wise):")
    print(tabulate(results, headers="keys", tablefmt="grid", floatfmt=".3f"))
    summarize_by_bias_type(results)

if __name__ == "__main__":
    run_experiment()
