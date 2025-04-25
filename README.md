# bias-cavs  
A framework for probing and explaining internal bias in large language models

This repository contains the code and materials for the **Bias-CAV** framework, featured in the research paper _"Explaining Bias in Internal Representations of Large Language Models via Concept Activation Vectors."_ Bias-CAV is a novel diagnostic tool that leverages concept activation vectors (CAVs) to probe the **internal representations** of large language models (LLMs) for **gender**, **racial**, **professional**, and **political** biases. Our method conducts a **layer-wise analysis** to uncover where bias is introduced, amplified, or mitigated—enabling both insight and intervention.

The manuscript has been accepted as a long paper at the **29th International Conference on Natural Language & Information Systems** ([NLDB 2025](https://www.jaist.ac.jp/event/nldb2025/)), to be held in July 2025 in Kanazawa, Japan, and will be published by **Springer**.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

While large language models have revolutionized natural language processing, they often inherit **societal biases** from training data. Bias-CAV extends the concept activation vector methodology to diagnose these internal biases across different sociolinguistic domains.

By extracting activations from key LLM layers and training simple linear classifiers, Bias-CAV computes a bias probability \(P_m(e)\) for each embedding. Our analysis identifies the layers where bias is most detectable and provides mechanisms for bias correction.

We further propose **two debiasing strategies**:
- A direct, closed-form **perturbation during inference**.
- A **training-time regularization penalty** that discourages biased internal activations.

## Features

- **Multi-Domain Bias Coverage**: Detects **gender**, **race**, **profession**, and **political** bias across varied datasets.
- **Layer-Wise Probing**: Explains how and where bias emerges inside LLMs by probing activations at early, middle, and final layers.
- **Closed-Form Debiasing**: Derives optimal perturbations to reduce internal bias without performance degradation.
- **Training Objective Augmentation**: Introduces a bias penalty term for fine-tuning with reduced internal bias.
- **Cross-Model Analysis**: Evaluated on multiple quantized models (e.g., LLaMA-3, Mistral, Phi, Gemma) via the Unsloth library.
- **Reproducibility**: All code and settings required to replicate our experiments are provided.

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/yourusername/bias-cavs.git
cd bias-cavs
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10+ is required. This project is GPU-optimized (recommended: NVIDIA A100 via Google Colab Pro).

## Usage

The pipeline is orchestrated in `run_experiment.py`. It loads the datasets, extracts layer activations, trains bias classifiers, and evaluates model performance using five-fold CV.

To run the full experiment:

```bash
python main.py
```

Generated metrics and visualizations (e.g., t-SNE plots, bias scores per layer) will be saved in output folders.

## Datasets

We evaluate bias across four dimensions using the following datasets:

### Gender Bias
From the [MDGenderBias](https://huggingface.co/datasets/md_gender_bias) corpus:
- `convai2_inferred`
- `light_inferred`
- `opensubtitles_inferred`
- `yelp_inferred`
- `image_chat`

### Race & Profession Bias
- **Stereoset**: Intersentence split for race- and profession-related stereotypes. Reformatted for binary classification.

### Political Bias
- **Political Bias Dataset**: Categorized as neutral vs. non-neutral (liberal/conservative) statements.

All datasets are automatically downloaded and preprocessed via Hugging Face.

## License

This project is licensed under the **Apache 2.0 License**.  
Dependencies include:
- Unsloth (Apache 2.0)
- MDGenderBias (MIT License)

See the [LICENSE](LICENSE) file for full details.

## Citation

If you use this framework in your research, please cite the following:

```
@misc{catapang2025biascavs,
  author = {Catapang, Jasper Kyle},
  title = {Bias-CAVs: Explainable Bias Discovery in Large Language Models Using Concept Activation Vectors},
  year = {2025},
  howpublished = {\url{https://github.com/jcatapang/bias-cavs}},
  note = {Accepted at NLDB 2025, Kanazawa, Japan}
}
```

## Acknowledgements

Special thanks to the [Unsloth](https://github.com/unslothai/unsloth) team for enabling efficient low-bit quantized model loading. We also thank the creators of MDGenderBias, StereoSet, and the Political Bias dataset for making this research possible.
