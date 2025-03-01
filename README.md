# bias-cavs
A framework for probing and explaining internal bias in large language models

This repository contains the code and materials for the Bias-CAV framework, a novel approach that leverages concept activation vectors to diagnose and explain internal gender bias in large language models (LLMs). Our method provides a layer-wise analysis of model activations to reveal where bias is introduced, amplified, or mitigated. In addition, we propose principled strategies for debiasing that are informed by our mathematical framework.

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

Large language models have demonstrated remarkable capabilities in natural language processing but often inherit subtle societal biases from their training data. Bias-CAVs extend the concept activation vector methodology to probe internal LLM representations. By extracting activations from key layers and training logistic regression classifiers, our framework computes a bias probability \(P_m(e)\) for each embedding, thus pinpointing bias propagation within the model.

We also propose debiasing strategies, including a direct corrective approach during inference and a regularization term added to the training objective. Both strategies leverage our closed-form perturbation derived from the classifier's decision boundary.

## Features

- **Layer-Wise Analysis:** Extracts and analyzes activations from multiple layers (e.g., layers 0, 8, 16, and final) to understand the internal bias propagation.
- **Robust Bias Detection:** Uses a logistic regression-based classifier (Bias-CAV) that consistently achieves strong cross-validation metrics.
- **Debiasing Proposals:** Provides two complementary debiasing strategies:
  - A direct corrective perturbation during inference.
  - A training objective augmentation with a bias regularization term.
- **Reproducibility:** Includes all scripts and instructions necessary to reproduce our experimental results.

## Installation

To set up the project, first clone this repository:

```bash
git clone https://github.com/yourusername/bias-cavs.git
cd bias-cavs
```

Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure that you have Python 3.10 or later installed. This project is optimized for running on a CUDA-enabled GPU (e.g., NVIDIA A100-SXM4-40GB) with Google Colab Pro.

## Usage

The main experimental pipeline is implemented in the script `run_experiment.py`. This script loads the MDGenderBias dataset from the Hugging Face Hub, extracts activations from selected layers of multiple LLM variants using the Unsloth library, and evaluates bias detection performance using five-fold cross-validation.

To run the experiments, simply execute:

```bash
python main.py
```

The script will output cross-validation metrics (Accuracy, Precision, Recall, F1 Score) for each model and configuration, and generate visualizations (e.g., t-SNE plots) that are saved in the `tsne_plots/` directory.

## Datasets

This work uses the MDGenderBias dataset, which is a multi-dimensional corpus capturing gender bias along various axes (ABOUT, AS, TO). We utilize five configurations:
- `convai2_inferred`
- `light_inferred`
- `opensubtitles_inferred`
- `yelp_inferred`
- `image_chat`

The dataset is available from the Hugging Face Hub, and our code applies standardized preprocessing to ensure consistent evaluation across configurations.

## License

This project is released under the Apache License 2.0. While the Unsloth library is licensed under Apache 2.0 and the MDGenderBias dataset is available under the MIT license, the combined work in this repository adopts the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.

## Citation

This work is currently submitted to a conference. Please stay tuned.

```
@misc{catapang2025biascavs,
  author = {Catapang, Jasper Kyle},
  title = {Bias-CAVs: Explainable Bias Discovery in Large Language Models Using Concept Activation Vectors},
  year = {2025},
  howpublished = {\url{https://github.com/jcatapang/bias-cavs/}},
  note = {Work in progress; manuscript submitted for review}
}
```

## Acknowledgements

We thank the [Unsloth](https://github.com/unslothai/unsloth) team for providing an efficient framework for low-bit quantization and model patching. Special thanks to contributors and open-source projects that have made this work possible.

For further details and updates, please visit our [GitHub repository](https://github.com/jcatapang/bias-cavs).
