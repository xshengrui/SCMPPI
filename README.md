# SCMPPI:Supervised Contrastive Multimodal Framework for Predicting Protein-Protein Interactions

## Overview

SCMPPI (Supervised Contrastive Multimodal Framework for Predicting Protein-Protein Interactions) is a novel supervised contrastive multimodal framework designed for predicting protein-protein interactions (PPI). This project aims to address the limitations of traditional experimental methods and existing computational methods in cross-modal feature fusion and false negative suppression. By effectively integrating sequence-based features (AAC, DPC, ESMC-CKSAAP) with network topology (Node2Vec embeddings) and combining enhanced contrastive learning strategies with negative sample filtering, SCMPPI achieves excellent predictive performance.

## Project Structure

The project includes the following main directories:

* `Data/`: Stores protein sequence and interaction data used for model training and testing.
    * `H.pylori/`: Data related to Helicobacter pylori.
    * `Human/`: Data related to human proteins.
    * `Multi-species/`: Data on interactions across multiple species.
    * `PIPR-cut/`: PIPR-cut dataset.
    * `Yeast/`: Data related to yeast proteins.
* `MF-encoder/`: Contains modules for protein feature encoding.
    * `graph-encoding/`: Scripts for feature encoding based on graph structures (such as protein interaction networks), including algorithms like `node2vec`.
    * `seq-encoding/`: Scripts for feature encoding based on protein sequences, such as `a_d_coding.py` and `ks_esmc300_coding.py`.
* `baseline/`: Stores performance results of various baseline models (such as Decision Tree, Random Forest, SVM, DeepFE-PPI, KSGPPI) on different datasets.
* `run/`: Contains YAML configuration files for configuring and running model training and evaluation.
* `scripts/`: Core scripts, including functions for data loading, model training, inference, and result visualization.
    * `SCMPPI.py`: Implementation of the main model or framework.
    * `data_loader.py`: Data loading and preprocessing.
    * `inference.py`: Model inference.
    * `main.py`: Main entry for model training.
    * `util.py`: General utility functions.
    * `visualize_embeddings.py`: Visualization of feature embeddings.

## Features

* **Excellent Performance:** Achieves advanced accuracy (98.13%) and AUC (99.69%) on eight benchmark datasets, and demonstrates outstanding cross-species generalization ability (AUC > 99%).
* **Multimodal Feature Fusion:** Effectively integrates sequence-based features (such as AAC, DPC, ESMC-CKSAAP) and graph information (such as Node2Vec embeddings) for prediction.
* **Enhanced Contrastive Learning:** Introduces an enhanced contrastive learning strategy, combined with negative sample filtering, to improve predictive accuracy and suppress false negatives.
* **Modular Design:** Clear code structure for easy understanding, modification, and extension.
* **Multiple Encoders:** Provides methods for protein feature encoding based on sequences and graphs.
* **Baseline Model Comparison:** Offers detailed comparison results with various traditional machine learning baseline models (such as Decision Tree, Random Forest, SVM, DeepFE-PPI, KSGPPI).
* **Configurable Operation:** Flexibly manages experimental parameters through `YAML` configuration files.

## Quick Start

### Environment Setup

This project relies on Python 3. It is recommended to create a virtual environment using `conda` or `venv`:

```bash
# Create a new conda environment
conda create -n scmppi_env python=3.8
conda activate scmppi_env

# Or use venv
# python3 -m venv scmppi_env
# source scmppi_env/bin/activate  # Linux/macOS
# .\scmppi_env\Scripts\activate   # Windows
```

### Data Preparation

Place the raw protein sequence and interaction data in the corresponding subfolders under the `Data/` directory.

## Running Examples

### Training

You can run the entire *training* process through the `scripts/main.py` script. Before running, please check the `*.yml` configuration files in the `run/` directory to adjust parameters.

For example, to run yeast PPI prediction training:

```bash
python scripts/main.py --config run/yeast-config.yml
```

### Inference

You can run the entire *inference* process through the `scripts/inference.py` script. Before running, please check the `*.yml` configuration files in the `run/` directory to adjust parameters.

For example, to run yeast PPI prediction inference:

```bash
python scripts/inference.py --config run/yeast-config.yml
```

### Feature Encoding

If you need to regenerate protein features, please adjust the commands according to your data path and requirements:

* **Sequence Encoding Example:**
    ```bash
    # For example, use a_d_coding.py to encode yeast protein sequences
    python MF-encoder/seq-encoding/a_d_coding.py
    
    # For example, use ks_esmc300_coding.py to encode yeast protein sequences
    python MF-encoder/seq-encoding/ks_esmc300_coding.py
    ```

* **Graph Encoding Example:**
    ```bash
    # Node2vec usually requires a network file as input and outputs an embedding file
    python MF-encoder/graph-encoding/node2vec.py 
    ```

## Contribution

Contributions and suggestions to this project are welcome! If you have any questions, bug reports, or feature requests, please submit them through GitHub Issues.

