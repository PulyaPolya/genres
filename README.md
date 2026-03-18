# Genre Classification and Clustering Pipeline

This repository contains the code used to build the genre classification and clustering pipeline developed as part of the Master's thesis:

**"Beat Tracking for Genre-Related Datasets Obtained by Clustering"**

## Overview

The project is divided into two main components:

---

## 1. AST Model (Genre Classification)

The first stage involves fine-tuning the **Audio Spectrogram Transformer (AST)** for a genre classification task.

This part of the repository includes:
- Code for fine-tuning the AST model
- A small-scale hyperparameter optimisation (HPO) setup

---

## 2. Embedding Extraction and Clustering

In the second stage, the fine-tuned AST model is used as an **encoder** to extract embeddings from its internal layers.

These embeddings are then used for:
- **Dimensionality Reduction (DR)**:
  - PCA
  - UMAP
  - t-SNE

- **Clustering**:
  - K-means
  - Agglomerative clustering
  - HDBSCAN

This pipeline enables the construction of **genre-related datasets** through unsupervised clustering, which are subsequently used for beat tracking experiments.
