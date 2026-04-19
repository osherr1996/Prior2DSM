# Prior2DSM – Test-Time Adaptation

This repository demonstrates a **test-time adaptation (TTO)** pipeline for converting **relative depth maps into metric DSM (Digital Surface Model)** using **DINOv3 features + LoRA fine-tuning**.

<p align="center">
  <a href="https://arxiv.org/abs/2604.02009">
    <img src="https://img.shields.io/badge/arXiv-2604.02009-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/spaces/osherr/Prior2DSM">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow.svg" alt="Hugging Face">
  </a>
</p>

## Overview

The method refines a monocular depth prediction by learning a **per-image scale and bias** using sparse **prior elevation data**.
It leverages:

* **DINOv3 ViT-L/16** for feature extraction
* **LoRA (Low-Rank Adaptation)** for efficient test-time tuning
* **MLP head** to estimate scale and bias
* **Huber loss** on valid prior regions
* **Dense sliding-window inference** for high-resolution output

The process is fully **training-free at dataset level** and adapts per image.

---

## Data Structure

Each sample consists of:

* `RGB/` – 3-band RGB image
* `GT_DSM/` – Ground truth DSM (for evaluation only)
* `MASK/` – Binary mask of valid target regions
* `RELATIVE_DEPTH/` – Monocular relative depth (e.g., Depth Anything)
* `PRIORS/` – Sparse or low-resolution elevation priors

Example file name:

```
000000000003.tif
```

---

## Pipeline

1. Load RGB, relative depth, priors, and mask
2. Extract patch tokens using DINOv3
3. Predict **scale (s) and bias (b)** via MLP
4. Optimize using:

   ```
   DSM = s * relative_depth + b
   ```
5. Apply **LoRA-based TTO** using prior supervision
6. Run **dense full-resolution inference**
7. Generate:

   * Final DSM prediction
   * Training video (optimization process)
   * Evaluation metrics (MAE)

---

## Outputs

* `.mp4` – optimization process visualization
* `.png` – final comparison figure
* Console logs – loss + MAE

---

## Key Features

* No full model training required
* Produces **high-resolution DSM**
* Efficient adaptation using LoRA
* Visual debugging via video generation

---

## Requirements

* PyTorch
* rasterio
* OpenCV
* matplotlib
* DINOv3 (local repo + weights)

---

## Notes

* Update all directory paths in the config section before running
* Designed for **single-image adaptation** (TTO setting)
* Ground truth is used only for evaluation, not training



## Citation
```bibtex
@misc{rafaeli2026testtimeadaptationheightcompletion,
      title={Test-Time Adaptation for Height Completion via Self-Supervised ViT Features and Monocular Foundation Models}, 
      author={Osher Rafaeli and Tal Svoray and Ariel Nahlieli},
      year={2026},
      eprint={2604.02009},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.02009}, 
}
