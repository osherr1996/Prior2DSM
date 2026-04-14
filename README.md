# Prior2DSM
Code for "Test-Time Adaptation for Height Completion via Self-Supervised ViT Features and Monocular Foundation Models"

<p align="center">
  <a href="https://arxiv.org/abs/2604.02009">
    <img src="https://img.shields.io/badge/arXiv-2604.02009-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/spaces/osherr/Prior2DSM">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow.svg" alt="Hugging Face">
  </a>
</p>

## Repository structure

```text
prior2dsm-tto/
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_example.py
│   └── evaluate_example.py
├── src/
│   └── prior2dsm/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── constants.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lora.py
│       │   ├── mlp_decoder.py
│       │   └── dino_loader.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── tto_trainer.py
│       │   ├── inference.py
│       │   └── metrics.py
│       └── utils/
│           ├── __init__.py
│           ├── logging_utils.py
│           ├── seed.py
│           └── visualization.py
├── tests/
│   ├── test_metrics.py
│   └── test_mlp_decoder.py
└── outputs/
```

## Installation

```bash
git clone <your-repo-url>
cd prior2dsm-tto
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Update `configs/default.yaml` with your local paths, then:

```bash
set PYTHONPATH=src
python scripts/run_example.py --config configs/default.yaml
```

On Linux/macOS:

```bash
export PYTHONPATH=src
python scripts/run_example.py --config configs/default.yaml
```

## Expected input files

For each `example_name`, the code expects matching filenames in:

- `rgb_dir`: RGB image
- `gt_dir`: DSM ground truth
- `mask_dir`: binary evaluation mask
- `da_dir`: relative depth raster
- `prior_dir`: prior raster

## Outputs

The script saves:

- `outputs/figures/<example>_summary.png`
- `outputs/logs/<example>_metrics.json`

You can extend it to export the final DSM as GeoTIFF.

## Notes

- Keep machine-specific paths only inside YAML config files.
- Do not commit big TIFF files or checkpoints to Git.
- Prefer small, focused commits during refactoring.

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
