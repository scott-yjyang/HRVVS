# HRVVS: A High-resolution Video Vasculature Segmentation Network via Hierarchical Autoregressive Residual Priors

<!-- <div align="center"> -->
[![HRVVS](https://img.shields.io/badge/Paper-HRVVS-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2507.22530)
<!-- </div> -->

## Introduction
Here is the official code for HRVVS, a High-resolution video vasculature segmentation network via hierarchical autoregressive residual priors.

## Environment
```bash
conda create -n HRVVS python
conda activate HRVVS
pip install -r requirements.txt
```

## Dataset
The Hepa-SEG dataset is undergoing qualification review in hospital.

## Checkpoints
```bash
mkdir checkpoints && cd checkpoints
wget https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
gdown https://drive.google.com/uc?id=1FixkCY4KOlZ0gae_7Uf9zhAG4sKDFZBV
```

## Train & Test
### Train
```bash
python train.py
```

### Test
```bash
python test.py
```

## Acknowledgement
We thank the great work from [VAR](https://github.com/FoundationVision/VAR).
